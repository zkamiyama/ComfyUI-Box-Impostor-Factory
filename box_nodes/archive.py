"""Losslessly package an entire Box directory for transport with ZIP/DEFLATE."""
import argparse
import hashlib
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import stat
import tempfile
import unicodedata
import zipfile

from . import store


# The limits apply to expanded input, including auxiliary images and optional maps.
DEFAULT_MAX_BYTES = 32 * 1024**3
MAX_ENTRIES = 8192


def _check_names(names):
    """Require unambiguous, portable filenames, including on case-insensitive disks."""
    kinds = {}
    for name, is_dir in names:
        path = PurePosixPath(name)
        if (not name or name == '.' or path.is_absolute() or str(path) != name
                or '..' in path.parts or any(c in '<>:"\\|?*' or ord(c) < 32 for c in name)):
            raise ValueError(f'Invalid Box ZIP member path: {name!r}')
        for i, part in enumerate(path.parts, 1):
            if part.endswith((' ', '.')) or PureWindowsPath(part).is_reserved():
                raise ValueError(f'Invalid Box ZIP member path: {name!r}')
            prefix = '/'.join(path.parts[:i])
            normalized = unicodedata.normalize('NFC', prefix).casefold()
            kind = is_dir or i < len(path.parts)
            prior = kinds.setdefault(normalized, (prefix, kind))
            if prior != (prefix, kind):
                raise ValueError(f'Conflicting Box ZIP paths: {name!r}')


def _members(archive, max_bytes):
    entries = archive.infolist()
    if len(entries) > MAX_ENTRIES or sum(entry.file_size for entry in entries) > max_bytes:
        raise ValueError('Box ZIP exceeds the expanded size or entry count limit.')
    members = {}
    for entry in entries:
        directory = entry.is_dir()
        name = entry.filename[:-1] if directory else entry.filename
        mode = stat.S_IFMT(entry.external_attr >> 16)
        if (entry.orig_filename != entry.filename or name in members
                or mode not in (0, stat.S_IFDIR if directory else stat.S_IFREG)
                or (directory and entry.file_size) or entry.flag_bits & 1
                or entry.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)):
            raise ValueError(f'Invalid, duplicate or unsupported Box ZIP member: {name!r}')
        members[name] = entry
    _check_names((name, entry.is_dir()) for name, entry in members.items())
    if 'manifest.json' not in members or members['manifest.json'].is_dir():
        raise ValueError('Box ZIP needs a manifest.json at its root.')
    return members


def _source_snapshot(directory):
    files = {}
    for path in sorted(directory.rglob('*')):
        info = path.lstat()
        if stat.S_ISDIR(info.st_mode):
            continue
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f'Box ZIP requires regular files and directories: {path.name!r}')
        name = path.relative_to(directory).as_posix()
        files[name] = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
    _check_names((name, False) for name in files)
    return files


def _check_existing(destination, files):
    if destination.is_symlink() or not destination.is_file():
        raise FileExistsError(f'Export destination already exists: {destination}')
    with zipfile.ZipFile(destination) as archive:
        entries = _members(archive, sum(path.stat().st_size for path in files.values()))
        if {name for name, entry in entries.items() if not entry.is_dir()} == files.keys():
            for name, path in files.items():
                if path.stat().st_size != entries[name].file_size:
                    break
                with archive.open(entries[name]) as stream:
                    if hashlib.file_digest(stream, 'sha256').hexdigest() != store.file_hash(path):
                        break
            else:
                return destination
    raise FileExistsError(f'Export destination contains a different Box: {destination}')


def pack_asset(directory, destination):
    """Publish a new archive atomically. Identical existing file sets can be reused."""
    directory = Path(directory).resolve()
    destination = Path(destination).absolute()
    if destination.resolve().is_relative_to(directory):
        raise ValueError('Place the Box ZIP outside the source asset directory.')
    snapshot = _source_snapshot(directory)
    files = {name: directory / name for name in snapshot}
    store.validate(directory)
    if len(files) > MAX_ENTRIES:
        raise ValueError('Box ZIP exceeds the entry count limit.')
    if destination.exists() or destination.is_symlink():
        return _check_existing(destination, files)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.box-pack-', dir=destination.parent) as tmp:
        staging = Path(tmp) / 'package.zip'
        with zipfile.ZipFile(staging, 'w', compression=zipfile.ZIP_DEFLATED,
                             strict_timestamps=False) as archive:
            for name, path in files.items():
                archive.write(path, name)
        if _source_snapshot(directory) != snapshot:
            raise ValueError('Source Box changed while packing. Retry with a stable asset.')
        try:
            # Same-filesystem link creation publishes atomically without clobbering
            # a competing exporter that claimed the name after the existence check.
            os.link(staging, destination)
        except FileExistsError:
            return _check_existing(destination, files)
    return destination


def unpack_asset(archive_path, destination, *, max_bytes=DEFAULT_MAX_BYTES):
    """Verify and restore the archive to a new directory on the same filesystem."""
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f'Box destination already exists: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    available = shutil.disk_usage(destination.parent).free
    with zipfile.ZipFile(archive_path) as archive:
        entries = _members(archive, min(max_bytes, available))
        with tempfile.TemporaryDirectory(prefix='.box-unpack-', dir=destination.parent) as tmp:
            staging = Path(tmp) / 'asset'
            staging.mkdir()
            for name, entry in entries.items():
                path = store.contained(staging, name)
                if entry.is_dir():
                    path.mkdir(parents=True, exist_ok=True)
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(entry) as source, path.open('xb') as target:
                    shutil.copyfileobj(source, target, 1024 * 1024)
            manifest = store.validate(staging)
            os.rename(staging, destination)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    pack = commands.add_parser('pack', help='compress a Box directory')
    pack.add_argument('directory', type=Path)
    pack.add_argument('destination', type=Path)
    unpack = commands.add_parser('unpack', help='restore to a new Box directory')
    unpack.add_argument('archive', type=Path)
    unpack.add_argument('destination', type=Path)
    unpack.add_argument('--max-bytes', type=int, default=DEFAULT_MAX_BYTES,
                        help='maximum expanded byte count (default: 32 GiB)')
    args = parser.parse_args()
    if args.command == 'pack':
        pack_asset(args.directory, args.destination)
    else:
        unpack_asset(args.archive, args.destination, max_bytes=args.max_bytes)


if __name__ == '__main__':
    main()
