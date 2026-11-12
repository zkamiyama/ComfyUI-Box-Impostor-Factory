import json
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from box_nodes import archive, store
from test_nodes import fixture


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / 'source'
        self.manifest = fixture(self.source, 'a' * 64)
        self.package = self.root / 'packed.box.zip'
        self.restored = self.root / 'restored'

    def test_complete_half_float_and_sidecar_roundtrip(self):
        # Exercise raw half words including signed zero, subnormal, NaN payload and Inf.
        for name in ('intervalMap', 'rangeMap'):
            spec = self.manifest['textures'][name]
            path = self.source / spec['path']
            words = np.resize(np.array([0x8000, 1, 0x7d01, 0xfc00, 0x3555], dtype='<u2'),
                              spec['byteLength'] // 2)
            path.write_bytes(words.tobytes())
            spec['sha256'] = store.file_hash(path)
        (self.source / 'manifest.json').write_text(json.dumps(self.manifest, indent=1))
        view = self.source / 'views' / 'px' / 'raw.npy'
        view.parent.mkdir(parents=True)
        np.save(view, np.array([1.25], dtype='<f4'))
        (self.source / 'qa').mkdir()
        (self.source / 'qa' / 'report.json').write_text('{"accepted":true}')
        self.assertEqual(archive.pack_asset(self.source, self.package), self.package)
        with zipfile.ZipFile(self.package) as z:
            self.assertEqual(z.getinfo(self.manifest['textures']['rangeMap']['path']).compress_type,
                             zipfile.ZIP_DEFLATED)
        original_mtime = self.package.stat().st_mtime_ns
        with patch.object(zipfile.ZipFile, 'write', side_effect=AssertionError('recompressed')):
            archive.pack_asset(self.source, self.package)
        self.assertEqual(self.package.stat().st_mtime_ns, original_mtime)
        self.assertEqual(archive.unpack_asset(self.package, self.restored), self.manifest)
        def files(directory):
            return {p.relative_to(directory): store.file_hash(p)
                    for p in directory.rglob('*') if p.is_file()}
        self.assertEqual(files(self.source), files(self.restored))
        with self.assertRaises(FileExistsError):
            archive.unpack_asset(self.package, self.restored)
        view.write_bytes(b'changed auxiliary')
        package_hash = store.file_hash(self.package)
        with self.assertRaises((ValueError, FileExistsError)):
            archive.pack_asset(self.source, self.package)
        self.assertEqual(store.file_hash(self.package), package_hash)

    def test_source_changes_never_publish_partial_archive(self):
        actual_write = zipfile.ZipFile.write
        def write_then_change(archive_file, path, name):
            actual_write(archive_file, path, name)
            if name == 'manifest.json':
                path.write_bytes(path.read_bytes() + b' ')
        with patch.object(zipfile.ZipFile, 'write', write_then_change):
            with self.assertRaisesRegex(ValueError, 'changed while packing'):
                archive.pack_asset(self.source, self.package)
        self.assertFalse(self.package.exists())
        self.assertEqual(list(self.root.glob('.box-pack-*')), [])

    def test_declared_size_and_digest_failure_leave_no_destination(self):
        archive.pack_asset(self.source, self.package)
        with self.assertRaisesRegex(ValueError, 'expanded size'):
            archive.unpack_asset(self.package, self.restored, max_bytes=16)
        self.assertFalse(self.restored.exists())
        bad = self.root / 'wrong-map.zip'
        with zipfile.ZipFile(self.package) as src, zipfile.ZipFile(bad, 'w') as target:
            for info in src.infolist():
                data = src.read(info)
                if info.filename == self.manifest['textures']['rangeMap']['path']:
                    data = bytes([data[0] ^ 1]) + data[1:]
                target.writestr(info, data)
        with self.assertRaisesRegex(ValueError, 'digest'):
            archive.unpack_asset(bad, self.restored)
        self.assertFalse(self.restored.exists())
        self.assertEqual(list(self.root.glob('.box-unpack-*')), [])

    def test_reject_ambiguous_paths_and_links(self):
        link_info = zipfile.ZipInfo('maps/symlink')
        link_info.create_system = 3
        link_info.external_attr = (stat.S_IFLNK | 0o777) << 16
        for names in (('../outside',), ('/outside',), ('a/../outside',), ('a\\b',),
                      ('maps/NUL.bin',), ('maps/odd.',), ('maps/file', 'MAPS/other'),
                      ('maps', 'maps/file'), (link_info,)):
            with self.subTest(names=names):
                with zipfile.ZipFile(self.package, 'w') as output:
                    output.writestr('manifest.json', '{}')
                    for name in names:
                        output.writestr(name, 'x')
                with self.assertRaises(ValueError):
                    archive.unpack_asset(self.package, self.restored)
                self.assertFalse(self.restored.exists())
        self.assertFalse((self.root / 'outside').exists())
        (self.source / 'bad-link').symlink_to(self.source / 'manifest.json')
        with self.assertRaisesRegex(ValueError, 'regular files'):
            archive.pack_asset(self.source, self.root / 'link.zip')


if __name__ == '__main__':
    unittest.main()
