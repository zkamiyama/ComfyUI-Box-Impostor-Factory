"""Encode paired actual Inspector captures as a widely playable comparison video."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from PIL import Image, ImageDraw, ImageFont


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def ffmpeg_path(value):
    path = value or os.environ.get('FFMPEG_BINARY') or shutil.which('ffmpeg')
    if not path:
        raise RuntimeError('Set --ffmpeg or FFMPEG_BINARY to an existing ffmpeg binary')
    return path


def run(command):
    subprocess.run(command, check=True, timeout=600)


def capture(path):
    data = json.loads((path / 'capture.json').read_text())
    assert data['completed'] and not data['errors'], str(path)
    assert len(data['frames']) == data['framesExpected']
    for index, frame in enumerate(data['frames']):
        png = path / f'frame-{index:04d}.png'
        assert frame['index'] == index and sha(png) == frame['sha256'], str(png)
    assert len({f['sha256'] for f in data['frames']}) > 1, 'Turntable has no changing frames'
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('before', type=Path)
    parser.add_argument('after', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--ffmpeg')
    parser.add_argument('--font', default='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    parser.add_argument('--gif', action='store_true')
    args = parser.parse_args()
    before, after = capture(args.before), capture(args.after)
    for key in ['assetId', 'sourceFaceResolution', 'framesExpected', 'playbackFPS']:
        assert before[key] == after[key], f'Comparison differs: {key}'
    for key in ['projection', 'pixels', 'target', 'halfExtent', 'expectedDistance',
                'startDegrees', 'sweepDegrees']:
        assert before['camera'][key] == after['camera'][key], f'Camera setup differs: {key}'
    max_matrix_difference = 0.0
    for a, b in zip(before['frames'], after['frames']):
        assert a['requestedYawDegrees'] == b['requestedYawDegrees']
        for key in ['matrixWorld', 'projectionMatrix']:
            error = max(abs(x-y) for x, y in zip(a['camera'][key], b['camera'][key]))
            max_matrix_difference = max(max_matrix_difference, error)
            assert error < 1e-6, f'Frame {a["index"]} camera differs'
    args.output.mkdir(parents=True, exist_ok=True)
    ffmpeg = ffmpeg_path(args.ffmpeg)
    width = before['camera']['pixels'][0]
    assert width % 2 == 0, 'H.264 yuv420p requires even dimensions'
    fps = before['playbackFPS']
    nframes = before['framesExpected']
    font = Path(args.font)
    assert font.is_file(), f'Font missing: {font}'
    band = 72
    filters = []
    smoke = before['htmlSHA256'] == after['htmlSHA256']
    titles = ['Before: native Box', 'After: angle-driven field morph']
    if smoke:
        titles = ['Encoder test: baseline', 'Encoder test: same baseline']
    header = Image.new('RGB', (2*width, band), '#15191e')
    draw = ImageDraw.Draw(header)
    title_font = ImageFont.truetype(str(font), 22)
    caption_font = ImageFont.truetype(str(font), 14)
    for i, title in enumerate(titles):
        draw.text((i*width+18, 10), title, font=title_font, fill='white')
        draw.text((i*width+18, 40),
                  f"{before['sourceFaceResolution']} px source faces  |  actual Box render",
                  font=caption_font, fill='#adb8c6')
        filters.append(f'color=c=0x15191e:s={width}x{width}:r={fps}[bg{i}]')
        filters.append(f'[bg{i}][{i}:v]overlay=shortest=1:format=auto,format=rgb24[p{i}]')
    header_path = args.output / 'video-header.png'
    header.save(header_path)
    filters.append('[p0][p1]hstack=inputs=2[body]')
    filters.append('[2:v][body]vstack=inputs=2:shortest=1,setsar=1,'
                   'scale=out_color_matrix=bt709:out_range=tv[v]')
    output_mp4 = args.output / 'box-depth-morph-turntable.mp4'
    command = [ffmpeg, '-hide_banner', '-loglevel', 'warning', '-y',
               '-framerate', str(fps), '-i', str(args.before / 'frame-%04d.png'),
               '-framerate', str(fps), '-i', str(args.after / 'frame-%04d.png'),
               '-loop', '1', '-framerate', str(fps), '-i', str(header_path),
               '-filter_complex', ';'.join(filters), '-map', '[v]', '-frames:v', str(nframes),
               '-an', '-c:v', 'libx264', '-preset', 'medium', '-crf', '16',
               '-pix_fmt', 'yuv420p', '-profile:v', 'high', '-level:v', '4.1',
               '-color_primaries', 'bt709', '-color_trc', 'bt709', '-colorspace', 'bt709',
               '-movflags', '+faststart', '-g', str(round(fps)), str(output_mp4)]
    run(command)
    run([ffmpeg, '-hide_banner', '-v', 'error', '-i', str(output_mp4), '-f', 'null', '-'])
    files = [output_mp4]
    # A real frame near 45 degrees helps inspect the region that motivated the change.
    nearest = min(before['frames'], key=lambda frame: abs(frame['requestedYawDegrees']-45))['index']
    poster = args.output / 'box-depth-morph-poster.png'
    run([ffmpeg, '-hide_banner', '-v', 'error', '-y', '-i', str(output_mp4),
         '-vf', f'select=eq(n\\,{nearest})', '-frames:v', '1', str(poster)])
    files.append(poster)
    if args.gif:
        gif = args.output / 'box-depth-morph-turntable.gif'
        run([ffmpeg, '-hide_banner', '-v', 'warning', '-y', '-i', str(output_mp4),
             '-filter_complex', 'fps=12,scale=768:-2:flags=lanczos,split[a][b];'
             '[a]palettegen=max_colors=128:stats_mode=diff[p];'
             '[b][p]paletteuse=dither=bayer:bayer_scale=3', '-loop', '0', str(gif)])
        files.append(gif)
    result = {
        'beforeCapture': str(args.before / 'capture.json'),
        'beforeCaptureSHA256': sha(args.before / 'capture.json'),
        'afterCapture': str(args.after / 'capture.json'),
        'afterCaptureSHA256': sha(args.after / 'capture.json'),
        'assetId': before['assetId'], 'sourceFaceResolution': before['sourceFaceResolution'],
        'frameCount': nframes, 'playbackFPS': fps, 'durationSeconds': nframes/fps,
        'posterYawDegrees': before['frames'][nearest]['actualYawDegrees'],
        'cameraMaximumMatrixDifference': max_matrix_difference,
        'width': 2*width, 'height': width+band,
        'codec': 'H.264 High / yuv420p / MP4 faststart',
        'note': 'Playback FPS is the encoding rate, not a realtime-rendering benchmark.',
        'comparison': ('Encoder smoke test with the same baseline on both sides; not an effect comparison.'
                       if smoke else
                       'Same source asset, same framing and measured camera matrices; two real render paths.'),
        'command': command,
        'files': [{'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)} for path in files],
    }
    (args.output / 'video-provenance.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
