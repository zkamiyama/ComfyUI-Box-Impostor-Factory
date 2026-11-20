"""Encode synchronized Inspector captures from a JSON list of tested depth assets."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw, ImageFont


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=True)
    ffmpeg = config['ffmpeg']
    captures = []
    for item in config['captures']:
        directory = Path(item['directory'])
        data = json.loads((directory / 'capture.json').read_text())
        assert data['completed'] and not data['errors'], item['label']
        assert len(data['frames']) == data['framesExpected']
        for frame in data['frames']:
            assert digest(frame['path']) == frame['sha256']
        assert len({frame['sha256'] for frame in data['frames']}) > 1
        captures.append((item, data))
    reference = captures[0][1]
    max_camera_error = 0.0
    for item, data in captures[1:]:
        for key in ('sourceFaceResolution', 'framesExpected', 'playbackFPS'):
            assert data[key] == reference[key], (item['label'], key)
        for key in ('projection', 'pixels', 'target', 'halfExtent', 'startDegrees', 'sweepDegrees'):
            assert data['camera'][key] == reference['camera'][key], (item['label'], key)
        assert data['display'] == reference['display'], 'Only depth assets may differ'
        for before, after in zip(reference['frames'], data['frames']):
            assert before['requestedYawDegrees'] == after['requestedYawDegrees']
            for key in ('matrixWorld', 'projectionMatrix'):
                error = max(abs(a-b) for a, b in zip(before['camera'][key], after['camera'][key]))
                max_camera_error = max(error, max_camera_error)
                assert error < 1e-6, (item['label'], before['index'], key, error)
    width = reference['camera']['pixels'][0]
    fps = reference['playbackFPS']
    count = reference['framesExpected']
    band = 64
    title_font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 22)
    small_font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
    files = []

    def run(command):
        subprocess.run(command, check=True, timeout=600)

    def encode(indices, name, columns):
        commands = [ffmpeg, '-hide_banner', '-loglevel', 'warning', '-y']
        for index in indices:
            commands += ['-framerate', str(fps), '-i', str(Path(captures[index][0]['directory']) / 'frame-%04d.png')]
        for panel, index in enumerate(indices):
            header = Image.new('RGB', (width, band), '#15191e')
            draw = ImageDraw.Draw(header)
            draw.text((16, 8), captures[index][0]['label'], font=title_font, fill='white')
            draw.text((16, 37), 'native Box / ortho / 1024px geometry atlas', font=small_font, fill='#b2c0cf')
            path = output / f'{name}-header-{panel}.png'
            header.save(path)
            commands += ['-loop', '1', '-framerate', str(fps), '-i', str(path)]
        filters = []
        for panel in range(len(indices)):
            filters += [f'color=c=0x15191e:s={width}x{width}:r={fps}[bg{panel}]',
                        f'[bg{panel}][{panel}:v]overlay=shortest=1:format=auto,format=rgb24[body{panel}]',
                        f'[{len(indices)+panel}:v][body{panel}]vstack=inputs=2:shortest=1[p{panel}]']
        layout = '|'.join(f'{i % columns * width}_{i // columns * (width+band)}' for i in range(len(indices)))
        inputs = ''.join(f'[p{i}]' for i in range(len(indices)))
        if len(indices) == 1:
            filters += ['[p0]setsar=1,scale=out_color_matrix=bt709:out_range=tv[v]']
        else:
            filters += [inputs + f'xstack=inputs={len(indices)}:layout={layout}:fill=0x15191e,setsar=1,scale=out_color_matrix=bt709:out_range=tv[v]']
        target = output / (name + '.mp4')
        commands += ['-filter_complex', ';'.join(filters), '-map', '[v]', '-frames:v', str(count),
                     '-an', '-c:v', 'libx264', '-preset', 'medium', '-crf', '19', '-pix_fmt', 'yuv420p',
                     '-color_primaries', 'bt709', '-color_trc', 'bt709', '-colorspace', 'bt709',
                     '-movflags', '+faststart', '-g', str(round(fps)), str(target)]
        run(commands)
        run([ffmpeg, '-v', 'error', '-i', str(target), '-f', 'null', '-'])
        files.append(target)
        return target

    combined = encode(list(range(len(captures))), 'gs-direct-box-comparison', 3)
    for index, (item, data) in enumerate(captures[1:], 1):
        encode([0, index], 'gs-direct-box-' + item['id'], 2)
    gif = output / 'gs-direct-box-preview.gif'
    run([ffmpeg, '-hide_banner', '-v', 'warning', '-y', '-i', str(combined),
         '-filter_complex', 'fps=6,scale=720:-2:flags=lanczos,split[a][b];[a]palettegen=max_colors=96:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=3',
         '-loop', '0', str(gif)])
    files.append(gif)
    for yaw in (0, 45, 315):
        index = min(reference['frames'], key=lambda f: abs((f['requestedYawDegrees']-yaw+180) % 360-180))['index']
        path = output / f'gs-direct-box-yaw-{yaw:03d}.png'
        run([ffmpeg, '-hide_banner', '-v', 'error', '-y', '-i', str(combined),
             '-vf', f'select=eq(n\\,{index})', '-frames:v', '1', str(path)])
        files.append(path)
    result = {'inputConfig': str(args.config), 'inputConfigSha256': digest(args.config),
              'sourceResolution': reference['sourceFaceResolution'], 'frames': count, 'fps': fps,
              'durationSeconds': count/fps, 'maximumCameraMatrixDifference': max_camera_error,
              'captures': [{'id': item['id'], 'label': item['label'], 'assetId': data['assetId'],
                            'htmlSha256': data['htmlSHA256'],
                            'captureSha256': digest(Path(item['directory'])/'capture.json')}
                           for item, data in captures],
              'note': 'Actual unchanged 3D-Inspector renders. No generated frames or motion interpolation. Encoding FPS is not a rendering benchmark.',
              'files': [{'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest(path)} for path in files]}
    (output/'media-provenance.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
