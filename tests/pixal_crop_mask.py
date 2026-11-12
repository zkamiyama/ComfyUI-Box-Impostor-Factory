"""Recreate the Pixal3D workflow's conditioning mask with its native ComfyUI crop node."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('comfy', type=Path, help='ComfyUI checkout containing the native nodes')
    parser.add_argument('asset', type=Path, help='Pixal3D Box asset directory')
    parser.add_argument('output', type=Path, help='Destination for the grayscale mask PNG')
    args = parser.parse_args()

    sys.path.insert(0, str(args.comfy.resolve()))
    # ComfyUI parses its own argv on import. The crop can run entirely on the CPU.
    sys.argv[1:] = ['--cpu']
    import comfy.options
    comfy.options.enable_args_parsing()
    from comfy_extras.nodes_images import ImageCropToMask

    manifest = json.loads((args.asset / 'manifest.json').read_text())
    reference = manifest['reference']
    camera = reference['camera']
    source = torch.from_numpy(np.load(args.asset / reference['samples'], allow_pickle=False).astype('float32'))
    mask = source[None, ..., 3]
    rgb = source[None, ..., :3]

    def crop(image):
        pixels = ImageCropToMask.execute(image, mask, camera['width'], camera['height'],
                                         1.1, 0, '#000000')[0][0]
        return np.uint8(np.clip(np.rint(pixels.cpu().numpy() * 255), 0, 255))

    with Image.open(args.asset / camera['image']) as saved:
        if not np.array_equal(crop(rgb), np.asarray(saved.convert('RGB'))):
            raise ValueError('The crop recipe did not reproduce the saved conditioning image')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(crop(torch.ones_like(rgb))[..., 0]).save(args.output)


if __name__ == '__main__':
    main()
