"""Optional appearance references shared by all six view-generation branches."""
import hashlib
from io import BytesIO

import numpy as np
from PIL import Image, ImageOps

from . import store


MAX_SECONDARY = 8  # Primary and the geometry guide occupy the first two model slots.


def paths_from_text(paths):
    import folder_paths
    names = [line.strip() for line in paths.splitlines() if line.strip()]
    if len(names) > MAX_SECONDARY:
        raise ValueError('Supply at most 8 secondary references in addition to the primary and guide.')
    return [store.contained(folder_paths.get_input_directory(), name) for name in names]


class SecondaryReferences:
    CATEGORY = 'Box Impostor'
    FUNCTION = 'load'
    RETURN_TYPES = ('BOX_SECONDARY_REFERENCES',)
    DESCRIPTION = ('Additional appearance references for the six face images. Enter one ComfyUI input '
                   'image filename per line, in order. Leave empty to use only the primary and guide.')

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'paths': ('STRING', {'default': '', 'multiline': True})}}

    @classmethod
    def IS_CHANGED(cls, paths):
        from .generation import fingerprint_file
        return tuple(fingerprint_file(path) for path in paths_from_text(paths))

    def load(self, paths):
        import torch
        hashes, images = [], []
        for path in paths_from_text(paths):
            data = path.read_bytes()
            hashes.append(hashlib.sha256(data).hexdigest())
            with Image.open(BytesIO(data)) as opened:
                if getattr(opened, 'n_frames', 1) != 1:
                    raise ValueError(f'Each secondary reference must be a single still image: {path.name}')
                rgba = ImageOps.exif_transpose(opened).convert('RGBA')
                rgb = Image.new('RGB', rgba.size, (255, 255, 255))
                rgb.paste(rgba, mask=rgba.getchannel('A'))
                images.append(torch.from_numpy(np.asarray(rgb, dtype=np.float32) / 255).unsqueeze(0))
        return ({'images': images, 'sha256': hashes},)


class SecondaryReferenceSlots:
    CATEGORY = 'Box Impostor'
    FUNCTION = 'unpack'
    RETURN_TYPES = ('IMAGE',) * MAX_SECONDARY
    RETURN_NAMES = tuple(f'image_{n}' for n in range(3, MAX_SECONDARY + 3))

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'references': ('BOX_SECONDARY_REFERENCES',)}}

    def unpack(self, references):
        images = tuple(references['images'])
        return images + (None,) * (MAX_SECONDARY - len(images))
