import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from box_nodes import generation, store
from box_nodes.secondary_references import SecondaryReferences, SecondaryReferenceSlots
from test_nodes import fixture


class SecondaryReferencesTests(unittest.TestCase):
    def test_order_limit_empty_slots_and_in_place_replacement(self):
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            self.enterContext(patch.dict(sys.modules, folder_paths=SimpleNamespace(get_input_directory=lambda: work)))
            self.enterContext(patch.object(store, 'root', return_value=root))
            loader, split = SecondaryReferences(), SecondaryReferenceSlots()
            image = root / 'side.png'
            Image.new('RGB', (6, 3), 'red').save(image)
            Image.new('RGB', (4, 2), 'blue').save(root / 'back.png')
            names = 'side.png\nback.png'
            first = loader.IS_CHANGED(names)
            self.assertNotEqual(first, loader.IS_CHANGED('back.png\nside.png'))
            self.assertEqual(split.unpack({'images':[]}), (None,) * 8)
            Image.new('RGB', (6, 3), 'lime').save(image)
            changed = loader.IS_CHANGED(names)
            self.assertNotEqual(first, changed)
            with self.assertRaisesRegex(ValueError, '8'):
                loader.IS_CHANGED('\n'.join(['side.png'] * 9))
            with self.assertRaisesRegex(ValueError, 'relative'):
                loader.IS_CHANGED('../side.png')
            asset = 'a' * 64
            fixture(root / asset, asset)
            prompt = {'out': {'inputs': {}, 'class_type': 'BICachedGeneratedFace'}}
            args = (asset, {'sha256': 'd' * 64}, '+Z', 'qwen_enhanced', '{}', 42, 'rgba', prompt, 'out')
            empty, _ = generation.generation_request(*args)
            self.assertEqual(empty, generation.generation_request(*args, secondary={'sha256': []})[0])
            old, provenance = generation.generation_request(*args, secondary={'sha256': first})
            self.assertEqual(provenance['secondaryReferenceSha256'], first)
            self.assertNotEqual(old, generation.generation_request(*args, secondary={'sha256': changed})[0])
