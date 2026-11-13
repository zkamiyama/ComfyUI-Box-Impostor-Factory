import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None

from box_nodes import material_cosmos


@unittest.skipIf(torch is None, 'Cosmos output conversion requires PyTorch')
class CosmosOutputs(unittest.TestCase):
    def test_pass_order_padding_threshold_and_output_dimensions(self):
        image = torch.full((1, 245, 512, 3), .25)
        class Inverse:
            def run_inverse_pass(self, pipeline, padded, guidance, seed, num_steps):
                assert padded.shape == (1, 256, 512, 3)
                assert torch.all(padded[:, :5] == .5)
                assert (guidance, seed, num_steps) == (0., 42, 15)
                metal = torch.full_like(padded, .49)
                metal[:, :, :256] = .5
                return padded, metal, torch.ones_like(padded), None, None
        nodes = SimpleNamespace(NODE_CLASS_MAPPINGS={'MateriaInverseRenderer': Inverse})
        with patch.dict(sys.modules, {'nodes': nodes}):
            albedo, packed = material_cosmos.estimate(image, None, 2048, 42)
        self.assertEqual(albedo.shape, (1, 980, 2048, 3))
        self.assertTrue(torch.allclose(albedo, torch.full_like(albedo, .25)))
        self.assertTrue(torch.all(packed[..., 0] == 1))
        self.assertTrue(torch.all(packed[:, :, :900, 1] == 1))
        self.assertTrue(torch.all(packed[:, :, 1100:, 1] == 0))
        self.assertTrue(torch.all(packed[..., 2] == 1))
