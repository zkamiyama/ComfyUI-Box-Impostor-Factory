import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None

from box_nodes import material_supermat


@unittest.skipIf(torch is None, 'SuperMat map conversion requires PyTorch')
class SuperMatOutputTests(unittest.TestCase):
    def test_uses_material_g_and_b_and_removes_padding_from_a_nonsquare_face(self):
        # R is intentionally arbitrary. G contains continuous roughness and B straddles the cutoff.
        image = torch.empty(1, 256, 512, 3)
        image[..., 0] = .73
        image[..., 1] = .37
        image[..., :256, 2] = .49
        image[..., 256:, 2] = .51

        class Decompose:
            def decompose(self, square, model, vae, size, batch_chunk, clip):
                assert square.shape == (1, 512, 512, 3)
                assert torch.all(square[:, :128] == .5)
                return square, square

        registry = {'SuperPBRDecompose': Decompose}
        with patch.dict(sys.modules, {'nodes': SimpleNamespace(NODE_CLASS_MAPPINGS=registry),
                                      'comfy.utils': SimpleNamespace(common_upscale=None)}):
            albedo, rma = material_supermat.estimate(image, None, None, None, 512)
            _, higher = material_supermat.estimate(image, None, None, None, 512, .7)
            self.assertTrue(torch.all(higher[..., 1] == 0))
        torch.testing.assert_close(albedo, image)
        torch.testing.assert_close(rma[..., 0], image[..., 1])
        self.assertTrue(torch.all(rma[..., :256, 1] == 0))
        self.assertTrue(torch.all(rma[..., 256:, 1] == 1))
        self.assertTrue(torch.all(rma[..., 2] == 1))
