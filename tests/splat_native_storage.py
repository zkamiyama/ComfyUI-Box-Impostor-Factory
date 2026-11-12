"""Verify restored SPLAT tensors with the actual pinned ComfyUI PLY reader."""
from pathlib import Path
import sys
import tempfile

root = Path(__file__).resolve().parents[1]
comfy = Path(sys.argv[1]).resolve()
sys.path[:0] = [str(comfy), str(root)]
sys.argv[1:] = ['--cpu']
import comfy.options
comfy.options.enable_args_parsing()
from comfy_api.latest import Types
from comfy_extras.nodes_gaussian_splat import File3DToSplat
import torch

from box_nodes.splat_cache import FIELDS, restore_splat, save_decode
from splat_native_smoke import write_splat_fixture

with tempfile.TemporaryDirectory() as tmp:
    directory = Path(tmp)
    write_splat_fixture(directory)
    original, = File3DToSplat.execute(Types.File3D(str(directory/'3d'/'two-gaussians.ply'))).result
    save_decode(directory, original, {'source': 'native-ply'})
    restored = restore_splat(directory)
    assert isinstance(restored, Types.SPLAT)
    for name in FIELDS:
        torch.testing.assert_close(getattr(original, name), getattr(restored, name), atol=0, rtol=0)
    print('Native PLY to Gaussian tensor archive round trip matched all five fields.', flush=True)
