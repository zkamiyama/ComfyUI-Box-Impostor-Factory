"""H runner controls on tiny meshes/files; no GPU, model reads or native bake."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import trimesh

from box_nodes import depth_mesh

SCRIPT = Path(__file__).resolve().parents[1]/'scripts/run_geometry_h_reference.py'
spec = importlib.util.spec_from_file_location('run_geometry_h_reference',SCRIPT)
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)


class GeometryHTests(unittest.TestCase):
    def tiny_legacy_export(self, root):
        source = trimesh.creation.icosphere(subdivisions=1)
        material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.new('RGBA',(2,2),(134,133,131,128)))
        source.visual = trimesh.visual.TextureVisuals(
            uv=np.full((len(source.vertices),2),.5),material=material)
        original, raw, output = root/'source.glb',root/'raw.glb',root/'prepared.glb'
        original.write_bytes(trimesh.exchange.gltf.export_glb(source,include_normals=False))
        geometry = trimesh.Trimesh(vertices=source.vertices.copy(),faces=source.faces.copy(),process=False)
        raw.write_bytes(trimesh.exchange.gltf.export_glb(geometry,include_normals=False))
        legacy = h.legacy_postprocess()
        mesh,report = legacy.prepare(original,raw,smooth=0)
        mesh.export(output)
        return mesh,report,output

    def test_pinned_snapshot_retains_b_color_and_normal_policy_on_tiny_export(self):
        self.assertEqual(h.digest(h.SNAPSHOT),h.SNAPSHOT_SHA)
        with tempfile.TemporaryDirectory() as tmp:
            mesh,report,output = self.tiny_legacy_export(Path(tmp))
            expected = np.tile([134,133,131,128],(len(mesh.vertices),1))
            np.testing.assert_array_equal(mesh.visual.vertex_colors,expected)
            self.assertEqual(report['smooth_iterations'],0)
            signature = h.glb_semantics(output,depth_mesh)
            attrs = signature['meshes'][0][0]['attributes']
            self.assertNotIn('NORMAL',attrs)
            self.assertEqual(attrs['COLOR_0']['componentType'],5121)
            self.assertTrue(attrs['COLOR_0']['normalized'])

    def test_replay_semantics_ignore_generator_text_but_detect_position_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _,_,output = self.tiny_legacy_export(root)
            before = h.glb_semantics(output,depth_mesh)
            doc,binary = depth_mesh._decode(output.read_bytes())
            doc['asset']['generator'] = 'metadata-only comparison'
            changed = root/'metadata.glb'
            changed.write_bytes(depth_mesh._encode(doc,bytearray(binary)))
            self.assertNotEqual(h.digest(output),h.digest(changed))
            self.assertEqual(before,h.glb_semantics(changed,depth_mesh))
            prim = doc['meshes'][0]['primitives'][0]
            xyz = depth_mesh._accessor(doc,binary,prim['attributes']['POSITION'],'VEC3',{5126}).copy()
            xyz[0,0] += .01
            data = bytearray(binary)
            depth_mesh._write_attribute(doc,data,prim,'POSITION',xyz)
            changed.write_bytes(depth_mesh._encode(doc,data))
            self.assertNotEqual(before,h.glb_semantics(changed,depth_mesh))

    def test_successful_b_environment_keeps_venv_path_and_regular_hardlinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp);case=run/'B';case.mkdir()
            model = run/'models/model.bin';model.parent.mkdir()
            model.write_bytes(b'fixture')
            linked = run/'snapshot/model.bin';linked.parent.mkdir();os.link(model,linked)
            python = run/'venv/bin/python';python.parent.mkdir(parents=True)
            python.symlink_to('/usr/bin/python3')
            h.save(case/'hi3d-process.json',{'environment':{
                'BI_HI3D_PYTHON':str(python),'HF_HUB_CACHE':str(linked.parent),'HF_HUB_OFFLINE':'1'}})
            h.save(case/'model-materialization.json',{'materializedHub':str(linked.parent),'files':[{
                'regularFile':str(linked),'sourceBlob':str(model),'bytes':7,'sha256':h.digest(model)}]})
            proof = h.execution_environment_proof(run)
            self.assertEqual(proof['pythonExecutable'],str(python))
            self.assertFalse(proof['executablePathResolved'])
            self.assertTrue(proof['modelFiles'][0]['sameHardlinkedFile'])
            self.assertTrue(proof['wslNvmlGuardBeforeCudaAndPipelineImport'])
            linked.unlink();linked.symlink_to(model)
            with self.assertRaisesRegex(ValueError,'materialization differs'):
                h.execution_environment_proof(run)


if __name__=='__main__':
    unittest.main()
