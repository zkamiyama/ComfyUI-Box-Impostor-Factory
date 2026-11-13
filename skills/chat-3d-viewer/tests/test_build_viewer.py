import base64
import hashlib
import io
import json
from pathlib import Path
import re
import tempfile
import unittest
import zlib
import zipfile
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from build_viewer import build

FIXTURES=Path(__file__).resolve().parent/'fixtures'


def scene_from_html(path):
    html=Path(path).read_text(encoding='utf-8')
    payload=re.search(r'<script id="scene-payload"[^>]+>(.*?)</script>',html,re.S)
    assert payload
    return json.loads(zlib.decompress(base64.b64decode(payload.group(1))))


class BuildViewerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (FIXTURES/'material_study.glb').exists():
            exec((FIXTURES.parent/'make_fixtures.py').read_text(),{'__file__':str(FIXTURES.parent/'make_fixtures.py')})

    def test_glb_stats_textures_are_real_and_offline(self):
        with tempfile.TemporaryDirectory() as out:
            page=Path(out)/'preview.html'
            result=build(FIXTURES/'material_study.glb',page)
            scene=scene_from_html(page)
            self.assertEqual(result['meshes'],3)
            self.assertEqual(result['materials'],3)
            self.assertEqual(result['textures'],3)
            self.assertEqual(result['triangles'],428)
            self.assertEqual(scene['original']['triangles'],428)
            self.assertTrue(all(x['dataUri'].startswith('data:image/png;base64,') for x in scene['textures']))
            self.assertIn('Unlit' if False else 'Base texture',page.read_text())
            self.assertNotRegex(page.read_text(),r'<script\s+src=')
            self.assertNotIn('fetch(',page.read_text())
            self.assertEqual(scene['meshes'][0]['name'],'Box')

    def test_box_archive_discloses_approximation_and_map_roles(self):
        with tempfile.TemporaryDirectory() as out:
            page=Path(out)/'box.html'
            stats=build(FIXTURES/'synthetic.box.zip',page,grid=24)
            scene=scene_from_html(page)
            self.assertEqual(stats['mode'],'surface')
            self.assertGreater(stats['vertices'],100)
            self.assertEqual(stats['meshes'],6)
            self.assertEqual(stats['textures'],11)
            self.assertIn('not the native',scene['note'])
            self.assertEqual(scene['assetId'],'a'*64)
            self.assertGreater(scene['original']['triangles'],400)
            self.assertTrue(any(t['role'].startswith('R: roughness') for t in scene['textures']))

    def test_deterministic_html_and_title_sanitized(self):
        with tempfile.TemporaryDirectory() as out:
            a,b=Path(out)/'a.html',Path(out)/'b.html'
            build(FIXTURES/'material_study.glb',a,title='<script>oops</script>')
            build(FIXTURES/'material_study.glb',b,title='<script>oops</script>')
            self.assertEqual(hashlib.sha256(a.read_bytes()).hexdigest(),hashlib.sha256(b.read_bytes()).hexdigest())
            scene=scene_from_html(a)
            self.assertNotIn('oops',a.read_text())
            self.assertEqual(scene['name'],'<script>oops</script>')

    def test_rejects_invalid_format_and_archive_traversal(self):
        with tempfile.TemporaryDirectory() as out:
            out=Path(out)
            (out/'test.foo').write_text('oops')
            with self.assertRaisesRegex(ValueError,'Unsupported input'):
                build(out/'test.foo',out/'out.html')
            with zipfile.ZipFile(out/'bad.box.zip','w') as z:
                z.writestr('../breakout','bad')
                z.writestr('manifest.json','{}')
            with self.assertRaisesRegex(ValueError,'Unsafe ZIP'):
                build(out/'bad.box.zip',out/'out.html')


    def test_stl_and_ply_are_supported(self):
        import trimesh
        with tempfile.TemporaryDirectory() as out:
            out=Path(out)
            mesh=trimesh.creation.box(extents=[1,2,3])
            for extension in ('.stl','.ply'):
                file=out/('part'+extension)
                mesh.export(file)
                page=out/(extension[1:]+'.html')
                result=build(file,page)
                self.assertGreater(result['vertices'],0)
                self.assertGreater(result['triangles'],0)
                self.assertEqual(scene_from_html(page)['format'],extension[1:].upper())

    def test_box_texture_checksum_rejection(self):
        from copy import deepcopy
        with tempfile.TemporaryDirectory() as out:
            out=Path(out)
            archive=out/'broken.box.zip'
            with zipfile.ZipFile(FIXTURES/'synthetic.box.zip') as source, zipfile.ZipFile(archive,'w') as target:
                for name in source.namelist():
                    data=source.read(name)
                    if name=='maps/colorMap.rgba16f.bin':
                        data=b'x'+data[1:]
                    target.writestr(name,data)
            with self.assertRaisesRegex(ValueError,'checksum mismatch'):
                build(archive,out/'viewer.html')

    def test_does_not_write_input_file(self):
        with tempfile.TemporaryDirectory() as out:
            source=FIXTURES/'material_study.glb'
            before=hashlib.sha256(source.read_bytes()).hexdigest()
            build(source,Path(out)/'safe.html')
            self.assertEqual(before,hashlib.sha256(source.read_bytes()).hexdigest())


if __name__=='__main__':unittest.main()
