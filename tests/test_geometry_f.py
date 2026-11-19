import copy
from pathlib import Path
import sys
import tempfile
import unittest

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
if str(SCRIPTS) not in sys.path:sys.path.insert(0,str(SCRIPTS))
import run_geometry_ab as ab
import run_geometry_d as d
import run_geometry_f as f
from test_geometry_d import recipe


class GeometryFTests(unittest.TestCase):
    def test_prompt_removes_only_declared_structural_instructions(self):
        original=ab.face_prompt('-X')
        result=f.color_prompt(original)
        self.assertTrue(result.startswith(ab.VIEW_INSTRUCTIONS['-X']))
        self.assertIn('Use Picture 2 only for identity, skin, hair, dress colors and material details.',result)
        self.assertIn('Use neutral even lighting and a flat mid-gray background.',result)
        for token in ('<Picture 3>','<Picture 4>','<Picture 5>','structural maps','within the mask'):
            self.assertNotIn(token,result)
        with self.assertRaisesRegex(ValueError,'structural prompt changed'):
            f.color_prompt(original.replace('brighter means nearer','darker means nearer'))

    def test_graph_removes_structural_loaders_and_keeps_sampler_color_links(self):
        source=recipe();source['branch']='D';source['faces']['-X']['prompt']=ab.face_prompt('-X')
        original=copy.deepcopy(source);before=d.build_prompt(source,'-X')
        target=copy.deepcopy(source);target.update(branch='F',pictureOrder=['guide','source'])
        target['faces']['-X']['pictures']={role:source['faces']['-X']['pictures'][role]
                                          for role in ('guide','source')}
        target['faces']['-X']['prompt']=f.color_prompt(source['faces']['-X']['prompt'])
        graph=f.build_prompt(target,'-X')
        self.assertEqual(source,original)
        self.assertEqual(set(graph),set(before)-{'3','4','5'})
        for key in set(graph)-{'10','14','18'}:
            self.assertEqual(graph[key],before[key],key)
        inputs=graph['10']['inputs']
        self.assertEqual(inputs['reference_latents'],'color_only')
        for key in ('clip','vae','width','height','image_1','image_2'):
            self.assertEqual(inputs[key],before['10']['inputs'][key])
        for key in ('image_3','image_4','image_5'):
            self.assertNotIn(key,inputs)
        self.assertEqual(graph['14']['inputs']['filename_prefix'],'F/nx/raw')
        self.assertEqual(graph['18']['inputs']['filename_prefix'],'F/nx/mask')

    def test_prepare_preserves_frozen_B_D_and_refuses_changed_color(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);b=recipe();b['faces']['-X']['prompt']=ab.face_prompt('-X')
            for role,picture in b['faces']['-X']['pictures'].items():
                path=run/f'{role}.png';path.write_bytes(role.encode())
                picture.update(path=str(path),sha256=ab.digest(path))
            raw=run/'B-raw.png';raw.write_bytes(b'original B raw')
            ab.write_json(run/'B/prepared.json',b)
            ab.write_json(run/'B/composed.json',{'sourceBaseline':'baseline','asset':'B-result',
                'generatedFaces':['+X','-X','+Y','-Y','+Z','-Z']})
            ab.write_json(run/'B/nx/result.json',{'raw':str(raw),'rawSha256':ab.digest(raw)})
            d.prepare(run,'-X')
            draw=run/'D-raw.png';draw.write_bytes(b'original D raw')
            ab.write_json(run/'D/nx/result.json',{'raw':str(draw),'rawSha256':ab.digest(draw)})
            ab.write_json(run/'D/composed.json',{'sourceBox':'B-result','asset':'D-result'})
            preserved={str(path):ab.digest(path) for branch in ('B','D')
                       for path in (run/branch).rglob('*') if path.is_file()}
            prepared=f.prepare(run,'-X')
            self.assertEqual(prepared['branch'],'F')
            self.assertEqual(prepared['pictureOrder'],['guide','source'])
            self.assertEqual(set(prepared['faces']['-X']['pictures']),{'guide','source'})
            for key in ('models','sampler','seeds'):
                self.assertEqual(prepared[key],b[key])
            self.assertEqual(prepared['experiment']['sourceBox'],'B-result')
            self.assertEqual(prepared['experiment']['vlmPictureIndices'],[1,2])
            self.assertEqual(prepared['experiment']['vaePictureIndices'],[1,2])
            self.assertFalse(prepared['experiment']['promptIdenticalToD'])
            self.assertEqual(ab.read_json(run/'F/prompt-diff.json')['before'],b['faces']['-X']['prompt'])
            self.assertEqual(f.prepare(run,'-X'),prepared)
            self.assertEqual({path:ab.digest(path) for path in preserved},preserved)
            with self.assertRaisesRegex(ValueError,'limited'):
                f.prepare(run,'+X')
            Path(b['faces']['-X']['pictures']['guide']['path']).write_bytes(b'corrected color')
            with self.assertRaisesRegex(ValueError,'conditioning image changed'):
                f.prepare(run,'-X')


if __name__=='__main__':unittest.main()
