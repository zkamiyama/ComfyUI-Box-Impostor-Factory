import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
if str(SCRIPTS) not in sys.path:sys.path.insert(0,str(SCRIPTS))
import run_geometry_ab as ab
import run_geometry_d as d


def recipe():
    return {'branch':'B','baseline':'baseline','baselinePath':'unused',
      'models':{'model':'same-model','clip':'same-clip','vae':'same-vae','matting':'same-matting'},
      'pictureOrder':['guide','source','axis-depth','normals','mask'],
      'seeds':{'-X':102},'sampler':{'steps':6,'cfg':1,'sampler_name':'euler','scheduler':'simple','denoise':1},
      'faces':{'-X':{'prompt':'Same five pictures and exact prompt.',
        'pictures':{role:{'image':f'B/nx/{role}.png'} for role in ['guide','source','axis-depth','normals','mask']}}}}


class GeometryDTests(unittest.TestCase):
    def test_graph_changes_only_conditioning_latent_policy_and_output_location(self):
        b=recipe();before=copy.deepcopy(b)
        source=ab.build_prompt(b,'-X');target=copy.deepcopy(b);target['branch']='D'
        graph=d.build_prompt(target,'-X')
        self.assertEqual(b,before)
        self.assertEqual(set(graph),set(source))
        for key in set(source)-{'10','14','18'}:
            self.assertEqual(graph[key],source[key],key)
        self.assertEqual(graph['10']['class_type'],'BIH3ReferenceConditioning')
        inputs=graph['10']['inputs'];original=source['10']['inputs']
        for key in ['clip','vae','prompt','width','height']:
            self.assertEqual(inputs[key],original[key])
        for i in range(1,6):
            self.assertEqual(inputs[f'image_{i}'],original[f'images.image_{i}'])
        self.assertEqual(inputs['reference_latents'],'color_first_two')
        self.assertEqual(graph['14']['inputs']['filename_prefix'],'D/nx/raw')
        self.assertEqual(graph['18']['inputs']['filename_prefix'],'D/nx/mask')

    def test_preparation_freezes_B_images_and_refuses_changed_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);b=recipe()
            for role,picture in b['faces']['-X']['pictures'].items():
                path=root/f'{role}.png';path.write_bytes(role.encode())
                picture.update(path=str(path),sha256=ab.digest(path))
            raw=root/'B-raw.png';raw.write_bytes(b'original raw')
            ab.write_json(root/'B/prepared.json',b)
            ab.write_json(root/'B/composed.json',{'sourceBaseline':'baseline','asset':'B-result',
                'generatedFaces':['+X','-X','+Y','-Y','+Z','-Z']})
            ab.write_json(root/'B/nx/result.json',{'raw':str(raw),'rawSha256':ab.digest(raw)})
            prepared=d.prepare(root,'-X')
            self.assertEqual(prepared['faces'],b['faces'])
            self.assertEqual(prepared['models'],b['models'])
            self.assertEqual(prepared['sampler'],b['sampler'])
            self.assertEqual(prepared['seeds'],b['seeds'])
            self.assertEqual(prepared['experiment']['vlmPictureIndices'],[1,2,3,4,5])
            self.assertEqual(prepared['experiment']['vaePictureIndices'],[1,2])
            self.assertEqual(d.prepare(root,'-X'),prepared)
            Path(b['faces']['-X']['pictures']['normals']['path']).write_bytes(b'changed control')
            with self.assertRaisesRegex(ValueError,'conditioning image changed'):
                d.prepare(root,'-X')


class GeometryDComposeTests(unittest.TestCase):
    def setUp(self):
        from test_nodes import fixture
        self.run=Path(self.enterContext(tempfile.TemporaryDirectory()))
        assets=self.run/'output/box_impostor';assets.mkdir(parents=True)
        self.enterContext(patch.object(ab.store,'root',return_value=assets))
        self.enterContext(patch.object(ab.store,'check_interrupted',return_value=None))
        self.enterContext(patch.object(ab.folder_paths,'set_output_directory'))
        self.base='a'*64
        fixture(assets/self.base,self.base)
        candidates={}
        for index,fid in enumerate(ab.store.FACES):
            rgba=np.zeros((8,8,4),dtype='f4')
            rgba[...,:3]=(.1+index*.08,.2,.5-index*.04)
            rgba[1:-1,1:-1,3]=1
            candidates[fid]=ab.faces.save_view(rgba)
        self.b=ab.faces.compose(self.base,ab.faces.align(candidates,self.base,.85,.10,.15))
        self.recipe=recipe();self.recipe.update(branch='D',baseline=self.base,
            conditioning='five VLM, two VAE',experiment={'sourceBox':self.b})
        self.case=self.run/'D/nx';self.case.mkdir(parents=True)
        Image.new('RGB',(8,8),(245,30,80)).save(self.case/'generated.png')
        self.mask=np.zeros((8,8),dtype='u1');self.mask[1:-1,1:-1]=255

    def result(self,mask):
        Image.fromarray(mask).save(self.case/'foreground.png')
        ab.write_json(self.case/'prompt.json',d.build_prompt(self.recipe,'-X'))
        ab.write_json(self.case/'result.json',{'prompt_id':'fresh-D-fixture',
            'rawSha256':ab.digest(self.case/'generated.png'),
            'maskSha256':ab.digest(self.case/'foreground.png'),
            'promptSha256':ab.digest(self.case/'prompt.json')})

    def atlas(self,asset):
        root=ab.store.asset_dir(asset);m=ab.store.validate(root,asset)
        desc=m['textures']['colorMap']
        return np.fromfile(root/desc['path'],'<f2').reshape(desc['height'],desc['width'],4)

    def test_one_face_replaces_full_B_and_keeps_five_generated_views_and_geometry(self):
        self.result(self.mask)
        before=self.atlas(self.b).copy()
        proof=d.compose_replacement(self.run,self.recipe,'-X')
        self.assertTrue(proof['applied'])
        self.assertEqual(proof['sourceBox'],self.b)
        self.assertNotEqual(proof['asset'],self.b)
        after=self.atlas(proof['asset'])
        self.assertFalse(np.array_equal(ab.faces.tile_view(before,'-X'),ab.faces.tile_view(after,'-X')))
        for fid in ab.store.FACES:
            if fid!='-X':
                np.testing.assert_array_equal(ab.faces.tile_view(before,fid).view('u2'),ab.faces.tile_view(after,fid).view('u2'))
        np.testing.assert_array_equal(before,self.atlas(self.b))
        self.assertEqual(len(proof['preservation']['unchangedFaceRGBAHashes']),5)
        self.assertTrue(proof['preservation']['nonColorMapsExact'])
        self.assertTrue(proof['preservation']['unchangedFaceSourcesAndMetadata'])

    def test_rejected_replacement_keeps_B_instead_of_restoring_coarse_face(self):
        self.result(np.zeros_like(self.mask))
        before=self.atlas(self.b).copy()
        proof=d.compose_replacement(self.run,self.recipe,'-X')
        self.assertFalse(proof['applied'])
        self.assertEqual(proof['asset'],self.b)
        np.testing.assert_array_equal(before,self.atlas(proof['asset']))


if __name__=='__main__':unittest.main()
