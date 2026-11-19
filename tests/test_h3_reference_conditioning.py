import types
import unittest
from unittest.mock import patch

from box_nodes.h3_reference_conditioning import H3ReferenceConditioning


class NativeFixture:
    IMAGE_REF_SHORT_EDGE=1024
    IMAGE_REF_VAE_MAX_SHORT_EDGE=2048

    def __init__(self):
        self.events=[]
        self.MiniMaxH3ImageConditioning=types.SimpleNamespace(execute=self.execute)
        self.node_helpers=types.SimpleNamespace(conditioning_set_values=
            lambda cond,values:[[embedding,{**meta,**values}] for embedding,meta in cond])

    @staticmethod
    def _size_short_edge(w,h,low,high):
        return w,h,low,high

    def _image_refs(self,images,vae,vision_size,vae_size):
        ordered=[images[key] for key in sorted(images) if images[key] is not None]
        refs=[]
        if vae is not None:
            self.events.append(('vae',list(ordered)))
            refs=[{'kind':'image','value':v,'resizeRule':vae_size(320,640)} for v in ordered]
        return ordered,list(ordered),refs

    def execute(self,clip,prompt,width,height,vae=None,images=None):
        ordered,vision,refs=self._image_refs(images,vae,
            lambda w,h:self._size_short_edge(w,h,1024,1024),
            lambda w,h:self._size_short_edge(w,h,1024,2048))
        self.events.append(('vlm',vision))
        metadata={'prompt':prompt,'nativeMarker':23}
        if refs:metadata['minimax_refs']=refs
        self.last=types.SimpleNamespace(result=([[tuple(vision),metadata]],{'size':(width,height)}))
        return self.last


class H3ReferenceConditioningTests(unittest.TestCase):
    def setUp(self):
        self.native=NativeFixture()
        patcher=patch('box_nodes.h3_reference_conditioning.importlib.import_module',return_value=self.native)
        patcher.start();self.addCleanup(patcher.stop)
        self.node=H3ReferenceConditioning()
        self.inputs={'clip':object(),'vae':object(),'prompt':'Pictures 1 through 5',
                     'width':1024,'height':1024,
                     **{f'image_{i}':v for i,v in enumerate([17,23,59,71,89],1)}}

    def test_default_is_unchanged_native_five_reference_result(self):
        result=self.node.condition(**self.inputs)
        self.assertIs(result,self.native.last.result)
        self.assertEqual(self.native.events,[('vae',[17,23,59,71,89]),('vlm',[17,23,59,71,89])])
        self.assertEqual(len(result[0][0][1]['minimax_refs']),5)
        self.assertEqual(self.node.INPUT_TYPES()['required']['reference_latents'][1]['default'],'all')

    def test_only_vae_references_change_and_picture_numbers_and_loading_order_stay(self):
        original=self.node.condition(**self.inputs)
        self.native.events.clear()
        split=self.node.condition(**self.inputs,reference_latents='color_first_two')
        self.assertEqual(self.native.events,[('vae',[17,23]),('vlm',[17,23,59,71,89])])
        self.assertEqual(split[0][0][0],original[0][0][0])
        self.assertEqual(split[1],original[1])
        self.assertEqual(split[0][0][1]['minimax_refs'],original[0][0][1]['minimax_refs'][:2])
        self.assertEqual(len(original[0][0][1]['minimax_refs']),5)
        self.assertEqual(split[0][0][1]['nativeMarker'],23)
        self.assertEqual(split[0][0][1]['prompt'],self.inputs['prompt'])

    def test_color_slots_cannot_be_replaced_by_later_structural_slots(self):
        for changes in ({'image_1':None},{'image_2':None},{'vae':None}):
            with self.subTest(changes=changes),self.assertRaisesRegex(ValueError,'Pictures 1 and 2'):
                self.node.condition(**{**self.inputs,**changes},reference_latents='color_first_two')
        self.assertEqual(self.native.events,[])

    def test_native_vision_only_mode_and_invalid_mode(self):
        cond,_=self.node.condition(**{**self.inputs,'vae':None})
        self.assertNotIn('minimax_refs',cond[0][1])
        self.assertEqual(self.native.events,[('vlm',[17,23,59,71,89])])
        with self.assertRaisesRegex(ValueError,'Unknown'):
            self.node.condition(**self.inputs,reference_latents='unsupported')


if __name__=='__main__':
    unittest.main()
