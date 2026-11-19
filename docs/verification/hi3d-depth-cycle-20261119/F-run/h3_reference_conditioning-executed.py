"""Opt-in H3 conditioning for color references plus structural pictures."""
import importlib


class H3ReferenceConditioning:
    CATEGORY='Box Impostor/Experimental'
    RETURN_TYPES=('CONDITIONING','LATENT')
    RETURN_NAMES=('positive','latent')
    FUNCTION='condition'

    @classmethod
    def INPUT_TYPES(cls):
        required={
            'clip':('CLIP',),
            'prompt':('STRING',{'multiline':True,'dynamicPrompts':True}),
            'width':('INT',{'default':0,'min':0,'max':16384,'step':32}),
            'height':('INT',{'default':0,'min':0,'max':16384,'step':32}),
            'reference_latents':(['all','color_first_two','color_only'],{'default':'all',
                'tooltip':'All anchors all five pictures. Color first two anchors the color guide and source photo; structural pictures still condition the vision/text encoder. Color only sends Pictures 1 and 2 to both encoders.'}),
        }
        labels=['color guide','source photo','depth guide','normal guide','foreground mask']
        optional={'vae':('VAE',)}
        for index,label in enumerate(labels,1):
            inputs=required if index<=2 else optional
            inputs[f'image_{index}']=('IMAGE',{'tooltip':f'Picture {index}: {label}. Picture numbers are preserved.'})
        return {'required':required,'optional':optional}

    def condition(self,clip,prompt,width,height,image_1,image_2,image_3=None,image_4=None,image_5=None,
                  reference_latents='all',vae=None):
        if reference_latents not in ('all','color_first_two','color_only'):
            raise ValueError('Unknown H3 reference latent mode.')
        images={f'image_{i}':image for i,image in enumerate(
            (image_1,image_2,image_3,image_4,image_5),1)}
        if reference_latents!='all' and (vae is None or image_1 is None or image_2 is None):
            raise ValueError('Color reference modes require a VAE and Pictures 1 and 2.')
        if reference_latents=='color_only':
            images={'image_1':image_1,'image_2':image_2}
        native=importlib.import_module('comfy_extras.nodes_minimax_h3')
        execute=native.MiniMaxH3ImageConditioning.execute
        if reference_latents in ('all','color_only'):
            return execute(clip,prompt,width,height,vae=vae,images=images).result

        # Preserve native resize rules and VAE-before-Qwen model loading order.
        vision_size=lambda w,h:native._size_short_edge(w,h,native.IMAGE_REF_SHORT_EDGE,native.IMAGE_REF_SHORT_EDGE)
        vae_size=lambda w,h:native._size_short_edge(w,h,native.IMAGE_REF_SHORT_EDGE,native.IMAGE_REF_VAE_MAX_SHORT_EDGE)
        _,_,refs=native._image_refs(
            {'image_1':image_1,'image_2':image_2},vae,vision_size,vae_size)
        positive,latent=execute(clip,prompt,width,height,vae=None,images=images).result
        positive=native.node_helpers.conditioning_set_values(positive,{'minimax_refs':refs})
        return positive,latent
