"""Classify K front residual pixels using existing captures, without new rendering."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy.ndimage import binary_erosion,label
from build_box_control_front import BG,TEXT,MUTED,FONT,sha,backdrop,aligned_front_crop

def components(mask,crop):
    labs,n=label(mask,structure=np.ones((3,3),bool))
    out=[]
    for i in range(1,n+1):
        yy,xx=np.where(labs==i)
        out.append({"pixels":int(len(xx)),"bboxFullImage":[int(xx.min()+crop[0]),int(yy.min()+crop[1]),int(xx.max()+1+crop[0]),int(yy.max()+1+crop[1])]})
    return sorted(out,key=lambda d:-d["pixels"])

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("work",type=Path);p.add_argument("output",type=Path)
    a=p.parse_args()
    fidelity_path=a.output/"F-J-K-front-fidelity.json"
    fidelity=json.loads(fidelity_path.read_text())
    meta_path=a.work/"F-provenance.json";meta=json.loads(meta_path.read_text())
    crop=fidelity["headCrop"];camera=fidelity["camera"];reference_info=fidelity["reference"]
    assert sha(reference_info["path"])==reference_info["sha256"]
    reference=aligned_front_crop(reference_info["path"],meta["bakeBox"],crop,camera)
    ref=np.asarray(reference)
    full,images={},{}
    for stage,data in fidelity["inputs"].items():
        entry=data["front"];assert sha(entry["path"])==entry["sha256"]
        full[stage]=np.asarray(Image.open(entry["path"]).convert("RGBA"))
        images[stage]=np.asarray(Image.fromarray(full[stage],"RGBA").crop(crop))
    assert np.array_equal(images["K-view"],images["K-dominant"])
    k=images["K-view"];source_opaque=ref[...,3]>250
    difference=np.abs(k[...,:3].astype(np.int16)-ref[...,:3].astype(np.int16))
    mae=difference.mean(axis=-1);max_channel=difference.max(axis=-1)
    common=source_opaque.copy()
    for im in images.values():common &= im[...,3]>250
    old_mask=binary_erosion(common,np.ones((3,3),bool),iterations=3,border_value=0)
    old_mae=float(difference[old_mask].mean())
    assert int(old_mask.sum())==fidelity["mask"]["pixelsAfterErosion"]
    assert abs(old_mae-fidelity["metrics"]["K-view"]["sRGBMAE255"])<1e-10
    regions={"head":np.ones(source_opaque.shape,bool)}
    # Fixed, visibly selected face rectangle; no semantic skin segmentation.
    face_rect=[478,165,543,231]
    face=np.zeros(source_opaque.shape,bool)
    face[face_rect[1]-crop[1]:face_rect[3]-crop[1],face_rect[0]-crop[0]:face_rect[2]-crop[0]]=True
    regions["faceRectangle"]=face
    thresholds=[8,16,32]
    results={}
    for radius in [0,1,3,5]:
        inner=source_opaque if radius==0 else binary_erosion(source_opaque,np.ones((3,3),bool),iterations=radius,border_value=0)
        results[str(radius)]={}
        for region_name,region in regions.items():
            mask=inner&region;opaque=mask&(k[...,3]>250);transparent=mask&(k[...,3]<=5)
            partial=mask&(k[...,3]>5)&(k[...,3]<=250)
            counts={"sourceOpaqueInteriorPixels":int(mask.sum()),"renderTransparentAlpha0to5":int(transparent.sum()),
                    "renderPartialAlpha6to250":int(partial.sum()),"renderOpaqueAlpha251to255":int(opaque.sum()),
                    "opaqueRGBMAE255":float(difference[opaque].mean()) if opaque.any() else None}
            counts["thresholds"]={}
            for t in thresholds:
                bad=opaque&(mae>t);bad_max=opaque&(max_channel>t)
                counts["thresholds"][str(t)]={"opaquePerPixelMAEGreater":int(bad.sum()),"opaqueMaxChannelDifferenceGreater":int(bad_max.sum()),
                    "opaqueMeanRGBDarkeningGreater":int((opaque&((k[...,:3].astype(float)-ref[...,:3]).mean(axis=-1)<-t)).sum()),
                    "opacityDeficitOrOpaqueMAEError":int((transparent|partial|bad).sum())}
            results[str(radius)][region_name]=counts
    inner=binary_erosion(source_opaque,np.ones((3,3),bool),iterations=3,border_value=0)
    masks={"transparent":inner&(k[...,3]<=5),"partial":inner&(k[...,3]>5)&(k[...,3]<=250),
           "opaqueRGBError":inner&(k[...,3]>250)&(mae>16)}
    colors={"transparent":(245,70,95),"partial":(245,190,55),"opaqueRGBError":(80,140,255)}
    classification=np.zeros((*inner.shape,3),np.uint8);classification[inner]=(60,65,75)
    overlay=np.asarray(backdrop(Image.fromarray(k,"RGBA"))).copy()
    for key,mask in masks.items():classification[mask]=colors[key];overlay[mask]=colors[key]
    width,height=reference.size;scale=4;cell=width*scale+24
    board=Image.new("RGB",(cell*4,height*scale+190),BG);draw=ImageDraw.Draw(board)
    font=lambda n:ImageFont.truetype(FONT,n)
    draw.text((18,14),"K front residuals / existing captures only",font=font(25),fill=TEXT)
    draw.text((18,49),"Mask: reference alpha >250, eroded 3px before inspecting render alpha. No render-mask intersection.",font=font(15),fill=MUTED)
    labels=["Registered +Z reference","K actual","Pixel classification","Classification over K"]
    displays=[backdrop(reference),backdrop(Image.fromarray(k,"RGBA")),Image.fromarray(classification),Image.fromarray(overlay)]
    for i,(title,im) in enumerate(zip(labels,displays)):
        draw.text((i*cell+12,84),title,font=font(19),fill=TEXT)
        board.paste(im.resize((width*scale,height*scale),Image.Resampling.NEAREST),(i*cell+12,114))
    y=height*scale+129
    for x,(key,name) in zip([18,430,850],[( "transparent","Transparent alpha<=5"),("partial","Partial alpha 6..250"),("opaqueRGBError","Opaque RGB MAE>16")]):
        draw.rectangle((x,y,x+12,y+12),fill=colors[key])
        draw.text((x+19,y-3),name+": "+str(int(masks[key].sum())),font=font(16),fill=TEXT)
    draw.text((18,y+27),"4x nearest-neighbor; same crop. Classification does not identify a mesh/depth/shader cause.",font=font(15),fill=MUTED)
    board_path=a.output/"K-front-residual-classification.webp";board.save(board_path,"WEBP",lossless=True)
    mask_path=a.output/"K-front-source-interior-mask.png";Image.fromarray(inner.astype(np.uint8)*255).save(mask_path)
    ref_path=a.output/"K-front-reference-rgba.png";reference.save(ref_path)
    proof={"kind":__doc__,"assetId":fidelity["assetId"],"headCrop":crop,"faceRectangleFullImage":face_rect,
        "reference":reference_info,"sourceEvidence":{"path":str(fidelity_path),"sha256":sha(fidelity_path)},
        "referenceSampler":{"path":str(Path(__file__).with_name("build_box_control_front.py")),"sha256":sha(Path(__file__).with_name("build_box_control_front.py"))},
        "inputFrames":{k:v["front"] for k,v in fidelity["inputs"].items()},
        "KViewAndDominantPixelsIdentical":True,
        "renderAlphaByteIdenticalToK":{stage:bool(np.array_equal(im[...,3],full["K-view"][...,3])) for stage,im in full.items()},
        "priorCommonMaskCheck":{"pixels":int(old_mask.sum()),"sRGBMAE255":old_mae},
        "definitions":{"erosion":"3x3 all-ones binary erosion, stated iteration count, border=0, applied only to reference alpha>250",
            "transparent":"render alpha<=5","partial":"6<=render alpha<=250","opaque":"render alpha>250",
            "opaqueError":"mean absolute difference of 8-bit sRGB channels > stated threshold; alpha-deficient pixels are classified separately",
            "faceRectangle":"Fixed manually selected rectangle; not semantic skin segmentation"},
        "countsByErosionRadius":results,
        "componentsAtRadius3Threshold16":{key:components(mask,crop) for key,mask in masks.items()},
        "outputs":{p.name:{"path":str(p),"sha256":sha(p),"bytes":p.stat().st_size} for p in [board_path,mask_path,ref_path]},
        "scriptSHA256":sha(__file__),"renderingPerformed":False,"sourceAssetsModified":False,
        "limitations":["Classifies existing rendered pixels only; does not prove an underlying geometry, mesh, depth, rasterization, or shader cause.",
            "Counts depend on alpha/RGB thresholds and reference-interior mask; radius and threshold sensitivity are reported.",
            "The reference uses the established registration/camera sampling. No new alignment, exposure fitting, sharpening, or colour correction."]}
    proof_path=a.output/"K-front-residual-classification.json";proof_path.write_text(json.dumps(proof,indent=2)+"\n")
    print(json.dumps({"radius3":results["3"],"alphaEqual":proof["renderAlphaByteIdenticalToK"],"components":proof["componentsAtRadius3Threshold16"],"outputs":proof["outputs"],"proof":str(proof_path),"proofSHA256":sha(proof_path)},indent=2))

if __name__=="__main__":main()
