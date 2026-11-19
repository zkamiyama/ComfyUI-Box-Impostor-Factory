"""Visualize native B/C rebaked axis-depth on shared world-unit scales.
No MoGe prediction image, per-image normalization, sharpening, or AI processing.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BG=(21,24,30)
FG=(235,239,245)
SUB=(175,188,205)
B_ONLY=(239,151,56)
C_ONLY=(45,205,222)
BOTH=(103,115,131)
BLUE=np.array([48,120,224],float)
WHITE=np.array([231,234,240],float)
RED=np.array([232,91,48],float)
font=lambda size:ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",size)
def sha(path):
    with Path(path).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def scalar_rgb(values,mask,lo,hi,difference=False):
    t=np.clip((np.nan_to_num(values,nan=lo)-lo)/(hi-lo),0,1)
    if difference:
        rgb=np.where((t<=.5)[...,None],BLUE+(WHITE-BLUE)*(t*2)[...,None],
                     WHITE+(RED-WHITE)*((t-.5)*2)[...,None])
    else:
        gray=(1-t)*255
        rgb=np.repeat(gray[...,None],3,axis=-1)
    rgb=np.rint(rgb).astype("u1")
    rgb[~mask]=BG
    return Image.fromarray(rgb)
def coverage_rgb(b,c):
    rgb=np.zeros(b.shape+(3,),dtype="u1");rgb[:]=BG
    rgb[b&c]=BOTH;rgb[b&~c]=B_ONLY;rgb[c&~b]=C_ONLY
    return Image.fromarray(rgb)
def physical_preview(image,width_world,height_world,max_size):
    ratio=width_world/height_world
    mw,mh=max_size
    w=min(mw,max(1,round(mh*ratio)))
    h=min(mh,max(1,round(w/ratio)))
    return image.resize((w,h),Image.Resampling.NEAREST)
def paste_center(board,image,box):
    x,y,w,h=box
    board.paste(image,(x+(w-image.width)//2,y+(h-image.height)//2))
def bar(board,x,y,width,height,lo,hi,difference=False):
    v=np.broadcast_to(np.linspace(lo,hi,width),(height,width))
    board.paste(scalar_rgb(v,np.ones_like(v,dtype=bool),lo,hi,difference),(x,y))
    d=ImageDraw.Draw(board)
    d.text((x,y+height+4),f"{lo:.4f}",font=font(12),fill=SUB)
    end=f"{hi:.4f}"
    d.text((x+width-d.textlength(end,font=font(12)),y+height+4),end,font=font(12),fill=SUB)
    if difference:
        d.text((x+width//2-5,y+height+4),"0",font=font(12),fill=SUB)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("b",type=Path);p.add_argument("c",type=Path)
    p.add_argument("work",type=Path);p.add_argument("output",type=Path)
    a=p.parse_args()
    assets={"B":a.b,"C":a.c}
    manifests={s:json.loads((path/"manifest.json").read_text()) for s,path in assets.items()}
    b,c=manifests["B"],manifests["C"]
    assert b["bakeBox"]==c["bakeBox"] and b["transform"]==c["transform"]
    bounds=b["bakeBox"];span=np.asarray(bounds["max"])-bounds["min"]
    height=float(b["tightBounds"]["max"][1]-b["tightBounds"]["min"][1])
    delta_limit=.02*height
    faces=["+Z","+X","-X","+Y","-Y","-Z"]
    arrays={};records={}
    for face in faces:
        records[face]={"sources":{}}
        for stage,path in assets.items():
            f=manifests[stage]["faces"][face]
            assert f["depthEncoding"]=="linear-axis-0-near-1-far;nan-background"
            assert f["imageRowOrder"]=="top-to-bottom"
            for field in ("frustum","viewMatrix","projectionMatrix","side","up"):
                assert f[field]==b["faces"][face][field],(face,field)
            guide_path=path/f["guide"];depth_path=path/f["depth"]
            assert sha(guide_path)==f["guideSha256"] and sha(depth_path)==f["depthSha256"]
            image=Image.open(guide_path).convert("RGBA")
            assert image.size==(1024,1024)
            depth=np.fromfile(depth_path,dtype="<f2").reshape(image.height,image.width).astype("f4")
            mask=(np.asarray(image)[...,3]>250)&np.isfinite(depth)
            assert np.all((depth[mask]>=0)&(depth[mask]<=1))
            axis_span=float(np.abs(f["side"])@span)
            arrays[stage,face]={"depth":depth*axis_span,"mask":mask}
            records[face]["sources"][stage]={"guidePath":str(guide_path),"guideSHA256":sha(guide_path),
                                               "depthPath":str(depth_path),"depthSHA256":sha(depth_path),
                                               "validPixels":int(mask.sum())}
        bm=arrays["B",face]["mask"];cm=arrays["C",face]["mask"];common=bm&cm
        delta=arrays["C",face]["depth"]-arrays["B",face]["depth"]
        frustum=b["faces"][face]["frustum"]
        records[face].update({
            "side":b["faces"][face]["side"],"depthRangeWorld":[0,axis_span],
            "guidePlaneWorldWidthHeight":[frustum[1]-frustum[0],frustum[3]-frustum[2]],
            "commonPixels":int(common.sum()),"bOnlyPixels":int((bm&~cm).sum()),"cOnlyPixels":int((cm&~bm).sum()),
            "geometryMaskIoU":float(common.sum()/np.count_nonzero(bm|cm)),
            "absoluteWorldDepthDifferenceQuantiles":dict(zip(["p50","p95","p99","max"],
                  np.quantile(np.abs(delta[common]),[.5,.95,.99,1]).tolist())),
            "signedWorldDepthDifferenceMean":float(delta[common].mean()),
            "deltaDisplayRangeWorld":[-delta_limit,delta_limit],
            "deltaClippedCommonPixels":int(np.count_nonzero(np.abs(delta[common])>delta_limit)),
            "deltaClippedCommonFraction":float(np.mean(np.abs(delta[common])>delta_limit)),
        })
    proof={
        "kind":"Actual native mesh rebake depth comparison; no raw MoGe prediction",
        "assets":{s:{"path":str(path),"assetId":path.name,"manifestSHA256":sha(path/"manifest.json")} for s,path in assets.items()},
        "bakeBox":bounds,"canonicalHeight":height,"cameraAndBakeBoxIdentical":True,
        "encoding":"little-endian float16 normalized axis depth d, 0=near box face, 1=opposite box face; NaN background",
        "worldDepthFormula":"inwardDepthWorld = d * dot(abs(side), bakeBox.max - bakeBox.min)",
        "differenceFormula":"C minus B inward axis depth on B/C common valid mask; positive means deeper, not surface motion toward the camera",
        "mask":"guide RGBA alpha > 250 and finite depth; same rule for B/C",
        "colorScales":"Each face uses one shared B/C range [0, axis span] in canonical bake units. Every difference panel uses the same fixed +/-0.02 * B height range.",
        "differenceClipping":"Clipped values use endpoint colors and are counted per face; no independent per-image normalization.",
        "units":"Canonical bake units, not measured meters. Mask/point differences are changes, not ground-truth errors.",
        "display":"Grayscale depth and blue-white-orange signed difference; nearest-neighbor sampling with physical aspect restored; no sharpening.",
        "coverageLegend":{"common":BOTH,"BOnly":B_ONLY,"COnly":C_ONLY},
        "faces":records,"outputs":{},
    }
    a.output.mkdir(parents=True,exist_ok=True)
    png_dir=a.work/"depth-evidence";png_dir.mkdir(exist_ok=True)
    def save(board,stem):
        png=png_dir/(stem+".png");out=a.output/(stem+".webp")
        board.save(png)
        board.save(out,"WEBP",lossless=True)
        assert np.array_equal(np.asarray(Image.open(out)),np.asarray(board))
        proof["outputs"][out.name]={"path":str(out),"sha256":sha(out),"bytes":out.stat().st_size,
                                     "size":list(board.size),"losslessPixelsEqual":True,
                                     "pngPath":str(png),"pngSHA256":sha(png)}
    # Front head: one shared crop derived from the previously published A world region.
    roi_path=a.output/"A-full-comparison-provenance.json"
    head=np.asarray(json.loads(roi_path.read_text())["headWorldBounds"],float)
    original_head=head.copy();head[1,1]=bounds["max"][1]
    f=b["faces"]["+Z"];l,r,bot,top=f["frustum"][:4]
    crop=[max(0,int(np.floor((head[0,0]-l)/(r-l)*1024))),
          max(0,int(np.floor((top-head[1,1])/(top-bot)*1024))),
          min(1024,int(np.ceil((head[1,0]-l)/(r-l)*1024))),
          min(1024,int(np.ceil((top-head[0,1])/(top-bot)*1024)))]
    x0,y0,x1,y1=crop
    sl=np.s_[y0:y1,x0:x1]
    dw=(x1-x0)/1024*(r-l);dh=(y1-y0)/1024*(top-bot)
    bm=arrays["B","+Z"]["mask"][sl];cm=arrays["C","+Z"]["mask"][sl];common=bm&cm
    bd=arrays["B","+Z"]["depth"][sl];cd=arrays["C","+Z"]["depth"][sl]
    axis_span=records["+Z"]["depthRangeWorld"][1]
    images=[scalar_rgb(bd,bm,0,axis_span),scalar_rgb(cd,cm,0,axis_span),
            scalar_rgb(cd-bd,common,-delta_limit,delta_limit,True),coverage_rgb(bm,cm)]
    board=Image.new("RGB",(1240,660),BG);d=ImageDraw.Draw(board)
    d.text((20,16),"B / C native rebaked depth: front head",font=font(26),fill=FG)
    d.text((20,55),"Shared world scales and crop; actual mesh depth only. Raw MoGe predictions are not shown.",font=font(16),fill=SUB)
    labels=["B / inward axis depth","C / inward axis depth","C - B / common mask","Geometry coverage"]
    for i,(im,label) in enumerate(zip(images,labels)):
        d.text((20+i*305,95),label,font=font(17),fill=FG)
        preview=physical_preview(im,dw,dh,(285,400))
        paste_center(board,preview,(20+i*305,125,285,400))
    for i in (0,1):bar(board,20+i*305,544,280,14,0,axis_span)
    bar(board,630,544,280,14,-delta_limit,delta_limit,True)
    for j,(label,color) in enumerate([("Both",BOTH),("B only",B_ONLY),("C only",C_ONLY)]):
        d.rectangle((935,538+j*24,949,552+j*24),fill=color)
        d.text((958,537+j*24),label,font=font(14),fill=SUB)
    d.text((20,605),"Units: canonical bake units. White depth = near. Blue difference = closer; orange = deeper.",font=font(14),fill=SUB)
    clipped=float(np.mean(np.abs((cd-bd)[common])>delta_limit))
    d.text((20,628),f"Depth 0..{axis_span:.5f} for both B/C; delta +/-{delta_limit:.5f}, head delta saturation {clipped:.2%}. Physical aspect; nearest-neighbor.",font=font(14),fill=SUB)
    proof["frontHead"]={"sourceWorldROI":original_head.tolist(),"worldROIAfterTopPadding":head.tolist(),"nativeGuideCrop":crop,
                         "cropDefinition":"A world head region, expanded upward to shared bakeBox top for both stages",
                         "sourceROIProofSHA256":sha(roi_path),"physicalWidthHeight":[dw,dh],"deltaClippedFraction":clipped}
    save(board,"B-C-depth-front")
    # All six faces, one shared B/C scale per axis and one delta scale across all rows.
    board=Image.new("RGB",(1250,2130),BG);d=ImageDraw.Draw(board)
    d.text((20,15),"B / C native depth: all six directions",font=font(26),fill=FG)
    d.text((20,54),"Actual rebakes; shared per-axis depth ranges. Common delta scale +/-0.02 height; no per-image normalization.",font=font(16),fill=SUB)
    for col,label in enumerate(["B depth","C depth","C - B depth","Coverage"]):d.text((135+col*275,91),label,font=font(18),fill=FG)
    for row,face in enumerate(faces):
        y=125+row*310;record=records[face];dw,dh=record["guidePlaneWorldWidthHeight"]
        bm=arrays["B",face]["mask"];cm=arrays["C",face]["mask"];common=bm&cm
        bd=arrays["B",face]["depth"];cd=arrays["C",face]["depth"];span_f=record["depthRangeWorld"][1]
        ims=[scalar_rgb(bd,bm,0,span_f),scalar_rgb(cd,cm,0,span_f),
             scalar_rgb(cd-bd,common,-delta_limit,delta_limit,True),coverage_rgb(bm,cm)]
        d.text((20,y),face,font=font(22),fill=FG)
        d.text((20,y+34),"Depth range",font=font(12),fill=SUB)
        d.text((20,y+52),f"0..{span_f:.4f}",font=font(12),fill=SUB)
        d.text((20,y+82),"Mask IoU",font=font(12),fill=SUB)
        d.text((20,y+100),f"{record['geometryMaskIoU']:.4f}",font=font(12),fill=SUB)
        d.text((20,y+130),"Delta clip",font=font(12),fill=SUB)
        d.text((20,y+148),f"{record['deltaClippedCommonFraction']:.2%}",font=font(12),fill=SUB)
        for col,im in enumerate(ims):
            preview=physical_preview(im,dw,dh,(255,275))
            paste_center(board,preview,(130+col*275,y,265,275))
    bar(board,135,2000,420,14,-delta_limit,delta_limit,True)
    d.text((135,2038),"Difference: blue = closer; orange = deeper.",font=font(14),fill=SUB)
    for j,(label,color) in enumerate([("Both",BOTH),("B only",B_ONLY),("C only",C_ONLY)]):
        d.rectangle((665+j*170,2000,682+j*170,2017),fill=color)
        d.text((690+j*170,1999),label,font=font(15),fill=SUB)
    d.text((20,2083),"Masks: alpha > 250 and finite float16 depth. Difference uses overlap only; coverage changes are separate.",font=font(14),fill=SUB)
    save(board,"B-C-depth-six-view")
    proof["script"]={"path":str(Path(__file__).resolve()),"sha256":sha(__file__)}
    path=a.output/"B-C-depth-provenance.json"
    path.write_text(json.dumps(proof,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"outputs":proof["outputs"],"provenance":str(path),"provenanceSHA256":sha(path),
                      "frontCrop":crop,"perFace":{face:{"geometryMaskIoU":record["geometryMaskIoU"],
                      "depthDeltaP95":record["absoluteWorldDepthDifferenceQuantiles"]["p95"]} for face,record in records.items()}}))
if __name__=="__main__":main()
