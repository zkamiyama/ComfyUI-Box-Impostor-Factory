"""Compare B/D raw -X with one identical source-pixel crop, before Box."""
import argparse,hashlib,json,shutil
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
p=argparse.ArgumentParser(description=__doc__)
p.add_argument("run",type=Path);p.add_argument("d_raw",type=Path);p.add_argument("output",type=Path)
a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True)
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
read=lambda path:json.loads(Path(path).read_text())
ref_path=a.output/"A-six-view-source-provenance.json"
crop=read(ref_path)["headCrops"]["-X"]
paths={"B":a.run/"B/nx/generated.png","D":a.d_raw}
expected={"B":"b1252bf14ca410884ef8f2cb2214893a372e994b59f31291995d73cf239f5103",
          "D":"290d900dc6a71d2542ff2fe17c77bb8db590bfe116d86b6caab693ab2a922025"}
recipes={s:read(a.run/s/"prepared.json") for s in ("B","D")}
keys=("models","sampler","pictureOrder","pictureSize","outputSize","baseline","proxySha256","bakeBox","transform")
for key in keys:assert recipes["B"][key]==recipes["D"][key],key
assert recipes["B"]["seeds"]["-X"]==recipes["D"]["seeds"]["-X"]==102
pictures={s:recipes[s]["faces"]["-X"]["pictures"] for s in ("B","D")}
assert {k:v["sha256"] for k,v in pictures["B"].items()}=={k:v["sha256"] for k,v in pictures["D"].items()}
assert recipes["B"]["faces"]["-X"].get("prompt")==recipes["D"]["faces"]["-X"].get("prompt")
records,images={},{}
for stage,path in paths.items():
    assert sha(path)==expected[stage]
    copy=a.output/f"{stage}-nx-raw.png"
    if not copy.exists():shutil.copyfile(path,copy)
    assert sha(copy)==sha(path)
    images[stage]=Image.open(path).convert("RGB")
    assert images[stage].size==(1024,1024)
    recipe=a.run/stage/"prepared.json"
    records[stage]={"rawPath":str(path),"rawSHA256":sha(path),"unchangedCopy":str(copy),"copySHA256":sha(copy),
                    "recipePath":str(recipe),"recipeSHA256":sha(recipe),"conditioning":recipes[stage]["conditioning"],
                    "guide":pictures[stage]["guide"],"seed":102}
BG,FG,SUB=(21,24,30),(237,241,247),(177,192,211)
font=lambda size:ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",size)
w,h=(crop[2]-crop[0])*2,(crop[3]-crop[1])*2
cw=max(w+24,470)
board=Image.new("RGB",(2*cw+40,h+530),BG);d=ImageDraw.Draw(board)
d.text((20,16),"B / D -X AI raw outputs, before Box composition",font=font(26),fill=FG)
d.text((20,55),"Same inputs, model, prompt and seed; VAE references changed from five to the first two.",font=font(17),fill=SUB)
labels={"B":"B / VLM 5 + VAE 5 references","D":"D / VLM 5 + VAE first 2 references"}
for i,stage in enumerate(("B","D")):
    x=20+i*cw
    d.text((x,102),labels[stage],font=font(20),fill=FG)
    im=images[stage].crop(crop).resize((w,h),Image.Resampling.NEAREST)
    board.paste(im,(x+(cw-w)//2,136))
    d.text((x,h+156),"Complete raw frame / 1/4 nearest",font=font(16),fill=SUB)
    board.paste(images[stage].resize((256,256),Image.Resampling.NEAREST),(x+(cw-256)//2,h+188))
d.text((20,h+465),f"Head crop {crop}: 2x nearest-neighbor. No registration, sharpening or color edits.",font=font(16),fill=SUB)
d.text((20,h+493),"Raw PNGs copied unchanged. 1024px / -X seed 102 / 6 steps / CFG 1.",font=font(16),fill=SUB)
out=a.output/"B-D-nx-raw-proof.webp";board.save(out,"WEBP",lossless=True)
decoded=np.asarray(Image.open(out).convert("RGB"))
assert np.array_equal(decoded,np.asarray(board))
for i,stage in enumerate(("B","D")):
    x=20+i*cw+(cw-w)//2
    assert np.array_equal(decoded[136:136+h:2,x:x+w:2],np.asarray(images[stage].crop(crop)))
proof={"kind":"B/D -X raw generation comparison before Box","isBoxRender":False,"stages":records,
       "headCrop":crop,"headCropReference":{"path":str(ref_path),"sha256":sha(ref_path)},
       "controlledSettingsEqual":{k:recipes["B"][k] for k in keys},
       "fiveInputPictureSHA256":{k:v["sha256"] for k,v in pictures["B"].items()},
       "singleDeclaredChange":"VAE reference pictures 1-5 -> pictures 1-2; VLM retains all five",
       "experiment":recipes["D"].get("experiment"),"pixelPreservationVerifiedAfterWebPDecode":True,
       "processing":"Identical crop, 2x nearest; full raw frames 1/4 nearest. No registration, sharpening, recoloring or AI postprocessing.",
       "limitation":"One raw direction; does not establish final Box quality or six-view consistency.",
       "panel":{"path":str(out),"sha256":sha(out),"bytes":out.stat().st_size,"size":list(board.size)},
       "scriptSHA256":sha(__file__)}
proof_path=a.output/"B-D-nx-raw-proof-provenance.json";proof_path.write_text(json.dumps(proof,indent=2,allow_nan=False)+"\n")
print(json.dumps({"panel":proof["panel"],"provenanceSHA256":sha(proof_path),"raws":records}))
