from pathlib import Path
import json,io,zipfile,hashlib
import numpy as np
from PIL import Image, ImageDraw
import trimesh

out=Path(__file__).parent/'fixtures';out.mkdir(parents=True,exist_ok=True)
# A textured glb, with three distinct object materials.
scene=trimesh.Scene()
palette=[(204,142,91,255),(101,155,177,255),(195,193,185,255)]
for i,color in enumerate(palette):
    tex=Image.new('RGBA',(128,128),color)
    d=ImageDraw.Draw(tex)
    for j in range(0,128,16):d.line((j,0,128-j,128),fill=(30+i*40,40,50,150),width=3)
    geom=trimesh.creation.box(extents=[.7,.65,.65]) if i==0 else (trimesh.creation.icosphere(subdivisions=2,radius=.35) if i==1 else trimesh.creation.cylinder(radius=.3,height=.8,sections=24))
    uv=np.zeros((len(geom.vertices),2))
    p=geom.vertices
    uv[:,0]=(p[:,0]-p[:,0].min())/max(.001,np.ptp(p[:,0]))
    uv[:,1]=(p[:,1]-p[:,1].min())/max(.001,np.ptp(p[:,1]))
    m=trimesh.visual.material.PBRMaterial(name=['Ceramic glaze','Anodized blue','Brushed aluminum'][i],
            baseColorFactor=np.array([255,255,255,255],dtype=np.uint8),
            baseColorTexture=tex, metallicFactor=[0,.6,.95][i],roughnessFactor=[.5,.3,.25][i])
    geom.visual=trimesh.visual.TextureVisuals(uv=uv,material=m)
    geom.apply_translation([(i-1)*.9,0,0])
    scene.add_geometry(geom,node_name=['Box','Sphere','Cylinder'][i])
(out/'material_study.glb').write_bytes(scene.export(file_type='glb'))
# Box atlas 32px/face; synthetic non-uniform depth & four half-float maps.
r=32
rng=np.random.default_rng(5)
color=np.zeros((r*2,r*3,4),dtype=np.float16)
for fi in range(6):
    row,col=divmod(fi,3)
    tile=color[row*r:(row+1)*r,col*r:(col+1)*r]
    for y in range(r):
        for x in range(r):
            nx=(x+0.5)/r*2-1;ny=(y+0.5)/r*2-1
            if nx*nx+ny*ny>.76:continue
            tile[y,x,3]=1
            tile[y,x,:3]=[(fi+1)/6,x/r,y/r]
depth=np.ones((r,r),dtype=np.float16)*.55
with zipfile.ZipFile(out/'synthetic.box.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
    def put(path,bytes_):
        z.writestr(path,bytes_)
        return hashlib.sha256(bytes_).hexdigest()
    texs={}
    for name in ['colorMap','normalMap','intervalMap','rangeMap','materialMap']:
        values=color if name=='colorMap' else np.full_like(color,.5)
        data=values.astype('<f2').tobytes()
        path='maps/'+name+'.rgba16f.bin';h=put(path,data)
        texs[name]=dict(path=path,width=r*3,height=r*2,channels=4,dtype='float16',endian='little',byteLength=len(data),sha256=h)
    faces={}
    for fi,face in enumerate(['+X','-X','+Y','-Y','+Z','-Z']):
        code=['px','nx','py','ny','pz','nz'][fi]
        dp='guides/'+code+'.depth16f.bin';put(dp,depth.astype('<f2').tobytes())
        faceim=Image.new('RGBA',(r,r),(80+fi*20,100,150,255))
        buf=io.BytesIO();faceim.save(buf,format='PNG')
        guide='guides/'+code+'.png';put(guide,buf.getvalue())
        faces[face]=dict(depth=dp,guide=guide)
    proxy='geometry/proxy.glb';proxyhash=put(proxy,scene.export(file_type='glb'))
    manifest=dict(schemaVersion=1,layout='box',asset_id='a'*64,mode='geometry_baseline',resolution=r,
            faceOrder=['+X','-X','+Y','-Y','+Z','-Z'],bakeBox={'min':[-.5,-.6,-.7],'max':[.5,.6,.7]},
            textures=texs,baseColor=texs['colorMap'],proxy=proxy,proxySha256=proxyhash,faces=faces,maps=['roughness','metalness','ao'])
    z.writestr('manifest.json',json.dumps(manifest))
print('fixtures:',[(f.name,f.stat().st_size) for f in out.iterdir()])
