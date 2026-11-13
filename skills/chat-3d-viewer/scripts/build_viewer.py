#!/usr/bin/env python3
"""Build a single-file, network-free 3D Inspector view for ChatGPT/artifact previews.

Supports GLB/GLTF/OBJ/PLY/STL/3MF through trimesh and Box Impostor .box.zip.
The latter is deliberately an approximate geometric reconstruction, not the
three-impostor raymarch renderer. Geometry and counts are marked accordingly.
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import re
import struct
import sys
import zipfile
import zlib

import numpy as np
from PIL import Image

VERSION = 1
FORMATS = {'.glb', '.gltf', '.obj', '.stl', '.ply', '.3mf'}
FACES = ['+X', '-X', '+Y', '-Y', '+Z', '-Z']
CHANNEL_MAPS = {'colorMap': 'Base color', 'normalMap': 'Normal', 'intervalMap': 'Intervals',
                'rangeMap': 'Depth range', 'materialMap': 'R: roughness · G: metallic · B: AO',
                'emissiveMap': 'Emissive', 'baseColor': 'Unedited color'}
MAX_INPUT = 512 * 1024 * 1024
MAX_TEXTURE_SIZE = 2048


def buffer64(arr, dtype):
    return base64.b64encode(np.ascontiguousarray(arr, dtype=dtype).tobytes()).decode('ascii')


def png_uri(image: Image.Image, cap=MAX_TEXTURE_SIZE):
    im = image.convert('RGBA')
    if max(im.size) > cap:
        im.thumbnail((cap, cap), Image.Resampling.LANCZOS)
    output = io.BytesIO()
    im.save(output, format='PNG', optimize=True)
    return 'data:image/png;base64,' + base64.b64encode(output.getvalue()).decode('ascii'), im.size


def finite(array):
    return bool(np.isfinite(array).all())


def clean_name(name: str, limit=100):
    return re.sub(r'[\x00-\x1f\x7f]', '', str(name))[:limit].strip() or 'Unnamed'


def convert_texture(image, name, role, textures, *, max_size=MAX_TEXTURE_SIZE, channels=None):
    uri, size = png_uri(image, max_size)
    index = len(textures)
    textures.append(dict(name=clean_name(name), role=role, width=size[0], height=size[1],
                         format='PNG', channels=channels or 'RGBA', dataUri=uri))
    return index


def quantized_color(value):
    a = np.asarray(value, dtype=np.float64).reshape(-1)
    if len(a) < 3:
        return [1, 1, 1, 1]
    if np.max(a) > 1:
        a = a / 255
    return np.clip(list(a[:4]) + [1] * (4-len(a[:4])), 0, 1).tolist()


def glb_scene(path, *, limit=250_000, texture_size=MAX_TEXTURE_SIZE):
    try:
        import trimesh
    except ImportError as exc:
        raise SystemExit('Missing trimesh. Install with: pip install numpy pillow trimesh') from exc
    if path.stat().st_size > MAX_INPUT:
        raise ValueError('Input file exceeds the 512 MiB limit')
    loaded = trimesh.load(str(path), force='scene', process=False)
    if isinstance(loaded, trimesh.Trimesh):
        scene = trimesh.Scene(loaded)
    else:
        scene = loaded
    textures, materials, meshes = [], [], []
    material_cache = {}
    source_vertex_total = 0
    source_triangle_total = 0
    for node_name in sorted(scene.graph.nodes_geometry):
        transform, geometry_name = scene.graph.get(node_name)
        mesh = scene.geometry[geometry_name]
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
            continue
        v = np.asarray(mesh.vertices, dtype=np.float64)
        transform = np.asarray(transform, dtype=np.float64)
        vertices = np.c_[v, np.ones(len(v))] @ transform.T
        vertices = vertices[:, :3]
        if not finite(vertices):
            raise ValueError('Mesh coordinates must be finite')
        # Material is keyed by node AND material object; submeshes can reuse.
        visual = mesh.visual
        matobj = getattr(visual, 'material', None)
        key = id(matobj) if matobj is not None else geometry_name + '_none'
        if key not in material_cache:
            name = clean_name(getattr(matobj, 'name', '') or geometry_name)
            base = quantized_color(getattr(matobj, 'baseColorFactor', None) if getattr(matobj, 'baseColorFactor', None) is not None else
                                   getattr(matobj, 'diffuse', [255,255,255,255]))
            material = dict(name=name, baseColor=base, metallic=float(getattr(matobj,'metallicFactor',0) or 0),
                            roughness=float(getattr(matobj,'roughnessFactor',1) if getattr(matobj,'roughnessFactor',None) is not None else 1),
                            alphaMode=str(getattr(matobj,'alphaMode','OPAQUE') or 'OPAQUE'),
                            baseColorTexture=-1, normalTexture=-1, metallicRoughnessTexture=-1,
                            emissiveTexture=-1, doubleSided=bool(getattr(matobj,'doubleSided',True)))
            for attr, dest, role in [('baseColorTexture','baseColorTexture','Base color'),
                                     ('normalTexture','normalTexture','Normal'),
                                     ('metallicRoughnessTexture','metallicRoughnessTexture','Metallic / roughness'),
                                     ('emissiveTexture','emissiveTexture','Emissive')]:
                img = getattr(matobj, attr, None)
                if isinstance(img, Image.Image):
                    material[dest] = convert_texture(img, f'{name} · {role}', role, textures, max_size=texture_size)
            # OBJ SimpleMaterial uses .image, not .baseColorTexture
            img = getattr(matobj, 'image', None)
            if material['baseColorTexture'] == -1 and isinstance(img, Image.Image):
                material['baseColorTexture'] = convert_texture(img, name+' · Albedo', 'Base color', textures, max_size=texture_size)
            material_cache[key] = len(materials)
            materials.append(material)
        mi = material_cache[key]
        idx = np.asarray(mesh.faces, dtype=np.uint32).reshape(-1)
        source_vertex_total += len(v)
        source_triangle_total += len(idx)//3
        vertex_normal = np.asarray(mesh.vertex_normals, dtype=np.float64)
        normal_matrix = np.linalg.pinv(transform[:3,:3]).T
        vertex_normal = vertex_normal @ normal_matrix.T
        vertex_normal /= np.maximum(np.linalg.norm(vertex_normal,axis=1,keepdims=True),1e-8)
        if not finite(vertex_normal):
            vertex_normal[:] = [0,1,0]
        uv = getattr(visual, 'uv', None)
        if uv is None or len(uv) != len(v):
            uv = np.zeros((len(v),2), dtype=np.float32)
        else:
            uv = np.asarray(uv, dtype=np.float32)
        if visual.kind == 'vertex':
            colors = np.asarray(visual.vertex_colors, dtype=np.uint8)
            if colors.shape[1] == 3:
                colors = np.c_[colors,np.full(len(v),255,dtype=np.uint8)]
        elif visual.kind == 'face' and len(visual.face_colors) == len(mesh.faces):
            # Expand vertices for per-face paints (PLY/OBJ colour variants).
            fids = np.asarray(mesh.faces, dtype=np.uint32).reshape(-1)
            colors = np.repeat(np.asarray(visual.face_colors,dtype=np.uint8),3,axis=0)
            vertices,vertex_normal,uv = vertices[fids],vertex_normal[fids],uv[fids]
            idx = np.arange(len(vertices),dtype=np.uint32)
        else:
            colors = np.full((len(vertices),4),255,dtype=np.uint8)
        # WebGL2 multiplies true vertex colors by the texture. Canvas2D uses
        # a separate UV-sampled color approximation when textures are present.
        webgl_colors = np.array(colors,copy=True)
        texidx = materials[mi]['baseColorTexture']
        if texidx >= 0:
            img = getattr(matobj,'baseColorTexture',None)
            if img is None: img = getattr(matobj,'image',None)
            if isinstance(img,Image.Image):
                pic = np.array(img.convert('RGBA'))
                px = np.clip((uv[:,0] % 1)*(pic.shape[1]-1),0,pic.shape[1]-1).astype(int)
                py = np.clip((1-uv[:,1] % 1)*(pic.shape[0]-1),0,pic.shape[0]-1).astype(int)
                colors = (colors.astype('float32')*pic[py,px].astype('float32')/255).astype('uint8')
        # Limit using deterministic face stride (no random sampling).
        if len(idx)//3 > limit:
            step = int(np.ceil((len(idx)//3)/limit))
            faces = np.asarray(mesh.faces[::step], dtype=np.uint32)
            used, inverse = np.unique(faces.ravel(), return_inverse=True)
            idx = inverse.astype(np.uint32)
            vertices, vertex_normal, uv, colors, webgl_colors = (vertices[used],vertex_normal[used],
                uv[used],colors[used],webgl_colors[used])
        meshes.append(dict(name=clean_name(node_name), material=mi, positions=buffer64(vertices,np.float32),
                           indices=buffer64(idx,np.uint32), normals=buffer64(vertex_normal,np.float32),
                           uv=buffer64(uv,np.float32), colors=buffer64(webgl_colors,np.uint8),
                           fallbackColors=buffer64(colors,np.uint8)))
    if not meshes:
        raise ValueError('No triangle meshes were found in the asset')
    return finalize_scene(path.stem,path.suffix.lstrip('.').upper(),meshes,materials,textures,
                          description='Mesh asset',unit='model',original=dict(vertices=source_vertex_total,triangles=source_triangle_total))


def image_from_half(data,desc,*, color=False):
    w,h=desc['width'],desc['height']
    if w<1 or h<1 or w*h>MAX_INPUT//8:
        raise ValueError('Unreasonable atlas dimensions')
    dtype = '<f2' if desc['dtype']=='float16' else 'u1'
    arr=np.frombuffer(data,dtype=dtype)
    if arr.size!=w*h*4: raise ValueError('Mismatched map byte count')
    arr=arr.reshape(h,w,4).astype(np.float32)
    if desc['dtype'] == 'uint8':arr/=255
    a=np.clip(np.nan_to_num(arr[:,:,3],nan=0),0,1)
    rgb=np.clip(np.nan_to_num(arr[:,:,:3],nan=0,posinf=1,neginf=0),0,1)
    if color:
        rgb=np.clip(rgb/np.maximum(a[:,:,None],.00001),0,1)
        rgb=np.where(rgb<=.0031308,rgb*12.92,1.055*np.power(rgb,1/2.4)-.055)
    rgba=np.concatenate([rgb,a[:,:,None]],axis=2)
    rgba=(np.flipud(rgba)*255+.5).astype(np.uint8)
    return Image.fromarray(rgba,'RGBA')


def box_scene(path, *, grid=72, texture_size=MAX_TEXTURE_SIZE, limit=250_000):
    if not 8<=grid<=256:
        raise ValueError('Box surface grid must be 8–256')
    if path.stat().st_size>MAX_INPUT:
        raise ValueError('Box ZIP exceeds the 512 MiB limit')
    with zipfile.ZipFile(path) as archive:
        entries=archive.infolist()
        if len(entries)>400 or sum(i.file_size for i in entries)>MAX_INPUT:
            raise ValueError('Archive is too large')
        if any(i.filename.startswith('/') or '..' in Path(i.filename).parts or '\\' in i.filename for i in entries):
            raise ValueError('Unsafe ZIP member path')
        m=json.loads(archive.read('manifest.json'))
        if m.get('schemaVersion')!=1 or m.get('layout')!='box' or m.get('faceOrder')!=FACES:
            raise ValueError('Not a supported six-face Box ZIP')
        r=int(m['resolution'])
        if not 8<=r<=4096:
            raise ValueError('Unsupported Box resolution')
        dims=np.array([m['bakeBox']['min'],m['bakeBox']['max']],dtype=np.float64)
        if dims.shape!=(2,3) or not finite(dims) or (dims[1]<=dims[0]).any():
            raise ValueError('Invalid Box bake bounds')
        textures=[]
        ids={}
        color_atlas=None
        for name in ['colorMap','normalMap','intervalMap','rangeMap','materialMap','emissiveMap']:
            desc=m.get('textures',{}).get(name)
            if not desc:continue
            data=archive.read(desc['path'])
            if hashlib.sha256(data).hexdigest()!=desc['sha256']:
                raise ValueError('Box map checksum mismatch: '+name)
            if name=='colorMap':
                color_atlas=np.frombuffer(data,dtype='<f2').reshape(2*r,3*r,4).astype(np.float32)
            im=image_from_half(data,desc,color=name in ('colorMap','emissiveMap'))
            ids[name]=convert_texture(im,name,CHANNEL_MAPS[name],textures,max_size=texture_size)
        if color_atlas is None:
            raise ValueError('Missing Box colorMap')
        base=m.get('baseColor')
        if base and base.get('sha256')!=m['textures']['colorMap']['sha256']:
            data=archive.read(base['path'])
            if hashlib.sha256(data).hexdigest()!=base['sha256']:
                raise ValueError('Base color checksum mismatch')
            ids['baseColor']=convert_texture(image_from_half(data,base,color=True),'baseColor',CHANNEL_MAPS['baseColor'],textures,max_size=texture_size)
        for face in FACES:
            g=m['faces'][face].get('guide')
            if g and g in archive.namelist():
                im=Image.open(io.BytesIO(archive.read(g)))
                convert_texture(im,'Guide '+face,'Orthographic guide',textures,max_size=texture_size)
        mesh_data=[]
        selected = np.linspace(1,r-2,min(grid,r)).round().astype(np.int32)
        # Sampling on a grid; consistent with storage: raw atlas row 0 = shader V=0.
        for fi,face in enumerate(FACES):
            f=m['faces'][face]
            depth=np.frombuffer(archive.read(f['depth']),dtype='<f2').reshape(r,r).astype(np.float32)
            atlas=color_atlas[fi//3*r:(fi//3+1)*r,fi%3*r:(fi%3+1)*r]
            vert,uvs,col=[],[],[]
            index=np.full((len(selected),len(selected)),-1,np.int32)
            depth_index=np.full_like(index,np.nan,dtype=np.float32)
            for yi,y in enumerate(selected):
                for xi,x in enumerate(selected):
                    delta=float(depth[r-1-y,x]);rgba=atlas[y,x]
                    if not np.isfinite(delta) or delta<0 or delta>1 or rgba[3]<.04:continue
                    u=(x+.5)/r;v=(y+.5)/r
                    if face=='+X':pos=[1-delta,v,1-u]
                    elif face=='-X':pos=[delta,v,u]
                    elif face=='+Y':pos=[u,1-delta,1-v]
                    elif face=='-Y':pos=[u,delta,v]
                    elif face=='+Z':pos=[u,v,1-delta]
                    else:pos=[1-u,v,delta]
                    xyz=dims[0]+np.clip(pos,0,1)*(dims[1]-dims[0])
                    texuv=[(fi%3+u)/3,(fi//3+v)/2]
                    index[yi,xi]=len(vert)
                    depth_index[yi,xi]=delta
                    vert.append(xyz);uvs.append(texuv)
                    color=np.clip(rgba[:3]/max(.001,float(rgba[3])),0,1)
                    srgb=np.where(color<=.0031308,12.92*color,1.055*np.power(color,1/2.4)-.055)
                    col.append(np.r_[srgb,rgba[3]]*255)
            tris=[]
            for y in range(len(selected)-1):
                for x in range(len(selected)-1):
                    verts=[index[y,x],index[y,x+1],index[y+1,x],index[y+1,x+1]]
                    zz=[depth_index[y,x],depth_index[y,x+1],depth_index[y+1,x],depth_index[y+1,x+1]]
                    if min(verts)<0 or max(zz)-min(zz)>.18:continue
                    a,b,c,d=verts
                    tris.extend([(a,b,c),(b,d,c)])
            if not tris:continue
            vertices=np.asarray(vert,dtype=np.float32);faces=np.asarray(tris,dtype=np.uint32)
            normals=np.zeros_like(vertices)
            for triangle in faces:
                a,b,c=vertices[triangle]
                n=np.cross(b-a,c-a)
                for t in triangle:normals[t]+=n
            nlen=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-9)
            normals=normals/nlen
            mesh_data.append(dict(name=face+' surface',material=0,positions=buffer64(vertices,np.float32),
                                  normals=buffer64(normals,np.float32),uv=buffer64(uvs,np.float32),
                                  indices=buffer64(faces.reshape(-1),np.uint32),
                                  colors=buffer64(np.full((len(vert),4),255,dtype=np.uint8),np.uint8),
                                  fallbackColors=buffer64(np.clip(col,0,255),np.uint8)))
        if not mesh_data:raise ValueError('No visible Box surface samples')
        material=dict(name='Box atlas',baseColor=[1,1,1,1],metallic=0,roughness=1,
                      alphaMode='BLEND',baseColorTexture=ids['colorMap'],normalTexture=ids.get('normalMap',-1),
                      metallicRoughnessTexture=ids.get('materialMap',-1),emissiveTexture=ids.get('emissiveMap',-1),doubleSided=True)
        # Source proxy counts, if trimesh can inspect its native GLB.
        original={}
        try:
            import trimesh
            proxy=trimesh.load(io.BytesIO(archive.read(m['proxy'])),file_type='glb',force='scene',process=False)
            original=dict(vertices=sum(len(x.vertices) for x in proxy.geometry.values()),
                          triangles=sum(len(x.faces) for x in proxy.geometry.values()))
        except Exception: original={}
        out=finalize_scene(path.name.removesuffix('.box.zip'),'Box impostor ZIP',mesh_data,[material],textures,
                           description='Derived surface mesh',unit='bake unit',original=original,
                           note='Preview geometry is sampled from the Box depth guides. It is not the native three-impostor ray-marched result.',
                           previewType='surface', fixed_bounds=dims)
        out['assetId']=m.get('asset_id')
        out['resolution']=r
        return out


def finalize_scene(name,fmt,meshes,materials,textures,*,description,unit,original,note=None,previewType='mesh',fixed_bounds=None):
    if fixed_bounds is None:
        allp=[np.frombuffer(base64.b64decode(m['positions']),dtype='<f4').reshape(-1,3) for m in meshes]
        mins=np.min([p.min(axis=0) for p in allp],axis=0)
        maxs=np.max([p.max(axis=0) for p in allp],axis=0)
    else:mins,maxs=fixed_bounds
    if not materials:materials=[dict(name='Default',baseColor=[1,1,1,1],metallic=0,roughness=1,
                                    baseColorTexture=-1,normalTexture=-1,alphaMode='OPAQUE')]
    return dict(version=VERSION,name=clean_name(name),format=fmt,description=description,
                unit=unit,original=original or {},bounds={'min':np.asarray(mins).tolist(),'max':np.asarray(maxs).tolist()},
                previewType=previewType,note=note,meshes=meshes,materials=materials,textures=textures)


def build(input_file: Path, output_file: Path, *, title=None, grid=72, max_triangles=250_000,
          max_texture=MAX_TEXTURE_SIZE):
    input_file=input_file.expanduser().resolve()
    if not input_file.is_file():raise ValueError('Input file does not exist: '+str(input_file))
    if input_file.name.lower().endswith('.box.zip'):
        scene=box_scene(input_file,grid=grid,texture_size=max_texture,limit=max_triangles)
    elif input_file.suffix.lower() in FORMATS:
        scene=glb_scene(input_file,limit=max_triangles,texture_size=max_texture)
    else:raise ValueError('Unsupported input format: '+input_file.name)
    if title:scene['name']=clean_name(title)
    data=json.dumps(scene,ensure_ascii=False,separators=(',',':'),allow_nan=False).replace('</script>','<\\/script>')
    compressed=zlib.compress(data.encode('utf-8'),level=9)
    payload=base64.b64encode(compressed).decode('ascii')
    template=Path(__file__).resolve().parent.parent/'assets/viewer.html'
    html=template.read_text(encoding='utf-8').replace('__SCENE_PAYLOAD__',payload)
    output_file=output_file.expanduser().resolve()
    output_file.parent.mkdir(parents=True,exist_ok=True)
    output_file.write_text(html,encoding='utf-8')
    return dict(file=str(output_file),size=output_file.stat().st_size,
                vertices=sum(len(base64.b64decode(m['positions']))//12 for m in scene['meshes']),
                triangles=sum(len(base64.b64decode(m['indices']))//12 for m in scene['meshes']),
                meshes=len(scene['meshes']),materials=len(scene['materials']),textures=len(scene['textures']),
                mode=scene['previewType'])


def main(argv=None):
    parser=argparse.ArgumentParser(description='Build a standalone, offline 3D Inspector HTML')
    parser.add_argument('input',type=Path,help='.glb, .gltf, .obj, .stl, .ply, .3mf, or .box.zip')
    parser.add_argument('-o','--output',type=Path,help='Output .html (default: input-stem.inspector.html)')
    parser.add_argument('--title',help='Optional display title')
    parser.add_argument('--grid',type=int,default=72,help='Box ZIP sampling grid 8–256 (default 72)')
    parser.add_argument('--max-triangles',type=int,default=250_000,help='Per mesh face cap')
    parser.add_argument('--max-texture',type=int,default=2048,help='Texture longest side in px')
    args=parser.parse_args(argv)
    output=args.output or Path(args.input.name+'.inspector.html')
    try: result=build(args.input,output,title=args.title,grid=args.grid,max_triangles=args.max_triangles,
                      max_texture=args.max_texture)
    except Exception as exc:
        print('Error:',str(exc),file=sys.stderr)
        return 1
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
