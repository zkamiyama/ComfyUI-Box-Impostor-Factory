import {
  Fn, If, abs, clamp, diffuseColor, dot, float, materialColor, max,
  normalize, property, select, smoothstep, texture, vec2, vec3, vec4,
} from 'three/tsl';
import { ImpostorNodeMaterial } from 'three-impostor';

// Pinned three-impostor 0.4.1 Box projection convention. These functions only
// locate colour samples; the upstream interval/range raymarch is unchanged.
const SIDES=[[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]];
const VIEW_UV=[p=>vec2(p.z.oneMinus(),p.y),p=>vec2(p.z,p.y),
  p=>vec2(p.x,p.z.oneMinus()),p=>vec2(p.x,p.z),
  p=>vec2(p.x,p.y),p=>vec2(p.x.oneMinus(),p.y)];
const AXIS_UV=[p=>vec2(p.z,p.y),p=>vec2(p.x,p.z),p=>vec2(p.x,p.y)];

class ViewColorImpostorMaterial extends ImpostorNodeMaterial {
  constructor(impostor,colorSampling) {
    super(impostor);
    this.boxColorSampling=colorSampling;
  }

  customProgramCacheKey() {
    return super.customProgramCacheKey()+':box-color-'+this.boxColorSampling;
  }

  _setupSpace(builder,normal,hitDepth,hitView) {
    const space=super._setupSpace(builder,normal,hitDepth,hitView);
    const ray=space.ray,setDepth=space.setDepth;
    space.boxDisplayHit=property('vec3');
    space.boxDisplayToRay=property('vec3');
    space.ray=()=>{
      const result=ray();
      // Physical local-space ray, not the anisotropically scaled unit-box ray.
      space.boxDisplayToRay.assign(normalize(result.direction).negate());
      return result;
    };
    space.setDepth=local=>{
      space.boxDisplayHit.assign(local);
      setDepth(local);
    };
    return space;
  }

  _setupBox(space) {
    const nativeColor=super._setupBox(space);
    const {box,resolution,colorMap,normalMap,intervalMap}=this.impostor;
    const boxMin=vec3(...box.min.toArray());
    const boxSize=vec3(box.max.x-box.min.x,box.max.y-box.min.y,box.max.z-box.min.z);
    const dominant=this.boxColorSampling==='dominant';
    return Fn(()=>{
      // Evaluate the native resolver first, including discard, normal and depth.
      const original=nativeColor.toVar();
      const p=space.boxDisplayHit.sub(boxMin).div(boxSize).toVar();
      const intervals=AXIS_UV.map((project,axis)=>{
        const q=clamp(project(p),.5/resolution,1-.5/resolution);
        return texture(intervalMap,vec2(q.x.add(axis),q.y).div(vec2(3,1))).level(0).toVar();
      });
      const selected=original.rgb.toVar();
      const accumulated=vec3(0).toVar(),total=float(0).toVar(),best=float(0).toVar();
      for(let i=0;i<6;i++) {
        const axis=Math.floor(i/2),uv=VIEW_UV[i](p).toVar();
        const q=clamp(uv,.5/resolution,1-.5/resolution);
        const atlas=vec2(q.x.add(i%3),q.y.add(Math.floor(i/3))).div(vec2(3,2));
        const c=texture(colorMap,atlas).level(0).toVar();
        const n=texture(normalMap,atlas).level(0).toVar();
        const interval=intervals[axis],high=interval.x,low=interval.y.oneMinus();
        const first=i%2===0?high:low;
        // Endpoints on the box boundary may be open-mesh sentinels. A normal
        // alpha is coverage, never depth. Unknown visibility keeps native RGB.
        const valid=uv.x.greaterThanEqual(0).and(uv.x.lessThanEqual(1))
          .and(uv.y.greaterThanEqual(0)).and(uv.y.lessThanEqual(1))
          .and(c.a.greaterThan(1e-3)).and(n.a.greaterThan(1e-3))
          .and(high.greaterThanEqual(low)).and(first.greaterThan(1e-6)).and(first.lessThan(1-1e-6));
        const distance=abs(p.element(axis).sub(first));
        // Native surfaces are expanded by 1.5 texels. Allow that hit offset,
        // then reduce candidate weights. With no candidate, keep native RGB;
        // normalized colour weights do not blend gradually into that fallback.
        const visible=select(valid,smoothstep(2/resolution,3/resolution,distance).oneMinus(),0).toVar();
        const angle=max(dot(space.boxDisplayToRay,vec3(...SIDES[i])),0).toVar();
        const rgb=c.rgb.div(max(c.a,1e-3)).mul(materialColor);
        if(dominant) {
          // Strict comparison gives a deterministic face-order tie break.
          If(visible.greaterThan(0).and(angle.greaterThan(best)),()=>{
            best.assign(angle);selected.assign(rgb);
          });
        } else {
          const weight=angle.pow(8).mul(visible).mul(c.a).toVar();
          accumulated.addAssign(rgb.mul(weight));total.addAssign(weight);
        }
      }
      if(!dominant) If(total.greaterThan(1e-8),()=>selected.assign(accumulated.div(total)));
      // Alpha and all geometry outputs remain those of the native resolver.
      return vec4(selected,original.a);
    })();
  }
}

// Display-only: setupDiffuseColor still resolves the native hit, discard,
// blended normal and fragment depth before NodeMaterial applies outputNode.
// Tone mapping belongs to the renderer; WebGPURenderer ignores toneMapped.
export function createBoxDisplayMaterial(impostor, {lighting='lit', sampling='native',colorSampling='native'}={}) {
  if (!['lit','unlit'].includes(lighting)) throw new Error('Unknown Box display lighting: '+lighting);
  // Retain the original helper option for old display payloads. Texture
  // interpolation is a separate Factory setting; this legacy value stays native.
  if (sampling!=='native') throw new Error('Unknown Box view sampling: '+sampling);
  if (!['native','view','dominant'].includes(colorSampling)) throw new Error('Unknown Box colour sampling: '+colorSampling);
  if (colorSampling!=='native' && impostor.layout!=='box') throw new Error('View colour sampling requires Box layout');
  const material=colorSampling==='native' ? new ImpostorNodeMaterial(impostor) : new ViewColorImpostorMaterial(impostor,colorSampling);
  if (lighting==='unlit') material.outputNode=vec4(diffuseColor.rgb,diffuseColor.a);
  return material;
}
