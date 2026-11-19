import {
  Fn, If, abs, clamp, diffuseColor, dot, float, floor, materialColor, max, min, mix,
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

function sourceDepthEdgeWeight(uv,axis,positive,resolution,intervalMap) {
  const pixel=uv.mul(resolution).sub(.5).toVar(),cell=floor(pixel).toVar();
  const nearest=float(6).toVar();
  for(let direction=0;direction<2;direction++) {
    const samples=[];
    for(let offset=-7;offset<=8;offset++) {
      const offsetUV=direction===0?vec2(offset,0):vec2(0,offset);
      const location=cell.add(offsetUV),q=clamp(location,0,resolution-1).add(.5).div(resolution);
      const interval=texture(intervalMap,vec2(q.x.add(axis),q.y).div(vec2(3,1))).level(0).toVar();
      const first=(positive?interval.x:interval.y.oneMinus()).toVar();
      const valid=location.x.greaterThanEqual(0).and(location.x.lessThan(resolution))
        .and(location.y.greaterThanEqual(0)).and(location.y.lessThan(resolution))
        .and(interval.x.greaterThanEqual(interval.y.oneMinus()))
        .and(first.greaterThan(1e-6)).and(first.lessThan(1-1e-6));
      samples.push({first,valid});
    }
    for(let i=1;i<14;i++) {
      const [a,b,c,d]=samples.slice(i-1,i+3);
      const jump=abs(b.first.sub(c.first));
      const outer=max(abs(a.first.sub(b.first)),abs(c.first.sub(d.first)));
      // Smooth steep slopes are not seams. Require an isolated depth jump
      // with valid samples on both sides; open endpoints are not observations.
      const step=a.valid.and(b.valid).and(c.valid).and(d.valid)
        .and(jump.greaterThan(16/resolution)).and(jump.greaterThan(outer.mul(4).add(8/resolution)));
      const distance=abs(pixel.element(direction).sub(cell.element(direction).add(i-7+.5)));
      If(step,()=>nearest.assign(min(nearest,distance)));
    }
  }
  return max(1e-3,smoothstep(0,6,nearest).pow(8));
}

class ViewColorImpostorMaterial extends ImpostorNodeMaterial {
  constructor(impostor,colorSampling) {
    super(impostor);
    this.boxColorSampling=colorSampling;
  }

  customProgramCacheKey() {
    return super.customProgramCacheKey()+':box-color-'+this.boxColorSampling+':debug-'+this.boxDebug;
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
    const fade=this.boxColorSampling==='view-fade'||this.boxColorSampling==='view-soft';
    const soft=this.boxColorSampling==='view-soft';
    const feather=this.boxColorSampling==='view-feather';
    return Fn(()=>{
      // Evaluate the native resolver first, including discard, normal and depth.
      const original=nativeColor.toVar();
      const p=space.boxDisplayHit.sub(boxMin).div(boxSize).toVar();
      const intervals=AXIS_UV.map((project,axis)=>{
        const q=clamp(project(p),.5/resolution,1-.5/resolution);
        return texture(intervalMap,vec2(q.x.add(axis),q.y).div(vec2(3,1))).level(0).toVar();
      });
      const selected=original.rgb.toVar(); const probes={weights:[],visible:[],distances:[],rgb:[],coverage:[],valid:[]};
      const accumulated=vec3(0).toVar(),total=float(0).toVar(),best=float(0).toVar();
      const confidence=float(0).toVar();
      const baseTotal=float(0).toVar();
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
        // Native surfaces are expanded by 1.5 texels. The old view mode keeps
        // its narrow visibility ramp and abrupt fallback as a comparison.
        const visible=select(valid,smoothstep(2/resolution,(soft?6:3)/resolution,distance).oneMinus(),0).toVar();
        const angle=max(dot(space.boxDisplayToRay,vec3(...SIDES[i])),0).toVar();
        const rgb=c.rgb.div(max(c.a,1e-3)).mul(materialColor);
        probes.rgb.push(rgb);probes.visible.push(visible);probes.distances.push(distance.mul(resolution));
        probes.coverage.push(min(c.a,n.a));probes.valid.push(select(valid,1,0));
        if(dominant) {
          // Strict comparison gives a deterministic face-order tie break.
          If(visible.greaterThan(0).and(angle.greaterThan(best)),()=>{
            best.assign(angle);selected.assign(rgb);
          });
        } else {
          let reliability=visible;
          if(fade) {
            const edge=min(min(uv.x,uv.y),min(uv.x.oneMinus(),uv.y.oneMinus()));
            const coverage=smoothstep(1e-3,.05,min(c.a,n.a));
            const boundary=smoothstep(0,.5/resolution,edge)
              .mul(smoothstep(1e-6,.5/resolution,min(first,first.oneMinus())));
            // A fully reliable single view must keep full strength, even at
            // 45 degrees. Normalized weights alone cannot fade a lone view.
            reliability=visible.mul(coverage).mul(boundary).toVar();
            confidence.assign(max(confidence,reliability.mul(smoothstep(0,.1,angle))));
          }
          const weight=angle.pow(soft?4:8).mul(reliability).mul(c.a).toVar();
          if(feather) {
            baseTotal.addAssign(weight);
            // Feather only competing source weights, never opacity or fallback
            // confidence. The small floor preserves a sole source at any edge.
            If(weight.greaterThan(0),()=>weight.mulAssign(sourceDepthEdgeWeight(AXIS_UV[axis](p),axis,i%2===0,resolution,intervalMap)));
          }
          accumulated.addAssign(rgb.mul(weight));total.addAssign(weight);probes.weights.push(weight);
        }
      }
      if(fade) {
        const resolved=accumulated.div(max(total,1e-12));
        selected.assign(mix(original.rgb,resolved,confidence.mul(smoothstep(0,1e-8,total))));
      } else if(!dominant) If((feather?baseTotal:total).greaterThan(1e-8),()=>selected.assign(accumulated.div(feather?max(total,1e-20):total)));
      // Alpha and all geometry outputs remain those of the native resolver.
      const pack=(v,offset=0)=>vec4(vec3(...v.slice(offset,offset+3)),original.a);
      if(this.boxDebug==='hit')return vec4(p,original.a);
      if(this.boxDebug==='weights0')return pack(probes.weights.map(v=>v.div(max(total,1e-12))));
      if(this.boxDebug==='weights1')return pack(probes.weights.map(v=>v.div(max(total,1e-12))),3);
      if(this.boxDebug==='visible0')return pack(probes.visible);
      if(this.boxDebug==='visible1')return pack(probes.visible,3);
      if(this.boxDebug==='distances0')return pack(probes.distances);
      if(this.boxDebug==='distances1')return pack(probes.distances,3);
      if(this.boxDebug==='coverage')return vec4(vec3(probes.coverage[0],probes.coverage[4],total),original.a);
      if(this.boxDebug==='valid')return vec4(vec3(probes.valid[0],probes.valid[4],select(total.greaterThan(1e-8),0,1)),original.a);
      if(this.boxDebug==='native')return vec4(original.rgb,original.a);
      if(this.boxDebug==='plusX')return vec4(probes.rgb[0],original.a);
      if(this.boxDebug==='plusZ')return vec4(probes.rgb[4],original.a);
      if(this.boxDebug==='plusX-active')return vec4(probes.rgb[0],select(probes.weights[0].greaterThan(0),original.a,0));
      if(this.boxDebug==='plusZ-active')return vec4(probes.rgb[4],select(probes.weights[4].greaterThan(0),original.a,0));
      return vec4(selected,original.a);
    })();
  }
}

// Display-only: setupDiffuseColor still resolves the native hit, discard,
// blended normal and fragment depth before NodeMaterial applies outputNode.
// Tone mapping belongs to the renderer; WebGPURenderer ignores toneMapped.
export function createBoxDisplayMaterial(impostor, {lighting='lit', sampling='native',colorSampling='native',debugChannel='color'}={}) {
  if (!['lit','unlit'].includes(lighting)) throw new Error('Unknown Box display lighting: '+lighting);
  // Retain the original helper option for old display payloads. Texture
  // interpolation is a separate Factory setting; this legacy value stays native.
  if (sampling!=='native') throw new Error('Unknown Box view sampling: '+sampling);
  if (!['native','view','dominant','view-fade','view-soft','view-feather'].includes(colorSampling)) throw new Error('Unknown Box colour sampling: '+colorSampling);
  if (colorSampling!=='native' && impostor.layout!=='box') throw new Error('View colour sampling requires Box layout');
  const material=colorSampling==='native' ? new ImpostorNodeMaterial(impostor) : new ViewColorImpostorMaterial(impostor,colorSampling);
  if (lighting==='unlit') material.outputNode=vec4(diffuseColor.rgb,diffuseColor.a);
  material.boxDebug=debugChannel;return material;
}
