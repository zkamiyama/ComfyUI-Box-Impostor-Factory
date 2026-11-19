// Internal RGB-preserving parallax experiments for three-impostor 0.4.1.
// The upstream ray marcher remains the sole owner of hit/discard/depth. Only
// the RGB sampled at its hit changes; none of the source textures are changed.
import { Vector3 } from 'three/webgpu';
import { ImpostorNodeMaterial } from 'three-impostor';
import {
  Fn, If, abs, clamp, diffuseColor, dot, exp, float, materialColor,
  max, min, normalize, property, select, smoothstep, texture, vec2, vec3, vec4,
} from 'three/tsl';

export const PARALLAX_MODES = Object.freeze([
  'native', 'camera-single', 'camera-blend', 'camera-depth', 'camera-hybrid',
]);

export const PARALLAX_MODE_DESCRIPTIONS = Object.freeze({
  native: 'Native six-view RGB mixing, including normal-dependent weights; lighting disabled.',
  'camera-single': 'One covered view with the greatest camera-axis weight; no RGB blending or normal weights.',
  'camera-blend': 'Camera-axis and coverage weighted RGB; no depth or normal weights except a no-coverage fallback.',
  'camera-depth': 'Camera-axis and coverage weighted RGB with relative captured-depth confidence; no normal weights.',
  'camera-hybrid': 'Camera-axis, coverage and relative depth RGB weights with bounded captured-normal facing confidence; no lighting.',
});

export const HYBRID_NORMAL_CONFIDENCE = Object.freeze({
  floor:.02,facingStart:.05,facingEnd:.6,towardRayStart:-.15,towardRayEnd:.25,
  facingPower:2,normalSampling:'native linear level0, no extra blur or renormalization',
  axisPreservation:'One active camera-axis view returns covered source RGB directly.',
});

// Shared presentation preset. Changing this does not rewrite the Box maps.
// Keep the experiment controls available separately for recorded comparisons.
export const RGB_PARALLAX_PRESET = Object.freeze({
  mode:'camera-hybrid',cameraPower:8,depthToleranceTexels:4,
});

// Match the pinned baker's atlas order and UV conventions exactly.
const SIDES = [[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]];
const viewUV = [
  b => vec2(b.z.oneMinus(), b.y), b => vec2(b.z, b.y),
  b => vec2(b.x, b.z.oneMinus()), b => vec2(b.x, b.z),
  b => vec2(b.x, b.y), b => vec2(b.x.oneMinus(), b.y),
];
const axisUV = [b => vec2(b.z,b.y), b => vec2(b.x,b.z), b => vec2(b.x,b.y)];

function checkedOptions({mode='native', cameraPower=8, depthToleranceTexels=4}={}) {
  if (!PARALLAX_MODES.includes(mode)) throw new RangeError('Unknown parallax RGB mode: '+mode);
  if (!Number.isFinite(cameraPower) || cameraPower < 1 || cameraPower > 64) {
    throw new RangeError('cameraPower must be finite and between 1 and 64');
  }
  if (!Number.isFinite(depthToleranceTexels) || depthToleranceTexels <= 0 || depthToleranceTexels > 64) {
    throw new RangeError('depthToleranceTexels must be finite, positive and at most 64');
  }
  return Object.freeze({mode,cameraPower,depthToleranceTexels,
    ...(mode==='camera-hybrid'?{normalConfidence:HYBRID_NORMAL_CONFIDENCE}:{})});
}

/**
 * Unlit experimental Box material. `data` is the unchanged native impostor
 * data (for example restored.mesh.material.impostor). Dispose the material
 * separately from the native data/textures, which it does not own.
 *
 * This deliberately calls upstream _setupBox, including its normal sampling.
 * Those normals do not affect RGB in camera-single/blend/depth modes. The
 * explicitly named hybrid mode uses bounded normal confidence for RGB view
 * selection, never for lighting. Normals remain available to diagnostic
 * outputs. This is an isolation experiment, not a performance
 * optimization or a replacement native ray marcher.
 */
export class ParallaxColorMaterial extends ImpostorNodeMaterial {
  static get type() { return 'ParallaxColorMaterial'; }

  constructor(data, options={}) {
    if (data?.layout !== 'box' || !Number.isInteger(data.resolution) || data.resolution < 1) {
      throw new TypeError('ParallaxColorMaterial requires native box impostor data');
    }
    const settings=checkedOptions(options);
    super(data);
    this.parallaxOptions=settings;
    this.steps=Math.max(this.steps,2*data.resolution);
    this.lights=false;
    this.toneMapped=false;
    this.fog=false;
    this.name='parallax-'+settings.mode;
  }

  customProgramCacheKey() {
    // The mode and constants change the graph even though they are not TSL
    // uniforms. Keep independent materials from sharing a stale shader.
    return super.customProgramCacheKey()+'|parallax:'+JSON.stringify(this.parallaxOptions);
  }

  setupLighting() {
    // Generated RGB may already contain illumination. Preserve it without
    // applying lights, material maps, emissive maps or a second tone curve.
    return diffuseColor.rgb;
  }

  _setupBox(space) {
    const {mode,cameraPower,depthToleranceTexels}=this.parallaxOptions;
    if (mode==='native') return super._setupBox(space);

    const {box,resolution,colorMap,normalMap,intervalMap}=this.impostor;
    const boxMin=vec3(...box.min.toArray());
    const boxSize=vec3(...box.getSize(new Vector3()).toArray());
    const localHit=property('vec3','ParallaxNativeHitPosition');
    // Capturing this existing callback preserves the exact native point and
    // depth computation. No alternate intersection/position is substituted.
    const nativeColor=super._setupBox({...space,setDepth:local=>{
      localHit.assign(local);
      space.setDepth(local);
    }});
    const half=.5/resolution;
    const intervalAt=(b,axis)=>{
      const uv=clamp(axisUV[axis](b),half,1-half);
      return texture(intervalMap,vec2(uv.x.add(axis),uv.y).div(vec2(3,1))).level(0);
    };
    const sampleColor=(b,i)=>{
      const uv=clamp(viewUV[i](b),half,1-half);
      return texture(colorMap,vec2(uv.x.add(i%3),uv.y.add(Math.floor(i/3))).div(vec2(3,2))).level(0);
    };
    const sampleNormal=(b,i)=>{
      const uv=clamp(viewUV[i](b),half,1-half);
      return texture(normalMap,vec2(uv.x.add(i%3),uv.y.add(Math.floor(i/3))).div(vec2(3,2))).level(0);
    };

    return Fn(()=>{
      // Using native alpha forces evaluation of the complete upstream hit
      // path before sampling localHit. Upstream discard/depth are retained.
      const native=nativeColor.toVar();
      const p=localHit.sub(boxMin).div(boxSize).toVar();
      const toCamera=normalize(space.ray().direction).negate().toVar();
      const x=intervalAt(p,0),y=intervalAt(p,1),z=intervalAt(p,2);
      const distances=[
        abs(x.x.sub(p.x)),abs(p.x.sub(x.y.oneMinus())),
        abs(y.x.sub(p.y)),abs(p.y.sub(y.y.oneMinus())),
        abs(z.x.sub(p.z)),abs(p.z.sub(z.y.oneMinus())),
      ].map(d=>d.mul(resolution).toVar());
      const samples=SIDES.map((side,i)=>{
        const c=sampleColor(p,i).toVar();
        let confidence=null;
        if(mode==='camera-hybrid') {
          const policy=HYBRID_NORMAL_CONFIDENCE;
          const n=sampleNormal(p,i).rgb.div(max(c.a,1e-3)).mul(2).sub(1).toVar();
          const facing=smoothstep(policy.facingStart,policy.facingEnd,max(dot(n,vec3(...side)),0));
          const toward=smoothstep(policy.towardRayStart,policy.towardRayEnd,dot(n,toCamera));
          confidence=facing.pow(policy.facingPower).mul(toward).mul(1-policy.floor).add(policy.floor).toVar();
        }
        return {
          coverage:c.a,
          // This value is reused inside conditional fallbacks and later in
          // unconditional blending. Materialize it here: otherwise TSL may
          // cache its first evaluation inside an If branch and reuse an
          // uninitialized GLSL variable when that branch does not execute.
          rgb:c.rgb.div(max(c.a,1e-3)).toVar(),
          camera:max(dot(toCamera,vec3(...side)),0).pow(cameraPower).toVar(),
          distance:distances[i],
          confidence,
        };
      });

      // A native hit can lack coverage in camera-facing source views (for
      // example a newly revealed cavity). All modes use the nearest covered
      // captured surface in this case. This fallback is also normal-free;
      // it cannot invent unseen RGB and may reveal a view disagreement.
      const fallback=vec3(0).toVar();
      const nearestCovered=float(1e6).toVar();
      const nearestCameraCandidate=float(1e6).toVar();
      const activeCameraViews=mode==='camera-hybrid'?float(0).toVar():null;
      const singleViewColor=mode==='camera-hybrid'?vec3(0).toVar():null;
      const singleViewCoverage=mode==='camera-hybrid'?float(0).toVar():null;
      for(const s of samples) {
        if(mode==='camera-hybrid') {
          If(s.camera.greaterThan(1e-8),()=>{
            activeCameraViews.addAssign(1);
            singleViewColor.assign(s.rgb);
            singleViewCoverage.assign(s.coverage);
          });
        }
        If(s.coverage.greaterThan(1e-3).and(s.distance.lessThan(nearestCovered)),()=>{
          nearestCovered.assign(s.distance);
          fallback.assign(s.rgb);
        });
        If(s.coverage.greaterThan(1e-3).and(s.camera.greaterThan(1e-8)),()=>{
          nearestCameraCandidate.assign(min(nearestCameraCandidate,s.distance));
        });
      }

      const color=vec3(0).toVar();
      const weight=float(0).toVar();
      if(mode==='camera-single') {
        // Coverage prevents choosing an empty sample. Equal scores use the
        // fixed atlas order. A dominant-axis switch is intentionally hard.
        for(const s of samples) {
          const score=s.camera.mul(s.coverage).toVar();
          If(score.greaterThan(weight),()=>{
            weight.assign(score);
            color.assign(s.rgb);
          });
        }
      } else {
        for(const s of samples) {
          let w=s.camera.mul(s.coverage);
          if(mode==='camera-depth'||mode==='camera-hybrid') {
            // Relative confidence avoids an all-zero result when every
            // captured surface is displaced by the native hit tolerance.
            // Allow a plateau of depthToleranceTexels past the nearest
            // camera-facing covered surface, then decay exponentially.
            // This is depth consistency, not a hard visibility proof.
            const excess=max(s.distance.sub(nearestCameraCandidate).sub(depthToleranceTexels),0);
            w=w.mul(exp(excess.div(-depthToleranceTexels)));
          }
          if(mode==='camera-hybrid') w=w.mul(s.confidence);
          color.addAssign(s.rgb.mul(w));
          weight.addAssign(w);
        }
        color.divAssign(max(weight,1e-8));
      }
      let rgb=select(weight.greaterThan(1e-8),color,fallback);
      if(mode==='camera-hybrid') {
        // Do not multiply/divide by normal confidence at the exact source
        // axes: even a mathematically cancelling factor can change an 8-bit
        // output through floating-point rounding. A direct covered sample
        // preserves the generated axial RGB, including facial detail.
        rgb=select(activeCameraViews.equal(1).and(singleViewCoverage.greaterThan(1e-8)),singleViewColor,rgb);
      }
      return vec4(rgb.mul(materialColor),native.a);
    })();
  }
}

export function createParallaxMaterial(data, options={}) {
  return new ParallaxColorMaterial(data,options);
}
