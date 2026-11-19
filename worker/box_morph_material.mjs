/*!
 * Source-color/material resolution and Box projection adapted from three-impostor 0.4.2:
 * https://github.com/richardmonette/three-impostor/tree/df6caf4f459f58cf424df32af07c4aed0de4905c
 *
 * MIT License
 *
 * Copyright (c) 2026 Richard Monette
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
import {
  Break, Discard, Fn, If, Loop, abs, bool, clamp, dot, exp, float, floor, int, length,
  materialColor, max, min, mix, normalize, select, smoothstep, step, texture, vec2, vec3, vec4,
} from 'three/tsl';
import { ImpostorNodeMaterial } from 'three-impostor';

const SIDES=[[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]];
const VIEW_UV=[p=>vec2(p.z.oneMinus(),p.y),p=>vec2(p.z,p.y),
  p=>vec2(p.x,p.z.oneMinus()),p=>vec2(p.x,p.z),
  p=>vec2(p.x,p.y),p=>vec2(p.x.oneMinus(),p.y)];
const AXIS_UV=[p=>vec2(p.z,p.y),p=>vec2(p.x,p.z),p=>vec2(p.x,p.y)];

// A bounded, angle-dependent projective-field morph, not a Euclidean SDF.
// It changes the single surface hit, its normal and its fragment depth. The
// original interval gaps remain hard exclusions. The native distanceMap is
// never used: only a separate cache built for the full morph band is safe.
export class BoxShapeImpostorMaterial extends ImpostorNodeMaterial {
  constructor(impostor,{shapeSampling='native',morphStrength=1,morphBandTexels=8,morphStepTexels=.75}={}) {
    super(impostor);
    if(!['native','interval-morph'].includes(shapeSampling)) throw new Error('Unknown Box shape sampling: '+shapeSampling);
    if(shapeSampling==='interval-morph'&&impostor.layout!=='box') throw new Error('Interval morph requires Box layout');
    if(!Number.isFinite(morphStrength)||morphStrength<0||morphStrength>1) throw new Error('Morph strength must be between 0 and 1');
    if(!Number.isFinite(morphBandTexels)||morphBandTexels<=0) throw new Error('Morph band must be positive');
    if(!Number.isFinite(morphStepTexels)||morphStepTexels<=0) throw new Error('Morph step must be positive');
    this.boxShapeSampling=shapeSampling;
    this.morphStrength=morphStrength;
    this.morphBandTexels=morphBandTexels;
    this.morphStepTexels=morphStepTexels;
    if(shapeSampling==='interval-morph') this.steps=Math.ceil(Math.sqrt(3)*impostor.resolution/morphStepTexels)+2;
  }

  customProgramCacheKey() {
    const skip=this.impostor.morphDistanceMap&&Number.isFinite(this.impostor.morphDistanceBandTexels)&&
      this.impostor.morphDistanceBandTexels>=this.morphBandTexels;
    return super.customProgramCacheKey()+':box-shape-'+this.boxShapeSampling+
      ':'+this.morphStrength+':'+this.morphBandTexels+':'+this.morphStepTexels+':skip-'+Number(Boolean(skip));
  }

  _setupBox(space) {
    if(this.boxShapeSampling==='native') return super._setupBox(space);
    const {box,resolution,intervalMap,colorMap,normalMap,materialMap,emissiveMap,
      morphDistanceMap,morphDistanceBandTexels}=this.impostor;
    const sizes=[box.max.x-box.min.x,box.max.y-box.min.y,box.max.z-box.min.z];
    const boxMin=vec3(...box.min.toArray()),boxSize=vec3(...sizes);
    const epsilon=1.5/resolution;
    const band=Math.min(...sizes)*this.morphBandTexels/resolution;
    const strength=this.morphStrength;
    const skipMap=morphDistanceMap&&Number.isFinite(morphDistanceBandTexels)&&morphDistanceBandTexels>=this.morphBandTexels
      ?morphDistanceMap:null;
    const cells=Math.ceil(resolution/8),columns=Math.ceil(Math.sqrt(cells)),rows=Math.ceil(cells/columns);
    const cellSize=8/resolution;
    const emptyDistance=cell=>{
      const row=floor(cell.z.add(.5).div(columns));
      const tile=vec2(cell.z.sub(row.mul(columns)),row);
      return floor(texture(skipMap,tile.mul(cells).add(cell.xy).add(.5).div(vec2(columns*cells,rows*cells)))
        .level(0).r.mul(255).add(.5));
    };
    const intervalAt=(p,axis)=>{
      const q=clamp(AXIS_UV[axis](p),.5/resolution,1-.5/resolution);
      return texture(intervalMap,vec2(q.x.add(axis),q.y).div(vec2(3,1))).level(0);
    };
    const viewAtlasUV=(p,i)=>{
      const q=clamp(VIEW_UV[i](p),.5/resolution,1-.5/resolution);
      return vec2(q.x.add(i%3),q.y.add(Math.floor(i/3))).div(vec2(3,2));
    };

    const fieldAt=Fn(([point,toRay])=>{
      const p=point.toVar(),fields=[],supports=[],weights=[];
      let nativeField=float(-1e20),gapGuard=float(-1e20);
      let minimumSpan=float(1e20),minimumSpanTexels=float(1e20);
      for(let axis=0;axis<3;axis++) {
        const interval=intervalAt(p,axis).toVar();
        const c=p.element(axis),high=interval.x,low=interval.y.oneMinus();
        const gap=min(c.sub(interval.w),interval.z.sub(c));
        const outer=max(c.sub(high),low.sub(c));
        // Scale each normalized Box-axis residual to the same physical local
        // units before combining axes. This does not make it a true SDF.
        const f=max(outer,gap).sub(epsilon).mul(sizes[axis]).toVar();
        const first=select(toRay.element(axis).greaterThanEqual(0),c.sub(high),low.sub(c)).sub(epsilon).mul(sizes[axis]).toVar();
        const endpoint=select(toRay.element(axis).greaterThanEqual(0),high,low);
        const support=high.greaterThanEqual(low).toVar();
        const observed=support.and(endpoint.greaterThan(1e-6)).and(endpoint.lessThan(1-1e-6));
        const nearFirst=smoothstep(band*.5,band,abs(first)).oneMinus();
        const outerSurface=smoothstep(band*.5,band,abs(f.sub(first))).oneMinus();
        const a=select(observed,toRay.element(axis).mul(toRay.element(axis)).mul(nearFirst).mul(outerSurface),0).toVar();
        fields.push(f);supports.push(support);weights.push(a);
        nativeField=max(nativeField,f);
        gapGuard=max(gapGuard,gap.sub(epsilon).mul(sizes[axis]));

        const hasGap=interval.z.greaterThan(interval.w);
        const lowerSpan=interval.w.sub(low),upperSpan=high.sub(interval.z);
        const segment=select(c.lessThanEqual(interval.w),lowerSpan,
          select(c.greaterThanEqual(interval.z),upperSpan,min(lowerSpan,upperSpan)));
        const span=max(select(hasGap,segment,high.sub(low)),0).toVar();
        minimumSpan=min(minimumSpan,span.mul(sizes[axis]));
        minimumSpanTexels=min(minimumSpanTexels,span.mul(resolution));
      }
      nativeField=nativeField.toVar();gapGuard=gapGuard.toVar();
      const total=weights.reduce((a,b)=>a.add(b)).toVar();
      const q=weights.map(w=>w.div(max(total,1e-12)));
      const mixed=fields.reduce((sum,f,i)=>sum.add(max(f,nativeField.sub(band)).mul(q[i])),float(0)).toVar();
      // One usable view keeps native geometry. Two overlapping views gradually
      // activate the morph; weights are continuous through cardinal directions.
      const overlap=clamp(float(1).sub(q.reduce((sum,w)=>sum.add(w.mul(w)),float(0))).mul(2),0,1);
      const amount=overlap.mul(smoothstep(.05,.25,total)).mul(strength);
      const thinProtection=smoothstep(4,8,minimumSpanTexels);
      const limit=min(band,minimumSpan.mul(.25)).mul(thinProtection);
      const candidate=max(max(mix(nativeField,mixed,amount),nativeField.sub(limit)),gapGuard);
      const supported=supports[0].and(supports[1]).and(supports[2]);
      // Empty/inside-out interval texels include the baker's thin-edge support.
      // Retain their native value instead of fabricating an observed surface.
      return select(supported,candidate,nativeField);
    }).setLayout({name:'boxMorphField',type:'float',inputs:[
      {name:'point',type:'vec3'},{name:'toRay',type:'vec3'},
    ]});

    return Fn(()=>{
      const {origin,direction}=space.ray();
      const toRay=normalize(direction).negate().toVar();
      const o=origin.sub(boxMin).div(boxSize).toVar(),d=direction.div(boxSize).toVar();
      const inverseD=vec3(1).div(d);
      const t0=o.negate().mul(inverseD).toVar(),t1=o.oneMinus().mul(inverseD).toVar();
      const tNear=max(max(min(t0.x,t1.x),min(t0.y,t1.y)),max(min(t0.z,t1.z),0)).toVar();
      const tFar=min(min(max(t0.x,t1.x),max(t0.y,t1.y)),max(t0.z,t1.z)).toVar();
      const hit=bool(false).toVar(),t=tNear.toVar(),previous=tNear.toVar(),sampleIndex=int(0).toVar();
      const fineStep=float(this.morphStepTexels/resolution).div(length(d)).toVar();
      // Fixed sampling brackets the first occupied sample, followed by bisection.
      // It does not claim sphere-tracing safety or guarantee tangent/thin hits.
      If(tFar.greaterThan(tNear),()=>{
        Loop({start:int(0),end:int(this.stepsNode),type:'int',condition:'<'},()=>{
          If(t.greaterThan(tFar),()=>Break());
          const p=o.add(d.mul(t)).toVar();
          const sample=()=>{
            If(fieldAt(p,toRay).lessThanEqual(0),()=>{hit.assign(true);Break();});
            previous.assign(t);sampleIndex.addAssign(1);
            t.assign(tNear.add(float(sampleIndex).mul(fineStep)));
          };
          if(skipMap){
            const cell=clamp(floor(p.div(cellSize)),0,cells-1).toVar();
            const empty=emptyDistance(cell).sub(1).toVar();
            If(empty.lessThan(0),sample).Else(()=>{
              const exits=cell.add(step(0,d).mul(empty.mul(2).add(1))).sub(empty).mul(cellSize).sub(o).mul(inverseD);
              const exit=max(min(min(exits.x,exits.y),exits.z),t);
              // Stay on the same tNear-anchored sampling lattice as fine-only.
              // Landing at/before the empty-block exit avoids skipping a sample
              // that could hit a thin surface in the next occupied cell.
              const next=max(sampleIndex.add(1),int(floor(exit.sub(tNear).div(fineStep)))).toVar();
              previous.assign(tNear.add(float(next.sub(1)).mul(fineStep)));
              sampleIndex.assign(next);t.assign(tNear.add(float(next).mul(fineStep)));
            });
          } else sample();
        });
      });
      If(hit.not(),()=>Discard());
      If(t.greaterThan(tNear),()=>{
        const low=previous.toVar(),high=t.toVar();
        Loop(8,()=>{
          const middle=low.add(high).mul(.5).toVar();
          If(fieldAt(o.add(d.mul(middle)),toRay).lessThanEqual(0),()=>high.assign(middle))
            .Else(()=>low.assign(middle));
        });
        t.assign(high);
      });
      const p=o.add(d.mul(t)).toVar();

      // Upstream 0.4.2 source-color/material resolution at the new single hit.
      // The separate display wrapper can apply the same RGB mode as a baseline.
      const x=intervalAt(p,0),y=intervalAt(p,1),z=intervalAt(p,2);
      const distances=[abs(x.x.sub(p.x)),abs(p.x.sub(x.y.oneMinus())),
        abs(y.x.sub(p.y)),abs(p.y.sub(y.y.oneMinus())),
        abs(z.x.sub(p.z)),abs(p.z.sub(z.y.oneMinus()))].map(value=>value.toVar());
      const closest=distances.reduce((a,b)=>min(a,b)).toVar();
      const color=vec3(0).toVar(),localNormal=vec3(0).toVar(),weight=float(1e-5).toVar();
      const surface=vec3(0).toVar(),glow=vec3(0).toVar();
      const anyColor=vec3(0).toVar(),anyWeight=float(1e-5).toVar();
      const anySurface=vec3(0).toVar(),anyGlow=vec3(0).toVar();
      for(let i=0;i<6;i++) {
        If(distances[i].sub(closest).lessThan(48/resolution),()=>{
          const uv=viewAtlasUV(p,i),c=texture(colorMap,uv).level(0),n=texture(normalMap,uv).level(0);
          const viewNormal=n.rgb.div(max(c.a,1e-3)).mul(2).sub(1).toVar();
          const facing=max(dot(viewNormal,vec3(...SIDES[i])),0);
          const towardRay=smoothstep(-.2,.1,dot(viewNormal,toRay));
          const anyW=exp(distances[i].sub(closest).mul(-.5*resolution)).mul(c.a).mul(facing.mul(facing).add(1e-3)).toVar();
          const w=anyW.mul(towardRay).toVar(),viewColor=c.rgb.div(max(c.a,1e-3)).toVar();
          color.addAssign(viewColor.mul(w));localNormal.addAssign(viewNormal.mul(w));weight.addAssign(w);
          anyColor.addAssign(viewColor.mul(anyW));anyWeight.addAssign(anyW);
          if(materialMap) {
            const sample=texture(materialMap,uv).level(0).rgb.div(max(c.a,1e-3)).toVar();
            surface.addAssign(sample.mul(w));anySurface.addAssign(sample.mul(anyW));
          }
          if(emissiveMap) {
            const sample=texture(emissiveMap,uv).level(0).rgb.div(max(c.a,1e-3)).toVar();
            glow.addAssign(sample.mul(w));anyGlow.addAssign(sample.mul(anyW));
          }
        });
      }
      const seen=weight.greaterThan(1e-3).toVar();
      color.assign(select(seen,color.div(weight),anyColor.div(anyWeight)));
      if(space.surface) space.surface.assign(select(seen,surface.div(weight),anySurface.div(anyWeight)));
      if(space.emissive) space.emissive.assign(select(seen,glow.div(weight),anyGlow.div(anyWeight)));
      const normal=select(seen.and(length(localNormal).greaterThan(1e-4)),normalize(localNormal),toRay).toVar();
      const h=.5/resolution;
      const gradient=vec3(
        fieldAt(p.add(vec3(h,0,0)),toRay).sub(fieldAt(p.sub(vec3(h,0,0)),toRay)),
        fieldAt(p.add(vec3(0,h,0)),toRay).sub(fieldAt(p.sub(vec3(0,h,0)),toRay)),
        fieldAt(p.add(vec3(0,0,h)),toRay).sub(fieldAt(p.sub(vec3(0,0,h)),toRay))
      ).div(boxSize).toVar();
      If(length(gradient).greaterThan(1e-12),()=>normal.assign(normalize(gradient)));
      space.setNormal(normal);
      space.setDepth(boxMin.add(p.mul(boxSize)));
      return vec4(color.mul(materialColor),1);
    })();
  }
}
