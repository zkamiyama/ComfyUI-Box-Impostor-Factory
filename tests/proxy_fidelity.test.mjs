import test from 'node:test';
import assert from 'node:assert/strict';
import {fidelity} from './compare_proxy.mjs';

function frame(colors,depth){
  const color=Buffer.from(colors.flat()).toString('base64');
  const z=Buffer.from(depth.flatMap(x=>[x,0,0,255])).toString('base64');
  return {id:'+X',width:2,height:1,color,depth:z};
}
test('proxy comparison distinguishes missed surfaces, shading and hit depth',()=>{
  const proxy=frame([[0,127,255,255],[50,40,30,255]],[64,128]);
  const box=frame([[30,127,225,255],[0,0,0,0]],[96,255]);
  const value=fidelity(proxy,box);
  assert.equal(value.silhouetteIoU,.5);
  assert.equal(value.overlapPixels,1);
  assert.equal(value.overlapColorMAE8,20);
  assert(Math.abs(value.overlapDepthP95-(32*65536/16777215))<1e-7);
});
test('a missing Box silhouette is reported as a miss with no overlap score',()=>{
  const proxy=frame([[23,34,45,255],[0,0,0,0]],[1,255]);
  const empty=frame([[0,0,0,0],[0,0,0,0]],[255,255]);
  const result=fidelity(proxy,empty);
  assert.equal(result.silhouetteIoU,0);
  assert.equal(result.overlapColorMAE8,null);
  assert.equal(result.overlapDepthP95,null);
  assert.throws(()=>fidelity(empty,empty),/Empty proxy/);
});
