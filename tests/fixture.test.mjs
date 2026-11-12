import assert from 'node:assert/strict';
import test from 'node:test';
import { makeFixture } from './make_fixture.mjs';
test('fixture has one embedded buffer, positive face winding and an asymmetric center', () => {
  const bytes = makeFixture();
  assert.equal(bytes.readUInt32LE(8), bytes.length);
  const jsonLength = bytes.readUInt32LE(12);
  const doc = JSON.parse(bytes.subarray(20, 20 + jsonLength));
  assert.equal(doc.buffers[0].uri, undefined);
  assert.equal(bytes.length - (28 + jsonLength), doc.buffers[0].byteLength);
  assert.notEqual(doc.accessors[0].min[0], -doc.accessors[0].max[0]);
  const data = bytes.subarray(28 + jsonLength);
  const vector = (view, i) => [0,1,2].map(k => data.readFloatLE(doc.bufferViews[view].byteOffset + i*12 + k*4));
  for (let i = 0; i < doc.accessors[0].count; i += 3) {
    const a = vector(0,i), b = vector(0,i+1), c = vector(0,i+2), n = vector(1,i);
    const u = b.map((v,k)=>v-a[k]), v = c.map((w,k)=>w-a[k]);
    const dot = n.reduce((s, axis, k) => s + axis * (u[(k+1)%3]*v[(k+2)%3] - u[(k+2)%3]*v[(k+1)%3]), 0);
    assert.ok(dot > 0, 'winding must agree with outward normal');
  }
});

test('open frame uses a watertight annulus with no internal caps',async()=>{
  const {makeOpenFrame}=await import('./make_open_frame.mjs');
  const glb=makeOpenFrame(), offset=glb.readUInt32LE(12);
  const doc=JSON.parse(glb.subarray(20,20+offset));
  const data=glb.subarray(28+offset), count=doc.accessors[0].count;
  const v=i=>[0,1,2].map(k=>data.readFloatLE(i*12+k*4));
  const edges=new Map();
  // The first two boxes form the protruding tab. The remaining mesh is the annulus.
  for(let i=2*36;i<count;i+=3){
    const tri=[v(i),v(i+1),v(i+2)].map(point=>point.join(','));
    for(let j=0;j<3;j++){
      const a=tri[j],b=tri[(j+1)%3],edge=a<b?a+'|'+b:b+'|'+a;
      const [uses,balance]=edges.get(edge)||[0,0];
      edges.set(edge,[uses+1,balance+(a<b?1:-1)]);
    }
  }
  assert.equal(count-72,16*6);
  for(const [edge,result] of edges)assert.deepEqual(result,[2,0],edge);
});
