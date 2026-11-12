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
