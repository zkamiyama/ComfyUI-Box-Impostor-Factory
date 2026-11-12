import test from 'node:test';
import assert from 'node:assert/strict';
import { compareFrames as compareAll } from '../worker/compare_frames.mjs';

const compareFrames = (before,after) => compareAll(before,after,{requireAllViews:false});

function frame({alpha = 255, color = 200, depth = 0x123456} = {}) {
  const rgba = Uint8Array.of(color,0,0,alpha, 0,0,0,0);
  const packed = Uint8Array.of(depth >>> 16,(depth >>> 8) & 255,depth & 255,255, 255,255,255,255);
  return {id:'+X', width:2, height:1, color:Buffer.from(rgba).toString('base64'),
    depth:Buffer.from(packed).toString('base64')};
}

test('color and real 24-bit depth detect changes independently', () => {
  const live = [frame()];
  assert.equal(compareFrames(live,[frame()]).pass,true);
  assert.equal(compareFrames(live,[frame({color:199,depth:0x123458})]).pass,true);
  const delta = compareFrames(live,[frame({color:194,depth:0x12345c})]);
  assert.equal(delta.pass,false);
  assert.equal(delta.worstColorLSB,6);
  assert.equal(delta.worstDepthLSB,6);
});

test('identical empty output and missing depth cannot pass', () => {
  for (const f of [frame({alpha:0,depth:0xffffff}),frame({depth:0xffffff}),frame({alpha:0})]) {
    assert.equal(compareFrames([f],[f]).pass,false);
  }
});

test('snapshot buffers and view identifiers must agree', () => {
  const a = frame();
  assert.throws(() => compareFrames([a],[{...a,depth:''}]),/byte length/);
  assert.throws(() => compareFrames([a],[{...a,id:'-X'}]),/views/);
  assert.throws(() => compareFrames([a,a],[a,a]),/views/);
});

test('integration policy requires all six axis and twelve oblique views', () => {
  assert.throws(() => compareAll([frame()],[frame()]),/twelve oblique/);
});
