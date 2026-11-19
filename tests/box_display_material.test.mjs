import test from 'node:test';
import assert from 'node:assert/strict';
import {BackSide} from 'three/webgpu';
import {ImpostorNodeMaterial} from 'three-impostor';
import {createBoxDisplayMaterial} from '../worker/box_display_material.mjs';
import {createBoxMaterial,restoreBox} from '../worker/runtime.mjs';
import {main} from '../worker/cli.mjs';

test('default display preserves the native material and Factory ray budget',()=>{
  const data={maps:['metalness'],resolution:1024};
  const native=new ImpostorNodeMaterial(data), plain=createBoxDisplayMaterial(data);
  const factory=createBoxMaterial(data), unlit=createBoxMaterial(data,{lighting:'unlit'});
  try {
    assert.equal(plain.constructor,native.constructor);
    for(const key of ['outputNode','fragmentNode','side','roughness','metalness','depthTest','depthWrite','transparent','steps'])
      assert.equal(plain[key],native[key],key);
    assert.equal(factory.steps,2048);
    assert.equal(unlit.steps,factory.steps);
    assert.equal(unlit.impostor,data);
    assert.equal(unlit.side,BackSide);
    assert.equal(unlit.fragmentNode,null);
    assert.ok(unlit.outputNode);
    // The native resolver, including hit depth/discard/normal, is not replaced.
    for(const key of ['setupDiffuseColor','setupDepth','_setupSpace','_setupBox'])
      assert.equal(unlit[key],native[key],key);
  } finally {for(const m of [native,plain,factory,unlit])m.dispose();}
});

test('unknown display options fail before asset I/O or browser launch',async()=>{
  assert.throws(()=>createBoxDisplayMaterial({maps:[]},{lighting:'other'}),/lighting/);
  assert.throws(()=>createBoxDisplayMaterial({maps:[]},{sampling:'dominant'}),/sampling/);
  await assert.rejects(restoreBox(null,null,null,{lighting:'other'}),/lighting/);
  for(const command of ['render','thumbnail','preview'])
    await assert.rejects(main([command,'--lighting','other']),/lighting/);
});
