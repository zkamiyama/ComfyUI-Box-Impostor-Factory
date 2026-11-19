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
  assert.throws(()=>createBoxDisplayMaterial({maps:[]},{colorSampling:'other'}),/colour sampling/);
  assert.throws(()=>createBoxDisplayMaterial({maps:[],layout:'octahedral'},{colorSampling:'view'}),/Box layout/);
  await assert.rejects(restoreBox(null,null,null,{colorSampling:'other'}),/colour sampling/);
  for(const command of ['render','thumbnail','preview']) {
    await assert.rejects(main([command,'--lighting','other']),/lighting/);
    await assert.rejects(main([command,'--color-sampling','other']),/colour sampling/);
  }
});


test('view colour modes get separate shader programs while native stays the original class',()=>{
  const data={maps:[],layout:'box',resolution:64};
  const native=createBoxDisplayMaterial(data,{colorSampling:'native'});
  const modes=['view','dominant','view-fade','view-soft','view-feather'].map(colorSampling=>createBoxDisplayMaterial(data,{colorSampling}));
  try {
    assert.equal(native.constructor,ImpostorNodeMaterial);
    assert.equal(new Set(modes.map(m=>m.customProgramCacheKey())).size,modes.length);
    for(const mode of modes) {
      assert.equal(mode.setupDiffuseColor,native.setupDiffuseColor);
      assert.equal(mode.setupDepth,native.setupDepth);
      assert.equal(mode.depthTest,native.depthTest);
      assert.equal(mode.depthWrite,native.depthWrite);
    }
  } finally {for(const m of [native,...modes])m.dispose();}
});
