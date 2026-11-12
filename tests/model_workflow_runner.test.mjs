import assert from 'node:assert/strict';
import { readFile, mkdtemp, readdir, rm } from 'node:fs/promises';
import { createServer } from 'node:http';
import { createHash } from 'node:crypto';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { checkCompiled, checkedMaterialTextures, configureMaterialMaps, configureFlatBackground, configureCompactProxy, configureEditGraph, packCompletedBox, preflight, workflows, run, inferenceReport, selectFace, verifyOutput, saveSplatOutput, verifySplatOutput } from '../scripts/run_model_workflow.mjs';

const dirs = ['+X', '-X', '+Y', '-Y', '+Z', '-Z'];
const digest = 'a'.repeat(64);
const load = name => readFile(new URL('../workflows/' + workflows[name] + '.json', import.meta.url), 'utf8').then(JSON.parse);

test('flat input background keeps the appearance mattes and drops unused model dependencies', async () => {
  for (const model of ['pixal3d', 'triposplat', 'qwen', 'h3']) {
    const graph = await load(model);
    configureFlatBackground(graph);
    const prepare = graph.nodes.find(n => n.type === 'BIPrepareReference');
    assert.equal(prepare.widgets_values[0], 'flat_background');
    assert.equal(prepare.inputs.find(p => p.name === 'mask').link, null);
    const inputImage = graph.nodes.find(n => n.type === 'LoadImage');
    const inputEdges = graph.links.filter(edge => edge[1] === inputImage.id);
    assert.equal(inputEdges.length, 1);
    assert.equal(inputEdges[0][3], prepare.id);
    const edited = model === 'qwen' || model === 'h3';
    assert.equal(graph.nodes.filter(n => n.type === 'RemoveBackground').length, edited ? 6 : 0);
    assert.equal(graph.nodes.filter(n => n.type === 'LoadBackgroundRemovalModel').length, edited ? 1 : 0);
    const ids = new Set(graph.nodes.map(n => n.id));
    assert(graph.links.every(edge => ids.has(edge[1]) && ids.has(edge[3])));
  }
});

test('portable export rejects invalid names and splat, and checks the additional registered nodes', async () => {
  await assert.rejects(run({model:'triposplat', pack:'splat'}), /requires a completed/);
  for (const pack of ['', '../box', 'two names', 'x'.repeat(101)])
    await assert.rejects(run({model:'qwen', pack}), /export name/);
  const graph = await load('pixal3d');
  const normal = preflight(graph, {}, 'source.png');
  const withPack = preflight(graph, {}, 'source.png', 'portable');
  assert.equal(withPack.pack, 'portable');
  assert(!Object.hasOwn(normal, 'pack'));
  assert.deepEqual(withPack.missingNodes.filter(n => !normal.missingNodes.includes(n)), ['BILoadBox', 'BIPackBox']);
});

test('portable export reloads the actual native pack output and verifies the restored asset before publishing', async t => {
  const dir = await mkdtemp(join(tmpdir(), 'bi-pack-'));
  // A ZIP with one manifest.json entry.
  const zip = Buffer.from(
    'UEsDBBQAAAAAAAAAIQD3xj9KUAAAAFAAAAANAAAAbWFuaWZlc3QuanNvbnsiYXNzZXRfaWQiOiAiYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYSJ9UEsBAhQDFAAAAAAAAAAhAPfGP0pQAAAAUAAAAA0AAAAAAAAAAAAAAIABAAAAAG1hbmlmZXN0Lmpzb25QSwUGAAAAAAEAAQA7AAAAewAAAAAA', 'base64');
  const calls = [];
  let restored = digest, download = zip;
  const server = createServer(async (req, res) => {
    const url = new URL(req.url, 'http://localhost');
    if (url.pathname === '/prompt') {
      let body = '';
      for await (const chunk of req) body += chunk;
      calls.push(JSON.parse(body).prompt);
      res.end(JSON.stringify({prompt_id:'pack-id'}));
    } else if (url.pathname === '/history/pack-id') {
      res.end(JSON.stringify({'pack-id':{status:{status_str:'success'}, outputs:{4:{box_impostor:[
        '/box-impostor/web/viewer.html?manifest=/box-impostor/assets/' + restored + '/manifest.json']}}}}));
    } else if (url.pathname.endsWith('/manifest.json')) {
      res.end(JSON.stringify({asset_id:restored}));
    } else if (url.pathname === '/view' && url.searchParams.get('type') === 'output' &&
      url.searchParams.get('subfolder') === 'box_impostor_exports' && url.searchParams.get('filename') === 'portable.box.zip') {
      res.end(download);
    } else { res.statusCode = 404; res.end(); }
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  t.after(async () => { await new Promise(resolve => server.close(resolve)); await rm(dir, {recursive:true, force:true}); });
  const args = {origin:'http://127.0.0.1:' + server.address().port, assetId:digest, name:'portable', outputDir:dir};
  const packed = await packCompletedBox(args);
  assert.deepEqual(calls[0], {
    1:{class_type:'BILoadBox',inputs:{source:digest}}, 2:{class_type:'BIPackBox',inputs:{asset:['1',0],name:'portable'}},
    3:{class_type:'BILoadBox',inputs:{source:['2',0]}}, 4:{class_type:'BIPreviewBox',inputs:{asset:['3',0]}},
  });
  assert.equal(packed.restoredAssetId, digest);
  assert.equal(packed.serverPath, 'box_impostor_exports/portable.box.zip');
  assert.equal(packed.sha256, createHash('sha256').update(zip).digest('hex'));
  assert.equal(packed.bytes, zip.length);
  assert.deepEqual(await readFile(packed.localPath), zip);
  assert.deepEqual(await packCompletedBox(args), packed);
  const mismatchedDir = join(dir, 'mismatched');
  restored = 'b'.repeat(64);
  await assert.rejects(packCompletedBox({...args, outputDir:mismatchedDir}), /restored Box/);
  assert(!(await readdir(mismatchedDir)).includes('portable.box.zip'));
  restored = digest;
  download = zip.subarray(0, -1);
  await assert.rejects(packCompletedBox({...args, outputDir:mismatchedDir}), /Incomplete Box ZIP/);
  assert(!(await readdir(mismatchedDir)).some(name => name.endsWith('.zip') || name.endsWith('.part')));
});

test('a nonemissive mesh retains material channels when emission was also requested', () => {
  const maps = ['roughness', 'metalness', 'ao'];
  const manifest = {maps, textures:{materialMap:{}}, provenance:{requestedMaterialMaps:[...maps, 'emissive']}};
  assert.deepEqual(checkedMaterialTextures('material+emissive', manifest), ['materialMap']);
  assert.throws(() => checkedMaterialTextures('material+emissive', {...manifest, maps:[]}), /channels/);
  assert.throws(() => checkedMaterialTextures('material+emissive', {...manifest, textures:{}}), /selection/);
});

test('guides and final bake request the same selected PBR maps', async () => {
  for (const model of ['pixal3d', 'qwen', 'h3']) {
    const graph = await load(model);
    configureMaterialMaps(graph, 'material+emissive');
    const bakers = graph.nodes.filter(n => ['BIMakeSixGuides', 'BIBakeBox'].includes(n.type));
    assert.equal(bakers.length, model === 'pixal3d' ? 1 : 2);
    assert(bakers.every(n => n.widgets_values[2] === 'material+emissive'));
    assert.throws(() => configureMaterialMaps(graph, 'unknown'), /Unknown material/);
  }
});

test('compact Pixal3D changes only geometry settings across the three workflows', async () => {
  for (const model of ['pixal3d', 'qwen', 'h3']) {
    const graph = await load(model);
    const before = structuredClone(graph);
    configureCompactProxy(graph);
    const changed = [];
    for (const [i, subgraph] of graph.definitions.subgraphs.entries()) {
      const original = before.definitions.subgraphs[i];
      assert.deepEqual(subgraph.links, original.links);
      for (const [j, node] of subgraph.nodes.entries()) {
        if (JSON.stringify(node) !== JSON.stringify(original.nodes[j])) changed.push(node.type);
      }
    }
    assert.deepEqual(changed.sort(), ['BakeAmbientOcclusion', 'BakeNormalMapFromMesh', 'DecimateMesh',
      'PrimitiveInt', 'RemeshMesh', 'Trellis2UpsampleStage']);
    assert.deepEqual(graph.nodes, before.nodes);
  }
  for (const model of ['triposplat', 'qwen']) {
    await assert.rejects(run({model, mesh:'mesh.glb', 'compact-proxy':true}), /requires generated Pixal3D/);
  }
});

test('installed dropdowns must contain each requested file, including the Qwen-only LoRA', async () => {
  for (const [name, count] of [['pixal3d', 6], ['h3', 9], ['qwen', 10]]) {
    const graph = await load(name);
    const missing = preflight(graph, {}, 'object.png');
    assert(!missing.ready);
    assert.equal(missing.models.length, count);
    assert(missing.missingNodes.includes('Pixal3DConditioning'));
    assert(!missing.missingNodes.includes(graph.definitions.subgraphs[0].id));
    const registry = Object.fromEntries(missing.missingNodes.map(type => [type, { input: { required: {} } }]));
    registry.LoadImage.input.required.image = [['object.png'], { image_upload: true }];
    for (const model of missing.models) {
      const pins = registry[model.loader].input.required;
      (pins.model ??= ['COMBO', { options: [] }])[1].options.push(model.name);
    }
    assert(preflight(graph, registry, 'object.png').ready);
    assert(!preflight(graph, registry, 'missing.png').ready);
    if (name === 'h3') {
      configureEditGraph(graph, 'h3', {h3Denoiser:'w4a8'});
      const unavailable = preflight(graph, registry, 'object.png');
      assert(!unavailable.ready);
      assert.deepEqual(unavailable.models.filter(m => !m.available).map(m => m.name), [h3Weights.w4a8[0]]);
      registry.UNETLoader.input.required.model[1].options.push(h3Weights.w4a8[0]);
      assert(preflight(graph, registry, 'object.png').ready);
    }
    if (name === 'qwen') {
      registry.LoraLoaderModelOnly.input.required.model[1].options = ['other.safetensors'];
      const report = preflight(graph, registry, 'object.png');
      assert.equal(report.ready, false);
      assert.deepEqual(report.models.filter(m => !m.available).map(m => m.name), ['QI2.1_AnyAngle.safetensors']);
    }
  }
});

const h3Weights = {
  int8: ['minimax_h3_image_pruned_int8_convrot.safetensors', '83eb140f1ff95c068f8a6886e6d3f3d0d4ab2e07bb2fe6d2ef992a51146ee622'],
  w4a8: ['minimax_h3_image_pruned_w4a8.safetensors', '517cc19b85348386f4df7c806dce80d839d0899e7cdaa3e1dbb09de964381dbb'],
};

test('H3 selection rewrites the shared loader and every recipe and keeps the default workflow intact', async () => {
  const graph = await load('h3'), original = structuredClone(graph);
  configureEditGraph(graph, 'h3');
  assert.deepEqual(graph, original);
  const loader = graph.nodes.find(n => n.type === 'UNETLoader');
  configureEditGraph(graph, 'h3', {h3Denoiser:'int8'});
  assert.equal(loader.properties.models[0].url, 'https://huggingface.co/Comfy-Org/MiniMax-H3-Image/resolve/0bb149f51e3156ed4b90dd45d5620346bfe1df8f/diffusion_models/'+h3Weights.int8[0]);
  assert.deepEqual(graph.nodes.filter(n => n.type !== 'UNETLoader'), original.nodes.filter(n => n.type !== 'UNETLoader'));
  loader.widgets_values_named = {unet_name:h3Weights.int8[0], weight_dtype:'default'};
  configureEditGraph(graph, 'h3', {h3Denoiser:'w4a8'});
  const name = h3Weights.w4a8[0];
  assert.equal(loader.widgets_values[0], name);
  assert.equal(loader.widgets_values_named.unet_name, name);
  assert.deepEqual(loader.properties.models, [{name, directory:'diffusion_models',
    url:'https://huggingface.co/Comfy-Org/MiniMax-H3-Image/resolve/0bb149f51e3156ed4b90dd45d5620346bfe1df8f/diffusion_models/'+name}]);
  const recipes = graph.nodes.filter(n => n.type === 'BICachedGeneratedFace');
  assert.deepEqual(recipes.map(n => n.widgets_values[0]), dirs);
  for (const node of recipes) {
    const before = original.nodes.find(n => n.id === node.id);
    const expected = {...JSON.parse(before.widgets_values[2]), model:name};
    assert.deepEqual(JSON.parse(node.widgets_values[2]), expected);
    assert.equal(node.widgets_values_named.recipe, node.widgets_values[2]);
  }
  assert.deepEqual(graph.links, original.links);
  assert.deepEqual(graph.definitions, original.definitions);
  assert.throws(() => configureEditGraph(graph, 'h3', {h3Denoiser:'other'}), /int8 or w4a8/);
  for (const model of ['pixal3d', 'qwen', 'triposplat'])
    await assert.rejects(run({model,'h3-denoiser':'int8'}), /--h3-denoiser needs h3/);
  await assert.rejects(run({model:'h3','h3-denoiser':'__proto__'}), /int8 or w4a8/);
});

test('H3 compiled and saved evidence requires the selected denoiser for all six face branches', () => {
  const node = (class_type,inputs={}) => ({class_type,inputs});
  for (const [selection,[name,hash]] of Object.entries(h3Weights)) {
    const report = {input:'ref.png', mesh:'proxy.glb', models:[{loader:'UNETLoader',name}], h3Denoiser:selection};
    const prompt = {load:node('LoadImage',{image:report.input}), ref:node('BIPrepareReference'),
      proxy:node('BICanonicalizeProxy',{filename:report.mesh}), save:node('BISaveBox'), preview:node('BIPreviewBox'),
      order:node('PrimitiveBoolean',{value:false}), orderSwitch:node('ComfySwitchNode',{switch:['order',0]}),
      loader:node('UNETLoader',{unet_name:name}), other:node('UNETLoader',{unet_name:'other.safetensors'})};
    const manifest = {layout:'box',asset_id:digest,mode:'geometry_baseline',reference:{sha256:digest},
      provenance:{originalGlbSha256:digest},faces:{}}, qa = {faces:{}};
    for (const [i,face] of dirs.entries()) {
      prompt['canvas'+i] = node('BIModelCanvas',{reference:['ref',0]});
      prompt['cond'+i] = node('MiniMaxH3ImageConditioning',{'images.image_1':['orderSwitch',0]});
      prompt['seed'+i] = node('PrimitiveInt',{value:101+i});
      prompt['sample'+i] = node('KSampler',{model:['loader',0],seed:['seed'+i,0]});
      const image = ['sample'+i,0];
      prompt['mask'+i] = node('RemoveBackground',{image});
      prompt['face'+i] = node('BICachedGeneratedFace',{face_id:face,mask_mode:'foreground',
        mask:['mask'+i,0],image,seed:['seed'+i,0],recipe:JSON.stringify({model:name})});
      manifest.faces[face] = {guideSha256:digest,edit:{source:'h3_enhanced',status:'fallback',
        reference:'reference/'+digest+'/source.png',provenance:{faceId:face,referenceSha256:digest,
          guideSha256:digest,recipe:{model:name},weights:{['diffusion_models/'+name]:hash}}}};
      qa.faces[face] = {status:'fallback'};
    }
    checkCompiled(prompt, 'h3', report);
    prompt.sample5.inputs.model = ['other',0];
    assert.throws(() => checkCompiled(prompt,'h3',report), /sampler denoiser on -Z/);
    prompt.sample5.inputs.model = ['loader',0];
    prompt.face4.inputs.recipe = JSON.stringify({model:'other.safetensors'});
    assert.throws(() => checkCompiled(prompt,'h3',report), /compiled H3 recipe on \+Z/);
    const verify = () => verifyOutput('h3',manifest,qa,undefined,digest,undefined,selection);
    const proof = verify();
    assert.deepEqual([proof.h3Denoiser,proof.denoiserFile,proof.denoiserSha256,proof.fallback],[selection,name,hash,6]);
    if (selection === 'int8') {
      const local = structuredClone(manifest);
      for (const dir of dirs) local.faces[dir].edit.provenance.weights['diffusion_models/'+name] = digest;
      const localProof = () => verifyOutput('h3',local,qa,undefined,digest);
      assert.equal(localProof().denoiserSha256, digest, 'The default reports the actual same-name local file');
      assert.throws(() => verifyOutput('h3',local,qa,undefined,digest,undefined,'int8'), /Wrong H3 weight digest/);
      local.faces['-Y'].edit.provenance.weights['diffusion_models/'+name] = hash;
      assert.throws(localProof, /Inconsistent H3 weight digest on -Y/);
    }
    const provenance = manifest.faces['-Y'].edit.provenance;
    provenance.weights['diffusion_models/'+name] = digest;
    assert.throws(verify, /Wrong H3 weight digest on -Y/);
    provenance.recipe.model = 'other.safetensors';
    provenance.weights['diffusion_models/other.safetensors'] = hash;
    assert.throws(verify, /Wrong H3 denoiser in the saved recipe on -Y/);
  }
});

test('an existing GLB and one face need only the appearance model, with consistent canvas and native dimensions', async () => {
  for (const [model, modelCount] of [['qwen', 5], ['h3', 4]]) {
    const graph = await load(model);
    const ids = graph.nodes.filter(n => n.type === 'BICachedGeneratedFace').map(n => n.id);
    configureEditGraph(graph, model, {mesh: 'examples/chair.glb', size: 512, face: '+Z'});
    const inventory = preflight(graph, {}, 'reference.png');
    assert.equal(inventory.models.length, modelCount);
    assert(!inventory.missingNodes.some(kind => kind.includes('Pixal')));
    assert.equal(graph.definitions.subgraphs.length, 1);
    const encoder = graph.definitions.subgraphs[0].nodes.find(n => n.type === (model === 'qwen' ? 'TextEncodeQwenImage21' : 'MiniMaxH3ImageConditioning'));
    assert.deepEqual(encoder.widgets_values.slice(model === 'qwen' ? 2 : 1), model === 'qwen' ? [512] : [512, 512]);
    const proxy = graph.nodes.find(n => n.type === 'BICanonicalizeProxy');
    assert.equal(proxy.widgets_values[0], 'examples/chair.glb');
    assert(proxy.inputs.every(p => p.link == null));
    const rgb = graph.nodes.find(n => n.type === 'BIPrepareReference').outputs[0];
    for (const n of graph.nodes.filter(n => n.type === 'BIModelCanvas')) {
      assert.equal(n.widgets_values[1], 512);
      assert(rgb.links.includes(n.inputs[0].link));
    }
    const collector = graph.nodes.find(n => n.type === 'BIFaceInputs');
    assert.deepEqual(collector.inputs.filter(p => p.link != null).map(p => p.name), ['pz']);
    assert.deepEqual(graph.nodes.filter(n => n.type === 'BICachedGeneratedFace').map(n => n.id), ids);
    for (const [id, source, slot, target, input] of graph.links) {
      assert(graph.nodes.find(n => n.id === source).outputs[slot].links.includes(id));
      assert.equal(graph.nodes.find(n => n.id === target).inputs[input].link, id);
    }
  }
  const graph = await load('qwen');
  for (const size of [-32, 16, 255, 4097, NaN]) assert.throws(() => configureEditGraph(graph, 'qwen', { size }), /multiple of 32/);
});

test('a failed preflight writes evidence without sending a queue request', async () => {
  const requests = [];
  const server = createServer((req, res) => {
    requests.push([req.method, req.url]);
    res.end('{}');
  });
  await new Promise(ok => server.listen(0, '127.0.0.1', ok));
  const output = await mkdtemp(join(tmpdir(), 'bi-modeltest-'));
  try {
    const opts = { model: 'h3', origin: 'http://127.0.0.1:' + server.address().port, input: 'object.png', output,
      run: true, 'timeout-minutes': '15' };
    await assert.rejects(() => run({...opts,'h3-denoiser':'w4a8'}), /minimax_h3_image_pruned_w4a8/);
    assert.deepEqual(requests, [['GET', '/object_info']]);
    const [dir] = await readdir(output);
    const report = JSON.parse(await readFile(join(output, dir, 'preflight.json')));
    assert(!report.ready && !report.inputAvailable);
    assert.equal(report.h3Denoiser, 'w4a8');
    assert.equal(report.models.find(m => m.name === h3Weights.w4a8[0]).expectedSha256, h3Weights.w4a8[1]);
    assert(report.models.find(m => m.loader === 'LoadMoGeModel' && m.directory === 'geometry_estimation'));
  } finally {
    await new Promise(ok => server.close(ok));
    await rm(output, { recursive: true, force: true });
  }
});

test('changing one seed keeps five seeds, every sampler link, and the full six-face result', async () => {
  for (const model of ['qwen', 'h3']) {
    const graph = await load(model);
    const before = structuredClone(graph);
    const faceSeed = { face: '-Y', seed: 321 };
    configureEditGraph(graph, model, {faceSeed});
    const changed = graph.nodes.filter((node, i) => JSON.stringify(node) !== JSON.stringify(before.nodes[i]));
    assert.equal(changed.length, 1);
    assert.equal(changed[0].type, 'PrimitiveInt');
    assert.equal(changed[0].widgets_values[0], 321);
    const target = graph.nodes.find(n => n.type === 'BICachedGeneratedFace' && n.widgets_values[0] === '-Y');
    assert.equal(graph.links.find(([id]) => id === target.inputs.find(p => p.name === 'seed').link)[1], changed[0].id);
    assert.deepEqual(graph.links, before.links);
    assert.deepEqual(graph.definitions, before.definitions);
    assert.throws(() => configureEditGraph(graph, model, {face:'+Z', faceSeed}), /target the requested/);
    assert.throws(() => configureEditGraph(graph, model, {faceSeed:{...faceSeed, seed:2**53}}), /safe integer/);
  }
  await assert.rejects(run({model:'qwen','face-seed':'-Y:bad'}), /such as/);
  await assert.rejects(run({model:'pixal3d','face-seed':'+X:300'}), /needs qwen or h3/);
});

test('one-face run keeps sampler IDs and labels fresh, reused and unobserved stages separately', () => {
  const node = (class_type, inputs) => ({ class_type, inputs });
  const prompt = {
    mesh: node('BICanonicalizeProxy', { model_3d: ['meshSampler', 0] }),
    meshSampler: node('KSampler', { latent_image: ['voxelSampler', 0] }),
    voxelSampler: node('KSampler', {}), guides: node('BIMakeSixGuides', { proxy: ['mesh', 0] }),
    surface: node('KSampler', { positive: ['guides', 3] }), decoded: node('VAEDecode', { samples: ['surface', 0] }),
    face: node('BICachedGeneratedFace', { face_id: '-Y', image: ['decoded', 0] }),
    collect: node('BIFaceInputs', { px: ['x', 0], ny: ['face', 0] }),
  };
  selectFace(prompt, '-Y');
  assert.deepEqual(prompt.collect.inputs, { ny: ['face', 0] });
  const trace = { started: true, succeeded: true, nodes: new Set(['surface', 'meshSampler']) };
  const report = inferenceReport(prompt, '-Y', trace);
  assert.equal(report['-Y'].state, 'sampled');
  assert.equal(report.pixal3d.state, 'partially_sampled');
  trace.nodes.clear();
  assert.equal(inferenceReport(prompt, '-Y', trace)['-Y'].state, 'reused');
  trace.started = false;
  assert.equal(inferenceReport(prompt, '-Y', trace)['-Y'].state, 'unknown');
});

test('report distinguishes six fallbacks, accepted edits, and untouched baseline faces', () => {
  const manifest = { layout: 'box', asset_id: digest, reference: { sha256: digest }, mode: 'qwen_enhanced',
    provenance: { meshGeneration: { backend: 'native_mesh', weights: { 'diffusion_models/pixal3d_int8_convrot.safetensors': digest } } },
    faces: {} };
  const qa = { faces: {} };
  for (const dir of dirs) {
    manifest.faces[dir] = { guideSha256: digest, edit: { source: 'qwen_enhanced', status: 'fallback', reason: 'silhouette-iou',
      provenance: { faceId: dir, referenceSha256: digest, guideSha256: digest, recipe: { model: 'qwen.safetensors' },
        weights: { 'diffusion_models/qwen.safetensors': digest } } } };
    qa.faces[dir] = { status: 'fallback' };
  }
  const failed = verifyOutput('qwen', manifest, qa);
  assert.equal(failed.accepted, 0);
  assert.equal(failed.fallback, 6);
  for (const dir of dirs.slice(1)) {
    delete manifest.faces[dir].edit;
    qa.faces[dir].status = 'baseline';
  }
  manifest.faces['+X'].edit.status = qa.faces['+X'].status = 'accepted';
  const one = verifyOutput('qwen', manifest, qa, '+X');
  assert.equal(one.accepted, 1);
  assert.equal(one.baselineFaces, 5);
  assert.equal(one.fallback, 0);
  const supplied = structuredClone(manifest);
  delete supplied.reference;
  supplied.provenance = { originalGlbSha256: 'b'.repeat(64) };
  supplied.faces['+X'].edit.reference = 'reference/' + digest + '/source.png';
  assert.equal(verifyOutput('qwen', supplied, qa, '+X', 'b'.repeat(64)).accepted, 1);
  assert.throws(() => verifyOutput('qwen', supplied, qa, '+X', digest), /GLB digest/);
  supplied.faces['+X'].edit.reference = 'wrong.png';
  assert.throws(() => verifyOutput('qwen', supplied, qa, '+X', 'b'.repeat(64)));
  manifest.faces['+X'].edit.provenance.weights = {'diffusion_models/wrong-name.safetensors':digest};
  assert.throws(() => verifyOutput('qwen', manifest, qa, '+X'), /weight digest/);
});

test('TripoSplat preflight checks five weights, rasterizer and matching input GLB without queueing', async () => {
  const graph = await load('triposplat');
  const required = preflight(graph, {}, 'sample.png');
  assert(required.missingNodes.includes('RenderSplat'));
  assert.equal(required.models.length, 5);
  const registry = Object.fromEntries(required.missingNodes.map(kind => [kind, {input: {required: {}}}]));
  registry.LoadImage.input.required.image = [['sample.png']];
  for (const m of required.models) (registry[m.loader].input.required.name ??= [[]])[0].push(m.name);
  const head = Buffer.alloc(12);
  head.write('glTF'); head.writeUInt32LE(2,4); head.writeUInt32LE(528,8);
  const paths = [];
  const server = createServer((req, res) => {
    const url = new URL(req.url, 'http://local');
    paths.push(url.pathname);
    if (url.pathname === '/object_info') return res.end(JSON.stringify(registry));
    assert.equal(url.pathname, '/view');
    assert.equal(url.searchParams.get('subfolder'), 'pair');
    assert.equal(url.searchParams.get('filename'), 'mesh.glb');
    assert.equal(url.searchParams.get('type'), 'input');
    assert.equal(req.headers.range, 'bytes=0-11');
    res.writeHead(206);
    res.end(head);
  });
  await new Promise(ok => server.listen(0, '127.0.0.1', ok));
  const output = await mkdtemp(join(tmpdir(), 'bi-tripo-test-'));
  try {
    const opts = {model:'triposplat', input:'sample.png', mesh:'pair/mesh.glb', face:'-Y', output,
      origin:'http://127.0.0.1:'+server.address().port, 'timeout-minutes':'1'};
    await run(opts);
    const [dir] = await readdir(output);
    const report = JSON.parse(await readFile(join(output,dir,'preflight.json')));
    assert(report.ready && report.meshAvailable);
    assert.equal(report.face, '-Y');
    assert.equal(report.coordinateSpace, 'canonical_box');
    head.write('json');
    await assert.rejects(() => run({...opts,run:true}), /pair\/mesh\.glb \(GLB v2\)/);
    await assert.rejects(() => run({...opts,mesh:'../mesh.glb'}), /relative/);
    assert.deepEqual(paths, ['/object_info','/view','/object_info','/view']);
  } finally {
    await new Promise(ok => server.close(ok));
    await rm(output, {recursive:true,force:true});
  }
});

test('splat guide evidence binds the actual face, mesh SHA, PNGs and sampler trace', async () => {
  const proxy = Buffer.from('mesh guide payload');
  const proxyHash = createHash('sha256').update(proxy).digest('hex');
  const png = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAYElEQVR4nO3PQQ0AIBDAMED5SUcEj4ZkVbDtmVk/OzrgVQNaA1oDWgNaA1oDWgNaA1oDWgNaA1oDWgNaA1oDWgNaA1oDWgNaA1oDWgNaA1oDWgNaA1oDWgNaA1oDWgPaBaIsAgBhHc02AAAAAElFTkSuQmCC','base64');
  const report = {asset:digest,splatSha256:digest,rendererSha256:digest,meshGuideSha256:proxyHash,stageKey:digest,
    faceId:'-Z',coordinateSpace:'source_glb',resolution:64,qa:{splatPixels:835,meshPixels:832,iou:.92,
      relativeDepth:{status:'measured',samples:700,spearman:.8}}};
  const manifest = {asset_id:digest,resolution:64,provenance:{originalGlbSha256:digest},
    faces:{'-Z':{guide:'guides/nz.png',guideSha256:proxyHash}}};
  const prompt = {img:{class_type:'PreviewImage',inputs:{}}, qa:{class_type:'PreviewAny',inputs:{}},
    sampler:{class_type:'KSampler',inputs:{}}, decoder:{class_type:'VAEDecodeTripoSplat',inputs:{samples:['sampler',0]}},
    cache:{class_type:'BICachedTripoSplat', inputs:{splat:['decoder',0]}},
    proxy:{class_type:'BICanonicalizeProxy',inputs:{filename:'mesh.glb'}}};
  const outputs = {img:{images:[{type:'temp',filename:'temporary.png',subfolder:''}]},qa:{text:[JSON.stringify(report)]}};
  const server = createServer((req,res) => {
    const url = new URL(req.url,'http://local');
    if (url.pathname.endsWith('/manifest.json')) return res.end(JSON.stringify(manifest));
    if (url.pathname.endsWith('/guides/nz.png')) return res.end(proxy);
    assert.equal(url.pathname,'/view');
    assert.equal(url.searchParams.get('type'),'temp');
    res.end(png);
  });
  await new Promise(ok => server.listen(0,'127.0.0.1',ok));
  const dir = await mkdtemp(join(tmpdir(),'bi-tripo-output-'));
  try {
    const summary = await saveSplatOutput(dir,'http://127.0.0.1:'+server.address().port,prompt,outputs,
      {face:'-Z','coordinate-space':'source_glb'},
      {started:true,succeeded:true,nodes:new Set(['sampler'])},workflows.triposplat,'prompt-id');
    assert.equal(summary.inference.triposplat.state,'sampled');
    assert.deepEqual(Object.keys(summary.inference),['triposplat']);
    assert.equal(summary.qa.iou,.92);
    assert.deepEqual(await readFile(join(dir,'splat-guide.png')),png);
    assert.deepEqual(await readFile(join(dir,'mesh-guide.png')),proxy);
    assert.throws(() => verifySplatOutput(report, manifest,'+Z','source_glb'));
    assert.throws(() => verifySplatOutput({...report,meshGuideSha256:digest}, manifest,'-Z','source_glb'));
    assert.equal(inferenceReport(prompt,null,{started:true,succeeded:true,nodes:new Set()}).triposplat.state,'reused');
  } finally {
    await new Promise(ok=>server.close(ok));
    await rm(dir,{recursive:true,force:true});
  }
});
