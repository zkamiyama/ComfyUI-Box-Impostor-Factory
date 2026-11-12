import assert from 'node:assert/strict';
import { readFile, mkdtemp, readdir, rm } from 'node:fs/promises';
import { createServer } from 'node:http';
import { createHash } from 'node:crypto';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { preflight, workflows, run, inferenceReport, selectFace, verifyOutput, saveSplatOutput, verifySplatOutput } from '../scripts/run_model_workflow.mjs';

const dirs = ['+X', '-X', '+Y', '-Y', '+Z', '-Z'];
const digest = 'a'.repeat(64);
const load = name => readFile(new URL('../workflows/' + workflows[name] + '.json', import.meta.url), 'utf8').then(JSON.parse);

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
    if (name === 'qwen') {
      registry.LoraLoaderModelOnly.input.required.model[1].options = ['other.safetensors'];
      const report = preflight(graph, registry, 'object.png');
      assert.equal(report.ready, false);
      assert.deepEqual(report.models.filter(m => !m.available).map(m => m.name), ['QI2.1_AnyAngle.safetensors']);
    }
  }
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
    await assert.rejects(() => run(opts), /input\/object.png/);
    assert.deepEqual(requests, [['GET', '/object_info']]);
    const [dir] = await readdir(output);
    const report = JSON.parse(await readFile(join(output, dir, 'preflight.json')));
    assert(!report.ready && !report.inputAvailable);
    assert(report.models.find(m => m.loader === 'LoadMoGeModel' && m.directory === 'geometry_estimation'));
  } finally {
    await new Promise(ok => server.close(ok));
    await rm(output, { recursive: true, force: true });
  }
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
  manifest.faces['+X'].edit.provenance.weights = {};
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
