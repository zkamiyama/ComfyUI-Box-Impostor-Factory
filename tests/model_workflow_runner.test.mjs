import assert from 'node:assert/strict';
import { readFile, mkdtemp, readdir, rm } from 'node:fs/promises';
import { createServer } from 'node:http';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { preflight, workflows, run, inferenceReport, selectFace, verifyOutput } from '../scripts/run_model_workflow.mjs';

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
