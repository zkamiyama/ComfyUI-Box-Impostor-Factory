// Use the installed Comfy frontend to compile and run the distributed visual graph.
import assert from 'node:assert/strict';
import { createHash, randomUUID } from 'node:crypto';
import { mkdir, mkdtemp, readFile, writeFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';
import { pathToFileURL } from 'node:url';
import { parseArgs } from 'node:util';
import { chromiumOptions } from '../worker/chromium.mjs';

export const workflows = {
  pixal3d: '05_pixal3d_baseline', qwen: '02_box_from_reference_qwen21', h3: '03_box_from_reference_h3',
};
const directions = ['+X', '-X', '+Y', '-Y', '+Z', '-Z'];
const slugs = { '+X': 'px', '-X': 'nx', '+Y': 'py', '-Y': 'ny', '+Z': 'pz', '-Z': 'nz' };
const sha256 = /^[a-f\d]{64}$/;
const jsonFile = (path, value) => writeFile(path, JSON.stringify(value, null, 2) + '\n');

function comboHas(spec, value) {
  const options = Array.isArray(spec?.[0]) ? spec[0] : spec?.[1]?.options;
  return Array.isArray(options) && options.includes(value);
}

// Read choices from the distributed graph before the frontend has a chance to
// replace a missing model selection with the loader's first installed model.
export function preflight(graph, registry, input) {
  const subgraphs = graph.definitions?.subgraphs ?? [];
  const subgraphIds = new Set(subgraphs.map(s => s.id));
  const missingNodes = new Set();
  const models = new Map();
  for (const scope of [graph, ...subgraphs]) {
    for (const node of scope.nodes) {
      if (subgraphIds.has(node.type)) continue;
      const info = registry[node.type];
      if (!info) missingNodes.add(node.type);
      const pins = { ...info?.input?.required, ...info?.input?.optional };
      for (const name of (node.widgets_values ?? []).filter(v => typeof v === 'string' && v.endsWith('.safetensors'))) {
        const decl = node.properties?.models?.find(m => m.name === name);
        const available = Object.values(pins).some(spec => comboHas(spec, name));
        models.set(node.type + '/' + name, { name, loader: node.type, directory: decl?.directory ??
          (node.type === 'LoraLoaderModelOnly' ? 'loras' : null), available });
      }
    }
  }
  const inputAvailable = comboHas(registry.LoadImage?.input?.required?.image, input);
  return { ready: !missingNodes.size && inputAvailable && [...models.values()].every(m => m.available),
    missingNodes: [...missingNodes].sort(), input, inputAvailable, models: [...models.values()] };
}

export function checkCompiled(prompt, model, report) {
  const nodes = Object.values(prompt);
  const byType = type => nodes.filter(n => n.class_type === type);
  assert.deepEqual(byType('LoadImage').map(n => n.inputs.image), [report.input], 'Frontend changed the reference image');
  for (const { name, loader } of report.models)
    assert(byType(loader).some(n => Object.values(n.inputs).includes(name)), 'Frontend lost ' + name);
  for (const type of ['Pixal3DConditioning', 'MeshToFile3D', 'BIPreviewBox', 'BISaveBox'])
    assert.equal(byType(type).length, 1, 'Missing or duplicate ' + type);
  if (model !== 'pixal3d') {
    const type = model === 'qwen' ? 'TextEncodeQwenImage21' : 'MiniMaxH3ImageConditioning';
    assert.equal(byType(type).length, 6, 'Missing generated view branches');
    assert.deepEqual(byType('BICachedGeneratedFace').map(n => n.inputs.face_id).sort(), [...directions].sort());
    const groups = samplerGroups(prompt);
    for (const n of byType('BICachedGeneratedFace')) {
      const { face_id, mask_mode, mask, image, seed } = n.inputs;
      assert.equal(mask_mode, 'foreground', 'Missing generated-image foreground coverage on ' + face_id);
      const matte = prompt[mask?.[0]];
      assert.equal(matte?.class_type, 'RemoveBackground');
      assert.deepEqual(matte.inputs.image, image);
      assert.equal(groups[face_id].length, 1);
      assert.deepEqual(prompt[groups[face_id][0]].inputs.seed, seed);
      assert.equal(prompt[seed[0]]?.inputs?.value, 101 + directions.indexOf(face_id), 'Unexpected face seed');
    }
  }
}

// Keep every upstream node ID intact so this test can warm one face of a subsequent six-face run.
export function selectFace(prompt, face) {
  if (!face) return;
  const collector = Object.values(prompt).find(n => n.class_type === 'BIFaceInputs');
  const pin = slugs[face];
  assert(pin && collector?.inputs?.[pin], 'No connected face ' + face);
  collector.inputs = { [pin]: collector.inputs[pin] };
}

function samplerGroups(prompt, face) {
  function samplers(link, seen = new Set()) {
    if (!Array.isArray(link) || !prompt[link[0]] || seen.has(link[0])) return [];
    const [id] = link;
    seen.add(id);
    const node = prompt[id];
    if (node.class_type === 'KSampler') return [id, ...Object.values(node.inputs).flatMap(v => samplers(v, seen))];
    if (['BIMakeSixGuides', 'BIPrepareReference'].includes(node.class_type)) return [];
    return Object.values(node.inputs).flatMap(v => samplers(v, seen));
  }
  const result = {};
  for (const node of Object.values(prompt)) {
    if (node.class_type === 'BICanonicalizeProxy') result.pixal3d = samplers(node.inputs.model_3d);
    if (node.class_type === 'BICachedGeneratedFace' && (!face || node.inputs.face_id === face)) {
      result[node.inputs.face_id] = samplers(node.inputs.image);
    }
  }
  return result;
}

// History records cached nodes at queue start. Only the prompt's live execution
// events can distinguish a newly run sampler from a lazy on-disk cache hit.
export function inferenceReport(prompt, face, trace) {
  const result = {};
  for (const [stage, ids] of Object.entries(samplerGroups(prompt, face))) {
    assert(ids.length, 'No samplers identified for ' + stage);
    const ran = ids.filter(id => trace?.nodes.has(id));
    const coverage = trace?.started && trace?.succeeded;
    const state = ran.length === ids.length ? 'sampled' : !coverage ? 'unknown'
      : ran.length ? 'partially_sampled' : 'reused';
    result[stage] = { state, samplerIds: ids, observedRunning: ran };
  }
  return result;
}

async function observe(origin, clientId) {
  const url = new URL('/ws', origin);
  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
  url.searchParams.set('clientId', clientId);
  const ws = new WebSocket(url);
  const traces = new Map();
  ws.addEventListener('message', e => {
    if (typeof e.data !== 'string') return;
    const { type, data } = JSON.parse(e.data);
    if (!data?.prompt_id) return;
    const trace = traces.get(data.prompt_id) ?? { nodes: new Set() };
    traces.set(data.prompt_id, trace);
    if (type === 'execution_start') trace.started = true;
    if (type === 'execution_success') trace.succeeded = true;
    if (type === 'executing' && data.node != null) trace.nodes.add(String(data.node));
  });
  await new Promise((ok, fail) => {
    const timer = setTimeout(() => { ws.close(); fail(new Error('ComfyUI WebSocket timed out')); }, 15_000);
    ws.addEventListener('open', () => { clearTimeout(timer); ok(); }, { once: true });
    ws.addEventListener('error', () => { clearTimeout(timer); fail(new Error('Could not open ' + url)); }, { once: true });
  });
  return { traces, close: () => ws.close() };
}

async function compile(origin, graph) {
  const { chromium } = await import('playwright');
  const browser = await chromium.launch(chromiumOptions('webgl2'));
  try {
    const page = await browser.newPage();
    await page.goto(origin, { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(() => window.app?.extensionManager && window.app?.rootGraph && window.app?.canvas,
      null, { timeout: 120_000 });
    return await page.evaluate(async workflow => {
      const file = new File([JSON.stringify(workflow)], 'workflow.json', { type: 'application/json' });
      await window.app.handleFile(file, 'file_drop', { deferWarnings: true });
      return (await window.app.graphToPrompt()).output;
    }, graph);
  } finally {
    await browser.close();
  }
}

async function response(url, init) {
  const result = await fetch(url, { signal: AbortSignal.timeout(60_000), ...init });
  if (!result.ok) throw new Error(url + ' HTTP ' + result.status + ' ' + (await result.text()).slice(0, 4000));
  return result;
}

export function verifyOutput(model, manifest, qa, face) {
  assert.equal(manifest.layout, 'box');
  assert.match(manifest.asset_id, sha256);
  assert.match(manifest.reference?.sha256 ?? '', sha256, 'Missing reference digest');
  const mesh = manifest.provenance?.meshGeneration;
  assert.equal(mesh?.backend, 'native_mesh');
  assert(Object.keys(mesh.weights).some(k => k.endsWith('/pixal3d_int8_convrot.safetensors') &&
    sha256.test(mesh.weights[k])), 'Missing Pixal3D weight digest');
  assert.deepEqual(Object.keys(manifest.faces).sort(), [...directions].sort());
  if (model === 'pixal3d') {
    assert.equal(manifest.mode, 'geometry_baseline');
    return { mode: manifest.mode };
  }
  const backend = model === 'qwen' ? 'qwen_enhanced' : 'h3_enhanced';
  const faces = {};
  for (const dir of directions) {
    const edit = manifest.faces[dir].edit;
    if (face && face !== dir) {
      assert.equal(edit, undefined, 'Unrequested appearance edit on ' + dir);
      assert.equal(qa?.faces?.[dir]?.status, 'baseline');
      continue;
    }
    assert.equal(edit?.source, backend);
    assert.equal(edit.provenance?.faceId, dir);
    assert.equal(edit.provenance?.referenceSha256, manifest.reference.sha256);
    assert.equal(edit.provenance?.guideSha256, manifest.faces[dir].guideSha256);
    const weight = edit.provenance.recipe.model;
    assert(Object.entries(edit.provenance.weights).some(([k, hash]) => k === 'diffusion_models/' + weight &&
      sha256.test(hash)), 'Missing appearance-model weight digest on ' + dir);
    assert(['accepted', 'fallback'].includes(edit.status));
    assert.equal(qa?.faces?.[dir]?.status, edit.status);
    faces[dir] = { status: edit.status, reason: edit.reason };
  }
  const accepted = Object.values(faces).filter(f => f.status === 'accepted').length;
  return { mode: manifest.mode, accepted, fallback: Object.keys(faces).length - accepted, baselineFaces: 6 - Object.keys(faces).length, faces };
}

function assetURL(value, base) {
  assert(value && typeof value === 'string', 'Missing asset URL');
  const url = new URL(value, base);
  assert.equal(url.origin, new URL(base).origin);
  return url;
}

export async function run(opts) {
  const name = workflows[opts.model];
  if (!Object.hasOwn(workflows, opts.model)) throw new Error('Choose --model pixal3d, qwen or h3.');
  if (opts.face && (!directions.includes(opts.face) || opts.model === 'pixal3d')) throw new Error('--face needs qwen or h3 and one of +X, -X, +Y, -Y, +Z, -Z.');
  const origin = new URL(opts.origin);
  if (!['http:', 'https:'].includes(origin.protocol)) throw new Error('--origin must be HTTP(S).');
  const timeout = Number(opts['timeout-minutes']);
  if (!(Number.isFinite(timeout) && timeout > 0)) throw new Error('--timeout-minutes must be positive.');
  await mkdir(opts.output, { recursive: true });
  const dir = await mkdtemp(join(resolve(opts.output), opts.model + '-'));
  console.log('Evidence:', dir);
  const graph = JSON.parse(await readFile(new URL('../workflows/' + name + '.json', import.meta.url), 'utf8'));
  const registry = await (await response(new URL('/object_info', origin))).json();
  const report = preflight(graph, registry, opts.input);
  await jsonFile(join(dir, 'preflight.json'), report);
  if (!report.ready) {
    const missing = [...report.missingNodes, ...report.models.filter(m => !m.available).map(m => (m.directory ?? m.loader) + '/' + m.name)];
    if (!report.inputAvailable) missing.push('input/' + opts.input);
    throw new Error('Missing on this ComfyUI server:\n' + missing.join('\n'));
  }
  if (!opts.run) { console.log('Preflight passed.'); return; }

  const image = graph.nodes.find(n => n.type === 'LoadImage');
  image.widgets_values[0] = opts.input;
  if (image.widgets_values_named) image.widgets_values_named.image = opts.input;
  const prompt = await compile(origin.href, graph);
  checkCompiled(prompt, opts.model, report);
  selectFace(prompt, opts.face);
  await jsonFile(join(dir, 'prompt.json'), prompt);
  const stats = await (await response(new URL('/system_stats', origin))).json();
  await jsonFile(join(dir, 'environment.json'), { devices: stats.devices, system: {
    comfyui_version: stats.system?.comfyui_version, pytorch_version: stats.system?.pytorch_version,
    python_version: stats.system?.python_version, os: stats.system?.os } });
  const clientId = randomUUID();
  const live = await observe(origin, clientId);
  try {
    const receipt = await (await response(new URL('/prompt', origin), { method: 'POST',
      headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ prompt, client_id: clientId }) })).json();
    assert.equal(typeof receipt.prompt_id, 'string', 'ComfyUI returned no prompt ID');
    await jsonFile(join(dir, 'receipt.json'), receipt);
    console.log('Queued:', receipt.prompt_id);
    const deadline = Date.now() + timeout * 60_000;
    const historyURL = new URL('/history/' + encodeURIComponent(receipt.prompt_id), origin);
    while (Date.now() < deadline) {
      const history = (await (await response(historyURL)).json())[receipt.prompt_id];
      if (!history) { await delay(2000); continue; }
      await jsonFile(join(dir, 'status.json'), history.status);
      if (history.status?.status_str !== 'success') throw new Error('Execution failed. See ' + join(dir, 'status.json'));
      const preview = Object.entries(prompt).find(([, n]) => n.class_type === 'BIPreviewBox')[0];
      const viewer = assetURL(history.outputs[preview]?.box_impostor?.[0], origin);
      const manifestURL = assetURL(viewer.searchParams.get('manifest'), origin);
      const manifest = await (await response(manifestURL)).json();
      await jsonFile(join(dir, 'manifest.json'), manifest);
      let qa;
      if (manifest.qa) {
        qa = await (await response(assetURL(manifest.qa, manifestURL))).json();
        await jsonFile(join(dir, 'qa.json'), qa);
      }
      const summary = { workflow: name, prompt_id: receipt.prompt_id, asset_id: manifest.asset_id,
        viewer: viewer.href, inference: inferenceReport(prompt, opts.face, live.traces.get(receipt.prompt_id)),
        ...verifyOutput(opts.model, manifest, qa, opts.face) };
      const views = join(dir, 'views');
      await mkdir(views);
      for (const side of opts.face ? [opts.face] : directions) {
        const entry = manifest.faces[side];
        const guide = Buffer.from(await (await response(assetURL(entry.guide, manifestURL))).arrayBuffer());
        assert.equal(createHash('sha256').update(guide).digest('hex'), entry.guideSha256, 'Guide digest mismatch on ' + side);
        await writeFile(join(views, slugs[side] + '-guide.png'), guide);
        if (entry.edit) {
          const generated = await response(assetURL(entry.edit.view, manifestURL));
          await writeFile(join(views, slugs[side] + '-generated.png'), new Uint8Array(await generated.arrayBuffer()));
        }
      }
      const hero = await response(assetURL('previews/hero.png', manifestURL));
      await writeFile(join(dir, 'preview.png'), new Uint8Array(await hero.arrayBuffer()));
      await jsonFile(join(dir, 'result.json'), summary);
      console.log(JSON.stringify(summary, null, 2));
      return;
    }
    throw new Error('Timed out waiting. The queued job may still be running. History: ' + historyURL);
  } finally {
    live.close();
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  const { values } = parseArgs({ options: { model: { type: 'string' }, origin: { type: 'string', default: 'http://127.0.0.1:8188' },
    input: { type: 'string', default: 'reference.png' }, output: { type: 'string', default: '.test-output/models' },
    face: { type: 'string' }, run: { type: 'boolean', default: false }, 'timeout-minutes': { type: 'string', default: '180' }, help: { type: 'boolean' } } });
  if (values.help) {
    console.log('node scripts/run_model_workflow.mjs --model pixal3d|qwen|h3 [--origin URL] [--input reference.png] [--output DIR] [--face=+Z] [--run] [--timeout-minutes 180]\nPlace the image in ComfyUI/input. Default: check installed models and input, without queuing.');
  } else {
    run(values).catch(error => { console.error(error.message); process.exitCode = 1; });
  }
}
