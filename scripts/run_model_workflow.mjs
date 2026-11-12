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
  triposplat: '08_triposplat_guide',
};
const directions = ['+X', '-X', '+Y', '-Y', '+Z', '-Z'];
const slugs = { '+X': 'px', '-X': 'nx', '+Y': 'py', '-Y': 'ny', '+Z': 'pz', '-Z': 'nz' };
const sha256 = /^[a-f\d]{64}$/;
const jsonFile = (path, value) => writeFile(path, JSON.stringify(value, null, 2) + '\n');

// Keep all four Pixal3D sampling stages and the PBR texture bakes. Reduce the
// sparse cascade, post-processing mesh and UV sizes together for a smaller GPU.
const compactProxy = {
  Trellis2UpsampleStage: { target_resolution: 1024 },
  RemeshMesh: { resolution: 256 },
  DecimateMesh: { target_face_count: 100_000 },
  PrimitiveInt: { value: 1024 }, // shared UV unwrap and voxel-to-texture bake size
  BakeNormalMapFromMesh: { resolution: 1024 },
  BakeAmbientOcclusion: { resolution: 512, samples: 32 },
};

export function configureCompactProxy(graph) {
  const geometry = graph.definitions?.subgraphs.find(s => s.nodes.some(n => n.type === 'Pixal3DConditioning'));
  assert(geometry, '--compact-proxy requires the Pixal3D geometry subgraph');
  for (const [type, settings] of Object.entries(compactProxy)) {
    const nodes = geometry.nodes.filter(n => n.type === type);
    assert.equal(nodes.length, 1, 'Missing or ambiguous compact proxy node ' + type);
    const [node] = nodes;
    // The changed inputs are the leading widgets of these pinned native nodes.
    Object.keys(settings).forEach((name, index) => {
      assert(Object.hasOwn(node.widgets_values_named, name));
      assert.equal(node.widgets_values[index], node.widgets_values_named[name], type + '.' + name);
      node.widgets_values[index] = node.widgets_values_named[name] = settings[name];
    });
  }
}

function disconnect(graph, pin) {
  if (pin.link === null) return;
  const link = graph.links.find(([id]) => id === pin.link);
  assert(link, 'Missing workflow link ' + pin.link);
  const [id, source, slot] = link;
  const output = graph.nodes.find(n => n.id === source).outputs[slot];
  output.links = output.links.filter(value => value !== id);
  graph.links = graph.links.filter(edge => edge !== link);
  pin.link = null;
}

// Derive the optional supplied-GLB route from the same editable model subgraph as
// the full workflow. Keep its node and link IDs so one-face runs can reuse caches.
export function configureEditGraph(graph, model, { mesh, size, face, faceSeed, h3ReferenceFirst } = {}) {
  const qwen = model === 'qwen';
  if (!qwen && model !== 'h3') throw new Error('An appearance workflow needs qwen or h3.');
  const canvases = graph.nodes.filter(n => n.type === 'BIModelCanvas');
  assert.equal(canvases.length, 6);
  if (h3ReferenceFirst) {
    assert.equal(model, 'h3', '--h3-reference-first needs h3.');
    const switches = graph.nodes.filter(n => n.type === 'PrimitiveBoolean');
    assert.equal(switches.length, 1, 'Expected the shared H3 image-order switch');
    switches[0].widgets_values[0] = true;
    if (switches[0].widgets_values_named) switches[0].widgets_values_named.value = true;
  }
  if (faceSeed) {
    assert(Object.hasOwn(slugs, faceSeed.face) && Number.isSafeInteger(faceSeed.seed) && faceSeed.seed >= 0,
      '--face-seed expects a direction and a nonnegative safe integer, for example +Z:321.');
    assert(!face || face === faceSeed.face, '--face-seed must target the requested --face.');
    const target = graph.nodes.find(n => n.type === 'BICachedGeneratedFace' && n.widgets_values[0] === faceSeed.face);
    const link = target.inputs.find(pin => pin.name === 'seed').link;
    const source = graph.links.find(([id]) => id === link)[1];
    const seed = graph.nodes.find(n => n.id === source);
    assert.equal(seed.type, 'PrimitiveInt');
    seed.widgets_values[0] = faceSeed.seed;
    if (seed.widgets_values_named) seed.widgets_values_named.value = faceSeed.seed;
  }
  if (size !== undefined) {
    assert(Number.isSafeInteger(size) && size >= 32 && size <= 4096 && size % 32 === 0,
      '--size must be a multiple of 32 from 32 through 4096.');
    for (const n of canvases) n.widgets_values[1] = size;
    const type = qwen ? 'TextEncodeQwenImage21' : 'MiniMaxH3ImageConditioning';
    const encoders = graph.definitions.subgraphs.flatMap(s => s.nodes.filter(n => n.type === type));
    assert.equal(encoders.length, 1, 'Expected one shared view encoder definition');
    encoders[0].widgets_values[2] = size; // Qwen resolution or H3 height
    if (!qwen) encoders[0].widgets_values[1] = size; // H3 width
  }
  if (face) {
    assert(Object.hasOwn(slugs, face), 'Unknown face ' + face);
    const collector = graph.nodes.find(n => n.type === 'BIFaceInputs');
    for (const pin of collector.inputs) if (pin.name !== slugs[face]) disconnect(graph, pin);
  }
  if (!mesh) return;
  const proxy = graph.nodes.find(n => n.type === 'BICanonicalizeProxy');
  const prepared = graph.nodes.find(n => n.type === 'BIPrepareReference');
  const pixalLink = proxy.inputs.find(p => p.name === 'model_3d').link;
  const pixalId = graph.links.find(([id]) => id === pixalLink)[1];
  const pixal = graph.nodes.find(n => n.id === pixalId);
  assert(graph.definitions.subgraphs.find(s => s.id === pixal.type)?.nodes.some(n => n.type === 'Pixal3DConditioning'));
  const rgb = prepared.outputs.findIndex(p => p.name === 'rgb');
  assert(rgb >= 0);
  for (const pin of pixal.inputs) disconnect(graph, pin);
  const pixalOutputs = new Set(pixal.outputs.flatMap(pin => pin.links ?? []));
  for (const pin of proxy.inputs) {
    if (pin.name === 'reference' || pixalOutputs.has(pin.link)) disconnect(graph, pin);
  }
  for (const link of graph.links.filter(([, source]) => source === pixalId)) {
    const [id, , , target, input] = link;
    assert(canvases.some(n => n.id === target && n.inputs[input].name === 'reference'), 'Unexpected Pixal output');
    link[1] = prepared.id;
    link[2] = rgb;
    prepared.outputs[rgb].links.push(id);
  }
  graph.nodes = graph.nodes.filter(n => n.id !== pixalId);
  graph.definitions.subgraphs = graph.definitions.subgraphs.filter(s => s.id !== pixal.type);
  proxy.widgets_values[0] = mesh;
}

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
  if (graph.nodes.some(n => n.type === 'BITripoSplatGuide') && !registry.RenderSplat) missingNodes.add('RenderSplat');
  const inputAvailable = comboHas(registry.LoadImage?.input?.required?.image, input);
  return { ready: !missingNodes.size && inputAvailable && [...models.values()].every(m => m.available),
    missingNodes: [...missingNodes].sort(), input, inputAvailable, models: [...models.values()] };
}

const materialChannels = {
  none: [], material: ['roughness', 'metalness', 'ao'], emissive: ['emissive'],
  'material+emissive': ['roughness', 'metalness', 'ao', 'emissive'],
};

export function checkedMaterialTextures(selection, manifest) {
  const requested = materialChannels[selection];
  assert.deepEqual(manifest.provenance.requestedMaterialMaps, requested, 'PBR bake request changed');
  // The baker omits emissive for a mesh with no emissive material.
  const channels = requested.filter(name => name !== 'emissive' || manifest.maps.includes('emissive'));
  assert.deepEqual(manifest.maps, channels, 'Unexpected baked PBR channels');
  const expected = {materialMap: channels.includes('roughness'), emissiveMap: channels.includes('emissive')};
  for (const [key, present] of Object.entries(expected))
    assert.equal(Boolean(manifest.textures[key]), present, 'Unexpected map selection for ' + key);
  return Object.keys(expected).filter(key => expected[key]);
}

export function configureMaterialMaps(graph, selection) {
  assert(Object.hasOwn(materialChannels, selection), 'Unknown material map selection');
  const bakers = graph.nodes.filter(n => ['BIMakeSixGuides', 'BIBakeBox'].includes(n.type));
  assert(bakers.length, 'Expected a geometry bake node');
  for (const n of bakers) {
    n.widgets_values[2] = selection;
    if (n.widgets_values_named) n.widgets_values_named.material_maps = selection;
  }
}

export function checkCompiled(prompt, model, report) {
  const nodes = Object.values(prompt);
  const byType = type => nodes.filter(n => n.class_type === type);
  assert.deepEqual(byType('LoadImage').map(n => n.inputs.image), [report.input], 'Frontend changed the reference image');
  for (const { name, loader } of report.models)
    assert(byType(loader).some(n => Object.values(n.inputs).includes(name)), 'Frontend lost ' + name);
  if (report.materialMaps !== undefined) {
    for (const n of [...byType('BIBakeBox'), ...byType('BIMakeSixGuides')])
      assert.equal(n.inputs.material_maps, report.materialMaps, 'Frontend changed the PBR bake selection');
  }
  if (model === 'triposplat') {
    const entry = type => {
      const found = Object.entries(prompt).filter(([,n]) => n.class_type === type);
      assert.equal(found.length, 1, 'Missing or duplicate ' + type);
      return found[0];
    };
    const [reference, prepared] = entry('BIPrepareReference');
    const [matte, removed] = entry('RemoveBackground');
    assert.equal(prepared.inputs.mask_mode, 'foreground');
    assert.deepEqual(prepared.inputs.mask, [matte, 0]);
    assert.deepEqual(removed.inputs.image, prepared.inputs.image);
    const [decode] = entry('VAEDecodeTripoSplat');
    const [cache, cached] = entry('BICachedTripoSplat');
    assert.deepEqual(cached.inputs, {reference: [reference, 2], splat: [decode, 0]});
    const [transform, aligned] = entry('TransformSplat');
    assert.deepEqual(aligned.inputs.splat, [cache, 0]);
    const [guide, rendered] = entry('BITripoSplatGuide');
    assert.deepEqual(rendered.inputs.splat, [transform, 0]);
    assert.deepEqual(rendered.inputs.guides, [entry('BIMakeSixGuides')[0], 6]);
    assert.deepEqual(entry('PreviewImage')[1].inputs.images, [guide, 0]);
    assert.deepEqual(entry('PreviewAny')[1].inputs.source, [guide, 2]);
    assert.equal(entry('BICanonicalizeProxy')[1].inputs.filename, report.mesh);
    assert.equal(rendered.inputs.face_id, report.face);
    assert.equal(rendered.inputs.coordinate_space, report.coordinateSpace);
    return;
  }
  for (const type of ['Pixal3DConditioning', 'MeshToFile3D'])
    assert.equal(byType(type).length, report.mesh ? 0 : 1, 'Wrong geometry route for ' + type);
  for (const type of ['BIPreviewBox', 'BISaveBox'])
    assert.equal(byType(type).length, 1, 'Missing or duplicate ' + type);
  const [proxy] = byType('BICanonicalizeProxy');
  assert(proxy, 'Missing canonical proxy');
  if (!report.mesh) {
    const nodeId = type => {
      const found = Object.keys(prompt).filter(id => prompt[id].class_type === type);
      assert.equal(found.length, 1, 'Missing or duplicate ' + type);
      return found[0];
    };
    assert.deepEqual(proxy.inputs.model_3d, [nodeId('MeshToFile3D'), 0]);
    assert.deepEqual(proxy.inputs.cropped, [nodeId('ImageCropToMask'), 0]);
    const camera = nodeId('MoGeGeometryToFOV');
    assert.deepEqual(proxy.inputs.horizontal_fov_degrees, [camera, 0]);
    assert.deepEqual(proxy.inputs.focal_pixels, [camera, 1]);
    if (report.compactProxy) {
      for (const [type, settings] of Object.entries(compactProxy)) {
        if (type === 'PrimitiveInt') continue; // also present in the appearance subgraphs
        for (const [name, value] of Object.entries(settings)) assert.equal(prompt[nodeId(type)].inputs[name], value);
      }
      const uv = prompt[nodeId('UnwrapMesh')].inputs.resolution;
      assert.deepEqual(prompt[nodeId('BakeTextureFromVoxel')].inputs.texture_size, uv);
      assert.equal(prompt[uv[0]]?.class_type, 'PrimitiveInt');
      assert.equal(prompt[uv[0]].inputs.value, compactProxy.PrimitiveInt.value);
    }
  }
  if (model !== 'pixal3d') {
    const type = model === 'qwen' ? 'TextEncodeQwenImage21' : 'MiniMaxH3ImageConditioning';
    assert.equal(byType(type).length, 6, 'Missing generated view branches');
    if (model === 'h3') {
      for (const encoder of byType(type)) {
        const orderSwitch = prompt[encoder.inputs['images.image_1'][0]];
        assert.equal(orderSwitch.class_type, 'ComfySwitchNode');
        assert.equal(prompt[orderSwitch.inputs.switch[0]].inputs.value, Boolean(report.h3ReferenceFirst));
      }
    }
    assert.deepEqual(byType('BICachedGeneratedFace').map(n => n.inputs.face_id).sort(), [...directions].sort());
    if (report.face) {
      assert.deepEqual(Object.keys(byType('BIFaceInputs')[0].inputs), [slugs[report.face]], 'Wrong generated face selection');
    }
    const canvases = byType('BIModelCanvas');
    assert.equal(canvases.length, 6);
    if (report.mesh) {
      assert.equal(proxy.inputs.filename, report.mesh);
      for (const pin of ['model_3d', 'cropped', 'horizontal_fov_degrees', 'focal_pixels', 'reference'])
        assert.equal(proxy.inputs[pin], undefined, 'Unexpected input on supplied mesh: ' + pin);
      const prepared = Object.keys(prompt).find(id => prompt[id].class_type === 'BIPrepareReference');
      for (const n of canvases) assert.deepEqual(n.inputs.reference, [prepared, 0]);
    }
    if (report.size !== undefined) {
      for (const n of canvases) assert.equal(n.inputs.size, report.size);
      for (const n of byType(type)) {
        for (const param of model === 'qwen' ? ['resolution'] : ['width', 'height']) assert.equal(n.inputs[param], report.size);
      }
    }
    const groups = samplerGroups(prompt);
    for (const n of byType('BICachedGeneratedFace')) {
      const { face_id, mask_mode, mask, image, seed } = n.inputs;
      assert.equal(mask_mode, 'foreground', 'Missing generated-image foreground coverage on ' + face_id);
      const matte = prompt[mask?.[0]];
      assert.equal(matte?.class_type, 'RemoveBackground');
      assert.deepEqual(matte.inputs.image, image);
      assert.equal(groups[face_id].length, 1);
      assert.deepEqual(prompt[groups[face_id][0]].inputs.seed, seed);
      const expected = report.faceSeed?.face === face_id ? report.faceSeed.seed : 101 + directions.indexOf(face_id);
      assert.equal(prompt[seed[0]]?.inputs?.value, expected, 'Unexpected face seed');
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
    if (node.class_type === 'BICachedTripoSplat') result.triposplat = samplers(node.inputs.splat);
    if (node.class_type === 'BICanonicalizeProxy' && node.inputs.model_3d) result.pixal3d = samplers(node.inputs.model_3d);
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

export function verifyOutput(model, manifest, qa, face, inputMeshSha256, faceSeed) {
  assert.equal(manifest.layout, 'box');
  assert.match(manifest.asset_id, sha256);
  const mesh = manifest.provenance?.meshGeneration;
  let referenceSha = manifest.reference?.sha256;
  if (inputMeshSha256 !== undefined) {
    assert.match(inputMeshSha256, sha256);
    assert.equal(manifest.provenance.originalGlbSha256, inputMeshSha256, 'Wrong input GLB digest');
    assert.equal(mesh, undefined, 'Unexpected generated mesh');
  } else {
    assert.match(referenceSha ?? '', sha256, 'Missing reference digest');
    assert.equal(mesh?.backend, 'native_mesh');
    assert(Object.keys(mesh.weights).some(k => k.endsWith('/pixal3d_int8_convrot.safetensors') &&
      sha256.test(mesh.weights[k])), 'Missing Pixal3D weight digest');
  }
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
    if (faceSeed?.face === dir) assert.equal(edit.provenance.recipeSeed, faceSeed.seed, 'Wrong generated face seed');
    referenceSha ??= edit.provenance?.referenceSha256;
    assert.match(referenceSha ?? '', sha256, 'Missing appearance reference digest');
    assert.equal(edit.provenance?.referenceSha256, referenceSha);
    if (inputMeshSha256) assert.equal(edit.reference, 'reference/' + referenceSha + '/source.png');
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

function inputMeshURL(origin, mesh) {
  const parts = mesh.split('/');
  const filename = parts.pop();
  const url = new URL('/view', origin);
  url.search = new URLSearchParams({type: 'input', subfolder: parts.join('/'), filename});
  return url;
}

async function inputMeshSha256(origin, mesh) {
  const result = await response(inputMeshURL(origin, mesh));
  const hash = createHash('sha256');
  for await (const chunk of result.body) hash.update(chunk);
  return hash.digest('hex');
}

async function inputMeshAvailable(origin, mesh) {
  const result = await fetch(inputMeshURL(origin, mesh), {headers: {Range: 'bytes=0-11'}, signal: AbortSignal.timeout(60_000)});
  if (result.status === 404) return false;
  if (!result.ok) throw new Error('Mesh lookup failed: HTTP ' + result.status);
  const head = Buffer.alloc(12);
  let n = 0;
  for await (const chunk of result.body) {
    const bytes = Buffer.from(chunk).subarray(0, 12-n);
    bytes.copy(head,n);
    n += bytes.length;
    if (n === 12) break;
  }
  return n === 12 && head.toString('ascii',0,4) === 'glTF' &&
    head.readUInt32LE(4) === 2 && head.readUInt32LE(8) >= 20;
}

export function verifySplatOutput(report, manifest, face, space) {
  for (const k of ['asset', 'splatSha256', 'rendererSha256', 'meshGuideSha256', 'stageKey'])
    assert.match(report[k] ?? '', sha256, 'Missing ' + k);
  assert.equal(report.faceId, face);
  assert.equal(report.coordinateSpace, space);
  assert.equal(manifest.asset_id, report.asset);
  assert.equal(report.resolution, manifest.resolution);
  assert(report.resolution >= 8 && Number.isInteger(report.resolution));
  assert.equal(manifest.faces[face].guideSha256, report.meshGuideSha256);
  assert.match(manifest.provenance?.originalGlbSha256 ?? '', sha256);
  const qa = report.qa;
  assert(Number.isInteger(qa.splatPixels) && qa.splatPixels >= 0 && qa.splatPixels <= report.resolution ** 2);
  assert(Number.isInteger(qa.meshPixels) && qa.meshPixels >= 0 && qa.meshPixels <= report.resolution ** 2);
  assert(Number.isFinite(qa.iou) && qa.iou >= 0 && qa.iou <= 1);
  assert(['measured','constant-depth','insufficient-overlap','mesh-depth-unavailable'].includes(qa.relativeDepth?.status));
  return { mode: 'splat-guide', face, coordinateSpace: space, splatSha256: report.splatSha256,
    meshSha256: manifest.provenance.originalGlbSha256, qa };
}

export async function saveSplatOutput(dir, origin, prompt, outputs, opts, trace, workflow, promptId) {
  const id = type => Object.keys(prompt).find(k => prompt[k].class_type === type);
  const raw = outputs[id('PreviewAny')]?.text?.[0];
  assert.equal(typeof raw, 'string', 'Missing splat guide QA from PreviewAny');
  const report = JSON.parse(raw);
  await jsonFile(join(dir, 'qa.json'), report);
  assert.match(report.asset ?? '', sha256);
  const url = new URL('/box-impostor/assets/' + report.asset + '/manifest.json', origin);
  const manifest = await (await response(url)).json();
  const verified = verifySplatOutput(report, manifest, opts.face ?? '+Z', opts['coordinate-space'] ?? 'canonical_box');
  await jsonFile(join(dir, 'manifest.json'), manifest);
  const image = outputs[id('PreviewImage')]?.images?.[0];
  assert(image?.filename && image.type === 'temp', 'Missing native guide preview');
  const previewURL = new URL('/view', origin);
  previewURL.search = new URLSearchParams({type: image.type, filename: image.filename, subfolder: image.subfolder ?? ''});
  const png = Buffer.from(await (await response(previewURL)).arrayBuffer());
  assert(png.length >= 57 && png.subarray(0,8).equals(Buffer.from('89504e470d0a1a0a','hex')) &&
    png.toString('ascii',12,16) === 'IHDR' && png.subarray(-12).equals(Buffer.from('0000000049454e44ae426082','hex')),
    'Invalid or truncated guide preview PNG');
  assert.equal(png.readUInt32BE(16), report.resolution);
  assert.equal(png.readUInt32BE(20), report.resolution);
  await writeFile(join(dir, 'splat-guide.png'), png);
  const proxy = Buffer.from(await (await response(assetURL(manifest.faces[report.faceId].guide, url))).arrayBuffer());
  assert.equal(createHash('sha256').update(proxy).digest('hex'), report.meshGuideSha256);
  await writeFile(join(dir, 'mesh-guide.png'), proxy);
  return { workflow, prompt_id: promptId, asset_id: report.asset, inference: inferenceReport(prompt, null, trace), ...verified };
}

export async function run(opts) {
  const name = workflows[opts.model];
  if (!Object.hasOwn(workflows, opts.model)) throw new Error('Choose --model ' + Object.keys(workflows).join(', ') + '.');
  if (opts.face && (!directions.includes(opts.face) || opts.model === 'pixal3d')) throw new Error('--face needs qwen, h3 or triposplat and one of +X, -X, +Y, -Y, +Z, -Z.');
  const isSplat = opts.model === 'triposplat';
  const isEdit = opts.model === 'qwen' || opts.model === 'h3';
  const h3ReferenceFirst = opts['h3-reference-first'];
  if (h3ReferenceFirst && opts.model !== 'h3') throw new Error('--h3-reference-first needs h3.');
  let faceSeed;
  if (opts['face-seed'] !== undefined) {
    const match = /^([+-][XYZ]):(\d+)$/.exec(opts['face-seed']);
    if (!isEdit || !match) throw new Error('--face-seed needs qwen or h3 and a value such as +Z:321.');
    faceSeed = { face: match[1], seed: Number(match[2]) };
  }
  const size = opts.size === undefined ? undefined : Number(opts.size);
  if (!isEdit && size !== undefined) throw new Error('--size applies to qwen or h3.');
  if (opts['compact-proxy'] && (isSplat || opts.mesh !== undefined))
    throw new Error('--compact-proxy requires generated Pixal3D geometry; omit --mesh.');
  if (opts.model === 'pixal3d' && opts.mesh !== undefined) throw new Error('--mesh applies to qwen, h3 or triposplat.');
  if ((isSplat || opts.mesh !== undefined) && (typeof opts.mesh !== 'string' || !opts.mesh.toLowerCase().endsWith('.glb') ||
      /[\\:\x00]/.test(opts.mesh) || opts.mesh.split('/').some(p => !p || p === '.' || p === '..')))
    throw new Error('--mesh must name a matching GLB relative to ComfyUI/input.');
  if (!isSplat && opts['coordinate-space']) throw new Error('--coordinate-space applies to triposplat.');
  const coordinateSpace = opts['coordinate-space'] ?? 'canonical_box';
  if (!['canonical_box','source_glb'].includes(coordinateSpace)) throw new Error('Choose --coordinate-space canonical_box or source_glb.');
  const origin = new URL(opts.origin);
  if (!['http:', 'https:'].includes(origin.protocol)) throw new Error('--origin must be HTTP(S).');
  const timeout = Number(opts['timeout-minutes']);
  if (!(Number.isFinite(timeout) && timeout > 0)) throw new Error('--timeout-minutes must be positive.');
  await mkdir(opts.output, { recursive: true });
  const dir = await mkdtemp(join(resolve(opts.output), opts.model + '-'));
  console.log('Evidence:', dir);
  const graph = JSON.parse(await readFile(new URL('../workflows/' + name + '.json', import.meta.url), 'utf8'));
  if (isEdit) configureEditGraph(graph, opts.model, { mesh: opts.mesh, size, face: opts.face, faceSeed, h3ReferenceFirst });
  if (opts['compact-proxy']) configureCompactProxy(graph);
  if (opts['material-maps'] !== undefined) configureMaterialMaps(graph, opts['material-maps']);
  const image = graph.nodes.find(n => n.type === 'LoadImage');
  image.widgets_values[0] = opts.input;
  if (image.widgets_values_named) image.widgets_values_named.image = opts.input;
  if (isSplat) {
    graph.nodes.find(n => n.type === 'BICanonicalizeProxy').widgets_values[0] = opts.mesh;
    graph.nodes.find(n => n.type === 'BITripoSplatGuide').widgets_values = [opts.face ?? '+Z', coordinateSpace];
  }
  await jsonFile(join(dir, 'workflow.json'), graph);
  const registry = await (await response(new URL('/object_info', origin))).json();
  const report = preflight(graph, registry, opts.input);
  if (opts['compact-proxy']) report.compactProxy = true;
  if (opts['material-maps'] !== undefined) report.materialMaps = opts['material-maps'];
  if (faceSeed) report.faceSeed = faceSeed;
  if (opts.model === 'h3') report.h3ReferenceFirst = Boolean(h3ReferenceFirst);
  if (size !== undefined) report.size = size;
  if (isSplat) Object.assign(report, { face: opts.face ?? '+Z', coordinateSpace });
  else if (opts.face) report.face = opts.face;
  if (opts.mesh) {
    Object.assign(report, { mesh: opts.mesh, meshAvailable: await inputMeshAvailable(origin, opts.mesh) });
    report.ready &&= report.meshAvailable;
  }
  await jsonFile(join(dir, 'preflight.json'), report);
  if (!report.ready) {
    const missing = [...report.missingNodes, ...report.models.filter(m => !m.available).map(m => (m.directory ?? m.loader) + '/' + m.name)];
    if (opts.mesh && !report.meshAvailable) missing.push('input/' + opts.mesh + ' (GLB v2)');
    if (!report.inputAvailable) missing.push('input/' + opts.input);
    throw new Error('Missing on this ComfyUI server:\n' + missing.join('\n'));
  }
  if (!opts.run) { console.log('Preflight passed.'); return; }

  const meshDigest = isEdit && opts.mesh ? await inputMeshSha256(origin, opts.mesh) : undefined;
  const prompt = await compile(origin.href, graph);
  checkCompiled(prompt, opts.model, report);
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
      if (isSplat) {
        const summary = await saveSplatOutput(dir, origin, prompt, history.outputs, opts,
          live.traces.get(receipt.prompt_id), name, receipt.prompt_id);
        await jsonFile(join(dir, 'result.json'), summary);
        console.log(JSON.stringify(summary, null, 2));
        return;
      }
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
      const materialSha256 = {};
      if (report.materialMaps !== undefined) {
        for (const key of checkedMaterialTextures(report.materialMaps, manifest)) {
          const descriptor = manifest.textures[key];
          const bytes = await (await response(assetURL(descriptor.path, manifestURL))).arrayBuffer();
          const digest = createHash('sha256').update(new Uint8Array(bytes)).digest('hex');
          assert.equal(digest, descriptor.sha256, key + ' digest mismatch');
          materialSha256[key] = digest;
        }
      }
      const summary = { workflow: name, ...(report.materialMaps !== undefined && {materialMaps: report.materialMaps, bakedChannels: manifest.maps, materialSha256}), prompt_id: receipt.prompt_id, asset_id: manifest.asset_id,
        viewer: viewer.href, inference: inferenceReport(prompt, opts.face, live.traces.get(receipt.prompt_id)),
        ...(opts.model === 'h3' && { imageOrder: h3ReferenceFirst ? ['reference', 'guide'] : ['guide', 'reference'] }),
        ...(meshDigest && { inputMesh: opts.mesh, inputMeshSha256: meshDigest }),
        ...(faceSeed && { faceSeed }), ...verifyOutput(opts.model, manifest, qa, opts.face, meshDigest, faceSeed) };
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
    face: { type: 'string' }, 'face-seed': { type: 'string' }, size: { type: 'string' }, mesh: { type: 'string' }, 'compact-proxy': { type: 'boolean' },
    'h3-reference-first': { type: 'boolean' }, 'material-maps': { type: 'string' }, 'coordinate-space': { type: 'string' }, run: { type: 'boolean', default: false }, 'timeout-minutes': { type: 'string', default: '180' }, help: { type: 'boolean' } } });
  if (values.help) {
    console.log('node scripts/run_model_workflow.mjs --model pixal3d|qwen|h3|triposplat [--mesh matching.glb | --compact-proxy] [--coordinate-space canonical_box|source_glb] [--origin URL] [--input reference.png] [--output DIR] [--face=+Z] [--face-seed=+Z:321] [--h3-reference-first] [--material-maps none|material|emissive|material+emissive] [--size 512] [--run] [--timeout-minutes 180]\n--compact-proxy reduces the Pixal3D cascade to 1024, remesh grid to 256, face target to 100k, UV/normal maps to 1024, and AO to 512 with 32 samples. Qwen/H3: --mesh skips Pixal3D, --size sets the square model canvas and output. --face-seed changes a single seed while keeping all six generated faces connected unless --face is supplied. H3 defaults to guide first. --h3-reference-first uses the reference as image_1; H3 internally scales reference images to at least 1024. Place inputs in ComfyUI/input. Default: check installed models and input, without queuing.');
  } else {
    run(values).catch(error => { console.error(error.message); process.exitCode = 1; });
  }
}
