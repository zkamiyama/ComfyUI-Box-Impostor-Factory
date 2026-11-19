// Rebuild both official Inspector native profiles from the pinned 0.4.1 entry.
// The source adapter changes only the required 0.4.2 distance-map restore path
// and retains the existing opt-in Box color display behavior.
import { build } from 'esbuild';
import { createHash } from 'node:crypto';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const worker = dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
function option(name) {
  const index = args.indexOf(name);
  if (index < 0 || !args[index + 1]) throw new Error(`Required option: ${name} PATH`);
  return resolve(args[index + 1]);
}
const inspector = option('--inspector-root');
const evidence = option('--evidence');
const assets = resolve(inspector, 'skills/3d-inspector/assets');
const sha = data => createHash('sha256').update(data).digest('hex');
const entryName = 'inspector_native_entry.mjs';
const entrySHA256 = 'ac0f29cd15b3bf9f3b6a95c216a34e39637474ca8c8b5613c51b94d2e10b7ff9';
const upstreamSHA256 = '18d82ffce940ec996c41e8c61018d38f5a0503cbabe06e69cd2f2c5bf7abcc56';
let source = await readFile(resolve(worker, entryName), 'utf8');
if (sha(source) !== entrySHA256) throw new Error('Pinned official Inspector native entry changed; review the adapter.');
const dependencies = {};
for (const [name, version] of [['three', '0.186.0'], ['three-impostor', '0.4.2'], ['esbuild', '0.28.2']]) {
  const json = JSON.parse(await readFile(resolve(worker, '../node_modules', name, 'package.json'), 'utf8'));
  if (json.version !== version) throw new Error(`Unexpected ${name} version: ${json.version}`);
  dependencies[name] = version;
}
if (sha(await readFile(resolve(worker, '../node_modules/three-impostor/Impostor.js'))) !== upstreamSHA256)
  throw new Error('Published three-impostor 0.4.2 source changed.');

function replaceOnce(text, before, after) {
  if (text.split(before).length !== 2) throw new Error(`Expected one pinned Inspector adapter location: ${before}`);
  return text.replace(before, after);
}
source = replaceOnce(source, 'n=[],t;try{for', 'n=[],t,rangePixels;try{for');
source = replaceOnce(source,
  'if(!y)throw new Error("Unsupported native Box texture dtype "+c.dtype);',
  'if(!y)throw new Error("Unsupported native Box texture dtype "+c.dtype);if(d==="rangeMap")rangePixels=y;');
source = replaceOnce(source, 't=new ee(m),t.steps=Math.max(t.steps,2*r.resolution);',
  'm.distanceMap=createDistanceTexture(rangePixels,r.resolution),n.push(m.distanceMap),t=new ee(m),t.steps=Math.max(t.steps,2*r.resolution);');
source = 'import { createDistanceTexture } from "./distance_texture.mjs";\n' + source;

const outputs = {};
for (const rgb of [false, true]) {
  let entry = source;
  if (rgb) {
    entry = replaceOnce(entry, 't=new ee(m),t.steps=Math.max(t.steps,2*r.resolution);',
      't=createBoxDisplayMaterial(m,p.display),p.display?.lighting==="unlit"&&(g.toneMapping=s.NoToneMapping),t.steps=Math.max(t.steps,2*r.resolution);');
    entry = 'import { createBoxDisplayMaterial } from "./box_display_material.mjs";\n' + entry;
  }
  for (const cdn of [false, true]) {
    const name = `${rgb ? 'native-rgb' : 'native'}-runtime.${cdn ? 'cdn.mjs' : 'bundle.js'}`;
    const outfile = resolve(assets, name);
    await build({
      stdin: { contents: entry, sourcefile: `inspector-${rgb ? 'rgb' : 'native'}-entry.mjs`, resolveDir: worker },
      outfile, bundle: true, packages: cdn ? 'external' : 'bundle',
      format: cdn ? 'esm' : 'iife', platform: 'browser', target: ['es2022'],
      minify: true, legalComments: 'none',
    });
    const bytes = await readFile(outfile);
    outputs[name] = { bytes: bytes.length, sha256: sha(bytes) };
  }
}
const helpers = {};
for (const name of ['distance_map.mjs', 'distance_texture.mjs', 'box_display_material.mjs'])
  helpers[name] = sha(await readFile(resolve(worker, name)));
const receipt = {
  method: 'official-inspector-three-impostor-0.4.2-runtime',
  baselineSource: { path: entryName, sha256: entrySHA256,
    origin: '3D-Inspector skills/3d-inspector/assets/native-runtime.cdn.mjs before the 0.4.2 upgrade' },
  dependencies, upstreamSHA256, helpers,
  builderSHA256: sha(await readFile(fileURLToPath(import.meta.url))), outputs,
  colorSamplingOptions: ['native', 'view', 'dominant', 'view-fade', 'view-soft', 'view-feather'],
  distanceMap: { derivation: 'Original decoded RGBA16F rangeMap; increasingShaderV; before WebGL2 row flip',
    storage: 'Runtime-only R8 DataTexture, nearest, NoColorSpace, disposed with the Box',
    legacyAssetBytesChanged: false },
  generatedHtmlPostprocessing: false,
  changes: ['Both native and native-rgb profiles use pinned three-impostor 0.4.2',
    'Legacy Box assets derive the required distance field once at restore time',
    'Existing native depth step budget and opt-in RGB display options are preserved'],
};
await mkdir(dirname(evidence), { recursive: true });
await writeFile(evidence, JSON.stringify(receipt, null, 2) + '\n');
process.stdout.write(JSON.stringify(receipt, null, 2) + '\n');
