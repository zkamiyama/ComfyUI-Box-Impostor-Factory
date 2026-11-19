// Rebuild an opt-in RGB display runtime from the pinned official Inspector source.
// The existing lit runtimes and every generated HTML remain unchanged.
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
const expected = {
  'native-runtime.cdn.mjs': 'ac0f29cd15b3bf9f3b6a95c216a34e39637474ca8c8b5613c51b94d2e10b7ff9',
  'native-runtime.bundle.js': '4e24d6ee46962c3bb8a6dd5236467f40b13d3519fb3e7f4224c36ca4f693c7aa',
};
for (const [name, hash] of Object.entries(expected)) {
  if (sha(await readFile(resolve(assets, name))) !== hash) {
    throw new Error(`Official baseline changed: ${name}; review the adapter before rebuilding.`);
  }
}
const dependencies = {};
for (const [name, version] of [['three', '0.186.0'], ['three-impostor', '0.4.1']]) {
  const json = JSON.parse(await readFile(resolve(worker, '../node_modules', name, 'package.json'), 'utf8'));
  if (json.version !== version) throw new Error(`Unexpected ${name} version: ${json.version}`);
  dependencies[name] = version;
}
let source = await readFile(resolve(assets, 'native-runtime.cdn.mjs'), 'utf8');
const before = 't=new ee(m),t.steps=Math.max(t.steps,2*r.resolution);';
const after = 't=createBoxDisplayMaterial(m,p.display),p.display?.lighting==="unlit"&&(g.toneMapping=s.NoToneMapping),t.steps=Math.max(t.steps,2*r.resolution);';
if (source.split(before).length !== 2) throw new Error('Expected exactly one native material construction.');
source = 'import { createBoxDisplayMaterial } from "./box_display_material.mjs";\n' + source.replace(before, after);
const outputs = {};
for (const cdn of [false, true]) {
  const name = `native-rgb-runtime.${cdn ? 'cdn.mjs' : 'bundle.js'}`;
  const outfile = resolve(assets, name);
  await build({
    stdin: {contents: source, sourcefile: 'inspector-rgb-entry.mjs', resolveDir: worker},
    outfile, bundle: true, packages: cdn ? 'external' : 'bundle',
    format: cdn ? 'esm' : 'iife', platform: 'browser', target: ['es2022'],
    minify: true, legalComments: 'none',
  });
  const bytes = await readFile(outfile);
  outputs[name] = {bytes: bytes.length, sha256: sha(bytes)};
}
for (const [name, hash] of Object.entries(expected)) {
  if (sha(await readFile(resolve(assets, name))) !== hash) throw new Error('Lit baseline runtime was modified.');
}
const receipt = {
  method: 'official-inspector-opt-in-rgb-runtime-v1',
  baselineSource: expected, dependencies,
  helperSha256: sha(await readFile(resolve(worker, 'box_display_material.mjs'))),
  builderSha256: sha(await readFile(fileURLToPath(import.meta.url))),
  outputs,
  baselineRuntimesUnchanged: true,
  generatedHtmlPostprocessing: false,
  changes: ['native material uses shared display helper', 'unlit renderer disables tone mapping'],
};
await mkdir(dirname(evidence), {recursive: true});
await writeFile(evidence, JSON.stringify(receipt, null, 2) + '\n');
process.stdout.write(JSON.stringify(receipt, null, 2) + '\n');
