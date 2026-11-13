import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import test from 'node:test';
import { configureFaceMaterials, preflightFaceMaterials } from '../scripts/run_model_workflow.mjs';

test('material size changes the selected texture editor independently of face generation', async () => {
  const graph = JSON.parse(await readFile(new URL('../workflows/02_box_from_reference_qwen21.json', import.meta.url)));
  configureFaceMaterials(graph, 'h3', '+Z', 512);
  const node = graph.nodes.find(n => n.type === 'BIGenerateFaceMaterials');
  assert.deepEqual(node.widgets_values, ['h3', '+Z', 0, 512]);
  assert.throws(() => configureFaceMaterials(graph, 'marigold', '+Z', 512), /material-size requires/);
  assert.throws(() => configureFaceMaterials(graph, 'qwen', '+Z', 63), /multiple of 32/);
});

test('preflight includes missing alternate material providers and skips keep', async t => {
  const calls = [];
  const server = createServer((req, res) => {
    calls.push(req.url);
    const source = req.url.split('/').at(-1);
    if (source === 'h3') {
      res.writeHead(404).end();
    } else {
      const missing = source === 'qwen' ? ['diffusion_models/absent.safetensors'] : [];
      res.end(JSON.stringify({source, missing, available: !missing.length}));
    }
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  t.after(() => new Promise(resolve => server.close(resolve)));
  const origin = new URL('http://127.0.0.1:' + server.address().port);
  for (const [source, ready] of [['keep', true], ['qwen', false], ['marigold', true], ['h3', false]]) {
    const report = {ready: true};
    await preflightFaceMaterials(origin, source, report);
    assert.equal(report.ready, ready);
    if (source === 'qwen') assert.deepEqual(report.materialProvider.missing, ['diffusion_models/absent.safetensors']);
    if (source === 'h3') assert.match(report.materialProvider.missing[0], /Update and restart/);
  }
  assert.deepEqual(calls, ['qwen', 'marigold', 'h3'].map(source => '/box-impostor/material-providers/' + source));
});
