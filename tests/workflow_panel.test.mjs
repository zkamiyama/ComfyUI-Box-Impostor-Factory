import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import test from 'node:test';
import {directions, inspectWorkflow, WorkflowStatus} from '../extension/workflow_state.mjs';
import {installWorkflowPanel} from '../extension/workflow_panel.js';

async function load(name = '02_box_from_reference_qwen21') {
  return JSON.parse(await readFile(new URL('../workflows/' + name + '.json', import.meta.url)));
}
function changed(before, after) {
  return Object.keys(before.signatures).filter(k => before.signatures[k] !== after.signatures[k]);
}

for (const name of ['02_box_from_reference_qwen21','03_box_from_reference_h3']) {
  test(name + ' follows actual links for primary, secondary, mesh, per-face, and material edits', async () => {
    const graph = await load(name);
    const baseline = inspectWorkflow(graph);
    const unchanged = structuredClone(graph);
    unchanged.nodes[0].pos = [1,2];
    assert.deepEqual(changed(baseline,inspectWorkflow(unchanged)), []);
    function check(edit, expected) {
      const copy = structuredClone(graph);
      edit(copy);
      assert.deepEqual(new Set(changed(baseline,inspectWorkflow(copy))), new Set(expected));
    }
    const all = ['geometry', ...directions, 'box'];
    check(g => g.nodes.find(n => n.type === 'LoadImage').widgets_values[0] = 'other.png', all);
    check(g => g.nodes.find(n => n.type === 'BISecondaryReferences').widgets_values[0] = 'other.png', [...directions, 'box']);
    check(g => g.definitions.subgraphs[0].nodes.find(n => n.type === 'KSampler').widgets_values[0]++, all);
    check(g => g.nodes.find(n => n.title === '-Y seed').widgets_values[0]++, ['-Y','box']);
    check(g => g.nodes.find(n => n.type === 'BIGenerateFaceMaterials').widgets_values[0] = 'qwen', ['box']);
  });
}

test('presentation-only widget sockets do not dirty successfully completed stages', async () => {
  const graph = await load();
  const submitted = inspectWorkflow(graph);
  for (const node of graph.nodes) {
    node.inputs ??= [];
    node.inputs.push({name:'widget-socket', link:null});
  }
  assert.deepEqual(inspectWorkflow(graph).signatures, submitted.signatures);
});

test('partial runs, late outputs, cached nodes, edits in flight, failures and retries', async () => {
  const graph = await load();
  const a = inspectWorkflow(graph);
  const state = new WorkflowStatus(a);
  const id = Object.keys(a.endIds).find(id => a.endIds[id] === '+X');
  assert.equal(state.stage('+X',a).label, '未実行');
  state.queue('first', a);
  for (const node of Object.keys(a.endIds)) state.observe('first',node);
  state.observe('first',id,{box_face:['first']});
  state.end('first');
  graph.nodes.find(n => n.title === '+X seed').widgets_values[0]++;
  const b = inspectWorkflow(graph);
  state.queue('retry',b);
  assert.match(state.stage('+X', b).label, /Dirty/);
  assert.equal(state.stage('-X', b).label, '完了');
  state.end('retry',{node_id: id, exception_message:'image failed'});
  assert.equal(state.stage('+X', b).message,'image failed');
  assert.match(state.stage('box', b).label, /Dirty/);
  state.queue('late', a);
  state.queue('partial', b);
  state.observe('partial',id,{box_face:['partial']});
  state.observe('late',id,{box_face:['old']});
  state.end('partial'); state.end('late');
  assert.equal(state.stage('+X', b).label,'完了');
  assert.deepEqual(state.stage('+X', b).output.box_face,['partial']);
  assert.match(state.stage('box', b).label, /Dirty/);
  graph.nodes.find(n => n.title === '+X seed').widgets_values[0]++;
  assert.match(state.stage('+X',inspectWorkflow(graph)).label,/Dirty/);
});

test('panel forwards original queue request only on explicit invocation', async () => {
  const graph = await load();
  const requests = [], events = new Map();
  const api = {
    async queuePrompt(...args) { requests.push(args); return {prompt_id:'1'}; },
    async fetchApi() {return {ok:true,json: async () => ({})};},
    addEventListener(type,handler) {events.set(type,handler);}
  };
  let tab;
  const app = {rootGraph:{serialize:()=>graph},extensionManager:{registerSidebarTab(value){tab=value;}}};
  installWorkflowPanel(app,api);
  assert.equal(tab.id,'box-impostor-workflow');
  assert.equal(requests.length,0);
  const prompt = {workflow:graph,output:{}}, options = {partialExecutionTargets:['23']};
  assert.deepEqual(await api.queuePrompt(1,prompt,options),{prompt_id:'1'});
  assert.deepEqual(requests,[[1,prompt,options]]);
  events.get('execution_cached')({detail:{prompt_id:'1',nodes:['23']}});
  events.get('execution_error')({detail:{prompt_id:'1',node_id:'23'}});
  assert.equal(requests.length,1);
});
