import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const names=[
  '01_box_from_glb','02_box_from_reference_qwen21','03_box_from_reference_h3',
  '04_manual_six_faces','05_pixal3d_baseline'
];
const faces=['+X','-X','+Y','-Y','+Z','-Z'];
const graph = async name => JSON.parse(await readFile(new URL('../workflows/'+name+'.json',import.meta.url),'utf8'));
const nodesOf = (g,t) => g.nodes.filter(n=>n.type===t);
const subgraphs = g => g.definitions?.subgraphs ?? [];
const linkList = g => g.links.map(l=>Array.isArray(l)?l:[l.id,l.origin_id,l.origin_slot,l.target_id,l.target_slot,l.type]);
const upstream = (g,node,input) => {
  const p=node.inputs.findIndex(s=>s.name===input);
  const link=linkList(g).find(l=>l[3]===node.id && l[4]===p);
  assert.ok(link,'Missing link on '+node.type+'.'+input);
  return {from:link[1],slot:link[2]};
};

function validate(g){
  const byId=new Map(g.nodes.map(n=>[n.id,n]));
  assert.equal(byId.size,g.nodes.length,'duplicate node ID');
  const links=linkList(g), listed=new Set(), connections=new Map();
  for(const [id,from,src,to,dst,type] of links){
    assert(!listed.has(id),'duplicate link');listed.add(id);
    const out=from===-10?g.inputs[src]:byId.get(from)?.outputs[src];
    const inp=to===-20?g.outputs[dst]:byId.get(to)?.inputs[dst];
    assert.ok(out && inp, 'unknown port on link '+id);
    assert(out.type==='*' || out.type===type);
    assert(inp.type==='*' || inp.type.split(',').includes(type),'type mismatch '+id);
    assert((from===-10?out.linkIds:out.links).includes(id));
    if(to===-20) assert(inp.linkIds.includes(id)); else assert.equal(inp.link,id);
    if(from>=0 && to>=0)(connections.get(to)??connections.set(to,[]).get(to)).push(from);
  }
  for(const n of g.nodes){
    for(const s of n.inputs??[])if(s.link!=null)assert(listed.has(s.link),'dangling input');
    for(const s of n.outputs??[])for(const id of s.links??[])assert(listed.has(id),'dangling output');
  }
  const visited=new Set(),path=new Set();
  function visit(id){
    if(visited.has(id))return;
    assert(!path.has(id),'graph cycle');path.add(id);
    for(const dep of connections.get(id)??[])visit(dep);
    path.delete(id);visited.add(id);
  }
  for(const n of g.nodes)visit(n.id);
}

for(const name of names)test(name+' is a closed, typed native GUI workflow',async()=>{
  const g=await graph(name);
  validate(g);
  for(const sub of subgraphs(g)){
    validate(sub);
    const instances=nodesOf(g,sub.id);
    assert(instances.length);
    for(const instance of instances){
      for(const side of ['inputs','outputs']){
        const labels=ports=>ports.map(p=>[p.name,p.type]);
        assert.deepEqual(labels(instance[side]),labels(sub[side]));
      }
    }
  }
  for(const node of g.nodes)if(/^[a-f0-9-]{36}$/.test(node.type))assert(subgraphs(g).some(s=>s.id===node.type));
  assert.equal(nodesOf(g,'BIBakeBox').length,1);
  assert.equal(nodesOf(g,'BIPreviewBox').length,1);
  assert.equal(nodesOf(g,'BISaveBox').length,1);
  const pixal=subgraphs(g).find(s=>s.name.startsWith('Pixal3D'));
  if(pixal){
    assert.equal(nodesOf(g,'LoadImage').length,1,'one reference for geometry and appearance');
    const mesh=nodesOf(pixal,'MeshToFile3D');assert.equal(mesh.length,1);
    const smooth=upstream(pixal,mesh[0],'mesh').from;
    const textured=upstream(pixal,pixal.nodes.find(n=>n.id===smooth),'mesh').from;
    assert.equal(pixal.nodes.find(n=>n.id===textured).type,'ApplyTextureToMesh');
    assert(nodesOf(pixal,'Pixal3DConditioning').length);
    const can=nodesOf(g,'BICanonicalizeProxy')[0];
    assert.equal(upstream(g,can,'model_3d').from,nodesOf(g,pixal.id)[0].id);
    assert.deepEqual(can.widgets_values,['',0,0,0]);
  }
  if(name.startsWith('04')){
    assert.equal(nodesOf(g,'LoadImage').length,6);
    assert.equal(nodesOf(g,'BIFaceImage').length,6);
  }
  const isQ=name.startsWith('02'),isH=name.startsWith('03');
  if(!isQ&&!isH)return;
  const sg=subgraphs(g).find(z=>z.name.startsWith('GenerateView'));
  const gen=nodesOf(g,sg.id);
  assert.equal(gen.length,6);
  const shared=(port)=>new Set(gen.map(n=>JSON.stringify(upstream(g,n,port)))).size;
  assert.equal(shared('model'),1);assert.equal(shared('vae'),1);assert.equal(shared('clip'),1);
  const encoder=nodesOf(sg,isQ?'TextEncodeQwenImage21':'MiniMaxH3ImageConditioning')[0];
  for(const [slot,name] of [['images.image_1',isQ?'reference':'guide'],['images.image_2',isQ?'guide':'reference']]){
    const u=upstream(sg,encoder,slot);
    if(isQ){assert.equal(u.from,-10);assert.equal(sg.inputs[u.slot].name,name);}
    else{
      const switchNode=sg.nodes.find(n=>n.id===u.from);
      assert.equal(switchNode.type,'ComfySwitchNode');
      assert.equal(sg.inputs[upstream(sg,switchNode,'on_false').slot].name,name);
      assert.equal(sg.inputs[upstream(sg,switchNode,'on_true').slot].name,name==='guide'?'reference':'guide');
      assert.equal(sg.inputs[upstream(sg,switchNode,'switch').slot].name,'ref_first');
    }
  }
  if(isH){
    const selector=nodesOf(g,'PrimitiveBoolean')[0];
    assert.deepEqual(selector.widgets_values,[false]);
    assert(gen.every(n=>upstream(g,n,'ref_first').from===selector.id));
    const promptSwitch=sg.nodes.find(n=>n.id===upstream(sg,encoder,'prompt').from);
    assert.equal(sg.inputs[upstream(sg,promptSwitch,'switch').slot].name,'ref_first');
    const prompt=port=>sg.nodes.find(n=>n.id===upstream(sg,promptSwitch,port).from).widgets_values[0];
    assert.match(prompt('on_false'),/colors and markings from <Picture 2>/);
    assert.match(prompt('on_true'),/colors and markings from <Picture 1>/);
  }
  const sampler=nodesOf(sg,'KSampler')[0];
  assert.equal(sampler.widgets_values[3],isQ?3:1);
  assert.equal(sampler.widgets_values[2],isQ?24:6);
  assert.equal(sg.inputs[upstream(sg,sampler,'seed').slot].name,'seed');
  assert.equal(nodesOf(g,'LoraLoaderModelOnly').length,isQ?1:0);
  const collectors=nodesOf(g,'BICachedGeneratedFace');
  assert.equal(collectors.length,6);
  for(const [i,n] of collectors.entries()){
    assert.equal(n.widgets_values[0],faces[i]);
    const generator=gen.find(z=>z.id===upstream(g,n,'image').from);assert(generator);
    assert.equal(upstream(g,generator,'guide').slot,i);
    assert.deepEqual(upstream(g,generator,'seed'),upstream(g,n,'seed'));
    const mask=g.nodes.find(z=>z.id===upstream(g,n,'mask').from);
    assert.equal(mask.type,'RemoveBackground');
    assert.equal(upstream(g,mask,'image').from,generator.id);
  }
});
