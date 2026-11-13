import { directions, inspectWorkflow, WorkflowStatus } from './workflow_state.mjs';

const hexId = /^[a-f0-9]{64}$/;
const stageNames = {geometry: '3D メッシュ・ガイド', box: 'Box Impostor'};
const liveNodes = graph => graph?.nodes ?? graph?._nodes ?? [];
const widget = (node, name) => node?.widgets?.find(w => w.name === name);

function setWidget(graph, node, w, value) {
  if (!w || w.value === value) return;
  const old = w.value;
  node.graph?.beforeChange?.(node);
  w.value = value;
  w.callback?.(value, undefined, node);
  node.onWidgetChanged?.(w.name, value, old, w);
  node.graph?.afterChange?.(node);
  graph.setDirtyCanvas?.(true, true);
}

function element(name, text) {
  const el = document.createElement(name);
  if (text !== undefined) el.textContent = text;
  return el;
}

export function installWorkflowPanel(app, api) {
  let state, root, snapshot, container, interval, lastDraw = '';
  const jobs = new Map();
  function current() {
    const graph = app.rootGraph;
    const next = graph && inspectWorkflow(graph.serialize());
    if (root !== graph || !next || !snapshot ||
        JSON.stringify(snapshot.endIds) !== JSON.stringify(next.endIds)) {
      root = graph;
      state = next ? new WorkflowStatus(next) : null;
      lastDraw = '';
    }
    snapshot = next;
    return state;
  }

  const queue = api.queuePrompt;
  api.queuePrompt = async function(number, prompt, ...rest) {
    const submitted = prompt?.workflow?.nodes && inspectWorkflow(prompt.workflow);
    const active = submitted && current();
    const response = await queue.call(this, number, prompt, ...rest);
    if (active && response?.prompt_id) {
      jobs.set(response.prompt_id, active);
      active.queue(response.prompt_id, submitted);
      // A cached run can finish before the HTTP response. Its own history also
      // recovers preview payloads that were computed in another browser session.
      reconcile(response.prompt_id);
    }
    return response;
  };

  async function reconcile(id, terminal) {
    const job = jobs.get(id);
    if (!job) return;
    try {
      const response = await api.fetchApi('/history/' + encodeURIComponent(id));
      if (!response.ok) return;
      const entry = (await response.json())[id];
      if (!entry) return;
      for (const [node, output] of Object.entries(entry.outputs ?? {})) job.observe(id, node, output);
      for (const [event, data] of entry.status?.messages ?? []) {
        if (event === 'execution_cached') for (const node of data.nodes ?? []) job.observe(id, node);
        if (event === 'execution_error' || event === 'execution_interrupted') terminal = data;
      }
      if (entry.status?.completed || entry.status?.status_str === 'error') terminal ??= true;
    } catch (error) {
      console.debug('Box workflow history unavailable', error);
    } finally {
      if (terminal) {
        job.end(id, terminal === true ? null : terminal);
        jobs.delete(id);
      }
      draw();
    }
  }

  api.addEventListener('executed', ({detail}) => {
    jobs.get(detail.prompt_id)?.observe(detail.prompt_id, detail.node, detail.output);
    draw();
  });
  api.addEventListener('execution_cached', ({detail}) => {
    const job = jobs.get(detail.prompt_id);
    for (const node of detail.nodes ?? []) job?.observe(detail.prompt_id, node);
    draw();
  });
  for (const type of ['execution_success','execution_error','execution_interrupted']) {
    api.addEventListener(type, ({detail}) => reconcile(detail.prompt_id, type === 'execution_success' || detail));
  }

  function draw(force = false) {
    if (!container) return;
    current();
    const statuses = snapshot && Object.fromEntries(Object.keys(snapshot.signatures).map(k => [k, state.stage(k, snapshot)]));
    const key = JSON.stringify([snapshot?.signatures, statuses]);
    if (!force && (lastDraw === key || container.contains(document.activeElement) &&
        ['INPUT','TEXTAREA','SELECT'].includes(document.activeElement?.tagName))) return;
    lastDraw = key;
    container.replaceChildren();
    const panel = element('div');
    panel.className = 'bi-workflow-panel';
    panel.style.cssText = 'padding:12px;display:grid;gap:8px;overflow:auto;max-height:100%;box-sizing:border-box';
    container.append(panel);
    if (!snapshot) {
      panel.append(element('p', 'Qwen または H3 の6面生成テンプレートを開いてください。'));
      return;
    }
    const nodes = liveNodes(root);
    const find = type => nodes.find(n => n.type === type || n.comfyClass === type);
    const all = type => nodes.filter(n => n.type === type || n.comfyClass === type);
    const sub = node => node?.subgraph ?? root.subgraphs?.get?.(node?.type);
    function write(node, w, value) {
      setWidget(root, node, w, value);
      draw(true);
    }
    function field(parent, label, node, name, {multiline = false, accept, update} = {}) {
      const w = widget(node, name);
      if (!w) return;
      let values = typeof w.options?.values === 'function' ? w.options.values() : w.options?.values;
      if (accept) values = [...new Set([w.value, ...(Array.isArray(values) ? values : [])])].filter(accept);
      const tag = Array.isArray(values) ? 'select' : multiline ? 'textarea' : 'input';
      const line = element('label', label);
      line.style.cssText = 'display:grid;gap:3px;min-width:0';
      const input = element(tag);
      input.style.cssText = 'max-width:100%;min-width:0;padding:4px';
      if (tag === 'select') for (const option of values) {
        const item = element('option', String(option)); item.value = String(option); input.append(item);
      }
      if (tag === 'input' && typeof w.value === 'number') {
        input.type = 'number';
        for (const k of ['min','max','step']) if (w.options?.[k] != null) input[k] = w.options[k];
      }
      input.value = w.value;
      input.onchange = () => {
        if (!input.checkValidity()) { input.reportValidity(); return; }
        const value = typeof w.value === 'number' ? Number(input.value) : input.value;
        if (update) update(value);
        write(node, w, value);
      };
      line.append(input); parent.append(line);
    }
    const image = find('LoadImage');
    field(panel, 'Primary 画像', image, 'image');
    field(panel, 'Secondary 画像（入力フォルダー内のファイル名を1行ずつ）', find('BISecondaryReferences'), 'paths', {multiline:true});
    const family = widget(find('BICachedGeneratedFace'),'backend')?.value;
    const h3 = family === 'h3_enhanced';
    const accepted = h3 ? ['minimax_h3_image_pruned_int8_convrot.safetensors','minimax_h3_image_pruned_w4a8.safetensors']
      : ['qwen_image_2.1_int8_convrot.safetensors'];
    const denoiser = find('UNETLoader');
    field(panel, h3 ? '画像生成モデル · H3' : '画像生成モデル · Qwen-Image 2.1', denoiser, 'unet_name', {
      accept: name => accepted.includes(String(name).split(/[\\/]/).pop()),
      update: value => {
        for (const node of all('BICachedGeneratedFace')) {
          const recipe = widget(node, 'recipe');
          if (recipe) {
            const metadata = JSON.parse(recipe.value || '{}');
            setWidget(root, node, recipe, JSON.stringify({...metadata, model:value}));
          }
        }
      }
    });
    panel.append(element('small', h3 ? 'Qwen-Image 2.1 はテンプレート 02 で使用します。' : 'H3 はテンプレート 03 で使用します。'));
    const guides = find('BIMakeSixGuides'), bake = find('BIBakeBox');
    field(panel, 'GLB からベイクするマップ', guides, 'material_maps', {
      update: value => setWidget(root, bake, widget(bake,'material_maps'), value)
    });
    field(panel, '深度の作成方法', bake, 'depth_source');
    const materials = find('BIGenerateFaceMaterials');
    field(panel, 'PBR マップの作成方法', materials, 'source');
    field(panel, 'PBR を生成する面', materials, 'face');
    const settings = element('details');
    settings.append(element('summary', 'モデルパラメータ'));
    field(settings, 'CLIP', find('CLIPLoader'), 'clip_name');
    field(settings, 'VAE', find('VAELoader'), 'vae_name');
    field(settings, 'LoRA', find('LoraLoaderModelOnly'), 'lora_name');
    field(settings, 'マテリアルのシード', materials, 'material_seed');
    if (['qwen','h3'].includes(widget(materials,'source')?.value))
      field(settings, 'PBR 編集サイズ', materials, 'edit_resolution');
    const keeper = find('BICachedGeneratedFace');
    const generator = keeper?.getInputNode?.(keeper?.inputs?.findIndex(p => p.name === 'image'));
    const modelNodes = liveNodes(sub(generator));
    const diffusion = modelNodes.find(n => n.type === 'KSampler');
    field(settings, '画像生成ステップ', diffusion, 'steps');
    field(settings, 'CFG', diffusion, 'cfg');
    const encoder = modelNodes.find(n => ['TextEncodeQwenImage21','MiniMaxH3ImageConditioning'].includes(n.type));
    const canvases = all('BIModelCanvas');
    if (encoder && canvases.length === 6) {
      const size = element('label', '6面の生成サイズ');
      const select = element('select');
      const old = widget(canvases[0],'size')?.value;
      for (const n of [...new Set([512,1024,2048,old].filter(Boolean))].sort((a,b) => a-b)) {
        const option = element('option', n + ' × ' + n); option.value = n; select.append(option);
      }
      select.value = old;
      select.onchange = () => {
        const value = Number(select.value);
        for (const canvas of canvases) setWidget(root, canvas, widget(canvas, 'size'), value);
        for (const key of ['width','height','resolution']) setWidget(root, encoder, widget(encoder,key), value);
        draw(true);
      };
      size.append(select); settings.append(size);
    }
    panel.append(settings);

    function action(parent, title, callback) {
      const button = element('button', title);
      button.type = 'button'; button.onclick = callback; parent.append(button);
    }
    function link(parent, text, path) {
      const a = element('a', text);
      a.href = api.apiURL(path); a.target = '_blank'; a.rel = 'noopener noreferrer'; parent.append(a);
      return a;
    }
    const guideAsset = statuses.geometry.output?.box_guides?.[0];
    const guideRoot = hexId.test(guideAsset) ? '/box-impostor/assets/' + guideAsset + '/' : null;
    const geometry = nodes.find(n => liveNodes(sub(n)).some(child => child.type === 'Pixal3DConditioning'));
    const sampler = liveNodes(sub(geometry)).find(n => n.type === 'KSampler' && widget(n, 'seed'));
    function retry(parent, node, w, label) {
      if (w) action(parent, label, () => {
        const next = (Number(w.value) + 1) % 2147483647;
        write(node, w, next);
      });
    }
    for (const stage of ['geometry', ...directions, 'box']) {
      const card = element('section');
      card.style.cssText = 'border-top:1px solid var(--border-color,#7776);padding-top:8px;display:grid;gap:5px';
      const title = element('strong', stageNames[stage] ?? stage);
      const status = statuses[stage];
      card.append(title, element('span', status.label));
      if (status.message) card.append(element('span', status.message));
      if (stage === 'geometry') {
        if (guideRoot) link(card, 'メッシュ・ベイク結果を見る', '/box-impostor/web/viewer.html?manifest=' + guideRoot + 'manifest.json&mode=proxy');
        retry(card, sampler, widget(sampler,'seed'), '3D のシードを変えて再生成対象にする');
      } else if (stage === 'box') {
        const path = status.output?.box_impostor?.[0];
        if (typeof path === 'string' && path.startsWith('/box-impostor/web/viewer.html?')) link(card, 'Box ビューアを開く', path);
      } else {
        const record = all('BICachedGeneratedFace').find(n => widget(n,'face_id')?.value === stage);
        const seedNode = record?.getInputNode?.(record?.inputs?.findIndex(p => p.name === 'seed'));
        const seed = widget(seedNode,'value');
        const digest = status.output?.box_face?.[0];
        const guide = statuses.geometry.output?.box_guide_images?.[0]?.[stage];
        const path = hexId.test(digest) ? '/box-impostor/face-previews/' + digest + '.png'
          : guideRoot && typeof guide === 'string' && guide.startsWith(guideRoot) && guide;
        if (path) {
          const a = link(card, '', path);
          const img = element('img'); img.src = a.href; img.alt = stage + (hexId.test(digest) ? ' 生成画像' : ' ガイド');
          img.style.cssText = 'width:100%;max-height:100px;object-fit:contain'; a.append(img);
          a.append(element('small', img.alt));
        }
        retry(card, seedNode, seed, stage + ' のみ再生成対象にする');
      }
      panel.append(card);
    }
  }

  app.extensionManager.registerSidebarTab({
    id: 'box-impostor-workflow', type: 'custom', icon: 'pi pi-box', title: 'Box Impostor', tooltip: 'Box Impostor ワークフロー',
    render(el) { container = el; lastDraw = ''; clearInterval(interval); draw(true); interval = setInterval(draw, 900); },
    destroy() { container = null; clearInterval(interval); }
  });
}
