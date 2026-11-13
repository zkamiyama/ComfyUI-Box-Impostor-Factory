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

function referencePreview(api, path, label, annotated) {
  // LoadImage accepts Comfy's folder annotations. Secondary paths are literal
  // file names relative to input, including spaces and URL metacharacters.
  const suffix = annotated && path.match(/\s*\[(input|output|temp)\]$/);
  const filePath = suffix ? path.slice(0, suffix.index) : path;
  const split = filePath.lastIndexOf('/');
  const filename = filePath.slice(split + 1);
  const query = new URLSearchParams({filename, type: suffix?.[1] || 'input'});
  if (split >= 0) query.set('subfolder', filePath.slice(0, split));

  const figure = element('figure');
  figure.dataset.reference = path;
  const anchor = element('a');
  anchor.href = api.apiURL('/view?' + query);
  anchor.target = '_blank';
  anchor.rel = 'noopener noreferrer';
  anchor.title = label + ' · ' + path;
  const img = element('img');
  img.alt = label + ' · ' + filename;
  img.onerror = () => {
    const message = element('span', '画像を表示できません');
    message.className = 'bi-reference-missing';
    anchor.replaceWith(message);
  };
  img.src = anchor.href;
  anchor.append(img);
  const caption = element('figcaption', label + ' · ' + filename);
  caption.title = path;
  figure.append(anchor, caption);
  return figure;
}

function addUpload(parent, {multiple, api, app, source, data, write}) {
  if (!parent) return;
  const picker = element('input');
  picker.type = 'file';
  picker.multiple = multiple;
  picker.accept = 'image/*';
  picker.setAttribute('aria-label', multiple ? 'Secondary 画像を追加' : 'Primary 画像をアップロード');
  // Keep the file input separate from the label's existing form control.
  const group = element('div');
  group.className = 'bi-reference';
  parent.replaceWith(group);
  group.append(parent, picker);
  const names = () => String(data.value ?? '').split(/\r?\n/).map(s => s.trim()).filter(Boolean);
  const paths = multiple ? names().slice(0, 8) : [String(data.value ?? '')].filter(Boolean);
  if (paths.length) {
    const gallery = element('div');
    gallery.className = 'bi-reference-previews';
    for (const [i, path] of paths.entries()) {
      gallery.append(referencePreview(api, path, multiple ? 'Secondary ' + (i + 1) : 'Primary', !multiple));
    }
    group.append(gallery);
  }
  picker.onchange = async () => {
    const files = [...picker.files];
    if (!files.length) return;
    const graph = source.graph;
    picker.setCustomValidity('');
    if (multiple && names().length + files.length > 8) {
      picker.setCustomValidity('Secondary 画像は最大8枚です。');
      picker.reportValidity();
      return;
    }
    picker.disabled = true;
    try {
      const paths = [];
      for (const file of files) {
        const body = new FormData();
        body.append('image', file);
        const response = await api.fetchApi('/upload/image', {method:'POST', body});
        if (!response.ok) throw Error(file.name + ' (HTTP ' + response.status + ')');
        const result = await response.json();
        paths.push([result.subfolder, result.name].filter(Boolean).join('/'));
      }
      if (source.graph !== graph || graph !== app.rootGraph) return;
      const value = multiple ? [...names(), ...paths].join('\n') : paths[0];
      if (!multiple && Array.isArray(data.options?.values) && !data.options.values.includes(value))
        data.options.values.push(value);
      write(source, data, value);
    } catch (error) {
      picker.disabled = false;
      picker.setCustomValidity('アップロードエラー · ' + error.message);
      picker.reportValidity();
    }
  };
}

export function installWorkflowPanel(app, api) {
  let state, root, snapshot, graphId, sidebar, interval;
  const views = new Map();
  const jobs = new Map();
  function current() {
    const graph = app.rootGraph;
    const serialized = graph?.serialize();
    const next = serialized && inspectWorkflow(serialized);
    if (root !== graph || graphId !== serialized?.id || !next || !snapshot ||
        JSON.stringify(snapshot.endIds) !== JSON.stringify(next.endIds)) {
      root = graph;
      graphId = serialized?.id;
      state = next ? new WorkflowStatus(next) : null;
      for (const view of views.values()) view.key = null;
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
    if (!views.size) return;
    current();
    const statuses = snapshot && Object.fromEntries(Object.keys(snapshot.signatures).map(k => [k, state.stage(k, snapshot)]));
    const key = JSON.stringify([snapshot?.signatures, statuses]);
    for (const [container, view] of views) {
      if (view.node && view.node.graph !== root) continue;
      if (!force && (view.key === key || container.contains(document.activeElement) &&
          ['INPUT','TEXTAREA','SELECT'].includes(document.activeElement?.tagName))) continue;
      view.key = key;
      render(container, statuses);
    }
  }

  function render(container, statuses) {
    const previous = container.firstElementChild;
    const openSettings = previous?.querySelector('details')?.open === true;
    const scroll = previous?.scrollTop ?? 0;
    const panel = element('div');
    panel.className = 'bi-workflow-panel';
    container.replaceChildren(panel);
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
      if (Array.isArray(values) && w.value != null && !values.includes(w.value)) values = [w.value, ...values];
      const tag = Array.isArray(values) ? 'select' : multiline ? 'textarea' : 'input';
      const line = element('label', label);
      line.style.cssText = 'display:grid;gap:3px;min-width:0';
      const input = element(tag);
      input.setAttribute('aria-label', label);
      input.style.cssText = 'max-width:100%;min-width:0;padding:4px';
      if (tag === 'select') for (const option of values) {
        const item = element('option', String(option)); item.value = String(option); input.append(item);
      }
      if (tag === 'input' && typeof w.value === 'number') {
        input.type = 'number';
        for (const k of ['min','max']) if (w.options?.[k] != null) input[k] = w.options[k];
        // LiteGraph's legacy step is ten times the actual increment.
        input.step = w.options?.step2 ?? (w.options?.step ? w.options.step / 10 : 'any');
      }
      input.value = w.value;
      input.onchange = () => {
        if (!input.checkValidity()) { input.reportValidity(); return; }
        const value = typeof w.value === 'number' ? Number(input.value) : input.value;
        if (update) update(value);
        write(node, w, value);
      };
      line.append(input); parent.append(line);
      return line;
    }
    const inputs = element('div');
    inputs.className = 'bi-controls';
    const references = element('div');
    references.className = 'bi-reference-inputs';
    panel.append(references, inputs);
    const image = find('LoadImage');
    const secondary = find('BISecondaryReferences');
    const primaryField = field(references, 'Primary 画像', image, 'image');
    const secondaryField = field(references, 'Secondary 画像（最大8枚・1行に1枚）', secondary, 'paths', {multiline:true});
    addUpload(primaryField, {multiple:false, api, app, source:image, data:widget(image,'image'), write});
    addUpload(secondaryField, {multiple:true, api, app, source:secondary, data:widget(secondary,'paths'), write});
    const family = widget(find('BICachedGeneratedFace'),'backend')?.value;
    const h3 = family === 'h3_enhanced';
    const accepted = h3 ? ['minimax_h3_image_pruned_int8_convrot.safetensors','minimax_h3_image_pruned_w4a8.safetensors']
      : ['qwen_image_2.1_int8_convrot.safetensors'];
    const denoiser = find('UNETLoader');
    field(inputs, h3 ? '画像生成モデル · H3' : '画像生成モデル · Qwen-Image 2.1', denoiser, 'unet_name', {
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
    const guides = find('BIMakeSixGuides'), bake = find('BIBakeBox');
    field(inputs, 'GLB からベイクするマップ', guides, 'material_maps', {
      update: value => setWidget(root, bake, widget(bake,'material_maps'), value)
    });
    field(inputs, '深度の作成方法', bake, 'depth_source');
    const materials = find('BIGenerateFaceMaterials');
    field(inputs, 'PBR マップの作成方法', materials, 'source');
    field(inputs, 'PBR を生成する面', materials, 'face');
    const settings = element('details');
    settings.open = openSettings;
    settings.append(element('summary', 'モデルパラメータ・解像度'));
    field(settings, 'Box とガイドの各面サイズ（px）', bake, 'resolution', {
      update: value => setWidget(root, guides, widget(guides, 'resolution'), value)
    });
    field(settings, 'CLIP', find('CLIPLoader'), 'clip_name');
    field(settings, 'VAE', find('VAELoader'), 'vae_name');
    field(settings, 'LoRA', find('LoraLoaderModelOnly'), 'lora_name');
    field(settings, 'マテリアルのシード', materials, 'material_seed');
    field(settings, 'PBR 推定サイズ（Qwen・H3）', materials, 'edit_resolution');
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
    const faces = element('div');
    faces.className = 'bi-faces';
    for (const stage of ['geometry', ...directions, 'box']) {
      const card = element('section');
      card.className = 'bi-stage';
      card.dataset.stage = stage;
      const title = element('strong', stageNames[stage] ?? stage);
      const status = statuses[stage];
      const badge = element('span', status.label);
      badge.className = 'bi-status';
      card.dataset.dirty = String(status.label.startsWith('Dirty'));
      card.append(title, badge);
      if (status.message) card.append(element('span', status.message));
      if (stage === 'geometry') {
        if (guideRoot) link(card, 'メッシュ・ベイク結果を見る', '/box-impostor/web/viewer.html?manifest=' + guideRoot + 'manifest.json&mode=proxy');
        retry(card, sampler, widget(sampler,'seed'), '3D のシードを変えて再生成対象にする');
      } else if (stage === 'box') {
        const path = status.output?.box_impostor?.[0];
        if (typeof path === 'string' && path.startsWith('/box-impostor/web/viewer.html?')) {
          const viewer = link(card, 'Box ビューアを開く', path);
          const image = status.output?.images?.[0];
          if (image?.type === 'output' && /^box_impostor\/[a-f0-9]{64}\/previews$/.test(image.subfolder)
              && ['hero.png','hero-reduce_bleeding.png'].includes(image.filename)) {
            const preview = element('img');
            preview.src = api.apiURL('/view?' + new URLSearchParams(image));
            preview.alt = 'Box Impostor プレビュー';
            viewer.prepend(preview);
          }
        }
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
          a.append(img);
          a.append(element('small', img.alt));
        }
        retry(card, seedNode, seed, stage + ' のみ再生成対象にする');
      }
      if (directions.includes(stage)) {
        if (!faces.parentElement) panel.append(faces);
        faces.append(card);
      } else panel.append(card);
    }
    panel.scrollTop = scroll;
  }

  function unmount(el) {
    views.delete(el);
    if (!views.size) {
      clearInterval(interval);
      interval = undefined;
    }
  }
  function mount(el, node) {
    views.set(el, {node, key: null});
    if (!interval) interval = setInterval(draw, 900);
    draw(true);
    return () => unmount(el);
  }


  app.extensionManager.registerSidebarTab({
    id: 'box-impostor-workflow', type: 'custom', icon: 'pi pi-box', title: 'Box Impostor', tooltip: 'Box Impostor ワークフロー',
    render(el) { sidebar?.(); sidebar = mount(el); },
    destroy() { sidebar?.(); sidebar = null; }
  });
  return {mount, draw};
}
