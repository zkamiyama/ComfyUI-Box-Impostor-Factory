import assert from 'node:assert/strict';

export async function checkWorkflowControls(page, name, initialPrompt) {
  const compile = () => page.evaluate(() => window.app.graphToPrompt().then(p => p.output));
  const inputs = (prompt, type) => Object.values(prompt).filter(n => n.class_type === type).map(n => n.inputs);
  const changed = (before, after) => Object.keys(before)
    .filter(id => JSON.stringify(before[id].inputs) !== JSON.stringify(after[id].inputs))
    .map(id => after[id].class_type);
  const host = page.locator('.bi-workflow-host:visible');
  await host.waitFor();
  const rect = await host.boundingBox();
  const viewport = page.viewportSize();
  assert(rect.width > 400 && rect.height > 300 && rect.x >= 0 && rect.y >= 0 &&
    rect.x + rect.width <= viewport.width && rect.y + rect.height <= viewport.height, name+' starting camera');
  assert.equal(await host.locator('.bi-stage').count(), 8);
  assert.equal(inputs(initialPrompt, 'BIWorkflowControls').length, 0);
  assert.equal(inputs(initialPrompt, 'BIMakeSixGuides')[0].resolution, 2048);
  assert.equal(inputs(initialPrompt, 'BIGenerateFaceMaterials')[0].edit_resolution, 2048);
  assert.deepEqual(inputs(initialPrompt, 'BIModelCanvas').map(n => n.size), Array(6).fill(2048));

  const toggleMode = () => page.evaluate(() => window.app.extensionManager.command.execute('Comfy.ToggleLinear'));
  await toggleMode();
  const appPanel = page.locator('[data-testid="linear-widgets"] .bi-workflow-host');
  await appPanel.waitFor();
  // Fresh frontend profiles can open Comfy's first-use tour over App Mode.
  const tour = page.getByRole('dialog', {name:'Welcome to Apps'});
  try { await tour.waitFor({timeout:1800}); }
  catch (error) { if (error.name !== 'TimeoutError') throw error; }
  if (await tour.isVisible()) await tour.getByRole('button', {name:'Skip'}).click();
  await appPanel.getByRole('button', {name:'+Z のみ再生成対象にする', exact:true}).click();
  const edited = await compile();
  assert.deepEqual(changed(initialPrompt, edited), ['PrimitiveInt']);
  const dirty = panel => panel.locator('[data-dirty="true"]').evaluateAll(items => items.map(item => item.dataset.stage));
  assert.deepEqual(await dirty(appPanel), ['+Z', 'box']);
  await toggleMode();
  await page.locator('[data-testid="linear-widgets"]').waitFor({state:'hidden'});
  assert.deepEqual(await dirty(host), ['+Z', 'box']);
  await host.locator('summary').click();
  await host.getByLabel('Box とガイドの各面サイズ（px）').fill('512');
  await host.getByLabel('Box とガイドの各面サイズ（px）').press('Tab');
  const resized = await compile();
  assert.equal(inputs(resized, 'BIBakeBox')[0].resolution, 512);
  assert.equal(inputs(resized, 'BIMakeSixGuides')[0].resolution, 512);
  assert.deepEqual(inputs(resized, 'BIModelCanvas').map(n => n.size), Array(6).fill(2048));
  assert.equal(await host.locator('details').getAttribute('open'), '');

  await host.getByLabel('6面の生成サイズ').selectOption('1024');
  const smaller = await compile();
  assert.deepEqual(inputs(smaller, 'BIModelCanvas').map(n => n.size), Array(6).fill(1024));
  const threshold = host.getByLabel('SuperMat 金属判定のしきい値');
  const preset = host.getByLabel('粗さ・金属度のプロンプト');
  assert.equal(await preset.count(), 0);
  assert.equal(await threshold.count(), 0);
  await host.getByLabel('PBR マップの作成方法').selectOption('supermat');
  assert.equal(await preset.count(), 0);
  await threshold.fill('0.70');
  await threshold.press('Tab');
  const tuned = inputs(await compile(), 'BIGenerateFaceMaterials')[0];
  assert.equal(tuned.source, 'supermat');
  assert.equal(tuned.supermat_metal_threshold, 0.70);
  await host.getByLabel('PBR マップの作成方法').selectOption('cosmos');
  assert.equal(await host.getByLabel('PBR 出力サイズ（Cosmos の推定は最大512 px）').inputValue(), '2048');
  assert.equal(await preset.count(), 0);
  await host.getByLabel('PBR マップの作成方法').selectOption('marigold');
  assert.equal(await preset.count(), 0);
  await host.getByLabel('PBR マップの作成方法').selectOption('h3');
  assert.equal(await preset.inputValue(), 'detailed');
  await preset.selectOption('short');
  await host.getByLabel('PBR マップの作成方法').selectOption('qwen');
  assert.equal(await preset.inputValue(), 'short');
  await host.getByLabel('PBR 推定サイズ（Qwen・H3・SuperMat）').fill('1024');
  await host.getByLabel('PBR 推定サイズ（Qwen・H3・SuperMat）').press('Tab');
  await host.locator('summary').click();
  const later = await compile();
  assert.equal(inputs(later,'BIGenerateFaceMaterials')[0].edit_resolution, 1024);
  assert.equal(inputs(later,'BIGenerateFaceMaterials')[0].source, 'qwen');
  assert.equal(inputs(later,'BIGenerateFaceMaterials')[0].prompt_preset, 'short');
  const sidebar = await page.evaluate(() => {
    const tab = window.app.extensionManager.getSidebarTabs().find(t => t.id === 'box-impostor-workflow');
    const el = document.createElement('div');
    try {
      tab.render(el);
      return el.querySelector('[data-stage="+Z"]').dataset.dirty;
    } finally {tab.destroy();}
  });
  assert.equal(sidebar, 'true');
  await host.getByRole('button', {name:'3D のシードを変えて再生成対象にする'}).click();
  const rerolled = await compile();
  assert.deepEqual(changed(later, rerolled), ['KSampler']);
  assert.equal((await dirty(host)).length, 8);

  async function previews(panel, paths) {
    const list = panel.locator('.bi-reference-previews figure');
    await panel.evaluate(async (el) => {
      await Promise.all([...el.querySelectorAll('.bi-reference-previews img')].map(img => img.decode()));
    });
    assert.deepEqual(await list.evaluateAll(figures => figures.map(f => f.dataset.reference)), paths);
    assert.deepEqual(await list.locator('img').evaluateAll(images => images.map(img => {
      const params = new URL(img.src).searchParams;
      if (!img.naturalWidth || !img.naturalHeight) throw Error(img.alt);
      return [params.get('subfolder'), params.get('filename')].filter(Boolean).join('/');
    })), paths);
    assert.deepEqual(await list.locator('figcaption').allTextContents(), paths.map((path, i) =>
      (i ? 'Secondary ' + i : 'Primary') + ' · ' + path.split('/').pop()));
  }
  const png = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAEklEQVR4nGP4WsMQJbeeAUIBACZ8BTFcSCtrAAAAAElFTkSuQmCC', 'base64');
  const file = suffix => ({name:name+'-'+suffix+'.png', mimeType:'image/png', buffer:png});
  await host.getByLabel('Primary 画像をアップロード').setInputFiles(file('primary'));
  await page.waitForFunction(() => window.app.rootGraph.nodes.find(n => n.type === 'LoadImage')
    .widgets.find(w => w.name === 'image').value.includes('-primary'));
  await host.getByLabel('Secondary 画像を追加').setInputFiles([file('second-a'), file('second-b')]);
  await page.waitForFunction(() => window.app.rootGraph.nodes.find(n => n.type === 'BISecondaryReferences')
    .widgets.find(w => w.name === 'paths').value.includes('-second-b'));
  const uploaded = await compile();
  assert.match(inputs(uploaded, 'LoadImage')[0].image, /-primary/);
  const secondary = inputs(uploaded, 'BISecondaryReferences')[0].paths.split('\n');
  assert.equal(secondary.length, 2);
  assert.match(secondary[0], /-second-a/);
  assert.match(secondary[1], /-second-b/);

  const primary = inputs(uploaded, 'LoadImage')[0].image;
  await previews(host, [primary, ...secondary]);
  await host.getByLabel('Primary 画像をアップロード').setInputFiles(file('replacement'));
  await page.waitForFunction(() => window.app.rootGraph.nodes.find(n => n.type === 'LoadImage')
    .widgets.find(w => w.name === 'image').value.includes('-replacement'));
  const replacement = inputs(await compile(), 'LoadImage')[0].image;
  await previews(host, [replacement, ...secondary]);
  await host.getByLabel('Primary 画像', {exact:true}).selectOption(primary);
  await previews(host, [primary, ...secondary]);
  const pathsField = host.getByLabel('Secondary 画像（最大8枚・1行に1枚）');
  async function setPaths(paths) {
    await pathsField.fill(paths.join('\n'));
    await pathsField.press('Tab');
  }
  await setPaths([secondary[1], secondary[0]]);
  await previews(host, [primary, secondary[1], secondary[0]]);
  await setPaths([secondary[1]]);
  await previews(host, [primary, secondary[1]]);
  await setPaths([]);
  await previews(host, [primary]);
  await setPaths(['nonexistent-folder/missing #1.png']);
  await host.getByText('画像を表示できません').waitFor();
  assert.equal(await host.locator('.bi-reference-previews img').count(), 1);
  await setPaths(secondary);
  await toggleMode();
  await previews(appPanel, [primary, ...secondary]);
  await toggleMode();
  await page.locator('[data-testid="linear-widgets"]').waitFor({state:'hidden'});

  // Reopen the same JSON via Comfy's file entry point. Custom camera and graph
  // values must survive independently of which surface the operator used.
  await page.evaluate(async () => {
    const graph = window.app.rootGraph.serialize();
    graph.extra.ds = {scale:0.8,offset:[818,118]};
    await window.app.handleFile(new File([JSON.stringify(graph)],'edited-box.json',{type:'application/json'}),
      'file_drop',{deferWarnings:true});
  });
  await host.waitFor();
  assert.deepEqual(await page.evaluate(() => [...window.app.canvas.ds.offset]), [818,118]);
  assert.equal(await host.count(), 1);
  assert.deepEqual(await compile(), uploaded);
  await previews(host, [primary, ...secondary]);
}
