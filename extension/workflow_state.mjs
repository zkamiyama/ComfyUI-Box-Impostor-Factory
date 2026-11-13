// Compare each milestone's actual upstream widgets and links. Layout and node IDs of
// unrelated branches do not invalidate a completed face.
export const directions = ['+X', '-X', '+Y', '-Y', '+Z', '-Z'];

function edges(graph) {
  return new Map(graph.links.map(link => Array.isArray(link)
    ? [link[0], [link[1], link[2]]] : [link.id, [link.origin_id, link.origin_slot]]));
}

function settings(node, links) {
  return [node.type, node.mode ?? 0, node.widgets_values ?? [],
    // Live serialization may add unconnected widget sockets that graphToPrompt()
    // drops. Widget values above and actual connections determine the result.
    (node.inputs ?? []).filter(pin => links.has(pin.link)).map(pin => [pin.name, links.get(pin.link)])];
}

export function inspectWorkflow(graph) {
  const nodes = new Map(graph.nodes.map(n => [String(n.id), n]));
  const keepers = graph.nodes.filter(n => n.type === 'BICachedGeneratedFace');
  const guides = graph.nodes.find(n => n.type === 'BIMakeSixGuides');
  const preview = graph.nodes.find(n => n.type === 'BIPreviewBox');
  const pairs = keepers.map(n => [n.widgets_values?.[0], n]);
  if (keepers.length !== 6 || !guides || !preview || directions.some(f => !pairs.some(([k]) => k === f))) return null;
  const milestones = new Map([['geometry', guides], ...pairs, ['box', preview]]);
  const links = edges(graph);
  const subgraphs = new Map((graph.definitions?.subgraphs ?? []).map(sub => {
    const sublinks = edges(sub);
    return [sub.id, sub.nodes.map(n => [n.id, settings(n, sublinks)])];
  }));
  const signatures = {}, upstream = {};
  for (const [stage, end] of milestones) {
    const visited = new Set();
    function visit(node) {
      if (!node || visited.has(node.id)) return;
      visited.add(node.id);
      for (const input of node.inputs ?? []) {
        const source = links.get(input.link);
        if (source) visit(nodes.get(String(source[0])));
      }
    }
    visit(end);
    upstream[stage] = visited;
    signatures[stage] = JSON.stringify([...visited].sort((a,b) => a-b).map(id => {
      const n = nodes.get(String(id));
      return [id, settings(n, links), subgraphs.get(n.type)];
    }));
  }
  return {signatures, upstream, endIds: Object.fromEntries([...milestones].map(([k,n]) => [String(n.id),k]))};
}

export class WorkflowStatus {
  constructor(snapshot) {
    this.initial = snapshot.signatures;
    this.finished = new Map();
    this.failed = new Map();
    this.jobs = new Map();
    this.sequence = 0;
  }

  queue(id, snapshot) {
    this.jobs.set(id, {snapshot, seen: new Map(), order: ++this.sequence});
  }

  observe(id, node, output = {}) {
    const job = this.jobs.get(id);
    const stage = job?.snapshot.endIds[String(node)];
    if (!stage) return;
    const prior = job.seen.get(stage);
    // A cached notification carries no preview and may follow an executed payload.
    job.seen.set(stage, {...prior, ...output});
    this.complete(job, stage);
  }

  complete(job, stage) {
    const sig = job.snapshot.signatures[stage];
    const old = this.finished.get(stage);
    if (old && old.order > job.order) return;
    const output = {...(old?.sig === sig ? old.output : {}), ...job.seen.get(stage)};
    this.finished.set(stage, {sig, output, order: job.order});
    if ((this.failed.get(stage)?.order ?? 0) <= job.order) this.failed.delete(stage);
  }

  end(id, error = null) {
    const job = this.jobs.get(id);
    if (!job) return;
    if (error) {
      const root = Number(String(error.node_id ?? error.node ?? '').split(':')[0]);
      // Milestones are ordered upstream-first. A shared dependency failure belongs
      // to the earliest affected stage. Other downstream outputs remain stale.
      const direct = job.snapshot.endIds[String(root)];
      const targets = Object.keys(job.snapshot.upstream).filter(k => job.snapshot.upstream[k].has(root));
      const stage = direct ?? targets[0];
      if (stage && (this.finished.get(stage)?.order ?? 0) <= job.order &&
          (this.failed.get(stage)?.order ?? 0) <= job.order) {
        this.failed.set(stage, {sig: job.snapshot.signatures[stage], message: error.exception_message || '実行中断', order: job.order});
      }
    }
    this.jobs.delete(id);
  }

  stage(stage, snapshot) {
    const sig = snapshot.signatures[stage];
    const done = this.finished.get(stage);
    const failed = this.failed.get(stage);
    if (failed?.sig === sig) return {label: 'エラー', message: failed.message, output: done?.output};
    if (done?.sig === sig) return {label: '完了', output: done.output};
    return {label: (done || this.initial[stage] !== sig) ? 'Dirty · 要更新' : '未実行', output: done?.output};
  }
}
