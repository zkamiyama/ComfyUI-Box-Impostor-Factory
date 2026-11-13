import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { installWorkflowPanel } from './workflow_panel.js';

const stylesheet = document.createElement('link');
stylesheet.rel = 'stylesheet';
stylesheet.href = new URL('./workflow_controls.css', import.meta.url).href;
document.head.append(stylesheet);

let controls;
app.registerExtension({
  name: "BoxImpostor.Viewer",
  setup() { controls ??= installWorkflowPanel(app, api); },
  afterConfigureGraph() { controls?.draw(true); },
  nodeCreated(node) {
    if (node.comfyClass !== 'BIWorkflowControls') return;
    node.isVirtualNode = true;
    const host = document.createElement('div');
    host.className = 'bi-workflow-host';
    // Scrolling the face grid and typing into fields should act on the panel.
    for (const event of ['wheel', 'keydown', 'pointerdown', 'dblclick'])
      host.addEventListener(event, e => e.stopPropagation());
    const panel = node.addDOMWidget('workflow', 'BI_WORKFLOW_PANEL', host, {
      serialize: false, hideOnZoom: false,
      getMinHeight: () => 300, getHeight: () => Math.max(300, node.size[1] - 14),
    });
    panel.label = 'Box Impostor';
    node.setSize([650, 640]);
    const added = node.onAdded, removed = node.onRemoved;
    let detach;
    node.onAdded = function (...args) {
      added?.apply(this, args);
      // Wait for Comfy to finish configuring the workflow and extension manager.
      queueMicrotask(() => {
        if (!node.graph || detach) return;
        controls ??= installWorkflowPanel(app, api);
        detach = controls.mount(host, node);
      });
    };
    node.onRemoved = function (...args) {
      detach?.();
      detach = null;
      return removed?.apply(this, args);
    };
  },
  async beforeRegisterNodeDef(type, data) {
    if (data.name !== "BIPreviewBox" && data.display_name !== "BI PreviewBox") return;
    const original = type.prototype.onExecuted;
    type.prototype.onExecuted = function (message) {
      original?.apply(this, arguments);
      const path = message?.box_impostor?.[0];
      if (typeof path !== "string" || !path.startsWith("/box-impostor/")) return;
      this.boxImpostorViewerURL = api.apiURL(path);
      if (!this.widgets?.some(widget => widget.name === "3Dプレビューを開く")) {
        this.addWidget("button", "3Dプレビューを開く", null, () => {
          if (this.boxImpostorViewerURL) window.open(this.boxImpostorViewerURL, "_blank", "noopener,noreferrer");
        }, {serialize:false});
        this.setSize(this.computeSize());
      }
      this.setDirtyCanvas(true, true);
    };
  }
});
