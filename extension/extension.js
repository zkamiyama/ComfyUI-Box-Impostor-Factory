import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { installWorkflowPanel } from './workflow_panel.js';

app.registerExtension({
  name: "BoxImpostor.Viewer",
  setup() { installWorkflowPanel(app, api); },
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
