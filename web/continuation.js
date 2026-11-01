import { app } from "../../../scripts/app.js";

const retiredInputs = new Set([
    "anchor_latent", "context_video_latent", "anchor_video_latent",
    "context_audio", "session", "audio_feather_ticks", "audio_tail_carryover",
]);

function removeRetiredInputs(node) {
    let changed = false;
    for (let index = (node.inputs?.length ?? 0) - 1; index >= 0; index--) {
        if (retiredInputs.has(node.inputs[index].name)) {
            node.removeInput(index);
            changed = true;
        }
    }
    if (changed) {
        node.setSize([node.size[0], node.computeSize()[1]]);
        node.setDirtyCanvas(true, true);
    }
}

app.registerExtension({
    name: "MiniMaxH3.ContinuationInputs",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "MiniMaxPrefixCacheApplier") return;
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = onConfigure?.apply(this, arguments);
            removeRetiredInputs(this);
            return result;
        };
    },
    loadedGraphNode(node) {
        if (node.comfyClass === "MiniMaxPrefixCacheApplier") removeRetiredInputs(node);
    },
});
