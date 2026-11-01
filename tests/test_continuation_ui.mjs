import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

const source = readFileSync(new URL("../web/continuation.js", import.meta.url), "utf8")
    .replace(/^import .*;\r?$/m, "");

function loadExtension() {
    let extension;
    vm.runInNewContext(source, { app: { registerExtension(value) { extension = value; } } });
    return extension;
}

test("old workflow inputs are removed while current connections and session output survive", () => {
    const extension = loadExtension();
    class Node {
        onConfigure() { this.originalCalled = true; return 42; }
        removeInput(index) { this.removed.push(index); this.inputs.splice(index, 1); }
        computeSize() { return [200, this.inputs.length * 20 + 60]; }
        setSize(size) { this.size = size; }
        setDirtyCanvas() { this.redraws++; }
    }
    extension.beforeRegisterNodeDef(Node, { name: "MiniMaxPrefixCacheApplier" });
    const node = new Node();
    node.comfyClass = "MiniMaxPrefixCacheApplier";
    node.inputs = ["model", "conditioning", "cache_config", "context_latent", "target_latent",
        "anchor_latent", "context_video_latent", "identity_latent", "anchor_video_latent",
        "context_audio", "session", "continuation_model", "audio_feather_ticks", "audio_tail_carryover"]
        .map((name, index) => ({ name, link: index + 100 }));
    const contextLink = node.inputs[3];
    const targetLink = node.inputs[4];
    node.outputs = [{ name: "model" }, { name: "conditioning" }, { name: "session", links: [500] }];
    node.size = [600, 800];
    node.removed = [];
    node.redraws = 0;
    assert.equal(node.onConfigure(), 42);
    assert.equal(node.originalCalled, true);
    assert.deepEqual(node.inputs.map(input => input.name), ["model", "conditioning", "cache_config",
        "context_latent", "target_latent", "identity_latent", "continuation_model"]);
    assert.equal(node.inputs[3], contextLink);
    assert.equal(node.inputs[4], targetLink);
    assert.deepEqual(node.outputs[2], { name: "session", links: [500] });
    assert.deepEqual(node.removed, [13, 12, 10, 9, 8, 6, 5]);
    assert.equal(node.size[0], 600);
    assert.ok(node.size[1] < 800);
    extension.loadedGraphNode(node);
    assert.equal(node.redraws, 1);
});

test("other nodes keep their session inputs", () => {
    const extension = loadExtension();
    class Other { onConfigure() { return "unchanged"; } }
    const original = Other.prototype.onConfigure;
    extension.beforeRegisterNodeDef(Other, { name: "MiniMaxTrimPrefixLatent" });
    assert.equal(Other.prototype.onConfigure, original);
    const node = { comfyClass: "MiniMaxTrimPrefixLatent", inputs: [{ name: "session" }] };
    extension.loadedGraphNode(node);
    assert.equal(node.inputs[0].name, "session");
});
