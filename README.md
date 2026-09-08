# ComfyUI-MiniMaxH3-PrefixStream

[![GitHub License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-green.svg)]()
[![Model](https://img.shields.io/badge/model-MiniMax--H3-orange.svg)]()

A native masked audiovisual continuation and streaming long-video suite for MiniMax H3 in ComfyUI.

The default **Native Masked AV** path follows [Herrgott's H3 Infinite Continuation Suite](https://github.com/HerrgottMargott/Herrgotts-H3-Infinite-Continuation-Suite): it copies the previous AV latent into the new target head and protects video and audio independently with ComfyUI's native denoise masks. `Safe Native` remains available as the keyframe-based fallback.

---

## Key Highlights

- 🛡️ **Native Masked AV by Default**: Copies clean previous video/audio latents directly into the next target and protects them in-place; no DiT monkey-patching or experimental KV injection.
- 🎯 **Exact Joint AV Boundaries**: Uses the MiniMax H3-compatible `39 / 90 / 141 / 192 / ...` frame grid. The default 39-frame context is about 1.625 seconds and maps exactly to 65 audio-latent ticks.
- 🎵 **Independent Audio Protection**: Video and audio receive separate native masks. Dialogue can keep the full previous audio tail, while an optional audio-only feather releases the final protected ticks gradually.
- 🔒 **Collision-Safe Conditioning**: Keyframes inside the protected video head are removed so a frame-0 guide cannot fight the copied latent context. Future endpoint and reference conditioning remain available.
- 🧩 **Safe Native Fallback**: The earlier native-attention/keyframe continuation path remains selectable for compatibility with workflows that cannot use masked latents.
- ✂️ **Seam-Ready Outputs**: Session metadata drives exact duplicate-head trimming, with pixel-space video trimming and synchronized audio handling for final assembly.

---

## At a Glance

### Choose the continuation path in one node

`MiniMax H3 Continuation Config` keeps the main choice deliberately small: use **Native Masked AV (Recommended)** for normal continuation, and switch to **Safe Native** only when a workflow needs the compatibility fallback. `continuation_frames` selects the protected overlap length on the H3 frame grid.

![MiniMax H3 Continuation Config](assets/readme/continuation-config.png)

### Browse and resume from the Clip Bin

`MiniMax H3 Clip Bin Picker` presents saved clips as a visual gallery. Pick a prior shot, filter by rating, and use its latent and tail frame as the next continuation context—without hunting through output folders.

![MiniMax H3 Clip Bin Picker gallery](assets/readme/clip-bin-picker-gallery.png)

---

## Architecture Overview

```text
Previous full AV latent ──► select canonical 39/90/141/... frame tail
                                      │
Target empty AV latent ───────────────┼──► copy tail into target head
                                      └──► noise_mask=(video mask, audio mask)
                                                        │
                                                        ▼
                                      Native ComfyUI MiniMax H3 sampling
```

---

## Custom Nodes

| Node Name | Category | Description |
| :--- | :--- | :--- |
| **`MiniMax H3 Continuation Config`** | `MiniMaxH3/PrefixStream` | Selects `Native Masked AV` (default) or `Safe Native` (fallback), plus a user-visible video context length. |
| **`MiniMax H3 Continuation Applier`** | `MiniMaxH3/PrefixStream` | Builds the native video/audio masks and outputs `masked_latent`, or applies collision-safe keyframe conditioning in fallback mode. |
| **`MiniMax Trim Prefix`** | `MiniMaxH3/PrefixStream` | Trims leading overlap frames in pixel and audio waveform space, guaranteeing zero VAE causal flicker and perfect sync. |
| **`MiniMax Long Video Stitcher`** | `MiniMaxH3/PrefixStream` | Seamlessly joins video in pixel space (with luminance gain matching) and audio waveforms (equal-power crossfade). |
| **`MiniMax Save AV Latent`** | `MiniMaxH3/PrefixStream` | Standalone node to save joint AV latents to safetensors without external dependencies. |
| **`MiniMax Load AV Latent`** | `MiniMaxH3/PrefixStream` | Standalone node to load joint AV latents with metadata for multi-clip continuous streaming. |
| **`MiniMax H3 Clip Bin Saver`** | `MiniMaxH3/PrefixStream` | Archives a generated clip with a preview, rating, shot tag, prompt, and continuation lineage. |
| **`MiniMax H3 Clip Bin Picker`** | `MiniMaxH3/PrefixStream` | Gallery-based loader for finding a saved clip and exposing its latent, tail frame, prompt, and ID. |
| **`MiniMax Cache Telemetry Monitor`**| `MiniMaxH3/PrefixStream` | Reports the active continuation mode, protected context geometry, and session progress. |

---

## Installation

Clone this repository into your ComfyUI `custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/knoic/ComfyUI-MiniMaxH3-PrefixStream.git
cd ComfyUI-MiniMaxH3-PrefixStream
pip install -r requirements.txt
```

Restart ComfyUI. The nodes will appear under the category `MiniMaxH3/PrefixStream`.

---

## Running Unit Tests

Verify everything locally:

```bash
python tests/test_cache_manager.py
python tests/test_fused_attention.py
python tests/test_nodes_and_pipeline.py
python tests/test_native_masked_av.py
```

---

## Example Workflow & Documentation

- 📄 **Ready-to-Use Workflow**: [`examples/MiniMaxH3_PrefixStream_v1.0.json`](examples/MiniMaxH3_PrefixStream_v1.0.json)
  - Drag and drop this complete 42-node JSON directly into your ComfyUI canvas to run PrefixStream long video generation.
- 📖 **Detailed Chinese User Guide (中文详细使用指南)**: [`docs/USER_GUIDE_CN.md`](docs/USER_GUIDE_CN.md)
  - Includes hardware recommendations, parameter tuning, multi-clip infinite continuation walkthroughs, and FAQ.

---

## Acknowledgments & References (致谢与参考项目)

本项目的诞生与时空续写设计深受以下开源项目与作者的启发，特此致以诚挚的感谢与敬意：

1. **[ComfyUI-H3-Motion-Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context)** by **[@NikoDemon80](https://github.com/NikoDemon80)**
   - 感谢 NikoDemon80 在 MiniMax H3 关键帧锚定算法、VAE 时空周期相位网格对齐公式（Snap to Run Grid）、音视频头部裁切及 5/3 音视频时间缩放比例方面的先驱性数学探索与启发。
   - *Pioneering formulations for MiniMax H3 keyframe anchoring, VAE phase grid alignment, and audio-video temporal ratios.*

2. **[Herrgotts-H3-Infinite-Continuation-Suite](https://github.com/HerrgottMargott/Herrgotts-H3-Infinite-Continuation-Suite)** by **[@HerrgottMargott](https://github.com/HerrgottMargott)**
   - 感谢 Native Masked AV 的原生分流遮罩、精确 AV 上下文边界及独立音频保护方案。
   - *Native per-stream masked AV continuation, exact joint AV context geometry, and independent audio protection.*

---

## License

This project is licensed under the [MIT License](LICENSE).
