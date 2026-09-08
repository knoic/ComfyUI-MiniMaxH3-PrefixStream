# ComfyUI-MiniMaxH3-PrefixStream

[![GitHub License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-green.svg)]()
[![Model](https://img.shields.io/badge/model-MiniMax--H3-orange.svg)]()

A high-performance **Prefix KV Caching & Streaming Long Video Suite** for MiniMax H3 (Hailuo DiT) in ComfyUI.

By mathematically decoupling the temporal causality of video diffusion and precomputing Key/Value activations for static context frames, **PrefixStream** eliminates redundant Linear, RMSNorm, and SwiGLU MLP computations across all 50 DiT blocks during sampling.

---

## Key Highlights

- ⚡ **40% ~ 48% Speedup per Chunk**: Slashes DiT forward FLOPs during diffusion sampling steps by processing Query for new target frames only.
- 🛡️ **Dual-Tier Identity Protection (Zero Degradation)**:
  - **Anchor KV (World Origin)**: Locks frame 0 identity (facial structure, costume, scene lighting) permanently across unlimited sequential clips.
  - **Rolling KV (Dynamic Motion)**: Preserves motion velocity and continuity using the last 6~8 latent frames of the preceding clip.
- 💾 **Consumer GPU Friendly (24GB RTX 3090/4090)**:
  - **Native FP8 KV Cache** (`torch.float8_e4m3fn`): Halves cache VRAM footprint to ~3.9 GB.
  - **CPU-Pinned Asynchronous Streaming**: Transparently streams layer-by-layer KV caches via dedicated CUDA streams, reducing GPU VRAM increment to **0 MB**.
- 🎵 **Seamless Audio-Video Handover**:
  - 50ms equal-power cosine crossfade for audio streams to prevent clicks and pops.
  - Latent boundary soft-blending and automatic overlap trimming.

---

## Architecture Overview

```
[Historical Clean Clip N] ──►【Phase 0: 1-Shot Warmup】──► Freeze 50-layer Prefix KV Cache
                                                                    │
                                                ┌───────────────────┘ (Read-Only KV)
                                                ▼
[Target Noisy Clip N+1]   ──►【Phase 1: 25 Denoise Steps】──► Compute Target QKV only
                                                        └──► Attn(Q_target, [KV_prefix, KV_target])
                                                        └──► Bypasses Prefix Linear/Norm/FFN completely!
```

---

## Custom Nodes

| Node Name | Category | Description |
| :--- | :--- | :--- |
| **`MiniMax Prefix Cache Config`** | `MiniMaxH3/PrefixStream` | Configures mode (`Safe Native` / `Step-1 Dynamic`), precision (`FP8` / `BF16`), device mode, and grid-aligned rolling window lengths. |
| **`MiniMax Prefix Cache Applier`** | `MiniMaxH3/PrefixStream` | Injects block-level DiT hooks or native conditioning keyframe anchors with frame-0 collision removal. |
| **`MiniMax Trim Prefix`** | `MiniMaxH3/PrefixStream` | Trims leading overlap frames in pixel and audio waveform space, guaranteeing zero VAE causal flicker and perfect sync. |
| **`MiniMax Long Video Stitcher`** | `MiniMaxH3/PrefixStream` | Seamlessly joins video in pixel space (with luminance gain matching) and audio waveforms (equal-power crossfade). |
| **`MiniMax Save AV Latent`** | `MiniMaxH3/PrefixStream` | Standalone node to save joint AV latents to safetensors without external dependencies. |
| **`MiniMax Load AV Latent`** | `MiniMaxH3/PrefixStream` | Standalone node to load joint AV latents with metadata for multi-clip continuous streaming. |
| **`MiniMax Cache Telemetry Monitor`**| `MiniMaxH3/PrefixStream` | Outputs live diagnostics on VRAM consumption, cache footprint, and session progress. |

---

## 100% Standalone & Independent Architecture

- 🛡️ **Zero External Patching**: Does **NOT** require any external file-patching scripts (`patch_model.py` is NEVER needed). Operates directly on stock, official ComfyUI.
- 🚀 **Zero Third-Party Suite Dependencies**: Completely replaces third-party latent loaders/savers or speed patches (e.g. `TE-Speed-MiniMaxH3`, `Herrgotts-H3-Infinite-Continuation-Suite`, `ReservedVRAM`). Everything needed for long video streaming continuation is built natively into this repository.
- 🔒 **Safe Native Mode**: 100% native ComfyUI attention with zero DiT monkey-patching for rock-solid stability and zero artifacts.

---

## Installation

Clone this repository into your ComfyUI `custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/knoic/ComfyUI-MiniMaxH3-PrefixStream.git
```

Restart ComfyUI. The nodes will appear under the category `MiniMaxH3/PrefixStream`.

---

## Example Workflow & Documentation

- 📄 **Ready-to-Use Workflow**: [`examples/MiniMaxH3_PrefixStream_LongVideo_Workflow.json`](examples/MiniMaxH3_PrefixStream_LongVideo_Workflow.json)
  - Drag and drop this JSON directly into your ComfyUI canvas to run PrefixStream long video generation.
- 📖 **Detailed Chinese User Guide (中文详细使用指南)**: [`docs/USER_GUIDE_CN.md`](docs/USER_GUIDE_CN.md)
  - Includes hardware recommendations, parameter tuning, multi-clip infinite continuation walkthroughs, and FAQ.

---

## Acknowledgments & References (致谢与参考项目)

本项目的诞生离不开开源社区与前行者的探索。在此向以下开源项目、作者与团队致以最诚挚的感谢与敬意（排名不分先后）：

1. **[MiniMax AI (Hailuo Team)](https://www.minimax.io/)**
   - 感谢 MiniMax 团队打造并开源了卓越的 **MiniMax H3** 全模态视音频基座模型，在复杂动作表现力、影视级质感与原生音视频联合生成领域树立了全新标杆。
   - *Special thanks to the MiniMax AI team for the revolutionary MiniMax H3 omni-modal audio-video foundation model.*

2. **[ComfyUI](https://github.com/comfyanonymous/ComfyUI)** by **[@comfyanonymous](https://github.com/comfyanonymous)**
   - 感谢 ComfyUI 奠定了模块化生成式 AI 的基石，以及对 MiniMax H3 架构、NestedTensor 视音频联合潜空间、内存管理系统的原生支持。
   - *The cornerstone modular generative AI framework and its native MiniMax H3 implementation.*

3. **[ComfyUI-H3-Motion-Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context)** by **[@NikoDemon80](https://github.com/NikoDemon80)**
   - 感谢 NikoDemon80 在 MiniMax H3 关键帧锚定算法、VAE 时空周期相位网格对齐公式（Snap to Run Grid）、音视频头部裁切及 5/3 音视频时间缩放比例方面的先驱性数学探索与启发。
   - *Pioneering formulations for MiniMax H3 keyframe anchoring, VAE phase grid alignment, and audio-video temporal ratios.*

4. **[Herrgotts-H3-Infinite-Continuation-Suite](https://github.com/Herrgotts/Herrgotts-H3-Infinite-Continuation-Suite)** by **[@Herrgotts](https://github.com/Herrgotts)**
   - 感谢 Herrgott 提出的无限长视频链式接力构想、上下文对齐平滑接缝理念、以及音视频无缝拼接工作流的探索。
   - *Groundbreaking principles of infinite video chaining, context-aligned seamless AV joining, and safe tail bridging.*

5. **[ComfyUI-VideoHelperSuite (VHS)](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)** by **[@Kosinkadink](https://github.com/Kosinkadink)**
   - 感谢 VHS 套件在 ComfyUI 社区中为视频加载、组合、编码与视音频复用（Muxing）树立的可靠标准与卓越工具链支持。
   - *The industry-standard toolchain for video loading, encoding, combining, and audio-video muxing in ComfyUI.*

6. **[ComfyUI-MiniMaxH3-TimelineDirector](https://github.com/AIGODLIKE/ComfyUI-MiniMaxH3-TimelineDirector)** by **[AIGODLIKE](https://github.com/AIGODLIKE)**
   - 感谢其在多镜头剧本式时间线规划、分镜推进与参考图/提示词协同调度方面的思路启发。
   - *Inspiration for multi-segment timeline orchestration and reference-driven prompt sequencing.*

7. **[TE-Speed-MiniMaxH3](https://github.com/AIGODLIKE/TE-Speed-MiniMaxH3)** / 社区加速探索
   - 感谢社区早期在 MiniMax H3 推理加速方面的探索，促使我们深入 DiT 注意力层机制并研发出原生的 Prefix KV Caching 解决方案。
   - *Early community explorations on MiniMax H3 acceleration that inspired our research into clean, mathematical Prefix KV caching.*

---

## License

This project is licensed under the [MIT License](LICENSE).
