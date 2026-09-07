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
| **`MiniMax Prefix Cache Config`** | `MiniMaxH3/PrefixStream` | Configures precision (`FP8` / `BF16`), device mode (`Auto` / `GPU` / `CPU_Pinned`), and anchor/rolling window lengths. |
| **`MiniMax Prefix Cache Applier`** | `MiniMaxH3/PrefixStream` | Injects block-level DiT hooks into `model.model_options` and executes single-pass warmup before KSampler. |
| **`MiniMax Long Video Stitcher`** | `MiniMaxH3/PrefixStream` | Smoothly joins video latents and performs equal-power cosine crossfade on audio waveforms between adjacent clips. |
| **`MiniMax Cache Telemetry Monitor`**| `MiniMaxH3/PrefixStream` | Outputs live diagnostics on VRAM consumption, cache footprint, and session progress. |

---

## Installation

Clone this repository into your ComfyUI `custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/knoic/ComfyUI-MiniMaxH3-PrefixStream.git
```

Restart ComfyUI. The nodes will appear under the category `MiniMaxH3/PrefixStream`.

---

## License

This project is licensed under the [MIT License](LICENSE).
