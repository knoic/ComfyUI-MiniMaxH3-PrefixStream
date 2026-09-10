# ComfyUI-MiniMaxH3-PrefixStream

[![GitHub License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-green.svg)]()
[![Model](https://img.shields.io/badge/model-MiniMax--H3-orange.svg)]()

English | [简体中文](README.md)

A ComfyUI node suite for **MiniMax H3 long-video generation**. It turns short generated clips into a manageable, selectable, and seamlessly connected long-video workflow: generate a clip, archive it in the Clip Bin, visually choose the shot to continue from, then generate the next clip.

It integrates with existing native MiniMax H3 workflows with minimal intrusion: keep your model, prompts, sampler, and decode chain, then add configuration and application nodes only where continuation is needed. The default **Native Masked AV** path follows ideas from [Herrgott's H3 Infinite Continuation Suite](https://github.com/HerrgottMargott/Herrgotts-H3-Infinite-Continuation-Suite), protecting the previous audiovisual context with ComfyUI's native denoise masks. `Safe Native` remains available as a compatibility fallback.

---

## Key Highlights

- 🎬 **Built for long-video continuation**: Every clip is both an output and reusable context for the next one. Overlap trimming and audiovisual synchronization are handled automatically, so segmented generations can reliably accumulate into a long video.
- 🧩 **Low-intrusion native H3 integration**: Does not modify ComfyUI source files or monkey-patch H3 DiT blocks. Keep your existing model, prompt, sampling, and decoding nodes; add only the few nodes needed for continuation.
- 🛡️ **Native Masked AV by default**: Writes clean video/audio latents from the previous clip into the head of the next target and protects them with native ComfyUI masks. Video and audio are controlled independently. `Safe Native` is retained as the compatibility fallback.
- 🗂️ **Visual continuation media library**: Archives clip previews, ratings, shot tags, prompts, and parent-child lineage. Choose any historical clip by image and rating from a gallery instead of guessing from filenames.
- 🎯 **Precise, controllable context**: Uses the H3-compatible `39 / 90 / 141 / 192 / ...` frame grid. The default 39-frame context is about 1.625 seconds and maps exactly to 65 audio-latent ticks.
- 🔒 **Reduced continuation conflicts**: Removes keyframes inside the protected head so a frame-0 guide cannot conflict with copied latent context. Later endpoint and reference conditioning remain available.

---

## At a Glance

```text
Native MiniMax H3 workflow → generate current clip → automatically archive to Clip Bin
                                             │
                                             ▼
                         choose any historical shot in the visual gallery
                                             │
                                             ▼
                  Continuation Config / Applier → continue native H3 sampling
                                             │
                                             ▼
                     trim overlapping head → stitch long video → archive again
```

### Choose the continuation path in one node

`MiniMax H3 Continuation Config` keeps the decision deliberately small: use **Native Masked AV (Recommended)** for normal continuation and switch to **Safe Native** only when a workflow requires the compatibility fallback. `continuation_frames` chooses the protected overlap length on the H3 frame grid.

![MiniMax H3 Continuation Config](assets/readme/continuation-config.png)

### Browse and continue from the Clip Bin

`MiniMax H3 Clip Bin Picker` displays saved clips as a visual gallery. Select a preceding shot, filter by rating, and use its latent and tail frame as the next continuation context—without searching through output folders.

![MiniMax H3 Clip Bin Picker gallery](assets/readme/clip-bin-picker-gallery.png)

---

## How It Works

```text
Previous full AV latent ──► select the canonical 39/90/141/... frame tail
                                      │
Target empty AV latent ───────────────┼──► copy tail into the target head
                                      └──► noise_mask = (video mask, audio mask)
                                                        │
                                                        ▼
                                      Native ComfyUI MiniMax H3 sampling
```

---

## Custom Nodes

| Node Name | Category | Description |
| :--- | :--- | :--- |
| **`MiniMax H3 Continuation Config`** | `MiniMaxH3/PrefixStream` | Selects default `Native Masked AV` or fallback `Safe Native`, and sets the user-visible video context length. |
| **`MiniMax H3 Continuation Applier`** | `MiniMaxH3/PrefixStream` | Builds native video/audio masks and outputs `masked_latent`; applies collision-safe keyframe conditioning in fallback mode. |
| **`MiniMax Trim Prefix`** | `MiniMaxH3/PrefixStream` | Trims leading overlap frames in pixel and audio waveform space, avoiding VAE causal flicker while retaining sync. |
| **`MiniMax Long Video Stitcher`** | `MiniMaxH3/PrefixStream` | Seamlessly stitches accumulated long video and new clips in pixel and audio waveform space with luminance matching and linear audio overlap crossfade. |
| **`MiniMax Save AV Latent`** | `MiniMaxH3/PrefixStream` | Saves joint audiovisual latents to safetensors without external dependencies. |
| **`MiniMax Load AV Latent`** | `MiniMaxH3/PrefixStream` | Loads joint audiovisual latents and their metadata for multi-clip continuous streaming. |
| **`MiniMax H3 Clip Bin Saver`** | `MiniMaxH3/PrefixStream` | Archives a generated clip with its preview, rating, shot tag, prompt, and continuation lineage. |
| **`MiniMax H3 Clip Bin Picker`** | `MiniMaxH3/PrefixStream` | Gallery-based loader that exposes a selected clip's latent, tail frame, prompt, and ID. |
| **`MiniMax Cache Telemetry Monitor`** | `MiniMaxH3/PrefixStream` | Shows the active continuation mode, protected-context geometry, and session progress. |

---

## Installation

Clone the repository into ComfyUI's `custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/knoic/ComfyUI-MiniMaxH3-PrefixStream.git
cd ComfyUI-MiniMaxH3-PrefixStream
pip install -r requirements.txt
```

Restart ComfyUI. The nodes appear under `MiniMaxH3/PrefixStream`.

---

## Running Unit Tests

Verify the project locally:

```bash
python -m unittest discover -s tests -v
```

---

## Example Workflow and Documentation

- 📄 **Ready-to-use workflow**: [`examples/MiniMaxH3_PrefixStream_v1.0.json`](examples/MiniMaxH3_PrefixStream_v1.0.json)
  - Drag this complete 42-node JSON directly onto a ComfyUI canvas to run PrefixStream long-video generation.
- 📖 **Detailed Chinese user guide**: [`docs/USER_GUIDE_CN.md`](docs/USER_GUIDE_CN.md)
  - Includes hardware guidance, parameter tuning, multi-clip infinite-continuation walkthroughs, and FAQs.

---

## Acknowledgments and References

This project and its temporal continuation design were strongly inspired by the following open-source projects and authors:

1. **[ComfyUI-H3-Motion-Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context)** by **[@NikoDemon80](https://github.com/NikoDemon80)**
   - For pioneering work on MiniMax H3 keyframe anchoring, VAE spatiotemporal phase-grid alignment (Snap to Run Grid), audiovisual head trimming, and 5/3 audiovisual temporal scaling.

2. **[Herrgotts-H3-Infinite-Continuation-Suite](https://github.com/HerrgottMargott/Herrgotts-H3-Infinite-Continuation-Suite)** by **[@HerrgottMargott](https://github.com/HerrgottMargott)**
   - For its native per-stream AV masking, exact audiovisual context boundaries, and independent audio-protection design.

---

## License

This project is released under the [MIT License](LICENSE).


## Disk-backed long video output

`MiniMax H3 Disk Video Stream` writes each decoded clip to disk and exports a final MP4 without accumulating the complete IMAGE timeline in RAM. FFmpeg must be on PATH. Connect only the current clip's images/audio and optionally its session to trim the prefix. Keep the project and stream names constant, and leave export disabled while appending. To export without appending again, disconnect images/audio and enable export on a node with the same names. Every execution with images appends another clip; use a new stream name for a new video.

This path uses hard cuts, eight-frame encoding batches and PCM intermediate audio. All clips must have consistent fps, even resolution, sample rate, channels and audio presence. The existing IMAGE stitcher remains available for brightness matching and overlap blending, and still retains the full output in memory. Project locks protect concurrent operations within one ComfyUI process; do not share a writable bin between multiple processes. VAE decoding errors now propagate instead of producing empty results. Restart ComfyUI and refresh the browser after updating.
