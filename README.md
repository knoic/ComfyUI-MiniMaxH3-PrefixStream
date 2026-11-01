# ComfyUI-MiniMaxH3-PrefixStream

[![GitHub License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-green.svg)]()
[![Model](https://img.shields.io/badge/model-MiniMax--H3-orange.svg)]()

[English](README_EN.md) | 简体中文

面向 **MiniMax H3 长视频生成** 的 ComfyUI 节点套件。它把一段段短片组织成可管理、可选择、可无缝承接的长视频流程：生成片段、归档到素材箱、可视化挑选要承接的镜头，再继续生成下一段。

它以低侵入方式接入现有的原生 MiniMax H3 工作流：保留你的模型、提示词、采样器与解码链路，只在续写处插入配置与应用节点。默认的 **Native Masked AV** 路径参考了 [Herrgott's H3 Infinite Continuation Suite](https://github.com/HerrgottMargott/Herrgotts-H3-Infinite-Continuation-Suite)，使用 ComfyUI 原生去噪遮罩保护上一段的音视频上下文；`Safe Native` 则是兼容性备选方案。

---

## 核心特性

- 🎬 **为长视频续写而生**：每一段既是成片，也是下一段可复用的上下文；自动处理重叠开头裁切与音画同步，让分段生成可以稳定累积为长视频。
- 🧩 **低侵入地兼容原生 H3 工作流**：不改动 ComfyUI 源码，也不 monkey-patch H3 DiT block。沿用原有模型、提示词、采样和解码节点，只增加续写所需的少量节点。
- 🛡️ **默认使用 Native Masked AV**：将上一段干净的视频/音频 latent 直接写入下一段开头，并用 ComfyUI 原生遮罩保护；视频、音频各自独立控制。`Safe Native` 保留为兼容性备选。
- 🗂️ **可视化接力素材箱**：自动归档片段预览、评分、镜头标签、提示词及前后镜头血缘；在画廊中按画面和评分选择任意历史片段，直接接力，而不是靠文件名猜测。
- 🎯 **准确且可控的上下文**：支持 H3 的 `5 / 22 / 39 / 56 / ... / 192` 帧网格。22 帧对应 7 个视频 latent 和 37 个音频 tick；默认 39 帧对应 12 和 65。
- 🔒 **减少续写冲突**：移除受保护开头内的关键帧，避免第 0 帧引导与复制的 latent 上下文相互冲突；后续端点和参考条件仍可正常使用。
- 🖼️ **保留整组原生图片参考**：随片段保存用户提供的图片参考，续写时缺失才恢复；当前参考优先，不自动将生成首帧或关键帧转换成新参考。支持 `identity_latent` 显式补接图片，以及保存、素材箱和裁切后的时序状态传递。

---

## 快速浏览

```text
原生 MiniMax H3 工作流 → 生成当前片段 → Clip Bin 自动归档
                                      │
                                      ▼
                     在画廊中选择任意历史镜头作为接力来源
                                      │
                                      ▼
                  Continuation Config / Applier → 原生 H3 继续采样
                                      │
                                      ▼
                        裁切重叠开头 → 拼接为长视频 → 再次归档
```

### 一个节点选择续写方案

`MiniMax H3 Continuation Config` 只保留必要的核心选择：普通续写使用 **Native Masked AV (Recommended)**；只有工作流需要兼容方案时才切换为 **Safe Native**。`continuation_frames` 用于选择 H3 帧网格上的受保护重叠长度。

![MiniMax H3 续写配置节点](assets/readme/continuation-config.png)

### 从素材箱浏览并继续镜头

`MiniMax H3 Clip Bin Picker` 将已保存的片段以可视化画廊呈现。选择上一镜头、按评分筛选，并将其 latent 与尾帧作为下一段的续写上下文，无需在输出目录中翻找文件。

![MiniMax H3 素材箱画廊选择器](assets/readme/clip-bin-picker-gallery.png)

---

## 工作原理

```text
上一段完整音视频 latent ──► 选择规范的 5/22/39/56/... 帧尾部
                                      │
目标空音视频 latent ───────────────────┼──► 将尾部复制到目标开头
                                      └──► noise_mask =（视频遮罩，音频遮罩）
                                                        │
                                                        ▼
                                      ComfyUI 原生 MiniMax H3 采样
```

---

## 自定义节点

| 节点名称 | 分类 | 说明 |
| :--- | :--- | :--- |
| **`MiniMax Video Chunk Slicer`** | `MiniMaxH3/VideoEdit` | 🎬 长视频智能切片器（零显存流式加载）：直接按需提取长视频片段，支持自动网格/自由区间切片，**支持直接输出上一片段末帧（单帧图片参考）与尾部序列（多帧视频参考）**，自动优先提取已编辑成果，为下游图生视频/视频延续提供无缝参考。 |
| **`MiniMax Video Patch Reassembler`** | `MiniMaxH3/VideoEdit` | 🧩 长视频无缝回填缝合器：自动将生成的片段内存替换进母带时间轴，支持余弦微融合消除缝合线，可视化追踪全片完成度并一键导出最终成品 MP4。 |
| **`MiniMax H3 Continuation Config`** | `MiniMaxH3/PrefixStream` | 选择默认的 `Native Masked AV` 或备选的 `Safe Native`，并设置用户可见的视频上下文长度。 |
| **`MiniMax H3 Continuation Applier`** | `MiniMaxH3/PrefixStream` | 构建原生视频/音频遮罩并输出 `masked_latent`；在备选模式下应用避免冲突的关键帧条件。 |
| **`MiniMax Trim Prefix`** | `MiniMaxH3/PrefixStream` | 在像素和音频波形空间裁切开头的重叠帧，避免 VAE 因果闪烁并保持同步。 |
| **`MiniMax Long Video Stitcher`** | `MiniMaxH3/PrefixStream` | 在像素和音频波形空间无缝拼接长视频与新片段，具备自适应亮度匹配、重叠区不增益的余弦交叉淡化及防二次裁剪守卫。 |
| **`MiniMax Save AV Latent`** | `MiniMaxH3/PrefixStream` | 独立地将联合音视频 latent 保存为 safetensors，无外部依赖。 |
| **`MiniMax Load AV Latent`** | `MiniMaxH3/PrefixStream` | 独立加载带有元数据的联合音视频 latent，用于多片段连续流式生成。 |
| **`MiniMax H3 Clip Bin Saver`** | `MiniMaxH3/PrefixStream` | 连同预览图、评分、镜头标签、提示词和续写血缘归档生成片段。 |
| **`MiniMax H3 Clip Bin Picker`** | `MiniMaxH3/PrefixStream` | 以画廊方式查找已保存的片段（支持卡片流与树形图谱一键切换），并输出其 latent、尾帧、提示词和 ID。 |
| **`MiniMax H3 Clip Bin Tree Picker`** | `MiniMaxH3/ClipBin` | 以 DAG 分支血缘拓扑树（Tree View）全景可视化历史片段、多分支探索版本与完整故事线祖先链路。 |
| **`MiniMax Cache Telemetry Monitor`**| `MiniMaxH3/PrefixStream` | 显示当前续写模式、受保护上下文几何信息和会话进度。 |

---

## 安装

将仓库克隆到 ComfyUI 的 `custom_nodes` 目录：

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/knoic/ComfyUI-MiniMaxH3-PrefixStream.git
cd ComfyUI-MiniMaxH3-PrefixStream
pip install -r requirements.txt
```

重启 ComfyUI 后，节点会出现在 `MiniMaxH3/PrefixStream` 与 `MiniMaxH3/VideoEdit` 分类下。

---

## 运行单元测试

可通过以下命令进行本地验证：

```bash
python -m unittest discover -s tests -v
node tests/test_clip_bin_frontend.cjs
```

---

## 低内存长视频输出

新增 `MiniMax H3 Disk Video Stream`：逐片段写入磁盘，最后合成 MP4，无需在内存中累计全部画面。需要 FFmpeg，采用硬切拼接；原拼接节点继续提供亮度匹配和接缝淡化。连接与导出步骤见[中文指南](docs/USER_GUIDE_CN.md#磁盘分段长视频低内存)。

## 示例工作流与文档

- 📄 **长视频切片与自动缝合工作流**：[`examples/MiniMaxH3_LongVideo_Slicer_Reassembler_Template.json`](examples/MiniMaxH3_LongVideo_Slicer_Reassembler_Template.json)
  - 专为长视频重绘与连续接力打造的轻量工作流，包含智能切片器、上一段参考帧连线、VHS 规格读取与自动回填缝合。
- 📄 **PrefixStream 连续生成工作流**：[`examples/MiniMaxH3_PrefixStream_v1.0.json`](examples/MiniMaxH3_PrefixStream_v1.0.json)
  - 将这份完整的 42 节点 JSON 直接拖入 ComfyUI 画布，即可运行 Native Masked AV 续写长视频生成。
- 📖 **中文详细使用指南**：[`docs/USER_GUIDE_CN.md`](docs/USER_GUIDE_CN.md)
  - 包含长视频切片器、参考帧输出连线、硬件建议、参数调节与常见问题。

---

## 致谢与参考项目

本项目的诞生与时空续写设计深受以下开源项目与作者的启发，特此致以诚挚的感谢与敬意：

1. **[ComfyUI-H3-Motion-Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context)** by **[@NikoDemon80](https://github.com/NikoDemon80)**
   - 感谢 NikoDemon80 在 MiniMax H3 关键帧锚定算法、VAE 时空周期相位网格对齐公式（Snap to Run Grid）、音视频头部裁切及 5/3 音视频时间缩放比例方面的先驱性数学探索与启发。

2. **[Herrgotts-H3-Infinite-Continuation-Suite](https://github.com/HerrgottMargott/Herrgotts-H3-Infinite-Continuation-Suite)** by **[@HerrgottMargott](https://github.com/HerrgottMargott)**
   - 感谢 Native Masked AV 的原生分流遮罩、精确 AV 上下文边界及独立音频保护方案。

---

## 许可证

本项目采用 [MIT 许可证](LICENSE) 发布。
