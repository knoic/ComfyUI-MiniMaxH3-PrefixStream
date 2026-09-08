# ComfyUI-MiniMaxH3-PrefixStream

[![GitHub License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-green.svg)]()
[![Model](https://img.shields.io/badge/model-MiniMax--H3-orange.svg)]()

一个用于 ComfyUI MiniMax H3 的原生音视频续写与长视频流式生成节点套件。

默认的 **Native Masked AV** 路径参考了 [Herrgott's H3 Infinite Continuation Suite](https://github.com/HerrgottMargott/Herrgotts-H3-Infinite-Continuation-Suite)：它将上一段的音视频 latent 复制到新目标的开头，并通过 ComfyUI 原生去噪遮罩独立保护视频与音频。`Safe Native` 保留为基于关键帧的兼容性备选方案。

---

## 核心特性

- 🛡️ **默认使用 Native Masked AV**：将干净的上一段视频/音频 latent 直接复制到下一段的开头并原地保护；无需修改 DiT，也不使用实验性的 KV 注入。
- 🎯 **精确的联合音视频边界**：使用 MiniMax H3 兼容的 `39 / 90 / 141 / 192 / ...` 帧网格。默认 39 帧上下文约为 1.625 秒，恰好对应 65 个音频 latent tick。
- 🎵 **独立的音频保护**：视频和音频分别使用原生遮罩。对白可完整保留上一段音频尾部，并可选用仅影响音频的羽化，使最后的保护 tick 平滑释放。
- 🔒 **避免条件冲突**：移除受保护视频开头内的关键帧，防止第 0 帧引导与复制的 latent 上下文相互冲突；之后的端点和参考条件仍可正常使用。
- 🧩 **Safe Native 兼容备选**：对于不能使用 masked latent 的工作流，仍可选择早期的原生注意力/关键帧续写路径。
- ✂️ **便于无缝拼接的输出**：会话元数据驱动精确的重复开头裁切，支持像素空间视频裁切和同步音频处理，便于最终合成。

---

## 快速浏览

### 一个节点选择续写方案

`MiniMax H3 Continuation Config` 只保留必要的核心选择：普通续写使用 **Native Masked AV (Recommended)**；只有工作流需要兼容方案时才切换为 **Safe Native**。`continuation_frames` 用于选择 H3 帧网格上的受保护重叠长度。

![MiniMax H3 续写配置节点](assets/readme/continuation-config.png)

### 从素材箱浏览并继续镜头

`MiniMax H3 Clip Bin Picker` 将已保存的片段以可视化画廊呈现。选择上一镜头、按评分筛选，并将其 latent 与尾帧作为下一段的续写上下文，无需在输出目录中翻找文件。

![MiniMax H3 素材箱画廊选择器](assets/readme/clip-bin-picker-gallery.png)

---

## 工作原理

```text
上一段完整音视频 latent ──► 选择规范的 39/90/141/... 帧尾部
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
| **`MiniMax H3 Continuation Config`** | `MiniMaxH3/PrefixStream` | 选择默认的 `Native Masked AV` 或备选的 `Safe Native`，并设置用户可见的视频上下文长度。 |
| **`MiniMax H3 Continuation Applier`** | `MiniMaxH3/PrefixStream` | 构建原生视频/音频遮罩并输出 `masked_latent`；在备选模式下应用避免冲突的关键帧条件。 |
| **`MiniMax Trim Prefix`** | `MiniMaxH3/PrefixStream` | 在像素和音频波形空间裁切开头的重叠帧，避免 VAE 因果闪烁并保持同步。 |
| **`MiniMax Long Video Stitcher`** | `MiniMaxH3/PrefixStream` | 在像素空间无缝合并视频（含亮度增益匹配）和音频波形（等功率交叉淡化）。 |
| **`MiniMax Save AV Latent`** | `MiniMaxH3/PrefixStream` | 独立地将联合音视频 latent 保存为 safetensors，无外部依赖。 |
| **`MiniMax Load AV Latent`** | `MiniMaxH3/PrefixStream` | 独立加载带有元数据的联合音视频 latent，用于多片段连续流式生成。 |
| **`MiniMax H3 Clip Bin Saver`** | `MiniMaxH3/PrefixStream` | 连同预览图、评分、镜头标签、提示词和续写血缘归档生成片段。 |
| **`MiniMax H3 Clip Bin Picker`** | `MiniMaxH3/PrefixStream` | 以画廊方式查找已保存的片段，并输出其 latent、尾帧、提示词和 ID。 |
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

重启 ComfyUI 后，节点会出现在 `MiniMaxH3/PrefixStream` 分类下。

---

## 运行单元测试

可通过以下命令进行本地验证：

```bash
python tests/test_cache_manager.py
python tests/test_fused_attention.py
python tests/test_nodes_and_pipeline.py
python tests/test_native_masked_av.py
```

---

## 示例工作流与文档

- 📄 **可直接使用的工作流**：[`examples/MiniMaxH3_PrefixStream_v1.0.json`](examples/MiniMaxH3_PrefixStream_v1.0.json)
  - 将这份完整的 42 节点 JSON 直接拖入 ComfyUI 画布，即可运行 PrefixStream 长视频生成。
- 📖 **中文详细使用指南**：[`docs/USER_GUIDE_CN.md`](docs/USER_GUIDE_CN.md)
  - 包含硬件建议、参数调节、多片段无限续写流程与常见问题。

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
