# ComfyUI-MiniMaxH3-PrefixStream 中文使用指南

本插件用于将 MiniMax H3 的多段生成组织为稳定的长视频续写流程。每次生成后，片段可以归档到可视化 **Clip Bin** 素材箱；下一次从画廊选取任意历史镜头作为上下文，即可继续生成或创建新的分支。

默认方案为 **Native Masked AV**：将上一段音视频 latent 的规范尾部复制到下一段的开头，并用 ComfyUI 原生视频/音频去噪遮罩分别保护。`Safe Native` 是短片段或不具备原生 AV-mask 支持时的兼容性备选。

---

## 一、安装与前提

1. 将本仓库放入 ComfyUI 的 `custom_nodes` 目录：

   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/knoic/ComfyUI-MiniMaxH3-PrefixStream.git
   cd ComfyUI-MiniMaxH3-PrefixStream
   pip install -r requirements.txt
   ```

2. 使用包含 MiniMax H3 音视频遮罩支持的新版 ComfyUI。`Native Masked AV` 依赖 ComfyUI 的 H3 AV-mask 实现；若运行时提示缺少支持，请先更新 ComfyUI，再完全重启后端。
3. 重启 ComfyUI，并在浏览器中按 `Ctrl + F5` 强制刷新前端。节点位于 `MiniMaxH3/PrefixStream` 和 `MiniMaxH3/ClipBin` 分类。

---

## 二、核心概念

```text
上一段完整 AV latent → 取规范尾部作为上下文
                                  │
新的 H3 目标 latent ───────────────┼→ Continuation Applier
                                  │       │
                                  │       ▼
                                  │   原生 H3 采样
                                  │       │
                                  │       ▼
                             解码 → Trim Prefix 裁切重叠开头 → 保存 / 合成
```

- **上下文**：上一段采样器输出的完整联合音视频 `LATENT`。MiniMax H3 的视频和音频封装在同一个粉色 latent 端口内，无需拆成两根线。
- **目标**：新一段由原生 H3 工作流创建的目标 `LATENT`，通常来自 `MiniMaxH3ReferenceToVideo`。
- **重叠开头**：为保证连续性，下一段开头会保留上一段的上下文。采样后用 `MiniMax H3 Trim Prefix` 裁掉这部分，再进行成片保存或外部合成。
- **素材箱**：保存片段的 latent、预览图、镜头标签、评分、提示词和父镜头 ID。它是续写来源的可视化选择器，而非视频拼接器。

---

## 三、节点说明

### 1. `MiniMax H3 Continuation Config`

这是唯一需要为续写策略做选择的配置节点。

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `cache_mode` | `Native Masked AV (Recommended)` | 默认方案。使用 ComfyUI 原生视频/音频遮罩保护复制到目标开头的上下文。`Safe Native (Fallback)` 用于兼容工作流或原生遮罩不可用的情形。 |
| `continuation_frames` | `39` | 可选择 `39`、`90`、`141` 或 `192`。这是视频中可见的重叠上下文长度；39 帧约为 1.625 秒。 |

推荐先使用默认的 `Native Masked AV + 39`。上下文越长，连续性参考越多，但目标片段必须有足够长度容纳它。

### 2. `MiniMax H3 Continuation Applier`

将它插入原生 H3 的模型、条件和采样 latent 链路中。

| 插槽 | 连接方式 |
| :--- | :--- |
| `model` | 原生 H3 模型输出。 |
| `conditioning` | 原生正向条件输出，例如 `MiniMaxH3ReferenceToVideo` 的正向条件。 |
| `cache_config` | 连接 `MiniMax H3 Continuation Config`。 |
| `context_latent` | 上一段完整的 H3 联合音视频 latent；首段生成时留空。可连接 Clip Bin Picker 的 `latent` 输出。 |
| `target_latent` | 新一段原生 H3 目标 latent，通常来自 `MiniMaxH3ReferenceToVideo`。 |

输出连接方式：

- `model`、`conditioning`：分别接回原生采样链路。
- `masked_latent`：接采样器的 `latent_image` 输入。
- `session`：接 `MiniMax H3 Trim Prefix`；也可以接 `MiniMax H3 Cache Telemetry Monitor` 查看本次保护的实际上下文。

首段没有 `context_latent` 时，节点会直接传递目标 latent，不创建保护前缀。若 Native Masked AV 的源片段或目标片段过短，节点会自动回退至 `Safe Native`，并在 ComfyUI 控制台记录原因。

### 3. `MiniMax H3 Trim Prefix (AV Master, Zero Flicker)`

续写片段采样完成后，用它移除重复的开头。推荐在**解码后的图像和音频波形空间**裁切，可避免再次 VAE 解码导致的画面闪烁或色彩偏差。

| 输入 | 建议 |
| :--- | :--- |
| `images` | 连接当前片段完整解码后的画面。 |
| `audio` | 连接当前片段完整解码后的音频。 |
| `session` | 连接 Continuation Applier 的 `session` 输出；`trim_frames=0` 时自动使用本次实际保护帧数。 |
| `trim_frames` | 保持 `0` 即可自动裁切；只有脱离会话单独使用时才手动填写。 |
| `latent` | 可选。需要裁切 latent 时再连接；不要把它当作像素/音频裁切的替代品。 |

输出 `trimmed_images` 和 `trimmed_audio` 用于保存或交给视频合成节点。`trimmed_latent` 仅在输入了 `latent` 时才有内容。

### 4. `MiniMax H3 Clip Bin Saver (Media Pool)`

将一段生成结果归档到项目素材箱，供之后视觉化挑选和接力。

- 必填：`latent`、`project_name`、`shot_tag`、`rating`。
- `shot_tag` 是自由文本标签，例如“雨夜登场”或“Take 2”；保留 `Auto (自动编号)` 会生成 `Shot 1`、`Shot 2` 等自动编号。
- 建议连接 `images` 和可选的 `audio`，让素材箱生成首尾预览卡片并可自动编码封装视频。
- `video_file_name`：连接合成保存节点（如 `VHS_VideoCombine`）的 Filenames 输出，系统会自动将实际生成的 MP4 视频复制归档到该镜头的素材包中。
- `save_video`：默认为 `True`。开启后若传入了视频文件名则自动归档，若未传入但输入了 `images`，将自动编码生成 `video.mp4`，让素材包成为真正独立、包含潜空间和成品视听的自包含媒体池。
- 可选连接 `prompt`、`parent_clip_id`，记录提示词与多分支承接血缘。

要让下一段正常续写，请保存采样器输出的**完整**联合 AV latent；预览图则可以使用裁切后的 `trimmed_images`。

### 5. `MiniMax H3 Clip Bin Picker (Gallery Loader)`

以画廊方式浏览和载入素材箱中的片段，支持全动态视听交互。

- `project_name`：选择项目素材箱。
- `mode`：`Auto` 在素材箱为空时作为首段生成；有素材时自动接力。`Force Initial` 强制开始新首段；`Strict Chaining` 在没有可用来源时直接报错。
- `filter_rating`：按星级过滤候选镜头。
- `clip_selection`：使用 `latest` 自动选择最新片段，或填写 `clip_id` / `shot_tag` 来选择指定历史镜头。
- **全新动态视听交互**：
  - **悬停动效预览（Hover-to-Play）**：鼠标移入带有视频的卡片，自动静音循环播放动态预览，移出自动还原；
  - **全功能声画视听弹窗（Modal Player）**：点击卡片角标 `▶ MP4` 或画面中央播放按钮（亦可双击卡片），弹出视听播放弹窗，支持声音控制、时间轨拖拽、查看完整提示词与镜头参数，并可一键「设为当前接力源」；
- 输出 `latent` 接 Continuation Applier 的 `context_latent`；`tail_frame`、`first_frame` 和 `prompt` 可作为下一段的参考或提示词素材；`clip_id` 接下一次 Clip Bin Saver 的 `parent_clip_id`。

### 6. 其他辅助节点

| 节点 | 用途 |
| :--- | :--- |
| `MiniMax H3 Cache Telemetry Monitor` | 读取 `session` 并输出当前模式、片段序号和实际保护帧数。 |
| `MiniMax H3 Save AV Latent (Standalone)` | 将完整 H3 联合 AV latent 保存为 safetensors。 |
| `MiniMax H3 Load AV Latent (Standalone)` | 从历史 safetensors 载入联合 AV latent。 |
| `MiniMax H3 Safe VAE Decode (Video)` | 在首段模式下可安全处理空的上文 latent。 |
| `MiniMax H3 Safe VAE Decode (Audio)` | 在首段模式下可安全处理空的上文 latent。 |

---

## 四、推荐工作流

### 首段生成

1. 保持原有 MiniMax H3 模型、提示词、目标 latent、采样和解码节点。
2. 将 `Continuation Config` 设为 `Native Masked AV (Recommended)` 和 `39`。
3. `Continuation Applier` 的 `context_latent` 留空；将原生目标 latent 接到 `target_latent`，再将 `masked_latent` 接采样器。
4. 采样完成后解码完整画面与音频；将它们以及 `session` 接到 `Trim Prefix`。首段没有重叠，裁切节点会保持内容不变。
5. 将采样器完整 latent 与裁切后的预览画面接到 `Clip Bin Saver`，保存第一段。

### 续写下一段或创建分支

1. 在 `Clip Bin Picker` 中选择项目、评分筛选和要接力的镜头；使用 `latest` 可自动选择最新片段。
2. 将 Picker 的 `latent` 接到 Applier 的 `context_latent`，将 `clip_id` 接到 Saver 的 `parent_clip_id`。
3. 继续使用原生 H3 工作流产生新的 `target_latent`，并连接到 Applier 的 `target_latent`。
4. 采样后按首段相同方式解码，再使用 Trim Prefix 依照 `session` 自动裁掉重叠开头。
5. 归档新片段。之后可从任意历史片段再次接力，因此可以形成分支，而非只能线性续写。

最终视频的合成可使用你现有的 ComfyUI 视频合成节点或剪辑软件；本插件负责为每一段输出已裁掉重叠开头且音画同步的内容。

---

## 五、常见问题

### 为什么配置节点只有两个选项？

这是有意简化的界面。创作者只需决定续写方案与可见视频上下文长度；缓存 dtype、设备、锚帧等旧版实现细节不再暴露。

### 为什么 `shot_tag` 是文本输入框？

它是镜头的自由标签，供素材箱检索与分镜备注使用，而不是固定编号。无需手填时，保留 `Auto (自动编号)` 即可。

### 为什么 Native Masked AV 报缺少支持？

请更新 ComfyUI 至包含 MiniMax H3 AV-mask 支持的版本，完全重启后端并按 `Ctrl + F5` 刷新浏览器。短源片段或短目标片段则会自动使用 `Safe Native`。

### 为什么不需要分别连接视频和音频 latent？

MiniMax H3 在 ComfyUI 中使用联合 AV `LATENT` 封装视频和音频。续写应用节点会在内部拆分、处理并重新封装；用户只需连接一个 latent 端口。

### 如何减少显存占用？

从 `39` 帧上下文开始，按需要降低分辨率、目标时长或采样步数，并避免同时加载不使用的模型。

---

## 六、致谢与参考项目

1. **[ComfyUI-H3-Motion-Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context)** by **[@NikoDemon80](https://github.com/NikoDemon80)**
   - 感谢其在 MiniMax H3 关键帧锚定、VAE 时空相位网格对齐、音视频开头裁切及时间缩放方面的探索与启发。
2. **[Herrgotts-H3-Infinite-Continuation-Suite](https://github.com/HerrgottMargott/Herrgotts-H3-Infinite-Continuation-Suite)** by **[@HerrgottMargott](https://github.com/HerrgottMargott)**
   - 感谢其在 Native Masked AV、精确音视频上下文边界和独立音频保护方面提供的思路。
