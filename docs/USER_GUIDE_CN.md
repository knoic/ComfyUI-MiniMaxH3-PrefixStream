# ComfyUI-MiniMaxH3-PrefixStream 使用说明指南

本插件专为 **MiniMax H3** 长视频续写设计。默认采用 **Native Masked AV**：把上一段音视频 Latent 的规范尾部直接复制到新目标开头，并通过 ComfyUI 原生、彼此独立的 video/audio denoise mask 保护；`Safe Native` 保留为关键帧续写兼容方案。

---

## 一、 快速安装

1. 打开终端，进入你的 ComfyUI 插件目录：
   ```bash
   cd ComfyUI/custom_nodes
   ```
2. 克隆本仓库：
   ```bash
   git clone https://github.com/knoic/ComfyUI-MiniMaxH3-PrefixStream.git
   ```
3. 重启 ComfyUI。
4. 启动后，在节点列表中搜索或右键找到分类：`MiniMaxH3/PrefixStream`。

---

## 二、 核心节点详解

### 1. `MiniMax H3 Continuation Config`（续写配置节点）
只提供创作者可见的视频续写参数：续写方案和上下文帧数。

| 参数项 | 可选值 / 默认值 | 推荐配置与说明 |
| :--- | :--- | :--- |
| **`cache_mode`** | `Native Masked AV (Recommended)`<br>`Safe Native (Fallback)`<br>*(默认 `Native Masked AV`)* | `Native Masked AV` 直接复制并原生保护上一段 AV Latent，不修改 DiT；视频和音频 mask 相互独立。`Safe Native` 使用原生 Attention 与关键帧条件续写，供旧工作流兼容。 |
| **`continuation_frames`** | `39` / `90` / `141` / `192`<br>*(默认 `39`)* | 用户可见的视频续写上下文帧数。39 帧约 1.625 秒，对应 12 个视频 Latent step 与 65 个音频 Latent tick；上下文不能占满整个目标。 |

---

### 2. `MiniMax H3 Continuation Applier`（续写应用节点）
连接在模型、条件与采样 Latent 之间。默认模式会把上一段 AV Latent 的规范尾部复制到 `target_latent` 开头，创建独立 video/audio `noise_mask`，并清理与保护区冲突的开头关键帧。

* **连接方式（关键插槽）**：
  * `model`：连接自模型加载或调度器的输出端。
  * `conditioning`：**必须连接自正面提示词条件输出（如 `MiniMaxH3ReferenceToVideo` 的 `positive`）**。
  * `cache_config`：连接自 `MiniMax H3 Continuation Config`。
  * `target_latent`：连接 `MiniMaxH3ReferenceToVideo` 输出的目标 `LATENT`。
  * `context_latent`：连接上一段完整的 H3 音视频 Latent（支持 NestedTensor 自动解包）。
* **输出**：
  * `model`：原生模型，不安装 DiT Hook。
  * `conditioning`：已移除保护区内冲突关键帧的条件，输入给采样器。
  * `session`：当前长视频生成会话对象，用于传递给裁切节点、缝合节点或监视器。
  * `masked_latent`：连接采样器的 `latent_image`；其中已包含原生音视频 denoise mask。

---

### 3. `MiniMax Trim Prefix Latent (Auto-Crop)`（自动剔除前缀重复帧节点 - 音视频统一）
**【彻底告别手动剪映/PR裁剪】** 专用于一段一段导出独立 MP4 的场景。
续写生成的视频开头必然包含前置过渡重叠帧。接入本节点后，节点将**在统一潜空间层面同时对画面 Latent 和音频 Latent 毫秒级无损裁切**：
- **输入**：`latent`（采样器输出的原生 unified LATENT，内置 NestedTensor 视频+音频）、`session` 或 `cache_config`。
- **输出**：`trimmed_latent`（纯净新切片统一 Latent，直接单线同时接入 `VAEDecode` 和 `VAEDecodeAudio`，0 帧重复回放，音画绝对对齐！）。
- **附加优势**：直接在 Latent 上裁除，使得 VAE Decode 少解码 20%~30% 图像与音频，显著节省显存并加快解码。

---

### 4. `MiniMax Long Video Stitcher`（无缝缝合与长视频合成节点 - 音画双轨 S 曲线）
**【全自动长视频拼接】** 用于将 Clip 1 与 Clip 2 在潜空间直接合成连续超长视频的场景。
- **输入**：`current_latent`、`previous_latent`、`session`、`cache_config`。
- **输出**：
  - `stitched_latent`：**统一无缝长视频 Latent**。自动将两段视频与音频的重合区在潜空间用平滑余弦 S 曲线混合消除硬缝，单线直接接 `VAEDecode` 与 `VAEDecodeAudio`，一键生成无缝超长大片！
  - `trimmed_current_latent`：当前切片的纯净新内容 Latent（音画合一）。

---

### 5. `MiniMax Cache Telemetry Monitor`（遥测监视器）
连接 `session`，输出当前续写模式、切片序号与实际裁切帧数，可在 ComfyUI 中接 `ShowText` 实时查看。

---

### 6. `MiniMax Save AV Latent`（独立联合音画 Latent 保存节点）
直接将采样器输出的 MiniMax H3 联合音画 Latent 以原生 safetensors 格式保存至输出目录，携带切片序号与帧数元数据，无需安装第三方续写扩展套件。

---

### 7. `MiniMax Load AV Latent`（独立联合音画 Latent 加载节点）
加载历史切片的联合音画 Latent，支持指定切片索引（如加载第 1 段用于为第 2 段提供 Rolling 上下文），或自动载入最新生成的切片文件。

---

### 8. `MiniMax H3 Clip Bin Saver`（素材箱智能归档节点 - 告别文件盲盒）
**【非线性剪辑素材库体系】** 针对“事后根本不知道哪一个 Latent 对应哪一个视频”设计的全新工程化媒体池归档节点：
- **核心能力**：
  - **自包含资产打包**：保存潜变量的同时，若连接了 `images`（来自 `VAEDecode`），自动截取**第 0 帧（角色锚点）**与**末尾交接帧**，生成高质首尾双联预览图 `preview.png` 与单帧卡片。在操作系统文件夹中直接大图可见！
  - **星标与分镜打标**：支持设置 `rating`（⭐1~5 星打分）与 `shot_tag`（如“雨夜拔刀”、“Take 2”），方便事后批量过滤废案。
  - **双向血缘与视频索引**：支持记录关联的 MP4 视频文件名 `video_file_name` 与父镜头 ID `parent_clip_id`，彻底告别错位。
  - **UI 即时预览**：节点执行后，直接在 ComfyUI 画布节点面板上渲染出首尾双联预览大图！

---

### 9. `MiniMax H3 Clip Bin Picker`（素材箱画廊选择器 - 零显存末帧即显）
**【可视化镜头挑选与接力】** 彻底废除手动打字与翻找序号的传统方式：
- **核心能力**：
  - **多维筛选与工程管理**：支持按项目 `project_name` 分组管理，支持按星级快速过滤（如只看 `⭐⭐⭐⭐+ (4+ ⭐)`），一键屏蔽废片。
  - **自动接力模式**：输入 `latest`（默认），自动选用本工程中最新符合星级标准的优质镜头，配合批处理队列实现全自动链式生成。
  - **零显存末帧即显 (`tail_frame`)**：直接输出上一段视频的最后一帧图像（`IMAGE` 端口），并在节点表面即时展示！**创作者无需再次消耗显存调用 VAE 解码器**，即可一眼核对续写起点画面，并可直接把该图片拉给后续节点作参考图！


---

## 三、 实战工作流指南

### 场景一：首段正常生成
首段没有历史上下文时，`target_latent` 会原样通过，不创建保护前缀；按普通 MiniMax H3 文生视频、图生视频或首尾帧流程采样即可。

### 场景二：无限多段长视频连续续写（Infinite Video Chaining）
适用于需要连续生成 10 秒、30 秒、1 分钟甚至更长故事视频的场景：
1. **第 1 切片（生成 0~5.16 秒）**：
   - 正常文生视频或图生视频。
   - 保存输出的 unified Latent。
2. **第 2 切片（续写 5~10 秒）**：
   - 将第 1 切片完整 AV Latent 接入 `context_latent`。
   - 将新一段空目标接入 `target_latent`，再把 `masked_latent` 接入采样器。
   - 默认保护 39 帧视频上下文和与之对齐的音频；采样完成后裁掉重复头部。
3. **第 3~N 切片**：
   - 始终把上一切片的完整 AV Latent 传入 `context_latent`，滚动向前推进。

---

## 四、 常见问题与避坑指南 (FAQ)

### Q1: 为什么只显示视频帧数？
* `continuation_frames` 是你在导出视频中看到的帧数；无需了解模型内部的 latent 时间步。
* 更新插件后必须完全重启 ComfyUI 后端，并在浏览器中按 `Ctrl + F5` 强制刷新，才能加载精简后的节点界面。

### Q2: 为什么画面和音频的 Latent 不用分开连线？
* **MiniMax H3 的原生特性**：MiniMax H3 与传统单模态视频模型不同，它在生成时是音视频联合生成的，ComfyUI 官方将其打包为 `NestedTensor((video, audio))` 封装在同一个 `LATENT`（粉色端口）中。
* **潜空间同步缝合/裁切**：本插件的 `TrimPrefixLatent` 与 `LongVideoStitcher` 原生识别并解包该结构，在 Latent 空间中同时按相同的时间比率无损裁切与余弦插值缝合，再原封不动打包为 unified LATENT。
* **极简连线**：无需在缝合前将音频解码为波形，缝合节点输出的单个粉色 `stitched_latent` 端口直接分发连接给 `VAEDecode`（解码画面）与 `VAEDecodeAudio`（解码声音），音画自动严丝合缝、声画对齐！

### Q3: 提示 `CUDA out of memory` (显存溢出) 怎么办？
* 将 `continuation_frames` 调低到 `39`；这是推荐起点。
* 降低目标视频分辨率、时长或采样步数，并关闭其他占用显存的模型与预览。

### Q4: 两段拼接处画面出现微小的闪烁或接缝怎么优化？
* MiniMax H3 原生包含轻微的随机噪声，将 `MiniMax Long Video Stitcher` 中的 `latent_blend_steps` 设置为 `2` 或 `3`，即可利用潜空间余弦插值消除光影跳跃。

### Q5: Native Masked AV 为什么更稳定？
* 上一段 AV Latent 被直接复制到目标开头，保护区 mask 为 0，因此无需让模型重新猜测这些历史内容。
* 视频和音频分别保护，画面交接点与对白尾音可以采用不同长度；这是连续性方案，不宣称 KV 缓存加速。

---

## 五、 致谢与参考项目 (Acknowledgments)

本项目的诞生与时空续写设计深受以下开源项目与作者的启发，特此致以诚挚的感谢与敬意：

1. **[ComfyUI-H3-Motion-Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context)** by **[@NikoDemon80](https://github.com/NikoDemon80)**
   - 感谢 NikoDemon80 在 MiniMax H3 关键帧锚定算法、VAE 时空周期相位网格对齐公式（Snap to Run Grid）、音视频头部裁切及 5/3 音视频时间缩放比例方面的先驱性数学探索与启发。
2. **[Herrgotts-H3-Infinite-Continuation-Suite](https://github.com/HerrgottMargott/Herrgotts-H3-Infinite-Continuation-Suite)** by **[@HerrgottMargott](https://github.com/HerrgottMargott)**
   - 感谢 Native Masked AV 的原生分流遮罩、精确 AV 上下文边界及独立音频保护方案。

