# ComfyUI-MiniMaxH3-PrefixStream 使用说明指南

本插件专为 **MiniMax H3（海螺视频 DiT 架构）** 设计，通过在底层解耦时空注意力因果性，实现 **Prefix KV Caching（时空前缀键值缓存复用）**。
在长视频续写与单切片生成中，**消除扩散去噪迭代过程中对已知历史前缀帧的全部冗余 FFN、Norm 与 Linear 计算**，带来约 **40% ~ 48% 的硬件级端到端提速**，并通过**双轨缓存机制（Anchor + Rolling）**彻底解决长距离视频生成中的画质崩塌与人物漂移。

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

### 1. `MiniMax Prefix Cache Config`（缓存策略配置节点）
用于定义缓存的存储精度、硬件内存分配策略以及双轨窗口长度（**以创作者直觉的真实画面帧数填入，无需换算 Latent**）。

| 参数项 | 可选值 / 默认值 | 推荐配置与说明 |
| :--- | :--- | :--- |
| **`cache_mode`** | `Safe Native (Zero Artifacts, Recommended)`<br>`Step-1 Dynamic Cache (Experimental Acceleration)`<br>*(默认 `Safe Native`)* | **模式选择与防闪烁核心**：<br>- `Safe Native`（推荐）：采用 100% 原生 ComfyUI Attention 机制，结合时空网格对齐与关键帧去重，**从数学底层彻底杜绝第 0 帧双重 Latent 冲突导致的剧烈闪烁、抽搐与色偏**，画质与原生生成 100% 一致。<br>- `Step-1 Dynamic Cache`：在线动态提取静态条件帧 KV 缓存以提供加速，并自动排除动态文本 Prompt 避免色调漂移。 |
| **`cache_dtype`** | `fp8` / `bf16` / `fp16`<br>*(默认 `fp8`)* | **强烈推荐 `fp8`**（在 24GB 显卡如 RTX 3090/4090 上，50 层 KV 显存占用从 7.8GB 骤降至 **~3.9GB**，肉眼画质无损）；若使用 48GB+ 专业显卡（A100/H100），可直接选 `bf16`。 |
| **`device_mode`** | `auto` / `gpu` / `cpu_pinned`<br>*(默认 `auto`)* | - `auto`：系统空闲显存 > 16GB 时走 GPU 常驻；显存不足时自动降级到 CPU 锁页内存。<br>- `gpu`：速度最快，全驻留显存。<br>- `cpu_pinned`：**零 GPU 显存增量**，通过异步 CUDA Stream 随层预取，杜绝爆显存。 |
| **`rolling_frames`** | `22` / `5` / `39` / `56` / `73` / `90` / `107` / `124`<br>*(默认 `22`)* | **动态滑动近景窗口（真实物理帧数，严格遵循 VAE 网格）**。默认 **22 帧**（在 24fps 下刚好约 **0.92 秒**，对应 7 步 Latent，起始起点对齐 cycle position 0），负责平滑传承上一段末尾的速度矢量、肢体动势与光照渐变。 |
| **`use_anchor`** | `True` / `False`<br>*(默认 `False`)* | 是否额外保留第一段的首帧锚点。连续片段接力推荐保持 `False`，由 rolling head 平滑接管开端；仅在希望主角服装面部在跨镜头时被永久强绑定时开启。 |
| **`anchor_frames`** | `1 ~ 30`<br>*(默认 `5`)* | **世界原点永久锚点（真实物理帧数）**。默认 **5 帧**（约 0.21 秒，对应 2 步 Latent），承载主角初始五官几何与场景基调。 |

---

### 2. `MiniMax Prefix Cache Applier`（核心注入器与条件增强器）
连接在模型调度链与提示词条件链上，负责在采样器运行前执行单步 Phase 0 预热（提取 KV），并将前缀帧绑定为 `minimax_keyframes` 注入到 `conditioning` 中，同时将 Phase 1 降噪 Hook 注入到模型的 `model_options` 中。

* **连接方式（关键插槽）**：
  * `model`：连接自模型加载或调度器的输出端。
  * `conditioning`：**必须连接自正面提示词条件输出（如 `MiniMaxH3ReferenceToVideo` 的 `positive`）**。
  * `cache_config`：连接自 `MiniMax Prefix Cache Config`。
  * `context_video_latent` *(可选)*：连接上一段视频尾部输出的 Latent（作为动态 Rolling 上下文，支持 NestedTensor 自动解包）。
  * `anchor_video_latent` *(可选)*：连接初始首帧/参考图像编码后的 Latent（作为永久 Anchor）。
  * `context_audio` *(可选)*：连接上一段视频尾部的音频（实现音视频同步连续性）。
* **输出**：
  * `model`：已挂载极速 Block 级 Hook 的模型，输入给采样器（如 `BasicGuider` / `KSampler`）。
  * `conditioning`：**已注入前置关键帧锚点的条件，输入给采样器的 `conditioning`**。
  * `session`：当前长视频生成会话对象，用于传递给裁切节点、缝合节点或监视器。

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
连接 `session`，输出当前显存开销、CPU 搬运占用、切片序号、实际裁切帧数等实时诊断信息，可在 ComfyUI 中接 `ShowText` 实时查看。

---

## 三、 实战工作流指南

### 场景一：单切片加速（图生视频 / 文生视频 40%+ 提速）
适用于每次只渲染一段视频（如 5 秒），但在输入端提供了首帧图像或参考图像的场景：
1. `Load Image` $\to$ `VAEEncode` 得到参考 Latent。
2. 将参考 Latent 接入 `MiniMax Prefix Cache Applier` 的 `anchor_latent` 插槽。
3. `EmptyLatentVideo` 生成目标画幅（如 1280x720，31 步）。
4. 执行采样器。在整个采样中，模型只对生成的 31 步计算 QKV 与 FFN，参考帧仅作为只读 KV 注入注意力层。
5. **实测表现**：计算量直降，且参考帧的角色还原度达到 100% 数学无损。

### 场景二：无限多段长视频连续续写（Infinite Video Chaining）
适用于需要连续生成 10 秒、30 秒、1 分钟甚至更长故事视频的场景：
1. **第 1 切片（生成 0~5.16 秒）**：
   - 正常文生视频或图生视频。
   - 保存输出的 unified Latent。
2. **第 2 切片（续写 5~10 秒）**：
   - 将第 1 切片的首部接入 `anchor_latent`（锁定主角五官外貌）。
   - 将第 1 切片的尾部接入 `context_latent`（引导动作、运镜与声音延续）。
   - 采样器生成速度相比传统全序列计算提升约 **1.8x 倍**。
   - 通过 `MiniMax Long Video Stitcher` 将两段 Latent 缝合，直接接 VAE 解码导出超长无缝视频！
3. **第 3~N 切片**：
   - 保持 `anchor_latent` 始终为 Clip 1 的首帧（主角永不崩塌）。
   - 将上一切片的尾部传入 `context_latent`，滚动向前推进。

---

## 四、 常见问题与避坑指南 (FAQ)

### Q1: 为什么之前滑动窗口参数最大只能填 16 帧？
* **原因**：早期版本参数是以 **Latent 步数**（`rolling_latent_frames`）计量的，由于 MiniMax H3 的时间轴约 4 帧压成 1 步 Latent，16 步 Latent 实际上相当于 **64 实际视频帧**。部分用户误以为只能滑 16 帧（不足 1 秒）。
* **已优化**：现已升级为直观的 **实际视频帧数** `rolling_frames`（范围 **4 ~ 124 帧**，最大可覆盖 124 帧单切片全长！）；对于保留旧节点的用户，兼容项上限也已放开至 64 步。
* **⚠️ 重要提醒**：若界面滑块依然卡在 16，是因为 ComfyUI 只在启动时加载一次 Python 节点定义。**更新插件代码后必须完全重启 ComfyUI 后端服务，并在浏览器中按 `Ctrl + F5` 强制刷新页面**，才能载入最新的 124 帧控件定义。

### Q2: 为什么画面和音频的 Latent 不用分开连线？
* **MiniMax H3 的原生特性**：MiniMax H3 与传统单模态视频模型不同，它在生成时是音视频联合生成的，ComfyUI 官方将其打包为 `NestedTensor((video, audio))` 封装在同一个 `LATENT`（粉色端口）中。
* **潜空间同步缝合/裁切**：本插件的 `TrimPrefixLatent` 与 `LongVideoStitcher` 原生识别并解包该结构，在 Latent 空间中同时按相同的时间比率无损裁切与余弦插值缝合，再原封不动打包为 unified LATENT。
* **极简连线**：无需在缝合前将音频解码为波形，缝合节点输出的单个粉色 `stitched_latent` 端口直接分发连接给 `VAEDecode`（解码画面）与 `VAEDecodeAudio`（解码声音），音画自动严丝合缝、声画对齐！

### Q3: 提示 `CUDA out of memory` (显存溢出) 怎么办？
* **方案 A**：检查 `MiniMax Prefix Cache Config`，确保 `cache_dtype` 设置为 `fp8`（显存占用直接砍半）。
* **方案 B**：将 `device_mode` 从 `gpu` 切换为 `cpu_pinned`。缓存将全部驻留在主机物理内存中，GPU 显存开销降为 0 MB。
* **方案 C**：适当缩减 `rolling_frames`（如设为 24 帧，约 1 秒过渡）。

### Q4: 两段拼接处画面出现微小的闪烁或接缝怎么优化？
* MiniMax H3 原生包含轻微的随机噪声，将 `MiniMax Long Video Stitcher` 中的 `latent_blend_steps` 设置为 `2` 或 `3`，即可利用潜空间余弦插值消除光影跳跃。

### Q5: 为什么这套方案比传统 Inpainting / Mask 续写更快更清晰？
* **传统 Mask 续写**：全序列 Token 都还在 Transformer 里跑完整的 QKV 投影和 SwiGLU FFN，且每次重绘都要加噪去噪，误差随切片数指数累加（第 4 段开始画面容易融化）。
* **Prefix KV Caching**：前缀帧只读冻结，没有重复加噪损耗，且跳过了前缀全部前向矩阵乘法，既快又稳。

---

## 五、 致谢与参考项目 (Acknowledgments)

本项目的诞生与演进离不开开源社区中多位优秀开发者与团队的先驱性探索。特此向以下项目和作者致以由衷的感谢与敬意：

1. **[MiniMax AI (Hailuo Team)](https://www.minimax.io/)**
   - 感谢 MiniMax 团队打造并开源了划时代的 **MiniMax H3** 全模态视音频基座模型，在动作表现力、影视级质感与原生音视频联合生成领域树立了业界标杆。
2. **[ComfyUI](https://github.com/comfyanonymous/ComfyUI)** by **[@comfyanonymous](https://github.com/comfyanonymous)**
   - 感谢 ComfyUI 奠定了模块化生成式 AI 的基石，以及对 MiniMax H3 架构、NestedTensor 视音频联合潜空间、内存管理系统的卓越原生支持。
3. **[ComfyUI-H3-Motion-Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context)** by **[@NikoDemon80](https://github.com/NikoDemon80)**
   - 感谢 NikoDemon80 在 MiniMax H3 关键帧锚定算法、VAE 时空周期相位网格对齐公式（Snap to Run Grid）、音视频头部裁切及 5/3 音视频时间缩放比例方面的先驱性数学探索与启发。
4. **[Herrgotts-H3-Infinite-Continuation-Suite](https://github.com/Herrgotts/Herrgotts-H3-Infinite-Continuation-Suite)** by **[@Herrgotts](https://github.com/Herrgotts)**
   - 感谢 Herrgott 提出的无限长视频链式接力构想、上下文对齐平滑接缝理念、以及音视频无缝拼接工作流的探索。
5. **[ComfyUI-VideoHelperSuite (VHS)](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)** by **[@Kosinkadink](https://github.com/Kosinkadink)**
   - 感谢 VHS 套件在 ComfyUI 社区中为视频加载、组合、编码与视音频复用（Muxing）树立的可靠标准与卓越工具链支持。
6. **[ComfyUI-MiniMaxH3-TimelineDirector](https://github.com/AIGODLIKE/ComfyUI-MiniMaxH3-TimelineDirector)** by **[AIGODLIKE](https://github.com/AIGODLIKE)**
   - 感谢其在多镜头剧本式时间线规划、分镜推进与参考图/提示词协同调度方面的思路启发。
7. **[TE-Speed-MiniMaxH3](https://github.com/AIGODLIKE/TE-Speed-MiniMaxH3)** / 社区加速探索
   - 感谢社区早期在 MiniMax H3 推理加速方面的探索，促使我们深入 DiT 注意力层机制并研发出原生的 Prefix KV Caching 解决方案。

