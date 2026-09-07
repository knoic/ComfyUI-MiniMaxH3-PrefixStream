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
| **`cache_dtype`** | `fp8` / `bf16` / `fp16`<br>*(默认 `fp8`)* | **强烈推荐 `fp8`**（在 24GB 显卡如 RTX 3090/4090 上，50 层 KV 显存占用从 7.8GB 骤降至 **~3.9GB**，肉眼画质无损）；若使用 48GB+ 专业显卡（A100/H100），可直接选 `bf16`。 |
| **`device_mode`** | `auto` / `gpu` / `cpu_pinned`<br>*(默认 `auto`)* | - `auto`：系统空闲显存 > 16GB 时走 GPU 常驻；显存不足时自动降级到 CPU 锁页内存。<br>- `gpu`：速度最快，全驻留显存。<br>- `cpu_pinned`：**零 GPU 显存增量**，通过异步 CUDA Stream 随层预取，杜绝爆显存。 |
| **`use_anchor`** | `True` / `False`<br>*(默认 `True`)* | **核心防漂移开关**。开启后将永久锁定第 0 帧世界原点（面部五官、服装纹理、核心光影），生成 100+ 切片人物也不变形。 |
| **`rolling_frames`** | `4 ~ 124`<br>*(默认 `22`)* | **动态滑动近景窗口（真实物理帧数）**。默认 **22 帧**（在 24fps 下刚好约 **0.92 秒**，对应 7 步 Latent），负责传承上一段末尾的速度矢量、肢体动势与光照渐变。 |
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

### 3. `MiniMax Trim Prefix Latent (Auto-Crop)`（自动剔除前缀重复帧节点）
**【彻底告别手动剪映/PR裁剪】** 专用于一段一段导出独立 MP4 的场景。
续写生成的视频开头必然包含约 1 秒的前置过渡重叠帧。接入本节点后，节点将**在潜空间层面直接无损切除注入的前缀**：
- **输入**：`video_latent`（采样器输出的原生 Latent）、`session` 或 `cache_config`。
- **输出**：`trimmed_video`（纯净新片段，直接连入 `VAEDecode` 导出即是干净的新画面，0 帧重复回放！）。
- **附加优势**：直接在 Latent 上裁除，使得 VAE Decode 少解码 25% 图像，显著节省显存并加快解码。

---

### 4. `MiniMax Long Video Stitcher`（无缝缝合与长视频合成节点）
**【全自动长视频拼接】** 用于将 Clip 1 与 Clip 2 直接合成连续长视频的场景。
- **输入**：`current_video`、`previous_video`、`session` 等。
- **输出**：
  - `stitched_video`：**无缝长视频**。自动将两段视频的重合区在潜空间用平滑余弦 S 曲线混合消除硬缝，一键生成无缝超长大片！
  - `trimmed_current_video`：当前切片的纯净新内容。
  - `stitched_audio`：经过 50ms 等功率立体声淡入淡出后的拼接音频。
  - `trimmed_current_audio`：纯净新切片音频。

---

### 5. `MiniMax Cache Telemetry Monitor`（遥测监视器）
连接 `session`，输出当前显存开销、CPU 搬运占用、切片序号、实际裁切帧数等实时诊断信息，可在 ComfyUI 中接 `ShowText` 实时查看。

---

## 三、 实战工作流指南

### 场景一：单切片加速（图生视频 / 文生视频 40%+ 提速）
适用于每次只渲染一段视频（如 6 秒），但在输入端提供了首帧图像或参考图像的场景：
1. `Load Image` $\to$ `VAEEncode` 得到参考 Latent。
2. 将参考 Latent 接入 `MiniMax Prefix Cache Applier` 的 `anchor_video_latent` 插槽。
3. `EmptyLatentVideo` 生成目标画幅（如 1280x720，37 步）。
4. 执行 `KSampler`。在整个 25 步采样中，模型只对生成的 37 步计算 QKV 与 FFN，参考帧仅作为只读 KV 注入注意力层。
5. **实测表现**：计算量直降，且参考帧的角色还原度达到 100% 数学无损。

### 场景二：无限多段长视频连续续写（Infinite Video Chaining）
适用于需要连续生成 10 秒、30 秒、1 分钟甚至更长故事视频的场景：
1. **第 1 切片（生成 0~6 秒）**：
   - 正常文生视频或图生视频。
   - 将生成的输出 Latent 命名为 `Clip1_Latent`。
2. **第 2 切片（续写 6~12 秒）**：
   - 将 `Clip1_Latent` 的首部接入 `anchor_video_latent`（锁定角色）。
   - 将 `Clip1_Latent` 的尾部接入 `context_video_latent`（引导动作延续）。
   - 运行采样器，生成速度相比从头计算全长序列提升约 **1.8x 倍**。
   - 通过 `MiniMax Long Video Stitcher` 将 Clip 1 与 Clip 2 拼接。
3. **第 3~N 切片**：
   - 保持 `anchor_video_latent` 始终为 Clip 1 的首帧（角色不崩塌）。
   - 将上一切片的尾部传入 `context_video_latent`，滚动向前推进。

---

## 四、 常见问题与避坑指南 (FAQ)

### Q1: 提示 `CUDA out of memory` (显存溢出) 怎么办？
* **方案 A**：检查 `MiniMax Prefix Cache Config`，确保 `cache_dtype` 设置为 `fp8`（显存占用直接砍半）。
* **方案 B**：将 `device_mode` 从 `gpu` 切换为 `cpu_pinned`。缓存将全部驻留在主机物理内存中，GPU 显存开销降为 0 MB。
* **方案 C**：适当缩减 `rolling_latent_frames`（如从 8 调至 4~6）。

### Q2: 两段拼接处画面出现微小的闪烁或接缝怎么优化？
* MiniMax H3 原生包含轻微的随机噪声，将 `MiniMax Long Video Stitcher` 中的 `latent_blend_steps` 设置为 `2` 或 `3`，即可利用潜空间余弦插值消除光影跳跃。

### Q3: 为什么这套方案比传统 Inpainting / Mask 续写更快更清晰？
* **传统 Mask 续写**：全序列 Token 都还在 Transformer 里跑完整的 QKV 投影和 SwiGLU FFN，且每次重绘都要加噪去噪，误差随切片数指数累加（第 4 段开始画面容易融化）。
* **Prefix KV Caching**：前缀帧只读冻结，没有重复加噪损耗，且跳过了前缀全部前向矩阵乘法，既快又稳。
