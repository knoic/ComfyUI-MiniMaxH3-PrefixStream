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
用于定义缓存的存储精度、硬件内存分配策略以及双轨窗口长度。

| 参数项 | 可选值 / 默认值 | 推荐配置与说明 |
| :--- | :--- | :--- |
| **`cache_dtype`** | `fp8` / `bf16` / `fp16`<br>*(默认 `fp8`)* | **强烈推荐 `fp8`**（在 24GB 显卡如 RTX 3090/4090 上，50 层 KV 显存占用从 7.8GB 骤降至 **~3.9GB**，肉眼画质无损）；若使用 48GB+ 专业显卡（A100/H100），可直接选 `bf16`。 |
| **`device_mode`** | `auto` / `gpu` / `cpu_pinned`<br>*(默认 `auto`)* | - `auto`：系统空闲显存 > 16GB 时走 GPU 常驻；显存不足时自动降级到 CPU 锁页内存。<br>- `gpu`：速度最快，全驻留显存。<br>- `cpu_pinned`：**零 GPU 显存增量**，通过异步 CUDA Stream 随层预取，杜绝爆显存。 |
| **`use_anchor`** | `True` / `False`<br>*(默认 `True`)* | **核心防漂移开关**。开启后将永久锁定第 0 帧世界原点（面部五官、服装纹理、核心光影），生成 100+ 切片人物也不变形。 |
| **`anchor_latent_frames`** | `1 ~ 10`<br>*(默认 `2`)* | 锚点潜空间步数。2 步潜空间覆盖约 5 帧真实图像，足以承载完整的五官与环境几何。 |
| **`rolling_latent_frames`** | `2 ~ 16`<br>*(默认 `6`)* | 动态运动滑动窗口步数。6 步潜空间覆盖约 19 帧真实图像，保证前后切片镜头运动、肢体动势与光影渐变的连续性。 |

---

### 2. `MiniMax Prefix Cache Applier`（核心注入器与条件增强器）
连接在模型调度链与提示词条件链上，负责在采样器运行前执行单步 Phase 0 预热（提取 KV），并将前缀帧绑定为 `minimax_keyframes` 注入到 `conditioning` 中，同时将 Phase 1 降噪 Hook 注入到模型的 `model_options` 中。

* **连接方式（关键插槽）**：
  * `model`：连接自 `MiniMaxH3SigmaShift` 的输出端。
  * `conditioning`：**必须连接自 `CLIPTextEncode` 的正面提示词输出**。
  * `cache_config`：连接自 `MiniMax Prefix Cache Config`。
  * `context_video_latent` *(可选)*：连接上一段视频尾部输出的 Latent（作为动态 Rolling 上下文）。
  * `anchor_video_latent` *(可选)*：连接初始首帧/参考图像编码后的 Latent（作为永久 Anchor）。
  * `context_audio` *(可选)*：连接上一段视频尾部的音频（实现音视频同步连续性）。
* **输出**：
  * `model`：已挂载极速 Block 级 Hook 的模型，输入给 `KSampler` 的 `model`。
  * `conditioning`：**已注入前置关键帧锚点的条件，输入给 `KSampler` 的 `positive`**。
  * `session`：当前长视频生成会话对象，用于传递给监视器或下一个切片。

---

### 3. `MiniMax Long Video Stitcher`（无缝缝合与音频平滑）
消除切片之间的硬切缝隙感与音频接缝咔哒声。

* **参数**：
  * `latent_blend_steps` *(默认 `2`)*：在潜空间时间轴上执行微小软过渡融合的步数。
  * `audio_crossfade_ms` *(默认 `50`)*：音频 50 毫秒等功率余弦交叉淡入淡出，消除由于相位断层产生的杂音。

---

### 4. `MiniMax Cache Telemetry Monitor`（遥测监视器）
连接 `session`，输出当前显存开销、CPU 搬运占用、切片序号等实时诊断信息，可在 ComfyUI 中接 `ShowText` 实时查看。

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
