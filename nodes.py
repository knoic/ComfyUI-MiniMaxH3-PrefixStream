"""ComfyUI Custom Nodes for MiniMax H3 Prefix KV Caching & Streaming Chaining.

Exposes:
- MiniMaxPrefixCacheConfig: Configure FP8/BF16, GPU/CPU Pinned, Anchor & Rolling parameters in real video frames.
- MiniMaxPrefixCacheApplier: Connects model, historical context latents, and hooks into KSampler.
- MiniMaxTrimPrefixLatent: Automatically trims leading overlap frames from generated clips (AV unified).
- MiniMaxLongVideoStitcher: Smoothly stitches video and audio latents between clips in unified LATENT space.
- MiniMaxCacheMonitor: Real-time diagnostics for VRAM, memory footprint, and session progress.
"""

import os
import json
import logging
from typing import Dict, Any, Tuple, Optional
import torch

try:
    from safetensors.torch import load_file as st_load, save_file as st_save
except ImportError:
    try:
        import safetensors.torch
        st_load = safetensors.torch.load_file
        st_save = safetensors.torch.save_file
    except ImportError:
        st_load = None
        st_save = None

logger = logging.getLogger("minimax_prefix_stream")

try:
    from .engine.cache_manager import (
        KVCacheConfig,
        PrefixKVCacheManager,
        pixel_frames_to_latent_steps,
        latent_steps_to_pixel_frames,
    )
    from .pipeline.long_video_director import LongVideoSession
    from .pipeline.seam_protector import (
        audio_equal_power_crossfade,
        latent_soft_blend,
        stitch_video_latents,
        stitch_audio_latents,
        trim_prefix_frames,
        trim_audio_latents,
        trim_audio_waveform,
        stitch_video_images,
        stitch_audio_waveforms,
        trim_images_and_audio,
        estimate_luminance_gain,
        apply_luminance_gain_fade,
    )
    from .engine.clip_bin_manager import (
        save_clip_asset,
        load_clip_asset,
        get_project_dir,
        list_projects,
        load_project_index,
        get_clips_for_selection,
        format_clip_label,
        pil_to_tensor,
        create_placeholder_card,
    )
except (ImportError, ValueError):
    from engine.cache_manager import (
        KVCacheConfig,
        PrefixKVCacheManager,
        pixel_frames_to_latent_steps,
        latent_steps_to_pixel_frames,
    )
    from pipeline.long_video_director import LongVideoSession
    from pipeline.seam_protector import (
        audio_equal_power_crossfade,
        latent_soft_blend,
        stitch_video_latents,
        stitch_audio_latents,
        trim_prefix_frames,
        trim_audio_latents,
        trim_audio_waveform,
        stitch_video_images,
        stitch_audio_waveforms,
        trim_images_and_audio,
        estimate_luminance_gain,
        apply_luminance_gain_fade,
    )
    from engine.clip_bin_manager import (
        save_clip_asset,
        load_clip_asset,
        get_project_dir,
        list_projects,
        load_project_index,
        get_clips_for_selection,
        format_clip_label,
        pil_to_tensor,
        create_placeholder_card,
    )





class MiniMaxPrefixCacheConfigNode:
    """Configures Prefix KV Cache precision, memory strategy, and window sizes in real video frames."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cache_mode": ([
                    "Safe Native (Zero Artifacts, Recommended)",
                    "Step-1 Dynamic Cache (Experimental Acceleration)",
                    "Decoupled Pure Prefix (Zero Overlap, Prompt-Aligned)"
                ], {
                    "default": "Safe Native (Zero Artifacts, Recommended)",
                    "tooltip": "模式选择: Safe Native 采用 100% 原生 ComfyUI Attention 运算；Step-1 Dynamic Cache 采用在线动态 KV 缓存；Decoupled Pure Prefix (Zero Overlap) 彻底解耦时间轴，前缀作为纯外部只读 KV 注入，新视频从 t=0 起跑，提示词动作完美对齐第 0 秒，免裁切零损耗。"
                }),
                "cache_dtype": (["fp8", "bf16", "fp16"], {"default": "fp8"}),
                "device_mode": (["auto", "gpu", "cpu_pinned"], {"default": "auto"}),
                "rolling_frames": (["22", "5", "39", "56", "73", "90", "107", "124"], {
                    "default": "22",
                    "tooltip": "滑动窗口实际视频帧数 (VAE 网格点)。推荐 22 帧 (~0.92s, 7 个 latent steps，严密对齐 cycle position 0)"
                }),
            },
            "optional": {
                "use_anchor": ("BOOLEAN", {"default": False, "tooltip": "是否额外保留第一段的首帧锚点。多片段连续接力建议 False，由 rolling head 平滑过渡"}),
                "anchor_frames": ("INT", {"default": 5, "min": 1, "max": 31, "step": 1, "tooltip": "首尾锚点保护实际视频帧数"}),
                "anchor_latent_frames": ("INT", {"default": 2, "min": 1, "max": 16, "step": 1}),
                "rolling_latent_frames": ("INT", {"default": 7, "min": 1, "max": 64, "step": 1}),
            }
        }

    RETURN_TYPES = ("MINIMAX_CACHE_CONFIG",)
    RETURN_NAMES = ("cache_config",)
    FUNCTION = "create_config"
    CATEGORY = "MiniMaxH3/PrefixStream"

    def create_config(
        self,
        cache_mode: str = "Safe Native (Zero Artifacts, Recommended)",
        cache_dtype: str = "fp8",
        device_mode: str = "auto",
        rolling_frames: Any = "22",
        use_anchor: bool = False,
        anchor_frames: int = 5,
        anchor_latent_frames: Optional[int] = None,
        rolling_latent_frames: Optional[int] = None,
        **kwargs
    ) -> Tuple[KVCacheConfig]:
        # Seamless backward compatibility for older saved workflow widget ordering:
        # e.g. ['fp8', 'auto', True, ...] where cache_dtype was first
        if cache_mode in ("fp8", "bf16", "fp16") and cache_dtype in ("auto", "gpu", "cpu_pinned"):
            actual_cache_dtype = cache_mode
            actual_device_mode = cache_dtype
            actual_use_anchor = bool(rolling_frames) if isinstance(rolling_frames, bool) else use_anchor
            actual_cache_mode = "Safe Native (Zero Artifacts, Recommended)"
            r_frames = 22
        else:
            actual_cache_mode = cache_mode
            actual_cache_dtype = cache_dtype
            actual_device_mode = device_mode
            actual_use_anchor = use_anchor
            try:
                r_frames = int(rolling_frames)
            except (ValueError, TypeError):
                r_frames = 22

        if rolling_latent_frames is not None:
            r_frames = latent_steps_to_pixel_frames(rolling_latent_frames)
        elif "rolling_latent_frames" in kwargs:
            r_frames = latent_steps_to_pixel_frames(kwargs["rolling_latent_frames"])

        if anchor_latent_frames is not None:
            anchor_frames = latent_steps_to_pixel_frames(anchor_latent_frames)
        elif "anchor_latent_frames" in kwargs:
            anchor_frames = latent_steps_to_pixel_frames(kwargs["anchor_latent_frames"])

        config = KVCacheConfig(
            cache_mode=actual_cache_mode,
            cache_dtype=actual_cache_dtype,
            device_mode=actual_device_mode,
            use_anchor=actual_use_anchor,
            anchor_frames=anchor_frames,
            rolling_frames=r_frames
        )
        return (config,)


def _unpack_latent(latent_dict: Optional[Dict[str, Any]]) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Unpacks video and audio tensors from an H3 latent dict, handling NestedTensor."""
    if latent_dict is None:
        return None, None
    samples = latent_dict.get("samples")
    if samples is None:
        return None, None
    if hasattr(samples, "unbind"):
        parts = list(samples.unbind())
        v = parts[0]
        a = parts[1] if len(parts) > 1 else None
    elif hasattr(samples, "tensors"):
        parts = samples.tensors
        v = parts[0]
        a = parts[1] if len(parts) > 1 else None
    elif isinstance(samples, (tuple, list)):
        v = samples[0]
        a = samples[1] if len(samples) > 1 else None
    elif isinstance(samples, torch.Tensor):
        v = samples
        a = None
    else:
        return None, None
    if v is not None and v.ndim == 4:
        v = v.unsqueeze(0)
    if a is not None and a.ndim == 3:
        a = a.unsqueeze(0)
    return v, a


def pack_av_latent(
    video: torch.Tensor,
    audio: Optional[torch.Tensor] = None,
    original_dict: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Packs video and audio tensors back into an H3 latent dict matching ComfyUI conventions."""
    out = dict(original_dict) if original_dict is not None else {}
    if audio is None:
        out["samples"] = video
        return out

    try:
        import comfy.nested_tensor
        out["samples"] = comfy.nested_tensor.NestedTensor([video, audio])
    except (ImportError, AttributeError):
        out["samples"] = (video, audio)
    return out


class MiniMaxPrefixCacheApplierNode:
    """Attaches Prefix KV Caching engine to MiniMax H3 model and injects keyframe conditioning before diffusion sampling."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "conditioning": ("CONDITIONING",),
            },
            "optional": {
                "cache_config": ("MINIMAX_CACHE_CONFIG",),
                "context_latent": ("LATENT",),
                "anchor_latent": ("LATENT",),
                "context_video_latent": ("LATENT",),  # Backward compatibility alias
                "anchor_video_latent": ("LATENT",),   # Backward compatibility alias
                "context_audio": ("AUDIO",),          # Optional raw audio waveform fallback
                "session": ("MINIMAX_SESSION",),
            }
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "MINIMAX_SESSION")
    RETURN_NAMES = ("model", "conditioning", "session")
    FUNCTION = "apply_cache"
    CATEGORY = "MiniMaxH3/PrefixStream"

    def apply_cache(
        self,
        model: Any,
        conditioning: Any,
        cache_config: Optional[KVCacheConfig] = None,
        context_latent: Optional[Dict[str, Any]] = None,
        anchor_latent: Optional[Dict[str, Any]] = None,
        context_video_latent: Optional[Dict[str, Any]] = None,
        anchor_video_latent: Optional[Dict[str, Any]] = None,
        context_audio: Optional[Dict[str, Any]] = None,
        session: Optional[LongVideoSession] = None
    ) -> Tuple[Any, Any, LongVideoSession]:
        cfg = cache_config or KVCacheConfig()
        sess = session or LongVideoSession(cfg)
        if session is not None and cache_config is not None:
            sess.config = cfg

        ctx_target = context_latent if context_latent is not None else context_video_latent
        anc_target = anchor_latent if anchor_latent is not None else anchor_video_latent

        # Unpack video and audio from latents (supports ComfyUI NestedTensor from H3ContinuousLoadLatent)
        v_ctx, a_ctx_from_latent = _unpack_latent(ctx_target)
        v_anc, _ = _unpack_latent(anc_target)

        # Audio priority: extracted from context_latent or explicit context_audio
        a_ctx = None
        if a_ctx_from_latent is not None:
            a_ctx = a_ctx_from_latent
        elif context_audio is not None and "waveform" in context_audio:
            a_ctx = context_audio["waveform"]

        # Prepare next clip (handles Phase 0 warmup, keyframe conditioning injection, and Phase 1 hook injection)
        patched_model, out_cond = sess.prepare_next_clip(
            model_patcher=model,
            conditioning=conditioning,
            previous_video_latent=v_ctx,
            previous_audio_latent=a_ctx,
            anchor_video_latent=v_anc
        )

        return (patched_model, out_cond, sess)


class MiniMaxTrimPrefixLatentNode:
    """Automatically trims redundant prefix overlap frames from video and audio.
    
    SUPPORTED MODES:
    1. Pixel & Waveform Space (RECOMMENDED): Connect decoded 'images' and 'audio'.
       Trims leading frames directly in pixel space, guaranteeing ZERO VAE flicker and ZERO color distortion!
    2. Latent Space: Connect 'latent'. Trims raw latent steps.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trim_frames": ("INT", {
                    "default": 0, "min": 0, "max": 124, "step": 1,
                    "tooltip": "裁切的前置重叠帧数 (如 22 帧)。设为 0 且连接了 session/config 时将自动识别"
                }),
            },
            "optional": {
                "images": ("IMAGE", {"tooltip": "【强烈推荐】解码后的完整画面。在像素空间裁切，彻底杜绝 VAE 闪烁与偏色！"}),
                "audio": ("AUDIO", {"tooltip": "【强烈推荐】解码后的音频。精确同步毫秒级样本截断，杜绝音画不同步"}),
                "latent": ("LATENT", {"tooltip": "原始采样 latent (可选，若已连接 images/audio 则无需裁切 latent)"}),
                "session": ("MINIMAX_SESSION",),
                "cache_config": ("MINIMAX_CACHE_CONFIG",),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0}),
                "match_tail": ("BOOLEAN", {"default": True, "tooltip": "尾部时长严格对齐：消除 H3 40Hz 音频与 24fps 画面约8ms的网格舍入累积误差"}),
                "video_latent": ("LATENT",),  # Backward compatibility alias
            }
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "LATENT")
    RETURN_NAMES = ("trimmed_images", "trimmed_audio", "trimmed_latent")
    FUNCTION = "trim"
    CATEGORY = "MiniMaxH3/PrefixStream"

    def trim(
        self,
        trim_frames: int = 0,
        images: Optional[torch.Tensor] = None,
        audio: Optional[Dict[str, Any]] = None,
        latent: Optional[Dict[str, Any]] = None,
        video_latent: Optional[Dict[str, Any]] = None,
        session: Optional[LongVideoSession] = None,
        cache_config: Optional[KVCacheConfig] = None,
        fps: float = 24.0,
        match_tail: bool = True,
        **kwargs
    ) -> Tuple[Optional[torch.Tensor], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        # 1. Determine trim frame count
        is_decoupled = False
        if session is not None and getattr(session.config, "is_decoupled_mode", lambda: False)():
            is_decoupled = True
        elif cache_config is not None and getattr(cache_config, "is_decoupled_mode", lambda: False)():
            is_decoupled = True

        actual_trim_frames = trim_frames
        if is_decoupled and trim_frames == 0:
            actual_trim_frames = 0
            logger.info("[Trim AV] Decoupled Pure Prefix Mode active: 0 overlap frames to trim, safely passing through.")
        elif actual_trim_frames <= 0:
            if session is not None:
                # Strictly respect session: 0 for initial clip, >0 for continuation clips
                actual_trim_frames = session.last_rolling_frames
            elif cache_config is not None:
                actual_trim_frames = cache_config.rolling_frames


        # 2. Pixel & audio waveform trimming (Golden Standard)
        out_images = None
        out_audio = None
        if images is not None:
            out_images, out_audio = trim_images_and_audio(
                images=images,
                audio=audio,
                trim_frames=actual_trim_frames,
                fps=fps,
                match_tail=match_tail
            )
            logger.info(
                "[Trim AV] Cleanly trimmed %d leading frames in pixel space. Output: %d frames (~%.2fs). Zero VAE flicker.",
                actual_trim_frames, out_images.shape[0], out_images.shape[0] / float(fps)
            )
        elif audio is not None:
            dummy_images = torch.empty((int(round(actual_trim_frames + 1)), 1, 1, 3))
            _, out_audio = trim_images_and_audio(
                images=dummy_images,
                audio=audio,
                trim_frames=actual_trim_frames,
                fps=fps,
                match_tail=match_tail
            )

        # 3. Latent trimming (fallback / passthrough)
        out_latent = None
        target_latent = latent if latent is not None else video_latent
        if target_latent is not None:
            trim_steps = 0
            if actual_trim_frames > 0:
                trim_steps = pixel_frames_to_latent_steps(actual_trim_frames)
            elif session is not None and session.last_rolling_steps > 0:
                trim_steps = session.last_rolling_steps
            elif cache_config is not None:
                trim_steps = cache_config.rolling_latent_frames

            v, a_from_latent = _unpack_latent(target_latent)
            if v is None:
                v = target_latent.get("samples")

            if v is not None:
                if trim_steps > 0 and trim_steps < v.shape[2]:
                    trimmed_v = v[:, :, trim_steps:]
                else:
                    trimmed_v = v
                trimmed_a = None
                if a_from_latent is not None:
                    if trim_steps > 0 and v.shape[2] > 0:
                        audio_trim_steps = int(round(trim_steps * (a_from_latent.shape[-1] / v.shape[2])))
                        trimmed_a = trim_audio_latents(a_from_latent, audio_trim_steps)
                    else:
                        trimmed_a = a_from_latent
                out_latent = pack_av_latent(trimmed_v, trimmed_a, original_dict=target_latent)
            else:
                out_latent = target_latent

        if out_images is None:
            out_images = images if images is not None else torch.empty((0, 768, 1344, 3), dtype=torch.float32)
        if out_audio is None:
            out_audio = audio

        return (out_images, out_audio, out_latent)


class MiniMaxLongVideoStitcherNode:
    """Seamlessly joins adjacent video and audio clips.
    
    MODES:
    1. Pixel & Audio Waveform Space (RECOMMENDED): Connect 'current_images', 'previous_images',
       'current_audio', 'previous_audio'.
       Performs luminance gain matching and cosine S-curve blending.
       100% immune to VAE causal collapse, gray/dirt corrupted frames, or color distortion!
    2. Latent Space (Fallback): Joins latents directly in latent space.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trim_frames": ("INT", {
                    "default": 0, "min": 0, "max": 124, "step": 1,
                    "tooltip": "重叠帧数。设为 0 且连接了 session/config 时将自动对齐"
                }),
                "crossfade_frames": ("INT", {
                    "default": 4, "min": 0, "max": 24, "step": 1,
                    "tooltip": "接缝余弦平滑过渡帧数 (推荐 4 帧，实现肉眼无痕拼接)"
                }),
                "luminance_match": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "亮度自适应匹配：自动分析前后片段接缝处的曝光，消除接缝亮度突变跳跃"
                }),
                "audio_crossfade_ms": ("INT", {
                    "default": 50, "min": 0, "max": 500, "step": 5,
                    "tooltip": "音频等功率微淡入淡出毫秒数，彻底消除拼接处咔哒爆音"
                }),
            },
            "optional": {
                "current_images": ("IMAGE", {"tooltip": "【强烈推荐】当前片段完整解码后的画面 (来自 VAEDecode)"}),
                "previous_images": ("IMAGE", {"tooltip": "【强烈推荐】前一片段完整解码后的画面 (来自上一段 VAEDecode 或 LoadVideo)"}),
                "current_audio": ("AUDIO", {"tooltip": "当前片段解码后的完整音频 (来自 VAEDecodeAudio)"}),
                "previous_audio": ("AUDIO", {"tooltip": "前一片段解码后的完整音频 (来自上一段音频)"}),
                "current_latent": ("LATENT", {"tooltip": "当前片段 raw latent (可选)"}),
                "previous_latent": ("LATENT", {"tooltip": "前一片段 raw latent (可选)"}),
                "session": ("MINIMAX_SESSION",),
                "cache_config": ("MINIMAX_CACHE_CONFIG",),
                "latent_blend_steps": ("INT", {"default": 2, "min": 0, "max": 8, "step": 1}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0}),
                "current_video": ("LATENT",),   # Backward compatibility alias
                "previous_video": ("LATENT",),  # Backward compatibility alias
            }
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "IMAGE", "AUDIO", "LATENT", "LATENT")
    RETURN_NAMES = (
        "stitched_images", "stitched_audio",
        "trimmed_current_images", "trimmed_current_audio",
        "stitched_latent", "trimmed_current_latent"
    )
    FUNCTION = "stitch"
    CATEGORY = "MiniMaxH3/PrefixStream"

    def stitch(
        self,
        trim_frames: int = 0,
        crossfade_frames: int = 4,
        luminance_match: bool = True,
        audio_crossfade_ms: int = 50,
        current_images: Optional[torch.Tensor] = None,
        previous_images: Optional[torch.Tensor] = None,
        current_audio: Optional[Dict[str, Any]] = None,
        previous_audio: Optional[Dict[str, Any]] = None,
        current_latent: Optional[Dict[str, Any]] = None,
        previous_latent: Optional[Dict[str, Any]] = None,
        current_video: Optional[Dict[str, Any]] = None,
        previous_video: Optional[Dict[str, Any]] = None,
        session: Optional[LongVideoSession] = None,
        cache_config: Optional[KVCacheConfig] = None,
        latent_blend_steps: int = 2,
        fps: float = 24.0,
        **kwargs
    ) -> Tuple[
        Optional[torch.Tensor], Optional[Dict[str, Any]],
        Optional[torch.Tensor], Optional[Dict[str, Any]],
        Optional[Dict[str, Any]], Optional[Dict[str, Any]]
    ]:
        # Determine overlap frame count
        overlap_frames = trim_frames
        if overlap_frames <= 0:
            if session is not None and session.last_rolling_frames > 0:
                overlap_frames = session.last_rolling_frames
            elif cache_config is not None:
                overlap_frames = cache_config.rolling_frames
            else:
                overlap_frames = 22

        # 1. Pixel-space video stitching
        out_stitched_images = None
        out_trimmed_images = None

        if current_images is not None and current_images.shape[0] > 0:
            out_trimmed_images, _ = trim_images_and_audio(
                images=current_images,
                audio=current_audio,
                trim_frames=overlap_frames,
                fps=fps
            )

            if previous_images is not None and previous_images.shape[0] > 0:
                out_stitched_images = stitch_video_images(
                    prev_images=previous_images,
                    curr_images=current_images,
                    trim_frames=overlap_frames,
                    crossfade_frames=crossfade_frames,
                    luminance_match=luminance_match
                )
                logger.info(
                    "[Stitch Video] Pixel join: %d + %d frames -> %d frames (Overlap: %df, Crossfade: %df, LumaMatch: %s)",
                    previous_images.shape[0], current_images.shape[0], out_stitched_images.shape[0],
                    overlap_frames, crossfade_frames, luminance_match
                )
            else:
                out_stitched_images = current_images
        elif previous_images is not None and previous_images.shape[0] > 0:
            out_stitched_images = previous_images

        # 2. Waveform-space audio stitching
        out_stitched_audio = None
        out_trimmed_audio = None

        if current_audio is not None and "waveform" in current_audio:
            curr_total_f = current_images.shape[0] if current_images is not None else 124
            _, out_trimmed_audio = trim_images_and_audio(
                images=current_images if current_images is not None else torch.empty((curr_total_f, 1, 1, 3)),
                audio=current_audio,
                trim_frames=overlap_frames,
                fps=fps
            )

            if previous_audio is not None and "waveform" in previous_audio:
                out_stitched_audio = stitch_audio_waveforms(
                    prev_audio=previous_audio,
                    curr_audio=current_audio,
                    curr_total_frames=curr_total_f,
                    trim_frames=overlap_frames,
                    crossfade_ms=float(audio_crossfade_ms),
                    fps=fps
                )
                logger.info(
                    "[Stitch Audio] Waveform join with %dms crossfade. Total samples: %d",
                    audio_crossfade_ms, out_stitched_audio["waveform"].shape[-1]
                )
            else:
                out_stitched_audio = current_audio
        elif previous_audio is not None and "waveform" in previous_audio:
            out_stitched_audio = previous_audio

        # 3. Latent stitching (backward compatibility fallback)
        out_stitched_latent = None
        out_trimmed_latent = None

        curr_target = current_latent if current_latent is not None else current_video
        prev_target = previous_latent if previous_latent is not None else previous_video

        if curr_target is not None:
            curr_v, curr_a = _unpack_latent(curr_target)
            if curr_v is None:
                curr_v = curr_target.get("samples")

            prev_v, prev_a = (None, None)
            if prev_target is not None:
                prev_v, prev_a = _unpack_latent(prev_target)
                if prev_v is None:
                    prev_v = prev_target.get("samples")

            overlap_steps = pixel_frames_to_latent_steps(overlap_frames)
            audio_overlap_steps = 0
            if curr_a is not None and curr_v is not None and curr_v.shape[2] > 0 and overlap_steps > 0:
                audio_overlap_steps = int(round(overlap_steps * (curr_a.shape[-1] / curr_v.shape[2])))

            if curr_v is not None and overlap_steps > 0 and overlap_steps < curr_v.shape[2]:
                trimmed_curr_v = curr_v[:, :, overlap_steps:]
            else:
                trimmed_curr_v = curr_v

            if curr_a is not None:
                if audio_overlap_steps > 0 and audio_overlap_steps < curr_a.shape[-1]:
                    trimmed_curr_a = curr_a[..., audio_overlap_steps:]
                else:
                    trimmed_curr_a = curr_a
            else:
                trimmed_curr_a = None

            if trimmed_curr_v is not None:
                out_trimmed_latent = pack_av_latent(trimmed_curr_v, trimmed_curr_a, original_dict=curr_target)

            if prev_v is None:
                stitched_v = curr_v
                stitched_a = curr_a
            else:
                stitched_v = stitch_video_latents(
                    prev_latent=prev_v,
                    curr_latent=curr_v,
                    overlap_steps=overlap_steps,
                    blend_steps=latent_blend_steps
                )
                if prev_a is not None and curr_a is not None:
                    stitched_a = stitch_audio_latents(
                        prev_latent=prev_a,
                        curr_latent=curr_a,
                        overlap_steps=audio_overlap_steps,
                        blend_steps=latent_blend_steps
                    )
                else:
                    stitched_a = curr_a if curr_a is not None else prev_a

            if stitched_v is not None:
                out_stitched_latent = pack_av_latent(stitched_v, stitched_a, original_dict=curr_target)

        if out_stitched_images is None:
            out_stitched_images = torch.empty((0, 768, 1344, 3), dtype=torch.float32)
        if out_trimmed_images is None:
            out_trimmed_images = torch.empty((0, 768, 1344, 3), dtype=torch.float32)

        return (
            out_stitched_images, out_stitched_audio,
            out_trimmed_images, out_trimmed_audio,
            out_stitched_latent, out_trimmed_latent
        )


class MiniMaxCacheMonitorNode:
    """Provides memory consumption and operational telemetry for Prefix KV Cache."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "session": ("MINIMAX_SESSION",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("telemetry_report",)
    FUNCTION = "report"
    CATEGORY = "MiniMaxH3/PrefixStream"

    def report(self, session: LongVideoSession) -> Tuple[str]:
        mem = session.cache_manager.get_memory_usage_mb()
        status_str = "ACTIVE (Reusing Prefix KV across sampling steps)" if session.cache_manager.has_cache(0) else "READY (Step-1 online capture armed)"
        report_str = (
            f"=== MiniMax H3 Prefix KV Cache Telemetry ===\n"
            f"Cache Engine Status: {status_str}\n"
            f"Current Clip: #{session.current_clip_index}\n"
            f"Precision: {session.config.cache_dtype.upper()}\n"
            f"Device Mode: {session.config.device_mode} (Resolved: {'CPU-Pinned' if session.cache_manager.is_cpu_pinned else 'GPU'})\n"
            f"Active Prefix Tokens: {session.cache_manager.captured_tokens}\n"
            f"Denoise Steps Accelerated: {session.cache_manager.skipped_steps_count}\n"
            f"GPU VRAM Usage: {mem['gpu_mb']:.2f} MB\n"
            f"CPU Pinned Usage: {mem['cpu_pinned_mb']:.2f} MB\n"
            f"Total Cache Footprint: {mem['total_mb']:.2f} MB\n"
            f"Anchor Window: {session.config.anchor_frames} frames ({session.config.anchor_latent_frames} latent steps, ~{session.config.anchor_frames / 24.0:.2f}s)\n"
            f"Rolling Window: {session.config.rolling_frames} frames ({session.config.rolling_latent_frames} latent steps, ~{session.config.rolling_frames / 24.0:.2f}s)\n"
            f"Last Overlap Trimmed: {session.last_rolling_frames} frames ({session.last_rolling_steps} latent steps)\n"
            f"Accumulated Clips: {len(session.accumulated_video_latents)}"
        )
        return (report_str,)


class MiniMaxSaveLatentNode:
    """Saves the complete joint MiniMax H3 AV latent to safetensors for seamless continuation.
    
    Guarantees 100% independent, standalone operation without requiring external continuation suites.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "Sampler output joint AV latent to save."}),
                "filename_prefix": ("STRING", {
                    "default": "minimax_h3/clip",
                    "tooltip": "Subfolder and filename prefix in ComfyUI output directory."
                }),
                "clip_index": ("INT", {
                    "default": 1, "min": 0, "max": 99999, "step": 1,
                    "tooltip": "Fixed chain slot (1, 2, 3...). 0 = auto-incrementing."
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("saved_path", "latent_info")
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = "MiniMaxH3/PrefixStream"

    def save(
        self,
        latent: Dict[str, Any],
        filename_prefix: str = "minimax_h3/clip",
        clip_index: int = 1,
        **kwargs
    ) -> Tuple[str, str]:
        if latent is None:
            raise ValueError("MiniMaxSaveLatent: 'latent' input is required.")

        video, audio = _unpack_latent(latent)
        if video is None:
            raise ValueError("MiniMaxSaveLatent: latent contains no video samples.")

        video_cpu = video.detach().cpu().contiguous()
        audio_cpu = audio.detach().cpu().contiguous() if audio is not None else None

        try:
            import folder_paths
            base_dir = folder_paths.get_output_directory()
        except Exception:
            base_dir = "output"

        target_dir = os.path.join(base_dir, os.path.dirname(filename_prefix))
        os.makedirs(target_dir, exist_ok=True)

        base_name = os.path.basename(filename_prefix)
        if clip_index > 0:
            filename = f"{base_name}_{clip_index:05d}.safetensors"
        else:
            filename = f"{base_name}_temp.safetensors"
        full_path = os.path.join(target_dir, filename)

        tensors = {"video": video_cpu}
        if audio_cpu is not None:
            tensors["audio"] = audio_cpu

        frame_count = latent_steps_to_pixel_frames(video_cpu.shape[2])

        if st_save is not None:
            st_save(
                tensors,
                full_path,
                metadata={
                    "format": "minimax_h3_av_latent",
                    "frame_count": str(frame_count),
                    "clip_index": str(clip_index),
                    "video_shape": json.dumps(list(video_cpu.shape)),
                    "audio_shape": json.dumps(list(audio_cpu.shape)) if audio_cpu is not None else "none",
                }
            )
        else:
            torch.save(tensors, full_path)

        info_str = f"{frame_count} frames | Video {tuple(video_cpu.shape)}"
        if audio_cpu is not None:
            info_str += f" | Audio {tuple(audio_cpu.shape)}"
        logger.info("[Save Latent] Successfully saved %s -> %s", info_str, full_path)
        return (full_path, info_str)


class MiniMaxLoadLatentNode:
    """Loads a saved MiniMax H3 joint AV latent for continuation and long-video stitching.
    
    Guarantees 100% independent, standalone operation: compatible with any saved H3 safetensors latent.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent_path": ("STRING", {
                    "default": "minimax_h3/clip",
                    "tooltip": "Folder or filepath relative to ComfyUI output, or absolute path."
                }),
                "clip_index": ("INT", {
                    "default": 1, "min": 0, "max": 99999, "step": 1,
                    "tooltip": "Clip index to load (e.g. 1 to continue Clip 2). 0 = latest file."
                }),
            }
        }

    RETURN_TYPES = ("LATENT", "STRING", "STRING")
    RETURN_NAMES = ("latent", "loaded_path", "latent_info")
    FUNCTION = "load"
    CATEGORY = "MiniMaxH3/PrefixStream"

    def load(
        self,
        latent_path: str = "minimax_h3/clip",
        clip_index: int = 1
    ) -> Tuple[Dict[str, Any], str, str]:
        try:
            import folder_paths
            base_dir = folder_paths.get_output_directory()
        except Exception:
            base_dir = "output"

        p = (latent_path or "").strip().strip('"').strip("'")
        if not os.path.isabs(p):
            full_target = os.path.join(base_dir, p)
        else:
            full_target = p

        if os.path.isfile(full_target):
            target_file = full_target
        else:
            parent_dir = os.path.dirname(full_target)
            base_name = os.path.basename(full_target)
            if os.path.isdir(parent_dir):
                if clip_index > 0:
                    candidates = [f for f in os.listdir(parent_dir) if f.startswith(base_name) and f.endswith(".safetensors")]
                    matched = [f for f in candidates if f"_{clip_index:05d}" in f or f"_{clip_index}." in f or f"_{clip_index}_" in f]
                    if matched:
                        target_file = os.path.join(parent_dir, matched[0])
                    else:
                        target_file = os.path.join(parent_dir, f"{base_name}_{clip_index:05d}.safetensors")
                else:
                    candidates = [os.path.join(parent_dir, f) for f in os.listdir(parent_dir) if f.endswith(".safetensors")]
                    if candidates:
                        target_file = max(candidates, key=os.path.getmtime)
                    else:
                        target_file = full_target
            else:
                target_file = full_target

        if not os.path.exists(target_file):
            raise FileNotFoundError(f"MiniMaxLoadLatent: file not found at '{target_file}'")

        if st_load is not None:
            tensors = st_load(target_file, device="cpu")
        else:
            try:
                tensors = torch.load(target_file, map_location="cpu")
            except RuntimeError as exc:
                if "safetensors is not installed" in str(exc):
                    with open(target_file, "rb") as f:
                        tensors = torch.load(f, map_location="cpu")
                else:
                    raise

        if "video" not in tensors:
            raise ValueError(f"MiniMaxLoadLatent: '{target_file}' does not contain 'video' tensor.")

        video = tensors["video"]
        audio = tensors.get("audio", None)

        out_latent = pack_av_latent(video, audio)
        frame_count = latent_steps_to_pixel_frames(video.shape[2])
        info_str = f"{frame_count} frames | Video {tuple(video.shape)}"
        if audio is not None:
            info_str += f" | Audio {tuple(audio.shape)}"

        logger.info("[Load Latent] Successfully loaded %s (%s)", target_file, info_str)
        return (out_latent, target_file, info_str)


class MiniMaxClipBinSaverNode:
    """Saves a unified MiniMax H3 AV Latent into the Clip Bin media pool with keyframes, preview card, and rich metadata."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "采样器输出的原生联合音画 Latent (支持 NestedTensor)"}),
                "project_name": ("STRING", {
                    "default": "Default_Project",
                    "tooltip": "素材箱项目名称（不同故事/场景独立归档管理）"
                }),
                "shot_tag": ("STRING", {
                    "default": "Auto (自动编号)",
                    "tooltip": "镜头名称或动作标签（填 'Auto (自动编号)' 将自动递增为 Shot 1, Shot 2...）"
                }),
                "rating": ("INT", {
                    "default": 4, "min": 1, "max": 5, "step": 1,
                    "tooltip": "镜头星标打分（1~5星），方便事后一键过滤废案"
                }),
            },
            "optional": {
                "images": ("IMAGE", {"tooltip": "VAE Decode 解码的视频帧。连接后自动截取首帧与最后一帧生成高清缩略图！"}),
                "prompt": ("STRING", {"default": "", "tooltip": "本镜头的正向提示词（便于回溯与承接）"}),
                "parent_clip_id": ("STRING", {"default": "", "tooltip": "父镜头 ID（记录分支血缘）"}),
                "video_file_name": ("STRING", {"default": "", "tooltip": "关联的 MP4 视频文件名（建立 1:1 双向索引）"}),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("clip_id", "preview_image", "bin_path")
    OUTPUT_NODE = True
    FUNCTION = "save_clip"
    CATEGORY = "MiniMaxH3/ClipBin"

    def save_clip(
        self,
        latent: Dict[str, Any],
        project_name: str = "Default_Project",
        shot_tag: str = "Auto (自动编号)",
        rating: int = 4,
        images: Optional[torch.Tensor] = None,
        prompt: str = "",
        parent_clip_id: str = "",
        video_file_name: str = "",
        **kwargs
    ) -> Dict[str, Any]:
        if latent is None:
            raise ValueError("MiniMaxClipBinSaver: 'latent' input is required.")

        video, audio = _unpack_latent(latent)
        if video is None:
            raise ValueError("MiniMaxClipBinSaver: latent contains no video samples.")

        actual_shot = (shot_tag or "").strip()
        if actual_shot.startswith("Auto") or not actual_shot:
            idx = load_project_index(project_name)
            actual_shot = f"Shot {len(idx.get('clips', [])) + 1}"

        meta_obj, clip_dir, preview_pil = save_clip_asset(
            video_tensor=video,
            audio_tensor=audio,
            images=images,
            project_name=project_name,
            shot_tag=actual_shot,
            prompt=prompt,
            rating=rating,
            parent_clip_id=parent_clip_id,
            associated_video_path=video_file_name,
        )

        preview_tensor = pil_to_tensor(preview_pil)

        try:
            import folder_paths
            base_dir = folder_paths.get_output_directory()
            subfolder = os.path.relpath(clip_dir, base_dir)
        except Exception:
            subfolder = ""

        ui_images = [{
            "filename": "preview.png",
            "subfolder": subfolder,
            "type": "output"
        }]

        logger.info("[Clip Bin Saver] Stored clip '%s' in '%s' (%s frames | ⭐%s | tag: %s)",
                    meta_obj.clip_id, project_name, meta_obj.frames, meta_obj.rating, actual_shot)

        return {
            "ui": {"images": ui_images},
            "result": (meta_obj.clip_id, preview_tensor, clip_dir)
        }


class MiniMaxClipBinPickerNode:
    """Visually browses, filters, and loads clips from the Clip Bin with instant tail-frame output."""

    @classmethod
    def INPUT_TYPES(cls):
        projects = list_projects()
        default_proj = projects[0] if projects else "Default_Project"
        return {
            "required": {
                "project_name": ("STRING", {
                    "default": default_proj,
                    "tooltip": "素材箱项目名称。可填已有项目名，或通过控制台查看可用项目。"
                }),
                "mode": ([
                    "Auto (首段全新 / 后续自动接力)",
                    "Force Initial (强制新建首段，无上下文)",
                    "Strict Chaining (必须接力指定或最新镜头)"
                ], {
                    "default": "Auto (首段全新 / 后续自动接力)",
                    "tooltip": "工作模式：'Auto' 最省心，首次运行自动开辟首段，后续自动接力上一段；'Force Initial' 强制全新生成；'Strict Chaining' 强校验接力。"
                }),
                "filter_rating": ([
                    "All (1-5 ⭐)",
                    "⭐⭐⭐+ (3+ ⭐)",
                    "⭐⭐⭐⭐+ (4+ ⭐)",
                    "⭐⭐⭐⭐⭐ (5 ⭐)"
                ], {
                    "default": "All (1-5 ⭐)",
                    "tooltip": "星级过滤器：只加载或选用大于等于该星级的优质镜头。"
                }),
                "clip_selection": ("STRING", {
                    "default": "latest",
                    "tooltip": "镜头选择：输入 'latest' (或留空) 自动加载本工程最新符合条件的优质镜头；也可输入具体的 clip_id (如 clip_2026...)。"
                }),
            },
            "optional": {
                "custom_clip_path": ("STRING", {
                    "default": "",
                    "tooltip": "可选绝对路径覆盖。"
                }),
            }
        }

    RETURN_TYPES = ("LATENT", "IMAGE", "IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("latent", "tail_frame", "first_frame", "prompt", "clip_id")
    FUNCTION = "pick_clip"
    CATEGORY = "MiniMaxH3/ClipBin"

    def pick_clip(
        self,
        project_name: str = "Default_Project",
        mode: str = "Auto (首段全新 / 后续自动接力)",
        filter_rating: str = "All (1-5 ⭐)",
        clip_selection: str = "latest",
        custom_clip_path: str = "",
        **kwargs
    ) -> Dict[str, Any]:
        p_name = (project_name or "Default_Project").strip()
        custom_p = (custom_clip_path or "").strip().strip('"').strip("'")

        if custom_p and os.path.isdir(custom_p):
            target_clip_dir = custom_p
            p_name = os.path.basename(os.path.dirname(custom_p)) or p_name
            target_clip_id = os.path.basename(custom_p)
        else:
            idx = load_project_index(p_name)
            clips = idx.get("clips", [])

            # Check if Initial Mode applies (Auto with empty bin, or Force Initial)
            is_initial_mode = mode.startswith("Force Initial") or (mode.startswith("Auto") and len(clips) == 0)

            if is_initial_mode:
                logger.info("[Clip Bin Picker] Operating in Initial Generation mode for project '%s' (Zero prior context).", p_name)
                card = create_placeholder_card("✨ Initial Clip Mode", f"Project: {p_name} | Ready for First Clip (No Context)")
                placeholder_tensor = pil_to_tensor(card)
                return {
                    "ui": {"images": []},
                    "result": (None, placeholder_tensor, placeholder_tensor, "", "[INITIAL_GENERATION]")
                }

            if not clips:
                raise ValueError(f"MiniMaxClipBinPicker: No clips found in project '{p_name}'. "
                                 f"Switch mode to 'Auto' to generate the first clip.")

            # Parse star rating filter
            min_stars = 1
            if filter_rating.startswith("⭐⭐⭐⭐⭐"):
                min_stars = 5
            elif filter_rating.startswith("⭐⭐⭐⭐"):
                min_stars = 4
            elif filter_rating.startswith("⭐⭐⭐"):
                min_stars = 3

            filtered = [c for c in clips if c.get("rating", 3) >= min_stars]
            if not filtered:
                logger.warning("[Clip Bin Picker] No clips match rating >= %s in '%s', falling back to all clips.",
                               min_stars, p_name)
                filtered = clips

            sel = (clip_selection or "latest").strip()
            if sel.lower() in ("latest", "", "0", "auto", "default"):
                target_clip = filtered[0]
                target_clip_id = target_clip["clip_id"]
            else:
                # Substring / exact match
                matched = [c for c in clips if sel in c.get("clip_id", "") or sel in c.get("shot_tag", "")]
                if matched:
                    target_clip_id = matched[0]["clip_id"]
                else:
                    target_clip_id = sel

        video, audio, tail_tensor, first_tensor, meta_dict = load_clip_asset(p_name, target_clip_id)
        out_latent = pack_av_latent(video, audio)

        project_dir = get_project_dir(p_name)
        clip_dir = os.path.join(project_dir, target_clip_id)

        try:
            import folder_paths
            base_dir = folder_paths.get_output_directory()
            subfolder = os.path.relpath(clip_dir, base_dir)
        except Exception:
            subfolder = ""

        # UI Preview: show tail_frame or preview.png
        preview_file = "tail_frame.png" if os.path.isfile(os.path.join(clip_dir, "tail_frame.png")) else "preview.png"
        ui_images = [{
            "filename": preview_file,
            "subfolder": subfolder,
            "type": "output"
        }]

        prompt_str = meta_dict.get("prompt", "")
        frames = meta_dict.get("frames", latent_steps_to_pixel_frames(video.shape[2]))
        logger.info("[Clip Bin Picker] Loaded clip '%s' (%s frames | ⭐%s | tag: '%s')",
                    target_clip_id, frames, meta_dict.get("rating", 3), meta_dict.get("shot_tag", ""))

        return {
            "ui": {"images": ui_images},
            "result": (out_latent, tail_tensor, first_tensor, prompt_str, target_clip_id)
        }



NODE_CLASS_MAPPINGS = {
    "MiniMaxPrefixCacheConfig": MiniMaxPrefixCacheConfigNode,
    "MiniMaxPrefixCacheApplier": MiniMaxPrefixCacheApplierNode,
    "MiniMaxTrimPrefix": MiniMaxTrimPrefixLatentNode,
    "MiniMaxTrimPrefixLatent": MiniMaxTrimPrefixLatentNode,
    "MiniMaxLongVideoStitcher": MiniMaxLongVideoStitcherNode,
    "MiniMaxCacheMonitor": MiniMaxCacheMonitorNode,
    "MiniMaxSaveLatent": MiniMaxSaveLatentNode,
    "MiniMaxLoadLatent": MiniMaxLoadLatentNode,
    "MiniMaxClipBinSaver": MiniMaxClipBinSaverNode,
    "MiniMaxClipBinPicker": MiniMaxClipBinPickerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxPrefixCacheConfig": "MiniMax H3 Prefix Cache Config",
    "MiniMaxPrefixCacheApplier": "MiniMax H3 Prefix Cache Applier",
    "MiniMaxTrimPrefix": "MiniMax H3 Trim Prefix (AV Master, Zero Flicker)",
    "MiniMaxTrimPrefixLatent": "MiniMax H3 Trim Prefix Latent (AV Master)",
    "MiniMaxLongVideoStitcher": "MiniMax H3 Long Video Stitcher (Seamless AV)",
    "MiniMaxCacheMonitor": "MiniMax H3 Cache Telemetry Monitor",
    "MiniMaxSaveLatent": "MiniMax H3 Save AV Latent (Standalone)",
    "MiniMaxLoadLatent": "MiniMax H3 Load AV Latent (Standalone)",
    "MiniMaxClipBinSaver": "MiniMax H3 Clip Bin Saver (Media Pool)",
    "MiniMaxClipBinPicker": "MiniMax H3 Clip Bin Picker (Gallery Loader)",
}

