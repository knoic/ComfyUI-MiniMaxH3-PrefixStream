"""ComfyUI Custom Nodes for MiniMax H3 Prefix KV Caching & Streaming Chaining.

Exposes:
- MiniMaxPrefixCacheConfig: Configure FP8/BF16, GPU/CPU Pinned, Anchor & Rolling parameters in real video frames.
- MiniMaxPrefixCacheApplier: Connects model, historical context latents, and hooks into KSampler.
- MiniMaxTrimPrefixLatent: Automatically trims leading overlap frames from generated clips (AV unified).
- MiniMaxLongVideoStitcher: Smoothly stitches video and audio latents between clips in unified LATENT space.
- MiniMaxCacheMonitor: Real-time diagnostics for VRAM, memory footprint, and session progress.
"""

from typing import Dict, Any, Tuple, Optional
import logging
import torch

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
    )


class MiniMaxPrefixCacheConfigNode:
    """Configures Prefix KV Cache precision, memory strategy, and window sizes in real video frames."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cache_dtype": (["fp8", "bf16", "fp16"], {"default": "fp8"}),
                "device_mode": (["auto", "gpu", "cpu_pinned"], {"default": "auto"}),
                "use_anchor": ("BOOLEAN", {"default": True}),
                "anchor_frames": ("INT", {"default": 5, "min": 1, "max": 31, "step": 1, "tooltip": "首尾锚点保护实际视频帧数 (Anchor Window, 推荐 5 帧，约 1~2 个 latent steps)"}),
                "rolling_frames": ("INT", {"default": 24, "min": 4, "max": 124, "step": 1, "tooltip": "滑动窗口实际视频帧数 (Rolling Window, 推荐 16~32 帧，最大可至单片段全长 124 帧)"}),
            },
            "optional": {
                "anchor_latent_frames": ("INT", {"default": 2, "min": 1, "max": 16, "step": 1}),
                "rolling_latent_frames": ("INT", {"default": 6, "min": 1, "max": 64, "step": 1}),
            }
        }

    RETURN_TYPES = ("MINIMAX_CACHE_CONFIG",)
    RETURN_NAMES = ("cache_config",)
    FUNCTION = "create_config"
    CATEGORY = "MiniMaxH3/PrefixStream"

    def create_config(
        self,
        cache_dtype: str,
        device_mode: str,
        use_anchor: bool,
        anchor_frames: int = 5,
        rolling_frames: int = 24,
        anchor_latent_frames: Optional[int] = None,
        rolling_latent_frames: Optional[int] = None,
        **kwargs
    ) -> Tuple[KVCacheConfig]:
        # Backward compatibility for workflows passing old *_latent_frames
        if anchor_latent_frames is not None:
            anchor_frames = latent_steps_to_pixel_frames(anchor_latent_frames)
        elif "anchor_latent_frames" in kwargs:
            anchor_frames = latent_steps_to_pixel_frames(kwargs["anchor_latent_frames"])

        if rolling_latent_frames is not None:
            rolling_frames = latent_steps_to_pixel_frames(rolling_latent_frames)
        elif "rolling_latent_frames" in kwargs:
            rolling_frames = latent_steps_to_pixel_frames(kwargs["rolling_latent_frames"])

        config = KVCacheConfig(
            cache_dtype=cache_dtype,
            device_mode=device_mode,
            use_anchor=use_anchor,
            anchor_frames=anchor_frames,
            rolling_frames=rolling_frames
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
    """Automatically trims redundant prefix overlap frames from generated video and audio latents.
    
    Both video and audio are trimmed synchronously within the unified LATENT.
    Connect trimmed_latent directly to VAEDecode and VAEDecodeAudio.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
            },
            "optional": {
                "video_latent": ("LATENT",),  # Backward compatibility alias
                "session": ("MINIMAX_SESSION",),
                "cache_config": ("MINIMAX_CACHE_CONFIG",),
                "trim_frames": ("INT", {"default": 0, "min": 0, "max": 124, "step": 1}),
                "audio": ("AUDIO",),          # Optional raw audio waveform fallback
            }
        }

    RETURN_TYPES = ("LATENT", "AUDIO")
    RETURN_NAMES = ("trimmed_latent", "trimmed_audio")
    FUNCTION = "trim"
    CATEGORY = "MiniMaxH3/PrefixStream"

    def trim(
        self,
        latent: Optional[Dict[str, Any]] = None,
        video_latent: Optional[Dict[str, Any]] = None,
        audio: Optional[Dict[str, Any]] = None,
        session: Optional[LongVideoSession] = None,
        cache_config: Optional[KVCacheConfig] = None,
        trim_frames: int = 0
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        target_latent = latent if latent is not None else video_latent
        if target_latent is None:
            raise ValueError("MiniMaxTrimPrefixLatentNode requires 'latent' input.")

        v, a_from_latent = _unpack_latent(target_latent)
        if v is None:
            v = target_latent.get("samples")
        if v is None:
            return (target_latent, audio)

        # Determine how many latent steps to trim
        trim_steps = 0
        if trim_frames > 0:
            trim_steps = pixel_frames_to_latent_steps(trim_frames)
        elif session is not None and session.last_rolling_steps > 0:
            trim_steps = session.last_rolling_steps
        elif cache_config is not None:
            trim_steps = cache_config.rolling_latent_frames

        # Trim video latent
        if trim_steps > 0 and trim_steps < v.shape[2]:
            trimmed_v = v[:, :, trim_steps:]
            p_frames = latent_steps_to_pixel_frames(trim_steps)
            logger.info(
                "Auto-trimmed %d prefix video latent steps (~%d frames, ~%.2fs). Remaining steps: %d",
                trim_steps, p_frames, p_frames / 24.0, trimmed_v.shape[2]
            )
        else:
            trimmed_v = v

        # Trim audio latent if present inside LATENT
        trimmed_a = None
        if a_from_latent is not None:
            if trim_steps > 0 and v.shape[2] > 0:
                audio_trim_steps = int(round(trim_steps * (a_from_latent.shape[-1] / v.shape[2])))
                trimmed_a = trim_audio_latents(a_from_latent, audio_trim_steps)
            else:
                trimmed_a = a_from_latent

        out_latent = pack_av_latent(trimmed_v, trimmed_a, original_dict=target_latent)

        # Trim raw audio waveform if provided
        out_audio = None
        if audio is not None and "waveform" in audio:
            sr = int(audio.get("sample_rate", 32000))
            if trim_steps > 0:
                p_frames = latent_steps_to_pixel_frames(trim_steps)
                trim_samples = int((p_frames / 24.0) * sr)
                w = trim_audio_waveform(audio["waveform"], trim_samples)
                out_audio = {"waveform": w, "sample_rate": sr}
            else:
                out_audio = audio

        return (out_latent, out_audio)


class MiniMaxLongVideoStitcherNode:
    """Seamlessly joins adjacent video and audio latents in unified LATENT space.
    
    Blends video latents with cosine S-curve and cross-blends audio latents.
    Connect stitched_latent directly to VAEDecode and VAEDecodeAudio.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "current_latent": ("LATENT",),
            },
            "optional": {
                "previous_latent": ("LATENT",),
                "current_video": ("LATENT",),   # Backward compatibility alias
                "previous_video": ("LATENT",),  # Backward compatibility alias
                "session": ("MINIMAX_SESSION",),
                "cache_config": ("MINIMAX_CACHE_CONFIG",),
                "trim_frames": ("INT", {"default": 0, "min": 0, "max": 124, "step": 1}),
                "latent_blend_steps": ("INT", {"default": 2, "min": 0, "max": 8, "step": 1}),
                "current_audio": ("AUDIO",),    # Optional raw audio waveform fallback
                "previous_audio": ("AUDIO",),   # Optional raw audio waveform fallback
                "audio_crossfade_ms": ("INT", {"default": 50, "min": 0, "max": 500, "step": 10}),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT", "AUDIO", "AUDIO")
    RETURN_NAMES = ("stitched_latent", "trimmed_current_latent", "stitched_audio", "trimmed_current_audio")
    FUNCTION = "stitch"
    CATEGORY = "MiniMaxH3/PrefixStream"

    def stitch(
        self,
        current_latent: Optional[Dict[str, Any]] = None,
        previous_latent: Optional[Dict[str, Any]] = None,
        current_video: Optional[Dict[str, Any]] = None,
        previous_video: Optional[Dict[str, Any]] = None,
        current_audio: Optional[Dict[str, Any]] = None,
        previous_audio: Optional[Dict[str, Any]] = None,
        session: Optional[LongVideoSession] = None,
        cache_config: Optional[KVCacheConfig] = None,
        trim_frames: int = 0,
        latent_blend_steps: int = 2,
        audio_crossfade_ms: int = 50
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        curr_target = current_latent if current_latent is not None else current_video
        if curr_target is None:
            raise ValueError("MiniMaxLongVideoStitcher requires 'current_latent' input.")
        prev_target = previous_latent if previous_latent is not None else previous_video

        curr_v, curr_a = _unpack_latent(curr_target)
        if curr_v is None:
            curr_v = curr_target.get("samples")

        prev_v, prev_a = (None, None)
        if prev_target is not None:
            prev_v, prev_a = _unpack_latent(prev_target)
            if prev_v is None:
                prev_v = prev_target.get("samples")

        # Determine overlap steps for video
        overlap_steps = 0
        if trim_frames > 0:
            overlap_steps = pixel_frames_to_latent_steps(trim_frames)
        elif session is not None and session.last_rolling_steps > 0:
            overlap_steps = session.last_rolling_steps
        elif cache_config is not None:
            overlap_steps = cache_config.rolling_latent_frames

        # Determine overlap steps for audio latent
        audio_overlap_steps = 0
        if curr_a is not None and curr_v is not None and curr_v.shape[2] > 0 and overlap_steps > 0:
            audio_overlap_steps = int(round(overlap_steps * (curr_a.shape[-1] / curr_v.shape[2])))

        # 1. Generate trimmed current video & audio latent (pure new content)
        if overlap_steps > 0 and overlap_steps < curr_v.shape[2]:
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

        out_trimmed_latent = pack_av_latent(trimmed_curr_v, trimmed_curr_a, original_dict=curr_target)

        # 2. Generate stitched video & audio latent
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

        out_stitched_latent = pack_av_latent(stitched_v, stitched_a, original_dict=curr_target)

        # 3. Optional raw waveform audio stitching if external AUDIO wires are connected
        out_stitched_audio = None
        out_trimmed_audio = None

        if current_audio is not None and "waveform" in current_audio:
            sr = int(current_audio.get("sample_rate", 32000))
            if overlap_steps > 0:
                p_frames = latent_steps_to_pixel_frames(overlap_steps)
                trim_samples = int((p_frames / 24.0) * sr)
                w_trim = trim_audio_waveform(current_audio["waveform"], trim_samples)
                out_trimmed_audio = {"waveform": w_trim, "sample_rate": sr}
            else:
                out_trimmed_audio = current_audio

            if previous_audio is None:
                out_stitched_audio = current_audio
            else:
                w1 = previous_audio.get("waveform")
                w2 = out_trimmed_audio["waveform"] if out_trimmed_audio is not None else current_audio["waveform"]
                if w1 is not None and w2 is not None:
                    cross_samples = int((audio_crossfade_ms / 1000.0) * sr)
                    stitched_w = audio_equal_power_crossfade(w1, w2, crossfade_samples=cross_samples)
                    out_stitched_audio = {"waveform": stitched_w, "sample_rate": sr}

        return (out_stitched_latent, out_trimmed_latent, out_stitched_audio, out_trimmed_audio)


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
        report_str = (
            f"=== MiniMax H3 Prefix KV Cache Telemetry ===\n"
            f"Current Clip: #{session.current_clip_index}\n"
            f"Precision: {session.config.cache_dtype.upper()}\n"
            f"Device Mode: {session.config.device_mode} (Resolved: {'CPU-Pinned' if session.cache_manager.is_cpu_pinned else 'GPU'})\n"
            f"GPU VRAM Usage: {mem['gpu_mb']:.2f} MB\n"
            f"CPU Pinned Usage: {mem['cpu_pinned_mb']:.2f} MB\n"
            f"Total Cache Footprint: {mem['total_mb']:.2f} MB\n"
            f"Anchor Window: {session.config.anchor_frames} frames ({session.config.anchor_latent_frames} latent steps, ~{session.config.anchor_frames / 24.0:.2f}s)\n"
            f"Rolling Window: {session.config.rolling_frames} frames ({session.config.rolling_latent_frames} latent steps, ~{session.config.rolling_frames / 24.0:.2f}s)\n"
            f"Last Overlap Trimmed: {session.last_rolling_frames} frames ({session.last_rolling_steps} latent steps)\n"
            f"Accumulated Clips: {len(session.accumulated_video_latents)}"
        )
        return (report_str,)


NODE_CLASS_MAPPINGS = {
    "MiniMaxPrefixCacheConfig": MiniMaxPrefixCacheConfigNode,
    "MiniMaxPrefixCacheApplier": MiniMaxPrefixCacheApplierNode,
    "MiniMaxTrimPrefixLatent": MiniMaxTrimPrefixLatentNode,
    "MiniMaxLongVideoStitcher": MiniMaxLongVideoStitcherNode,
    "MiniMaxCacheMonitor": MiniMaxCacheMonitorNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxPrefixCacheConfig": "MiniMax H3 Prefix Cache Config",
    "MiniMaxPrefixCacheApplier": "MiniMax H3 Prefix Cache Applier",
    "MiniMaxTrimPrefixLatent": "MiniMax H3 Trim Prefix Latent (Auto-Crop)",
    "MiniMaxLongVideoStitcher": "MiniMax H3 Long Video Stitcher",
    "MiniMaxCacheMonitor": "MiniMax H3 Cache Telemetry Monitor",
}
