"""ComfyUI Custom Nodes for MiniMax H3 Prefix KV Caching & Streaming Chaining.

Exposes:
- MiniMaxPrefixCacheConfig: Configure FP8/BF16, GPU/CPU Pinned, Anchor & Rolling parameters in real video frames.
- MiniMaxPrefixCacheApplier: Connects model, historical context latents, and hooks into KSampler.
- MiniMaxTrimPrefixLatent: Automatically trims leading overlap frames from generated clips.
- MiniMaxLongVideoStitcher: Smoothly stitches video latents and cross-fades audio between clips.
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
        trim_prefix_frames,
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
        trim_prefix_frames,
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
                "rolling_frames": ("INT", {"default": 22, "min": 4, "max": 124, "step": 1}),
                "anchor_frames": ("INT", {"default": 5, "min": 1, "max": 30, "step": 1}),
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
        rolling_frames: int = 22,
        anchor_frames: int = 5,
        **kwargs
    ) -> Tuple[KVCacheConfig]:
        # Backward compatibility for workflows passing old *_latent_frames
        if "rolling_latent_frames" in kwargs and "rolling_frames" not in kwargs:
            rolling_frames = latent_steps_to_pixel_frames(kwargs["rolling_latent_frames"])
        if "anchor_latent_frames" in kwargs and "anchor_frames" not in kwargs:
            anchor_frames = latent_steps_to_pixel_frames(kwargs["anchor_latent_frames"])

        config = KVCacheConfig(
            cache_dtype=cache_dtype,
            device_mode=device_mode,
            use_anchor=use_anchor,
            rolling_frames=rolling_frames,
            anchor_frames=anchor_frames
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
                "context_video_latent": ("LATENT",),
                "anchor_video_latent": ("LATENT",),
                "context_audio": ("AUDIO",),
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
        context_video_latent: Optional[Dict[str, Any]] = None,
        anchor_video_latent: Optional[Dict[str, Any]] = None,
        context_audio: Optional[Dict[str, Any]] = None,
        session: Optional[LongVideoSession] = None
    ) -> Tuple[Any, Any, LongVideoSession]:
        # Initialize or retrieve active session
        cfg = cache_config or KVCacheConfig()
        sess = session or LongVideoSession(cfg)

        # Unpack video and audio from latents (supports ComfyUI NestedTensor from H3ContinuousLoadLatent)
        v_ctx, a_ctx_from_latent = _unpack_latent(context_video_latent)
        v_anc, _ = _unpack_latent(anchor_video_latent)

        # Audio priority: explicit context_audio or extracted from context_video_latent
        a_ctx = None
        if context_audio is not None and "waveform" in context_audio:
            a_ctx = context_audio["waveform"]
        elif a_ctx_from_latent is not None:
            a_ctx = a_ctx_from_latent

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
    """Automatically trims the redundant prefix overlap frames from generated video/audio latents.
    
    Connect directly to VAE Decode to save a clean, stutter-free continuous clip without any manual cropping.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_latent": ("LATENT",),
            },
            "optional": {
                "audio": ("AUDIO",),
                "session": ("MINIMAX_SESSION",),
                "cache_config": ("MINIMAX_CACHE_CONFIG",),
                "trim_frames": ("INT", {"default": 0, "min": 0, "max": 124, "step": 1}),
            }
        }

    RETURN_TYPES = ("LATENT", "AUDIO")
    RETURN_NAMES = ("trimmed_video", "trimmed_audio")
    FUNCTION = "trim"
    CATEGORY = "MiniMaxH3/PrefixStream"

    def trim(
        self,
        video_latent: Dict[str, Any],
        audio: Optional[Dict[str, Any]] = None,
        session: Optional[LongVideoSession] = None,
        cache_config: Optional[KVCacheConfig] = None,
        trim_frames: int = 0
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        v, a_from_latent = _unpack_latent(video_latent)
        if v is None:
            v = video_latent.get("samples")
        if v is None:
            return (video_latent, audio)

        # Determine how many latent steps to trim
        trim_steps = 0
        if trim_frames > 0:
            trim_steps = pixel_frames_to_latent_steps(trim_frames)
        elif session is not None and session.last_rolling_steps > 0:
            trim_steps = session.last_rolling_steps
        elif cache_config is not None:
            trim_steps = cache_config.rolling_latent_frames

        # Trim video
        if trim_steps > 0 and trim_steps < v.shape[2]:
            trimmed_v = v[:, :, trim_steps:]
            p_frames = latent_steps_to_pixel_frames(trim_steps)
            logger.info(
                "Auto-trimmed %d prefix latent steps (~%d frames, ~%.2fs). Remaining steps: %d",
                trim_steps, p_frames, p_frames / 24.0, trimmed_v.shape[2]
            )
        else:
            trimmed_v = v

        out_video = {"samples": trimmed_v}

        # Trim audio
        out_audio = None
        target_audio = audio or ({"waveform": a_from_latent, "sample_rate": 32000} if a_from_latent is not None else None)
        if target_audio is not None and "waveform" in target_audio:
            sr = int(target_audio.get("sample_rate", 32000))
            if trim_steps > 0:
                p_frames = latent_steps_to_pixel_frames(trim_steps)
                trim_samples = int((p_frames / 24.0) * sr)
                w = trim_audio_waveform(target_audio["waveform"], trim_samples)
                out_audio = {"waveform": w, "sample_rate": sr}
            else:
                out_audio = target_audio

        return (out_video, out_audio)


class MiniMaxLongVideoStitcherNode:
    """Seamlessly joins adjacent video latents with smooth overlap blending and crossfades audio waveforms.
    
    Provides both stitched continuous full video and cleanly trimmed current segment.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "current_video": ("LATENT",),
            },
            "optional": {
                "previous_video": ("LATENT",),
                "current_audio": ("AUDIO",),
                "previous_audio": ("AUDIO",),
                "session": ("MINIMAX_SESSION",),
                "cache_config": ("MINIMAX_CACHE_CONFIG",),
                "trim_frames": ("INT", {"default": 0, "min": 0, "max": 124, "step": 1}),
                "latent_blend_steps": ("INT", {"default": 2, "min": 0, "max": 8, "step": 1}),
                "audio_crossfade_ms": ("INT", {"default": 50, "min": 0, "max": 500, "step": 10}),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT", "AUDIO", "AUDIO")
    RETURN_NAMES = ("stitched_video", "trimmed_current_video", "stitched_audio", "trimmed_current_audio")
    FUNCTION = "stitch"
    CATEGORY = "MiniMaxH3/PrefixStream"

    def stitch(
        self,
        current_video: Dict[str, Any],
        previous_video: Optional[Dict[str, Any]] = None,
        current_audio: Optional[Dict[str, Any]] = None,
        previous_audio: Optional[Dict[str, Any]] = None,
        session: Optional[LongVideoSession] = None,
        cache_config: Optional[KVCacheConfig] = None,
        trim_frames: int = 0,
        latent_blend_steps: int = 2,
        audio_crossfade_ms: int = 50
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        curr_v, curr_a_latent = _unpack_latent(current_video)
        if curr_v is None:
            curr_v = current_video.get("samples")

        # Determine overlap steps
        overlap_steps = 0
        if trim_frames > 0:
            overlap_steps = pixel_frames_to_latent_steps(trim_frames)
        elif session is not None and session.last_rolling_steps > 0:
            overlap_steps = session.last_rolling_steps
        elif cache_config is not None:
            overlap_steps = cache_config.rolling_latent_frames

        # 1. Generate trimmed current video (pure new content)
        if overlap_steps > 0 and overlap_steps < curr_v.shape[2]:
            trimmed_curr_v = curr_v[:, :, overlap_steps:]
        else:
            trimmed_curr_v = curr_v

        # 2. Generate stitched video
        if previous_video is None:
            stitched_v = curr_v
        else:
            prev_v, _ = _unpack_latent(previous_video)
            if prev_v is None:
                prev_v = previous_video.get("samples")
            stitched_v = stitch_video_latents(
                prev_latent=prev_v,
                curr_latent=curr_v,
                overlap_steps=overlap_steps,
                blend_steps=latent_blend_steps
            )

        out_stitched_video = {"samples": stitched_v}
        out_trimmed_video = {"samples": trimmed_curr_v}

        # 3. Handle audio
        target_curr_audio = current_audio or ({"waveform": curr_a_latent, "sample_rate": 32000} if curr_a_latent is not None else None)
        out_stitched_audio = None
        out_trimmed_audio = None

        if target_curr_audio is not None and "waveform" in target_curr_audio:
            sr = int(target_curr_audio.get("sample_rate", 32000))
            # Trim audio
            if overlap_steps > 0:
                p_frames = latent_steps_to_pixel_frames(overlap_steps)
                trim_samples = int((p_frames / 24.0) * sr)
                w_trim = trim_audio_waveform(target_curr_audio["waveform"], trim_samples)
                out_trimmed_audio = {"waveform": w_trim, "sample_rate": sr}
            else:
                out_trimmed_audio = target_curr_audio

            # Stitch audio
            if previous_audio is None:
                out_stitched_audio = target_curr_audio
            else:
                w1 = previous_audio.get("waveform")
                w2 = out_trimmed_audio["waveform"] if out_trimmed_audio is not None else target_curr_audio["waveform"]
                if w1 is not None and w2 is not None:
                    cross_samples = int((audio_crossfade_ms / 1000.0) * sr)
                    stitched_w = audio_equal_power_crossfade(w1, w2, crossfade_samples=cross_samples)
                    out_stitched_audio = {"waveform": stitched_w, "sample_rate": sr}

        return (out_stitched_video, out_trimmed_video, out_stitched_audio, out_trimmed_audio)


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
