"""ComfyUI Custom Nodes for MiniMax H3 Prefix KV Caching & Streaming Chaining.

Exposes:
- MiniMaxPrefixCacheConfig: Configure FP8/BF16, GPU/CPU Pinned, Anchor & Rolling parameters.
- MiniMaxPrefixCacheApplier: Connects model, historical context latents, and hooks into KSampler.
- MiniMaxLongVideoStitcher: Smoothly stitches video latents and cross-fades audio between clips.
- MiniMaxCacheMonitor: Real-time diagnostics for VRAM, memory footprint, and session progress.
"""

from typing import Dict, Any, Tuple, Optional
import torch

try:
    from .engine.cache_manager import KVCacheConfig, PrefixKVCacheManager
    from .pipeline.long_video_director import LongVideoSession
    from .pipeline.seam_protector import audio_equal_power_crossfade, latent_soft_blend
except (ImportError, ValueError):
    from engine.cache_manager import KVCacheConfig, PrefixKVCacheManager
    from pipeline.long_video_director import LongVideoSession
    from pipeline.seam_protector import audio_equal_power_crossfade, latent_soft_blend


class MiniMaxPrefixCacheConfigNode:
    """Configures Prefix KV Cache precision, memory strategy, and window sizes."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cache_dtype": (["fp8", "bf16", "fp16"], {"default": "fp8"}),
                "device_mode": (["auto", "gpu", "cpu_pinned"], {"default": "auto"}),
                "use_anchor": ("BOOLEAN", {"default": True}),
                "anchor_latent_frames": ("INT", {"default": 2, "min": 1, "max": 10, "step": 1}),
                "rolling_latent_frames": ("INT", {"default": 6, "min": 2, "max": 16, "step": 1}),
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
        anchor_latent_frames: int,
        rolling_latent_frames: int
    ) -> Tuple[KVCacheConfig]:
        config = KVCacheConfig(
            cache_dtype=cache_dtype,
            device_mode=device_mode,
            use_anchor=use_anchor,
            anchor_latent_frames=anchor_latent_frames,
            rolling_latent_frames=rolling_latent_frames
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


class MiniMaxLongVideoStitcherNode:
    """Seamlessly joins adjacent video latents and crossfades audio waveforms."""

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
                "latent_blend_steps": ("INT", {"default": 2, "min": 0, "max": 6, "step": 1}),
                "audio_crossfade_ms": ("INT", {"default": 50, "min": 0, "max": 500, "step": 10}),
            }
        }

    RETURN_TYPES = ("LATENT", "AUDIO")
    RETURN_NAMES = ("stitched_video", "stitched_audio")
    FUNCTION = "stitch"
    CATEGORY = "MiniMaxH3/PrefixStream"

    def stitch(
        self,
        current_video: Dict[str, Any],
        previous_video: Optional[Dict[str, Any]] = None,
        current_audio: Optional[Dict[str, Any]] = None,
        previous_audio: Optional[Dict[str, Any]] = None,
        latent_blend_steps: int = 2,
        audio_crossfade_ms: int = 50
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        curr_v = current_video["samples"]

        if previous_video is None:
            # First clip, passthrough
            stitched_v = curr_v
        else:
            prev_v = previous_video["samples"]
            stitched_v = latent_soft_blend(prev_v, curr_v, blend_steps=latent_blend_steps)

        out_video = {"samples": stitched_v}

        # Handle audio stitching
        out_audio = None
        if current_audio is not None:
            if previous_audio is None:
                out_audio = current_audio
            else:
                sr = int(current_audio.get("sample_rate", 32000))
                cross_samples = int((audio_crossfade_ms / 1000.0) * sr)
                w1 = previous_audio["waveform"]
                w2 = current_audio["waveform"]
                stitched_w = audio_equal_power_crossfade(w1, w2, crossfade_samples=cross_samples)
                out_audio = {"waveform": stitched_w, "sample_rate": sr}

        return (out_video, out_audio)


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
            f"Anchor Latent Frames: {session.config.anchor_latent_frames}\n"
            f"Rolling Latent Frames: {session.config.rolling_latent_frames}\n"
            f"Accumulated Clips: {len(session.accumulated_video_latents)}"
        )
        return (report_str,)


NODE_CLASS_MAPPINGS = {
    "MiniMaxPrefixCacheConfig": MiniMaxPrefixCacheConfigNode,
    "MiniMaxPrefixCacheApplier": MiniMaxPrefixCacheApplierNode,
    "MiniMaxLongVideoStitcher": MiniMaxLongVideoStitcherNode,
    "MiniMaxCacheMonitor": MiniMaxCacheMonitorNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxPrefixCacheConfig": "MiniMax H3 Prefix Cache Config",
    "MiniMaxPrefixCacheApplier": "MiniMax H3 Prefix Cache Applier",
    "MiniMaxLongVideoStitcher": "MiniMax H3 Long Video Stitcher",
    "MiniMaxCacheMonitor": "MiniMax H3 Cache Telemetry Monitor",
}
