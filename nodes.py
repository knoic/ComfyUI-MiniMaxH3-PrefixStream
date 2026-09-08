"""ComfyUI custom nodes for MiniMax H3 masked AV continuation and chaining.

Exposes:
- MiniMaxPrefixCacheConfig: Select continuation mode and a visible-video context length.
- MiniMaxPrefixCacheApplier: Builds native AV masks or Safe Native fallback conditioning.
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
    from .pipeline.native_masked_av import apply_native_masked_av
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
        _standardize_image_tensor,
        _standardize_audio_dict,
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
    from pipeline.native_masked_av import apply_native_masked_av
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
        _standardize_image_tensor,
        _standardize_audio_dict,
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


class AnyType(str):
    """Wildcard type for ComfyUI input slots to accept multiple types (STRING, VHS_FILENAMES, etc.)."""
    def __ne__(self, __value: object) -> bool:
        return False

    def __eq__(self, __value: object) -> bool:
        return True


any_type = AnyType("*")


class MiniMaxPrefixCacheConfigNode:
    """Configures the user-facing continuation mode and video context length."""

    @classmethod
    def INPUT_TYPES(cls):

        return {
            "required": {
                "cache_mode": ([
                    "Native Masked AV (Recommended)",
                    "Safe Native (Fallback)"
                ], {
                    "default": "Native Masked AV (Recommended)",
                    "tooltip": "Native Masked AV 将上一段 AV latent 直接复制到目标开头，并用 ComfyUI 原生 video/audio denoise mask 分别保护；Safe Native 是短片段或旧工作流的兼容备选。"
                }),
                "continuation_frames": (["39", "90", "141", "192"], {
                    "default": "39",
                    "tooltip": "用户可见的视频续写上下文帧数。39 帧约 1.625 秒；较长选项会自动下取整到精确音视频公共边界。"
                }),
            }
        }

    RETURN_TYPES = ("MINIMAX_CACHE_CONFIG",)
    RETURN_NAMES = ("cache_config",)
    FUNCTION = "create_config"
    CATEGORY = "MiniMaxH3/PrefixStream"

    def create_config(
        self,
        cache_mode: str = "Native Masked AV (Recommended)",
        continuation_frames: Any = "39",
        **kwargs
    ) -> Tuple[KVCacheConfig]:
        # Accept a saved legacy rolling_frames value if an old workflow sends it.
        actual_rolling = kwargs.get("rolling_frames", continuation_frames)
        try:
            r_frames = int(actual_rolling)
        except (ValueError, TypeError):
            r_frames = 39

        config = KVCacheConfig(
            cache_mode=cache_mode,
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


def _pack_nested_streams(video: torch.Tensor, audio: torch.Tensor):
    """Pack two H3 streams without importing ComfyUI during standalone tests."""
    try:
        import comfy.nested_tensor
        return comfy.nested_tensor.NestedTensor((video, audio))
    except (ImportError, AttributeError):
        return (video, audio)


def _drop_head_keyframes(conditioning: Any, protected_frames: int):
    """Remove native keyframes that collide with the hard-protected masked head."""
    if not conditioning:
        return conditioning
    out = []
    for item in conditioning:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            out.append(item)
            continue
        embedding, extra = item
        updated = dict(extra)
        prior = updated.get("minimax_keyframes") or []
        updated["minimax_keyframes"] = [
            dict(kf) for kf in prior
            if float(kf.get("resolved_frame_index", 0)) >= float(protected_frames)
        ]
        out.append([embedding, updated])
    return out


def _require_native_masked_av_support() -> None:
    """Probe for ComfyUI's native MiniMax H3 per-stream mask implementation."""
    import inspect
    try:
        import comfy.model_base as model_base
        import comfy.ldm.minimax.model as h3_model
    except Exception as exc:
        raise RuntimeError(
            "Native Masked AV requires a current ComfyUI build with MiniMax H3 AV-mask support (PR #15375)."
        ) from exc

    base_cls = getattr(model_base, "MiniMaxH3", None)
    model_cls = getattr(h3_model, "MiniMaxH3Model", None)
    forward = getattr(model_cls, "forward", None) if model_cls is not None else None
    inner = getattr(model_cls, "_forward", None) if model_cls is not None else None
    scale = base_cls.__dict__.get("scale_latent_inpaint") if base_cls is not None else None
    extra_conds = getattr(base_cls, "extra_conds", None) if base_cls is not None else None

    def has_params(fn, *names):
        try:
            params = inspect.signature(fn).parameters
        except (TypeError, ValueError):
            return False
        return all(name in params for name in names)

    def code_names(fn):
        import types
        found = set()
        def walk(code):
            if not isinstance(code, types.CodeType):
                return
            found.update(code.co_names)
            found.update(value for value in code.co_consts if isinstance(value, str))
            for value in code.co_consts:
                if isinstance(value, types.CodeType):
                    walk(value)
        walk(getattr(fn, "__code__", None))
        return found

    extra_names = code_names(extra_conds) if callable(extra_conds) else set()

    available = all((
        base_cls is not None,
        callable(getattr(h3_model, "mask_row_values", None)),
        callable(forward) and has_params(forward, "denoise_mask", "audio_denoise_mask"),
        callable(inner) and has_params(inner, "denoise_mask", "audio_denoise_mask"),
        callable(base_cls.__dict__.get("_token_grid_masks")) if base_cls is not None else False,
        callable(base_cls.__dict__.get("_denoise_mask_conds")) if base_cls is not None else False,
        callable(scale) and has_params(scale, "x", "denoise_mask"),
        callable(extra_conds) and "denoise_mask" in extra_names and "_denoise_mask_conds" in extra_names,
    ))
    if not available:
        raise RuntimeError(
            "Native Masked AV requires a current ComfyUI build with MiniMax H3 AV-mask support from PR #15375. "
            "Update ComfyUI, restart it completely, and reload the workflow."
        )


class MiniMaxPrefixCacheApplierNode:
    """Builds Native Masked AV sampling input or Safe Native fallback conditioning."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "conditioning": ("CONDITIONING",),
            },
            "optional": {
                "cache_config": ("MINIMAX_CACHE_CONFIG",),
                "context_latent": ("LATENT", {"tooltip": "上一段完整的 H3 音视频 LATENT。首段生成时留空。"}),
                "target_latent": ("LATENT", {"tooltip": "连接 MiniMaxH3ReferenceToVideo 的目标 LATENT；Native Masked AV 会输出带独立音视频 noise_mask 的采样 latent。"}),
            }
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "MINIMAX_SESSION", "LATENT")
    RETURN_NAMES = ("model", "conditioning", "session", "masked_latent")
    FUNCTION = "apply_cache"
    CATEGORY = "MiniMaxH3/PrefixStream"

    def apply_cache(
        self,
        model: Any,
        conditioning: Any,
        cache_config: Optional[KVCacheConfig] = None,
        context_latent: Optional[Dict[str, Any]] = None,
        target_latent: Optional[Dict[str, Any]] = None,
        **legacy: Any,
    ) -> Tuple[Any, Any, LongVideoSession, Optional[Dict[str, Any]]]:
        cfg = cache_config or KVCacheConfig()
        session = legacy.get("session")
        sess = session or LongVideoSession(cfg)
        if session is not None and cache_config is not None:
            sess.config = cfg

        # Legacy aliases are accepted only for previously saved workflows; they
        # are intentionally absent from the user-facing node interface.
        ctx_target = context_latent if context_latent is not None else legacy.get("context_video_latent")

        # Unpack video and audio from latents (supports ComfyUI NestedTensor from H3ContinuousLoadLatent)
        v_ctx, a_ctx_from_latent = _unpack_latent(ctx_target)
        a_ctx = a_ctx_from_latent

        def safe_native_result(reason: Optional[Exception] = None):
            """Run the compatibility path when a native AV mask cannot be built."""
            if reason is not None:
                logger.warning(
                    "[Native Masked AV] %s Falling back to Safe Native for this clip; "
                    "use a source and target of at least 39 frames to enable native masks.",
                    reason,
                )
            patched_model, out_cond = sess.prepare_next_clip(
                model_patcher=model,
                conditioning=conditioning,
                previous_video_latent=v_ctx,
                previous_audio_latent=a_ctx,
                anchor_video_latent=None,
            )
            return (patched_model, out_cond, sess, target_latent)

        if cfg.is_native_masked_av_mode():
            if ctx_target is None:
                logger.info("[Native Masked AV] Initial clip: target latent passes through without a protected prefix.")
                return (model, conditioning, sess, target_latent)
            if target_latent is None:
                raise ValueError(
                    "Native Masked AV mode requires target_latent from MiniMaxH3ReferenceToVideo. "
                    "Connect the applier's masked_latent output to the sampler latent input."
                )
            if v_ctx is None or a_ctx_from_latent is None:
                raise ValueError("Native Masked AV requires a previous latent containing both video and audio streams")
            target_video, target_audio = _unpack_latent(target_latent)
            if target_video is None or target_audio is None:
                raise ValueError("Native Masked AV requires a target latent containing both video and audio streams")
            logger.info(
                "[Native Masked AV] Source %d video steps/%d frames, target %d video steps/%d frames.",
                v_ctx.shape[2], latent_steps_to_pixel_frames(v_ctx.shape[2]),
                target_video.shape[2], latent_steps_to_pixel_frames(target_video.shape[2]),
            )
            _require_native_masked_av_support()
            try:
                out_v, out_a, video_mask, audio_mask, plan = apply_native_masked_av(
                    target_video=target_video,
                    target_audio=target_audio,
                    source_video=v_ctx,
                    source_audio=a_ctx_from_latent,
                    context_frames=cfg.rolling_frames,
                    audio_tail_carryover="Full Previous Tail",
                    audio_feather_ticks=0,
                )
            except ValueError as exc:
                geometry_errors = (
                    "at least 39 source frames",
                    "consume the whole target",
                    "has no phase-aligned",
                )
                if any(text in str(exc) for text in geometry_errors):
                    return safe_native_result(exc)
                raise
            masked_latent = pack_av_latent(out_v, out_a, target_latent)
            masked_latent["noise_mask"] = _pack_nested_streams(video_mask, audio_mask)
            sess.last_rolling_steps = int(plan["context_steps"])
            sess.last_rolling_frames = int(plan["actual_context_frames"])
            out_cond = _drop_head_keyframes(conditioning, sess.last_rolling_frames)
            logger.info(
                "[Native Masked AV] Protected %d video frames/%d audio ticks; source latent %d:%d.",
                sess.last_rolling_frames,
                plan["audio_steps"],
                plan["start_t"],
                plan["end_t"],
            )
            return (model, out_cond, sess, masked_latent)

        # Safe Native fallback: inject grid-aligned keyframe conditioning only.
        return safe_native_result()


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
        actual_trim_frames = trim_frames
        if actual_trim_frames <= 0:
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


class MiniMaxCacheMonitorNode:
    """Provides operational telemetry for the current continuation session."""

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
        native_masked = session.config.is_native_masked_av_mode()
        status_str = "Native Masked AV" if native_masked else "Safe Native fallback"
        report_str = (
            f"=== MiniMax H3 Continuation Telemetry ===\n"
            f"Mode: {status_str}\n"
            f"Current Clip: #{session.current_clip_index}\n"
            f"Configured Protected Context: {session.config.rolling_frames} frames\n"
            f"Last Protected Context: {session.last_rolling_frames} frames ({session.last_rolling_steps} latent steps)\n"
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
                "latent": ("LATENT", {"tooltip": "【核心音画潜空间】采样器输出的原生联合音画 Latent (支持 MiniMax H3 官方 NestedTensor，包含完整视频与音频潜空间)"}),
                "project_name": ("STRING", {
                    "default": "Default_Project",
                    "tooltip": "【项目/分镜箱名称】指定当前镜头归档的项目库（例如：科幻短片、广告场景1）。不同项目之间素材完全隔离，方便多故事独立管理"
                }),
                "shot_tag": ("STRING", {
                    "default": "Auto (自动编号)",
                    "tooltip": "【镜头标签/备注】镜头的编号或简要动作描述（如：'Shot 1'、'男主回眸'、'远景空镜'）。填 'Auto (自动编号)' 时系统将根据项目内已有镜头数量自动顺延递增为 Shot 1, Shot 2..."
                }),
                "rating": ("INT", {
                    "default": 4, "min": 1, "max": 5, "step": 1,
                    "tooltip": "【镜头星标打分】1~5 星质量评级。后续使用 Clip Bin Picker 加载接力时，可按星级一键过滤掉废案镜头，只接力高分镜头"
                }),
            },
            "optional": {
                "images": ("IMAGE", {"tooltip": "【渲染像素画面】连接当前片段解码后的画面 (来自 VAEDecode 或 TrimPrefix)。连接后系统将自动截取真实的首帧与尾帧，生成超高清并排缩略图卡片！"}),
                "audio": ("AUDIO", {"tooltip": "【音频流】连接当前片段的音频 (来自 TrimPrefix 或 VAEDecodeAudio)。当自动编码保存 MP4 视频时，将作为音轨同步封装"}),
                "prompt": ("STRING", {"default": "", "tooltip": "【本段正向提示词】连接输入文本 (Input Text/Prompt)。自动入库保存到 meta.json，以便后续回顾镜头剧情与接力参考"}),
                "parent_clip_id": ("STRING", {"default": "", "tooltip": "【父镜头血缘ID】连接上一段 Clip Bin Picker 输出的 clip_id。用于在元数据中清晰记录多版本分支历史与承接血缘"}),
                "video_file_name": (any_type, {"default": "", "tooltip": "【关联合成视频名】连接当前片段合成保存节点 (VHS_VideoCombine) 的 Filenames 输出，或手动输入关联的 MP4 文件名，系统将自动将该视频归档到资产包中"}),
                "save_video": ("BOOLEAN", {"default": True, "tooltip": "【归档完整视频】是否在资产包内归档或编码生成完整 MP4 视频文件。开启后 Clip Bin Picker 画廊将支持悬停实时微动播放与声画视听弹窗！"}),
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
        audio: Optional[Dict[str, Any]] = None,
        prompt: str = "",
        parent_clip_id: str = "",
        video_file_name: Any = "",
        save_video: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        if latent is None:
            raise ValueError("MiniMaxClipBinSaver: 'latent' input is required.")

        video, audio_lat = _unpack_latent(latent)
        if video is None:
            raise ValueError("MiniMaxClipBinSaver: latent contains no video samples.")

        images = _standardize_image_tensor(images)
        audio = _standardize_audio_dict(audio)

        actual_shot = (shot_tag or "").strip()
        if actual_shot.startswith("Auto") or not actual_shot:
            idx = load_project_index(project_name)
            actual_shot = f"Shot {len(idx.get('clips', [])) + 1}"

        # Handle video_file_name if passed as list/tuple from VHS_VideoCombine Filenames
        resolved_video_name = ""
        def _extract_filename(val: Any) -> str:
            if isinstance(val, (list, tuple)):
                if not val:
                    return ""
                # Recursively inspect the last element (VHS format: [bool_or_subfolder, [paths...]])
                return _extract_filename(val[-1])
            return str(val).strip()

        if video_file_name is not None:
            resolved_video_name = _extract_filename(video_file_name)
            # If it's a full path, keep basename for friendly display
            if resolved_video_name:
                resolved_video_name = os.path.basename(resolved_video_name)

        meta_obj, clip_dir, preview_pil = save_clip_asset(
            video_tensor=video,
            audio_tensor=audio_lat,
            images=images,
            project_name=project_name,
            shot_tag=actual_shot,
            prompt=prompt if isinstance(prompt, str) else str(prompt),
            rating=rating,
            parent_clip_id=parent_clip_id if isinstance(parent_clip_id, str) else str(parent_clip_id),
            associated_video_path=resolved_video_name,
            raw_video_source=video_file_name,
            audio_dict=audio,
            save_video=save_video,
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

        logger.info("[Clip Bin Saver] Stored clip '%s' in '%s' (%s frames | ⭐%s | tag: %s | video: '%s')",
                    meta_obj.clip_id, project_name, meta_obj.frames, meta_obj.rating, actual_shot, resolved_video_name)

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
                    "tooltip": "【选择项目库】要读取素材的项目文件夹名称（如 Default_Project）。可在 ComfyUI 运行控制台查看已存在的项目名称列表"
                }),
                "mode": ([
                    "Auto (首段全新 / 后续自动接力)",
                    "Force Initial (强制新建首段，无上下文)",
                    "Strict Chaining (必须接力指定或最新镜头)"
                ], {
                    "default": "Auto (首段全新 / 后续自动接力)",
                    "tooltip": "【运行工作模式】\n• Auto（强烈推荐）：若项目库为空自动作为首段全新生成；后续运行时全自动接续上一段，无需任何拔线或手动操作！\n• Force Initial：强制开辟首段，忽略库内所有历史素材。\n• Strict Chaining：严格接力模式，库内无镜头时直接报错提示"
                }),
                "filter_rating": ([
                    "All (1-5 ⭐)",
                    "⭐⭐⭐+ (3+ ⭐)",
                    "⭐⭐⭐⭐+ (4+ ⭐)",
                    "⭐⭐⭐⭐⭐ (5 ⭐)"
                ], {
                    "default": "All (1-5 ⭐)",
                    "tooltip": "【星级过滤器】只读取大于等于该评级的镜头（如过滤掉 1~3 星的测试废案，只接续 4 星或 5 星的满意镜头）"
                }),
                "clip_selection": ("STRING", {
                    "default": "latest",
                    "tooltip": "【镜头定位】\n• 填 'latest'（默认）：自动调取最新生成的优质镜头进行无缝接续\n• 填 clip_id（如 clip_20260908...）：精确跳转或回溯到指定的历史镜头开启新分支\n• 填 shot 名称：按镜头标签名称匹配"
                }),
            },
            "optional": {
                "custom_clip_path": ("STRING", {
                    "default": "",
                    "tooltip": "【自定义物理路径覆盖】可选高级选项。填入绝对路径可直接载入任意磁盘目录下的 Clip Bin 镜头文件夹"
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

        prompt_str = meta_dict.get("prompt", "")
        frames = meta_dict.get("frames", latent_steps_to_pixel_frames(video.shape[2]))
        logger.info("[Clip Bin Picker] Loaded clip '%s' (%s frames | ⭐%s | tag: '%s')",
                    target_clip_id, frames, meta_dict.get("rating", 3), meta_dict.get("shot_tag", ""))

        return {
            "ui": {"images": []},
            "result": (out_latent, tail_tensor, first_tensor, prompt_str, target_clip_id)
        }


class MiniMaxSafeVAEDecodeNode:
    """Safe VAE Video Decoder that gracefully handles None in Initial Clip Mode.
    
    When samples is None (e.g. initial generation with zero prior context),
    it safely returns an empty IMAGE tensor instead of crashing with TypeError.
    When samples is present, it delegates to vae.decode() with full fidelity.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
            },
            "optional": {
                "samples": ("LATENT",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "MiniMaxH3/ClipBin"

    def decode(self, vae: Any, samples: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor]:
        if samples is None:
            logger.info("[Safe VAE Decode] No previous video latent provided (Initial Clip Mode). Passing through empty.")
            return (torch.empty((0, 768, 1344, 3), dtype=torch.float32),)

        v, _ = _unpack_latent(samples)
        if v is None:
            raw_s = samples.get("samples")
            if raw_s is None:
                return (torch.empty((0, 768, 1344, 3), dtype=torch.float32),)
            v = raw_s

        try:
            images = vae.decode(v)
            images = _standardize_image_tensor(images)
            return (images,)
        except Exception as e:
            logger.warning("[Safe VAE Decode] Failed to decode samples (%s), returning empty: %s", getattr(v, 'shape', None), e)
            return (torch.empty((0, 768, 1344, 3), dtype=torch.float32),)


class MiniMaxSafeVAEDecodeAudioNode:
    """Safe VAE Audio Decoder that gracefully handles None in Initial Clip Mode.
    
    When samples is None (e.g. initial generation with zero prior context),
    it safely returns an empty AUDIO dict instead of crashing with TypeError.
    When samples is present, it delegates to vae.decode() with full audio fidelity.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
            },
            "optional": {
                "samples": ("LATENT",),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "decode"
    CATEGORY = "MiniMaxH3/ClipBin"

    def decode(self, vae: Any, samples: Optional[Dict[str, Any]] = None) -> Tuple[Optional[Dict[str, Any]]]:
        if samples is None:
            logger.info("[Safe VAE Decode Audio] No previous audio latent provided (Initial Clip Mode). Passing through None.")
            return (None,)

        _, a = _unpack_latent(samples)
        if a is None:
            # Check if samples itself is an audio latent or dictionary
            raw_s = samples.get("samples")
            if raw_s is not None and hasattr(raw_s, "ndim") and raw_s.ndim <= 4:
                a = raw_s
            else:
                logger.info("[Safe VAE Decode Audio] No audio stream found in latent. Passing through None.")
                return (None,)

        try:
            audio = vae.decode(a)
            audio = _standardize_audio_dict(audio)
            return (audio,)
        except Exception as e:
            logger.warning("[Safe VAE Decode Audio] Failed to decode audio (%s), returning None: %s", getattr(a, 'shape', None), e)
            return (None,)


NODE_CLASS_MAPPINGS = {
    "MiniMaxPrefixCacheConfig": MiniMaxPrefixCacheConfigNode,
    "MiniMaxPrefixCacheApplier": MiniMaxPrefixCacheApplierNode,
    "MiniMaxTrimPrefix": MiniMaxTrimPrefixLatentNode,
    "MiniMaxTrimPrefixLatent": MiniMaxTrimPrefixLatentNode,
    "MiniMaxCacheMonitor": MiniMaxCacheMonitorNode,
    "MiniMaxSaveLatent": MiniMaxSaveLatentNode,
    "MiniMaxLoadLatent": MiniMaxLoadLatentNode,
    "MiniMaxClipBinSaver": MiniMaxClipBinSaverNode,
    "MiniMaxClipBinPicker": MiniMaxClipBinPickerNode,
    "MiniMaxSafeVAEDecode": MiniMaxSafeVAEDecodeNode,
    "MiniMaxSafeVAEDecodeAudio": MiniMaxSafeVAEDecodeAudioNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxPrefixCacheConfig": "MiniMax H3 Continuation Config",
    "MiniMaxPrefixCacheApplier": "MiniMax H3 Continuation Applier",
    "MiniMaxTrimPrefix": "MiniMax H3 Trim Prefix (AV Master, Zero Flicker)",
    "MiniMaxTrimPrefixLatent": "MiniMax H3 Trim Prefix Latent (AV Master)",
    "MiniMaxCacheMonitor": "MiniMax H3 Cache Telemetry Monitor",
    "MiniMaxSaveLatent": "MiniMax H3 Save AV Latent (Standalone)",
    "MiniMaxLoadLatent": "MiniMax H3 Load AV Latent (Standalone)",
    "MiniMaxClipBinSaver": "MiniMax H3 Clip Bin Saver (Media Pool)",
    "MiniMaxClipBinPicker": "MiniMax H3 Clip Bin Picker (Gallery Loader)",
    "MiniMaxSafeVAEDecode": "MiniMax H3 Safe VAE Decode (Video)",
    "MiniMaxSafeVAEDecodeAudio": "MiniMax H3 Safe VAE Decode (Audio)",
}


