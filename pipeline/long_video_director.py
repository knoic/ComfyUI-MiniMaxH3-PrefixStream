"""Long Video Director & Streaming Chaining Pipeline for MiniMax H3.

Coordinates multi-segment long video generation with Dual-Tier Prefix KV Caching
(Permanent Anchor + Dynamic Rolling), timeline advancement, and audio-video stitching.
"""

from typing import Optional, Dict, Any, List, Tuple
import logging
import torch

try:
    from ..engine.cache_manager import (
        PrefixKVCacheManager,
        KVCacheConfig,
        latent_steps_to_pixel_frames,
        pixel_frames_to_latent_steps,
        snap_to_run_grid,
        steps_for_frames,
        VIDEO_RUN_GRID,
    )
    from ..engine.rope_aligner import TemporalCursorTracker
    from ..engine.warmup_executor import WarmupExecutor
    from ..engine.block_hook import create_prefix_dit_hook
    from .seam_protector import audio_equal_power_crossfade, latent_soft_blend, trim_prefix_frames
except (ImportError, ValueError):
    from engine.cache_manager import (
        PrefixKVCacheManager,
        KVCacheConfig,
        latent_steps_to_pixel_frames,
        pixel_frames_to_latent_steps,
        snap_to_run_grid,
        steps_for_frames,
        VIDEO_RUN_GRID,
    )
    from engine.rope_aligner import TemporalCursorTracker
    from engine.warmup_executor import WarmupExecutor
    from engine.block_hook import create_prefix_dit_hook
    from pipeline.seam_protector import audio_equal_power_crossfade, latent_soft_blend, trim_prefix_frames

logger = logging.getLogger("minimax_prefix_stream")

FRAME_PER_TOKEN = (1, 4, 4, 4, 4)


def step_offsets(latent_t: int) -> List[int]:
    """Pixel-frame index at which each latent step begins."""
    out, acc = [], 0
    for k in range(latent_t):
        out.append(acc)
        acc += FRAME_PER_TOKEN[k % 5]
    return out


def inject_minimax_keyframes(
    conditioning: List[Any],
    video_tail: Optional[torch.Tensor] = None,
    audio_tail: Optional[torch.Tensor] = None,
    anchor_video: Optional[torch.Tensor] = None
) -> List[Any]:
    """Injects keyframe anchors into MiniMax H3 conditioning payload so the model binds context.

    CRITICAL FIX FOR FLICKERING & COLOR SHIFT:
    When video_tail is injected, it defines the start of the timeline [0..head_end).
    Any prior keyframe (such as the initial prompt image at frame 0 from MiniMaxH3ReferenceToVideo)
    residing within [0..head_end) MUST be dropped! If both exist at frame 0, the model receives
    two conflicting conditioning latents at the exact same 3D-RoPE coordinates, causing violent
    diffusion trajectory oscillation, flashing, and severe color shifts.
    """
    if not conditioning:
        return conditioning

    keyframes = []
    head_end = 0

    # 1. Rolling Context Keyframes (Priority: Head of current clip continues tail of previous clip)
    if video_tail is not None:
        tail_t = video_tail.shape[2]
        tail_offsets = step_offsets(tail_t)
        head_end = latent_steps_to_pixel_frames(tail_t)
        for k in range(tail_t):
            keyframes.append({
                "resolved_frame_index": tail_offsets[k],
                "latent": video_tail[:, :, k:k+1].clone()
            })

        if audio_tail is not None:
            # End-align the audio window with the pinned video (ending at head_end)
            rt = int(audio_tail.shape[-1])
            # 40Hz audio step to 24fps pixel frame coordinate conversion: 5/3
            audio_start_frame = float(head_end) - (rt / (5.0 / 3.0))
            keyframes.append({
                "resolved_frame_index": audio_start_frame,
                "audio_latent": audio_tail.clone()
            })

    # 2. Origin Anchor Keyframes (Only used if no rolling tail is present, e.g. initial generation)
    elif anchor_video is not None:
        anc_t = anchor_video.shape[2]
        anc_offsets = step_offsets(anc_t)
        head_end = latent_steps_to_pixel_frames(anc_t)
        for k in range(anc_t):
            keyframes.append({
                "resolved_frame_index": anc_offsets[k],
                "latent": anchor_video[:, :, k:k+1].clone()
            })

    if not keyframes:
        return conditioning

    out_cond = []
    dropped_count = 0
    for emb, extra in conditioning:
        d = extra.copy()
        prior = d.get("minimax_keyframes") or []
        kept = []
        for kf in prior:
            p = kf.get("resolved_frame_index", 0)
            # Drop any prior anchor that falls inside [0..head_end) to prevent double-anchor collision!
            if p < head_end:
                dropped_count += 1
                logger.info(
                    "Dropped conflicting prior keyframe at frame %s (< head_end %s) to prevent frame-0 collision.",
                    p, head_end
                )
                continue
            kept.append(dict(kf))
        d["minimax_keyframes"] = kept + keyframes
        out_cond.append([emb, d])

    logger.info(
        "Conditioning updated: injected %d keyframes (span [0..%d]), dropped %d conflicting prior keyframes.",
        len(keyframes), head_end, dropped_count
    )
    return out_cond


class LongVideoSession:
    """Manages state across infinite clips in an H3 video generation stream."""

    def __init__(self, config: Optional[KVCacheConfig] = None):
        self.config = config or KVCacheConfig()
        self.cache_manager = PrefixKVCacheManager(self.config)
        self.cursor_tracker = TemporalCursorTracker()
        self.warmup_executor = WarmupExecutor(self.cache_manager)

        self.current_clip_index: int = 0
        self.last_rolling_steps: int = 0
        self.last_rolling_frames: int = 0
        self.accumulated_video_latents: List[torch.Tensor] = []
        self.accumulated_audio_latents: List[torch.Tensor] = []

    def prepare_next_clip(
        self,
        model_patcher: Any,
        conditioning: List[Any],
        previous_video_latent: Optional[torch.Tensor] = None,
        previous_audio_latent: Optional[torch.Tensor] = None,
        anchor_video_latent: Optional[torch.Tensor] = None,
        text_context: Optional[Any] = None
    ) -> Tuple[Any, List[Any]]:
        """Prepares model patcher and conditioning for the upcoming clip generation.

        1. Slices grid-aligned rolling context from previous clip latent.
        2. Injects keyframe anchors into conditioning while purging colliding prior anchors.
        3. Configures model: in Safe Native mode, retains 100% native ComfyUI DiT attention.
        """
        # Reset cache manager state for upcoming clip's Step-1 online capture if enabled
        self.cache_manager.reset_for_next_clip()

        tail_video = None
        tail_audio = None
        if previous_video_latent is not None:
            # Snap requested rolling frames to VIDEO_RUN_GRID (5, 22, 39, 56, 73, 90, 107, 124)
            snapped_frames = snap_to_run_grid(self.config.rolling_frames)
            total_steps = previous_video_latent.shape[2]
            rolling_steps = min(pixel_frames_to_latent_steps(snapped_frames), total_steps)

            # Ensure start lands on cycle position 0 (start % 5 == 0) for consistent (1,4,4,4,4) phase
            start = total_steps - rolling_steps
            if start % 5 != 0:
                adjusted_steps = rolling_steps - (start % 5)
                if adjusted_steps > 0:
                    rolling_steps = adjusted_steps
                    start = total_steps - rolling_steps

            self.last_rolling_steps = rolling_steps
            self.last_rolling_frames = latent_steps_to_pixel_frames(rolling_steps)
            tail_video = previous_video_latent[:, :, start:start + rolling_steps].clone()

            if previous_audio_latent is not None:
                # Audio runs at 40Hz, video at 24fps -> 40/24 = 5/3
                audio_steps = int(round(self.last_rolling_frames / 24.0 * 40.0))
                total_a = previous_audio_latent.shape[-1]
                audio_steps = min(audio_steps, total_a)
                tail_audio = previous_audio_latent[..., total_a - audio_steps:].clone()

            logger.info(
                "Configured Grid-Aligned Rolling Context: %d frames (%d latent steps, start %d @ phase %d).",
                self.last_rolling_frames, rolling_steps, start, start % 5
            )

        # Branch A: Decoupled Pure Prefix Mode (Zero Timeline Overlap, Prompt-Aligned)
        if self.config.is_decoupled_mode():
            if tail_video is not None and not self.cache_manager.has_cache(0):
                logger.info("[Decoupled Pure Prefix] Extracting pure prefix KV from previous clip tail...")
                self.warmup_executor.precompute_rolling(
                    model_patcher=model_patcher,
                    prefix_video_latent=tail_video,
                    prefix_audio_latent=tail_audio,
                    text_context=text_context
                )

            # In decoupled mode, NEVER inject keyframes into timeline conditioning!
            # The target timeline starts purely at t=0, fully aligned with user prompt.
            updated_conditioning = conditioning
            logger.info("[Decoupled Pure Prefix] Zero-overlap mode: timeline kept clean at t=0 (no minimax_keyframes injected).")

            patched_model = self._attach_decoupled_hooks(model_patcher)
            return patched_model, updated_conditioning

        # Branch B: Standard In-Timeline Overlap (Safe Native / Step-1 Dynamic Cache)
        # Inject keyframes into conditioning payload with conflict resolution
        updated_conditioning = inject_minimax_keyframes(
            conditioning=conditioning,
            video_tail=tail_video,
            audio_tail=tail_audio,
            anchor_video=anchor_video_latent if self.config.use_anchor else None
        )

        # Select mode: Safe Native (Recommended) vs Step-1 Dynamic Cache (Experimental)
        if self.config.is_cache_enabled():
            logger.info("Applying Step-1 Dynamic KV Caching Hook to DiT blocks.")
            patched_model = self._attach_denoise_hooks(model_patcher)
        else:
            logger.info("Operating in Safe Native Mode: 100% native ComfyUI attention with zero DiT patching (guaranteed zero flicker).")
            patched_model = model_patcher

        return patched_model, updated_conditioning

    def _attach_decoupled_hooks(self, model_patcher: Any) -> Any:
        """Injects Decoupled Pure Prefix Hook into model patcher's transformer_options."""
        if hasattr(model_patcher, "clone"):
            patched = model_patcher.clone()
        else:
            patched = model_patcher

        opts = getattr(patched, "model_options", {}).copy()
        transformer_options = opts.get("transformer_options", {}).copy()
        transformer_options["minimax_prefix_mode"] = "decoupled_pure"

        patches_replace = transformer_options.get("patches_replace", {}).copy()
        dit_patches = patches_replace.get("dit", {}).copy()

        new_hooks = create_prefix_dit_hook(
            cache_manager=self.cache_manager,
            model=getattr(patched, "model", patched),
            is_anchor_warmup=False,
            is_rolling_warmup=False
        )
        dit_patches.update(new_hooks)
        patches_replace["dit"] = dit_patches
        transformer_options["patches_replace"] = patches_replace

        opts["transformer_options"] = transformer_options
        if hasattr(patched, "model_options"):
            patched.model_options = opts

        return patched

    def _attach_denoise_hooks(self, model_patcher: Any) -> Any:
        """Injects Phase 1 Denoising Hook into model patcher's transformer_options."""
        if hasattr(model_patcher, "clone"):
            patched = model_patcher.clone()
        else:
            patched = model_patcher

        opts = getattr(patched, "model_options", {}).copy()
        transformer_options = opts.get("transformer_options", {}).copy()
        transformer_options["minimax_prefix_mode"] = "denoise"

        patches_replace = transformer_options.get("patches_replace", {}).copy()
        dit_patches = patches_replace.get("dit", {}).copy()

        # Generate block hooks with model instance passed for direct module resolution
        new_hooks = create_prefix_dit_hook(
            cache_manager=self.cache_manager,
            model=getattr(patched, "model", patched),
            is_anchor_warmup=False,
            is_rolling_warmup=False
        )
        dit_patches.update(new_hooks)
        patches_replace["dit"] = dit_patches
        transformer_options["patches_replace"] = patches_replace

        opts["transformer_options"] = transformer_options
        if hasattr(patched, "model_options"):
            patched.model_options = opts

        return patched

    def commit_generated_clip(
        self,
        video_latent: torch.Tensor,
        audio_latent: Optional[torch.Tensor] = None,
        trim_prefix: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Registers a completed clip into session history and advances clip index."""
        if self.config.is_decoupled_mode():
            # In decoupled mode, generated clip is already 100% pure target without overlap frames
            delivered_video = video_latent
            delivered_audio = audio_latent
            self.accumulated_video_latents.append(delivered_video)
            if delivered_audio is not None:
                self.accumulated_audio_latents.append(delivered_audio)
            self.current_clip_index += 1
            logger.info("Committed Decoupled Clip #%d (0 overlap frames). Total accumulated: %d",
                        self.current_clip_index, len(self.accumulated_video_latents))
            return delivered_video, delivered_audio

        rolling_steps = self.config.rolling_latent_frames if self.current_clip_index > 0 else 0

        delivered_video = trim_prefix_frames(video_latent, rolling_steps) if trim_prefix else video_latent
        self.accumulated_video_latents.append(delivered_video)

        if audio_latent is not None:
            audio_steps = int(rolling_steps * 1.6) if trim_prefix else 0
            delivered_audio = audio_latent[..., audio_steps:] if audio_steps > 0 else audio_latent
            self.accumulated_audio_latents.append(delivered_audio)
        else:
            delivered_audio = None

        self.current_clip_index += 1
        logger.info("Committed Clip #%d. Total accumulated segments: %d",
                    self.current_clip_index, len(self.accumulated_video_latents))
        return delivered_video, delivered_audio

    def reset(self):
        """Clears all caches and restarts session from scratch."""
        self.cache_manager.clear_all()
        self.cursor_tracker.reset()
        self.current_clip_index = 0
        self.accumulated_video_latents.clear()
        self.accumulated_audio_latents.clear()
