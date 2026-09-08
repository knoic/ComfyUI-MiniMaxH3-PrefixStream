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

        # Safe Native fallback uses in-timeline native keyframe conditioning.
        updated_conditioning = inject_minimax_keyframes(
            conditioning=conditioning,
            video_tail=tail_video,
            audio_tail=tail_audio,
            anchor_video=anchor_video_latent if self.config.use_anchor else None
        )

        logger.info("Operating in Safe Native fallback mode with native ComfyUI attention and keyframes.")
        return model_patcher, updated_conditioning

    def commit_generated_clip(
        self,
        video_latent: torch.Tensor,
        audio_latent: Optional[torch.Tensor] = None,
        trim_prefix: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Registers a completed clip into session history and advances clip index."""
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
