"""Long Video Director & Streaming Chaining Pipeline for MiniMax H3.

Coordinates multi-segment long video generation with Dual-Tier Prefix KV Caching
(Permanent Anchor + Dynamic Rolling), timeline advancement, and audio-video stitching.
"""

from typing import Optional, Dict, Any, List, Tuple
import logging
import torch

try:
    from ..engine.cache_manager import PrefixKVCacheManager, KVCacheConfig, latent_steps_to_pixel_frames, pixel_frames_to_latent_steps
    from ..engine.rope_aligner import TemporalCursorTracker
    from ..engine.warmup_executor import WarmupExecutor
    from ..engine.block_hook import create_prefix_dit_hook
    from .seam_protector import audio_equal_power_crossfade, latent_soft_blend, trim_prefix_frames
except (ImportError, ValueError):
    from engine.cache_manager import PrefixKVCacheManager, KVCacheConfig, latent_steps_to_pixel_frames, pixel_frames_to_latent_steps
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
    """Injects keyframe anchors into MiniMax H3 conditioning payload so the model binds context."""
    if not conditioning:
        return conditioning

    keyframes = []

    # 1. Anchor Keyframes (Frame 0 World Origin)
    if anchor_video is not None:
        anc_t = anchor_video.shape[2]
        anc_offsets = step_offsets(anc_t)
        for k in range(anc_t):
            keyframes.append({
                "resolved_frame_index": anc_offsets[k],
                "latent": anchor_video[:, :, k:k+1].clone()
            })

    # 2. Rolling Context Keyframes (Motion Continuity)
    if video_tail is not None:
        tail_t = video_tail.shape[2]
        tail_offsets = step_offsets(tail_t)
        for k in range(tail_t):
            keyframes.append({
                "resolved_frame_index": tail_offsets[k],
                "latent": video_tail[:, :, k:k+1].clone()
            })

        if audio_tail is not None:
            keyframes.append({
                "resolved_frame_index": 0,
                "audio_latent": audio_tail.clone()
            })

    if not keyframes:
        return conditioning

    out_cond = []
    for emb, extra in conditioning:
        d = extra.copy()
        prior = d.get("minimax_keyframes") or []
        d["minimax_keyframes"] = prior + keyframes
        out_cond.append([emb, d])

    logger.info("Injected %d MiniMax keyframes into conditioning payload.", len(keyframes))
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

        1. Warms up Anchor KV (if provided).
        2. Warms up Rolling KV (if context provided).
        3. Injects keyframe anchors into conditioning payload.
        4. Injects denoising block patches into model_options.
        """
        # Reset cache manager state for upcoming clip's Step-1 online capture
        self.cache_manager.reset_for_next_clip()

        tail_video = None
        tail_audio = None
        if previous_video_latent is not None:
            rolling_steps = min(self.config.rolling_latent_frames, previous_video_latent.shape[2])
            self.last_rolling_steps = rolling_steps
            self.last_rolling_frames = latent_steps_to_pixel_frames(rolling_steps)
            tail_video = previous_video_latent[:, :, -rolling_steps:]
            if previous_audio_latent is not None:
                audio_steps = min(int(rolling_steps * 1.6), previous_audio_latent.shape[-1])
                tail_audio = previous_audio_latent[..., -audio_steps:]

            logger.info(
                "Configured Rolling Context with %d latent steps (~%d frames, ~%.2fs) for Step-1 Dynamic Capture.",
                rolling_steps, self.last_rolling_frames, self.last_rolling_frames / 24.0
            )

        # Inject keyframes into conditioning payload so the model is bound to the context
        updated_conditioning = inject_minimax_keyframes(
            conditioning=conditioning,
            video_tail=tail_video,
            audio_tail=tail_audio,
            anchor_video=anchor_video_latent if self.config.use_anchor else None
        )

        # Attach Denoising hooks to model patcher
        patched_model = self._attach_denoise_hooks(model_patcher)
        return patched_model, updated_conditioning

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
