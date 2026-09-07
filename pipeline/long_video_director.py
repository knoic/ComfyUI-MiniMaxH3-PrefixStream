"""Long Video Director & Streaming Chaining Pipeline for MiniMax H3.

Coordinates multi-segment long video generation with Dual-Tier Prefix KV Caching
(Permanent Anchor + Dynamic Rolling), timeline advancement, and audio-video stitching.
"""

from typing import Optional, Dict, Any, List, Tuple
import logging
import torch

try:
    from ..engine.cache_manager import PrefixKVCacheManager, KVCacheConfig
    from ..engine.rope_aligner import TemporalCursorTracker
    from ..engine.warmup_executor import WarmupExecutor
    from ..engine.block_hook import create_prefix_dit_hook
    from .seam_protector import audio_equal_power_crossfade, latent_soft_blend, trim_prefix_frames
except (ImportError, ValueError):
    from engine.cache_manager import PrefixKVCacheManager, KVCacheConfig
    from engine.rope_aligner import TemporalCursorTracker
    from engine.warmup_executor import WarmupExecutor
    from engine.block_hook import create_prefix_dit_hook
    from pipeline.seam_protector import audio_equal_power_crossfade, latent_soft_blend, trim_prefix_frames

logger = logging.getLogger("minimax_prefix_stream")


class LongVideoSession:
    """Manages state across infinite clips in an H3 video generation stream."""

    def __init__(self, config: Optional[KVCacheConfig] = None):
        self.config = config or KVCacheConfig()
        self.cache_manager = PrefixKVCacheManager(self.config)
        self.cursor_tracker = TemporalCursorTracker()
        self.warmup_executor = WarmupExecutor(self.cache_manager)

        self.current_clip_index: int = 0
        self.accumulated_video_latents: List[torch.Tensor] = []
        self.accumulated_audio_latents: List[torch.Tensor] = []

    def prepare_next_clip(
        self,
        model_patcher: Any,
        previous_video_latent: Optional[torch.Tensor] = None,
        previous_audio_latent: Optional[torch.Tensor] = None,
        anchor_video_latent: Optional[torch.Tensor] = None,
        text_context: Optional[Any] = None
    ) -> Any:
        """Prepares the ComfyUI model patcher for the upcoming clip generation.

        1. If first clip and anchor provided: warmup anchor KV.
        2. If subsequent clip: warmup rolling KV from previous clip's tail frames.
        3. Injects denoising block patches into transformer_options.
        """
        # Phase 0: Anchor Warmup (once at clip 0)
        if self.current_clip_index == 0 and anchor_video_latent is not None and self.config.use_anchor:
            logger.info("Initializing World Origin Anchor KV from initial keyframe...")
            self.warmup_executor.precompute_anchor(
                model_patcher=model_patcher,
                anchor_video_latent=anchor_video_latent,
                text_context=text_context
            )

        # Phase 0: Rolling Warmup (for clip >= 1)
        if self.current_clip_index > 0 and previous_video_latent is not None:
            # Extract tail frames for rolling window
            rolling_steps = min(self.config.rolling_latent_frames, previous_video_latent.shape[2])
            tail_video = previous_video_latent[:, :, -rolling_steps:]
            tail_audio = None
            if previous_audio_latent is not None:
                # Audio runs at 40Hz (~1.6x video latent rate)
                audio_steps = min(int(rolling_steps * 1.6), previous_audio_latent.shape[-1])
                tail_audio = previous_audio_latent[..., -audio_steps:]

            logger.info("Extracting %d rolling latent frames for motion continuity...", rolling_steps)
            self.warmup_executor.precompute_rolling(
                model_patcher=model_patcher,
                prefix_video_latent=tail_video,
                prefix_audio_latent=tail_audio,
                text_context=text_context
            )

        # Attach Denoising hooks to model patcher
        patched_model = self._attach_denoise_hooks(model_patcher)
        return patched_model

    def _attach_denoise_hooks(self, model_patcher: Any) -> Any:
        """Injects Phase 1 Denoising Hook into model patcher's transformer_options."""
        # Create clone of model patcher if supported by ComfyUI
        if hasattr(model_patcher, "clone"):
            patched = model_patcher.clone()
        else:
            patched = model_patcher

        # Configure transformer options
        opts = getattr(patched, "model_options", {}).copy()
        transformer_options = opts.get("transformer_options", {}).copy()
        transformer_options["minimax_prefix_mode"] = "denoise"

        patches_replace = transformer_options.get("patches_replace", {}).copy()
        dit_patches = patches_replace.get("dit", {}).copy()

        # Generate block hooks
        new_hooks = create_prefix_dit_hook(
            cache_manager=self.cache_manager,
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

        # Trim overlap frames if requested
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
