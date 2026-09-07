"""Phase 0: Warmup Executor for Precomputing Prefix KV Cache.

Performs a single, offline forward pass on prefix/historical latent frames
with timestep pinned at clean state (t=0.999), populating the 50-layer Key/Value
cache before diffusion sampling starts.
"""

from typing import Optional, Dict, Any
import logging
import torch

try:
    from .cache_manager import PrefixKVCacheManager
    from .block_hook import create_prefix_dit_hook
except (ImportError, ValueError):
    from cache_manager import PrefixKVCacheManager
    from block_hook import create_prefix_dit_hook

logger = logging.getLogger("minimax_prefix_stream")


class WarmupExecutor:
    """Executes single-shot precomputation of Prefix Key-Value caches."""

    def __init__(self, cache_manager: PrefixKVCacheManager):
        self.cache_manager = cache_manager

    def precompute_anchor(
        self,
        model_patcher: Any,
        anchor_video_latent: torch.Tensor,
        anchor_audio_latent: Optional[torch.Tensor] = None,
        text_context: Optional[Any] = None
    ) -> bool:
        """Precomputes and stores the permanent Anchor Key/Value (World Origin / Character Anchor)."""
        logger.info("Executing Phase 0 Anchor Warmup across 50 DiT layers...")
        return self._run_warmup_forward(
            model_patcher=model_patcher,
            video_latent=anchor_video_latent,
            audio_latent=anchor_audio_latent,
            text_context=text_context,
            is_anchor=True
        )

    def precompute_rolling(
        self,
        model_patcher: Any,
        prefix_video_latent: torch.Tensor,
        prefix_audio_latent: Optional[torch.Tensor] = None,
        text_context: Optional[Any] = None
    ) -> bool:
        """Precomputes and stores the Rolling Key/Value (Dynamic Motion Continuity from previous clip)."""
        logger.info("Executing Phase 0 Rolling Warmup across 50 DiT layers...")
        return self._run_warmup_forward(
            model_patcher=model_patcher,
            video_latent=prefix_video_latent,
            audio_latent=prefix_audio_latent,
            text_context=text_context,
            is_anchor=False
        )

    def _run_warmup_forward(
        self,
        model_patcher: Any,
        video_latent: torch.Tensor,
        audio_latent: Optional[torch.Tensor],
        text_context: Optional[Any],
        is_anchor: bool
    ) -> bool:
        """Executes a single forward pass with warmup hooks attached."""
        try:
            # Prepare dummy or zero audio latent if None
            if audio_latent is None:
                # [1, 32, 2, T_audio]
                t_audio = max(1, int(video_latent.shape[2] * 1.6))
                audio_latent = torch.zeros(
                    (1, 32, 2, t_audio),
                    device=video_latent.device,
                    dtype=video_latent.dtype
                )

            # Build model input pair [video, audio]
            x_in = [video_latent, audio_latent]

            # Timestep for condition frames is pinned near 1.0 (sigma near 0.001)
            sigma = torch.tensor([0.001], device=video_latent.device)

            # Setup warmup transformer options
            transformer_options = {
                "minimax_prefix_mode": "warmup",
                "patches_replace": {
                    "dit": create_prefix_dit_hook(
                        self.cache_manager,
                        is_anchor_warmup=is_anchor,
                        is_rolling_warmup=not is_anchor
                    )
                }
            }

            # Run forward pass in evaluation mode without tracking gradients
            with torch.no_grad():
                model = getattr(model_patcher, "model", model_patcher)
                diffusion_model = getattr(model, "diffusion_model", model)

                if hasattr(diffusion_model, "forward"):
                    diffusion_model.forward(
                        x=x_in,
                        timestep=sigma,
                        context=text_context if text_context is not None else [torch.zeros((1, 1, 5120), device=video_latent.device)],
                        transformer_options=transformer_options
                    )
                elif callable(model_patcher):
                    model_patcher(x_in, sigma, context=text_context, transformer_options=transformer_options)

            mem_info = self.cache_manager.get_memory_usage_mb()
            logger.info("Warmup complete. Total KV Cache Footprint: %.2f MB (GPU: %.2f MB, CPU: %.2f MB)",
                        mem_info["total_mb"], mem_info["gpu_mb"], mem_info["cpu_pinned_mb"])
            return True
        except Exception as exc:
            logger.error("Warmup forward failed: %s", exc, exc_info=True)
            return False
