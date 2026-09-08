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
    from .block_hook import create_prefix_dit_hook, _apply_rope_and_norm
except (ImportError, ValueError):
    from cache_manager import PrefixKVCacheManager
    from block_hook import create_prefix_dit_hook, _apply_rope_and_norm

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
            model = getattr(model_patcher, "model", model_patcher)
            diffusion_model = getattr(model, "diffusion_model", model)

            target_device = video_latent.device
            target_dtype = video_latent.dtype if video_latent.dtype.is_floating_point else torch.float32

            if hasattr(diffusion_model, "parameters"):
                try:
                    p = next(diffusion_model.parameters())
                    target_device = p.device
                    if p.dtype.is_floating_point:
                        target_dtype = p.dtype
                except (StopIteration, Exception):
                    pass

            # 1. Format video latent: ensure [1, C, T, H, W]
            v = video_latent
            if v.ndim == 4:
                v = v.unsqueeze(0)
            v = v.to(device=target_device, dtype=target_dtype)

            # 2. Format audio latent: ensure [1, 32, 2, T_audio]
            a = audio_latent
            if a is None:
                t_audio = max(1, int(round(v.shape[2] * (5.0 / 3.0))))
                a = torch.zeros((1, 32, 2, t_audio), device=target_device, dtype=target_dtype)
            else:
                if a.ndim == 3:
                    a = a.unsqueeze(0)
                a = a.to(device=target_device, dtype=target_dtype)

            # 3. Format text context: ensure a single Tensor [1, L, text_dim]
            text_tensor = None
            if isinstance(text_context, torch.Tensor):
                text_tensor = text_context
            elif isinstance(text_context, (list, tuple)) and len(text_context) > 0 and isinstance(text_context[0], torch.Tensor):
                text_tensor = text_context[0]

            if text_tensor is None:
                text_dim = getattr(diffusion_model, "text_dim", 5120)
                text_tensor = torch.zeros((1, 1, text_dim), device=target_device, dtype=target_dtype)
            else:
                if text_tensor.ndim == 2:
                    text_tensor = text_tensor.unsqueeze(0)
                text_tensor = text_tensor.to(device=target_device, dtype=target_dtype)

            # 4. Setup warmup transformer options with diffusion_model bound
            transformer_options = {
                "minimax_prefix_mode": "warmup",
                "patches_replace": {
                    "dit": create_prefix_dit_hook(
                        cache_manager=self.cache_manager,
                        model=diffusion_model,
                        is_anchor_warmup=is_anchor,
                        is_rolling_warmup=not is_anchor
                    )
                }
            }

            # 5. Timestep: pass 1.0 (sigma near 0.001 for clean condition state)
            sigma_step = torch.tensor([1.0], device=target_device)
            x_in = [v, a]

            # 6. Primary execution: model forward pass
            with torch.no_grad():
                forward_succeeded = False
                if hasattr(diffusion_model, "forward"):
                    try:
                        diffusion_model.forward(
                            x=x_in,
                            timestep=sigma_step,
                            context=text_tensor,
                            transformer_options=transformer_options
                        )
                        forward_succeeded = True
                    except Exception as fwd_err:
                        logger.warning("Primary diffusion_model.forward failed in warmup (%s), falling back to block execution.", fwd_err)

                if not forward_succeeded and callable(model_patcher):
                    try:
                        model_patcher(x_in, sigma_step, context=text_tensor, transformer_options=transformer_options)
                        forward_succeeded = True
                    except Exception as patch_err:
                        logger.warning("model_patcher call failed in warmup (%s), attempting direct block extraction.", patch_err)

                # 7. Resilient Fallback: direct block-by-block KV extraction
                if not self.cache_manager.has_cache(0) and hasattr(diffusion_model, "blocks"):
                    logger.info("Executing direct block-level Phase 0 warmup fallback...")
                    patch_size = getattr(diffusion_model, "patch_size", (1, 2, 2))
                    b, c, t, h, w = v.shape
                    pt, ph, pw = patch_size
                    pad_h = (ph - (h % ph)) % ph
                    pad_w = (pw - (w % pw)) % pw
                    v_pad = torch.nn.functional.pad(v, (0, pad_w, 0, pad_h))
                    _, _, _, h_p, w_p = v_pad.shape
                    v_reshaped = v_pad.reshape(b, c, t, pt, h_p // ph, ph, w_p // pw, pw)
                    v_rows = torch.einsum("nctrhpwq->nthwcrpq", v_reshaped).reshape(-1, c * pt * ph * pw)

                    a_b, a_c, a_ch, a_t = a.shape
                    a_rows = a[0].permute(1, 2, 0).reshape(a_ch * a_t, a_c)

                    if hasattr(diffusion_model, "video_patch_proj") and hasattr(diffusion_model, "audio_patch_proj"):
                        v_emb = diffusion_model.video_patch_proj(v_rows.float()).to(target_dtype)
                        a_emb = diffusion_model.audio_patch_proj(a_rows.float()).to(target_dtype)
                        h_stream = torch.cat([a_emb, v_emb], dim=0)

                        for idx, block in enumerate(diffusion_model.blocks):
                            attn = getattr(block, "attn", None)
                            if attn is not None:
                                q, k, val = attn.qkv_proj(h_stream).split(attn.heads * attn.head_dim, dim=-1)
                                q, k, val = _apply_rope_and_norm(attn, q, k, val, rope_freqs=None)
                                if is_anchor:
                                    self.cache_manager.set_anchor_kv(idx, k, val)
                                else:
                                    self.cache_manager.set_rolling_kv(idx, k, val)

            mem_info = self.cache_manager.get_memory_usage_mb()
            if self.cache_manager.has_cache(0):
                logger.info(
                    "Phase 0 Warmup complete. Cached 50 DiT layers: %.2f MB (GPU: %.2f MB, CPU: %.2f MB)",
                    mem_info["total_mb"], mem_info["gpu_mb"], mem_info["cpu_pinned_mb"]
                )
                return True
            else:
                logger.error("Phase 0 Warmup completed but cache is still empty.")
                return False
        except Exception as exc:
            logger.error("Warmup forward failed: %s", exc, exc_info=True)
            return False
