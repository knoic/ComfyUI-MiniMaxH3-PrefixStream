"""DiTBlock Patch & Hook Engine for MiniMax H3 Prefix KV Caching.

Intercepts DiTBlock forward passes via ComfyUI transformer_options["patches_replace"]["dit"].
In Warmup mode (Phase 0): extracts and caches Key/Value tensors for prefix frames.
In Denoise mode (Phase 1): executes target-only QKV projection, attending to
cached [K_prefix, K_target] and [V_prefix, V_target].
"""

from typing import Dict, Any, Callable, Optional, Tuple
import logging
import torch
import torch.nn as nn

logger = logging.getLogger("minimax_prefix_stream")

try:
    from .cache_manager import PrefixKVCacheManager
    from .fused_attention import asymmetric_cached_attention
except (ImportError, ValueError):
    from cache_manager import PrefixKVCacheManager
    from fused_attention import asymmetric_cached_attention


def _apply_rope_and_norm(
    attn_module: nn.Module,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rope_freqs: Optional[torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Applies RMSNorm and partial split-half 3D-RoPE to Q, K, V."""
    s = q.shape[0]
    heads = attn_module.heads
    head_dim = attn_module.head_dim

    v = v.view(s, heads, head_dim)
    if rope_freqs is not None:
        q = q.view(1, s, heads, head_dim)
        k = k.view(1, s, heads, head_dim)
        try:
            import comfy.quant_ops
            import comfy.model_management
            qw = comfy.model_management.cast_to(attn_module.q_norm.weight, device=q.device)
            kw = comfy.model_management.cast_to(attn_module.k_norm.weight, device=k.device)
            rot = rope_freqs.shape[-3] * 2
            if getattr(comfy.model_management, "in_training", False):
                q, k = comfy.quant_ops.ck.rms_rope_split_half(
                    q, k, rope_freqs, qw, kw, epsilon=attn_module.q_norm.eps, rot_dim=rot)
            else:
                comfy.quant_ops.ck.rms_rope_split_half_(
                    q, k, rope_freqs, qw, kw, epsilon=attn_module.q_norm.eps, rot_dim=rot)
            q = q[0]
            k = k[0]
        except Exception:
            q = attn_module.q_norm(q.view(s, heads, head_dim))
            k = attn_module.k_norm(k.view(s, heads, head_dim))
    else:
        q = attn_module.q_norm(q.view(s, heads, head_dim))
        k = attn_module.k_norm(k.view(s, heads, head_dim))

    q = q.transpose(0, 1).unsqueeze(0)
    k = k.transpose(0, 1).unsqueeze(0)
    v = v.transpose(0, 1).unsqueeze(0)
    return q, k, v


def create_prefix_dit_hook(
    cache_manager: PrefixKVCacheManager,
    model: Optional[Any] = None,
    is_anchor_warmup: bool = False,
    is_rolling_warmup: bool = False
) -> Dict[Tuple[str, int], Callable]:
    """Generates the patches_replace dictionary for all 50 DiT blocks."""
    patches = {}
    num_layers = cache_manager.config.num_layers

    # Extract blocks directly from model if available
    blocks = None
    if model is not None:
        diff_model = getattr(model, "diffusion_model", model)
        blocks = getattr(diff_model, "blocks", None)

    for layer_idx in range(num_layers):
        attn_mod = None
        if blocks is not None and layer_idx < len(blocks):
            attn_mod = getattr(blocks[layer_idx], "attn", None)

        patches[("double_block", layer_idx)] = _make_block_patch(
            layer_idx=layer_idx,
            cache_manager=cache_manager,
            attn_module=attn_mod,
            is_anchor_warmup=is_anchor_warmup,
            is_rolling_warmup=is_rolling_warmup
        )
    return patches


def _make_block_patch(
    layer_idx: int,
    cache_manager: PrefixKVCacheManager,
    attn_module: Optional[nn.Module],
    is_anchor_warmup: bool,
    is_rolling_warmup: bool
) -> Callable:
    """Creates a closure for block i."""

    def _resolve_attn(extra_options: Dict[str, Any]) -> Optional[nn.Module]:
        if attn_module is not None:
            return attn_module
        if "attn_module" in extra_options and extra_options["attn_module"] is not None:
            return extra_options["attn_module"]
        # Introspect original_block closure
        bw = extra_options.get("original_block")
        if hasattr(bw, "__closure__") and bw.__closure__:
            for cell in bw.__closure__:
                val = cell.cell_contents
                if hasattr(val, "attn"):
                    return val.attn
        return None

    def block_hook_fn(args: Dict[str, Any], extra_options: Dict[str, Any]) -> Dict[str, Any]:
        block_wrap = extra_options["original_block"]
        transformer_options = args.get("transformer_options", {})
        mode = transformer_options.get("minimax_prefix_mode", "normal")
        attn_mod = _resolve_attn(extra_options)

        if attn_mod is None or not getattr(cache_manager.config, "is_cache_enabled", lambda: True)():
            return block_wrap(args)

        # -------------------------------------------------------------
        # Mode 1: Legacy Warmup Mode (Offline forward pass if requested)
        # -------------------------------------------------------------
        if mode == "warmup":
            def warmup_attention(h_in, rope_freqs=None, transformer_options={}):
                s = h_in.shape[0]
                q, k, v = attn_mod.qkv_proj(h_in).split(attn_mod.heads * attn_mod.head_dim, dim=-1)
                q, k, v = _apply_rope_and_norm(attn_mod, q, k, v, rope_freqs)

                if is_anchor_warmup:
                    cache_manager.set_anchor_kv(layer_idx, k, v)
                elif is_rolling_warmup:
                    cache_manager.set_rolling_kv(layer_idx, k, v)

                out = asymmetric_cached_attention(
                    q, k, v,
                    num_heads=attn_mod.heads,
                    transformer_options=transformer_options
                )
                return attn_mod.out_proj(out)

            args_with_attn = dict(args)
            args_with_attn["attention"] = warmup_attention
            return block_wrap(args_with_attn)

        # -------------------------------------------------------------
        # Mode 2: Denoising Mode (Step-1 Dynamic Capture & Step 2..N Reuse)
        # -------------------------------------------------------------
        if mode == "denoise":
            # Identify condition token bounds (cond & cond_audio).
            # Text tokens (0..text_len) MUST NOT be cached, as their timestep t_v = 1 - sigma varies every step.
            layout = args.get("layout")
            cond_start = None
            cond_end = None
            if layout is not None and hasattr(layout, "segments"):
                for a, b, kind in layout.segments:
                    if kind in ("cond", "cond_audio"):
                        if cond_start is None:
                            cond_start = a
                        cond_end = b

            # If no prefix condition tokens exist, pass through natively
            if cond_start is None or cond_end is None or cond_end <= cond_start:
                return block_wrap(args)

            # Update step counter when layer 0 is encountered
            if layer_idx == 0:
                cache_manager.step_counter += 1

            # Branch 2A: Step 1 Dynamic Capture (Cache is empty)
            if not cache_manager.has_cache(layer_idx):
                def capture_attention(h_in, rope_freqs=None, transformer_options={}):
                    try:
                        s = h_in.shape[0]
                        q, k, v = attn_mod.qkv_proj(h_in).split(attn_mod.heads * attn_mod.head_dim, dim=-1)
                        q, k, v = _apply_rope_and_norm(attn_mod, q, k, v, rope_freqs)

                        # Extract ONLY invariant condition Key and Value (cond_start..cond_end)
                        k_cond = k[:, :, cond_start:cond_end, :]
                        v_cond = v[:, :, cond_start:cond_end, :]

                        # Store in cache manager (quantizes & pins if configured)
                        cache_manager.set_rolling_kv(layer_idx, k_cond, v_cond)
                        if layer_idx == 0:
                            cache_manager.captured_tokens = cond_end - cond_start
                            logger.info(
                                "[Prefix KV Cache] Step 1: Captured %d invariant condition tokens into 50-layer DiT cache (%s, %s).",
                                cache_manager.captured_tokens,
                                cache_manager.config.cache_dtype.upper(),
                                cache_manager._resolved_device_mode
                            )

                        # Full standard attention on Step 1
                        out = asymmetric_cached_attention(
                            q=q,
                            k_cached=k,
                            v_cached=v,
                            num_heads=attn_mod.heads,
                            transformer_options=transformer_options
                        )
                        return attn_mod.out_proj(out)
                    except Exception as err:
                        logger.warning("[Prefix KV Cache] Step 1 fallback in layer %d: %s", layer_idx, err)
                        return attn_mod(h_in, rope_freqs=rope_freqs, transformer_options=transformer_options)

                args_with_attn = dict(args)
                args_with_attn["attention"] = capture_attention
                return block_wrap(args_with_attn)

            # Branch 2B: Steps 2..N Prefix KV Reuse
            next_layer = layer_idx + 1
            cache_manager.prefetch_next_layer(next_layer, args["img"].device)

            if layer_idx == 0:
                cache_manager.skipped_steps_count += 1
                if cache_manager.skipped_steps_count <= 2 or cache_manager.skipped_steps_count % 5 == 0:
                    logger.info(
                        "[Prefix KV Cache] Step %d: Reusing %d cached condition tokens.",
                        cache_manager.step_counter,
                        cond_end - cond_start
                    )

            def cached_target_attention(h_in, rope_freqs=None, transformer_options={}):
                try:
                    s = h_in.shape[0]
                    q, k_dyn, v_dyn = attn_mod.qkv_proj(h_in).split(attn_mod.heads * attn_mod.head_dim, dim=-1)
                    q, k_dyn, v_dyn = _apply_rope_and_norm(attn_mod, q, k_dyn, v_dyn, rope_freqs)

                    # Fetch cached invariant condition KV
                    k_cond, v_cond = cache_manager.get_combined_kv(
                        layer_idx=layer_idx,
                        target_device=q.device,
                        compute_dtype=q.dtype
                    )

                    # Concatenate dynamically updated text, cached condition, and dynamically updated target
                    k_total = torch.cat([k_dyn[:, :, :cond_start, :], k_cond, k_dyn[:, :, cond_end:, :]], dim=2)
                    v_total = torch.cat([v_dyn[:, :, :cond_start, :], v_cond, v_dyn[:, :, cond_end:, :]], dim=2)

                    out = asymmetric_cached_attention(
                        q=q,
                        k_cached=k_total,
                        v_cached=v_total,
                        num_heads=attn_mod.heads,
                        transformer_options=transformer_options
                    )
                    return attn_mod.out_proj(out)
                except Exception as err:
                    logger.warning("[Prefix KV Cache] Step %d fallback in layer %d: %s", cache_manager.step_counter, layer_idx, err)
                    return attn_mod(h_in, rope_freqs=rope_freqs, transformer_options=transformer_options)

            args_with_attn = dict(args)
            args_with_attn["attention"] = cached_target_attention
            return block_wrap(args_with_attn)

        # -------------------------------------------------------------
        # Mode 3: Normal / Passthrough
        # -------------------------------------------------------------
        return block_wrap(args)

    return block_hook_fn
