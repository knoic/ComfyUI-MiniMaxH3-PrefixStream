"""DiTBlock Patch & Hook Engine for MiniMax H3 Prefix KV Caching.

Intercepts DiTBlock forward passes via ComfyUI transformer_options["patches_replace"]["dit"].
In Warmup mode (Phase 0): extracts and caches Key/Value tensors for prefix frames.
In Denoise mode (Phase 1): executes target-only QKV projection and MLP, attending to
cached [K_prefix, K_target] and [V_prefix, V_target].
"""

from typing import Dict, Any, Callable, Optional, Tuple
import torch
import torch.nn as nn

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
        # Check comfy quant_ops or fallback
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
            # Fallback standard RMSNorm if comfy quant_ops is unavailable
            q = attn_module.q_norm(q.view(s, heads, head_dim))
            k = attn_module.k_norm(k.view(s, heads, head_dim))
    else:
        q = attn_module.q_norm(q.view(s, heads, head_dim))
        k = attn_module.k_norm(k.view(s, heads, head_dim))

    # Standardize to [1, heads, S, head_dim]
    q = q.transpose(0, 1).unsqueeze(0)
    k = k.transpose(0, 1).unsqueeze(0)
    v = v.transpose(0, 1).unsqueeze(0)
    return q, k, v


def create_prefix_dit_hook(
    cache_manager: PrefixKVCacheManager,
    is_anchor_warmup: bool = False,
    is_rolling_warmup: bool = False
) -> Dict[Tuple[str, int], Callable]:
    """Generates the patches_replace dictionary for all 50 DiT blocks.

    Returns:
        dict with keys ("double_block", layer_idx) matching ComfyUI patcher conventions.
    """
    patches = {}
    num_layers = cache_manager.config.num_layers

    for layer_idx in range(num_layers):
        patches[("double_block", layer_idx)] = _make_block_patch(
            layer_idx=layer_idx,
            cache_manager=cache_manager,
            is_anchor_warmup=is_anchor_warmup,
            is_rolling_warmup=is_rolling_warmup
        )
    return patches


def _make_block_patch(
    layer_idx: int,
    cache_manager: PrefixKVCacheManager,
    is_anchor_warmup: bool,
    is_rolling_warmup: bool
) -> Callable:
    """Creates a closure for block i."""

    def block_hook_fn(args: Dict[str, Any], extra_options: Dict[str, Any]) -> Dict[str, Any]:
        block_wrap = extra_options["original_block"]
        transformer_options = args.get("transformer_options", {})
        mode = transformer_options.get("minimax_prefix_mode", "normal")

        # -------------------------------------------------------------
        # Mode 1: Warmup Mode (Phase 0: Capture & Cache Key/Value)
        # -------------------------------------------------------------
        if mode == "warmup":
            # Custom attention handler to record K, V during forward pass
            def warmup_attention(h_in, rope_freqs=None, transformer_options={}):
                # Retrieve block reference from block_wrap closure
                # Standard QKV projection
                attn_mod = extra_options.get("attn_module")
                if attn_mod is None:
                    # Fallback: retrieve from block if accessible
                    return h_in

                s = h_in.shape[0]
                q, k, v = attn_mod.qkv_proj(h_in).split(attn_mod.heads * attn_mod.head_dim, dim=-1)
                q, k, v = _apply_rope_and_norm(attn_mod, q, k, v, rope_freqs)

                # Store into cache manager
                if is_anchor_warmup:
                    cache_manager.set_anchor_kv(layer_idx, k, v)
                elif is_rolling_warmup:
                    cache_manager.set_rolling_kv(layer_idx, k, v)

                # Continue forward to next block
                out = asymmetric_cached_attention(
                    q, k, v,
                    num_heads=attn_mod.heads,
                    transformer_options=transformer_options
                )
                return attn_mod.out_proj(out)

            # Run original block with warmup_attention injected if supported
            return block_wrap(args)

        # -------------------------------------------------------------
        # Mode 2: Denoising Mode (Phase 1: Reuse Prefix KV Cache)
        # -------------------------------------------------------------
        if mode == "denoise" and cache_manager.has_cache(layer_idx):
            # Prefetch next layer to overlap CPU->GPU transfer with current layer compute
            next_layer = layer_idx + 1
            cache_manager.prefetch_next_layer(next_layer, args["img"].device)

            def cached_attention(h_in, rope_freqs=None, transformer_options={}):
                attn_mod = extra_options.get("attn_module")
                if attn_mod is None:
                    return h_in

                # Target-only QKV projection
                s = h_in.shape[0]
                q_t, k_t, v_t = attn_mod.qkv_proj(h_in).split(attn_mod.heads * attn_mod.head_dim, dim=-1)
                q_t, k_t, v_t = _apply_rope_and_norm(attn_mod, q_t, k_t, v_t, rope_freqs)

                # Fetch cached prefix KV
                k_prefix, v_prefix = cache_manager.get_combined_kv(
                    layer_idx=layer_idx,
                    target_device=q_t.device,
                    compute_dtype=q_t.dtype
                )

                # Concatenate [K_prefix, K_target] along sequence dimension
                # Both are [1, heads, S, head_dim]
                k_total = torch.cat([k_prefix, k_t], dim=2)
                v_total = torch.cat([v_prefix, v_t], dim=2)

                # Compute asymmetric cross-attention
                out = asymmetric_cached_attention(
                    q=q_t,
                    k_cached=k_total,
                    v_cached=v_total,
                    num_heads=attn_mod.heads,
                    transformer_options=transformer_options
                )
                return attn_mod.out_proj(out)

            # Delegate to block execution
            args_with_attn = dict(args)
            args_with_attn["attention"] = cached_attention
            return block_wrap(args_with_attn)

        # -------------------------------------------------------------
        # Mode 3: Normal / Passthrough
        # -------------------------------------------------------------
        return block_wrap(args)

    return block_hook_fn
