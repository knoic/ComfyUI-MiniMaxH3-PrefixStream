"""Asymmetric Fused Cross/Self Attention for Prefix KV Caching (Experimental).

NOTE: Reference implementation for asymmetric attention in Prefix KV caching exploration.
Current production workflows use Native Masked AV without DiT monkey-patching.
"""

from typing import Optional, Dict, Any
import torch
import torch.nn.functional as F

try:
    from comfy.ldm.modules.attention import AttentionTensorContainer, optimized_attention
    _HAS_COMFY_ATTN = True
except ImportError:
    AttentionTensorContainer = None
    optimized_attention = None
    _HAS_COMFY_ATTN = False


def asymmetric_cached_attention(
    q: torch.Tensor,
    k_cached: torch.Tensor,
    v_cached: torch.Tensor,
    num_heads: int,
    mask: Optional[torch.Tensor] = None,
    transformer_options: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """Computes attention where Query is target-only, but Key and Value contain [Prefix + Target].

    Args:
        q: [1, heads, S_target, head_dim] or [S_target, heads, head_dim]
        k_cached: [1, heads, S_total, head_dim]
        v_cached: [1, heads, S_total, head_dim]
        num_heads: number of attention heads (56 in MiniMax H3)
        mask: optional attention mask
        transformer_options: optional dict for ComfyUI patcher

    Returns:
        out: [S_target, heads * head_dim] (flattened for out_proj)
    """
    # Standardize Query shape to [1, heads, S_target, head_dim]
    if q.ndim == 3:
        # [S_target, heads, head_dim] -> [1, heads, S_target, head_dim]
        q = q.transpose(0, 1).unsqueeze(0)

    # Ensure contiguous for FlashAttention kernels
    q = q.contiguous()
    k_cached = k_cached.contiguous()
    v_cached = v_cached.contiguous()

    # If ComfyUI optimized_attention is available, try it with container
    if _HAS_COMFY_ATTN and AttentionTensorContainer is not None:
        try:
            qc = AttentionTensorContainer(q)
            kc = AttentionTensorContainer(k_cached)
            vc = AttentionTensorContainer(v_cached)
            out = optimized_attention(
                qc, kc, vc,
                num_heads,
                mask=mask,
                skip_reshape=True,
                transformer_options=transformer_options or {}
            )
            if isinstance(out, AttentionTensorContainer):
                out = out.tensor
            if hasattr(out, "unwrap"):
                out = out.unwrap()

            # ComfyUI optimized_attention with skip_reshape=True returns:
            # - 3D [1, S_target, heads * head_dim] (standard ComfyUI path)
            # - or 4D [1, heads, S_target, head_dim]
            s_q = q.shape[2]
            if out.ndim == 3:
                return out.squeeze(0).contiguous()
            elif out.ndim == 4:
                return out.squeeze(0).transpose(0, 1).contiguous().reshape(s_q, -1)
            elif out.ndim == 2:
                return out.contiguous()
            return out.reshape(s_q, -1).contiguous()
        except Exception:
            # Fall back to PyTorch native SDPA if Comfy container rejects non-square shapes
            pass

    # Direct PyTorch 2.0+ F.scaled_dot_product_attention (supports S_q != S_k natively)
    # q: [1, heads, S_q, dim], k: [1, heads, S_k, dim], v: [1, heads, S_k, dim]
    out = F.scaled_dot_product_attention(
        q, k_cached, v_cached,
        attn_mask=mask,
        dropout_p=0.0,
        is_causal=False
    )

    # SDPA returns [1, heads, S_q, dim] -> [S_q, heads * dim]
    s_q = q.shape[2]
    if out.ndim == 4:
        return out.squeeze(0).transpose(0, 1).contiguous().reshape(s_q, -1)
    elif out.ndim == 3:
        return out.squeeze(0).contiguous()
    return out.reshape(s_q, -1).contiguous()
