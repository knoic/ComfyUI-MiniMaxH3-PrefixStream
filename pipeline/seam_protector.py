"""Seam Protector & Audio-Video Seamless Handover for Long Video Chaining.

Provides:
1. Audio waveform micro-crossfade (50ms equal-power cosine curve) to eliminate click/pop artifacts.
2. Latent-space soft overlap blending across chunk boundaries.
3. Clean trim of overlapping prefix frames to produce seamless concatenated final media.
"""

from typing import Optional, Tuple, Dict, Any
import math
import torch
import torch.nn.functional as F


def audio_equal_power_crossfade(
    wave1: torch.Tensor,
    wave2: torch.Tensor,
    crossfade_samples: int = 1600  # 50ms at 32kHz
) -> torch.Tensor:
    """Concatenates two audio waveforms with an equal-power cosine crossfade."""
    if crossfade_samples <= 0:
        return torch.cat([wave1, wave2], dim=-1)

    c = min(crossfade_samples, wave1.shape[-1], wave2.shape[-1])
    if c <= 0:
        return torch.cat([wave1, wave2], dim=-1)

    # Fade curves
    t = torch.linspace(0, math.pi / 2, c, device=wave1.device, dtype=wave1.dtype)
    fade_out = torch.cos(t)
    fade_in = torch.sin(t)

    tail1 = wave1[..., -c:] * fade_out
    head2 = wave2[..., :c] * fade_in
    blended = tail1 + head2

    return torch.cat([wave1[..., :-c], blended, wave2[..., c:]], dim=-1)


def latent_soft_blend(
    latent1: torch.Tensor,
    latent2: torch.Tensor,
    blend_steps: int = 2
) -> torch.Tensor:
    """Softly blends the tail of latent1 with the head of latent2 along the time dimension.

    latent shape: [B, C, T, H, W]
    """
    b = min(blend_steps, latent1.shape[2], latent2.shape[2])
    if b <= 0:
        return torch.cat([latent1, latent2], dim=2)

    # Linear or cosine alpha ramp
    alpha = torch.linspace(0.0, 1.0, b, device=latent1.device, dtype=latent1.dtype)
    # Reshape for broadcasting [1, 1, b, 1, 1]
    alpha = alpha.view(1, 1, -1, 1, 1)

    overlap1 = latent1[:, :, -b:]
    overlap2 = latent2[:, :, :b]
    blended = (1.0 - alpha) * overlap1 + alpha * overlap2

    return torch.cat([latent1[:, :, :-b], blended, latent2[:, :, b:]], dim=2)


def trim_prefix_frames(
    full_video_latent: torch.Tensor,
    prefix_latent_steps: int
) -> torch.Tensor:
    """Trims the leading prefix/condition frames from the generated output."""
    if prefix_latent_steps <= 0:
        return full_video_latent
    if prefix_latent_steps >= full_video_latent.shape[2]:
        return full_video_latent
    return full_video_latent[:, :, prefix_latent_steps:]
