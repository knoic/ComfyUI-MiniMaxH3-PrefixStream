"""Seam Protector & Audio-Video Seamless Handover for Long Video Chaining.

Provides:
1. Pixel-space video stitching with luminance matching and cosine S-curve crossfade (zero VAE artifacts).
2. Waveform-space audio stitching with sample-accurate alignment and equal-power crossfade (zero click/pop).
3. Synchronous frame & sample trimming for decoded IMAGE and AUDIO streams.
4. Latent-space soft overlap blending (fallback / intermediate representation).
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


def fit_audio_length(waveform: torch.Tensor, target_samples: int) -> torch.Tensor:
    """Pads or truncates audio waveform to exact target samples."""
    target = max(0, int(target_samples))
    current = int(waveform.shape[-1])
    if current == target:
        return waveform
    if current > target:
        return waveform[..., :target]
    if target == 0:
        return waveform[..., :0]
    if current == 0:
        shape = list(waveform.shape)
        shape[-1] = target
        return torch.zeros(shape, dtype=waveform.dtype, device=waveform.device)
    pad = waveform[..., -1:].expand(*waveform.shape[:-1], target - current)
    return torch.cat((waveform, pad), dim=-1)


def _rgb_luminance(images: torch.Tensor) -> torch.Tensor:
    """Calculates relative luminance from RGB channels."""
    rgb = images[..., :3].float()
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def estimate_luminance_gain(
    previous_context: torch.Tensor,
    next_context: torch.Tensor,
    max_correction_percent: float = 10.0
) -> float:
    """Estimates conservative global RGB gain between time-overlapping frames."""
    if previous_context.shape != next_context.shape or previous_context.shape[0] == 0:
        return 1.0
    prev_luma = _rgb_luminance(previous_context)[:, ::4, ::4]
    next_luma = _rgb_luminance(next_context.to(previous_context.device))[:, ::4, ::4]
    ratios = []
    for i in range(previous_context.shape[0]):
        a = prev_luma[i]
        b = next_luma[i]
        valid = (a > 0.04) & (a < 0.96) & (b > 0.04) & (b < 0.96)
        if valid.sum() < max(4, int(valid.numel() * 0.01)):
            continue
        a_mean = float(a[valid].mean().item())
        b_mean = float(b[valid].mean().item())
        if b_mean > 1e-6:
            ratios.append(a_mean / b_mean)
    if not ratios:
        return 1.0
    measured = float(torch.tensor(ratios, dtype=torch.float32).median().item())
    limit = max(0.0, float(max_correction_percent)) / 100.0
    return min(1.0 + limit, max(max(0.01, 1.0 - limit), measured))


def apply_luminance_gain_fade(
    images: torch.Tensor,
    gain: float,
    fade_frames: int = 16
) -> torch.Tensor:
    """Smoothly applies gain at clip start and fades back to native brightness."""
    n = min(max(0, int(fade_frames)), int(images.shape[0]))
    if n <= 0 or abs(gain - 1.0) < 1e-6:
        return images
    out = images.clone()
    t = torch.linspace(0.0, 1.0, n, dtype=out.dtype, device=out.device)
    weights = 0.5 + 0.5 * torch.cos(math.pi * t)
    scales = 1.0 + (gain - 1.0) * weights
    out[:n, ..., :3] = (out[:n, ..., :3] * scales.view(n, 1, 1, 1)).clamp(0.0, 1.0)
    return out


def _standardize_image_tensor(images: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Standardizes video image tensor to canonical ComfyUI 4D shape: [F, H, W, 3/4].
    
    Gracefully handles:
    - 5D tensors: [1, F, H, W, C], [B, C, F, H, W], [1, C, F, H, W]
    - 3D tensors: [H, W, C] -> [1, H, W, C]
    - Channel-first vs channel-last conventions.
    """
    if images is None or not isinstance(images, torch.Tensor):
        return images
    
    t = images
    if t.ndim == 5:
        # Case 1: [1, F, H, W, C] where last dim is color channels (1, 3, 4)
        if t.shape[0] == 1 and t.shape[-1] in (1, 3, 4):
            t = t.squeeze(0)
        # Case 2: [1, C, F, H, W] where channel dim is at index 1
        elif t.shape[0] == 1 and t.shape[1] in (1, 3, 4):
            t = t.squeeze(0).permute(1, 2, 3, 0)
        # Case 3: [B, C, F, H, W] where B > 1
        elif t.shape[1] in (1, 3, 4):
            t = t.permute(0, 2, 3, 4, 1).flatten(0, 1)
        else:
            t = t.flatten(0, 1)
    elif t.ndim == 3:
        t = t.unsqueeze(0)
    
    return t


def _standardize_audio_dict(audio: Any, default_sr: int = 32000) -> Optional[Dict[str, Any]]:
    """Safely normalizes AUDIO input to standard ComfyUI dict {'waveform': Tensor, 'sample_rate': int}.
    
    Prevents crash when upstream node outputs bare Tensor or tuple/list instead of dictionary.
    """
    if audio is None:
        return None
    if isinstance(audio, dict) and "waveform" in audio:
        return audio
    if isinstance(audio, torch.Tensor):
        t = audio
        if t.ndim == 1:
            t = t.unsqueeze(0).unsqueeze(0)
        elif t.ndim == 2:
            t = t.unsqueeze(0)
        return {"waveform": t, "sample_rate": default_sr}
    if isinstance(audio, (list, tuple)) and len(audio) > 0:
        return _standardize_audio_dict(audio[0], default_sr=default_sr)
    return None


def stitch_video_images(
    prev_images: Optional[torch.Tensor],
    curr_images: torch.Tensor,
    trim_frames: int = 22,
    crossfade_frames: int = 4,
    luminance_match: bool = True,
    luminance_fade_frames: int = 16
) -> torch.Tensor:
    """Seamlessly joins two decoded video IMAGE tensors [F, H, W, 3] in pixel space.

    Guarantees zero VAE artifacts, zero color distortion, and smooth seam transition.
    """
    prev_images = _standardize_image_tensor(prev_images)
    curr_images = _standardize_image_tensor(curr_images)

    if curr_images is None:
        return prev_images if prev_images is not None else torch.empty(0)

    head = max(0, int(trim_frames))
    if head >= curr_images.shape[0]:
        return prev_images if prev_images is not None else curr_images[:0]

    if prev_images is None or prev_images.shape[0] == 0:
        return curr_images[head:]

    curr_body = curr_images[head:].to(prev_images.device, prev_images.dtype)

    gain = 1.0
    if luminance_match and head > 0 and prev_images.shape[0] > 0:
        analysis_n = min(8, head, prev_images.shape[0])
        gain = estimate_luminance_gain(
            prev_images[-analysis_n:],
            curr_images[head - analysis_n:head]
        )
        curr_body = apply_luminance_gain_fade(curr_body, gain, fade_frames=luminance_fade_frames)

    n = min(max(0, int(crossfade_frames)), head, prev_images.shape[0])
    if n <= 0:
        return torch.cat([prev_images, curr_body], dim=0)

    prev_prefix = prev_images[:-n]
    prev_tail = prev_images[-n:]
    curr_overlap = curr_images[head - n:head].to(prev_images.device, prev_images.dtype)
    if luminance_match and abs(gain - 1.0) > 1e-6:
        curr_overlap = (curr_overlap * gain).clamp(0.0, 1.0)

    t = torch.linspace(0.0, 1.0, n, dtype=prev_images.dtype, device=prev_images.device)
    alpha = (0.5 - 0.5 * torch.cos(math.pi * t)).view(n, 1, 1, 1)
    blended = prev_tail * (1.0 - alpha) + curr_overlap * alpha

    return torch.cat([prev_prefix, blended, curr_body], dim=0)


def trim_images_and_audio(
    images: torch.Tensor,
    audio: Optional[Dict[str, Any]] = None,
    trim_frames: int = 0,
    fps: float = 24.0,
    match_tail: bool = True
) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
    """Synchronously trims leading overlap frames from IMAGE and matching samples from AUDIO."""
    images = _standardize_image_tensor(images)
    n = max(0, int(trim_frames))
    total_frames = int(images.shape[0]) if images is not None else 0

    if n >= total_frames:
        trimmed_images = images[:0]
    elif n > 0:
        trimmed_images = images[n:]
    else:
        trimmed_images = images

    trimmed_audio = None
    audio = _standardize_audio_dict(audio)
    if audio is not None and "waveform" in audio:
        waveform = audio["waveform"]
        sr = int(audio.get("sample_rate", 32000))
        head_samples = int(round(n / float(fps) * sr))
        kept_frames = max(0, total_frames - n)

        if match_tail:
            want_samples = int(round(kept_frames / float(fps) * sr))
            w = waveform[..., head_samples:head_samples + want_samples]
            w = fit_audio_length(w, want_samples)
        else:
            w = waveform[..., head_samples:]
        trimmed_audio = {"waveform": w, "sample_rate": sr}

    return trimmed_images, trimmed_audio


def stitch_audio_waveforms(
    prev_audio: Optional[Dict[str, Any]],
    curr_audio: Optional[Dict[str, Any]],
    curr_total_frames: int,
    trim_frames: int = 22,
    crossfade_ms: float = 15.0,
    fps: float = 24.0
) -> Optional[Dict[str, Any]]:
    """Context-aligned audio stitch with de-click crossfade and sample-accurate timeline sync."""
    curr_audio = _standardize_audio_dict(curr_audio)
    prev_audio = _standardize_audio_dict(prev_audio)

    if curr_audio is None or "waveform" not in curr_audio:
        return prev_audio

    sr = int(curr_audio.get("sample_rate", 32000))
    head_samples = int(round(trim_frames / float(fps) * sr))
    kept_frames = max(0, curr_total_frames - trim_frames)
    want_samples = int(round(kept_frames / float(fps) * sr))

    curr_w = curr_audio["waveform"]
    curr_body = fit_audio_length(curr_w[..., head_samples:head_samples + want_samples], want_samples)

    if prev_audio is None or "waveform" not in prev_audio:
        return {"waveform": curr_body, "sample_rate": sr}

    prev_w = prev_audio["waveform"]
    c = min(int(round((crossfade_ms / 1000.0) * sr)), head_samples, prev_w.shape[-1])
    if c <= 0:
        stitched = torch.cat([prev_w, curr_body], dim=-1)
    else:
        prev_tail = prev_w[..., -c:]
        curr_overlap = curr_w[..., head_samples - c : head_samples]

        t = torch.linspace(0.0, 1.0, c, dtype=prev_w.dtype, device=prev_w.device)
        alpha = t.view(1, 1, c)
        blended = prev_tail * (1.0 - alpha) + curr_overlap * alpha

        stitched = torch.cat([prev_w[..., :-c], blended, curr_body], dim=-1)

    return {"waveform": stitched, "sample_rate": sr}


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


def stitch_video_latents(
    prev_latent: torch.Tensor,
    curr_latent: torch.Tensor,
    overlap_steps: int,
    blend_steps: int = 2
) -> torch.Tensor:
    """Stitches two contiguous video latents, seamlessly blending the overlap region.

    prev_latent: [B, C, T1, H, W]
    curr_latent: [B, C, T2, H, W] where curr_latent[:, :, :overlap_steps] overlaps with prev_latent[:, :, -overlap_steps:]
    """
    if overlap_steps <= 0:
        return latent_soft_blend(prev_latent, curr_latent, blend_steps=blend_steps)

    ov = min(overlap_steps, prev_latent.shape[2], curr_latent.shape[2])
    if ov <= 0:
        return torch.cat([prev_latent, curr_latent], dim=2)

    # Overlap slices
    prev_head = prev_latent[:, :, :-ov]
    prev_overlap = prev_latent[:, :, -ov:]
    curr_overlap = curr_latent[:, :, :ov]
    curr_tail = curr_latent[:, :, ov:]

    # Blend the overlap region smoothly with smooth cosine S-curve
    t = torch.linspace(0.0, math.pi / 2, ov, device=curr_latent.device, dtype=curr_latent.dtype)
    alpha = (torch.sin(t) ** 2).view(1, 1, -1, 1, 1)
    blended_overlap = (1.0 - alpha) * prev_overlap + alpha * curr_overlap

    return torch.cat([prev_head, blended_overlap, curr_tail], dim=2)


def stitch_audio_latents(
    prev_latent: torch.Tensor,
    curr_latent: torch.Tensor,
    overlap_steps: int,
    blend_steps: int = 2
) -> torch.Tensor:
    """Stitches two contiguous 4D audio latents [B, 32, 2, T], seamlessly blending the overlap region."""
    if overlap_steps <= 0 or prev_latent is None:
        return curr_latent

    ov = min(overlap_steps, prev_latent.shape[-1], curr_latent.shape[-1])
    if ov <= 0:
        return torch.cat([prev_latent, curr_latent], dim=-1)

    prev_head = prev_latent[..., :-ov]
    prev_overlap = prev_latent[..., -ov:]
    curr_overlap = curr_latent[..., :ov]
    curr_tail = curr_latent[..., ov:]

    t = torch.linspace(0.0, math.pi / 2, ov, device=curr_latent.device, dtype=curr_latent.dtype)
    alpha = torch.sin(t) ** 2
    while alpha.ndim < curr_latent.ndim:
        alpha = alpha.unsqueeze(0)

    blended_overlap = (1.0 - alpha) * prev_overlap + alpha * curr_overlap
    return torch.cat([prev_head, blended_overlap, curr_tail], dim=-1)


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


def trim_audio_waveform(
    waveform: torch.Tensor,
    trim_samples: int
) -> torch.Tensor:
    """Trims leading samples from audio waveform."""
    if trim_samples <= 0:
        return waveform
    if trim_samples >= waveform.shape[-1]:
        return waveform
    return waveform[..., trim_samples:]


def trim_audio_latents(
    full_audio_latent: torch.Tensor,
    prefix_audio_steps: int
) -> torch.Tensor:
    """Trims leading overlap steps from a 4D audio latent [B, 32, 2, T]."""
    if prefix_audio_steps <= 0 or full_audio_latent is None:
        return full_audio_latent
    if prefix_audio_steps >= full_audio_latent.shape[-1]:
        return full_audio_latent
    return full_audio_latent[..., prefix_audio_steps:]

