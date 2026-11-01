"""Taper pixel noise to zero and transfer its latent change without round-trip bias."""

import math
import logging
import torch

from .seam_protector import _standardize_image_tensor

logger = logging.getLogger("minimax_prefix_stream")

def reference_noise_strengths(frame_count):
    frame_count = int(frame_count)
    if frame_count <= 1:
        return [0.0] * int(frame_count)
    return [0.2 * (1.0 + math.cos(math.pi * i / (frame_count - 1))) for i in range(frame_count)]


def noise_reference_frames(images):
    """Cosine taper from 0.4 to 0, leaving the final pixel frame untouched."""
    result = images.clone()
    active = max(0, images.shape[0] - 1)
    if active == 0:
        return result
    frames = result[:active]
    generator = torch.Generator(device="cpu").manual_seed(0)
    noise = torch.randn(frames.shape, generator=generator, device="cpu").to(frames)
    strengths = torch.tensor(reference_noise_strengths(images.shape[0])[:active], device=frames.device,
                             dtype=frames.dtype).view(-1, 1, 1, 1)
    frames[..., :3] = (frames[..., :3] + strengths * noise[..., :3]).clamp(0.0, 1.0)
    return result


def encode_noisy_prefix(vae, source_video, plan):
    """Apply E(noisy pixels) - E(clean pixels) to the original selected latent.

    The paired encode cancels the clean decode/encode baseline. It does not
    guarantee that the learned VAE's noise response improves generated images.
    """
    images = _standardize_image_tensor(vae.decode(source_video))
    start, end = plan["source_start_frame"], plan["source_end_frame"]
    if images.shape[0] < end:
        raise ValueError("Reference noise: VAE decoded fewer frames than the selected source window")
    pixels = images[start:end, ..., :3].clone()
    del images
    clean_encoded = vae.encode(pixels)
    noisy_encoded = vae.encode(noise_reference_frames(pixels))
    expected = (source_video.shape[0], source_video.shape[1], plan["context_steps"], *source_video.shape[3:])
    for encoded in (clean_encoded, noisy_encoded):
        if tuple(encoded.shape) != expected:
            raise ValueError(f"Reference noise requires the matching H3 video VAE: expected {expected}, got {tuple(encoded.shape)}")
    original = source_video[:, :, plan["start_t"]:plan["end_t"]]
    delta = noisy_encoded.float() - clean_encoded.to(noisy_encoded.device, dtype=torch.float32)
    logger.info("Reference noise latent delta RMS %.6f; source RMS %.6f.",
                float(delta.square().mean().sqrt()), float(original.float().square().mean().sqrt()))
    return original.to(delta) + delta
