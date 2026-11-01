"""Native MiniMax H3 video/audio masked-continuation helpers.

Copied video context follows H3's 17k+5 frame grid. Audio context is rounded
to the 40 Hz tick grid, as in the native LatentCut/LatentConcat workflow.
"""

from typing import Any, Dict, Tuple

import torch


FPS = 24
AUDIO_HZ = 40
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
MASKED_AV_CONTEXTS = tuple(range(5, 193, 17))


def latent_boundaries(video_latent_t: int, start_step: int = 0):
    """Return exclusive pixel-frame boundaries for an H3 video latent."""
    boundaries = [0]
    for step in range(int(video_latent_t)):
        boundaries.append(boundaries[-1] + FRAME_PER_TOKEN[(start_step + step) % 5])
    return boundaries


def is_exact_masked_av_context(frame_count: int) -> bool:
    """Whether a canonical video run also has an integer audio duration (no rounding)."""
    frame_count = int(frame_count)
    return frame_count >= 39 and (frame_count - 39) % 51 == 0


def snap_masked_av_context_length(requested: int, available: int, target_frames: int) -> int:
    """Snap down to a canonical video context that leaves new content."""
    cap = min(int(requested), int(available), int(target_frames) - 1)
    if cap < 5:
        raise ValueError(
            "Native Masked AV needs at least 5 source frames and a target longer than 5 frames"
        )
    run = 5 + ((cap - 5) // 17) * 17
    return run


def masked_av_tail_slice(video_latent_t: int, context_frames: int, target_video_t: int, source_start_step: int = 0) -> Dict[str, int]:
    """Select the latest canonical source tail for a target masked prefix."""
    source_boundaries = latent_boundaries(video_latent_t, source_start_step)
    target_frames = latent_boundaries(target_video_t)[-1]
    previous_frames = source_boundaries[-1]
    frames = snap_masked_av_context_length(context_frames, previous_frames, target_frames)
    context_steps = 2 + 5 * ((frames - 5) // 17)

    end_t = None
    for candidate in range(int(video_latent_t), context_steps - 1, -1):
        start_t = candidate - context_steps
        if (source_start_step + candidate) % 5 == 2 and (source_start_step + start_t) % 5 == 0:
            end_t = candidate
            break
    if end_t is None:
        raise ValueError("Source latent has no phase-aligned Native Masked AV tail")

    start_t = end_t - context_steps
    source_start_frame = source_boundaries[start_t]
    source_end_frame = source_boundaries[end_t]
    if source_end_frame - source_start_frame != frames:
        raise RuntimeError("Internal Native Masked AV source geometry mismatch")
    return {
        "start_t": start_t,
        "end_t": end_t,
        "context_steps": context_steps,
        "actual_context_frames": frames,
        "source_start_frame": source_start_frame,
        "source_end_frame": source_end_frame,
        "previous_frame_count": previous_frames,
        "ignored_tail_frames": previous_frames - source_end_frame,
    }


def apply_native_masked_av(
    target_video: torch.Tensor,
    target_audio: torch.Tensor,
    source_video: torch.Tensor,
    source_audio: torch.Tensor,
    context_frames: int = 39,
    audio_tail_carryover: str = "Match Video Handover",
    audio_feather_ticks: int = 0,
    source_start_step: int = 0,
    source_audio_start_tick: int = 0,
    video_context_denoise: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Copy a previous AV tail into a target head and build native denoise masks."""
    if target_video.ndim != 5 or source_video.ndim != 5:
        raise ValueError("Native Masked AV expects 5D video latents")
    if target_audio.ndim != 4 or source_audio.ndim != 4:
        raise ValueError("Native Masked AV expects 4D audio latents")
    if target_video.shape[0] != 1 or source_video.shape[0] != 1:
        raise ValueError("Native Masked AV currently supports batch size 1")
    if target_audio.shape[0] != 1 or source_audio.shape[0] != 1:
        raise ValueError("Native Masked AV currently supports audio batch size 1")
    if tuple(source_video.shape[1:2] + source_video.shape[3:]) != tuple(
        target_video.shape[1:2] + target_video.shape[3:]
    ):
        raise ValueError("Native Masked AV requires matching source/target video latent geometry")
    if tuple(source_audio.shape[1:3]) != tuple(target_audio.shape[1:3]):
        raise ValueError("Native Masked AV requires matching source/target audio latent geometry")

    plan = masked_av_tail_slice(source_video.shape[2], context_frames, target_video.shape[2], source_start_step)
    video_steps = plan["context_steps"]
    source_run = source_video[:, :, plan["start_t"]:plan["end_t"]]

    mode = str(audio_tail_carryover).strip().lower()
    if mode not in ("full previous tail", "match video handover"):
        raise ValueError(f"Unknown audio tail carryover mode: {audio_tail_carryover!r}")
    audio_end_frame = (
        plan["previous_frame_count"] if mode == "full previous tail" else plan["source_end_frame"]
    )
    # Round the duration once, then end-align. Rounding both absolute endpoints
    # independently can copy an extra tick (e.g. a 5-frame tail of a 124-frame clip).
    audio_steps = round((audio_end_frame - plan["source_start_frame"]) * AUDIO_HZ / FPS)
    origin_frame = latent_boundaries(source_start_step)[-1]
    a1 = min(source_audio.shape[-1], round((origin_frame + audio_end_frame) * AUDIO_HZ / FPS) - source_audio_start_tick)
    a0 = max(0, a1 - audio_steps)
    if a1 <= a0:
        raise ValueError("Native Masked AV audio context is empty")
    audio_steps = a1 - a0
    if audio_steps >= target_audio.shape[-1]:
        raise ValueError(
            "Protected audio would consume the whole target; increase target duration or use Match Video Handover"
        )

    out_video = target_video.clone()
    out_audio = target_audio.clone()
    out_video[:, :, :video_steps] = source_run.to(out_video)
    out_audio[..., :audio_steps] = source_audio[..., a0:a1].to(out_audio)

    video_mask = torch.ones(
        (1, 1, out_video.shape[2], out_video.shape[3], out_video.shape[4]),
        device=out_video.device,
        dtype=torch.float32,
    )
    audio_mask = torch.ones(
        (1, 1, out_audio.shape[2], out_audio.shape[3]),
        device=out_audio.device,
        dtype=torch.float32,
    )
    context_denoise = max(0.0, min(1.0, float(video_context_denoise)))
    video_mask[:, :, :video_steps] = context_denoise

    feather = max(0, min(int(audio_feather_ticks), audio_steps))
    hard = audio_steps - feather
    if hard:
        audio_mask[..., :hard] = 0.0
    if feather:
        i = torch.arange(1, feather + 1, device=audio_mask.device, dtype=audio_mask.dtype)
        audio_mask[..., hard:audio_steps] = 0.5 - 0.5 * torch.cos(torch.pi * i / float(feather))

    plan.update({
        "audio_start_tick": a0,
        "audio_end_tick": a1,
        "audio_steps": audio_steps,
        "audio_tail_carryover": audio_tail_carryover,
        "audio_feather_ticks": feather,
        "video_context_denoise": context_denoise,
    })
    return out_video, out_audio, video_mask, audio_mask, plan
