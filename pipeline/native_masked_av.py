"""Native MiniMax H3 video/audio masked-continuation helpers.

The geometry follows Herrgott's H3 Infinite Continuation Suite: copied
video context must be a canonical H3 run whose duration is exact on both the
24 fps video and 40 Hz audio latent timelines (39, 90, 141, ... frames).
"""

from typing import Any, Dict, Tuple

import torch


FPS = 24
AUDIO_HZ = 40
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
MASKED_AV_CONTEXTS = (39, 90, 141, 192)


def latent_boundaries(video_latent_t: int):
    """Return exclusive pixel-frame boundaries for an H3 video latent."""
    boundaries = [0]
    for step in range(int(video_latent_t)):
        boundaries.append(boundaries[-1] + FRAME_PER_TOKEN[step % 5])
    return boundaries


def is_exact_masked_av_context(frame_count: int) -> bool:
    frame_count = int(frame_count)
    return frame_count >= 39 and (frame_count - 39) % 51 == 0


def snap_masked_av_context_length(requested: int, available: int, target_frames: int) -> int:
    """Snap down to the largest joint video/audio context that leaves new content."""
    cap = min(int(requested), int(available), int(target_frames) - 1)
    if cap < 39:
        raise ValueError(
            "Native Masked AV needs at least 39 source frames and a target longer than 39 frames"
        )
    run = 39 + ((cap - 39) // 51) * 51
    if not is_exact_masked_av_context(run):
        raise RuntimeError("Internal Native Masked AV context snap failed")
    return run


def masked_av_tail_slice(video_latent_t: int, context_frames: int, target_video_t: int) -> Dict[str, int]:
    """Select the latest canonical source tail for a target masked prefix."""
    source_boundaries = latent_boundaries(video_latent_t)
    target_frames = latent_boundaries(target_video_t)[-1]
    previous_frames = source_boundaries[-1]
    frames = snap_masked_av_context_length(context_frames, previous_frames, target_frames)
    context_steps = 2 + 5 * ((frames - 5) // 17)

    end_t = None
    for candidate in range(int(video_latent_t), context_steps - 1, -1):
        start_t = candidate - context_steps
        if candidate % 5 == 2 and start_t % 5 == 0:
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


def _audio_window(audio_t: int, start_frame: int, end_frame: int) -> Tuple[int, int]:
    a0 = max(0, min(int(audio_t), round(int(start_frame) * AUDIO_HZ / FPS)))
    a1 = max(0, min(int(audio_t), round(int(end_frame) * AUDIO_HZ / FPS)))
    if a1 <= a0:
        raise ValueError("Native Masked AV audio context is empty")
    return a0, a1


def apply_native_masked_av(
    target_video: torch.Tensor,
    target_audio: torch.Tensor,
    source_video: torch.Tensor,
    source_audio: torch.Tensor,
    context_frames: int = 39,
    audio_tail_carryover: str = "Full Previous Tail",
    audio_feather_ticks: int = 0,
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

    plan = masked_av_tail_slice(source_video.shape[2], context_frames, target_video.shape[2])
    video_steps = plan["context_steps"]
    source_run = source_video[:, :, plan["start_t"]:plan["end_t"]]

    mode = str(audio_tail_carryover).strip().lower()
    if mode not in ("full previous tail", "match video handover"):
        raise ValueError(f"Unknown audio tail carryover mode: {audio_tail_carryover!r}")
    audio_end_frame = (
        plan["previous_frame_count"] if mode == "full previous tail" else plan["source_end_frame"]
    )
    a0, a1 = _audio_window(source_audio.shape[-1], plan["source_start_frame"], audio_end_frame)
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
    video_mask[:, :, :video_steps] = 0.0

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
    })
    return out_video, out_audio, video_mask, audio_mask, plan
