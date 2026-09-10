"""Append decoded clips to disk and export without retaining the full IMAGE timeline."""

import json
import os
import re
import shutil
import subprocess
import uuid

try:
    from ..engine.clip_bin_manager import (
        atomic_write_json, encode_images_to_mp4, get_project_dir, project_locked,
    )
except ImportError:
    from engine.clip_bin_manager import (
        atomic_write_json, encode_images_to_mp4, get_project_dir, project_locked,
    )
from .seam_protector import trim_images_and_audio


@project_locked
def append_disk_clip(project_name, stream_name, images, audio=None, trim_frames=0,
                     fps=24.0, export=False):
    """Use PCM intermediate audio to avoid adding AAC encoder delay at every join."""
    if not re.fullmatch(r"[\w-]+", stream_name):
        raise ValueError("Stream name must contain only letters, digits, underscores or hyphens")
    directory = os.path.join(get_project_dir(project_name), ".streams", stream_name)
    if os.path.normcase(os.path.realpath(directory)) != os.path.normcase(os.path.abspath(directory)):
        raise ValueError("Linked stream directories are not supported")
    os.makedirs(directory, exist_ok=True)
    manifest_path = os.path.join(directory, "manifest.json")
    state = {"version": 1, "clips": [], "total_frames": 0}
    if os.path.isfile(manifest_path):
        with open(manifest_path, encoding="utf-8") as source:
            state = json.load(source)
    if images is not None:
        images, audio = trim_images_and_audio(images, audio, trim_frames, fps)
        if images.shape[0] == 0:
            raise ValueError("No frames remain after trimming")
        # The encoder uses even RGB dimensions and a single audio batch.
        if images.shape[-1] not in (3, 4) or any(n % 2 for n in images.shape[1:3]):
            raise ValueError("Disk streams require RGB/RGBA images with even height and width")
        if audio is not None and (audio["waveform"].ndim != 3 or audio["waveform"].shape[0] != 1):
            raise ValueError("Disk streams require audio shape [1, channels, samples]")
        geometry = {"fps": float(fps), "height": images.shape[1], "width": images.shape[2],
                    "sample_rate": int(audio["sample_rate"]) if audio else None,
                    "channels": audio["waveform"].shape[1] if audio else None}
        if state["clips"] and state["geometry"] != geometry:
            raise ValueError("Stream resolution, fps, sample rate and channels must remain consistent")
        filename = f"segment_{uuid.uuid4().hex}.mkv"
        path = os.path.join(directory, filename)
        if not encode_images_to_mp4(images, path, fps, audio):
            raise RuntimeError("Segment encoding failed; stream manifest was not changed")
        state["geometry"] = geometry
        state["clips"].append({"file": filename, "frames": int(images.shape[0])})
        state["total_frames"] += int(images.shape[0])
        atomic_write_json(manifest_path, state)

    output_path = ""
    if export:
        if not state["clips"]:
            raise ValueError("Cannot export an empty stream")
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg executable is required for disk stream export")
        # Only generated local segment names are accepted in the concat manifest.
        concat = os.path.join(directory, "concat.txt")
        with open(concat, "w", encoding="utf-8") as target:
            for clip in state["clips"]:
                if not re.fullmatch(r"segment_[0-9a-f]{32}\.mkv", clip["file"]):
                    raise ValueError("Invalid segment filename in stream manifest")
                target.write(f"file '{clip['file']}'\n")
                target.write(f"duration {clip['frames'] / state['geometry']['fps']:.12f}\n")
        temp_output = os.path.join(directory, f"export_{uuid.uuid4().hex}.mp4")
        command = [ffmpeg, "-v", "error", "-f", "concat", "-safe", "1", "-i", concat,
                   "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", temp_output]
        try:
            subprocess.run(command, check=True, capture_output=True, timeout=3600)
            output_path = os.path.join(directory, "video.mp4")
            os.replace(temp_output, output_path)
        finally:
            if os.path.exists(temp_output):
                os.remove(temp_output)
    return manifest_path, output_path, state["total_frames"]


class MiniMaxDiskVideoStreamNode:
    """Disk-backed hard-cut assembly; use the IMAGE stitcher for seam effects."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "project_name": ("STRING", {"default": "Default_Project"}),
            "stream_name": ("STRING", {"default": "long_video"}),
            "trim_frames": ("INT", {"default": 0, "min": 0, "max": 192}),
            "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0}),
            "export": ("BOOLEAN", {"default": False}),
        }, "optional": {"images": ("IMAGE",), "audio": ("AUDIO",), "session": ("MINIMAX_SESSION",)}}

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("manifest_path", "video_path", "total_frames")
    FUNCTION = "append"
    CATEGORY = "MiniMaxH3/PrefixStream"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def append(self, project_name, stream_name, trim_frames=0, fps=24.0, export=False,
               images=None, audio=None, session=None):
        if trim_frames == 0 and session is not None:
            trim_frames = session.last_rolling_frames
        return append_disk_clip(project_name, stream_name, images, audio, trim_frames, fps, export)
