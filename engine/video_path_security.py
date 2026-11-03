"""Shared filesystem boundary for HTTP-accessible timeline videos."""

import os

VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".webm", ".avi", ".flv", ".wmv", ".m4v", ".gif")


def validate_video_path(file_path: str) -> str:
    """Return a canonical allowed video path, or raise ValueError.

    Clip bins live beneath output, so allowing output also covers clip bins.
    Fail closed when ComfyUI's configured directories are unavailable.
    """
    import folder_paths

    if not file_path:
        raise ValueError("Empty video path")
    target = os.path.realpath(file_path)
    normalized = os.path.normcase(target)
    if not normalized.endswith(VIDEO_EXTENSIONS):
        raise ValueError("Unsupported video extension")
    for directory in (folder_paths.get_input_directory(), folder_paths.get_output_directory()):
        root = os.path.normcase(os.path.realpath(directory))
        try:
            if os.path.commonpath([root, normalized]) == root:
                return target
        except ValueError:
            # Different Windows drives or UNC shares cannot contain each other.
            continue
    raise ValueError("Video path is outside allowed directories")
