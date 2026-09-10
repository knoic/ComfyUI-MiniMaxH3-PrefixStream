"""3D-RoPE Coordinate Alignment & Temporal Cursor Manager for MiniMax H3 (Reference).

Provides continuous temporal coordinate grid computation and temporal tracking
for MiniMax H3 spatio-temporal alignment.
"""

from typing import List, Tuple
import math
import torch

FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FRAME_RESCALE = 5.0 / 3.0
FPS = 24


def video_t_spans(n_latent_steps: int) -> List[float]:
    """Span in continuous time units that each latent step occupies."""
    return [FRAME_RESCALE * FRAME_PER_TOKEN[k % 5] for k in range(n_latent_steps)]


def video_t_grid(n_latent_steps: int, origin: float = 0.0) -> torch.Tensor:
    """Calculates continuous temporal coordinates for n latent steps starting at origin."""
    spans = torch.tensor(video_t_spans(n_latent_steps), dtype=torch.float64)
    if n_latent_steps == 1:
        return torch.tensor([float(origin)], dtype=torch.float64)
    return float(origin) + torch.cat([torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)])


def total_span_for_steps(n_latent_steps: int) -> float:
    """Total time span advanced by n latent steps."""
    return sum(video_t_spans(n_latent_steps))


class TemporalCursorTracker:
    """Tracks global video timeline positions across continuous clips for 3D-RoPE."""

    def __init__(self):
        self.current_origin: float = 0.0
        self.clip_history: List[dict] = []

    def register_clip(
        self,
        clip_index: int,
        latent_steps: int,
        anchor_steps: int = 2,
        rolling_steps: int = 6
    ) -> dict:
        """Computes timeline coordinates for a new clip."""
        span = total_span_for_steps(latent_steps)
        clip_info = {
            "clip_index": clip_index,
            "origin": self.current_origin,
            "latent_steps": latent_steps,
            "span": span,
            "end_time": self.current_origin + span,
            "anchor_steps": anchor_steps,
            "rolling_steps": rolling_steps,
        }
        self.clip_history.append(clip_info)
        # Advance timeline for the next generated chunk
        # Note: In rolling mode, new frames start after the rolling overlap
        generated_steps = max(1, latent_steps - rolling_steps)
        self.current_origin += total_span_for_steps(generated_steps)
        return clip_info

    def get_time_coords(
        self,
        n_steps: int,
        origin: float
    ) -> torch.Tensor:
        """Get 1D temporal coordinate tensor for n steps."""
        return video_t_grid(n_steps, origin)

    def get_decoupled_coords(
        self,
        prefix_steps: int,
        target_steps: int,
        base_origin: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculates perfectly contiguous, non-negative 3D-RoPE temporal coordinates.

        Prefix spans [base_origin, base_origin + prefix_span).
        Target generation seamlessly continues at [base_origin + prefix_span, ...).
        Zero overlap on target timeline while preserving full temporal causality (Δt > 0).
        """
        prefix_coords = video_t_grid(prefix_steps, origin=base_origin)
        target_origin = base_origin + total_span_for_steps(prefix_steps)
        target_coords = video_t_grid(target_steps, origin=target_origin)
        return prefix_coords, target_coords

    def reset(self):
        self.current_origin = 0.0
        self.clip_history.clear()
