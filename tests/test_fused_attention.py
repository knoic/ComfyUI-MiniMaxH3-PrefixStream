"""Unit tests for asymmetric attention and RoPE / Seam protector."""

import sys
import os
import math
import torch
import torch.nn.functional as F

# Ensure parent directory is in sys.path for local standalone testing
_repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_dir not in sys.path:
    sys.path.insert(0, _repo_dir)

try:
    from engine.fused_attention import asymmetric_cached_attention
    from engine.rope_aligner import video_t_spans, video_t_grid, TemporalCursorTracker
    from pipeline.seam_protector import (
        audio_equal_power_crossfade,
        latent_soft_blend,
        trim_prefix_frames,
        estimate_luminance_gain,
        apply_luminance_gain_fade,
        stitch_video_images,
    )
except ImportError:
    from ComfyUI_MiniMaxH3_PrefixStream.engine.fused_attention import asymmetric_cached_attention
    from ComfyUI_MiniMaxH3_PrefixStream.engine.rope_aligner import video_t_spans, video_t_grid, TemporalCursorTracker
    from ComfyUI_MiniMaxH3_PrefixStream.pipeline.seam_protector import (
        audio_equal_power_crossfade,
        latent_soft_blend,
        trim_prefix_frames,
        estimate_luminance_gain,
        apply_luminance_gain_fade,
        stitch_video_images,
    )


def test_asymmetric_cached_attention():
    heads = 8
    head_dim = 64
    s_target = 40
    s_prefix = 60
    s_total = s_prefix + s_target

    q_t = torch.randn(s_target, heads, head_dim, dtype=torch.float32)
    k_cached = torch.randn(1, heads, s_total, head_dim, dtype=torch.float32)
    v_cached = torch.randn(1, heads, s_total, head_dim, dtype=torch.float32)

    out = asymmetric_cached_attention(
        q=q_t,
        k_cached=k_cached,
        v_cached=v_cached,
        num_heads=heads
    )

    # Expected output: [s_target, heads * head_dim]
    assert out.shape == (s_target, heads * head_dim)
    assert not torch.isnan(out).any()


def test_rope_video_grids():
    spans = video_t_spans(5)
    # (1, 4, 4, 4, 4) * (5/3)
    assert len(spans) == 5
    assert math.isclose(spans[0], 1.0 * 5.0 / 3.0)
    assert math.isclose(spans[1], 4.0 * 5.0 / 3.0)

    grid = video_t_grid(3, origin=10.0)
    assert grid.shape == (3,)
    assert math.isclose(float(grid[0]), 10.0)
    assert float(grid[1]) > 10.0


def test_cursor_tracker():
    tracker = TemporalCursorTracker()
    c0 = tracker.register_clip(clip_index=0, latent_steps=20, rolling_steps=6)
    assert c0["origin"] == 0.0

    c1 = tracker.register_clip(clip_index=1, latent_steps=20, rolling_steps=6)
    assert c1["origin"] > 0.0
    assert len(tracker.clip_history) == 2


def test_seam_protector():
    # Audio crossfade test
    w1 = torch.ones(1, 2, 8000)
    w2 = torch.full((1, 2, 8000), 2.0)
    blended_audio = audio_equal_power_crossfade(w1, w2, crossfade_samples=1600)
    assert blended_audio.shape == (1, 2, 8000 + 8000 - 1600)

    # Latent soft blend test
    l1 = torch.ones(1, 24, 10, 16, 16)
    l2 = torch.full((1, 24, 10, 16, 16), 3.0)
    blended_latent = latent_soft_blend(l1, l2, blend_steps=2)
    assert blended_latent.shape == (1, 24, 18, 16, 16)

    # Trim test
    trimmed = trim_prefix_frames(blended_latent, prefix_latent_steps=6)
    assert trimmed.shape == (1, 24, 12, 16, 16)


def test_luminance_and_pixel_stitching():
    # 24 frames of 64x64 images
    prev = torch.full((24, 64, 64, 3), 0.5, dtype=torch.float32)
    curr = torch.full((24, 64, 64, 3), 0.6, dtype=torch.float32)

    gain = estimate_luminance_gain(prev[-4:], curr[:4])
    assert 0.5 < gain < 1.5

    faded = apply_luminance_gain_fade(curr, gain=gain, fade_frames=8)
    assert faded.shape == curr.shape

    stitched = stitch_video_images(prev, curr, trim_frames=8, crossfade_frames=4, luminance_match=True)
    # Total frames: 24 - 4 (overlap crossfade) + (24 - 8) = 24 + 16 = 40 frames
    assert stitched.shape[0] == (24 + 24 - 8)


if __name__ == "__main__":
    test_asymmetric_cached_attention()
    test_rope_video_grids()
    test_cursor_tracker()
    test_seam_protector()
    test_luminance_and_pixel_stitching()
    print("All fused_attention & seam_protector tests passed successfully!")
