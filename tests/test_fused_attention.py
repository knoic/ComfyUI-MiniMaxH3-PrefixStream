"""Unit tests for asymmetric attention and RoPE / Seam protector."""

import sys
import os
import math
import unittest
import torch

_repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_dir not in sys.path:
    sys.path.insert(0, _repo_dir)

from engine.fused_attention import asymmetric_cached_attention
from engine.rope_aligner import video_t_spans, video_t_grid, TemporalCursorTracker
from pipeline.seam_protector import (
    audio_equal_power_crossfade,
    latent_soft_blend,
    trim_prefix_frames,
    estimate_luminance_gain,
    apply_luminance_gain_fade,
    stitch_video_images,
    stitch_audio_waveforms,
)


class TestFusedAttentionAndSeamProtector(unittest.TestCase):
    def test_asymmetric_cached_attention(self):
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

        self.assertEqual(out.shape, (s_target, heads * head_dim))
        self.assertFalse(torch.isnan(out).any())

    def test_rope_video_grids(self):
        spans = video_t_spans(5)
        self.assertEqual(len(spans), 5)
        self.assertTrue(math.isclose(spans[0], 1.0 * 5.0 / 3.0))
        self.assertTrue(math.isclose(spans[1], 4.0 * 5.0 / 3.0))

        grid = video_t_grid(3, origin=10.0)
        self.assertEqual(grid.shape, (3,))
        self.assertTrue(math.isclose(float(grid[0]), 10.0))
        self.assertGreater(float(grid[1]), 10.0)

    def test_cursor_tracker(self):
        tracker = TemporalCursorTracker()
        c0 = tracker.register_clip(clip_index=0, latent_steps=20, rolling_steps=6)
        self.assertEqual(c0["origin"], 0.0)

        c1 = tracker.register_clip(clip_index=1, latent_steps=20, rolling_steps=6)
        self.assertGreater(c1["origin"], 0.0)
        self.assertEqual(len(tracker.clip_history), 2)

    def test_seam_protector_audio_and_latent(self):
        w1 = torch.ones(1, 2, 8000)
        w2 = torch.full((1, 2, 8000), 2.0)
        blended_audio = audio_equal_power_crossfade(w1, w2, crossfade_samples=1600)
        self.assertEqual(blended_audio.shape, (1, 2, 8000 + 8000 - 1600))

        l1 = torch.ones(1, 24, 10, 16, 16)
        l2 = torch.full((1, 24, 10, 16, 16), 3.0)
        blended_latent = latent_soft_blend(l1, l2, blend_steps=2)
        self.assertEqual(blended_latent.shape, (1, 24, 18, 16, 16))

        trimmed = trim_prefix_frames(blended_latent, prefix_latent_steps=6)
        self.assertEqual(trimmed.shape, (1, 24, 12, 16, 16))

    def test_stitch_video_and_audio(self):
        prev_img = torch.ones(30, 64, 64, 3, dtype=torch.float32)
        curr_img = torch.ones(25, 64, 64, 3, dtype=torch.float32)
        stitched_img = stitch_video_images(
            prev_images=prev_img,
            curr_images=curr_img,
            trim_frames=10,
            crossfade_frames=2,
        )
        self.assertEqual(stitched_img.shape[0], 30 + (25 - 10))

        prev_aud = {"waveform": torch.zeros(1, 2, 32000), "sample_rate": 32000}
        curr_aud = {"waveform": torch.zeros(1, 2, 32000), "sample_rate": 32000}
        stitched_aud = stitch_audio_waveforms(
            prev_audio=prev_aud,
            curr_audio=curr_aud,
            curr_total_frames=25,
            trim_frames=10,
            fps=24.0,
        )
        self.assertIsNotNone(stitched_aud)
        self.assertIn("waveform", stitched_aud)


if __name__ == "__main__":
    unittest.main()
