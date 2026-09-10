"""Unit tests for audio processing enhancements and fixes."""

import os
import sys
import unittest
import torch

_repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_dir not in sys.path:
    sys.path.insert(0, _repo_dir)

from engine.cache_manager import KVCacheConfig
from pipeline.long_video_director import LongVideoSession
from pipeline.native_masked_av import apply_native_masked_av, masked_av_tail_slice
from pipeline.seam_protector import fit_audio_length, stitch_audio_waveforms
import nodes


class AudioEnhancementsTests(unittest.TestCase):
    def test_fit_audio_length_no_dc_plateau(self):
        # A waveform ending at non-zero amplitude 0.8
        w = torch.ones(1, 2, 100) * 0.8
        target = 200
        padded = fit_audio_length(w, target)
        self.assertEqual(padded.shape, (1, 2, target))

        # Check that the tail (from index 100 to 200) is padded with zeros, NOT repeating 0.8
        pad_region = padded[..., 100:]
        self.assertTrue(torch.all(pad_region == 0.0), "Padded region should be all zeros to prevent DC plateau")
        # Check that the last sample before padding was smoothly attenuated
        self.assertLess(float(padded[..., 99].abs().max()), 0.8, "Tail should fade out before zero padding")

    def test_fit_audio_length_truncate_and_exact(self):
        w = torch.randn(1, 1, 150)
        self.assertEqual(fit_audio_length(w, 150).shape[-1], 150)
        self.assertEqual(fit_audio_length(w, 50).shape[-1], 50)
        self.assertEqual(fit_audio_length(w, 0).shape[-1], 0)

    def test_stitch_audio_waveforms_equal_power_and_clamping(self):
        sr = 32000
        fps = 24.0
        # Create two 1-second audio clips with high amplitude to test clamping
        prev_audio = {"waveform": torch.ones(1, 2, 32000) * 0.9, "sample_rate": sr}
        curr_audio = {"waveform": torch.ones(1, 2, 32000) * 0.9, "sample_rate": sr}

        stitched_dict = stitch_audio_waveforms(
            prev_audio=prev_audio,
            curr_audio=curr_audio,
            curr_total_frames=24,
            trim_frames=6,
            crossfade_ms=20.0,
            fps=fps
        )
        self.assertIsNotNone(stitched_dict)
        stitched = stitched_dict["waveform"]
        self.assertEqual(stitched.shape[0], 1)
        self.assertEqual(stitched.shape[1], 2)
        # Verify clamp: no sample should exceed 1.0 or fall below -1.0
        self.assertLessEqual(float(stitched.max()), 1.0)
        self.assertGreaterEqual(float(stitched.min()), -1.0)

    def test_audio_handover_vs_full_previous_tail(self):
        # Source video with 65 latent steps (226 frames).
        # Snap will find canonical end_t = 62 (209 frames), leaving 17 ignored tail frames.
        source_video = torch.randn(1, 24, 65, 2, 2)
        source_audio = torch.randn(1, 32, 2, 377)
        target_video = torch.zeros(1, 24, 37, 2, 2)
        target_audio = torch.zeros(1, 32, 2, 207)

        # 1. Match Video Handover: audio aligns with video handover boundary (no ghost tail)
        _, _, _, _, plan_handover = apply_native_masked_av(
            target_video=target_video,
            target_audio=target_audio,
            source_video=source_video,
            source_audio=source_audio,
            context_frames=39,
            audio_tail_carryover="Match Video Handover"
        )
        # 2. Full Previous Tail: audio extends all the way to frame 226
        _, _, _, _, plan_full = apply_native_masked_av(
            target_video=target_video,
            target_audio=target_audio,
            source_video=source_video,
            source_audio=source_audio,
            context_frames=39,
            audio_tail_carryover="Full Previous Tail"
        )

        # Handover mode should have shorter protected audio steps than Full mode
        # because it does not copy the 17 ignored tail frames (~0.7s)
        self.assertLess(plan_handover["audio_steps"], plan_full["audio_steps"])
        self.assertEqual(plan_handover["audio_steps"], 65)  # 39 frames * 40 / 24 = 65 steps

    def test_stitcher_double_trim_guard(self):
        cfg = KVCacheConfig(rolling_frames=39)
        session = LongVideoSession(cfg)
        session.last_rolling_frames = 39

        stitcher = nodes.MiniMaxLongVideoStitcherNode()
        curr_images = torch.zeros(48, 64, 64, 3)
        curr_audio = {"waveform": torch.zeros(1, 2, 64000), "sample_rate": 32000}
        prev_images = torch.zeros(72, 64, 64, 3)
        prev_audio = {"waveform": torch.zeros(1, 2, 96000), "sample_rate": 32000}

        # Case 1: Clip has NOT been trimmed upstream -> stitcher trims 39 frames
        session.is_current_clip_trimmed = False
        out_img1, out_aud1, count1 = stitcher.stitch(
            trim_frames=0,
            crossfade_frames=2,
            prev_images=prev_images,
            curr_images=curr_images,
            prev_audio=prev_audio,
            curr_audio=curr_audio,
            session=session
        )
        # Expected frames: prev (72) + curr (48 - 39) = 81 frames
        self.assertEqual(out_img1.shape[0], 81)

        # Case 2: Clip was ALREADY trimmed upstream -> stitcher skips trimming (0 frames)
        session.is_current_clip_trimmed = True
        out_img2, out_aud2, count2 = stitcher.stitch(
            trim_frames=0,
            crossfade_frames=2,
            prev_images=prev_images,
            curr_images=curr_images,
            prev_audio=prev_audio,
            curr_audio=curr_audio,
            session=session
        )
        # Expected frames: prev (72) + curr (48 - 0) = 120 frames
        self.assertEqual(out_img2.shape[0], 120)


if __name__ == "__main__":
    unittest.main()
