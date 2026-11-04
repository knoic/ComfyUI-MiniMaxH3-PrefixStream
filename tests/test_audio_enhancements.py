"""Unit tests for audio processing enhancements and fixes."""

import os
import sys
import unittest
from types import SimpleNamespace
import torch

_repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_dir not in sys.path:
    sys.path.insert(0, _repo_dir)

from engine.cache_manager import KVCacheConfig
from pipeline.long_video_director import LongVideoSession
from pipeline.native_masked_av import apply_native_masked_av, masked_av_tail_slice
from pipeline.seam_protector import fit_audio_length, stitch_audio_waveforms, trim_images_and_audio
import nodes


class AudioEnhancementsTests(unittest.TestCase):
    def test_safe_decoder_converts_comfy_vae_layout_and_sample_rate(self):
        raw = torch.stack((torch.linspace(-0.1, 0.1, 4800),
                           torch.linspace(0.1, -0.1, 4800)), dim=-1)[None]
        vae = SimpleNamespace(decode=lambda _: raw, audio_sample_rate=32000,
                              audio_sample_rate_output=48000)
        audio, = nodes.MiniMaxSafeVAEDecodeAudioNode().decode(
            vae, {"samples": torch.zeros(1, 32, 2, 4)})
        self.assertEqual(audio["sample_rate"], 48000)
        self.assertEqual(audio["waveform"].shape, (1, 2, 4800))
        torch.testing.assert_close(audio["waveform"], raw.movedim(-1, 1))

    def test_trim_then_stitch_matches_direct_stitch(self):
        sr = 24000
        signal = 0.8 * torch.sin(torch.arange(3000) * (2 * torch.pi * 440 / sr))
        prev = {"waveform": signal[None, None, :2000], "sample_rate": sr}
        # Deliberately different decoded overlap: a hard join would pop.
        curr = {"waveform": -signal[None, None, 1000:], "sample_rate": sr}
        images = torch.zeros(2, 2, 2, 3)
        direct = stitch_audio_waveforms(prev, curr, 2, trim_frames=1)
        trimmed_images, trimmed = trim_images_and_audio(images, curr, 1)
        joined = stitch_audio_waveforms(prev, trimmed, len(trimmed_images), trim_frames=0)
        torch.testing.assert_close(joined["waveform"], direct["waveform"])
        self.assertEqual(joined["waveform"].shape[-1], 3000)
        self.assertNotIn("minimax_audio_context", joined)
        # The samples at the handover must be consecutive samples of curr.
        torch.testing.assert_close(joined["waveform"][..., 1999:2001], curr["waveform"][..., 999:1001])

    def test_contextless_join_declicks_without_shortening_or_mutating_inputs(self):
        prev = {"waveform": torch.full((1, 2, 2000), 0.8), "sample_rate": 24000}
        curr = {"waveform": torch.full((1, 2, 1000), -0.8), "sample_rate": 24000}
        result = stitch_audio_waveforms(prev, curr, 1, trim_frames=0)["waveform"]
        self.assertEqual(result.shape[-1], 3000)
        torch.testing.assert_close(result[..., 1999:2001], torch.zeros(1, 2, 2))
        self.assertLess(float(torch.diff(result).abs().max()), 0.01)
        self.assertTrue(torch.all(prev["waveform"] == 0.8))
        self.assertTrue(torch.all(curr["waveform"] == -0.8))
        torch.testing.assert_close(result[..., 2360:], curr["waveform"][..., 360:])

    def test_trim_and_stitch_nodes_keep_audio_context_with_session(self):
        prev_images = torch.zeros(2, 2, 2, 3)
        curr_images = torch.zeros(2, 2, 2, 3)
        prev = {"waveform": torch.full((1, 2, 2000), 0.8), "sample_rate": 24000}
        curr = {"waveform": torch.full((1, 2, 2000), -0.8), "sample_rate": 24000}
        for with_images in (False, True):
            with self.subTest(with_images=with_images):
                session = LongVideoSession(KVCacheConfig())
                session.last_rolling_frames = 1
                images, audio, _ = nodes.MiniMaxTrimPrefixLatentNode().trim(
                    audio=curr, images=curr_images if with_images else None, session=session)
                images = images if with_images else curr_images[1:]
                out_images, out_audio, count = nodes.MiniMaxLongVideoStitcherNode().stitch(
                    prev_images=prev_images, curr_images=images, prev_audio=prev,
                    curr_audio=audio, session=session)
                self.assertEqual(count, 3)
                self.assertEqual(out_audio["waveform"].shape[-1], 3000)
                torch.testing.assert_close(out_audio["waveform"][..., 1999:2001],
                                           torch.full((1, 2, 2), -0.8))

    def test_stitch_does_not_clip_unrelated_audio_or_change_empty_join(self):
        prev = {"waveform": torch.full((1, 1, 2000), 1.2), "sample_rate": 24000}
        curr = {"waveform": torch.full((1, 1, 1000), 1.2), "sample_rate": 24000}
        result = stitch_audio_waveforms(prev, curr, 1, trim_frames=0)
        torch.testing.assert_close(result["waveform"][..., :1000], prev["waveform"][..., :1000])
        empty = stitch_audio_waveforms(prev, curr, 1, trim_frames=1)
        torch.testing.assert_close(empty["waveform"], prev["waveform"])

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

    def test_identical_audio_overlap_does_not_amplify_or_clip(self):
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
        torch.testing.assert_close(stitched, torch.full_like(stitched, 0.9), atol=1e-7, rtol=0)

    def test_identical_sine_overlap_preserves_waveform_and_timeline(self):
        sr = 24000
        signal = 0.95 * torch.sin(torch.arange(3000) * (2 * torch.pi * 440 / sr))
        prev = {"waveform": signal[None, None, :2000], "sample_rate": sr}
        curr = {"waveform": signal[None, None, 1000:], "sample_rate": sr}
        result = stitch_audio_waveforms(prev, curr, curr_total_frames=2, trim_frames=1, fps=24, crossfade_ms=20)
        self.assertEqual(result["waveform"].shape[-1], 3000)
        torch.testing.assert_close(result["waveform"][0, 0], signal, atol=1e-7, rtol=0)

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
