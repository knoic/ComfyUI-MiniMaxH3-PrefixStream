"""Tests for the Native Masked AV continuation path."""

import os
import sys
import unittest
from unittest import mock

import torch


_repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_dir not in sys.path:
    sys.path.insert(0, _repo_dir)

from engine.cache_manager import KVCacheConfig
import nodes
from pipeline.native_masked_av import (
    apply_native_masked_av,
    is_exact_masked_av_context,
    masked_av_tail_slice,
)


class NativeMaskedAVTests(unittest.TestCase):
    def test_default_mode_and_exact_context_grid(self):
        cfg = KVCacheConfig()
        self.assertTrue(cfg.is_native_masked_av_mode())
        self.assertEqual(cfg.rolling_frames, 39)
        self.assertTrue(is_exact_masked_av_context(39))
        self.assertTrue(is_exact_masked_av_context(90))
        self.assertFalse(is_exact_masked_av_context(22))

    def test_canonical_tail_slice(self):
        plan = masked_av_tail_slice(video_latent_t=37, context_frames=39, target_video_t=37)
        self.assertEqual(plan["context_steps"], 12)
        self.assertEqual(plan["actual_context_frames"], 39)
        self.assertEqual(plan["start_t"] % 5, 0)
        self.assertEqual(plan["end_t"] % 5, 2)
        self.assertEqual(plan["ignored_tail_frames"], 0)

    def test_copies_av_and_builds_independent_masks(self):
        source_video = torch.randn(1, 24, 37, 4, 4)
        source_audio = torch.randn(1, 32, 2, 207)
        target_video = torch.zeros_like(source_video)
        target_audio = torch.zeros_like(source_audio)
        out_v, out_a, video_mask, audio_mask, plan = apply_native_masked_av(
            target_video, target_audio, source_video, source_audio, context_frames=39
        )
        self.assertTrue(torch.equal(out_v[:, :, :12], source_video[:, :, 25:37]))
        self.assertEqual(plan["audio_steps"], 65)
        self.assertTrue(torch.equal(out_a[..., :65], source_audio[..., 142:207]))
        self.assertTrue(torch.all(video_mask[:, :, :12] == 0))
        self.assertTrue(torch.all(video_mask[:, :, 12:] == 1))
        self.assertTrue(torch.all(audio_mask[..., :65] == 0))
        self.assertTrue(torch.all(audio_mask[..., 65:] == 1))

    def test_audio_feather_releases_only_audio_tail(self):
        source_video = torch.randn(1, 24, 37, 2, 2)
        source_audio = torch.randn(1, 32, 2, 207)
        target_video = torch.zeros_like(source_video)
        target_audio = torch.zeros_like(source_audio)
        _, _, video_mask, audio_mask, _ = apply_native_masked_av(
            target_video, target_audio, source_video, source_audio,
            context_frames=39, audio_feather_ticks=4,
        )
        self.assertTrue(torch.all(video_mask[:, :, :12] == 0))
        self.assertTrue(torch.all(audio_mask[..., :61] == 0))
        self.assertTrue(torch.all(audio_mask[..., 61:65] > 0))

    def test_applier_outputs_masked_latent_and_removes_head_keyframe(self):
        source = nodes.pack_av_latent(
            torch.randn(1, 24, 37, 2, 2),
            torch.randn(1, 32, 2, 207),
        )
        target = nodes.pack_av_latent(
            torch.zeros(1, 24, 37, 2, 2),
            torch.zeros(1, 32, 2, 207),
        )
        conditioning = [[torch.zeros(1), {"minimax_keyframes": [
            {"resolved_frame_index": 0, "latent": torch.zeros(1)},
            {"resolved_frame_index": 123, "latent": torch.ones(1)},
        ]}]]
        with mock.patch.object(nodes, "_require_native_masked_av_support", return_value=None):
            model, out_cond, session, masked = nodes.MiniMaxPrefixCacheApplierNode().apply_cache(
                model=object(),
                conditioning=conditioning,
                cache_config=KVCacheConfig(),
                context_latent=source,
                target_latent=target,
            )
        self.assertIsNotNone(model)
        self.assertEqual(session.last_rolling_frames, 39)
        self.assertIn("noise_mask", masked)
        self.assertEqual(len(out_cond[0][1]["minimax_keyframes"]), 1)
        self.assertEqual(out_cond[0][1]["minimax_keyframes"][0]["resolved_frame_index"], 123)

    def test_applier_falls_back_for_a_short_target(self):
        source = nodes.pack_av_latent(
            torch.randn(1, 24, 2, 2, 2),
            torch.randn(1, 32, 2, 8),
        )
        target = nodes.pack_av_latent(
            torch.zeros(1, 24, 2, 2, 2),
            torch.zeros(1, 32, 2, 8),
        )
        with mock.patch.object(nodes, "_require_native_masked_av_support", return_value=None):
            model, _cond, session, out_latent = nodes.MiniMaxPrefixCacheApplierNode().apply_cache(
                model=object(),
                conditioning=[[torch.zeros(1), {}]],
                cache_config=KVCacheConfig(),
                context_latent=source,
                target_latent=target,
            )
        self.assertIsNotNone(model)
        self.assertIs(out_latent, target)
        self.assertEqual(session.last_rolling_frames, 5)


if __name__ == "__main__":
    unittest.main()
