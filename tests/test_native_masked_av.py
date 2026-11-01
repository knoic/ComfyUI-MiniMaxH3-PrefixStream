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
from engine.continuation_state import prepare_identity_references
import nodes
from pipeline.native_masked_av import (
    apply_native_masked_av,
    is_exact_masked_av_context,
    masked_av_tail_slice,
)
from pipeline.long_video_director import LongVideoSession


class NativeMaskedAVTests(unittest.TestCase):
    def test_all_native_images_survive_missing_references_without_growth(self):
        refs = [{"kind": "image", "latent": torch.randn(1, 24, 1, 2, 2), "latent_h": 2, "latent_w": 2}
                for _ in range(3)]
        cond = [[torch.zeros(1), {"minimax_refs": refs}]]
        original, state = prepare_identity_references(cond)
        self.assertIs(original, cond)
        for _ in range(5):
            restored, state = prepare_identity_references([[torch.zeros(1), {}]], state)
            self.assertEqual(len(restored[0][1]["minimax_refs"]), 3)
            for expected, ref in zip(refs, restored[0][1]["minimax_refs"]):
                self.assertTrue(torch.equal(expected["latent"], ref["latent"]))
            restored_again, state = prepare_identity_references(restored, state)
            self.assertIs(restored_again, restored)

    def test_keyframes_and_legacy_anchor_do_not_silently_become_refs(self):
        image = torch.randn(1, 24, 1, 2, 2)
        cond = [[torch.zeros(1), {"minimax_keyframes": [{"resolved_frame_index": 0, "latent": image}]}]]
        result, state = prepare_identity_references(cond, {"minimax_identity_anchor": image})
        self.assertIs(result, cond)
        self.assertEqual(state["minimax_identity_refs"], [])
        self.assertNotIn("minimax_refs", result[0][1])

    def test_explicit_legacy_identity_uses_saved_image_not_generated_samples(self):
        original = torch.randn(1, 24, 1, 2, 2)
        target = nodes.pack_av_latent(torch.zeros(1, 24, 37, 2, 2), torch.zeros(1, 32, 2, 207))
        identity = {"samples": torch.ones(1, 24, 37, 2, 2), "minimax_identity_anchor": original}
        _, cond, _, latent = nodes.MiniMaxPrefixCacheApplierNode().apply_cache(
            object(), [[torch.zeros(1), {}]], target_latent=target, identity_latent=identity)
        self.assertTrue(torch.equal(cond[0][1]["minimax_refs"][0]["latent"], original))
        self.assertTrue(torch.equal(latent["minimax_identity_refs"][0]["latent"], original))

    def test_soft_video_context_is_opt_in_and_leaves_audio_independent(self):
        v = torch.randn(1, 24, 37, 2, 2)
        a = torch.randn(1, 32, 2, 207)
        hard = apply_native_masked_av(torch.zeros_like(v), torch.zeros_like(a), v, a)
        soft = apply_native_masked_av(torch.zeros_like(v), torch.zeros_like(a), v, a,
                                      video_context_denoise=0.06)
        for x, y in zip(hard[:2], soft[:2]):
            self.assertTrue(torch.equal(x, y))
        self.assertTrue(torch.all(hard[2][:, :, :12] == 0))
        self.assertTrue(torch.all(soft[2][:, :, :12] == 0.06))
        self.assertTrue(torch.all(soft[2][:, :, 12:] == 1))
        self.assertTrue(torch.equal(hard[3], soft[3]))

    def test_continuation_model_and_soft_mask_reach_native_node_output(self):
        initial, corrective = object(), object()
        source = nodes.pack_av_latent(torch.randn(1, 24, 37, 2, 2), torch.randn(1, 32, 2, 207))
        target = nodes.pack_av_latent(torch.zeros(1, 24, 37, 2, 2), torch.zeros(1, 32, 2, 207))
        cfg, = nodes.MiniMaxPrefixCacheConfigNode().create_config(video_context_denoise=0.03)
        applier = nodes.MiniMaxPrefixCacheApplierNode()
        model, _, _, _ = applier.apply_cache(initial, [], target_latent=target, continuation_model=corrective)
        self.assertIs(model, initial)
        with mock.patch.object(nodes, "_require_native_masked_av_support"):
            model, _, _, masked = applier.apply_cache(initial, [], cfg, source, target, continuation_model=corrective)
        self.assertIs(model, corrective)
        vm, _ = nodes._unpack_latent({"samples": masked["noise_mask"]})
        self.assertTrue(torch.all(vm[:, :, :12] == 0.03))

    def test_trimmed_context_matches_full_context_without_tail_loss(self):
        v = torch.randn(1, 24, 37, 2, 2)
        a = torch.randn(1, 32, 2, 207)
        source = nodes.pack_av_latent(v, a)
        target = nodes.pack_av_latent(torch.zeros_like(v), torch.zeros_like(a))
        _, _, cropped = nodes.MiniMaxTrimPrefixLatentNode().trim(latent=source, trim_frames=39)
        self.assertEqual(cropped["minimax_video_start_step"], 12)
        self.assertEqual(cropped["minimax_audio_start_tick"], 65)
        with mock.patch.object(nodes, "_require_native_masked_av_support"):
            for frames in (22, 39, 56, 73):
                cfg = KVCacheConfig(rolling_frames=frames, audio_feather_ticks=0)
                _, _, _, full = nodes.MiniMaxPrefixCacheApplierNode().apply_cache(object(), [], cfg, source, target)
                _, _, _, cut = nodes.MiniMaxPrefixCacheApplierNode().apply_cache(object(), [], cfg, cropped, target)
                for x, y in zip(nodes._unpack_latent(full), nodes._unpack_latent(cut)):
                    self.assertTrue(torch.equal(x, y), f"Mismatch at {frames} frames")

    def test_original_identity_survives_drifting_generated_context(self):
        original = torch.randn(1, 24, 1, 2, 2)
        cond = [[torch.zeros(1), {"minimax_refs": [{"kind": "image", "latent": original, "latent_h": 2, "latent_w": 2}]}]]
        target = nodes.pack_av_latent(torch.zeros(1, 24, 37, 2, 2), torch.zeros(1, 32, 2, 207))
        applier = nodes.MiniMaxPrefixCacheApplierNode()
        _, _, _, source = applier.apply_cache(object(), cond, target_latent=target)
        with mock.patch.object(nodes, "_require_native_masked_av_support"):
            for drift in (0.2, 0.4, 0.6):
                # Supply deliberately changed generated latents; the original anchor must not follow them.
                v, a = nodes._unpack_latent(source)
                source = nodes.pack_av_latent(v + drift, a, source)
                _, result_cond, _, source = applier.apply_cache(object(), [[torch.zeros(1), {}]], context_latent=source, target_latent=target)
                refs = result_cond[0][1]["minimax_refs"]
                self.assertEqual(len(refs), 1)
                self.assertTrue(torch.equal(refs[0]["latent"], original))
                self.assertEqual(refs[0]["latent_h"], 2)
                self.assertTrue(torch.equal(source["minimax_identity_anchor"], original))
                self.assertTrue(torch.equal(original, cond[0][1]["minimax_refs"][0]["latent"]))

    def test_repeated_latent_trims_track_actual_boundaries(self):
        v = torch.randn(1, 24, 37, 2, 2)
        a = torch.randn(1, 32, 2, 207)
        trimmer = nodes.MiniMaxTrimPrefixLatentNode()
        _, _, cropped = trimmer.trim(latent=nodes.pack_av_latent(v, a), trim_frames=39)
        _, _, cropped = trimmer.trim(latent=cropped, trim_frames=22)
        # The second cut starts at phase 2: seven more steps span 25, not 22 frames.
        self.assertEqual(cropped["minimax_video_start_step"], 19)
        self.assertEqual(cropped["minimax_audio_start_tick"], 107)
        cv, ca = nodes._unpack_latent(cropped)
        self.assertTrue(torch.equal(cv, v[:, :, 19:]))
        self.assertTrue(torch.equal(ca, a[..., 107:]))

    def test_explicit_identity_and_native_references(self):
        identity = torch.randn(1, 24, 1, 2, 2)
        source = nodes.pack_av_latent(torch.randn(1, 24, 37, 2, 2), torch.randn(1, 32, 2, 207))
        target = nodes.pack_av_latent(torch.zeros(1, 24, 37, 2, 2), torch.zeros(1, 32, 2, 207))
        ref = {"kind": "image", "latent": identity}
        with mock.patch.object(nodes, "_require_native_masked_av_support"):
            _, cond, _, result = nodes.MiniMaxPrefixCacheApplierNode().apply_cache(
                object(), [[torch.zeros(1), {"minimax_refs": [ref]}]], context_latent=source,
                target_latent=target, identity_latent={"samples": identity})
        self.assertEqual(len(cond[0][1]["minimax_refs"]), 1)
        self.assertTrue(torch.equal(result["minimax_identity_anchor"], identity))

    def test_current_image_reference_replaces_saved_origin(self):
        original = torch.randn(1, 24, 1, 2, 2)
        generated = {"kind": "image", "latent": original + 1}
        cond, state = prepare_identity_references([[torch.zeros(1), {"minimax_refs": [generated]}]],
            {"minimax_identity_refs": [{"kind": "image", "latent": original}]})
        self.assertEqual(len(cond[0][1]["minimax_refs"]), 1)
        self.assertTrue(torch.equal(state["minimax_identity_refs"][0]["latent"], generated["latent"]))

    def test_old_full_clip_does_not_invent_an_image_reference(self):
        v = torch.randn(1, 24, 37, 2, 2)
        a = torch.randn(1, 32, 2, 207)
        source = nodes.pack_av_latent(v, a)
        target = nodes.pack_av_latent(torch.zeros_like(v), torch.zeros_like(a))
        with mock.patch.object(nodes, "_require_native_masked_av_support"):
            _, cond, _, masked = nodes.MiniMaxPrefixCacheApplierNode().apply_cache(
                object(), [[torch.zeros(1), {}]], context_latent=source, target_latent=target)
        self.assertNotIn("minimax_refs", cond[0][1])
        self.assertNotIn("minimax_identity_anchor", masked)

    def test_matches_official_22_frame_cut_concat_masks(self):
        source_v = torch.randn(1, 24, 37, 2, 2)
        source_a = torch.randn(1, 32, 2, 207)
        target_v, target_a = torch.randn_like(source_v), torch.randn_like(source_a)
        v, a, vm, am, plan = apply_native_masked_av(
            target_v, target_a, source_v, source_a, context_frames=22)
        self.assertEqual(plan["actual_context_frames"], 22)
        self.assertTrue(torch.equal(v, torch.cat((source_v[:, :, -7:], target_v[:, :, 7:]), dim=2)))
        self.assertTrue(torch.equal(a, torch.cat((source_a[..., -37:], target_a[..., 37:]), dim=-1)))
        self.assertTrue(torch.all(vm[:, :, :7] == 0))
        self.assertTrue(torch.all(vm[:, :, 7:] == 1))
        self.assertTrue(torch.all(am[..., :37] == 0))
        self.assertTrue(torch.all(am[..., 37:] == 1))

    def test_short_target_stays_masked_and_rounds_audio_duration_once(self):
        source_v = torch.randn(1, 24, 37, 2, 2)
        source_a = torch.randn(1, 32, 2, 207)
        source = nodes.pack_av_latent(source_v, source_a)
        target = nodes.pack_av_latent(torch.zeros(1, 24, 7, 2, 2), torch.zeros(1, 32, 2, 37))
        with mock.patch.object(nodes, "_require_native_masked_av_support"):
            _, _, session, masked = nodes.MiniMaxPrefixCacheApplierNode().apply_cache(
                object(), [[torch.zeros(1), {}]], KVCacheConfig(), source, target)
        self.assertEqual(session.last_rolling_frames, 5)
        self.assertIn("noise_mask", masked)
        v, a = nodes._unpack_latent(masked)
        self.assertTrue(torch.equal(v[:, :, :2], source_v[:, :, -2:]))
        self.assertTrue(torch.equal(a[..., :8], source_a[..., -8:]))
        self.assertTrue(torch.all(a[..., 8:] == 0))

    def test_repeated_continuations_keep_references_and_source_unchanged(self):
        refs = [{"kind": "image", "latent": torch.randn(1, 24, 1, 2, 2)}]
        cond = [[torch.zeros(1), {"minimax_refs": refs}]]
        source = nodes.pack_av_latent(torch.randn(1, 24, 37, 2, 2), torch.randn(1, 32, 2, 207))
        cfg = KVCacheConfig(rolling_frames=22, audio_feather_ticks=0)
        with mock.patch.object(nodes, "_require_native_masked_av_support"):
            for _ in range(4):
                sv, sa = nodes._unpack_latent(source)
                before_v, before_a = sv.clone(), sa.clone()
                target = nodes.pack_av_latent(torch.randn_like(sv), torch.randn_like(sa))
                _, out_cond, _, masked = nodes.MiniMaxPrefixCacheApplierNode().apply_cache(
                    object(), cond, cfg, source, target)
                v, a = nodes._unpack_latent(masked)
                self.assertTrue(torch.equal(v[:, :, :7], sv[:, :, -7:]))
                self.assertTrue(torch.equal(a[..., :37], sa[..., -37:]))
                self.assertTrue(torch.equal(sv, before_v))
                self.assertTrue(torch.equal(sa, before_a))
                self.assertIs(out_cond[0][1]["minimax_refs"], refs)
                source = masked

    def test_safe_native_phase_alignment_and_commit_use_actual_prefix(self):
        session = LongVideoSession(KVCacheConfig(cache_mode="Safe Native", rolling_frames=39))
        video = torch.randn(1, 24, 35, 2, 2)
        _, cond = session.prepare_next_clip(object(), [[torch.zeros(1), {}]], video)
        # start=23 must advance to 25, not 26; ten steps remain.
        self.assertEqual(session.last_rolling_steps, 10)
        self.assertTrue(torch.equal(cond[0][1]["minimax_keyframes"][0]["latent"], video[:, :, 25:26]))
        delivered, _ = session.commit_generated_clip(video)
        self.assertEqual(delivered.shape[2], 25)
        session.is_current_clip_trimmed = True
        session.prepare_next_clip(object(), [[torch.zeros(1), {}]])
        self.assertEqual(session.last_rolling_frames, 0)
        self.assertFalse(session.is_current_clip_trimmed)
        delivered, _ = session.commit_generated_clip(video)
        self.assertEqual(delivered.shape[2], 35)

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
        self.assertIs(out_latent["samples"], target["samples"])
        self.assertEqual(session.last_rolling_frames, 5)


if __name__ == "__main__":
    unittest.main()
