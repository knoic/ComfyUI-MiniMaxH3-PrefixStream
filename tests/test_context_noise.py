import unittest
from unittest import mock

import torch

import nodes
from engine.cache_manager import KVCacheConfig, pixel_frames_to_latent_steps
from pipeline.context_noise import noise_reference_frames, reference_noise_strengths
from pipeline.native_masked_av import latent_boundaries


class RecordingVideoVAE:
    def __init__(self):
        self.encode_inputs = []
        self.encode_outputs = []

    def decode(self, latent):
        self.decoded_source = latent
        frames = latent_boundaries(latent.shape[2])[-1]
        return torch.full((frames, 32, 32, 3), 0.5)

    def encode(self, pixels):
        self.encode_inputs.append(pixels.clone())
        steps = pixel_frames_to_latent_steps(pixels.shape[0])
        latent = torch.full((1, 24, steps, 2, 2), float(pixels.mean()) + 0.125)
        self.encode_outputs.append(latent.clone())
        return latent


class ContextNoiseTests(unittest.TestCase):
    def setUp(self):
        self.video = torch.randn(1, 24, 67, 2, 2)  # 226 frames, accommodates every context choice.
        self.audio = torch.randn(1, 32, 2, 377)
        self.source = nodes.pack_av_latent(self.video, self.audio)
        self.target = nodes.pack_av_latent(torch.zeros_like(self.video), torch.zeros_like(self.audio))
        self.cond = [[torch.zeros(1), {}]]

    def test_recipe_reproducible_without_mutating_images_or_global_rng(self):
        images = torch.full((22, 4, 4, 4), 0.5)
        images[..., 3] = 0.7
        before = images.clone()
        rng = torch.random.get_rng_state()
        actual = noise_reference_frames(images)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertTrue(torch.equal(images, before))
        self.assertTrue(torch.equal(actual[-1:], before[-1:]))
        self.assertTrue(torch.equal(actual[..., 3], before[..., 3]))
        self.assertFalse(torch.equal(actual[0], before[0]))
        self.assertTrue(torch.equal(actual, noise_reference_frames(images)))

    def test_every_context_has_monotone_noise_and_clean_tail(self):
        for count in range(5, 193, 17):
            profile = reference_noise_strengths(count)
            self.assertEqual(len(profile), count)
            self.assertEqual(profile[-1], 0.0)
            self.assertGreater(profile[-2], 0.0)
            self.assertTrue(all(a >= b >= 0 for a, b in zip(profile, profile[1:])))
            self.assertEqual(profile[0], 0.4)

    def test_five_frame_context_noises_first_four_and_preserves_final_pixel_frame(self):
        image = torch.full((5, 16, 16, 3), 0.5)
        actual = noise_reference_frames(image)
        self.assertTrue(torch.equal(actual[-1], image[-1]))
        self.assertTrue(all(not torch.equal(actual[i], image[i]) for i in range(4)))
        expected = [0.4, 0.3414213562, 0.2, 0.0585786438, 0.0]
        for strength, value in zip(reference_noise_strengths(5), expected):
            self.assertAlmostEqual(strength, value)

    def test_all_context_choices_keep_masks_and_audio_unchanged(self):
        choices = nodes.MiniMaxPrefixCacheConfigNode.INPUT_TYPES()["required"]["continuation_frames"][0]
        original_video, original_audio = self.video.clone(), self.audio.clone()
        with mock.patch.object(nodes, "_require_native_masked_av_support"):
            for choice in choices:
                with self.subTest(frames=choice):
                    base_cfg = KVCacheConfig(rolling_frames=int(choice))
                    noisy_cfg, = nodes.MiniMaxPrefixCacheConfigNode().create_config(
                        continuation_frames=choice, reference_noise=True)
                    _, _, _, baseline = nodes.MiniMaxPrefixCacheApplierNode().apply_cache(
                        object(), self.cond, base_cfg, self.source, self.target)
                    vae = RecordingVideoVAE()
                    _, _, session, result = nodes.MiniMaxPrefixCacheApplierNode().apply_cache(
                        object(), self.cond, noisy_cfg, self.source, self.target, vae=vae)
                    v, a = nodes._unpack_latent(result)
                    steps = pixel_frames_to_latent_steps(int(choice))
                    self.assertEqual(session.last_rolling_frames, int(choice))
                    self.assertEqual(len(vae.encode_inputs), 2)
                    self.assertEqual(vae.encode_inputs[1].shape[0], int(choice))
                    self.assertIs(vae.decoded_source, self.video)
                    self.assertTrue(torch.equal(vae.encode_inputs[0][-1], vae.encode_inputs[1][-1]))
                    self.assertFalse(torch.equal(vae.encode_inputs[0][0], vae.encode_inputs[1][0]))
                    expected = self.video[:, :, -steps:] + (vae.encode_outputs[1] - vae.encode_outputs[0])
                    self.assertTrue(torch.equal(v[:, :, :steps], expected))
                    self.assertTrue(torch.all(v[:, :, steps:] == 0))
                    self.assertTrue(torch.equal(a, nodes._unpack_latent(baseline)[1]))
                    for expected, actual in zip(nodes._unpack_latent({"samples": baseline["noise_mask"]}),
                                                nodes._unpack_latent({"samples": result["noise_mask"]})):
                        self.assertTrue(torch.equal(expected, actual))
        self.assertTrue(torch.equal(self.video, original_video))
        self.assertTrue(torch.equal(self.audio, original_audio))

    def test_zero_noise_cancels_codec_bias_instead_of_replacing_original(self):
        vae = RecordingVideoVAE()
        cfg = KVCacheConfig(reference_noise=True, rolling_frames=22)
        with mock.patch.object(nodes, "_require_native_masked_av_support"), mock.patch(
            "pipeline.context_noise.noise_reference_frames", side_effect=lambda pixels: pixels.clone()):
            _, _, _, result = nodes.MiniMaxPrefixCacheApplierNode().apply_cache(
                object(), self.cond, cfg, self.source, self.target, vae=vae)
        v, _ = nodes._unpack_latent(result)
        self.assertTrue(torch.all(vae.encode_outputs[0] == 0.625))
        self.assertTrue(torch.equal(v[:, :, :7], self.video[:, :, -7:]))

    def test_22_frame_noise_is_not_overwritten_by_old_tail_tokens(self):
        vae = RecordingVideoVAE()
        with mock.patch.object(nodes, "_require_native_masked_av_support"):
            _, _, _, result = nodes.MiniMaxPrefixCacheApplierNode().apply_cache(
                object(), self.cond, KVCacheConfig(reference_noise=True, rolling_frames=22), self.source, self.target, vae=vae)
        v, _ = nodes._unpack_latent(result)
        self.assertFalse(torch.equal(v[:, :, 5:7], self.video[:, :, -2:]))

    def test_noise_hard_protects_audio_even_with_saved_feather_setting(self):
        cfg = KVCacheConfig(reference_noise=True, audio_feather_ticks=2)
        with mock.patch.object(nodes, "_require_native_masked_av_support"):
            _, _, _, result = nodes.MiniMaxPrefixCacheApplierNode().apply_cache(
                object(), self.cond, cfg, self.source, self.target, vae=RecordingVideoVAE())
        _, mask = nodes._unpack_latent({"samples": result["noise_mask"]})
        self.assertTrue(torch.all(mask[..., :65] == 0))
        self.assertTrue(torch.all(mask[..., 65:] == 1))

    def test_requested_context_snaps_to_actual_target_duration(self):
        target = nodes.pack_av_latent(torch.zeros(1, 24, 37, 2, 2), torch.zeros(1, 32, 2, 207))
        vae = RecordingVideoVAE()
        with mock.patch.object(nodes, "_require_native_masked_av_support"):
            _, _, session, result = nodes.MiniMaxPrefixCacheApplierNode().apply_cache(
                object(), self.cond, KVCacheConfig(rolling_frames=192, reference_noise=True), self.source, target, vae=vae)
        self.assertEqual(session.last_rolling_frames, 107)
        self.assertEqual(vae.encode_inputs[1].shape[0], 107)
        self.assertEqual(result["minimax_prefix_frames"], 107)

    def test_initial_clip_needs_no_vae_and_off_switch_never_calls_it(self):
        applier = nodes.MiniMaxPrefixCacheApplierNode()
        _, _, _, first = applier.apply_cache(object(), self.cond,
            KVCacheConfig(reference_noise=True), target_latent=self.target)
        self.assertIs(first["samples"], self.target["samples"])
        vae = mock.Mock()
        with mock.patch.object(nodes, "_require_native_masked_av_support"):
            applier.apply_cache(object(), self.cond, KVCacheConfig(), self.source, self.target, vae=vae)
        vae.decode.assert_not_called()
        vae.encode.assert_not_called()

    def test_original_image_references_are_not_noised(self):
        original = torch.randn(1, 24, 1, 2, 2)
        cond = [[torch.zeros(1), {"minimax_refs": [
            {"kind": "image", "latent": original, "latent_h": 2, "latent_w": 2}]}]]
        before = original.clone()
        with mock.patch.object(nodes, "_require_native_masked_av_support"):
            _, out_cond, _, result = nodes.MiniMaxPrefixCacheApplierNode().apply_cache(
                object(), cond, KVCacheConfig(reference_noise=True), self.source, self.target, vae=RecordingVideoVAE())
        self.assertIs(out_cond[0][1]["minimax_refs"][0]["latent"], original)
        self.assertTrue(torch.equal(original, before))
        self.assertTrue(torch.equal(result["minimax_identity_refs"][0]["latent"], before))

    def test_missing_vae_and_phase_shifted_sources_are_explicit(self):
        cfg = KVCacheConfig(reference_noise=True)
        with self.assertRaisesRegex(ValueError, "VAE"):
            nodes.MiniMaxPrefixCacheApplierNode().apply_cache(object(), self.cond, cfg, self.source, self.target)
        cropped = dict(self.source, minimax_video_start_step=12)
        with self.assertRaisesRegex(ValueError, "完整 latent"):
            nodes.MiniMaxPrefixCacheApplierNode().apply_cache(object(), self.cond, cfg, cropped, self.target, vae=RecordingVideoVAE())
