"""Run with ComfyUI's Python: native mask preparation, H3 injection and Euler sampling.

Uses a deterministic test denoiser, not model weights or a visual-quality metric.
"""

import sys
import importlib.util
from pathlib import Path
from types import SimpleNamespace

repo = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(repo.parents[1]), str(repo)]
sys.argv = [sys.argv[0], "--cpu"]

import torch
import comfy.utils
import comfy.sampler_helpers
from comfy.samplers import KSamplerX0Inpaint
from comfy.k_diffusion.sampling import sample_euler, sample_res_multistep
from comfy.model_base import MiniMaxH3
from comfy.model_sampling import CONST
from comfy.ldm.minimax.model import PackedLayout, MiniMaxH3Model
from comfy.ldm.minimax.model import time_shift_sigma
from comfy_extras.nodes_images import ImageAddNoise
from pipeline.context_noise import noise_reference_frames
plugin_spec = importlib.util.spec_from_file_location("prefixstream_test_nodes", repo / "nodes.py")
nodes = importlib.util.module_from_spec(plugin_spec)
plugin_spec.loader.exec_module(nodes)


def check_scaled_audio_sampling():
    """Exercise the 7-second geometry and the actual 12/3 AV schedule."""
    torch.manual_seed(42)
    video = torch.randn(1, 24, 52, 2, 2)
    audio = torch.randn(1, 32, 2, 292)
    cfg, = nodes.MiniMaxPrefixCacheConfigNode().create_config(continuation_frames="22")
    _, _, _, masked = nodes.MiniMaxPrefixCacheApplierNode().apply_cache(
        object(), [], cache_config=cfg, context_latent=nodes.pack_av_latent(video, audio),
        target_latent=nodes.pack_av_latent(torch.zeros_like(video), torch.zeros_like(audio)))
    streams = nodes._unpack_latent(masked)
    scaled_streams = [streams[0], streams[1] * 4.0]
    clean, shapes = comfy.utils.pack_latents(scaled_streams)
    masks = [comfy.sampler_helpers.prepare_mask(m, s.shape, "cpu")
             for m, s in zip(masked["noise_mask"].unbind(), streams)]
    mask, _ = comfy.utils.pack_latents(masks)
    h3 = SimpleNamespace(latent_shapes=shapes, diffusion_model=SimpleNamespace(patch_size=(1, 2, 2)),
                         audio_scale=lambda: 4.0,
                         model_sampling=SimpleNamespace(shift=12.0, audio_shift=3.0))
    h3._pool_masks_to_token_grid = lambda ms: MiniMaxH3._pool_masks_to_token_grid(h3, ms)
    h3._token_grid_masks = lambda m, ss: MiniMaxH3._token_grid_masks(h3, m, ss)
    h3.scale_latent_inpaint = lambda **kw: MiniMaxH3.scale_latent_inpaint(h3, **kw)

    class TestDenoiser:
        inner_model = h3
        model_patcher = SimpleNamespace(get_model_object=lambda _: SimpleNamespace(noise_scale=1.0))

        def __call__(self, x, sigma, **kwargs):
            vm, am = h3._token_grid_masks(mask, shapes)

            def forward(inputs, *args, **kw):
                # Check what the network sees, not just the final pinned output.
                torch.testing.assert_close(inputs[1][..., :37], streams[1][..., :37], atol=1e-6, rtol=0)
                return [torch.ones_like(t) for t in inputs]

            dit = SimpleNamespace(_forward=forward, sigma_shift_video=12.0, sigma_shift_audio=3.0)
            velocity = MiniMaxH3Model.forward(
                dit, comfy.utils.unpack_latents(x, shapes), sigma * 1000, torch.zeros(1),
                minimax_payload={"audio_scale": 4.0}, denoise_mask=vm, audio_denoise_mask=am)
            return CONST.calculate_denoised(None, sigma, comfy.utils.pack_latents(velocity)[0], x)

    sigmas = time_shift_sigma(torch.linspace(1, 0, 21), 1.0, 12.0)
    for sample in (sample_euler, sample_res_multistep):
        sampler = KSamplerX0Inpaint(TestDenoiser(), sigmas)
        sampler.latent_image = clean
        sampler.noise = torch.randn_like(clean)
        result = sample(sampler, sampler.noise.clone(), sigmas,
                        extra_args={"denoise_mask": mask}, disable=True)
        torch.testing.assert_close(result[mask == 0], clean[mask == 0], atol=1e-6, rtol=0)
        assert torch.isfinite(result).all()
    print("PASS: 175-frame/292-tick continuation, scaled audio prefix seen clean by DiT, Euler and res_multistep")

    cfg.audio_continuation_mode = "Native Audio Guide"
    _, cond, _, guided = nodes.MiniMaxPrefixCacheApplierNode().apply_cache(
        object(), [[torch.zeros(1), {}]], cache_config=cfg,
        context_latent=nodes.pack_av_latent(video, audio),
        target_latent=nodes.pack_av_latent(torch.zeros_like(video), torch.zeros_like(audio)))
    guides = cond[0][1]["minimax_keyframes"]
    payload = MiniMaxH3._cond_latents(guides, [])
    torch.testing.assert_close(payload["cond_audio_latents"][0], audio[..., -37:])
    layout = PackedLayout(2, 52, 2, 2, 292, keyframes=guides)
    assert sum(b - a for a, b, kind in layout.segments if kind == "cond_audio") == 74
    assert sum(b - a for a, b, kind in layout.segments if kind == "audio") == 584
    gv, ga = nodes._unpack_latent({"samples": guided["noise_mask"]})
    assert (gv[:, :, :7] == 0).all() and (ga == 1).all()
    print("PASS: native guide payload/layout keeps 37 stereo condition ticks separate from 292 generated ticks")


def main():
    nodes._require_native_masked_av_support()
    pixels = torch.full((22, 4, 4, 3), 0.5)
    noised = noise_reference_frames(pixels)
    assert torch.equal(noised[:1], ImageAddNoise.execute(pixels[:1], 0, 0.4)[0])
    assert torch.equal(noised[-1:], pixels[-1:])
    torch.manual_seed(42)
    v, a = torch.randn(1, 24, 37, 2, 2), torch.randn(1, 32, 2, 207)
    original = v[:, :, :1].clone()
    source = nodes.pack_av_latent(v, a)
    source["minimax_identity_refs"] = [{"kind": "image", "latent": original, "latent_h": 2, "latent_w": 2}]
    target = nodes.pack_av_latent(torch.zeros_like(v), torch.zeros_like(a))
    _, cond, _, masked = nodes.MiniMaxPrefixCacheApplierNode().apply_cache(
        object(), [[torch.zeros(1), {}]], context_latent=source, target_latent=target)
    streams = nodes._unpack_latent(masked)
    clean, shapes = comfy.utils.pack_latents(streams)
    masks = [comfy.sampler_helpers.prepare_mask(m, s.shape, "cpu")
             for m, s in zip(masked["noise_mask"].unbind(), streams)]
    mask, _ = comfy.utils.pack_latents(masks)

    h3 = SimpleNamespace(latent_shapes=shapes, diffusion_model=SimpleNamespace(patch_size=(1, 2, 2)),
                         audio_scale=lambda: 1.0)
    h3._pool_masks_to_token_grid = lambda ms: MiniMaxH3._pool_masks_to_token_grid(h3, ms)
    h3._token_grid_masks = lambda m, ss: MiniMaxH3._token_grid_masks(h3, m, ss)
    h3.scale_latent_inpaint = lambda **kw: MiniMaxH3.scale_latent_inpaint(h3, **kw)

    class TestDenoiser:
        inner_model = h3
        denoise_mask = mask

        def __call__(self, x, sigma, **kwargs):
            video_mask, audio_mask = h3._token_grid_masks(self.denoise_mask, shapes)
            dit = SimpleNamespace(_forward=lambda inputs, *args, **kw: [torch.ones_like(t) for t in inputs])
            velocity = MiniMaxH3Model.forward(dit, comfy.utils.unpack_latents(x, shapes), sigma * 1000,
                torch.zeros(1), denoise_mask=video_mask, audio_denoise_mask=audio_mask)
            packed_velocity, _ = comfy.utils.pack_latents(velocity)
            return CONST.calculate_denoised(None, sigma, packed_velocity, x)

    denoiser = TestDenoiser()
    sampler = KSamplerX0Inpaint(denoiser, torch.linspace(1, 0, 5))
    sampler.latent_image = clean
    sampler.noise = torch.randn_like(clean)
    result = sample_euler(sampler, sampler.noise.clone(), torch.linspace(1, 0, 5),
                          extra_args={"denoise_mask": mask}, disable=True)
    torch.testing.assert_close(result[mask == 0], clean[mask == 0], atol=1e-6, rtol=0)
    refs = cond[0][1]["minimax_refs"]
    payload = MiniMaxH3._cond_latents([], refs)
    assert torch.equal(payload["cond_video_latents"][0], original)
    layout = PackedLayout(2, 37, 2, 2, 207, refs=refs)
    assert layout is not None
    cfg, = nodes.MiniMaxPrefixCacheConfigNode().create_config(video_context_denoise=0.06)
    _, _, _, soft = nodes.MiniMaxPrefixCacheApplierNode().apply_cache(
        object(), [[torch.zeros(1), {}]], cache_config=cfg, context_latent=source, target_latent=target)
    soft_masks = [comfy.sampler_helpers.prepare_mask(m, s.shape, "cpu")
                  for m, s in zip(soft["noise_mask"].unbind(), streams)]
    soft_mask, _ = comfy.utils.pack_latents(soft_masks)
    native_video_mask = h3._token_grid_masks(soft_mask, shapes)[0]
    torch.testing.assert_close(native_video_mask[:, :, :12], torch.full_like(native_video_mask[:, :, :12], 16 / 256))
    denoiser.denoise_mask = soft_mask
    softened = sample_euler(sampler, sampler.noise.clone(), torch.linspace(1, 0, 5),
                            extra_args={"denoise_mask": soft_mask}, disable=True)
    assert (softened[mask == 0] - result[mask == 0]).abs().max() > 1e-4
    print("PASS: native noise amplitude and clean tail, hard/soft H3 masks, four Euler steps, protected prefix and reference layout")
    check_scaled_audio_sampling()


if __name__ == "__main__":
    main()
