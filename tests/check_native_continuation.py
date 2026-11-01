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
from comfy.k_diffusion.sampling import sample_euler
from comfy.model_base import MiniMaxH3
from comfy.model_sampling import CONST
from comfy.ldm.minimax.model import PackedLayout, MiniMaxH3Model
from comfy_extras.nodes_images import ImageAddNoise
from pipeline.context_noise import noise_reference_frames
plugin_spec = importlib.util.spec_from_file_location("prefixstream_test_nodes", repo / "nodes.py")
nodes = importlib.util.module_from_spec(plugin_spec)
plugin_spec.loader.exec_module(nodes)


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


if __name__ == "__main__":
    main()
