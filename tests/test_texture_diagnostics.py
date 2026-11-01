import unittest

import torch

from engine.texture_diagnostics import texture_band_rms


class TextureDiagnosticsTests(unittest.TestCase):
    def test_removes_flat_color_and_linear_lighting(self):
        x = torch.linspace(0, 0.1, 128)
        y = torch.linspace(0, 0.2, 128)[:, None]
        rgb = (0.3 + x + y)[..., None].expand(-1, -1, 3)
        report = texture_band_rms(rgb)
        for channel in report["band_rms"].values():
            self.assertLess(max(channel.values()), 1e-6)

    def test_detects_mid_band_pattern_and_scales_with_amplitude(self):
        pattern = torch.sin(torch.arange(128) * (2 * torch.pi / 16))
        def measure(amplitude):
            image = (0.5 + amplitude * pattern)[None, :, None].expand(128, -1, 3)
            return texture_band_rms(image)["band_rms"]["luma"]
        small, large = measure(0.02), measure(0.04)
        self.assertGreater(small["8_32px"], small["2_8px"] * 10)
        self.assertAlmostEqual(large["8_32px"] / small["8_32px"], 2, places=4)

    def test_measures_chroma_pattern_separately(self):
        image = torch.full((128, 128, 3), 0.5)
        pattern = 0.03 * torch.sin(torch.arange(128) * (2 * torch.pi / 16))
        image[..., 0] += pattern
        image[..., 1] -= pattern * (0.2126 / 0.7152)
        report = texture_band_rms(image)["band_rms"]
        self.assertLess(report["luma"]["8_32px"], 1e-6)
        self.assertGreater(report["red_minus_green"]["8_32px"], 0.01)

    def test_non_finite_image_is_explicit(self):
        with self.assertRaisesRegex(ValueError, "non-finite"):
            texture_band_rms(torch.full((16, 16, 3), float("nan")))
