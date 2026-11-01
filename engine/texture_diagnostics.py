"""Spatial texture measurements for matching, static image regions; not quality scores."""

import torch


def texture_band_rms(image):
    """Measure luminance/chroma variation after removing a linear lighting gradient.

    Wavelengths are in input pixels. Results require comparable framing, lighting
    and compression; skin detail, fabric and blemishes can occupy the same bands.
    """
    if image.ndim != 3 or image.shape[-1] < 3 or min(image.shape[:2]) < 4:
        raise ValueError("Expected an RGB image region with height and width >= 4")
    rgb = image[..., :3].detach().to(device="cpu", dtype=torch.float64)
    h, w = rgb.shape[:2]
    if not bool(torch.isfinite(rgb).all()):
        raise ValueError("Image region contains non-finite values")
    planes = torch.stack((rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722,
                          rgb[..., 0] - rgb[..., 1], rgb[..., 2] - rgb[..., 1]))
    x = torch.arange(w, dtype=rgb.dtype) - (w - 1) / 2
    y = torch.arange(h, dtype=rgb.dtype) - (h - 1) / 2
    trend = planes.mean(dim=(-2, -1), keepdim=True)
    trend = trend + (planes * x).mean(dim=(-2, -1), keepdim=True) / x.square().mean() * x
    trend = trend + (planes * y[:, None]).mean(dim=(-2, -1), keepdim=True) / y.square().mean() * y[:, None]
    window = torch.hann_window(h, periodic=False, dtype=rgb.dtype)[:, None] * torch.hann_window(w, periodic=False, dtype=rgb.dtype)
    spectrum = torch.fft.fft2((planes - trend) * window).abs().square()
    fy, fx = torch.fft.fftfreq(h, dtype=rgb.dtype), torch.fft.fftfreq(w, dtype=rgb.dtype)
    radius = (fy[:, None].square() + fx.square()).sqrt()
    normalizer = (h * w) ** 2 * window.square().mean()
    measurements = {}
    for name, power in zip(("luma", "red_minus_green", "blue_minus_green"), spectrum):
        measurements[name] = {}
        for lo, hi in ((2, 8), (8, 32), (32, 128)):
            band = (radius >= 1 / hi) & (radius < 1 / lo)
            measurements[name][f"{lo}_{hi}px"] = float((power[band].sum() / normalizer).sqrt())
    return {"height": h, "width": w, "band_rms": measurements}
