"""Compare identical static skin/background regions in exported PNG frames.

Example: python tools/analyze_texture_drift.py clip1.png clip5.png --roi 20 20 180 180
Does not modify images, denoise them or assign a quality score.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.texture_diagnostics import texture_band_rms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frames", nargs="+", type=Path)
    parser.add_argument("--roi", nargs=4, type=int, metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"))
    args = parser.parse_args()
    rows = []
    for path in args.frames:
        with Image.open(path) as opened:
            image = opened.convert("RGB")
            if args.roi:
                left, top, right, bottom = args.roi
                if not (0 <= left < right <= image.width and 0 <= top < bottom <= image.height):
                    parser.error(f"ROI lies outside {path}")
                image = image.crop(tuple(args.roi))
            pixels = torch.from_numpy(np.asarray(image).copy()).to(torch.float32) / 255
        rows.append({"frame": str(path), **texture_band_rms(pixels)})
    print(json.dumps({"roi": args.roi, "measurements": rows,
                      "interpretation": "Texture energy only. Compare matching static ROIs under comparable lighting; blur can lower all bands without improving quality."}, indent=2))


if __name__ == "__main__":
    main()
