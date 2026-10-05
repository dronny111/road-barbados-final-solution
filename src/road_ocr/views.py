"""Deterministic preprocessing views of a line crop, for test-time multi-view decoding.

Each view is applied to the source-resolution crop, before any resize, and returns RGB so
every recognizer takes it unchanged. The parameters are fixed in advance and scale with
the crop height; none is tuned on held-out rows.

- ``flatten``: divide by a heavily blurred copy (the paper and illumination), then a 1%
  autocontrast. Removes stains and uneven scan light and keeps stroke gradients.
- ``binary``: Sauvola thresholding of the flattened view (window ~H/4, k=0.2, R=128).
"""
from __future__ import annotations

VIEWS = ("original", "flatten", "binary")


def _flatten_gray(image):
    import numpy as np
    from PIL import ImageFilter, ImageOps

    gray = image.convert("L")
    background = gray.filter(ImageFilter.GaussianBlur(radius=max(8, gray.height // 2)))
    ratio = np.asarray(gray, dtype=np.float32) / np.maximum(np.asarray(background, dtype=np.float32), 1.0)
    flat = np.clip(ratio * 255.0, 0, 255).astype(np.uint8)
    from PIL import Image

    return ImageOps.autocontrast(Image.fromarray(flat), cutoff=1)


def _sauvola(gray, k: float = 0.2, r: float = 128.0):
    import numpy as np
    from PIL import Image

    pixels = np.asarray(gray, dtype=np.float64)
    window = max(15, (gray.height // 4) | 1)
    pad = window // 2
    padded = np.pad(pixels, pad, mode="reflect")
    # Integral images give every window's mean and variance in O(1).
    s1 = np.pad(padded.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    s2 = np.pad((padded ** 2).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    h, w = pixels.shape
    box = lambda s: (s[window:window + h, window:window + w] - s[:h, window:window + w]
                     - s[window:window + h, :w] + s[:h, :w])
    area = window * window
    mean = box(s1) / area
    std = np.sqrt(np.maximum(box(s2) / area - mean ** 2, 0.0))
    threshold = mean * (1 + k * (std / r - 1))
    return Image.fromarray(np.where(pixels > threshold, 255, 0).astype(np.uint8))


def apply_view(image, view: str):
    if view not in VIEWS:
        raise ValueError(f"unknown view {view!r}; expected one of {VIEWS}")
    if view == "original":
        return image
    flat = _flatten_gray(image)
    return (flat if view == "flatten" else _sauvola(flat)).convert("RGB")
