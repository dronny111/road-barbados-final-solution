"""Deterministic image cleaning for training and inference: images/ID.jpg -> images_clean/ID.png.

Keeps RGB and gradients (no binarization, no crop, no resize). Order: illumination division
(ink removed by closing before blurring, so strokes stay dark) -> mild deskew -> CLAHE on LAB L ->
light NLM denoise -> light unsharp. Run with .venv-vlm (cv2).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def flatten(bgr, k_frac=1.0):
    """Divide by the paper estimate. Closing runs on a downscaled copy (background is smooth; full-res ellipse was ~12 s/img)."""
    h, w = bgr.shape[:2]
    s = max(1, h // 32)
    small = cv2.resize(bgr, (max(1, w // s), max(1, h // s)), interpolation=cv2.INTER_AREA)
    k = max(5, int(small.shape[0] * k_frac)) | 1
    bg = cv2.morphologyEx(small, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = cv2.resize(cv2.GaussianBlur(bg, (0, 0), max(2, k / 3)), (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(bgr.astype(np.float32) / np.maximum(bg, 1) * 245, 0, 255).astype(np.uint8)


def deskew(bgr, max_deg=1.0, step=0.5):
    """Rotate by the tilt (|angle| <= max_deg) maximizing row-profile sharpness of the ink."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    ink = (255 - gray).astype(np.float32)
    h, w = ink.shape
    best, best_a = -1.0, 0.0
    for a in np.arange(-max_deg, max_deg + step / 2, step):
        m = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0)
        score = cv2.warpAffine(ink, m, (w, h)).sum(1).var()
        if score > best * (1 + 1e-6):
            best, best_a = score, a
    if abs(best_a) < step:
        return bgr
    m = cv2.getRotationMatrix2D((w / 2, h / 2), best_a, 1.0)
    return cv2.warpAffine(bgr, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(245, 245, 245))


def clean(bgr, clip=1.5, h_nlm=3, amount=0.4):
    bgr = deskew(flatten(bgr))
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(1, 8)).apply(lab[..., 0])
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    bgr = cv2.fastNlMeansDenoisingColored(bgr, None, h_nlm, h_nlm, 5, 15)
    blur = cv2.GaussianBlur(bgr, (0, 0), 1.0)
    return cv2.addWeighted(bgr, 1 + amount, blur, -amount, 0)


def main():
    cv2.setNumThreads(1)  # determinism
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="images")
    ap.add_argument("--out", default="images_clean")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--reencode-jpeg", metavar="DIR",
                    help="after cleaning, also write DIR/ID.jpg (quality 100, 4:4:4): the Kaggle notebooks "
                         "audit JPEG files only, and this is the form the cleaned runs trained on")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    files = sorted(Path(a.images).glob("*.jpg"))[: a.limit or None]
    for i, f in enumerate(files):
        cv2.imwrite(str(out / f"{f.stem}.png"), clean(cv2.imread(str(f), cv2.IMREAD_COLOR)))
        if i % 500 == 0:
            print(i, len(files), flush=True)
    if a.reencode_jpeg:
        from PIL import Image

        jpeg = Path(a.reencode_jpeg)
        jpeg.mkdir(parents=True, exist_ok=True)
        for f in files:
            with Image.open(out / f"{f.stem}.png") as im:
                im.convert("RGB").save(jpeg / f"{f.stem}.jpg", quality=100, subsampling=0, optimize=False)


if __name__ == "__main__":
    main()
