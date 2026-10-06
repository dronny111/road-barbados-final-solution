"""Pad line images onto a small set of XLA-friendly width buckets."""

from __future__ import annotations


DEFAULT_PATCH_BUCKETS = (16, 32, 64, 128, 256, 512, 1024)


def pad_to_bucket(image, *, patch: int = 28, max_pixels: int | None = None,
                  patch_buckets=DEFAULT_PATCH_BUCKETS):
    """Center ``image`` on the next patch-width bucket without resizing it.

    When the next configured bucket would exceed ``max_pixels``, the current
    patch-aligned width is retained.  This preserves the caller's memory bound
    while still collapsing the common shapes to a handful of widths.
    """

    from PIL import Image, ImageStat

    if patch <= 0:
        raise ValueError("patch must be positive")
    if image.width % patch or image.height % patch:
        raise ValueError("image dimensions must be patch-aligned before bucketing")
    buckets = tuple(int(value) for value in patch_buckets)
    if not buckets or any(value <= 0 for value in buckets) or tuple(sorted(set(buckets))) != buckets:
        raise ValueError("patch_buckets must be unique positive values in ascending order")
    if max_pixels is not None and max_pixels <= 0:
        raise ValueError("max_pixels must be positive")

    width_patches = image.width // patch
    bucket_patches = next((value for value in buckets if value >= width_patches), width_patches)
    bucket_width = bucket_patches * patch
    if max_pixels is not None and bucket_width * image.height > max_pixels:
        bucket_width = image.width
    if bucket_width == image.width:
        return image

    rgb = image.convert("RGB")
    # Median border tone is a robust paper colour and avoids introducing a
    # bright-white seam into stained scans.
    border = Image.new("RGB", (rgb.width, 2))
    border.paste(rgb.crop((0, 0, rgb.width, 1)), (0, 0))
    border.paste(rgb.crop((0, rgb.height - 1, rgb.width, rgb.height)), (0, 1))
    fill = tuple(int(round(value)) for value in ImageStat.Stat(border).median)
    canvas = Image.new("RGB", (bucket_width, rgb.height), fill)
    canvas.paste(rgb, ((bucket_width - rgb.width) // 2, 0))
    return canvas
