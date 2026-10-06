"""Deterministically trim empty vertical margins from a line crop.

The competition images already contain one text line, so autocropping only needs
to find the occupied row band.  It deliberately leaves the horizontal extent
unchanged: clipping ascenders or the ends of a very wide line is much more
damaging than retaining a little paper.
"""

from __future__ import annotations


def crop_to_line(image, *, threshold: int = 18, padding_fraction: float = 0.08):
    """Return ``image`` cropped to its ink-bearing rows.

    Ink is measured relative to a robust paper estimate taken from the image's
    grayscale histogram.  A small amount of vertical padding is retained.  If
    no convincing foreground is present, the original-sized image is returned.
    """

    from PIL import Image, ImageChops, ImageFilter

    if not 0 <= threshold <= 255:
        raise ValueError("threshold must be between 0 and 255")
    if padding_fraction < 0:
        raise ValueError("padding_fraction must not be negative")

    source = image.convert("RGB")
    gray = source.convert("L")
    # A light blur makes the row decision insensitive to isolated JPEG noise.
    smooth = gray.filter(ImageFilter.GaussianBlur(radius=max(0.5, gray.height / 200)))
    histogram = smooth.histogram()
    count = sum(histogram)
    target = max(0, count * 9 // 10)
    cumulative = 0
    paper = 255
    for value, frequency in enumerate(histogram):
        cumulative += frequency
        if cumulative >= target:
            paper = value
            break

    darkness = ImageChops.subtract(Image.new("L", smooth.size, paper), smooth)
    # BOX resizing computes the row-average darkness in optimized Pillow code.
    # A value of one is enough to retain thin abbreviation marks, while the
    # preceding blur suppresses isolated JPEG noise.
    profile = darkness.resize((1, smooth.height), Image.Resampling.BOX)
    occupied = [y for y in range(smooth.height) if profile.getpixel((0, y)) >= 1]

    if not occupied:
        return source
    pad = max(1, round((occupied[-1] - occupied[0] + 1) * padding_fraction))
    top = max(0, occupied[0] - pad)
    bottom = min(source.height, occupied[-1] + pad + 1)
    if top == 0 and bottom == source.height:
        return source
    return source.crop((0, top, source.width, bottom))
