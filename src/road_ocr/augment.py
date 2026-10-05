"""Photometric and geometric jitter for training crops only.

Nothing here touches validation or inference: the recogniser must see the corpus
exactly as the organiser cropped it. These operations exist because the measured
failure is reading the ink -- a quarter of the character error is casing and
abbreviation marks, which are stroke-level distinctions -- and the training set
is 3,280 lines of one archive's hands. Jitter is the cheapest way to stop the
adapter memorising those particular strokes.

Every operation preserves the canvas size, so token counts, the pixel budget and
the batch memory profile are identical to an unaugmented run.
"""

from __future__ import annotations

# ponytail: affine + photometric + codec noise only. Elastic warp is the usual
# next rung for handwriting and needs numpy; add it if this pass measures a gain.
DEFAULTS = dict(rotate=1.5, shear=0.04, thickness=0.30, brightness=0.15,
                contrast=0.15, blur=0.25, jpeg=0.30)


def augment_line(image, rng, **overrides):
    """Return a jittered copy of ``image`` at exactly its original size.

    ``rng`` is a ``random.Random``, so a run is reproducible from its seed.
    """

    from PIL import Image, ImageEnhance, ImageFilter

    settings = {**DEFAULTS, **overrides}
    for name, value in settings.items():
        if value < 0:
            raise ValueError(f"{name} must not be negative, got {value}")
    image = image.convert("RGB")
    width, height = image.size

    angle = rng.uniform(-settings["rotate"], settings["rotate"])
    shear = rng.uniform(-settings["shear"], settings["shear"])
    if angle or shear:
        import math

        # Rotate about the centre and shear horizontally in one inverse affine
        # map, filling exposed corners with the page's own average tone rather
        # than white, which would read as a crop edge.
        radians = math.radians(angle)
        cos, sin = math.cos(radians), math.sin(radians)
        centre_x, centre_y = width / 2, height / 2
        a, b = cos, sin + shear
        d, e = -sin, cos
        matrix = (a, b, centre_x - a * centre_x - b * centre_y,
                  d, e, centre_y - d * centre_x - e * centre_y)
        fill = image.resize((1, 1), Image.Resampling.BILINEAR).getpixel((0, 0))
        image = image.transform((width, height), Image.AFFINE, matrix,
                                resample=Image.Resampling.BILINEAR, fillcolor=fill)

    if rng.random() < settings["thickness"]:
        # MinFilter darkens and thickens strokes, MaxFilter thins them.
        image = image.filter(ImageFilter.MinFilter(3) if rng.random() < 0.5
                             else ImageFilter.MaxFilter(3))
    for name, enhancer in (("brightness", ImageEnhance.Brightness),
                           ("contrast", ImageEnhance.Contrast)):
        if settings[name]:
            image = enhancer(image).enhance(1 + rng.uniform(-settings[name], settings[name]))
    if rng.random() < settings["blur"]:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 0.7)))
    if rng.random() < settings["jpeg"]:
        import io

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=rng.randint(45, 90))
        buffer.seek(0)
        with Image.open(buffer) as reopened:
            image = reopened.convert("RGB")

    if image.size != (width, height):  # pragma: no cover - guards a future edit
        raise AssertionError("Augmentation changed the canvas size")
    return image
