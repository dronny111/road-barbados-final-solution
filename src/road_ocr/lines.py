"""Shared helpers for line-image recognizers.

The competition crops are single text lines with a median aspect ratio near
16.8:1 and a maximum near 40.5:1. Fixed square or thumbnail resizing destroys
these; every recognizer here instead preserves aspect and controls the rendered
line height directly.
"""

from __future__ import annotations

import csv
from pathlib import Path

from .records import CsvValidationError

PROMPT = "Transcribe this handwritten line of text exactly. Reply with the transcription only."

# Qwen2-VL tiles images into 28x28 patches, so both sides must land on a multiple.
PATCH = 28


def resolve_device(requested: str = "auto"):
    """Return (torch device, name) for the backend actually available.

    XLA is deliberately not auto-selected. ``torch_xla`` being importable does
    not mean a TPU is the intended target, and silently moving a run onto one
    would change both the numerics and the shape requirements without anyone
    asking. A TPU run has to say ``--device xla``.
    """

    import torch

    if requested == "xla":
        try:
            import torch_xla.core.xla_model as xm
        except ImportError as error:  # pragma: no cover - needs a TPU runtime
            raise RuntimeError(
                "--device xla needs torch_xla, which is absent. On a Kaggle TPU "
                "instance install the wheel matching the runtime's torch."
            ) from error
        return xm.xla_device(), "xla"
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda"), "cuda"
        if torch.backends.mps.is_available():
            return torch.device("mps"), "mps"
        return torch.device("cpu"), "cpu"
    return torch.device(requested), requested


def resolve_precision(requested: str, device: str = ""):
    """Return (torch dtype, name) for the accelerator actually present.

    Free Colab and Kaggle hand out a T4, which is Turing and has no bfloat16
    support, so a hard-coded bf16 either refuses to run or falls back to something
    far slower. Ampere and newer, and Apple MPS, all take bf16.

    torch.cuda.is_bf16_supported() counts software emulation, so it answers True
    on Turing; the compute capability is what distinguishes real bf16 hardware.
    """

    import torch

    if requested == "auto":
        if device == "xla":
            # TPUs have native bfloat16 and no fp16 path worth using.
            requested = "bf16"
        elif torch.cuda.is_available():
            requested = "bf16" if torch.cuda.get_device_capability()[0] >= 8 else "fp16"
        else:
            requested = "bf16"
    if requested not in ("bf16", "fp16"):
        raise ValueError(f"unknown precision {requested!r}")
    return (torch.bfloat16 if requested == "bf16" else torch.float16), requested


def clean_text(value: str) -> str:
    """Collapse whitespace so labels and predictions share one single-line form."""

    return " ".join(str(value).split())


def line_size(
    width: int,
    height: int,
    target_height: int,
    max_pixels: int,
    patch: int = PATCH,
) -> tuple[int, int]:
    """Return a patch-aligned (width, height) that preserves the line's aspect.

    The line is scaled to ``target_height`` so character strokes keep a constant
    rendered size regardless of how wide the crop is. Only if the result exceeds
    ``max_pixels`` is it scaled back down, which bounds the visual token count on
    the most extreme crops instead of shrinking every line to fit the worst one.
    """

    if width <= 0 or height <= 0:
        raise ValueError(f"degenerate image size {width}x{height}")
    aspect = width / height
    # Align the height to whole patches first and derive the width from it, so
    # rounding never distorts the aspect. Scaling back down therefore drops one
    # patch row at a time rather than squashing the widest lines vertically.
    for rows in range(max(1, round(target_height / patch)), 0, -1):
        aligned_height = rows * patch
        aligned_width = max(patch, round(aligned_height * aspect / patch) * patch)
        if aligned_width * aligned_height <= max_pixels or rows == 1:
            return aligned_width, aligned_height
    raise AssertionError("unreachable")


def model_patch(model_id) -> int:
    """Pixels per merged visual token for a local checkpoint: 28 for Qwen2/2.5-VL, 32 for Qwen3-VL.

    Lines are pre-aligned to this grid so the processor never resizes them again. A
    28-aligned line fed to Qwen3-VL would be re-rounded to 32 and lose its exact size.
    Falls back to 28 when the id is not a local directory with a preprocessor config.
    """
    import json

    config = Path(model_id) / "preprocessor_config.json"
    if not config.is_file():
        return PATCH
    data = json.loads(config.read_text())
    return int(data.get("patch_size", 14)) * int(data.get("merge_size", 2))


def load_line_image(
    path: Path, target_height: int, max_pixels: int, autocrop: bool = False,
    shape_buckets: bool = False, patch: int = PATCH, view: str = "original",
):
    """Open a crop as RGB at a patch-aligned size that preserves its aspect.

    ``view`` (road_ocr.views) preprocesses the source-resolution crop first; "original"
    leaves the pixels untouched.

    ``target_height`` scales the whole crop, so how large the handwriting renders
    depends on how tightly the organizer cropped the line -- from 28px to 1131px
    for the same kind of content. Two options address that:

    ``autocrop`` trims to the text band first. Measured worse than doing nothing
    and kept only so that result stays reproducible.

    The cheaper lever needs no flag: raising ``target_height`` past what
    ``max_pixels`` allows makes every crop fill its pixel budget, which lifts
    exactly the loose crops that were leaving most of theirs unused.
    """

    from PIL import Image

    with Image.open(path) as image:
        image = image.convert("RGB")
        if view != "original":
            from .views import apply_view

            image = apply_view(image, view)
        if autocrop:
            from .line_crop import crop_to_line

            image = crop_to_line(image)
        size = line_size(image.width, image.height, target_height, max_pixels, patch)
        resized = image.resize(size, Image.Resampling.BICUBIC)
        if shape_buckets:
            # XLA compiles per input shape, so a TPU needs a handful of widths
            # rather than the hundred-odd that preserving aspect produces.
            from .buckets import pad_to_bucket

            return pad_to_bucket(resized, patch=patch, max_pixels=max_pixels)
        return resized


def read_fold_ids(path: str | Path, fold: int) -> set[str]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["ID", "fold"]:
            raise CsvValidationError(
                f"{path}: expected columns ['ID', 'fold'], got {reader.fieldnames}"
            )
        return {row["ID"] for row in reader if int(row["fold"]) == fold}


def split_by_fold(
    rows: list[dict[str, str]], fold_manifest: str | Path, fold: int
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Split rows into (train, validation) using the deterministic fold manifest.

    The organizer starter draws a random split on every run, which makes its
    results unreproducible and leaks across experiments. This uses the same
    frozen manifest as every other run in the repository.
    """

    validation_ids = read_fold_ids(fold_manifest, fold)
    if not validation_ids:
        raise CsvValidationError(f"{fold_manifest}: fold {fold} selects no rows")
    train = [row for row in rows if row["ID"] not in validation_ids]
    validation = [row for row in rows if row["ID"] in validation_ids]
    return train, validation


def select_training_rows(rows, fold_manifest, fold, *, full_data=False, exclude_fold=None):
    """Make all-label training explicit so it cannot masquerade as validation.

    ``exclude_fold`` also drops a second fold from training (validation is unchanged), so
    sibling models trained on different fold subsets all stay held out on ``fold``.
    """
    if full_data:
        if fold is not None or exclude_fold is not None:
            raise ValueError("full-data training cannot also hold out a fold")
        if not rows:
            raise ValueError("training selection is empty")
        return list(rows), []
    if fold is None:
        raise ValueError("a held-out fold or full-data mode is required")
    training, validation = split_by_fold(rows, fold_manifest, fold)
    if exclude_fold is not None:
        if exclude_fold == fold:
            raise ValueError("the excluded fold must differ from the held-out fold")
        dropped = read_fold_ids(fold_manifest, exclude_fold)
        training = [row for row in training if row["ID"] not in dropped]
    if not training or not validation:
        raise ValueError("training and validation selections must both be nonempty")
    return training, validation


def load_vlm(model_id: str, dtype, load_4bit: bool = False, device=None, **kwargs):
    """Load a vision-language recognizer without hard-coding its architecture.

    The recipe here is a 2B Qwen2-VL, and every measured result in the registry
    comes from that checkpoint. Naming the class directly pinned the whole
    pipeline to one model family, which makes trying a stronger recognizer a
    code change in nine places rather than a config change in one.

    ``AutoModelForImageTextToText`` resolves a Qwen2-VL checkpoint to exactly
    ``Qwen2VLForConditionalGeneration``, so this is behaviour-preserving for
    every existing run, and it also accepts Qwen2.5-VL and the other VL families.

    ``load_4bit`` is what makes a 7B reachable at all on the free tier. A 7B in
    fp16 is about 16.6GB of weights against a T4's 16GB, and LoRA training needs
    roughly 21.6GB, so it does not fit; in NF4 the weights are about 5.6GB and it
    does. Quantization is lossy, so a 4-bit 7B is not a clean comparison against
    an fp16 3B: it tests whether the extra capacity survives the quantization,
    not what the 7B is worth at full precision.
    """

    from transformers import AutoModelForImageTextToText

    on_xla = device is not None and "xla" in str(device)
    on_mps = device is not None and "mps" in str(device)
    place_here = on_xla or on_mps
    if on_xla and load_4bit:
        raise ValueError(
            "bitsandbytes 4-bit is CUDA-only, so --load-4bit cannot be combined "
            "with --device xla. A 7B on TPU needs sharding across cores instead."
        )
    if place_here:
        # accelerate's device_map dispatches against CUDA/CPU and does not place
        # onto an XLA device, so load unplaced and move the whole model over.
        #
        # MPS needs the same treatment for a different reason: accelerate budgets
        # it at whatever host RAM is free right now and deliberately omits a "cpu"
        # entry, since both share the same memory. So on a busy machine the tail
        # of the checkpoint is mapped to "disk", and a partly offloaded base model
        # sends PeftModel.from_pretrained down an offload path that dies rebuilding
        # its index. Placing the model here makes adapter loads deterministic
        # rather than failing only when free memory happens to dip.
        kwargs.pop("device_map", None)
    if load_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
    if is_kimi(model_id):
        from transformers import AutoModelForCausalLM

        # Its remote code targets transformers 4.48-4.50 (requirements-kimi.txt), which
        # names the dtype argument torch_dtype; dtype= would silently load fp32 there.
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, trust_remote_code=True, **kwargs)
        prepare_kimi(model)
    else:
        model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=dtype, **kwargs)
    return model.to(device) if place_here else model


def is_kimi(model_id) -> bool:
    """True for a local Kimi-VL snapshot, read from config.json so no remote code runs.

    Like ``model_patch``, this expects a local directory (the notebooks pass the snapshot
    path). A bare hub id falls through to the Auto class, which fails loudly for Kimi-VL.
    """
    import json

    config = Path(model_id) / "config.json"
    return config.is_file() and json.loads(config.read_text()).get("model_type") == "kimi_vl"


def prepare_kimi(model) -> None:
    """Make Kimi-VL's inference-only remote code trainable, batch-safe and T4-sized.

    1. MoE. The router's noaux_tc top-k asserts ``not self.training``, and the train-mode
       path re-implements dispatch with a per-expert loop and a balance loss for the
       router. Router and routed experts stay frozen, so each MoE block is pinned to eval
       mode (inference routing, no aux loss), and ``moe_infer`` is unwrapped from its
       ``@torch.no_grad()`` so gradients reach lower layers through the routed experts.
       ponytail: LoRA dropout on the shared experts is therefore off; 0.05 elsewhere.
    2. Vision attention. The eager path adds its block-diagonal mask as +1 instead of
       masking, so lines packed in one batch attend to each other. SDPA masks properly.
    3. Checkpointing. The model declares none, and its decoder loop ignores the flag, so
       each vision block and decoder layer gets HF's standard ``gradient_checkpointing``
       hook, which ``gradient_checkpointing_enable`` then switches on. HF's input-grad
       hook turns the embeddings into a leaf, which the in-place image merge rejects, so
       it returns a non-leaf copy instead.
    """
    import types

    import torch

    blocks = [m for m in model.modules() if type(m).__name__ == "DeepseekV3MoE"]
    if not blocks:
        raise ValueError("Kimi-VL model has no DeepseekV3MoE blocks to patch")
    cls = type(blocks[0])
    cls.moe_infer = getattr(cls.moe_infer, "__wrapped__", cls.moe_infer)
    for block in blocks:
        if getattr(block, "ep_size", 1) != 1:
            raise ValueError("expert parallelism is not supported")
        block.train = types.MethodType(lambda self, mode=True: torch.nn.Module.train(self, False), block)
        block.train()

    def checkpointable(layer):
        forward = layer.forward
        layer.gradient_checkpointing = False

        def run(*args, **kwargs):
            if layer.gradient_checkpointing and layer.training and torch.is_grad_enabled():
                return layer._gradient_checkpointing_func(forward, *args, **kwargs)
            return forward(*args, **kwargs)
        layer.forward = run

    # 4. Vision memory. MoonViT packs every image of a batch into one sequence behind a
    #    block-diagonal mask, so attention costs (total patches)^2: 4 rows x 5 images asked
    #    for 36 GiB on a T4. The mask makes images independent, so encoding one at a time is
    #    the same computation at the cost of the largest single image.
    tower = model.vision_tower
    packed = tower.forward

    def per_image(pixel_values, grid_hws):
        sizes = (grid_hws[:, 0] * grid_hws[:, 1]).tolist()
        return [feature for pixels, grid in zip(pixel_values.split(sizes), grid_hws.split(1))
                for feature in packed(pixels, grid)]
    tower.forward = per_image

    # 5. Generation logits. The remote forward projects every prompt position onto the
    #    163,840-token vocabulary and upcasts it to fp32, but generate() reads only the last
    #    one: a 4-shot batch of 4 needed 4.4 GiB for that alone. Without gradients (always
    #    so in generate, never in training) only the last position is projected.
    head = model.language_model.lm_head
    project = head.forward
    head.forward = lambda hidden: project(hidden if torch.is_grad_enabled() else hidden[:, -1:, :])

    for layer in model.modules():
        kind = type(layer).__name__
        if kind == "MoonVitEncoderLayer":
            layer.attn_implementation = "sdpa"
        if kind in ("MoonVitEncoderLayer", "DeepseekV3DecoderLayer"):
            checkpointable(layer)
    type(model).supports_gradient_checkpointing = True

    def enable_input_require_grads(self):
        hook = lambda module, inputs, output: output.requires_grad_(True).clone()
        self._require_grads_hook = self.get_input_embeddings().register_forward_hook(hook)
    model.enable_input_require_grads = types.MethodType(enable_input_require_grads, model)
