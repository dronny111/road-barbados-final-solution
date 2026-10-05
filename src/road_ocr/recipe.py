"""Training-recipe pieces: curriculum order, height-routed experts, raw/cleaned view mix.

Pure and seeded so they test without a model. Used by scripts/vlm_finetune.py and vlm_infer.py.
"""
from __future__ import annotations

import random

EXPERT_HEIGHT = 150  # px; the tight/loose split the vote already stratifies on
EXPERTS = ("default", "loose")  # "default" is the tight expert and the shared warm-up adapter


def stratum(height: int, threshold: int = EXPERT_HEIGHT) -> str:
    return "tight" if height < threshold else "loose"


def curriculum_order(meta: list[dict], seed: int, epoch: int = 0) -> list[int]:
    """Easy to hard: real tight lines (short first), real loose lines, then pseudo-labels by
    vote support (highest first). Shuffled inside each bin, reseeded per epoch."""
    rng = random.Random(seed * 1_000_003 + epoch)

    def key(i):
        m = meta[i]
        if m["pseudo"]:
            return (2, -round(m["support"], 3), 0)
        return (0 if stratum(m["height"]) == "tight" else 1, 0, min(m["length"] // 20, 6))

    bins: dict[tuple, list[int]] = {}
    for i in range(len(meta)):
        bins.setdefault(key(i), []).append(i)
    order = []
    for k in sorted(bins):
        rng.shuffle(bins[k])
        order += bins[k]
    return order


def use_alt_view(rng: random.Random | None, prob: float) -> bool:
    return rng is not None and rng.random() < prob


def fork_adapter(model, src: str = "default", dst: str = "loose") -> int:
    """Copy every LoRA tensor of adapter `src` into `dst`; returns how many were copied."""
    params = dict(model.named_parameters())
    copied = 0
    for name, tensor in params.items():
        if "lora_" in name and f".{src}." in name:
            target = params.get(name.replace(f".{src}.", f".{dst}."))
            if target is None:
                raise ValueError(f"adapter {dst!r} has no tensor matching {name}")
            target.data.copy_(tensor.data)
            copied += 1
    if not copied:
        raise ValueError(f"adapter {src!r} has no LoRA tensors to copy")
    return copied
