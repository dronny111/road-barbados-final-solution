"""Validate single-GPU LoRA checkpoints before spending compute on recovery."""

from __future__ import annotations

import json
from pathlib import Path


def validate_checkpoint(path: str | Path, *, resume: bool = False) -> dict:
    path = Path(path)
    required = ["adapter_config.json", "trainer_state.json"]
    if resume:
        required += ["optimizer.pt", "scheduler.pt", "rng_state.pth"]
    missing = [name for name in required if not (path / name).is_file()
               or (path / name).stat().st_size == 0]
    weights = [path / name for name in ("adapter_model.safetensors", "adapter_model.bin")]
    if not any(p.is_file() and p.stat().st_size > 0 for p in weights):
        missing.append("adapter_model.safetensors (or adapter_model.bin)")
    if missing:
        raise ValueError(f"Incomplete checkpoint {path}: missing {', '.join(missing)}")
    adapter = json.loads((path / "adapter_config.json").read_text())
    state = json.loads((path / "trainer_state.json").read_text())
    if adapter.get("peft_type") != "LORA":
        raise ValueError(f"{path}: expected a LoRA adapter")
    if not isinstance(state.get("global_step"), int) or state["global_step"] <= 0:
        raise ValueError(f"{path}: missing or invalid saved global step")
    return state
