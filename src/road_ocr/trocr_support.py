"""Dependency-free run guards for the standalone TrOCR notebook."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

from .metrics import score_pairs

FOLD_SHA256 = "fff8a4994bc606cdd9987ae521712976bd83c180223dce596340d192ff58a448"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_folds(path):
    actual = sha256_file(path)
    if actual != FOLD_SHA256:
        raise ValueError("Fold checksum differs from the frozen repository manifest")
    return actual


def exact_metrics(references, predictions):
    if len(references) != len(predictions):
        raise ValueError("Prediction count differs from original reference count")
    # Normalization belongs to the recognizer; never normalize references here.
    score = score_pairs(zip(references, predictions))
    return {"cer": score.cer, "wer": score.wer, "combined": score.combined}


def estimate_hours(*, train_seconds, smoke_steps, train_rows, effective_batch,
                   epochs, decode_seconds, smoke_rows, validation_rows, test_rows,
                   safety_factor=1.5):
    values = (train_seconds, smoke_steps, train_rows, effective_batch, epochs,
              decode_seconds, smoke_rows, validation_rows, safety_factor)
    if any(not math.isfinite(v) or v <= 0 for v in values) or test_rows < 0:
        raise ValueError("Compute estimate needs positive, finite smoke measurements")
    steps = math.ceil(train_rows / effective_batch) * epochs
    training = train_seconds / smoke_steps * steps
    # Smoke uses final beam/TTA settings. Use that slower per-row cost for epoch
    # evaluation too, plus final validation and optional test inference.
    decoding = decode_seconds / smoke_rows * (validation_rows * (epochs + 1) + test_rows)
    return safety_factor * (training + decoding) / 3600


def require_budget(estimated_hours, elapsed_seconds, max_hours):
    if (not all(math.isfinite(v) for v in (estimated_hours, elapsed_seconds, max_hours))
            or estimated_hours < 0 or elapsed_seconds < 0 or max_hours <= 0):
        raise ValueError("Invalid compute budget")
    if estimated_hours + elapsed_seconds / 3600 > max_hours:
        raise RuntimeError("Budget gate: estimated remaining work exceeds the configured hours")
