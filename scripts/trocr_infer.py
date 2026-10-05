#!/usr/bin/env python3
"""Transcribe line images with a fine-tuned TrOCR model; the TrOCR twin of vlm_infer.py.

Same selection flags, ID,Target output and metadata keys (samples, empty_predictions,
runtime_seconds, score) as vlm_infer.py. --adapter is the fine-tuned model directory
(TrOCR is fully fine-tuned, so there is no separate adapter); --model-id is the base
used only when --adapter is absent. VLM-only size flags are accepted and ignored.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from road_ocr.lines import read_fold_ids, resolve_device  # noqa: E402
from road_ocr.metrics import score_pairs  # noqa: E402
from road_ocr.records import index_unique, read_csv  # noqa: E402
from road_ocr.trocr_model import load_trocr  # noqa: E402
from road_ocr.views import VIEWS, apply_view  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--images", default="images")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--adapter", help="fine-tuned TrOCR model directory")
    parser.add_argument("--processor", help="accepted for parity; the model directory's processor is used")
    parser.add_argument("--fold-manifest")
    parser.add_argument("--fold", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--target-height", type=int)
    parser.add_argument("--max-pixels", type=int)
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--view", default="original", choices=VIEWS,
                        help="test-time preprocessing of each crop (road_ocr.views)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata")
    args = parser.parse_args()
    if args.load_4bit:
        parser.error("--load-4bit is not supported for TrOCR")
    if (args.fold_manifest is None) != (args.fold is None):
        parser.error("--fold-manifest and --fold must be provided together")

    with Path(args.input).open("r", encoding="utf-8-sig", newline="") as handle:
        fieldnames = csv.DictReader(handle).fieldnames
    if fieldnames not in (["ID"], ["ID", "Target"]):
        parser.error(f"{args.input}: expected ID or ID,Target columns, got {fieldnames}")
    rows = read_csv(args.input, fieldnames)
    index_unique(rows, args.input)
    if args.fold_manifest is not None:
        fold_ids = read_fold_ids(args.fold_manifest, args.fold)
        rows = [row for row in rows if row["ID"] in fold_ids]
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    if not rows:
        parser.error("selection contains no rows")

    import torch
    from PIL import Image

    device, backend = resolve_device(args.device)
    if args.precision == "auto":
        precision = "fp16" if backend == "cuda" else "fp32"
    else:
        precision = args.precision
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[precision]
    source = args.adapter or args.model_id
    processor, model = load_trocr(source, device, dtype)
    model.eval()
    print(f"device: {backend}\nprecision: {precision}", flush=True)

    started = time.time()
    predictions = []
    images = Path(args.images)
    for start in range(0, len(rows), args.batch_size):
        chunk = rows[start:start + args.batch_size]
        batch = []
        for row in chunk:
            path = images / f"{row['ID']}.jpg"
            if not path.is_file():  # cleaned sets (scripts/clean_images.py) are PNG
                path = path.with_suffix(".png")
            if not path.is_file():
                raise FileNotFoundError(path)
            with Image.open(path) as image:
                batch.append(apply_view(image.convert("RGB"), args.view))
        pixel_values = processor(images=batch, return_tensors="pt").pixel_values.to(device, dtype)
        controls = {"max_new_tokens": args.max_new_tokens, "num_beams": args.num_beams}
        if args.no_repeat_ngram_size:
            controls["no_repeat_ngram_size"] = args.no_repeat_ngram_size
        with torch.inference_mode():
            generated = model.generate(pixel_values, **controls)
        for row, text in zip(chunk, processor.batch_decode(generated, skip_special_tokens=True), strict=True):
            predictions.append({"ID": row["ID"], "Target": text.strip()})
        print(f"recognized {len(predictions)}/{len(rows)}", flush=True)
    runtime = time.time() - started

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ID", "Target"])
        writer.writeheader()
        writer.writerows(predictions)
    empty = sum(1 for p in predictions if not p["Target"])
    metadata = {
        "family": "trocr", "adapter": args.adapter, "model_id": args.model_id, "view": args.view,
        "command": " ".join(sys.argv), "device": backend, "precision": precision,
        "fold": args.fold, "input": args.input, "num_beams": args.num_beams,
        "no_repeat_ngram_size": args.no_repeat_ngram_size, "max_new_tokens": args.max_new_tokens,
        "output": str(output), "platform": platform.platform(), "runtime_seconds": runtime,
        "samples": len(rows), "empty_predictions": empty, "torch_version": torch.__version__,
    }
    if fieldnames == ["ID", "Target"]:
        result = score_pairs((row["Target"], p["Target"]) for row, p in zip(rows, predictions, strict=True))
        metadata["score"] = result.__dict__
        print(json.dumps(result.__dict__, indent=2, sort_keys=True))
    if args.metadata:
        path = Path(args.metadata)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
