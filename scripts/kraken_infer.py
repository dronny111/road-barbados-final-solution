#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import shlex
import sys
import time
from pathlib import Path

from road_ocr.metrics import score_pairs
from road_ocr.records import CsvValidationError, index_unique, read_csv


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_fold_ids(path: str | Path, fold: int) -> set[str]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["ID", "fold"]:
            raise CsvValidationError(
                f"{path}: expected columns ['ID', 'fold'], got {reader.fieldnames}"
            )
        return {row["ID"] for row in reader if int(row["fold"]) == fold}


def make_full_crop_segmentation(image, image_path: Path):
    from kraken.containers import BaselineLine, Segmentation

    width, height = image.size
    right = max(0, width - 1)
    bottom = max(0, height - 1)
    baseline_y = max(0, height - 2)
    line = BaselineLine(
        id=image_path.stem,
        baseline=[(0, baseline_y), (right, baseline_y)],
        boundary=[(0, 0), (right, 0), (right, bottom), (0, bottom), (0, 0)],
        base_dir="L",
    )
    return Segmentation(
        type="baselines",
        imagename=image_path,
        text_direction="horizontal-lr",
        script_detection=False,
        lines=[line],
    )


def predict_rows(rows, image_dir: Path, model_path: Path, device: str, pad: int):
    from PIL import Image
    from kraken.lib.models import load_any
    from kraken.rpred import rpred

    new_api = model_path.suffix == ".safetensors"  # kraken >= 7 ppocrv6 weights; load_any is legacy VGSL only
    if new_api:
        from kraken.configs import RecognitionInferenceConfig
        from kraken.tasks import RecognitionTaskModel

        task = RecognitionTaskModel.load_model(model_path)
        config = RecognitionInferenceConfig(accelerator="cpu" if device == "cpu" else "auto")
    else:
        model = load_any(model_path, device=device)
    predictions: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        image_path = image_dir / f"{row['ID']}.jpg"
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        with Image.open(image_path) as image:
            segmentation = make_full_crop_segmentation(image, image_path)
            records = (
                task.predict(image, segmentation, config)
                if new_api
                else rpred(
                    model,
                    image,
                    segmentation,
                    pad=pad,
                    bidi_reordering=True,
                    no_legacy_polygons=True,
                )
            )
            prediction = " ".join(record.prediction for record in records)
        prediction = " ".join(prediction.split())
        predictions.append({"ID": row["ID"], "Target": prediction})
        if index == 1 or index % 25 == 0 or index == len(rows):
            print(f"recognized {index}/{len(rows)}", flush=True)
    return predictions


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a Kraken baseline reproducibly.")
    parser.add_argument("--input", required=True, help="ID or ID,Target CSV")
    parser.add_argument("--images", default="images")
    parser.add_argument("--model", required=True)
    parser.add_argument("--fold-manifest")
    parser.add_argument("--fold", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--pad", type=int, default=16)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata")
    args = parser.parse_args()
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

    model_path = Path(args.model)
    if not model_path.is_file():
        parser.error(f"model not found: {model_path}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    predictions = predict_rows(
        rows, Path(args.images), model_path, device=args.device, pad=args.pad
    )
    runtime = time.time() - started
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ID", "Target"])
        writer.writeheader()
        writer.writerows(predictions)

    metadata = {
        "command": " ".join(shlex.quote(value) for value in sys.argv),
        "device": args.device,
        "fold": args.fold,
        "input": args.input,
        "kraken_version": importlib.metadata.version("kraken"),
        "model": str(model_path),
        "model_sha256": sha256(model_path),
        "output": str(output),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "runtime_seconds": runtime,
        "samples": len(rows),
        "torch_version": importlib.metadata.version("torch"),
    }
    if fieldnames == ["ID", "Target"]:
        result = score_pairs(
            (row["Target"], prediction["Target"])
            for row, prediction in zip(rows, predictions, strict=True)
        )
        metadata["score"] = result.__dict__
        print(json.dumps(result.__dict__, indent=2, sort_keys=True))
    if args.metadata:
        metadata_path = Path(args.metadata)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(f"wrote {len(predictions)} predictions to {output} in {runtime:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
