#!/usr/bin/env python3
"""Recognize line crops with a Qwen2-VL model and score them if labels exist.

The command-line surface mirrors scripts/kraken_infer.py so the same scoring,
routing, and submission tools work unchanged against either recognizer.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import platform
import random
import shlex
import sys
import time
from pathlib import Path

from road_ocr.lines import (
    PROMPT,
    clean_text,
    is_kimi,
    load_line_image,
    model_patch,
    load_vlm,
    read_fold_ids,
    resolve_device,
    resolve_precision,
)
from road_ocr.metrics import score_pairs
from road_ocr.records import index_unique, read_csv


def decode_controls(args) -> dict:
    """Optional generate() knobs, omitted entirely when left at their defaults.

    Passing these unconditionally would change the greedy baseline, so each one
    is only forwarded when explicitly set. no-repeat-ngram is genuinely risky
    here: these are legal formulae and some references really do repeat a phrase
    ("or either of their or either of theire"), so it must be measured rather
    than assumed.
    """

    controls = {}
    if args.no_repeat_ngram_size:
        controls["no_repeat_ngram_size"] = args.no_repeat_ngram_size
    if args.length_penalty is not None:
        controls["length_penalty"] = args.length_penalty
    return controls


def sampling_controls(args) -> dict:
    """Sampling knobs, again omitted entirely unless --num-samples asks for them.

    Preference data needs candidates that differ; greedy decoding returns the
    same string every time, which is exactly why the earlier GRPO run had no
    advantage to learn from. One sample keeps the frozen greedy path untouched.
    """

    if args.num_samples == 1:
        return {"do_sample": False}
    controls = {"do_sample": True, "num_return_sequences": args.num_samples,
                "temperature": args.temperature}
    if args.top_p is not None:
        controls["top_p"] = args.top_p
    return controls


def select_shots(source, manifest, fold, count, seed):
    """A fixed, seeded set of labelled lines from outside the held-out fold.

    The same examples precede every query, so the prompt is one constant and the
    comparison against zero-shot changes nothing but their presence.
    """

    held_out = read_fold_ids(manifest, fold)
    pool = sorted((row for row in read_csv(source, ["ID", "Target"]) if row["ID"] not in held_out),
                  key=lambda row: row["ID"])
    if count > len(pool):
        raise ValueError(f"--shots {count} exceeds the {len(pool)} rows outside fold {fold}")
    return random.Random(seed).sample(pool, count)


def predict_rows(rows, images: Path, model, processor, args, shots=()):
    """Return (first sample per row, every sampled candidate)."""

    import torch

    predictions: list[dict[str, str]] = []
    candidates: list[dict[str, str]] = []
    user = {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}
    # In-context examples are earlier turns: the same user message, then its transcription.
    turns = [message for shot in shots
             for message in (user, {"role": "assistant", "content": clean_text(shot["Target"])})]
    prompt = processor.apply_chat_template(turns + [user], tokenize=False, add_generation_prompt=True)
    kimi = type(processor).__name__.startswith("KimiVL")
    line = lambda record_id, view=args.view: load_line_image(
        images / f"{record_id}.jpg", args.target_height, args.max_pixels,
        args.autocrop, args.shape_buckets, model_patch(args.model_id), view,
    )
    shot_images = [line(shot["ID"], "original") for shot in shots]  # examples stay as trained

    for start in range(0, len(rows), args.batch_size):
        chunk = rows[start : start + args.batch_size]
        batch_images = []
        for row in chunk:
            image_path = images / f"{row['ID']}.jpg"
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            row_images = [*shot_images, line(row["ID"])]
            # Kimi-VL's processor takes one flat image list rather than one list per text.
            if kimi:
                batch_images.extend(row_images)
            else:
                batch_images.append(row_images)

        batch = processor(
            text=[prompt] * len(chunk),
            images=batch_images,
            return_tensors="pt",
            padding=True,
        ).to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **batch,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
                **decode_controls(args),
                **sampling_controls(args),
            )
        trimmed = generated[:, batch["input_ids"].shape[1] :]
        decoded = processor.batch_decode(trimmed, skip_special_tokens=True)
        # generate() returns each row's samples contiguously, so regroup by row.
        grouped = [decoded[i * args.num_samples : (i + 1) * args.num_samples]
                   for i in range(len(chunk))]
        for row, texts in zip(chunk, grouped, strict=True):
            predictions.append({"ID": row["ID"], "Target": clean_text(texts[0])})
            for index, text in enumerate(texts):
                candidates.append({"ID": row["ID"], "sample": str(index),
                                   "Target": clean_text(text)})
        done = min(start + args.batch_size, len(rows))
        print(f"recognized {done}/{len(rows)}", flush=True)
    return predictions, candidates


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="ID or ID,Target CSV")
    parser.add_argument("--images", default="images")
    parser.add_argument("--model-id", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--adapter", help="LoRA adapter directory; omit for zero-shot")
    parser.add_argument(
        "--processor", help="Processor directory or model ID; use the base model ID "
                            "when evaluating a Trainer checkpoint without processor files",
    )
    parser.add_argument("--fold-manifest")
    parser.add_argument("--fold", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument(
        "--num-samples", type=int, default=1,
        help="draw this many sampled candidates per row; 1 keeps greedy decoding",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260906,
                        help="Random seed for sampled decoding; recorded in metadata")
    parser.add_argument("--top-p", type=float)
    parser.add_argument(
        "--candidates", help="write every sampled candidate here as ID,sample,Target",
    )
    parser.add_argument(
        "--no-repeat-ngram-size", type=int, default=0,
        help="block any token n-gram from repeating; 0 leaves generation unchanged",
    )
    parser.add_argument(
        "--length-penalty", type=float,
        help="beam-search length penalty; unset leaves the model default",
    )
    parser.add_argument("--target-height", type=int, default=112)
    parser.add_argument("--max-pixels", type=int, default=451_584)
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cuda", "mps", "cpu", "xla"],
        help="xla targets a TPU and requires torch_xla plus --shape-buckets",
    )
    parser.add_argument(
        "--shape-buckets", action="store_true",
        help="pad rendered widths onto a few fixed buckets; required for XLA/TPU, "
             "where every distinct input shape triggers a recompilation",
    )
    parser.add_argument(
        "--load-4bit", action="store_true",
        help="NF4-quantize the base weights; must match how the adapter was trained",
    )
    parser.add_argument(
        "--autocrop", action="store_true",
        help="trim each crop to its own text band before resizing "
             "(measured worse than doing nothing; kept for reproducibility)",
    )
    parser.add_argument(
        "--shots", type=int, default=0,
        help="in-context examples: this many labelled lines from outside --shot-fold, the "
             "same seeded set before every query; 0 keeps the single-turn prompt",
    )
    parser.add_argument("--shot-fold", type=int, help="fold the examples must come from outside")
    parser.add_argument("--shot-source", default="Train.csv")
    parser.add_argument("--shot-manifest", default="data/splits/folds.csv")
    parser.add_argument("--view", default="original", choices=["original", "flatten", "binary"],
                        help="test-time preprocessing of each query crop (road_ocr.views)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata")
    parser.add_argument("--precision", choices=["auto", "bf16", "fp16"], default="auto")
    args = parser.parse_args()
    if args.device == "xla" and not args.shape_buckets:
        parser.error(
            "--device xla requires --shape-buckets: XLA compiles per input shape, and "
            "preserving aspect otherwise produces over a hundred distinct widths"
        )
    if (args.fold_manifest is None) != (args.fold is None):
        parser.error("--fold-manifest and --fold must be provided together")
    if args.num_samples < 1:
        parser.error("--num-samples must be at least 1")
    if args.num_samples > 1:
        if args.num_beams > 1:
            parser.error("--num-samples draws independent samples; it cannot be combined "
                         "with beam search")
        if args.temperature <= 0:
            parser.error("--num-samples above 1 needs a positive --temperature, or every "
                         "candidate is the same string")
    if args.shots and args.shot_fold is None:
        parser.error("--shots needs --shot-fold so the examples cannot be held-out rows")
    if args.candidates and args.num_samples == 1:
        parser.error("--candidates without --num-samples above 1 would only repeat --output")

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
    from transformers import AutoProcessor

    processor_source = args.processor or args.adapter or args.model_id
    processor = AutoProcessor.from_pretrained(processor_source, trust_remote_code=is_kimi(args.model_id))
    processor.tokenizer.padding_side = "left"
    device, backend = resolve_device(args.device)
    dtype, precision = resolve_precision(args.precision, backend)
    print(f"device: {backend}", flush=True)
    print(f"precision: {precision}", flush=True)
    model = load_vlm(
        args.model_id, dtype, load_4bit=args.load_4bit, device=device, device_map="auto"
    )
    experts = False
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
        # Height-routed experts: the root adapter is the tight expert, <adapter>/loose the loose one.
        if (Path(args.adapter) / "loose").is_dir():
            model.load_adapter(Path(args.adapter) / "loose", adapter_name="loose")
            experts = True
    model.eval()

    if args.num_samples > 1:
        from transformers import set_seed
        set_seed(args.seed)
    started = time.time()
    shots = (select_shots(args.shot_source, args.shot_manifest, args.shot_fold, args.shots, args.seed)
             if args.shots else [])
    if {shot["ID"] for shot in shots} & {row["ID"] for row in rows}:
        raise ValueError("an in-context example is also a query row")
    if experts:
        from PIL import Image

        from road_ocr.recipe import EXPERT_HEIGHT, stratum

        def height(row):
            with Image.open(Path(args.images) / f"{row['ID']}.jpg") as probe:
                return probe.height
        # Each stratum is decoded with its own expert; output keeps the input row order.
        by_id, by_id_candidates = {}, []
        for name, adapter in (("tight", "default"), ("loose", "loose")):
            part = [row for row in rows if stratum(height(row), EXPERT_HEIGHT) == name]
            if part:
                model.set_adapter(adapter)
                got, cand = predict_rows(part, Path(args.images), model, processor, args, shots)
                by_id.update((p["ID"], p) for p in got)
                by_id_candidates += cand
        predictions = [by_id[row["ID"]] for row in rows]
        candidates = by_id_candidates
    else:
        predictions, candidates = predict_rows(rows, Path(args.images), model, processor, args, shots)
    runtime = time.time() - started

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ID", "Target"])
        writer.writeheader()
        writer.writerows(predictions)

    if args.candidates:
        candidate_path = Path(args.candidates)
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        with candidate_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["ID", "sample", "Target"])
            writer.writeheader()
            writer.writerows(candidates)

    metadata = {
        "adapter": args.adapter,
        "experts": experts,
        "num_samples": args.num_samples,
        "view": args.view,
        "shots": [shot["ID"] for shot in shots],
        "shot_fold": args.shot_fold,
        "shot_seed": args.seed if shots else None,
        "seed": args.seed if args.num_samples > 1 else None,
        "temperature": args.temperature if args.num_samples > 1 else None,
        "top_p": args.top_p if args.num_samples > 1 else None,
        "candidates": args.candidates,
        "autocrop": args.autocrop,
        "device": backend,
        "load_4bit": args.load_4bit,
        "shape_buckets": args.shape_buckets,
        "command": " ".join(shlex.quote(value) for value in sys.argv),
        "fold": args.fold,
        "input": args.input,
        "max_pixels": args.max_pixels,
        "model_id": args.model_id,
        "processor": processor_source,
        "num_beams": args.num_beams,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
        "length_penalty": args.length_penalty,
        "output": str(output),
        "platform": platform.platform(),
        "precision": precision,
        "python": platform.python_version(),
        "runtime_seconds": runtime,
        "samples": len(rows),
        "target_height": args.target_height,
        "torch_version": torch.__version__,
        "transformers_version": importlib.metadata.version("transformers"),
    }
    empty = sum(1 for prediction in predictions if not prediction["Target"])
    if empty:
        print(f"WARNING: {empty} predictions are empty", flush=True)
    metadata["empty_predictions"] = empty
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
