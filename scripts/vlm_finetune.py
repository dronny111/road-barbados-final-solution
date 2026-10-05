#!/usr/bin/env python3
"""LoRA fine-tune a Qwen2-VL recognizer on the deterministic training folds.

This replaces the organizer VLM starter, which resolved the repository one
directory too high, drew a random row split on every run, asked the model to
preserve line breaks against single-line labels, thumbnailed very wide crops,
and saved its adapter to a path inconsistent with its configured output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import shlex
import sys
import time
from pathlib import Path

from road_ocr.checkpoints import validate_checkpoint
from road_ocr.lines import (
    PROMPT,
    clean_text,
    is_kimi,
    load_line_image,
    model_patch,
    load_vlm,
    resolve_device,
    resolve_precision,
    select_training_rows,
)
from road_ocr.records import index_unique, read_csv
from road_ocr.augment import augment_line
from road_ocr.lora import AttentionGradientAudit, resolve_targets
from road_ocr.recipe import (
    EXPERT_HEIGHT, curriculum_order, fork_adapter, use_alt_view,
)


def build_example(
    row: dict[str, str], images: Path, target_height: int, max_pixels: int,
    autocrop: bool = False, shape_buckets: bool = False, patch: int = 28,
    alt_images: "Path | None" = None,
):
    from PIL import Image

    image_path = images / f"{row['ID']}.jpg"
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    with Image.open(image_path) as probe:
        height = probe.height
    example = {
        "image": load_line_image(image_path, target_height, max_pixels, autocrop, shape_buckets, patch),
        "target": clean_text(row["Target"]),
        "height": int(height),
    }
    if alt_images is not None:
        alt_path = alt_images / f"{row['ID']}.jpg"
        if not alt_path.is_file():
            raise FileNotFoundError(alt_path)
        example["image_alt"] = load_line_image(alt_path, target_height, max_pixels, autocrop, shape_buckets, patch)
    return example


def with_candidates(example: dict, candidates: "list | None") -> dict:
    # Every row gets both fields: Dataset.from_list takes its schema from the first row and
    # would silently drop them. A real label is its own single candidate.
    candidates = candidates or [[example["target"], 1.0]]
    example["candidates"] = [clean_text(text) for text, _ in candidates]
    example["weights"] = [float(weight) for _, weight in candidates]
    return example


def pick_target(feature: dict, rng: "random.Random | None") -> str:
    """Soft distillation: a pseudo-labelled line carries every member's reading and is
    trained on one of them per visit, drawn by vote support. Real labels have one target."""
    candidates = feature.get("candidates")
    if not candidates or rng is None:
        return feature["target"]
    return rng.choices(candidates, weights=feature["weights"])[0]


class Collator:
    """Tokenize a batch and mask every prompt token out of the loss."""

    def __init__(self, processor, debug: bool = False, verify_targets: bool = False,
                 augment: "random.Random | None" = None, sampler: "random.Random | None" = None,
                 alt_rng: "random.Random | None" = None, alt_prob: float = 0.5):
        self.sampler = sampler
        # Raw/cleaned mix: each visit draws the alternate image with probability alt_prob.
        self.alt_rng, self.alt_prob, self.alt_visits = alt_rng, alt_prob, 0
        self.processor = processor
        self.debug = debug
        self.verify_targets = verify_targets
        self.verified_examples = 0
        # Augmenting here rather than at load time gives every epoch fresh
        # jitter at no extra cost, and keeps the validation path untouched.
        self.augment = augment
        self.augmented_examples = 0
        # Qwen's template closes a turn with "<|im_end|>\n", Kimi-VL's with "<|im_end|>";
        # Kimi-VL's processor also takes one flat image list rather than one list per text.
        self.kimi = type(processor).__name__.startswith("KimiVL")
        self.end = "<|im_end|>" if self.kimi else "<|im_end|>\n"

    def __call__(self, features):
        import torch

        texts, answer_lengths, answer_ids, images = [], [], [], []
        for feature in features:
            user = {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": PROMPT}],
            }
            prompt = self.processor.apply_chat_template(
                [user], tokenize=False, add_generation_prompt=True
            )
            answer = pick_target(feature, self.sampler) + self.end
            texts.append(prompt + answer)
            # Measure the answer, not the prompt. The processor expands the single
            # image placeholder into one token per visual patch, so a prompt length
            # taken before that call is hundreds of tokens short and leaves the
            # image padding inside the loss. The answer is unaffected by expansion.
            answer_lengths.append(
                len(self.processor.tokenizer(answer, add_special_tokens=False)["input_ids"])
            )
            if self.verify_targets:
                answer_ids.append(self.processor.tokenizer(answer, add_special_tokens=False)["input_ids"])
            image = feature["image"]
            if feature.get("image_alt") is not None and use_alt_view(self.alt_rng, self.alt_prob):
                image = feature["image_alt"]
                self.alt_visits += 1
            if self.augment is not None:
                image = augment_line(image, self.augment)
                self.augmented_examples += 1
            images.append(image if self.kimi else [image])

        batch = self.processor(
            text=texts, images=images, return_tensors="pt", padding=True
        )
        mask = batch["attention_mask"]
        labels = batch["input_ids"].clone()
        labels[mask == 0] = -100
        for index, answer_length in enumerate(answer_lengths):
            # Anchor on the last real token so this holds under either padding side.
            end = int(mask[index].nonzero()[-1]) + 1
            labels[index, : end - answer_length] = -100
            labels[index, end:] = -100
            if self.verify_targets:
                kept = labels[index][labels[index] != -100].tolist()
                if not kept or kept != answer_ids[index]:
                    raise ValueError("Loss mask does not contain exactly the transcription and end marker")
                self.verified_examples += 1
        batch["labels"] = labels

        if self.debug:
            kept = labels[0][labels[0] != -100]
            print(
                "loss target of first example: "
                f"{self.processor.tokenizer.decode(kept)!r}",
                flush=True,
            )
            self.debug = False
        return batch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="Train.csv")
    parser.add_argument("--images", default="images")
    parser.add_argument("--fold-manifest", default="data/splits/folds.csv")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--fold", type=int, help="fold held out from training")
    selection.add_argument("--full-data", action="store_true", help="train on all labels; no validation score")
    parser.add_argument("--exclude-fold", type=int,
                        help="also drop this fold from training; the --fold held-out rows are unchanged")
    parser.add_argument("--model-id", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--output", required=True, help="adapter output directory")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lrate", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-scope", choices=["baseline", "vision-attention", "vision-mlp", "kimi"], default="baseline")
    parser.add_argument(
        "--augment", action="store_true",
        help="jitter training crops only: rotation, shear, stroke thickness, "
             "brightness, contrast, blur and JPEG noise, seeded from --seed",
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
        help="NF4-quantize the base weights (QLoRA). Needed for a 7B on a 16GB T4, "
             "where fp16 weights alone exceed the card.",
    )
    parser.add_argument(
        "--autocrop", action="store_true",
        help="trim each crop to its own text band before resizing "
             "(measured worse than doing nothing; kept for reproducibility)",
    )
    parser.add_argument("--max-train", type=int, help="smoke-test cap on training rows")
    parser.add_argument(
        "--extra-train", default="",
        help="ID,Target CSV of extra rows (test pseudo-labels) appended to a full training run; "
             "their images come from --images. Ignored with --max-train, so smokes stay 8 rows.",
    )
    parser.add_argument(
        "--max-source-height", type=int, default=0,
        help="drop training rows whose SOURCE crop is taller than this many pixels; "
             "0 keeps everything. Tests the 'some of the data is bad' hypothesis directly: "
             "loose crops are 26.5%% of rows and carry 48.2%% of held-out character edits. "
             "Validation is never filtered, so the score stays comparable.",
    )
    parser.add_argument("--curriculum", action="store_true",
                        help="train easy to hard: real tight lines, real loose lines, then pseudo-labels "
                             "by vote support (road_ocr.recipe.curriculum_order)")
    parser.add_argument("--experts", action="store_true",
                        help="height-routed LoRA experts: one shared epoch, then the adapter is forked into "
                             "a tight and a loose expert, each trained on its own stratum")
    parser.add_argument("--expert-height", type=int, default=EXPERT_HEIGHT)
    parser.add_argument("--train-images-alt", default="",
                        help="second image directory (e.g. raw scans); each training visit uses it with "
                             "probability --alt-prob. Validation and inference only ever use --images.")
    parser.add_argument("--alt-prob", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument(
        "--precision", choices=["auto", "bf16", "fp16"], default="auto",
        help="auto picks bf16 where the accelerator supports it, else fp16",
    )
    parser.add_argument(
        "--save-steps", type=int, default=0,
        help="save the adapter every N steps; 0 saves only at the end. "
             "Set it on a hosted runtime that can disconnect mid-run.",
    )
    parser.add_argument("--metadata")
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="trade compute for lower activation memory")
    parser.add_argument(
        "--resume-from-checkpoint",
        help="Resume adapter, optimizer, scheduler, and RNG state from a Trainer checkpoint. "
             "Keep --epochs at the original total, not the number of remaining epochs.",
    )
    args = parser.parse_args()
    if args.device == "xla" and not args.shape_buckets:
        parser.error(
            "--device xla requires --shape-buckets: XLA compiles per input shape, and "
            "preserving aspect otherwise produces over a hundred distinct widths"
        )
    if args.experts and args.resume_from_checkpoint:
        parser.error("--experts does not support --resume-from-checkpoint")
    # One epoch has nothing to fork after the shared epoch (the notebook's 8-row smoke), so it trains plain.
    use_experts = args.experts and args.epochs > 1
    resume_state = None
    if args.resume_from_checkpoint:
        resume_state = validate_checkpoint(args.resume_from_checkpoint, resume=True)

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoProcessor,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    rows = read_csv(args.train, ["ID", "Target"])
    index_unique(rows, args.train)
    candidates = None
    # Seed before LoRA initialization, not just inside Trainer.
    set_seed(args.seed)
    train_rows, validation_rows = select_training_rows(
        rows, args.fold_manifest, args.fold, full_data=args.full_data, exclude_fold=args.exclude_fold
    )
    images = Path(args.images)
    if args.max_source_height:
        from PIL import Image

        kept = []
        for row in train_rows:
            with Image.open(images / f"{row['ID']}.jpg") as probe:
                if probe.height <= args.max_source_height:
                    kept.append(row)
        print(
            f"--max-source-height {args.max_source_height}: keeping {len(kept)} of "
            f"{len(train_rows)} training rows; validation is untouched",
            flush=True,
        )
        if not kept:
            raise ValueError("--max-source-height dropped every training row")
        train_rows = kept
    if args.max_train is not None:
        train_rows = train_rows[: args.max_train]
    print(
        f"fold {args.fold}: training on {len(train_rows)} rows, "
        f"holding out {len(validation_rows)}",
        flush=True,
    )

    extra_rows = []
    if args.extra_train and args.max_train is None:
        extra_rows = read_csv(args.extra_train, ["ID", "Target"])
        index_unique(extra_rows, args.extra_train)
        # Pseudo-labels must be test lines: an overlap with Train.csv would let a
        # model-made label override, or duplicate, a real one.
        overlap = {row["ID"] for row in extra_rows} & {row["ID"] for row in rows}
        if overlap:
            raise ValueError(f"--extra-train shares {len(overlap)} IDs with --train")
        held_out = {row["ID"] for row in validation_rows} & {row["ID"] for row in extra_rows}
        if held_out:
            raise ValueError(f"--extra-train contains {len(held_out)} validation rows")
        print(f"--extra-train: adding {len(extra_rows)} rows", flush=True)
        # Optional sidecar from scripts/make_candidate_targets.py: {ID: [[text, weight], ...]}.
        sidecar = Path(args.extra_train).with_suffix(".candidates.json")
        if sidecar.is_file():
            candidates = json.loads(sidecar.read_text(encoding="utf-8"))
            if set(candidates) != {row["ID"] for row in extra_rows}:
                raise ValueError(f"{sidecar} IDs differ from {args.extra_train}")
            print(f"--extra-train: soft targets from {sidecar.name}", flush=True)
    examples = [
        build_example(
            row, images, args.target_height, args.max_pixels,
            args.autocrop, args.shape_buckets, model_patch(args.model_id),
            Path(args.train_images_alt) if args.train_images_alt else None,
        )
        for row in train_rows + extra_rows
    ]
    for example, row in zip(examples, train_rows + extra_rows):
        with_candidates(example, candidates.get(row["ID"]) if candidates else None)
    extra_ids = {row["ID"] for row in extra_rows}
    for example, row in zip(examples, train_rows + extra_rows):
        example["pseudo"] = row["ID"] in extra_ids
        example["support"] = max(example["weights"]) if row["ID"] in extra_ids else 1.0
        example["length"] = len(example["target"])
    meta = [{k: example[k] for k in ("pseudo", "support", "length", "height")} for example in examples]
    dataset = Dataset.from_list(examples)

    device, backend = resolve_device(args.device)
    dtype, precision = resolve_precision(args.precision, backend)
    print(f"device: {backend}", flush=True)
    if args.resume_from_checkpoint and precision == "fp16":
        if not (Path(args.resume_from_checkpoint) / "scaler.pt").is_file():
            raise ValueError("FP16 recovery requires scaler.pt in the checkpoint")
    print(f"precision: {precision}", flush=True)
    kimi = is_kimi(args.model_id)
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=kimi)
    model = load_vlm(
        args.model_id, dtype, load_4bit=args.load_4bit, device=device, device_map="auto"
    )
    if args.load_4bit:
        from peft import prepare_model_for_kbit_training

        # Casts layer norms and the head to fp32 and enables input grads, without
        # which the LoRA gradients do not flow through frozen quantized weights.
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=args.gradient_checkpointing
        )
    model.config.use_cache = False
    if kimi:  # its decoder reads use_cache from text_config, not the top-level config
        model.config.text_config.use_cache = False
    targets, lora_audit = resolve_targets(model, args.lora_scope, args.lora_r)
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    lora_audit["actual_trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if lora_audit["actual_trainable_parameters"] != lora_audit["expected_adapter_parameters"]:
        raise ValueError("Resolved LoRA parameter count differs from the adapted model")
    gradient_audit = (AttentionGradientAudit(model, lora_audit["added_modules"])
                      if args.lora_scope == "vision-attention" else None)
    if use_experts:
        model.add_adapter("loose", lora)
        model.set_adapter("default")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "lora_audit.json").write_text(json.dumps(lora_audit, indent=2) + "\n")
    def make_args(tag, epochs):
      return TrainingArguments(
        output_dir=str(output / f"trainer{tag}"),
        num_train_epochs=epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lrate,
        bf16=precision == "bf16",
        fp16=precision == "fp16",
        logging_steps=25,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 500,
        save_total_limit=1,
        report_to=[],
        seed=args.seed,
        remove_unused_columns=False,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if args.gradient_checkpointing else None,
        # macOS spawns workers, which re-pickles the decoded images per worker and
        # duplicates this collator. MPS gains nothing from it.
        dataloader_num_workers=0,
      )
    collator = Collator(processor, debug=True, verify_targets=gradient_audit is not None or kimi,
                        augment=random.Random(args.seed) if args.augment else None,
                        sampler=random.Random(args.seed + 1) if candidates else None,
                        alt_rng=random.Random(args.seed + 2) if args.train_images_alt else None,
                        alt_prob=args.alt_prob)

    class OrderSampler(torch.utils.data.Sampler):
        # Fixed easy-to-hard order per epoch, reseeded each epoch; the Trainer calls __iter__ once per epoch.
        def __init__(self, sub_meta):
            self.meta, self.epoch = sub_meta, 0

        def __iter__(self):
            order = curriculum_order(self.meta, args.seed, self.epoch)
            self.epoch += 1
            return iter(order)

        def __len__(self):
            return len(self.meta)

    class RecipeTrainer(Trainer):
        sub_meta = None

        def _get_train_sampler(self, *a, **k):
            return OrderSampler(self.sub_meta) if self.sub_meta is not None else super()._get_train_sampler(*a, **k)

    def run_phase(tag, adapter, indices, epochs):
        if adapter:
            model.set_adapter(adapter)
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if trainable != lora_audit["expected_adapter_parameters"]:
                raise ValueError(f"adapter {adapter!r}: {trainable} trainable parameters, expected "
                                 f"{lora_audit['expected_adapter_parameters']}")
        trainer = RecipeTrainer(model=model, args=make_args(tag, epochs),
                                train_dataset=dataset.select(indices), data_collator=collator)
        if args.curriculum:
            trainer.sub_meta = [meta[i] for i in indices]
        print(f"phase {tag or 'train'}: adapter={adapter or 'default'} rows={len(indices)} epochs={epochs}", flush=True)
        result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        loss = float(result.training_loss)
        if not math.isfinite(loss):
            raise ValueError(f"Nonfinite training loss: {loss}")
        return loss, len(indices) * epochs

    everyone = list(range(len(examples)))
    started = time.time()
    if use_experts:
        # One shared epoch, then fork the adapter into a tight and a loose expert.
        phases = [("_shared", "default", everyone, 1)]
        loss_parts = [run_phase(*phases[0])]
        print(f"forked {fork_adapter(model)} LoRA tensors into the loose expert", flush=True)
        tight = [i for i in everyone if meta[i]["height"] < args.expert_height]
        loose = [i for i in everyone if meta[i]["height"] >= args.expert_height]
        if not tight or not loose:
            raise ValueError("--experts needs training rows on both sides of --expert-height")
        loss_parts += [run_phase("_tight", "default", tight, args.epochs - 1),
                       run_phase("_loose", "loose", loose, args.epochs - 1)]
    else:
        loss_parts = [run_phase("", None, everyone, args.epochs)]
    runtime = time.time() - started
    train_loss = sum(l * w for l, w in loss_parts) / sum(w for _, w in loss_parts)
    gradient_report = gradient_audit.report() if gradient_audit else None
    if gradient_audit:
        gradient_audit.close()
        (output / "attention_gradients.json").write_text(json.dumps(gradient_report, indent=2) + "\n")
        if not gradient_report["passed"]:
            raise ValueError("Added visual attention adapters have missing, zero, or nonfinite gradients")

    model.save_pretrained(str(output))
    processor.save_pretrained(str(output))
    print(f"saved adapter to {output} after {runtime/60:.1f} min", flush=True)

    metadata = {
        "command": " ".join(shlex.quote(value) for value in sys.argv),
        "epochs": args.epochs,
        "fold": args.fold,
        "exclude_fold": args.exclude_fold,
        "full_data": args.full_data,
        "gradient_checkpointing": args.gradient_checkpointing,
        "batch_size": args.batch_size,
        "train_loss": train_loss,
        "grad_accum": args.grad_accum,
        "lora_alpha": args.lora_alpha,
        "lora_r": args.lora_r,
        "lora_scope": args.lora_scope,
        "augment": args.augment,
        "augmented_examples": collator.augmented_examples,
        "lora_audit": lora_audit,
        "attention_gradients": gradient_report,
        "loss_mask_verified_examples": collator.verified_examples,
        "lrate": args.lrate,
        "autocrop": args.autocrop,
        "max_source_height": args.max_source_height,
        "device": backend,
        "load_4bit": args.load_4bit,
        "shape_buckets": args.shape_buckets,
        "max_pixels": args.max_pixels,
        "model_id": args.model_id,
        "output": str(output),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "runtime_seconds": runtime,
        "resume_from_checkpoint": args.resume_from_checkpoint,
        "resumed_global_step": resume_state["global_step"] if resume_state else None,
        "precision": precision,
        "seed": args.seed,
        "target_height": args.target_height,
        "torch_version": torch.__version__,
        "train_rows": len(train_rows),
        "train_ids_sha256": hashlib.sha256(
            "\n".join(row["ID"] for row in train_rows).encode()
        ).hexdigest(),
        "validation_rows": len(validation_rows),
        "extra_train_rows": len(extra_rows),
        "curriculum": args.curriculum,
        "experts": use_experts,
        "expert_height": args.expert_height if use_experts else None,
        "train_images_alt": args.train_images_alt or None,
        "alt_visits": collator.alt_visits,
        "phase_losses": [loss for loss, _ in loss_parts],
        "soft_targets": bool(candidates),
        "extra_train_sha256": hashlib.sha256(
            "\n".join(f"{row['ID']}\t{row['Target']}" for row in extra_rows).encode()
        ).hexdigest() if extra_rows else None,
    }
    if args.metadata:
        metadata_path = Path(args.metadata)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
