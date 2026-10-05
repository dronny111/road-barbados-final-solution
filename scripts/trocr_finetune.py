#!/usr/bin/env python3
"""Fully fine-tune a TrOCR checkpoint on the frozen folds; the TrOCR twin of vlm_finetune.py.

Same command line, row selection, row hash, loss-target log marker and metadata keys as
vlm_finetune.py, so the notebook's audit, smoke, budget and fold checks apply unchanged.
VLM-only options (--target-height, --max-pixels, --lora-*) are accepted and recorded but
have no effect: TrOCR is fully fine-tuned at its processor's fixed input size.
--load-4bit, --augment and --max-source-height are refused rather than silently ignored.
No intermediate checkpoints are written (a full TrOCR-large checkpoint with optimizer
state is about 6.7 GB); the final model is saved to --output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from road_ocr.lines import clean_text, resolve_device, select_training_rows  # noqa: E402
from road_ocr.records import index_unique, read_csv  # noqa: E402
from road_ocr.trocr_model import load_trocr  # noqa: E402


class Collator:
    def __init__(self, processor, max_length, debug=True):
        self.processor, self.max_length, self.debug = processor, max_length, debug

    def __call__(self, features):
        from PIL import Image

        images = []
        for feature in features:
            with Image.open(feature["image"]) as image:
                images.append(image.convert("RGB"))
        pixel_values = self.processor(images=images, return_tensors="pt").pixel_values
        tokenizer = self.processor.tokenizer
        labels = tokenizer([f["target"] for f in features], padding="longest", truncation=True,
                           max_length=self.max_length, return_tensors="pt").input_ids
        labels[labels == tokenizer.pad_token_id] = -100
        if self.debug:
            kept = labels[0][labels[0] != -100]
            print(f"loss target of first example: {tokenizer.decode(kept, skip_special_tokens=True)!r}",
                  flush=True)
            self.debug = False
        return {"pixel_values": pixel_values, "labels": labels}


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
    parser.add_argument("--model-id", required=True, help="local TrOCR checkpoint directory")
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lrate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--max-train", type=int)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--extra-train", default="")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--aux-loss", choices=["none", "ctc"], default="none")
    parser.add_argument("--aux-weight", type=float, default=0.3)
    parser.add_argument("--aux-head-lrate", type=float, default=1e-3,
                        help="learning rate of the freshly initialised aux heads (the model keeps --lrate)")
    parser.add_argument("--recon-weight", type=float, default=0.0,
                        help="text-to-visual-frame reconstruction loss weight (needs --aux-loss ctc)")
    parser.add_argument("--align-weight", type=float, default=0.0,
                        help="last-layer cross-attention entropy weight (needs --aux-loss ctc)")
    parser.add_argument("--save-steps", type=int, default=0, help="accepted for interface parity; unused")
    # Accepted for interface parity with vlm_finetune.py; recorded, no effect on TrOCR.
    for name in ("--target-height", "--max-pixels", "--lora-r", "--lora-alpha"):
        parser.add_argument(name, type=int)
    parser.add_argument("--lora-scope", default="baseline")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--max-source-height", type=int, default=0)
    args = parser.parse_args()
    for flag, on in (("--load-4bit", args.load_4bit), ("--augment", args.augment),
                     ("--max-source-height", args.max_source_height)):
        if on:
            parser.error(f"{flag} is not supported for TrOCR")
    if (args.recon_weight or args.align_weight) and args.aux_loss != "ctc":
        parser.error("--recon-weight/--align-weight build on --aux-loss ctc")

    import torch
    from transformers import Trainer, TrainingArguments, set_seed

    rows = read_csv(args.train, ["ID", "Target"])
    index_unique(rows, args.train)
    set_seed(args.seed)
    train_rows, validation_rows = select_training_rows(
        rows, args.fold_manifest, args.fold, full_data=args.full_data, exclude_fold=args.exclude_fold)
    if args.max_train is not None:
        train_rows = train_rows[: args.max_train]
    extra_rows = []
    if args.extra_train and args.max_train is None:
        extra_rows = read_csv(args.extra_train, ["ID", "Target"])
        index_unique(extra_rows, args.extra_train)
        overlap = {r["ID"] for r in extra_rows} & {r["ID"] for r in rows}
        if overlap:
            raise ValueError(f"--extra-train shares {len(overlap)} IDs with --train")
    print(f"fold {args.fold}: training on {len(train_rows)} rows (+{len(extra_rows)} extra), "
          f"holding out {len(validation_rows)}", flush=True)
    images = Path(args.images)
    examples = []
    for row in train_rows + extra_rows:
        path = images / f"{row['ID']}.jpg"
        if not path.is_file():  # cleaned sets (scripts/clean_images.py) are PNG
            path = path.with_suffix(".png")
        if not path.is_file():
            raise FileNotFoundError(path)
        examples.append({"image": str(path), "target": clean_text(row["Target"])})

    device, backend = resolve_device(args.device)
    if args.precision == "auto":
        precision = "fp16" if backend == "cuda" and torch.cuda.get_device_capability()[0] < 8 else (
            "bf16" if backend == "cuda" else "fp32")
    else:
        precision = args.precision
    print(f"device: {backend}\nprecision: {precision}", flush=True)
    # Full fine-tune: fp32 master weights, mixed-precision compute.
    processor, model = load_trocr(args.model_id, device)

    from torch.utils.data import Dataset

    class Rows(Dataset):
        def __len__(self):
            return len(examples)

        def __getitem__(self, index):
            return examples[index]

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(output / "trainer"), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lrate, fp16=precision == "fp16", bf16=precision == "bf16",
        logging_steps=25, save_strategy="no", report_to=[], seed=args.seed,
        remove_unused_columns=False, gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if args.gradient_checkpointing else None,
        dataloader_num_workers=0, warmup_steps=0.05, lr_scheduler_type="linear",  # 5% warmup (a float is a ratio)
    )
    trainer_cls = Trainer
    if args.aux_loss == "ctc":
        from road_ocr.ctc_aux import CtcHead, ReconHead, alignment_entropy, ctc_loss, recon_loss
        tokenizer = processor.tokenizer
        head = CtcHead(model.config.encoder.hidden_size, model.config.decoder.vocab_size)
        special = list(tokenizer.all_special_ids)
        patch = model.config.encoder.patch_size
        patch = patch if isinstance(patch, int) else patch[0]
        image_h, image_w = processor.image_processor.size["height"], processor.image_processor.size["width"]
        # Built only when used, so the CTC-only run's RNG stream (head init, then Trainer) is unchanged.
        recon = (ReconHead(model.config.decoder.d_model, frames=image_w // patch)
                 if args.recon_weight else None)
        heads = [head] + ([recon] if recon is not None else [])

        class CtcTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                pixel_values, labels = inputs["pixel_values"], inputs["labels"]
                enc = model.encoder(pixel_values=pixel_values)
                out = model(encoder_outputs=enc, labels=labels, output_attentions=bool(args.align_weight))
                h, w = pixel_values.shape[-2] // patch, pixel_values.shape[-1] // patch
                assert (h * patch, w * patch) == (image_h, image_w), "unexpected input size"
                hidden = enc.last_hidden_state
                loss = out.loss + args.aux_weight * ctc_loss(head, hidden, labels, h, w, special)
                if recon is not None:
                    loss = loss + args.recon_weight * recon_loss(
                        recon, model.decoder.get_input_embeddings(), hidden, labels, h, w,
                        tokenizer.pad_token_id)
                if args.align_weight:
                    loss = loss + args.align_weight * alignment_entropy(out.cross_attentions[-1], labels)
                return (loss, out) if return_outputs else loss

            def create_optimizer(self, *a, **k):
                # Same optimizer as the control for the model; the random head gets its own
                # group, since at 2e-5 it barely moves and would feed the encoder noise.
                optimizer = super().create_optimizer(*a, **k)
                device = next(self.model.parameters()).device  # Trainer placed the model; follow it
                params = [p for module in heads for p in module.to(device).parameters()]
                optimizer.add_param_group({"params": params, "lr": args.aux_head_lrate, "weight_decay": 0.0})
                return optimizer

        trainer_cls = CtcTrainer
    trainer = trainer_cls(model=model, args=training_args, train_dataset=Rows(),
                          data_collator=Collator(processor, args.max_length))
    started = time.time()
    result = trainer.train()
    runtime = time.time() - started
    train_loss = float(result.training_loss)
    if not math.isfinite(train_loss):
        raise ValueError(f"Nonfinite training loss: {train_loss}")
    if args.aux_loss == "ctc":
        torch.save(head.state_dict(), output / "ctc_head.pt")  # not part of the checkpoint
        if recon is not None:
            torch.save(recon.state_dict(), output / "recon_head.pt")
    trainer.save_model(str(output))
    processor.save_pretrained(str(output))

    metadata = {
        "family": "trocr", "model_id": args.model_id, "command": " ".join(sys.argv),
        "device": backend, "precision": precision, "seed": args.seed,
        "epochs": args.epochs, "batch_size": args.batch_size, "grad_accum": args.grad_accum,
        "learning_rate": args.lrate, "max_length": args.max_length,
        "gradient_checkpointing": args.gradient_checkpointing,
        "aux_loss": args.aux_loss, "aux_weight": args.aux_weight, "aux_head_lrate": args.aux_head_lrate,
        "recon_weight": args.recon_weight, "align_weight": args.align_weight,
        "fold": args.fold, "exclude_fold": args.exclude_fold, "full_data": args.full_data, "train_loss": train_loss,
        "runtime_seconds": runtime, "platform": platform.platform(),
        "torch_version": torch.__version__,
        "train_rows": len(train_rows),
        "train_ids_sha256": hashlib.sha256("\n".join(r["ID"] for r in train_rows).encode()).hexdigest(),
        "validation_rows": len(validation_rows),
        "extra_train_rows": len(extra_rows),
        "ignored_vlm_options": {"target_height": args.target_height, "max_pixels": args.max_pixels,
                                "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
                                "lora_scope": args.lora_scope},
    }
    if args.metadata:
        path = Path(args.metadata)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"train_loss": train_loss, "runtime_seconds": runtime}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
