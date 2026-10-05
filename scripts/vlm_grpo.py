#!/usr/bin/env python3
"""Continue a Qwen2-VL LoRA adapter with GRPO against the competition metric.

Supervised fine-tuning optimizes next-token likelihood, which is not the thing
the leaderboard scores. GRPO samples several transcriptions per line and pushes
the policy toward the ones with the lower ``0.5 * CER + 0.5 * WER``, so the
training signal is the metric itself.

This continues an existing SFT adapter rather than starting a new one: RL from a
cold policy on a 2B model wastes its budget rediscovering what SFT already knows.
Fold discipline is unchanged -- rows held out for validation are never sampled.
"""

from __future__ import annotations

import argparse
import functools
import json
import platform
import shlex
import sys
import time
from pathlib import Path

from road_ocr.lines import (
    PROMPT,
    clean_text,
    load_line_image,
    load_vlm,
    resolve_device,
    resolve_precision,
    select_training_rows,
)
from road_ocr.metrics import combined_error, edit_distance
from road_ocr.records import index_unique, read_csv


def completion_text(completion) -> str:
    """Pull the assistant text out of either a chat completion or a raw string."""

    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        content = completion[-1].get("content", "")
        if isinstance(content, list):
            return "".join(part.get("text", "") for part in content)
        return content
    return ""


def transcription_reward(completions, target, **kwargs):
    """Reward in [0, 1]; 1 is an exact transcription.

    Both sides are whitespace-normalized because inference emits normalized text,
    so the raw label's stray double spaces are noise the policy cannot act on.
    A line worse than "completely wrong" earns 0 rather than a negative reward,
    which keeps a runaway repetition loop from dominating its group's advantage.
    """

    rewards = []
    for completion, label in zip(completions, target, strict=True):
        prediction = clean_text(completion_text(completion))
        reference = clean_text(label)
        rewards.append(max(0.0, 1.0 - combined_error(reference, prediction)))
    return rewards


def corpus_edits_reward(completions, target, mean_chars, mean_words, **kwargs):
    """Minus each sample's contribution to the micro-averaged corpus error; 0 is exact.

    The leaderboard sums edits over the corpus, so a long line weighs more than a
    short one. ``transcription_reward`` is a per-line ratio, and TRL's default
    per-group std scaling then gives every line equal pull; this keeps the scale
    (pair it with ``--scale-rewards none``). ``mean_chars`` and ``mean_words``
    come from the GRPO training rows only, so the reward is in "average line"
    units. Unclipped on purpose: a loop really does cost that many edits, and
    ``--max-completion-length`` bounds it.
    """

    rewards = []
    for completion, label in zip(completions, target, strict=True):
        prediction = clean_text(completion_text(completion))
        reference = clean_text(label)
        char_edits = edit_distance(reference, prediction)
        word_edits = edit_distance(reference.split(), prediction.split())
        rewards.append(-(0.5 * char_edits / mean_chars + 0.5 * word_edits / mean_words))
    return rewards


def build_dataset(rows, images: Path, args):
    """Build the GRPO dataset with the same prompt the SFT adapter was trained on.

    ``datasets`` unifies the content list's schema and adds a ``text: None`` key to
    the image part. That was checked against the processor's chat template and the
    rendered prompt is byte-identical to the one ``vlm_infer.py`` builds, so the
    adapter sees exactly the prompt it was tuned for.
    """

    from datasets import Dataset

    from PIL import Image

    records = []
    for row in rows:
        path = images / f"{row['ID']}.jpg"
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as probe:
            height = probe.height
        records.append(
            {
                "height": int(height),
                "prompt": [
                    {
                        "role": "user",
                        "content": [{"type": "image"}, {"type": "text", "text": PROMPT}],
                    }
                ],
                "image": load_line_image(path, args.target_height, args.max_pixels, args.autocrop),
                "target": clean_text(row["Target"]),
            }
        )
    return Dataset.from_list(records)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="Train.csv")
    parser.add_argument("--images", default="images")
    parser.add_argument("--model-id", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--processor")
    parser.add_argument("--adapter", required=True, help="SFT adapter to continue")
    parser.add_argument("--output", required=True)
    parser.add_argument("--fold-manifest")
    parser.add_argument("--fold", type=int)
    parser.add_argument("--full-data", action="store_true")
    parser.add_argument("--max-train", type=int, help="cap on training rows; use for smokes")
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.04, help="KL weight to the SFT policy")
    parser.add_argument("--lrate", type=float, default=1e-5)
    parser.add_argument(
        "--reward", choices=["clipped_combined", "corpus_edits"], default="clipped_combined",
        help="corpus_edits weights each line by its share of corpus error",
    )
    parser.add_argument(
        "--scale-rewards", choices=["group", "batch", "none"], default="group",
        help="TRL advantage scaling; corpus_edits needs none to keep its weights",
    )
    parser.add_argument("--max-completion-length", type=int, default=96)
    parser.add_argument("--target-height", type=int, default=112)
    parser.add_argument("--max-pixels", type=int, default=451_584)
    parser.add_argument(
        "--load-4bit", action="store_true",
        help="NF4-quantize the base weights; must match how the adapter was trained",
    )
    parser.add_argument("--autocrop", action="store_true")
    parser.add_argument("--experts", action="store_true",
                        help="--adapter holds height-routed experts (root = tight, loose/ = loose): run one "
                             "GRPO phase per expert on its own height stratum and save both")
    parser.add_argument("--expert-height", type=int, default=150)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cuda", "mps", "cpu", "xla"],
        help="xla targets a TPU and requires torch_xla",
    )
    parser.add_argument(
        "--shape-buckets", action="store_true",
        help="pad rendered widths onto a few fixed buckets; required for XLA/TPU",
    )
    parser.add_argument("--precision", choices=["auto", "bf16", "fp16"], default="auto")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--save-steps", type=int, default=0)
    parser.add_argument("--metadata")
    args = parser.parse_args()
    if (args.fold_manifest is None) != (args.fold is None):
        parser.error("--fold-manifest and --fold must be provided together")
    if args.batch_size * args.grad_accum % args.num_generations:
        parser.error(
            "--num-generations must divide --batch-size * --grad-accum so every "
            "group of samples comes from one line"
        )

    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, set_seed
    from trl import GRPOConfig, GRPOTrainer

    rows = read_csv(args.train, ["ID", "Target"])
    index_unique(rows, args.train)
    set_seed(args.seed)
    train_rows, validation_rows = select_training_rows(
        rows, args.fold_manifest, args.fold, full_data=args.full_data
    )
    if args.max_train is not None:
        train_rows = train_rows[: args.max_train]
    print(
        f"fold {args.fold}: GRPO on {len(train_rows)} rows, holding out {len(validation_rows)}",
        flush=True,
    )

    device, backend = resolve_device(args.device)
    if backend == "xla" and not args.shape_buckets:
        parser.error(
            "--device xla requires --shape-buckets: XLA compiles per input shape and "
            "this recipe otherwise produces over a hundred distinct widths"
        )
    dtype, precision = resolve_precision(args.precision, backend)
    print(f"device: {backend}", flush=True)
    print(f"precision: {precision}", flush=True)
    processor = AutoProcessor.from_pretrained(args.processor or args.model_id)
    model = load_vlm(
        args.model_id, dtype, load_4bit=args.load_4bit, device=device, device_map="auto"
    )
    # Continue the SFT adapter in place. GRPOTrainer takes the reference policy
    # from the same model with the adapter disabled, so no second copy is loaded.
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)
    if args.experts:
        loose_dir = Path(args.adapter) / "loose"
        if not loose_dir.is_dir():
            parser.error(f"--experts needs {loose_dir}; train the adapter with vlm_finetune.py --experts")
        model.load_adapter(loose_dir, adapter_name="loose", is_trainable=True)
    model.config.use_cache = False

    dataset = build_dataset(train_rows, Path(args.images), args)
    reward_normalizers = None
    reward_func = transcription_reward
    if args.reward == "corpus_edits":
        reward_normalizers = {
            "mean_chars": sum(len(t) for t in dataset["target"]) / len(dataset),
            "mean_words": sum(len(t.split()) for t in dataset["target"]) / len(dataset),
        }
        reward_func = functools.partial(corpus_edits_reward, **reward_normalizers)
        # TRL logs rewards under the function's __name__, which a partial lacks.
        reward_func.__name__ = corpus_edits_reward.__name__
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    phases = [(None, list(range(len(dataset))))]
    if args.experts:
        heights = dataset["height"]
        phases = [("default", [i for i, h in enumerate(heights) if h < args.expert_height]),
                  ("loose", [i for i, h in enumerate(heights) if h >= args.expert_height])]
        if not all(indices for _, indices in phases):
            parser.error("--experts needs GRPO rows on both sides of --expert-height; raise --max-train")
    all_metrics = []
    started = time.time()
    for name, indices in phases:
        if name:
            model.set_adapter(name)  # only the active expert trains; the other stays frozen
            print(f"GRPO phase: expert={name} rows={len(indices)}", flush=True)
        config = GRPOConfig(
            output_dir=str(output / (f"trainer_{name}" if name else "")),
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            num_generations=args.num_generations,
            temperature=args.temperature,
            beta=args.beta,
            scale_rewards=args.scale_rewards,
            learning_rate=args.lrate,
            max_completion_length=args.max_completion_length,
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
            seed=args.seed,
            bf16=(precision == "bf16"),
            fp16=(precision == "fp16"),
            gradient_checkpointing=args.gradient_checkpointing,
            save_steps=args.save_steps or 500,
            save_strategy="steps" if args.save_steps else "no",
            logging_steps=1,
            report_to=[],
            # The reward needs the label, and "target" is not one of TRL's signature
            # columns ("prompt", "image", "images"). TRL already defaults this to
            # False for GRPO so the extra column survives, but the reward silently
            # loses its labels if that default ever changes, so pin it here.
            remove_unused_columns=False,
        )
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=reward_func,
            args=config,
            train_dataset=dataset.select(indices),
            processing_class=processor,
        )
        all_metrics.append(trainer.train().metrics)
    runtime = time.time() - started
    result_metrics = all_metrics[0] if len(all_metrics) == 1 else all_metrics
    trainer.model.save_pretrained(output / "adapter")  # saves every adapter: root + loose/
    processor.save_pretrained(output / "adapter")

    metadata = {
        "adapter_in": args.adapter,
        "autocrop": args.autocrop,
        "device": backend,
        "load_4bit": args.load_4bit,
        "beta": args.beta,
        "reward": args.reward,
        "reward_normalizers": reward_normalizers,
        "scale_rewards": args.scale_rewards,
        "command": " ".join(shlex.quote(value) for value in sys.argv),
        "fold": args.fold,
        "full_data": args.full_data,
        "max_pixels": args.max_pixels,
        "model_id": args.model_id,
        "num_generations": args.num_generations,
        "output": str(output),
        "platform": platform.platform(),
        "precision": precision,
        "python": platform.python_version(),
        "runtime_seconds": runtime,
        "seed": args.seed,
        "target_height": args.target_height,
        "temperature": args.temperature,
        "torch_version": torch.__version__,
        "train_rows": len(train_rows),
        "train_metrics": result_metrics,
        "experts": args.experts,
        "validation_rows": len(validation_rows),
    }
    if args.metadata:
        path = Path(args.metadata)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result_metrics, indent=2, sort_keys=True))
    print(f"saved GRPO adapter to {output / 'adapter'} in {runtime:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
