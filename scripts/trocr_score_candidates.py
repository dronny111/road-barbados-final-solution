#!/usr/bin/env python3
"""Teacher-forced NLL of every MBR candidate under a fine-tuned TrOCR model.

The pool per line is exactly what ``vote_predictions.mbr_vote(..., with_word_vote=True)``
compares (``candidate_pools``), so ``--scores`` can rerank it. Each candidate's score is its
total negative log-likelihood given the line image: every label token, end-of-sequence
included, with labels built as the fine-tuning collator builds them (no truncation). The
image is preprocessed as trocr_infer.py does and encoded once per line.

Output: JSON {ID: {candidate: NLL}}. ``--own-member K`` reports how often member K's reading
(the scorer's own greedy decode, when K is the scorer) gets the lowest NLL: a sanity check
that scoring matches decoding.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from road_ocr.lines import resolve_device  # noqa: E402
from road_ocr.trocr_model import load_trocr  # noqa: E402
from vote_predictions import candidate_pools, read_predictions  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", nargs="+", help="member CSVs, in the vote's order (first is the pivot)")
    parser.add_argument("--model", required=True, help="fine-tuned TrOCR model directory")
    parser.add_argument("--images", default="images")
    parser.add_argument("--device", default="cpu", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--own-member", type=int, help="index of the scorer's own member CSV")
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata")
    args = parser.parse_args()

    import torch
    from PIL import Image
    from transformers.modeling_outputs import BaseModelOutput

    members = [read_predictions(p) for p in args.predictions]
    pools = candidate_pools(members, with_word_vote=True)[0]
    ids = list(pools)[: args.max_samples]
    device, backend = resolve_device(args.device)
    processor, model = load_trocr(args.model, device, torch.float32)
    model.eval()
    tokenizer = processor.tokenizer

    started, scores, own_best, contested = time.time(), {}, 0, 0
    for n, record_id in enumerate(ids, 1):
        candidates = pools[record_id]
        with Image.open(Path(args.images) / f"{record_id}.jpg") as image:
            pixel_values = processor(images=[image.convert("RGB")], return_tensors="pt").pixel_values.to(device)
        labels = tokenizer(candidates, padding="longest", return_tensors="pt").input_ids.to(device)
        labels[labels == tokenizer.pad_token_id] = -100
        with torch.inference_mode():
            hidden = model.encoder(pixel_values=pixel_values).last_hidden_state
            enc = BaseModelOutput(last_hidden_state=hidden.expand(len(candidates), -1, -1))
            logp = model(encoder_outputs=enc, labels=labels).logits.float().log_softmax(-1)
        mask = labels != -100
        token_logp = logp.gather(-1, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        nll = (-(token_logp * mask).sum(-1)).tolist()
        scores[record_id] = dict(zip(candidates, nll, strict=True))
        if args.own_member is not None and len(candidates) > 1:
            contested += 1
            own_best += min(candidates, key=scores[record_id].get) == members[args.own_member][record_id]
        if n % 50 == 0 or n == len(ids):
            print(f"scored {n}/{len(ids)} ({(time.time() - started) / n:.2f} s/line)", flush=True)
    runtime = time.time() - started

    with open(args.output, "x", encoding="utf-8") as handle:  # never overwrite a recorded run
        json.dump(scores, handle, ensure_ascii=False)
    metadata = {
        "command": " ".join(sys.argv), "members": args.predictions, "model": args.model,
        "device": backend, "rows": len(ids), "candidates": sum(map(len, scores.values())),
        "runtime_seconds": runtime, "seconds_per_line": runtime / len(ids),
        "own_member": args.own_member, "contested_rows": contested,
        "own_reading_lowest_nll": own_best / contested if contested else None,
        "platform": platform.platform(), "torch_version": torch.__version__, "output": args.output,
    }
    if args.metadata:
        Path(args.metadata).write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
