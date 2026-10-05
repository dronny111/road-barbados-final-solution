# TrOCR-large (Kansallisarkisto multicentury HTR), fold 0, Kaggle T4

**Why.** Model-family diversity has given the largest vote gains (+0.0039 for the vmlp4B over
the second 7B; MBR over three families gave 0.909303). This adds a fourth, non-VLM family: an
encoder-decoder already fine-tuned on **913,000 historical handwritten lines** (16th-20th
century, Swedish and Finnish; CER 2.8 on its own test set). The license is Apache-2.0 and it
is permitted under ruling 34734 (see `docs/COMPETITION.md`).

**Hypothesis.** Added to the fold-0 MBR pool (pseudo 7B + vmlp2B + vmlp4B), it lowers
`--mode mbr` below **0.09837**. The single-model gate is below **0.12437** (the vmlp4B, the weakest
current member).

**What runs.** `MODEL_FAMILY = "trocr"` makes the shared notebook call `scripts/trocr_finetune.py`
and `scripts/trocr_infer.py`. They use the same command line, fold selection, row hash, loss
marker and metadata as the VLM scripts, so the audit, smoke, budget gate, fold-0 scoring and
submission checks are unchanged.
- **Setup:** full fine-tune at the processor's 192x1024; the encoder interpolates position
  embeddings (`road_ocr/trocr_model.py`, checked by `tests/test_trocr_model.py`).
- **Recipe:** learning rate 2e-5 (the card's), 6 epochs, batch 8 x 2, fp16 AMP, checkpointing.
- **Decoding:** greedy with repetition ban 4.
- **Revision:** `794fd62b8b87df39b39530c37ad737eedbb12c2b`.
- **Not tuned:** epochs and learning rate are fixed in advance, with no search.

**Compute.** Unmeasured; the smoke estimate gates it at 9 h. The expectation is a few hours
on one T4.

```bash
python3 scripts/build_kaggle_notebook.py --config baselines/vlm/runs_trocr_multicentury_kaggle/config.json --output baselines/vlm/runs_trocr_multicentury_kaggle/kaggle_trocr_multicentury_fold0.ipynb
.venv-vlm/bin/kaggle kernels push -p baselines/vlm/runs_trocr_multicentury_kaggle --timeout 36000 --accelerator NvidiaTeslaT4
.venv-vlm/bin/kaggle kernels output <your-kaggle-user>/road-trocr-multicentury-fold0 -p experiments/runs/trocr_mc_small \
  --file-pattern '(validation_predictions|validation_reference|test_predictions|run_config|submission|commands)\.(csv|json|jsonl)$'
```
