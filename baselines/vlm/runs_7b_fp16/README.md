# Qwen2.5-VL-7B fp16 across two T4s, canonical fold 0

**Why.** Our best upload is the 7B NF4 fold-0 adapter (public LB 0.893432). A public
7B LoRA trained in bf16 (`ModarIbrahim/road-qwen25vl-lora`) reports 0.90413 on its model
card; we have not reproduced that. Of that recipe's six differences from ours, quantization
is the one most likely to cost accuracy, and the only one that free hardware can isolate.
bf16 is impossible on a T4 (it has no bf16 units), so this tests unquantized fp16 instead.
It is still the closest free-tier match to bf16.

**Hypothesis.** Removing NF4 quantization reduces complete fold-0 combined error by at
least 0.005 against the 7B NF4 fold-0 result, **0.11439015522447366**
(`kaggle_qwen2vl_20260922T023624Z_32dd77ff`). Success requires combined
**<= 0.10939015522447366** on all 818 held-out rows. The 3B fp16 baseline 0.1123217190 is
context only.

**Controlled change.** Base-weight precision only: `LOAD_4BIT` true -> false, with
`GPU_INDEX = "0,1"` so `device_map="auto"` can shard the 16.6 GB of fp16 weights across
Kaggle's two T4s. The layers run one GPU at a time (naive model parallelism). This
changes where the model runs, not what it computes.

**Fixed controls**, copied unchanged from `../runs_7b/config.json`: model revision
`cc594898137f460bfe9f0759e9844b3ce807cfb5`, seed 20260906, 3 epochs, batch 2 x
accumulation 8, learning rate 1e-4, LoRA 16/32, height 112, 451,584 pixels, gradient
checkpointing, greedy 96-token decoding with repetition ban 4, frozen folds, prompt, and
scorer.

**Compute estimate (unmeasured).** The NF4 run took 9.18 h. fp16 skips dequantization but
moves activations between the two GPUs, so expect 6-11 h. The smoke test's estimate is
gated at `MAX_ESTIMATED_HOURS = 9.0`, with a 36,000 s kernel timeout.

**Stop conditions.** Any of these stops the run; do not relax the gate or relaunch
automatically:
- a failed audit or smoke test
- a nonfinite loss (fp16 overflow in the 7B is a known risk)
- a device-placement error across the two GPUs
- OOM
- a smoke estimate above 9 h
- 3 completed epochs

**Not in this run.** Full-data training is deliberately off, because it cannot be
validated and would not fit the same session. It follows only if the gate passes. Height
256 and the larger pixel budget stay out: raising height alone was disproved at the 3B
(see the config comment in `scripts/build_kaggle_notebook.py`).

```bash
make preflight
python3 scripts/build_kaggle_notebook.py --config baselines/vlm/runs_7b_fp16/config.json --output baselines/vlm/runs_7b_fp16/kaggle_7b_fp16_fold0.ipynb
.venv-vlm/bin/kaggle kernels push -p baselines/vlm/runs_7b_fp16 --timeout 36000 --accelerator NvidiaTeslaT4
.venv-vlm/bin/kaggle kernels status <your-kaggle-user-2>/road-7b-fp16-fold0
```

Build the notebook from a clean commit so its source snapshot is marked `dirty: false`.
