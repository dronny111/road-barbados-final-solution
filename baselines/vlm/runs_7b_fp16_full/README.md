# Qwen2.5-VL-7B fp16 across two T4s, trained on all 4,098 labels

**Why.** The 7B fp16 fold-0 adapter is our best upload, 0.894223196
(`kaggle_qwen2vl_20260926T005721Z_c645047a`). Its fold model saw 3,280 rows. The final
submission should use every label; the public 7B recipe that reports 0.90413 trains on all
of them.

**Hypothesis.** Training on all 4,098 labels instead of fold 0's 3,280 improves the public
leaderboard over 0.894223196. **This cannot be verified locally**, because a full-data
model has no held-out rows. The leaderboard confirms it, one upload.

**Controlled change.** Training rows only. `TRAIN_FULL_DATA` and the new `FULL_DATA_ONLY`
skip the fold model and its validation. Every recipe value is copied from
`../runs_7b_fp16/config.json`: fp16 sharded across two T4s, seed 20260906, 3 epochs,
batch 2 x accumulation 8, learning rate 1e-4, LoRA 16/32, height 112, 451,584 pixels,
gradient checkpointing, greedy 96-token decoding with repetition ban 4.

**Compute (projected from measurement, not measured).** The fold run trained 9,840
row-epochs in 17,924 s (1.82 s each), so 12,294 row-epochs take about 6.2 h. Test inference
took 1.0 h and setup about 8 min: **about 7.4 h in total**. The 8-row smoke estimate is
noisy, so the gate is raised to 11 h and the kernel timeout to Kaggle's 12 h maximum
(43,200 s). About 7.4 h of the weekly 30 h quota.

**Stop conditions.** Any of these stops the run:
- a failed audit or smoke test
- a nonfinite loss
- OOM
- a smoke estimate above 11 h
- 3 completed epochs

A missing or failed submission check means no upload.

```bash
python3 scripts/build_kaggle_notebook.py --config baselines/vlm/runs_7b_fp16_full/config.json --output baselines/vlm/runs_7b_fp16_full/kaggle_7b_fp16_full.ipynb
.venv-vlm/bin/kaggle kernels push -p baselines/vlm/runs_7b_fp16_full --timeout 43200 --accelerator NvidiaTeslaT4
.venv-vlm/bin/kaggle kernels status <your-kaggle-user-2>/road-7b-fp16-full
```
