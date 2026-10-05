# 7B fp16 on all labels plus test pseudo-labels, two T4s

**Why.** Two gains measured separately, both against the fold-0-pivot vote (0.898238522):
- full-data training: pivot vote 0.900878253, **+0.0026**
- test pseudo-labels: pivot vote 0.900322392, **+0.0021** (fold 0 alone -0.00335)

They come from different sources (held-out labels versus test lines), so this run stacks
them.

**Pseudo-labels v2.** 665 test lines where the full-data 7B
(`kaggle_qwen2vl_20260926T104455Z_21f851e2`) and the pseudo-label fold-0 7B
(`kaggle_qwen2vl_20260926T192036Z_3c18c04f`) decode the identical string. Built by
`scripts/make_pseudo_labels.py`, sha256 `8dddbc89...6ebdc`. It is in the private dataset
`<your-kaggle-user-2>/road-pseudo-labels` next to v1. **Caveat:** the pseudo-label 7B trained on
the v1 labels, so on the 478 lines shared with v1 their agreement is partly inherited
(476 of 478 agree). No fold-0 quality measurement exists for this pair.

**Hypothesis.** As the pivot of the vote with NF4 + full-data 2B, it beats **0.901086809**
on the leaderboard. **Not locally verifiable**; one upload decides.

**Controlled change.** Training rows only: 4,098 labels + 665 pseudo-labelled lines. Every
recipe value is from `../runs_7b_fp16_full/config.json` (fp16 across two T4s,
`FULL_DATA_ONLY`).

**Compute (projected).** 4,763 rows x 3 epochs x 1.82 s is 7.2 h, plus 1.0 h test and
0.15 h setup: **about 8.4 h**. Gate 11 h, timeout 12 h.

```bash
python3 scripts/build_kaggle_notebook.py --config baselines/vlm/runs_7b_fp16_full_pseudo/config.json --output baselines/vlm/runs_7b_fp16_full_pseudo/kaggle_7b_fp16_full_pseudo.ipynb
.venv-vlm/bin/kaggle kernels push -p baselines/vlm/runs_7b_fp16_full_pseudo --timeout 43200 --accelerator NvidiaTeslaT4
.venv-vlm/bin/kaggle kernels output <your-kaggle-user-2>/road-7b-fp16-full-pseudo -p experiments/runs/full_pseudo_small \
  --file-pattern '(test_predictions|run_config|submission|commands)\.(csv|json|jsonl)$'
```
