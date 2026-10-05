# 7B fp16 fold 0 plus test pseudo-labels, two T4s

**Why.** Test pseudo-labelling is allowed: an organizer ruled on 2026-08-17 that it must
be fully automated (thread 34459). Confident self-training showed up in several winning
solutions of similar competitions (for example Kuzushiji 2nd place, +0.003).

**Pseudo-labels.** 707 of 1,374 test lines are lines where the two 7B fold-0 models (fp16
and NF4) decode the identical string. The file is built by `scripts/make_pseudo_labels.py`,
with sha256 `573b01ad...a9d5`. Its quality, measured on fold 0 where the truth is known:
lines where both 7Bs agree are 50% of rows, with combined error **0.071** (35% exact),
against 0.114 over all rows. **No leakage:** both labelling models trained on folds 1-4
only, so the fold-0 score stays honest.

**Hypothesis.** Adding the 707 lines to fold-0 training reduces complete fold-0 combined
error of the 7B fp16 by at least 0.003: **<= 0.11108619** against 0.11408619.

**Controlled change.** Training rows only (3,280 + 707). Every recipe value is copied from
`../runs_7b_fp16/config.json`. `--extra-train` excludes pseudo-labels from the 8-row smoke
and keeps `train_ids_sha256` over the competition rows, so the fold assertions still hold.

**Input.** The file is not in Git (derived competition data). It must be uploaded as the
private Kaggle dataset `<your-kaggle-user-2>/road-pseudo-labels`; the notebook finds
`pseudo_labels_7bpair_fold0.csv` under `/kaggle/input`.

**Compute (projected).** 3,987 training rows x 3 epochs at the measured 1.82 s per
row-epoch is 6.05 h. Plus 0.62 h validation, 1.0 h test and 0.15 h setup: **about 7.8 h**.
Gate 9 h, timeout 10 h.

**Stop conditions.** Any of these stops the run:
- a failed audit or smoke test
- the pseudo-label file missing or ambiguous
- IDs outside the test set
- a nonfinite loss
- OOM
- an estimate above 9 h

```bash
# once, private: upload the pseudo-label file (the operator approves this transfer)
.venv-vlm/bin/kaggle datasets create -p <dir with pseudo_labels_7bpair_fold0.csv + dataset-metadata.json>
python3 scripts/build_kaggle_notebook.py --config baselines/vlm/runs_7b_fp16_pseudo/config.json --output baselines/vlm/runs_7b_fp16_pseudo/kaggle_7b_fp16_pseudo_fold0.ipynb
.venv-vlm/bin/kaggle kernels push -p baselines/vlm/runs_7b_fp16_pseudo --timeout 36000 --accelerator NvidiaTeslaT4
```
