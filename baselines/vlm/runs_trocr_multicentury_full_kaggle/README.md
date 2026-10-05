# TrOCR-large (Kansallisarkisto multicentury HTR), all 4,098 labels, Kaggle T4

The TrOCR fold-0 member (`<your-kaggle-user>/road-trocr-multicentury-fold0`) was weak alone
(0.15076, which failed its gate) but the most valuable vote member: the five-member MBR vote with
it scored **0.914874169** on the leaderboard (+0.0056). This trains the same recipe
(`../runs_trocr_multicentury_kaggle/config.json`) on all labels. That is the member's
full-data version, as the full-data 7B pivot was (+0.0022 in the vote). **No local score**;
the verdict is replacing the fold-0 TrOCR in the five-member MBR vote, one upload against
0.914874169.

Compute: fold training took 1.15 h for 3,280 rows, so full data is about 1.45 h, plus 0.25 h
test and setup: **about 1.9 h**. Gate 9 h, timeout 10 h. Account: `<your-kaggle-user>`.

```bash
python3 scripts/build_kaggle_notebook.py --config baselines/vlm/runs_trocr_multicentury_full_kaggle/config.json --output baselines/vlm/runs_trocr_multicentury_full_kaggle/kaggle_trocr_multicentury_full.ipynb
.venv-vlm/bin/kaggle kernels push -p baselines/vlm/runs_trocr_multicentury_full_kaggle --timeout 36000 --accelerator NvidiaTeslaT4
.venv-vlm/bin/kaggle kernels output <your-kaggle-user>/road-trocr-multicentury-full -p experiments/runs/trocr_mc_full_small \
  --file-pattern '(test_predictions|run_config|submission|commands)\.(csv|json|jsonl)$'
```
