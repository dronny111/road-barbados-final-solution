# Reproducing the final submission

Nothing derived from the competition data is shipped: no labels, IDs, predictions, adapters or candidate
scores. You rebuild the ten members from the competition data and the recipes in `baselines/`, then run
the reranker. The submitted CSV, `submission/20261004_allvote10_hill.csv`, is not shipped either (the data licence
forbids publishing derived data). Its checksum is in `submission/SHA256SUMS`; after rebuilding the file
in `submission/`, `make verify-submission` checks it against that checksum.

**Expect this to take about 50 T4-hours of training plus Colab time for PP-OCRv6 (not recorded), and about
25 minutes of CPU for scoring and reranking.** Training is not bit-reproducible across GPUs, torch builds
or decode batch sizes (batch 8 changed a recipe's score by 0.0025 against batch 1), so a retrained pool
will give a score near, not equal to, 0.9240. Section "What is and is not reproducible" lists the gaps.

## 0. Inputs and environments

1. Put `Train.csv`, `Test.csv`, `SampleSubmission.csv` and `images/` (the competition download) in the
   repository root. They are git-ignored.
2. Three environments, because the stacks conflict:
   - Kaggle T4 notebooks: `requirements/train-kaggle.txt` (the notebooks install it themselves).
   - Local CPU scoring, stacking, tests: `requirements/local-scoring.txt` (Python 3.11 or newer).
   - Image cleaning: `requirements/cleaning.txt`. PP-OCRv6: `requirements/kraken-ppocr.txt` with
     `TORCHDYNAMO_DISABLE=1`.
3. `make test` (needs `KMP_DUPLICATE_LIB_OK=TRUE` on macOS; the Makefile sets it), `make preflight`,
   `make folds`. `make folds` regenerates `data/splits/folds.csv`; its checksum must equal `FOLD_SHA256` in
   `src/road_ocr/trocr_support.py`, and every run checks it.

## 1. Train the members, in dependency order

Rebuild a notebook with `make notebook CONFIG=... OUT=...` and run it on a Kaggle T4 (see
`baselines/README.md`). Every notebook runs the same audit first (tests, preflight, frozen folds, an
8-row smoke test with a time-budget gate), then trains a fold-0 model, scores its 818 held-out lines,
and writes test predictions.

| Step | What | Needs |
|---|---|---|
| 1 | Member 3: 7B NF4, fold 0 (`runs_7b`) | folds |
| 2 | 7B fp16 fold-0 model (`runs_7b_fp16`, not itself a final member), then `python scripts/make_pseudo_labels.py <7B fp16 test CSV> <7B NF4 test CSV> --output pseudo_v1.csv` (the lines on which they agree) | step 1 |
| 3 | Member 0, fold-0 twin: `runs_7b_fp16_pseudo` with `pseudo_v1.csv` as `PSEUDO_LABELS` | step 2 |
| 4 | Full-data 7B fp16 (`runs_7b_fp16_full`, not itself a final member), then `make_pseudo_labels.py` on the full-data and the pseudo fold-0 7B test CSVs for `pseudo_v2.csv`; then `runs_7b_fp16_full_pseudo` (member 0, test side) | steps 2, 3 |
| 5 | Members 1, 2 (`runs_2b_vmlp_kaggle`, `runs_qwen3vl4b_vmlp_kaggle`) and member 4 (`runs_trocr_multicentury_kaggle`, `_full_kaggle`) | folds |
| 6 | Member 5: PP-OCRv6 medium, `baselines/vlm/runs_ppocrv6` on Colab (kraken 7.1.1) | folds |
| 7 | Member 7 (soft-KD): vote the other members' test predictions with `scripts/vote_predictions.py --mode mbr-vote`, then `scripts/make_candidate_targets.py <vote> <members...> --output softkd.csv` (fold-0 twin: fold-0 members only; full twin: all members), then `runs_2b_vmlp_softkd_fold0_kaggle` and `_full_kaggle` | steps 3-6 |
| 8 | Cleaned images (members 6, 8, 9): `make clean-images`, upload `images_clean_jpg/` as a private Kaggle dataset | folds |
| 9 | Member 6: `runs_trocr_clean_kaggle` | step 8 |
| 10 | Member 8: `python scripts/make_confident_pseudo.py <vote> <fold-0-safe members...> --output pseudo_f0.csv` (lines read identically by at least two thirds of the members; use only members trained without fold 0), then `runs_2b_vmlp_recipe_fold0_kaggle` | steps 8, 3-7 |
| 11 | Member 9: `runs_2b_vmlp_recipe_grpo_experts_fast_kaggle` (it re-trains member 8's SFT, then runs GRPO) | step 10 |

Pseudo-label files and candidate sidecars are derived competition data: keep them in a private Kaggle
dataset, never in Git. Every training notebook refuses pseudo-label IDs that are not test IDs, and
`vlm_finetune.py` refuses any that overlap training or validation rows.

## 2. Collect the prediction files

Copy each member's fold-0 validation predictions and test predictions to the paths in
`configs/final_ensemble.json` (`predictions/fold0/<name>.csv`, `predictions/test/<name>.csv`), plus the
fold-0 reference labels (`predictions/fold0/validation_reference.csv`, written by every notebook) and the
member-4 fold-0 TrOCR adapter (`adapters/trocr_multicentury_fold0`, used only to score candidates).
Order matters: member 0 is the pivot, and ties go to the earliest member.

## 3. Rescore and rerank

```
make final-ensemble DRY=1     # print the commands
make final-ensemble           # 1) fold-0 candidate NLL  2) test candidate NLL  3) hill-climb  4) validate and hash
```

This runs `scripts/trocr_score_candidates.py` twice (about 10 and 15 minutes on a laptop CPU),
then `scripts/stack_hill_test.py`, which fits the member, NLL and character-LM weights on the 818 fold-0
lines and applies them to the test candidates. The shipped file used the weights
`[1,1,1,1,1,3,1,1,1,1]`, NLL `0.01`, LM `0.3`. The script prints whether your file's sha256 equals the
shipped one.

## What is and is not reproducible

- **Deterministic:** the folds, the cleaning, the pseudo-label builders, the vote, the candidate pools and
  the hill-climb (no randomness; the CV stacker comparison in `scripts/stack_candidates.py` uses 20 seeds).
- **Not bit-reproducible:** GPU training and decoding (different GPU, torch build, batch size), and the
  TrOCR NLL scores across torch builds or machines.
- **Checked on the machine that built the submission:** re-running the fold-0 scoring pass from this
  repository reproduced all 5,409 candidate NLLs bit for bit, and `make final-ensemble` from the saved
  fold-0 and test scores reproduced the shipped file byte for byte (sha256 `236f551e...`). The test-side
  scoring pass was not re-run; it uses the same code.
- **Gaps:** the PP-OCRv6 Colab run has no recorded `run_config.json` or runtime; two Kaggle torch builds
  were used; the cleaned-image datasets were deleted after the competition and must be rebuilt; the scorer
  is the *fold-0* TrOCR adapter on both sides (the full-data adapter was too slow to download).
- **Licences:** the Qwen2-VL-2B model card's licence was not verified against the organizers' list. The
  Qwen2.5-VL-3B models used early on are `qwen-research` licensed and are not part of this submission.
