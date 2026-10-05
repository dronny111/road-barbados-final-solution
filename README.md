# R.O.A.D. Barbados Historic Handwriting Challenge: final solution

**Public leaderboard 0.923964** (lower-is-better objective `0.5 * weighted WER + 0.5 * weighted CER`, shown on the board as `1 - combined`).
Decoded WER 0.1133, CER 0.0387. I do not know the private score or the final rank, and this solution does not claim a top-5 finish.

The submitted file is a **hill-climbing reranker over the transcriptions of ten recognizers**: seven Qwen vision-language models, two TrOCR-large models and a PP-OCRv6 recognizer. For each test line it picks one member's own reading, scored by a weighted sum of the leaderboard line cost against all members, a TrOCR likelihood and a character language model. Weights are fitted on 818 held-out training lines, never on test.

The full story, with what worked, what failed and the numbers, is in **[docs/SOLUTION.md](docs/SOLUTION.md)**. How to rebuild it is in **[docs/REPRODUCE.md](docs/REPRODUCE.md)**.

![Public leaderboard over time](docs/figures/lb_progression.png)

Visual walkthrough with diagrams of the pipeline and the reranker: **[docs/demo.html](docs/demo.html)** (download or serve with GitHub Pages; rebuild with `make demo`).

## What is in this repository

```
submission/SHA256SUMS                    checksum of the submitted CSV (the CSV itself is not shipped, see docs/REPRODUCE.md)
docs/demo.html                           visual walkthrough (pipeline, reranker, charts); open it in a browser
docs/SOLUTION.md  docs/REPRODUCE.md  docs/COMPETITION.md  docs/results/*.csv  docs/figures/*.png
configs/final_ensemble.json              member order, file roles and the fitted weights
baselines/                               per-member Kaggle/Colab configs (notebooks are rebuilt, see baselines/README.md)
scripts/                                 folds, training, inference, pseudo-labels, voting, reranking, checks
src/road_ocr/                            metric, line loading, LoRA scopes, TrOCR model, character LM, recipe pieces
tests/                                   unit tests for the kept code
requirements/                            one file per environment (Kaggle training, local scoring, cleaning, PP-OCRv6)
```

Not shipped, by design: the competition data, labels and IDs, every prediction file other than the final one,
model adapters and candidate scores (they are derived from the competition data), and the experiment history.

## Quick start

```bash
pip install -r requirements/local-scoring.txt
make test                                   # unit tests
# put Train.csv, Test.csv, SampleSubmission.csv and images/ in the repository root, then:
make preflight && make folds                # data audit; the fold manifest must match FOLD_SHA256
make verify-submission                      # checksum and format of the shipped submission
make final-ensemble DRY=1                   # the commands that rebuild it from the ten members' predictions
```

Rebuilding the members takes about 50 T4-hours plus Colab time for PP-OCRv6; see `docs/REPRODUCE.md` for the
order, which matters because several members are trained on pseudo-labels produced by earlier ones.

## Results at a glance

| Model or stage | Fold-0 combined error | Public LB |
|---|---|---|
| Qwen2-VL-2B LoRA (first big jump) | 0.1746 | 0.83236 |
| + vision-encoder MLP in the LoRA scope | 0.1144 | |
| Best single models: soft-KD 2B | 0.1055 | 0.9049 (fold-0 student), 0.9067 (distilled full-data twin) |
| MBR vote of 7 members + TrOCR likelihood rescore | 0.0851 | 0.92171 |
| **Hill-climb over 10 members (submitted)** | **0.0829 (cross-validated)** | **0.92396** |

Fold-0 numbers come from one fold of 818 lines and ranked the leaderboard unreliably (SOLUTION.md section 3).

## Compliance

Only the provided data and openly licensed pretrained models were used (Qwen2.5-VL-7B, Qwen3-VL-4B, Qwen2-VL-2B,
the Kansallisarkisto multicentury TrOCR, PP-OCRv6 medium). No manual labelling or editing of predictions, no
hidden labels, no leaderboard probing. Pseudo-labels are code-generated from test predictions, which the
organizers confirmed is allowed. The Qwen2-VL-2B model card's licence was not verified against the organizers'
list. See `docs/COMPETITION.md` and SOLUTION.md section 9.

## Provenance

Built from the working repository at commit `68734a2` by an export script that lives in that repository (not shipped here). It keeps the code
reachable from the final pipeline, drops the experiment history and removes account names and local paths.
Code is released under the Apache-2.0 licence (an assumption; change it if the organizers require otherwise).
