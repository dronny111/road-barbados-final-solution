# Competition contract

Source reviewed on 2026-09-06: [official Zindi competition page](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge).

## Task and data

- Transcribe cropped line images from 18th- and 19th-century Barbados archival records.
- Local download: 4,098 labeled train rows, 1,374 test rows, and 5,472 JPEGs.
- Submission schema: exactly `ID,Target`, with every test ID present.
- The data license limits the dataset to this challenge and prohibits publishing or transmitting it. The repository therefore ignores all CSVs, images, archives, and derived data.

## Leaderboard objective

Lower is better. Zindi describes two length-weighted error metrics:

```text
combined = 0.5 * weighted_WER + 0.5 * weighted_CER
```

Longer reference transcriptions contribute more. `src/road_ocr/metrics.py` implements this as corpus/micro WER and CER (summed edit distance divided by summed reference length) without text normalization. This is the repository's versioned working interpretation until Zindi releases executable scorer code or an organizer confirms a different formula.

### Displayed score decoding, confirmed 2026-09-17

The leaderboard displays a higher-is-better score beside two component figures. For
submission `20260917_qwen_fold_vote3.csv` the operator reported score `0.833560673`,
`WER Weighted 2.672003799` and `CER Weighted 6.061641875`. Those components reproduce
the score exactly:

```text
score = 1 - 0.5 * (weighted_WER / 12 + weighted_CER / 55)
      = 1 - 0.5 * (0.22266698325 + 0.11021167045)
      = 0.8335606731
```

So a displayed score maps to `combined_error = 1 - score` under this repository's
definition, and the two published components are edit counts per row that the divisors 12
and 55 convert into rates. The arithmetic is verified to nine decimals on one submission
with published components; the meaning of the divisors is inferred, and no executable
organizer scorer has been released. Do not upload further variants merely to test this
decoding: repeated scoring of near-identical submissions is leaderboard probing.

Participants in [thread 33847](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/33847)
and [thread 34861](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/34861)
report the components as **mean Levenshtein edit counts per line** (words split on
whitespace, characters raw), and say that reproduces their submissions to nine decimals.
That is not the same as the corpus/micro rates in `src/road_ocr/metrics.py`: under the
leaderboard form, every line counts equally and one word edit costs as much as 55/12 = 4.58
character edits. This is an unverified participant claim and an open risk. The local
scorer is unchanged.

The supplied starter evaluators are not authoritative: they import `wer.py` and `cer.py` files absent from the starter archive and combine `0.7 * CER + 0.3 * WER`, conflicting with the live page's `0.5/0.5` definition.

## Operational rules

- Open to all; maximum team size 4.
- Maximum 5 submissions per day and 200 overall.
- Only provided datasets may be used. Openly available pretrained models are allowed.
- Organizer clarifications rechecked on 2026-09-14: [automated test-image
  self-training is allowed](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/34459),
  with reproducible code and no manual transcription or sample-specific fixes.
  Whether to use it is a separate experiment decision; it is disabled in the
  initial word-localisation comparison.
- The [2026-09-11 model-licensing clarification](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/34734)
  requires weights that allow downstream use, modification, reproduction and
  commercial deployment. Further adaptation must use competition data only.
  Upstream dataset licences need not be independently audited unless restrictions
  explicitly carry through to downstream use.
- **Checked 2026-09-25: `Qwen/Qwen2.5-VL-3B-Instruct` is licensed `qwen-research`
  (non-commercial),** so under the ruling above it is not a permitted final-solution
  backbone. The organizers already denied `stanford-oval/churro-3B` on this same licence
  ([thread 34481](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/34481)).
  `Qwen2.5-VL-7B-Instruct`, `Qwen3-VL-4B-Instruct` and `Qwen3-VL-8B-Instruct` are
  Apache-2.0, and PP-OCRv6 medium was explicitly approved. 3B results are recipe evidence
  only.
- **Checked 2026-09-29: `Kansallisarkisto/multicentury-htr-model` is permitted.** Its weights
  are Apache-2.0 and public (not gated). It is a fine-tune of `microsoft/trocr-large-handwritten`
  (MIT) on 913,000 16th-20th-century lines in Swedish and Finnish. Its card carries no
  downstream restriction from that training data, so under the ruling above no upstream
  data audit is needed. Any further adaptation must use competition data only.
- **Checked 2026-09-29:** `microsoft/trocr-base-handwritten` declares **MIT** on its card (permitted).
  `microsoft/trocr-large-handwritten` declares **no license** on its card (the unilm GitHub
  repository is MIT, but the weights card does not say so), so it is **not used as a
  member**. Its fine-tune, the Kansallisarkisto model, is Apache-2.0 in its own right.
- **Checked 2026-09-29: `moonshotai/Kimi-VL-A3B-Instruct` is permitted.** Its card and the Hub
  metadata declare **MIT**, and it is public (not gated). Its base, `moonshotai/Moonlight-16B-A3B`,
  is also MIT and ungated. It runs with `trust_remote_code` at revision
  `398eede0903cd983a2bfa0cc634e9ac1d843f375`, and that code is open source in the same repository.
- [StackMix-style augmentation](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/34514)
  is permitted when it is fully automated and built from training images only.
  [Excluding clearly corrupted training rows](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/33891)
  is permitted; document which rows were excluded and why.
- Only open-source languages and packages; AutoML is prohibited.
- No manual labeling or spreadsheet manipulation; transformations must be code.
- Top 10 must provide reproducible code within 48 hours after the closing request; the rules also reserve the right to request code during the challenge with 24 hours to respond.
- Competition closes and private leaderboard reveals on 2026-10-04 according to the live page.

## Source discrepancies and unknowns

- The summary rules say the public/private split is approximately 30%/70%; detailed boilerplate later says 20%/80%. Do not tune to either assumption.
- The leaderboard HTML page renders its rows in the browser, so a static fetch shows only the benchmark row. The full public leaderboard is available without logging in from `https://api.zindi.world/v1/competitions/road-barbados-historic-handwriting-challenge/participations?page=0&per_page=100`. As of 2026-09-25 it has 609 ranked: #1 0.935147, #5 0.932322, #10 0.931877, #50 0.926330, #100 0.920350. Our best, 0.893432, is #291. These are public scores and will change before close.
- Treat dates, limits, and rules as changeable external state; re-check the official page before final submission selection.

## Benchmark and known pitfalls

- Benchmark: Zindi's benchmark row is visible on the leaderboard page. Record its score here and reproduce it locally (`make score`) before claiming improvement; not yet recorded in this file.
- Pitfalls: starter evaluators use the wrong `0.7/0.3` blend; the displayed score is `1 - combined` (see decoding above); do not use leaderboard probing to calibrate.
