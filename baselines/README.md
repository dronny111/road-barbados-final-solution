# Member recipes

Each folder holds what is needed to regenerate one Kaggle notebook: `config.json` (the config-cell
overrides), `kernel-metadata.json` (the kernel and its private dataset inputs, with the account name
replaced by `<your-kaggle-user>`) and, where it existed, a `README.md` with the hypothesis and the exact
command. The notebooks themselves are about 1 MB each because they embed the repository source, so they
are rebuilt rather than shipped:

```
make notebook CONFIG=baselines/vlm/<folder>/config.json OUT=<name>.ipynb
```

then push it to Kaggle (T4 accelerator) with the private datasets listed in `kernel-metadata.json`
(the competition files, plus the cleaned images or pseudo-label files where noted). The embedded source
of the notebook that actually ran is recorded in each run's `source_snapshot.json`, which is not shipped.

| # | Member | Folder(s) under `baselines/vlm/` | Notes |
|---|---|---|---|
| 0 | Qwen2.5-VL-7B fp16 + pseudo-labels | `runs_7b_fp16_pseudo` (fold 0), `runs_7b_fp16_full_pseudo` (full data); the pseudo-label chain also needs `runs_7b_fp16` and `runs_7b_fp16_full` | pseudo-labels from `scripts/make_pseudo_labels.py` on the agreeing lines of the 7B fp16 and NF4 fold-0 models; 2 T4 |
| 1 | Qwen2-VL-2B, vision-MLP LoRA | `runs_2b_vmlp_kaggle` | no `config.json`; the README gives the `--set` flags (`LORA_SCOPE=vision-mlp`) |
| 2 | Qwen3-VL-4B, vision-MLP LoRA | `runs_qwen3vl4b_vmlp_kaggle` | 2 T4, 128 px |
| 3 | Qwen2.5-VL-7B NF4 | `runs_7b` | 4-bit LoRA, about 9 h |
| 4 | TrOCR multicentury | `runs_trocr_multicentury_kaggle` (fold 0), `runs_trocr_multicentury_full_kaggle` (full) | `MODEL_FAMILY=trocr`; also the scorer of the likelihood feature (fold-0 adapter) |
| 5 | PP-OCRv6 medium | `runs_ppocrv6` | Colab notebook, `requirements/kraken-ppocr.txt`; run provenance incomplete |
| 6 | TrOCR on cleaned images | `runs_trocr_clean_kaggle` | needs `make clean-images` and a private dataset of `images_clean_jpg/` |
| 7 | Qwen2-VL-2B soft-KD | `runs_2b_vmlp_softkd_fold0_kaggle`, `runs_2b_vmlp_softkd_full_kaggle` | soft targets from `scripts/make_candidate_targets.py` over a vote of the other members |
| 8 | Qwen2-VL-2B recipe SFT | `runs_2b_vmlp_recipe_fold0_kaggle` | curriculum, tight/loose experts, raw/cleaned mix, `scripts/make_confident_pseudo.py` |
| 9 | Qwen2-VL-2B recipe + GRPO | `runs_2b_vmlp_recipe_grpo_experts_fast_kaggle` | member 8's recipe plus 200 GRPO rows; batch-8 decoding |

The fold-0 twin of members 0, 4 and 7 feeds the reranker's fitting; the full-data twin supplies the test
predictions. Several members depend on earlier members' predictions (pseudo-labels, soft targets), so the
order in `docs/REPRODUCE.md` matters.
