# PP-OCRv6 medium (kraken 7.1.1) fine-tune, fold 0 — NOT YET RUN ON COLAB

- **Hypothesis:** fine-tuning PP-OCRv6 medium (Apache-2.0, Zenodo 10.5281/zenodo.21788410, sha256 `15313b51…c081ac9`) on the fold-0 train rows beats the TrOCR control (fold-0 combined 0.15076), and/or is a diverse vote member (vote gain >= 0.002, 95% bootstrap low > 0).
- **Controls:** frozen folds (checksum-verified in the notebook), seed 20260906, no test data in training.
- **Recipe (first guess, untuned):** AdamW lr 2e-4, wd 0.01, cosine, warmup 100, 12 epochs, batch 8, 16-mixed, `--augment`, `--resize union`, final-epoch weights (`-F 12`, so no best-of-N selection on fold 0).
- **Stop:** smoke forecast over 9 h, nonfinite loss, or fold-0 combined above 0.15076.
- **Known risks:** lines wider than 2560 px at height 96 (~2%) are dropped from training; a local CPU/MPS timing was inconsistent (54 s/step in two runs, one 8-row step in ~24 s total), so Colab speed is unmeasured; whole tall crops are downscaled to height 96 (loose-crop handling untouched).
- Local `.venv-kraken` (Python 3.13) needs `TORCHDYNAMO_DISABLE=1`. `scripts/kraken_infer.py` now loads `.safetensors` via `RecognitionTaskModel` (legacy `.mlmodel` path unchanged).

Build: `python3 scripts/build_colab_ppocr_notebook.py` (embeds sources + SHA-256; records dirty state of embedded files). Inputs: Drive `road/` with Train.csv, Test.csv, images. Output: `road/results/colab_ppocrv6_medium_fold0_*`.
