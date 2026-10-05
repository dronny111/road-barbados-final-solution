# Qwen2.5-VL-7B NF4, canonical fold 0

Pursue the 0.93 displayed-score goal with a falsifiable capacity experiment.
The last recorded leaderboard measurement is 0.890343557. The target corresponds
to combined error 0.07 under the repository's working score interpretation.
Neither the target nor this experiment's gain is guaranteed.

The hypothesis is that the deployable 7B NF4 recognizer beats the 3B fp16 fold-0
baseline (CER 0.0520808841, WER 0.1725625539, combined 0.1123217190) by at least
0.010 combined. Success requires combined <0.1023217190 on all 818 held-out rows.
Capacity and quantization change together; this does not isolate capacity alone.

`config.json` fixes three epochs, seed 20260906, batch 2 / accumulation 8,
learning rate 1e-4, LoRA rank 16 / alpha 32, gradient checkpointing, height 112,
451584 pixels, and greedy 96-token decoding with repetition ban 4. The folds,
scorer, prompt and normalization are unchanged. No filtering, GRPO or full-data
training is enabled. A success requires independent-fold replication before
full-data selection; a failure retains the 3B baseline.

The official Apache-2.0 model is pinned to revision
`cc594898137f460bfe9f0759e9844b3ce807cfb5`. The existing Kaggle dataset named
`qwen-7b-instruct` contains a text-only Qwen2ForCausalLM; it cannot replace this
vision model and is deliberately not attached.

Unmeasured compute forecast: roughly 6–10 GPU hours. Run one free GPU kernel.
The eight-row train/inference smoke must pass with finite loss, correct masking,
nonempty predictions, and an estimated training/inference cost <=9 hours.
The kernel has a 36000-second timeout including setup. OOM, audit/smoke failure,
nonfinite loss, quota exhaustion, forecast >9h, or three completed epochs stops
work. Do not relax the gate or automatically launch a second run.

```bash
python3 scripts/build_kaggle_notebook.py --config baselines/vlm/runs_7b/config.json --output baselines/vlm/runs_7b/kaggle_7b_nf4_fold0.ipynb
KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/check_kaggle_notebook.py --notebook baselines/vlm/runs_7b/kaggle_7b_nf4_fold0.ipynb
.venv-vlm/bin/kaggle kernels push -p baselines/vlm/runs_7b --timeout 36000 --accelerator NvidiaTeslaT4
.venv-vlm/bin/kaggle kernels status <your-kaggle-user-2>/road-7b-nf4-fold0
```

The upload contains code only and mounts the authorized competition input already
on Kaggle. The kernel is private. Model calls in the local notebook checker are
stubbed; a passing check is not evidence of GPU compatibility or OCR quality.
Run metadata, exact subprocess commands, source snapshot, environment, adapter,
and CER/WER/combined results live in a unique remote `experiments/runs` directory.
Retain and independently audit the outputs. No Zindi submission is authorized.

## Measured smoke and free Colab continuation

Kaggle smoke version 1 completed and passed 42 local artifact checks. Model-only
training runtime was 20.1092s for eight fitting rows; inference was 24.0768s for
eight held-out rows. The full-run forecast is **8.70315h**. This is technical
feasibility evidence, not a validation improvement. Kaggle has 0.64 GPU hours
remaining until September 26; the operator chose free Colab for the full run.

`colab_7b_nf4_fold0.ipynb` is self-contained. Select a free T4, set `DATA_ROOT` to
existing inputs and `OUTPUT_ROOT` to private persistent storage, and run in order.
Defaults use `/content/drive/MyDrive/road` and its `results` directory. Drive mount
may require the operator's authentication. No competition data is uploaded.
Checkpoints save every 50 optimizer steps and retain the latest checkpoint.
The Colab runtime must pass its own smoke and nine-hour forecast gate. A ten-hour
elapsed deadline kills a running subprocess; private checkpoints remain on Drive.
Colab execution is not yet verified because no browser/runtime is connected.

```bash
python3 scripts/build_colab_7b_notebook.py
KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/check_kaggle_notebook.py --notebook baselines/vlm/runs_7b/colab_7b_nf4_fold0.ipynb
PYTHONPATH=src python3 scripts/audit_vlm_smoke.py --run experiments/runs/qwen25vl7b_nf4_smoke_20260920T164517Z/downloaded/experiments/runs/kaggle_qwen2vl_20260920T164628Z_b9b569a7 --notebook baselines/vlm/runs_7b_smoke/kaggle_7b_nf4_smoke.ipynb --config baselines/vlm/runs_7b/config.json --output /tmp/road-7b-smoke-audit.json
```

The notebook was installed into the operator-supplied Drive folder
`1xtrrUnn-nGPjrRLfqi5oZTVfpT954RrG` and is ready to run:

**Run this one:**
[colab_7b_nf4_fold0.ipynb](https://colab.research.google.com/drive/1EC1Eeh1CP_SMkcUH4fbRa3zLDDyeF7Kj)
(file id `1EC1Eeh1CP_SMkcUH4fbRa3zLDDyeF7Kj`), uploaded 2026-09-21 23:38 UTC, 650,458
bytes, which was an exact size match for the repository file **as it stood at
`e43fd94`**.

**The repository file no longer matches that Drive copy, on purpose.** It was rebuilt
onto the cell-rerun fix and is now 662,456 bytes, SHA-256 `e328c62e...`, pinning
`055e0cb2`. The copy executing on Colab is the pre-fix build and does **not** have it,
so re-running a cell inside that live session still raises
`FileExistsError: .../setup/kaggle_install.log`. Do not swap the Drive copy mid-run;
install this rebuilt one for the next run, whether that is a retry or fold 1. A lost
runtime is unaffected, since a new session creates a new run directory. Verified in the live
notebook: the 7B title, the `TARGET_HEIGHT` DISPROVED comment, and a closing cell that
prints Drive paths with no `chdir('/kaggle/working')` and no `FileLink`. That is a
size-exact plus content spot check, **not** a SHA-256 readback: reading 650 KB back out
of Drive was not done, so do not record this as byte-for-byte verified.

The older copy is renamed, not deleted. `13ysYToGpCsXZKc4S519GCkeIBWPoZ7zD` is now
`STALE-do-not-run-colab_t4_7b.ipynb`, so the name says so at the point of opening it.
It still carries the closing-cell defect: it chdirs to `/kaggle/working`, which does not
exist on Colab, so the last cell of a finished nine-hour run raises `FileNotFoundError`
and a completed experiment looks failed. It was never executed.

A third file, `colab_augment_fold0.ipynb` (`1Bj9iwfYMtaTIjVSnX0LfmgaVaFuVrpzj`), was
uploaded to the same folder in error during this install and has been moved to Drive
trash at the operator's instruction; the id no longer resolves. It was a valid build of
the augmentation arm, which is a separate single-variable ablation and must not share a
runtime with this one. Rebuild it from `runs_augment/` when that arm is actually run.

Retain a free T4 runtime, choose **Runtime -> Run all**, and complete the Drive mount
prompt. The existing Drive folder contains all three input CSVs and `images.zip`;
runtime audit still checks their content and folds.

The rebuild also corrects the provenance the old copy carried. It pinned
`dirty: True` at `f866f1b`, so its embedded source was not reproducible from git; this
one pins `e43fd94` clean. The recipe is unchanged: the only other difference is an
inert `AUGMENT = False` flag, which never reaches `TRAIN_ARGS`.
