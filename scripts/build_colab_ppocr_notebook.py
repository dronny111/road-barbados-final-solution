#!/usr/bin/env python3
"""Build a self-contained free-Colab notebook that fine-tunes PP-OCRv6 medium (kraken) on a fold.

Embeds the repo sources it needs (so the notebook runs from Drive inputs alone), records their
SHA-256 and the git commit/dirty state, and contains no competition data.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRIVE_ROOT = '/content/drive/MyDrive/road'
EMBED = ['src/road_ocr/__init__.py', 'src/road_ocr/metrics.py', 'src/road_ocr/records.py',
         'scripts/make_folds.py', 'scripts/prepare_kraken_data.py', 'scripts/kraken_infer.py']
WEIGHTS_URL = 'https://zenodo.org/records/21788410/files/medium.safetensors'
WEIGHTS_SHA = '15313b51ace64cbfa81f8f6ef25ad64f04e5a6fb7f7823e67b107527bc081ac9'
FOLDS_SHA = 'fff8a4994bc606cdd9987ae521712976bd83c180223dce596340d192ff58a448'

HYPOTHESIS = ('Fine-tuning the Apache-2.0 PP-OCRv6 medium CTC line recognizer (kraken 7.1.1) on the fold-0 '
              'training rows gives fold-0 combined error below the TrOCR control 0.15076, and/or adds a '
              'diverse vote member (gain >= 0.002, 95% bootstrap low bound > 0).')


def git(*args):
    return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def cell(kind, text):
    lines = text.strip('\n').splitlines(True)
    if kind == 'markdown':
        return {'cell_type': 'markdown', 'metadata': {}, 'source': lines}
    compile(text, 'cell', 'exec')
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': lines}


def build(fold, epochs, lr, batch, seed):
    sources = {p: (ROOT / p).read_text(encoding='utf-8') for p in EMBED}
    provenance = {'commit': git('rev-parse', 'HEAD'), 'branch': git('rev-parse', '--abbrev-ref', 'HEAD'),
                  'dirty_embedded_files': git('status', '--porcelain', '--', *EMBED).splitlines(),
                  'sha256': {p: hashlib.sha256(s.encode()).hexdigest() for p, s in sources.items()}}
    config = f'''# Configuration: edit before Run All.
DATA_ROOT = "{DRIVE_ROOT}"   # Train.csv, Test.csv, images/ (or images.zip); private, never uploaded
OUTPUT_ROOT = "{DRIVE_ROOT}/results"
FOLD, SEED = {fold}, {seed}
EPOCHS, LRATE, BATCH, PRECISION = {epochs}, {lr}, {batch}, "16-mixed"
MAX_ESTIMATED_HOURS = 9.0
WEIGHTS_URL, WEIGHTS_SHA = "{WEIGHTS_URL}", "{WEIGHTS_SHA}"
FOLDS_SHA = "{FOLDS_SHA}"   # sha256 of the frozen data/splits/folds.csv; regenerated fold file must match
SOURCE_FILES = {json.dumps(sources)}
SOURCE_PROVENANCE = {json.dumps(provenance)}
'''
    setup = '''import os, sys, json, time, hashlib, shutil, subprocess, zipfile, uuid
from pathlib import Path
from datetime import datetime, timezone
from google.colab import drive
drive.mount("/content/drive")
import torch
assert torch.cuda.is_available(), "Select a T4 GPU runtime"
print(torch.cuda.get_device_name(0))
RUN_ID = "colab_ppocrv6_medium_fold%d_%s_%s" % (FOLD, datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"), uuid.uuid4().hex[:8])
WORK = Path("/content/work"); WORK.mkdir(exist_ok=True)
RUN = Path(OUTPUT_ROOT) / RUN_ID; RUN.mkdir(parents=True, exist_ok=False)
for rel, text in SOURCE_FILES.items():
    dest = WORK / rel; dest.parent.mkdir(parents=True, exist_ok=True); dest.write_text(text, encoding="utf-8")
(RUN / "source_provenance.json").write_text(json.dumps(SOURCE_PROVENANCE, indent=1))
data = Path(DATA_ROOT)
for name in ["Train.csv", "Test.csv"]:
    shutil.copy(data / name, WORK / name)
images = WORK / "images"
src_dir = next((p for p in data.rglob("images") if p.is_dir() and next(p.glob("*.jpg"), None)), None)
if src_dir:
    shutil.copytree(src_dir, images)
else:
    with zipfile.ZipFile(next(data.rglob("images.zip"))) as z:
        images.mkdir()
        for m in z.namelist():
            if m.lower().endswith(".jpg"):
                (images / Path(m).name).write_bytes(z.read(m))
print("images:", len(list(images.glob("*.jpg"))))
env = dict(os.environ, PYTHONPATH=str(WORK / "src"))
def sh(cmd, log=None, **kw):
    print("$", cmd, flush=True)
    with open(log or os.devnull, "w") as f:
        p = subprocess.Popen(cmd, shell=True, cwd=WORK, env={**env, **kw}, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            f.write(line); f.flush(); print(line, end="")
        assert p.wait() == 0, "command failed"
'''
    folds = f'''# Regenerate the frozen folds and require the repo checksum (never change folds to improve a score).
sh("python scripts/make_folds.py --train Train.csv --output data/splits/folds.csv")
got = hashlib.sha256((WORK / "data/splits/folds.csv").read_bytes()).hexdigest()
assert got == FOLDS_SHA, f"fold manifest mismatch: {{got}}"
print("folds verified")
'''
    install = '''sh("pip install -q kraken==7.1.1")
# kraken 7.1.1 pins safetensors~=0.7.0, but only uses the stable save_file/load_file/safe_open
# API; something else in this image needs safetensors>=0.8.0 at import time (transformers'
# own version check). --no-deps skips re-checking kraken's declared range for this one upgrade.
sh("pip install -q --no-deps -U 'safetensors>=0.8.0'")
import importlib.metadata as md
print("kraken", md.version("kraken"), "torch", md.version("torch"))
import importlib; importlib.invalidate_caches()
import torch; assert torch.cuda.is_available(), "pip replaced torch with a non-CUDA build"
sh(f"curl -sL -o /content/medium.safetensors {WEIGHTS_URL}")
got = hashlib.sha256(Path("/content/medium.safetensors").read_bytes()).hexdigest()
assert got == WEIGHTS_SHA, f"weights sha mismatch {got}"
sh(f"python scripts/prepare_kraken_data.py --fold {FOLD} --seed 20260906 --output /content/work/kdata")
'''
    smoke = '''# Smoke: 32 rows, 1 epoch. Measures seconds/step and gates the full run by forecast hours.
sh("head -32 kdata/train.txt > kdata/smoke_train.txt; head -8 kdata/validation.txt > kdata/smoke_val.txt")
t0 = time.time()
sh(("ketos -d cuda:0 --precision %s --workers 2 --threads 2 -s %d train --arch ppocrv6 --variant medium "
    "-i /content/medium.safetensors --resize union -f path -t kdata/smoke_train.txt -e kdata/smoke_val.txt "
    "-B %d -r %g --optimizer AdamW --augment -q fixed -N 2 -o /content/smoke_out") % (PRECISION, SEED, BATCH, LRATE),
   log=str(RUN / "smoke.log"), TORCHDYNAMO_DISABLE="1")
smoke_s = time.time() - t0
steps = -(-32 // BATCH) * 2
train_rows = sum(1 for _ in open(WORK / "kdata/train.txt"))
per_step = max(smoke_s - 60, 1) / steps   # subtract ~60 s of start-up/validation; conservative lower bound is fine
forecast_h = per_step * (train_rows / BATCH) * EPOCHS / 3600
print(f"smoke {smoke_s:.0f}s, ~{per_step:.1f}s/step (upper bound), forecast {forecast_h:.1f}h")
(RUN / "smoke.json").write_text(json.dumps({"smoke_seconds": smoke_s, "steps": steps, "forecast_hours_upper": forecast_h}))
assert forecast_h <= MAX_ESTIMATED_HOURS, "forecast over the gate: lower EPOCHS or stop (do not loosen the gate)"
'''
    train = '''(RUN / "run_config.json").write_text(json.dumps(dict(
    run_id=RUN_ID, fold=FOLD, seed=SEED, epochs=EPOCHS, lrate=LRATE, batch=BATCH, precision=PRECISION,
    model="PP-OCRv6 medium (Zenodo 10.5281/zenodo.21788410)", model_license="Apache-2.0", weights_sha256=WEIGHTS_SHA,
    kraken="7.1.1", runtime="free Colab T4", provenance=SOURCE_PROVENANCE), indent=1))
t0 = time.time()
sh(("ketos -d cuda:0 --precision %s --workers 2 --threads 2 -s %d train --arch ppocrv6 --variant medium "
    "-i /content/medium.safetensors --resize union -f path -t kdata/train.txt -e kdata/validation.txt "
    "-B %d -r %g --optimizer AdamW -w 0.01 --schedule cosine --cos-max %d --warmup 100 --augment "
    "-q fixed -N %d -F %d -o /content/out") % (PRECISION, SEED, BATCH, LRATE, EPOCHS, EPOCHS, EPOCHS),
   log=str(RUN / "train.log"), TORCHDYNAMO_DISABLE="1")
(RUN / "train_minutes.txt").write_text("%.1f" % ((time.time() - t0) / 60))
weights = sorted(Path("/content/out").glob("best_*.safetensors"))
assert weights, "no weights written"
shutil.copy(weights[-1], RUN / "model.safetensors")
print("saved", RUN / "model.safetensors")
'''
    infer = '''model = RUN / "model.safetensors"
sh(f"python scripts/kraken_infer.py --input Train.csv --fold-manifest data/splits/folds.csv --fold {FOLD} "
   f"--model {model} --device cuda:0 --output {RUN}/fold{FOLD}_predictions.csv --metadata {RUN}/fold{FOLD}_metadata.json",
   log=str(RUN / "infer_fold.log"), TORCHDYNAMO_DISABLE="1")
score = json.loads((RUN / f"fold{FOLD}_metadata.json").read_text())["score"]
print("FOLD-%d  CER %.5f  WER %.5f  combined %.5f  (TrOCR control 0.15076; gate: lower)" % (FOLD, score["cer"], score["wer"], score["combined"]))
sh(f"python scripts/kraken_infer.py --input Test.csv --model {model} --device cuda:0 --output {RUN}/test_predictions.csv "
   f"--metadata {RUN}/test_metadata.json", log=str(RUN / "infer_test.log"), TORCHDYNAMO_DISABLE="1")
print("Artifacts on Drive:", RUN)
'''
    title = f'''# R.O.A.D. Barbados: PP-OCRv6 medium (kraken) fine-tune, canonical fold {fold}, free Colab T4

Run in order on a T4 runtime. Inputs come from your private Drive `road/`; nothing is uploaded.
Hypothesis: {HYPOTHESIS}
Controls: same frozen folds (checksum-verified), seed {seed}, no test-set use for training. Weights: Apache-2.0.
Hyper-parameters (AdamW lr {lr}, {epochs} epochs, batch {batch}, cosine, warmup 100) are a first guess, not tuned.
Stop: smoke forecast above the 9 h gate, nonfinite loss, or fold-0 combined above the control.
Caveat: PP-OCRv6 training drops lines wider than 2560 px at height 96 (about 2% of rows).
'''
    return {'cells': [cell('markdown', title), cell('code', config), cell('code', setup), cell('code', folds),
                      cell('code', install), cell('code', smoke), cell('code', train), cell('code', infer)],
            'metadata': {'accelerator': 'GPU', 'road_runtime': 'colab',
                         'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
            'nbformat': 4, 'nbformat_minor': 5}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'baselines/vlm/runs_ppocrv6/colab_ppocrv6_medium_fold0.ipynb')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--seed', type=int, default=20260906)
    args = parser.parse_args()
    notebook = build(args.fold, args.epochs, args.lr, args.batch, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n')
    print(f'Built {args.output} (code only; no competition data)')


if __name__ == '__main__':
    main()
