#!/usr/bin/env python3
"""Build the self-contained Kaggle notebook from allowlisted repository source."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "baselines/vlm/kaggle_train_infer.ipynb"
FOLD_SHA256 = "fff8a4994bc606cdd9987ae521712976bd83c180223dce596340d192ff58a448"


def build(overrides=None, *, smoke_only=False):
    # Only code and documentation: never discover or embed competition data.
    paths = [ROOT / "Makefile", ROOT / "pyproject.toml", ROOT / "baselines/vlm/requirements-kaggle.txt"]
    # Only the Kimi-VL path (REMOTE_CODE) installs this one; a trimmed tree may not ship it.
    paths += [p for p in [ROOT / "baselines/vlm/requirements-kimi.txt"] if p.exists()]
    # Small fixtures that embedded tests read. Without them the notebook's own
    # `make test` gate aborts the run on Kaggle before any training starts.
    # Optional: only the tests that read them need these, and a trimmed tree may not ship them.
    paths += [p for p in (ROOT / "baselines/vlm/fold1_plan.json",
                          ROOT / "baselines/mmdetection/pilot_v1.json",
                          ROOT / "baselines/mmdetection/rtmdet_tiny_word.py",
                          ROOT / "baselines/mmdetection/requirements-cu118.txt") if p.exists()]
    paths += sorted((ROOT / "src/road_ocr").glob("*.py"))
    # This builder is embedded too: tests/test_sampling_probe.py imports the probe
    # builder, which imports this module, so leaving it out fails the notebook's
    # own `make test` gate with ModuleNotFoundError.
    paths += sorted(p for p in (ROOT / "scripts").glob("*.py")
                    if p.name != "check_kaggle_notebook.py")
    # test_colab_setup asserts about baselines/vlm/colab_fold1.ipynb, a 123KB
    # notebook for a different runtime. Embedding it to satisfy one test would
    # grow this notebook by a quarter for nothing this pipeline uses.
    paths += sorted(p for p in (ROOT / "tests").glob("test_*.py")
                    if p.name not in {"test_colab_setup.py"})
    sources = {p.relative_to(ROOT).as_posix(): p.read_text() for p in paths}
    source_digest = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    provenance = {
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
        "source_sha256": source_digest,
        "note": "Embedded source is authoritative, including any uncommitted edits at build time.",
    }
    overrides = overrides or {}
    cells = []

    def add(kind, source):
        cell = {"cell_type": kind, "metadata": {}, "source": textwrap.dedent(source).strip().splitlines(True)}
        # Stable cell IDs and trailing newlines keep diffs reviewable.
        cell["source"] = [line if line.endswith("\n") else line + "\n" for line in cell["source"]]
        cell["id"] = f"road-{len(cells):02d}"
        if kind == "code":
            compile("".join(cell["source"]), f"cell-{len(cells)}", "exec")
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    def md(source):
        add("markdown", source)

    def code(source):
        add("code", source)

    md('''
    # R.O.A.D. Barbados: Kaggle training → validation → submission

    This notebook embeds the repository's Qwen2-VL LoRA trainer, image preprocessing,
    frozen fold generator, scorer, tests, and submission validator. No code archive or
    Git clone is needed. No competition data or model weights are embedded.

    1. Import this `.ipynb` into a **private** Kaggle notebook using a free GPU runtime.
    2. The competition inputs must already be available under `/kaggle/input`: `Train.csv`,
       `Test.csv`, `SampleSubmission.csv`, and either `images/*.jpg` or `images.zip`.
       Use only an authorized private compute copy; the repository's data restrictions
       still apply. This notebook does not upload data or publish anything.
    3. Enable Internet for package installation and the public pretrained-model download.
       Set paths in the next cell if automatic discovery finds multiple datasets.
    4. Run cells in order. The default trains fold 0 for three epochs, evaluates every
       held-out row, predicts every test row, and writes a strictly checked `submission.csv`.
       Set `TRAIN_FULL_DATA=True` to additionally train a fresh adapter on all labels
       after validation. This adds a second training run.
    5. Retain private notebook outputs/checkpoints before ending the session. Files in
       `/kaggle/working` are session-local until saved; periodic checkpoints alone do not
       guarantee survival of a terminated runtime. No Zindi upload is performed.

    CUDA throughput and peak memory for this notebook remain unmeasured. The smoke gate
    checks execution and estimates cost; its eight-row score is not model-quality evidence.
    A fold-0 result is preliminary and needs independent-fold confirmation.
    ''')
    code('''
    # Configuration: edit this cell before Run All.
    DATA_ROOT = ""           # e.g. /kaggle/input/your-private-input; empty = auto-discover
    IMAGES_SOURCE = ""       # optional exact images directory or ZIP, even on another mount
    MODE = "train"           # "train" or "infer" (use an existing adapter without retraining)
    TRAIN_FULL_DATA = False  # optional fresh all-label model after fold validation
    FULL_DATA_ONLY = False   # skip the fold model: only for a recipe already validated on a fold
    PSEUDO_LABELS = ""       # ID,Target CSV of test pseudo-labels in a private input; appended to training
    RAW_IMAGES_SOURCE = ""   # VLM recipe: raw-scan directory mixed into training (validation uses IMAGES_SOURCE only)
    CURRICULUM = False       # VLM recipe: easy-to-hard training order (road_ocr.recipe)
    EXPERTS = False          # VLM recipe: shared epoch, then height-routed tight/loose LoRA experts
    ALT_PROB = 0.5           # chance a training visit uses the RAW_IMAGES_SOURCE image
    LORA_SCOPE = "baseline"  # "vision-mlp" also adapts the vision MLP where its names differ (Qwen2-VL, Qwen3-VL)
    MODEL_FAMILY = "vlm"     # "trocr" runs scripts/trocr_finetune.py and trocr_infer.py instead
    ADAPTER_PATH = ""        # infer mode: final adapter or intermediate checkpoint directory
    SOURCE_TRAIN_METADATA = ""  # infer mode: matching train_metadata.json is required
    # Intermediate checkpoints can use matching run_config.json from this notebook
    # or the Colab notebook, because train_metadata.json may not exist after interruption.
    SOURCE_RUN_CONFIG = ""

    FOLD = 0
    SEED = 20260906
    EPOCHS = 3
    BATCH_SIZE = 4
    GRAD_ACCUM = 4
    LEARNING_RATE = 1e-4
    LORA_R, LORA_ALPHA = 16, 32
    TARGET_HEIGHT, MAX_PIXELS = 112, 451584
    # Raising TARGET_HEIGHT past what MAX_PIXELS allows makes every crop fill its
    # pixel budget, which gained about 0.0049 combined at inference only. DISPROVED
    # on retraining: at 896 the complete fold 0 scored 0.114591 against the 112
    # baseline's 0.112322, roughly 19x run-to-run variance, both CER and WER worse
    # (kaggle_3b_fold0_fillbudget_20260919T212248Z). Every height from 224 up
    # renders identically, a mean 121.3 of the 144 tokens per image against 68.5
    # at 112, so raising this is the same disproved experiment. Needs a new reason.
    NO_REPEAT_NGRAM = 0  # 4 blocks degenerate repetition loops; about 0.0088 combined
    LOAD_4BIT = False    # required for a 7B on a 16GB T4; 2B and 3B fit in fp16
    # 0 keeps every training row. Set 150 to train only on tight crops, which
    # tests the "some of the data is bad" hypothesis directly. Validation is
    # never filtered, so the fold score stays comparable. Expect this to HURT:
    # test carries the same 25% loose crops, so a model that never saw one
    # still has to read them.
    MAX_SOURCE_HEIGHT = 0
    EXCLUDE_FOLD = None  # also drop this fold from training: a sibling still held out on FOLD
    AUX_LOSS = 'none'    # 'ctc': TrOCR-only auxiliary CTC head (train-time; inference unchanged)
    AUX_WEIGHT = 0.3
    AUX_HEAD_LRATE = 1e-3  # the freshly initialised head's own optimizer group
    RECON_WEIGHT = 0.0   # >0: text-to-visual-frame reconstruction on top of the CTC term
    ALIGN_WEIGHT = 0.0   # >0: cross-attention entropy on top of the CTC term
    # GRPO continues the fold adapter against the 0.5/0.5 metric itself. It runs
    # only after the fold is trained and scored, so the before/after is measured on
    # the same held-out rows. Budget: about 20 s per row-with-4-rollouts, so the
    # full 3,280 rows would be 18 h and does not fit a session; a few hundred does.
    RUN_GRPO = False
    GRPO_ROWS, GRPO_GENERATIONS, GRPO_BETA, GRPO_LRATE = 600, 4, 0.04, 1e-5
    GRPO_EXTRA_ARGS = []  # e.g. ['--reward', 'corpus_edits', '--scale-rewards', 'none']
    MAX_NEW_TOKENS = 96
    INFER_BATCH_SIZE = 1     # decode batch (left-padded); 1 reproduces every earlier run, larger is faster but not bit-identical
    SAVE_STEPS = 50
    GRADIENT_CHECKPOINTING = False  # measured lines are short; recompute is not worth it
    AUGMENT = False  # jitter training crops only; validation and test stay untouched
    GPU_INDEX = "0"  # one visible GPU; "0,1" shards an unquantized 7B across Kaggle's two T4s
    MAX_ESTIMATED_HOURS = 10.0  # local budget gate, not a promise about Kaggle quota
    MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
    MODEL_REVISION = "895c3a49bc3fa70a340399125c650a463535e71c"
    MODEL_LICENSE = "Apache-2.0"
    REMOTE_CODE = False  # Kimi-VL: fetch its modeling *.py and install requirements-kimi.txt
    # Frozen revision from the repository's downloaded public base-model snapshot.
    HYPOTHESIS = "Three-epoch Qwen2-VL LoRA reduces fold-0 combined error by >=0.02 versus Kraken zero-shot."
    CONTROLLED_CHANGE = "Recognizer and its fixed training recipe; model comparison."
    SUCCESS_THRESHOLD = 0.3867929160
    ''')
    if overrides:
        # Rewrite the assignments in the config cell rather than appending a second
        # one, so the committed notebook always shows the values it will run with.
        config = cells[-1]
        rendered, applied = [], set()
        for line in config['source']:
            targets = [t.strip() for t in line.split('=')[0].split(',')] if '=' in line else []
            comment = line.partition('#')[1] + line.partition('#')[2].rstrip('\n')
            if len(targets) == 1 and targets[0] in overrides:
                name = targets[0]
                line = f"{name} = {overrides[name]!r}" + (f"  {comment}" if comment else '') + '\n'
                applied.add(name)
            elif len(targets) > 1 and any(t in overrides for t in targets):
                # A tuple assignment such as "A, B = 1, 2". Splitting on '=' yields
                # "A, B" as the name, so a naive match silently ignored the override
                # and the run used the default -- which cost a Kaggle run before this
                # was caught. Rewrite each target on its own line instead.
                values = [v.strip() for v in line.split('=', 1)[1].partition('#')[0].split(',')]
                assert len(values) == len(targets), f'cannot split tuple assignment: {line!r}'
                pieces = []
                for target, value in zip(targets, values):
                    if target in overrides:
                        pieces.append(f"{target} = {overrides[target]!r}\n")
                        applied.add(target)
                    else:
                        pieces.append(f"{target} = {value}\n")
                if comment:
                    pieces[0] = pieces[0].rstrip('\n') + f"  {comment}\n"
                rendered.extend(pieces)
                continue
            rendered.append(line)
        missing = set(overrides) - applied
        assert not missing, (
            f"--set named {sorted(missing)}, which do not appear in the config cell. "
            "Refusing to build a notebook that silently ignores an override."
        )
        config['source'] = rendered

    md('''
    ## Restore the exact source and create an isolated run

    This cell is intentionally long because it embeds source files as text. Each run
    uses a unique `experiments/runs/<run_id>/` directory. Input links and model caches
    live under `/tmp`; only code, logs, adapters, and derived results go into output.
    ''')
    code(textwrap.dedent('''
    from pathlib import Path
    from datetime import datetime, timezone
    import ast
    import csv
    import hashlib
    import json
    import math
    import os
    import re
    import shutil
    import subprocess
    import sys
    import time
    import uuid
    import zipfile

    assert sys.version_info >= (3, 11), "This repository requires Python 3.11+."
    assert MODE in {"train", "infer"}
    assert FOLD in range(5) and EPOCHS > 0 and MAX_ESTIMATED_HOURS > 0
    assert MODE == "train" or not TRAIN_FULL_DATA, "Full-data retraining needs MODE='train'."
    assert TRAIN_FULL_DATA or not FULL_DATA_ONLY, "FULL_DATA_ONLY needs TRAIN_FULL_DATA=True."
    assert EXCLUDE_FOLD is None or (EXCLUDE_FOLD in range(5) and EXCLUDE_FOLD != FOLD), \
        "EXCLUDE_FOLD must be another fold; it shapes only the fold model, never a full-data retrain."
    assert Path('/kaggle/input').is_dir(), "Run this notebook inside Kaggle."
    os.environ.update(
        CUDA_VISIBLE_DEVICES=GPU_INDEX, TOKENIZERS_PARALLELISM="false",
        HF_HUB_DISABLE_TELEMETRY="1", DO_NOT_TRACK="1", WANDB_DISABLED="true",
        PYTHONHASHSEED=str(SEED), CUBLAS_WORKSPACE_CONFIG=":4096:8",
    )
    SESSION_ID = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8]
    WORK = Path('/tmp') / ('road_' + SESSION_ID)
    WORK.mkdir(exist_ok=False)
    RUN = Path('/kaggle/working/experiments/runs') / ('kaggle_qwen2vl_' + SESSION_ID)
    RUN.mkdir(parents=True, exist_ok=False)
    os.environ['HF_HOME'] = str(WORK / 'hf_cache')
    os.environ['PYTHONPATH'] = str(WORK / 'src')
    ''') + '\nSOURCE_FILES = ' + repr(sources) + '\nSOURCE_PROVENANCE = ' + repr(provenance) + '\n' + textwrap.dedent('''
    source_hash = hashlib.sha256(json.dumps(SOURCE_FILES, sort_keys=True).encode()).hexdigest()
    assert source_hash == SOURCE_PROVENANCE['source_sha256']
    for name, contents in SOURCE_FILES.items():
        destination = WORK / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(contents, encoding='utf-8')
    (RUN / 'source_snapshot.json').write_text(json.dumps(SOURCE_FILES, indent=2) + '\\n')
    os.chdir(WORK)
    sys.path.insert(0, str(WORK / 'src'))
    sys.path.insert(0, str(WORK))

    run_config = dict(
        run_id=RUN.name, created_utc=SESSION_ID, source=SOURCE_PROVENANCE,
        mode=MODE, status='prepared', fold=FOLD, seed=SEED, epochs=EPOCHS,
        batch_size=BATCH_SIZE, grad_accum=GRAD_ACCUM, learning_rate=LEARNING_RATE,
        lora_r=LORA_R, lora_alpha=LORA_ALPHA, target_height=TARGET_HEIGHT,
        max_pixels=MAX_PIXELS, max_new_tokens=MAX_NEW_TOKENS,
        no_repeat_ngram=NO_REPEAT_NGRAM, load_4bit=LOAD_4BIT,
        max_source_height=MAX_SOURCE_HEIGHT, run_grpo=RUN_GRPO, grpo_rows=GRPO_ROWS,
        grpo_extra_args=GRPO_EXTRA_ARGS,
        save_steps=SAVE_STEPS, gradient_checkpointing=GRADIENT_CHECKPOINTING, augment=AUGMENT,
        train_full_data=TRAIN_FULL_DATA, full_data_only=FULL_DATA_ONLY, model_id=MODEL_ID, model_revision=MODEL_REVISION,
        model_license='Apache-2.0', max_estimated_hours=MAX_ESTIMATED_HOURS,
        hypothesis=HYPOTHESIS,
        controlled_change=CONTROLLED_CHANGE,
        fixed_controls='Frozen grouped folds, line preprocessing, seed, greedy decoding, whitespace handling, 0.5/0.5 scorer.',
        success_threshold=SUCCESS_THRESHOLD if FOLD == 0 else None,
        stop_condition='Failed audit/smoke, nonfinite loss, OOM, budget estimate exceeded, or fixed epochs completed.',
    )
    def save_config():
        destination = RUN / 'run_config.json'
        temporary = destination.with_suffix('.tmp')
        temporary.write_text(json.dumps(run_config, indent=2) + '\\n')
        temporary.replace(destination)

    def run(command, log_name):
        command = list(map(str, command))
        log_path = RUN / log_name
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # A rerun must not silently overwrite an earlier command or artifact log, so
        # take the next free suffix instead of clobbering it. Raising here made any
        # rerun fatal, which is routine in Colab and is the entire point of a recovery
        # notebook; commands.jsonl records which log each invocation actually wrote.
        if log_path.exists():
            stem, suffix, attempt = log_path.stem, log_path.suffix, 2
            while True:
                candidate = log_path.with_name(stem + '.' + str(attempt) + suffix)
                if not candidate.exists():
                    log_path = candidate
                    break
                attempt += 1
        with log_path.open('x') as handle:
            record = dict(command=command, cwd=str(WORK), started_utc=datetime.now(timezone.utc).isoformat())
            started = time.monotonic()
            print('$', ' '.join(command), flush=True)
            returncode = None
            try:
                with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      text=True, bufsize=1) as process:
                    for line in process.stdout:
                        print(line, end='', flush=True)
                        handle.write(line)
                        handle.flush()
                    returncode = process.wait()
                if returncode:
                    raise subprocess.CalledProcessError(returncode, command)
            except BaseException as exc:
                run_config.update(status='failed_or_interrupted', failed_log=str(log_path), error=str(exc))
                save_config()
                raise
            finally:
                record.update(returncode=returncode, wall_seconds=time.monotonic() - started, log=str(log_path))
                with (RUN / 'commands.jsonl').open('a') as history:
                    history.write(json.dumps(record) + '\\n')

    save_config()
    print('Private output directory:', RUN)
    '''))
    md('''
    ## Install the VLM stack and verify CUDA

    Hugging Face package versions match the locally exercised environment. The runtime's
    own CUDA PyTorch/torchvision build is pinned as a pip constraint, so a dependency
    cannot quietly swap it for a wheel built against another CUDA stack; a genuine
    conflict fails here instead of at training time. The constraint file and the full
    transitive environment are saved with the run, and CUDA is checked here and by smoke
    execution. The assigned GPU is rejected up front when this torch build carries no
    kernels for its architecture, which on Kaggle means a P100 rather than a T4. Optional `torchao` is removed because this baseline does not use it.
    ''')
    code('''
    # Kaggle assigns a P100 (sm_60) or a T4 (sm_75) at random, and a torch build
    # without kernels for the assigned card fails only later, inside the first CUDA
    # op, as an unexplained "no kernel image is available for execution". Compare
    # against the lowest built architecture: a newer card than any listed still runs
    # through forward-compatible PTX.
    import torch
    assert torch.cuda.is_available(), 'Enable a GPU accelerator'
    capability = torch.cuda.get_device_capability(0)
    device_arch = capability[0] * 10 + capability[1]
    built = [int(arch.removeprefix('sm_')) for arch in torch.cuda.get_arch_list()
             if arch.startswith('sm_')]
    assert built and device_arch >= min(built), (
        f'{torch.cuda.get_device_name(0)} is sm_{device_arch}, below every architecture '
        f'this torch build carries kernels for ({torch.cuda.get_arch_list()}). Switch the '
        'Kaggle accelerator to T4 and start a fresh session.')

    # Pin the runtime's own CUDA build so pip resolves around it. Without this a
    # dependency can silently replace it with a PyPI wheel for another CUDA stack.
    import importlib.metadata as metadata
    pins = []
    for name in ('torch', 'torchvision'):
        try:
            pins.append(f'{name}=={metadata.version(name)}')
        except metadata.PackageNotFoundError:
            pass
    assert any(pin.startswith('torch==') for pin in pins), 'No preinstalled torch to pin.'
    CONSTRAINTS = RUN / 'setup/runtime_constraints.txt'
    CONSTRAINTS.parent.mkdir(parents=True, exist_ok=True)
    CONSTRAINTS.write_text('\\n'.join(pins) + '\\n')
    run_config['runtime_constraints'] = pins
    print('Held at the preinstalled versions:', ', '.join(pins))
    run([sys.executable, '-m', 'pip', 'install', '-q', '-c', CONSTRAINTS, '-r',
         'baselines/vlm/requirements-kimi.txt' if REMOTE_CODE else 'baselines/vlm/requirements-kaggle.txt'],
        'setup/install.log')
    if RUN_GRPO:
        # vlm_grpo.py imports TRL, which requirements-kaggle.txt leaves out; the import check
        # below fails the run in setup rather than after hours of SFT.
        run([sys.executable, '-m', 'pip', 'install', '-q', '-c', CONSTRAINTS, 'trl==1.13.0'], 'setup/install_trl.log')
    run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchao'], 'setup/remove_torchao.log')
    run([sys.executable, '-c',
         ('from trl import GRPOConfig, GRPOTrainer; ' if RUN_GRPO else '') +
         'import torch, torchvision; '
         'from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, Trainer, TrainingArguments; '
         'from peft import LoraConfig, get_peft_model; from datasets import Dataset; '
         'assert torch.cuda.is_available(), "Enable a GPU accelerator"; '
         f'assert torch.cuda.device_count() == {len(GPU_INDEX.split(","))}, "Visible GPUs must match GPU_INDEX"; '
         'print("Fresh-process VLM/CUDA imports passed")'], 'setup/imports.log')
    run(['nvidia-smi'], 'setup/gpu.log')
    from road_ocr.lines import resolve_precision
    PRECISION = resolve_precision('auto')[1]
    run_config.update(gpu=torch.cuda.get_device_name(0), gpu_count=torch.cuda.device_count(), precision=PRECISION,
                      gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory)
    (RUN / 'environment.txt').write_text(subprocess.check_output(
        [sys.executable, '-m', 'pip', 'freeze'], text=True))
    save_config()
    print('Selected precision:', PRECISION)
    ''')
    md('''
    ## Locate inputs, audit every image, and verify frozen folds

    Automatic discovery requires one unambiguous dataset. Set `DATA_ROOT` and
    `IMAGES_SOURCE` explicitly when attaching several datasets. ZIP extraction accepts
    JPEGs only, checks paths, and rejects duplicate filenames. Fold regeneration happens
    only in this new workspace and must match the existing repository checksum.
    ''')
    code(textwrap.dedent('''
    if DATA_ROOT:
        data_root = Path(DATA_ROOT)
    else:
        candidates = sorted({p.parent for p in Path('/kaggle/input').rglob('Train.csv')
                             if all((p.parent / name).is_file()
                                    for name in ['Test.csv', 'SampleSubmission.csv'])})
        assert len(candidates) == 1, f'Set DATA_ROOT; found {len(candidates)} candidate roots: {candidates}'
        data_root = candidates[0]
    for name in ['Train.csv', 'Test.csv', 'SampleSubmission.csv']:
        source = data_root / name
        assert source.is_file(), f'Missing {source}'
        (WORK / name).symlink_to(source.resolve())

    if IMAGES_SOURCE:
        image_source = Path(IMAGES_SOURCE)
    else:
        directories = sorted(p for p in data_root.rglob('images')
                             if p.is_dir() and next(p.glob('*.jpg'), None) is not None)
        archives = sorted(data_root.rglob('images.zip'))
        candidates = directories if directories else archives
        assert len(candidates) == 1, f'Set IMAGES_SOURCE; found {candidates}'
        image_source = candidates[0]
    assert image_source.exists(), f'Missing {image_source}'
    if image_source.is_dir():
        (WORK / 'images').symlink_to(image_source.resolve(), target_is_directory=True)
    else:
        (WORK / 'images').mkdir()
        with zipfile.ZipFile(image_source) as archive:
            seen = set()
            for member in archive.infolist():
                path = Path(member.filename)
                if member.is_dir() or '__MACOSX' in path.parts or path.suffix.lower() != '.jpg':
                    continue
                assert not path.is_absolute() and '..' not in path.parts, 'Unsafe image archive path'
                assert path.name not in seen, f'Duplicate image filename: {path.name}'
                seen.add(path.name)
                with archive.open(member) as source, (WORK / 'images' / path.name).open('xb') as dest:
                    shutil.copyfileobj(source, dest)
    if RAW_IMAGES_SOURCE:
        assert Path(RAW_IMAGES_SOURCE).is_dir(), f'Missing {RAW_IMAGES_SOURCE}'
        (WORK / 'images_raw').symlink_to(Path(RAW_IMAGES_SOURCE).resolve(), target_is_directory=True)
        assert ({p.stem for p in (WORK / 'images_raw').glob('*.jpg')}
                == {p.stem for p in (WORK / 'images').glob('*.jpg')}), 'Raw and cleaned image IDs differ'
    run(['make', 'test', f'PYTHON={sys.executable}'], 'audit/tests.log')
    run(['make', 'preflight', f'PYTHON={sys.executable}'], 'audit/preflight.log')
    # Decode every image as well as checking JPEG markers and content hashes.
    from PIL import Image
    image_digest = hashlib.sha256()
    for path in sorted((WORK / 'images').glob('*.jpg')):
        with Image.open(path) as im:
            im.load()
            assert im.width > 0 and im.height > 0
        image_digest.update(path.name.encode())
        image_digest.update(hashlib.sha256(path.read_bytes()).digest())
    run(['make', 'folds', f'PYTHON={sys.executable}'], 'audit/folds.log')
    fold_hash = hashlib.sha256(Path('data/splits/folds.csv').read_bytes()).hexdigest()
    ''') + f"assert fold_hash == {FOLD_SHA256!r}, 'Frozen fold mismatch; stop without changing the split.'\n" + textwrap.dedent('''
    from road_ocr.records import read_csv, index_unique
    from road_ocr.lines import read_fold_ids, split_by_fold
    train_rows = read_csv('Train.csv', ['ID', 'Target'])
    test_rows = read_csv('Test.csv', ['ID'])
    fit_rows, val_rows = split_by_fold(train_rows, 'data/splits/folds.csv', FOLD)

    def apply_source_height_filter(rows):
        # Mirrors scripts/vlm_finetune.py --max-source-height. The trainer hashes
        # the rows it actually fitted, so the notebook has to filter identically
        # or the integrity assertion below compares against the wrong set.
        if not MAX_SOURCE_HEIGHT:
            return rows
        from PIL import Image
        kept = []
        for row in rows:
            with Image.open(WORK / 'images' / f"{row['ID']}.jpg") as probe:
                if probe.height <= MAX_SOURCE_HEIGHT:
                    kept.append(row)
        return kept

    if EXCLUDE_FOLD is not None:
        # Mirrors --exclude-fold in the trainers, so the fitted-row hash audit still holds.
        _excluded = read_fold_ids('data/splits/folds.csv', EXCLUDE_FOLD)
        fit_rows = [row for row in fit_rows if row['ID'] not in _excluded]
        print(f'EXCLUDE_FOLD={EXCLUDE_FOLD}: training on {len(fit_rows)} rows; fold {FOLD} validation untouched')
    expected_fit_rows = apply_source_height_filter(fit_rows)
    if MAX_SOURCE_HEIGHT:
        print(f'MAX_SOURCE_HEIGHT={MAX_SOURCE_HEIGHT}: training on '
              f'{len(expected_fit_rows)} of {len(fit_rows)} fold rows; validation untouched')
    assert len(train_rows) == 4098 and len(test_rows) == 1374
    def ids_hash(rows):
        return hashlib.sha256('\\n'.join(row['ID'] for row in rows).encode()).hexdigest()
    def write_rows(path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['ID', 'Target'])
            writer.writeheader()
            writer.writerows(rows)
    write_rows(RUN / 'validation_reference.csv', val_rows)
    shutil.copy2('data/splits/folds.csv', RUN / 'folds.csv')
    run_config.update(fold_sha256=fold_hash, image_corpus_sha256=image_digest.hexdigest(),
                      input_sha256={name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                                    for name in ['Train.csv', 'Test.csv', 'SampleSubmission.csv']},
                      train_rows=len(fit_rows), validation_rows=len(val_rows), test_rows=len(test_rows))
    save_config()
    print('Audit and frozen-fold verification passed.')
    '''))
    md('''
    ## Download the fixed public model and prepare commands

    The base model is [Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct),
    licensed Apache-2.0. A specific snapshot is used for both training and inference.
    Adapters require this base model plus the saved processor; preserve the model ID and
    revision in `run_config.json` when moving an adapter to another session.
    ''')
    code('''
    import shutil
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'  # read when huggingface_hub is imported
    from huggingface_hub import snapshot_download
    # Kimi-VL's 32.8GB download went silent and the kernel stopped answering (IOPub timeout):
    # progress-bar floods or RAM from 8 parallel ~5GB shards. Two workers, no bars, and a
    # disk/RAM line first so a repeat failure is diagnosable.
    meminfo = Path('/proc/meminfo')
    print('free /tmp GB:', round(shutil.disk_usage('/tmp').free / 1e9, 1), '| available RAM GB:',
          round(int(meminfo.read_text().split('MemAvailable:')[1].split()[0]) / 1e6, 1) if meminfo.exists() else '?',
          flush=True)
    MODEL_PATH = snapshot_download(
        repo_id=MODEL_ID, revision=MODEL_REVISION, max_workers=2,
        allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja', 'README.md', 'LICENSE*']
                       + (['*.bin'] if MODEL_FAMILY == 'trocr' else [])  # TrOCR repos ship pytorch_model.bin
                       + (['*.py'] if REMOTE_CODE else []),  # trust_remote_code modeling files
    )
    run_config.update(model_snapshot=MODEL_PATH, model_license=MODEL_LICENSE, remote_code=REMOTE_CODE)
    assert MODEL_FAMILY in ('vlm', 'trocr'), MODEL_FAMILY
    FINETUNE_SCRIPT = f'scripts/{MODEL_FAMILY}_finetune.py'
    INFER_SCRIPT = f'scripts/{MODEL_FAMILY}_infer.py'
    run_config.update(model_family=MODEL_FAMILY, finetune_script=FINETUNE_SCRIPT, infer_script=INFER_SCRIPT)
    COMMON = ['--model-id', MODEL_PATH, '--precision', PRECISION,
              '--target-height', str(TARGET_HEIGHT), '--max-pixels', str(MAX_PIXELS)]
    TRAIN_ARGS = [*COMMON, '--batch-size', str(BATCH_SIZE), '--grad-accum', str(GRAD_ACCUM),
                  '--seed', str(SEED), '--lrate', str(LEARNING_RATE),
                  '--lora-r', str(LORA_R), '--lora-alpha', str(LORA_ALPHA)]
    if GRADIENT_CHECKPOINTING:
        TRAIN_ARGS += ['--gradient-checkpointing']
    if AUGMENT:
        TRAIN_ARGS += ['--augment']
    if LORA_SCOPE != 'baseline':
        TRAIN_ARGS += ['--lora-scope', LORA_SCOPE]
    run_config['lora_scope'] = LORA_SCOPE
    if LOAD_4BIT:
        TRAIN_ARGS += ['--load-4bit']
    if MAX_SOURCE_HEIGHT:
        TRAIN_ARGS += ['--max-source-height', str(MAX_SOURCE_HEIGHT)]
    assert MODEL_FAMILY == 'vlm' or not (RAW_IMAGES_SOURCE or CURRICULUM or EXPERTS), 'Recipe flags are VLM-only'
    if CURRICULUM:
        TRAIN_ARGS += ['--curriculum']
    if EXPERTS:
        TRAIN_ARGS += ['--experts']
    if RAW_IMAGES_SOURCE:
        TRAIN_ARGS += ['--train-images-alt', str(WORK / 'images_raw'), '--alt-prob', str(ALT_PROB)]
    run_config.update(curriculum=CURRICULUM, experts=EXPERTS, raw_images_source=RAW_IMAGES_SOURCE or None)
    assert AUX_LOSS == 'none' or MODEL_FAMILY == 'trocr', 'AUX_LOSS is TrOCR-only'
    if AUX_LOSS != 'none':
        TRAIN_ARGS += ['--aux-loss', AUX_LOSS, '--aux-weight', str(AUX_WEIGHT),
                       '--aux-head-lrate', str(AUX_HEAD_LRATE)]
    assert not (RECON_WEIGHT or ALIGN_WEIGHT) or AUX_LOSS == 'ctc', 'RECON/ALIGN build on AUX_LOSS ctc'
    if RECON_WEIGHT:
        TRAIN_ARGS += ['--recon-weight', str(RECON_WEIGHT)]
    if ALIGN_WEIGHT:
        TRAIN_ARGS += ['--align-weight', str(ALIGN_WEIGHT)]
    run_config.update(aux_loss=AUX_LOSS, aux_weight=AUX_WEIGHT, aux_head_lrate=AUX_HEAD_LRATE,
                      recon_weight=RECON_WEIGHT, align_weight=ALIGN_WEIGHT)
    # Fold-model runs only: a full-data retrain keeps every label.
    EXCLUDE_ARGS = [] if EXCLUDE_FOLD is None else ['--exclude-fold', str(EXCLUDE_FOLD)]
    run_config['exclude_fold'] = EXCLUDE_FOLD
    PSEUDO_ROWS = 0
    if PSEUDO_LABELS:
        pseudo = Path(PSEUDO_LABELS)
        if not pseudo.is_absolute():
            # A bare filename: find it in the attached private inputs, wherever Kaggle mounts them.
            matches = sorted(Path('/kaggle/input').rglob(PSEUDO_LABELS))
            assert len(matches) == 1, f'Expected one {PSEUDO_LABELS} under /kaggle/input, found {matches}'
            pseudo = matches[0]
        assert pseudo.is_file(), f'PSEUDO_LABELS not found: {pseudo}'
        pseudo_rows = read_csv(pseudo, ['ID', 'Target'])
        assert {row['ID'] for row in pseudo_rows} <= {row['ID'] for row in test_rows}, 'Pseudo-labels must be test IDs only.'
        PSEUDO_ROWS = len(pseudo_rows)
        run_config.update(pseudo_labels=str(pseudo), pseudo_label_rows=PSEUDO_ROWS,
                          pseudo_labels_sha256=hashlib.sha256(pseudo.read_bytes()).hexdigest())
        TRAIN_ARGS += ['--extra-train', str(pseudo)]
    INFER_ARGS = [*COMMON, '--processor', MODEL_PATH, '--batch-size', str(INFER_BATCH_SIZE),
                  '--max-new-tokens', str(MAX_NEW_TOKENS), '--num-beams', '1']
    if NO_REPEAT_NGRAM:
        INFER_ARGS += ['--no-repeat-ngram-size', str(NO_REPEAT_NGRAM)]
    if LOAD_4BIT:
        INFER_ARGS += ['--load-4bit']
    run_config['train_args'] = TRAIN_ARGS
    save_config()

    def infer(input_csv, adapter, output, metadata, log_name, extra=()):
        run([sys.executable, '-u', INFER_SCRIPT, *INFER_ARGS,
             '--input', input_csv, *(['--adapter', adapter] if adapter else []),
             '--output', output, '--metadata', metadata, *extra], log_name)
        meta = json.loads(Path(metadata).read_text())
        assert meta['empty_predictions'] == 0, 'Empty predictions: inspect the saved run before proceeding.'
        return meta

    def check_training(metadata, log_path):
        meta = json.loads(Path(metadata).read_text())
        assert math.isfinite(float(meta['train_loss'])), 'Nonfinite training loss'
        text = re.sub(r'\\x1b\\[[0-?]*[ -/]*[@-~]', '', Path(log_path).read_text())
        marker = 'loss target of first example:'
        targets = [line.partition(marker)[2].strip() for line in text.splitlines() if marker in line]
        assert targets, 'Loss target marker missing; inspect training log.'
        target = ast.literal_eval(targets[0])
        assert isinstance(target, str) and target.removesuffix('<|im_end|>\\n').strip()
        for forbidden in ('<|image_pad|>', '<|vision_start|>', '<|vision_end|>', '<|im_start|>',
                          'Transcribe this handwritten line of text exactly.'):
            assert forbidden not in target, f'Invalid loss mask: {forbidden}'
        return meta
    ''')
    md('''
    ## Smoke test or load an existing adapter

    Training mode runs eight training rows and eight held-out rows. It verifies finite
    loss, answer-only loss masking, and inference. Full work is blocked if the rough
    estimate exceeds `MAX_ESTIMATED_HOURS`. Estimates exclude setup and checkpoint I/O
    and can understate sustained cost; leave headroom in your available GPU budget.

    Inference mode needs `ADAPTER_PATH` and matching `SOURCE_TRAIN_METADATA` for a final
    adapter. For an intermediate checkpoint, provide its matching `SOURCE_RUN_CONFIG`
    instead. The latter supports this notebook's fold run or the existing Colab fold-0
    recipe; such scores are explicitly labeled partial training. This mode does not
    resume optimizer updates. Exact training resume is available through the embedded
    trainer's `--resume-from-checkpoint` CLI with unchanged original controls and state.
    ''')
    code('''
    SMOKE_OK = False
    ACTIVE_ADAPTER = None
    ACTIVE_SCOPE = 'fold'
    PARTIAL_TRAINING = False
    if MODE == 'train':
        SMOKE = RUN / 'smoke'
        SMOKE.mkdir()
        run([sys.executable, '-u', FINETUNE_SCRIPT, *TRAIN_ARGS,
             '--fold', str(FOLD), *EXCLUDE_ARGS, '--max-train', '8', '--epochs', '1',
             '--output', SMOKE / 'adapter', '--metadata', SMOKE / 'train_metadata.json'], 'smoke/train.log')
        smoke_train = check_training(SMOKE / 'train_metadata.json', SMOKE / 'train.log')
        smoke_infer = infer('Train.csv', SMOKE / 'adapter', SMOKE / 'predictions.csv',
                            SMOKE / 'infer_metadata.json', 'smoke/infer.log',
                            ['--fold-manifest', 'data/splits/folds.csv', '--fold', str(FOLD), '--max-samples', '8'])
        assert smoke_infer['samples'] == 8
        assert all(math.isfinite(smoke_infer['score'][key]) for key in ['cer', 'wer', 'combined'])
        total_train_rows = ((0 if FULL_DATA_ONLY else len(expected_fit_rows)) + (len(train_rows) if TRAIN_FULL_DATA else 0)
                            + PSEUDO_ROWS * ((0 if FULL_DATA_ONLY else 1) + (1 if TRAIN_FULL_DATA else 0)))
        training_hours = smoke_train['runtime_seconds'] / 8 * total_train_rows * EPOCHS / 3600
        inference_rows = (0 if FULL_DATA_ONLY else len(val_rows)) + len(test_rows) + (8 if TRAIN_FULL_DATA else 0)
        inference_hours = smoke_infer['runtime_seconds'] / 8 * inference_rows / 3600
        run_config.update(estimated_training_hours=training_hours, estimated_inference_hours=inference_hours)
        save_config()
        print(f'Rough training: {training_hours:.2f} h; inference: {inference_hours:.2f} h')
        assert training_hours + inference_hours <= MAX_ESTIMATED_HOURS, 'Budget gate: stop and review the saved estimate.'
    else:
        assert ADAPTER_PATH, 'Set ADAPTER_PATH for inference mode.'
        ACTIVE_ADAPTER = Path(ADAPTER_PATH)
        adapter_config = json.loads((ACTIVE_ADAPTER / 'adapter_config.json').read_text())
        assert adapter_config.get('peft_type') == 'LORA'
        assert any((ACTIVE_ADAPTER / name).is_file() and (ACTIVE_ADAPTER / name).stat().st_size > 0
                   for name in ['adapter_model.safetensors', 'adapter_model.bin'])
        base_name = adapter_config.get('base_model_name_or_path', '')
        # A trained adapter records its base as the local snapshot directory, not the
        # repo id, so compare against the cache name MODEL_ID maps to. The old check
        # hard-coded the 2B name and therefore rejected every 3B and 7B adapter, making
        # infer mode unusable for them. The revision pin below still applies.
        cache_name = 'models--' + MODEL_ID.replace('/', '--')
        assert (base_name == MODEL_ID or cache_name in base_name
                or 'models--Qwen--Qwen2-VL-2B-Instruct' in base_name), 'Unexpected base model.'
        if '/snapshots/' in base_name:
            assert Path(base_name).name == MODEL_REVISION, 'Base-model snapshot differs from this adapter.'
        if SOURCE_TRAIN_METADATA:
            source_metadata_path = Path(SOURCE_TRAIN_METADATA)
            expected_adapter = source_metadata_path.parent / 'adapter'
            assert (ACTIVE_ADAPTER.resolve() == expected_adapter.resolve()
                    or expected_adapter.resolve() in ACTIVE_ADAPTER.resolve().parents), 'Metadata must belong to this adapter directory.'
            source_meta = json.loads(Path(SOURCE_TRAIN_METADATA).read_text())
            ACTIVE_SCOPE = 'full' if source_meta.get('full_data', False) else 'fold'
            expected_rows = train_rows if ACTIVE_SCOPE == 'full' else expected_fit_rows
            assert source_meta['train_ids_sha256'] == ids_hash(expected_rows), 'Source training rows mismatch.'
            assert source_meta['train_rows'] == len(expected_rows)
            assert source_meta['target_height'] == TARGET_HEIGHT and source_meta['max_pixels'] == MAX_PIXELS
            if ACTIVE_SCOPE == 'fold':
                assert source_meta['fold'] == FOLD
            shutil.copy2(SOURCE_TRAIN_METADATA, RUN / 'source_train_metadata.json')
        else:
            assert SOURCE_RUN_CONFIG, 'Provide matching metadata or run config; do not guess checkpoint provenance.'
            from road_ocr.checkpoints import validate_checkpoint
            state = validate_checkpoint(ACTIVE_ADAPTER)
            source_meta = json.loads(Path(SOURCE_RUN_CONFIG).read_text())
            if 'model_revision' in source_meta:
                assert source_meta['model_revision'] == MODEL_REVISION, 'Source model revision mismatch.'
            source_root = Path(SOURCE_RUN_CONFIG).resolve().parent
            relative_checkpoint = ACTIVE_ADAPTER.resolve().relative_to(source_root)
            expected_parents = {'fold_model/adapter/trainer', 'adapter/trainer'}
            assert relative_checkpoint.parent.as_posix() in expected_parents, 'Run config must belong to this fold checkpoint.'
            assert source_meta['fold_sha256'] == fold_hash
            old_args = source_meta.get('train_args', [])
            def old_setting(flag, key):
                if key in source_meta:
                    return source_meta[key]
                assert flag in old_args, f'Missing source control: {flag}'
                return old_args[old_args.index(flag) + 1]
            assert int(old_setting('--fold', 'fold')) == FOLD
            assert int(old_setting('--target-height', 'target_height')) == TARGET_HEIGHT
            assert int(old_setting('--max-pixels', 'max_pixels')) == MAX_PIXELS
            PARTIAL_TRAINING = True
            run_config['source_checkpoint_step'] = state['global_step']
            shutil.copy2(SOURCE_RUN_CONFIG, RUN / 'source_run_config.json')
        if (ACTIVE_ADAPTER / 'trainer_state.json').is_file():
            PARTIAL_TRAINING = True
        run_config.update(source_adapter=str(ACTIVE_ADAPTER), source_training_scope=ACTIVE_SCOPE,
                          partial_training=PARTIAL_TRAINING)
        # Use test images for the inference-only smoke: no validation leakage from a full-data model.
        smoke_infer = infer('Test.csv', ACTIVE_ADAPTER, RUN / 'smoke/predictions.csv',
                            RUN / 'smoke/infer_metadata.json', 'smoke/infer.log', ['--max-samples', '8'])
        assert smoke_infer['samples'] == 8
        remaining_rows = len(test_rows) + (len(val_rows) if ACTIVE_SCOPE == 'fold' else 0)
        estimate = smoke_infer['runtime_seconds'] / 8 * remaining_rows / 3600
        run_config['estimated_inference_hours'] = estimate
        save_config()
        print(f'Rough inference estimate: {estimate:.2f} h')
        assert estimate <= MAX_ESTIMATED_HOURS, 'Budget gate exceeded.'
    SMOKE_OK = True
    run_config['status'] = 'smoke_passed'
    save_config()
    ''')
    md('''
    ## Train the fold model from the public base weights

    Three fixed epochs; no validation-driven early stopping or checkpoint selection.
    The smoke adapter is discarded as a training starting point. Microbatch 4 and
    accumulation 4 give an effective batch of 16. Activation checkpointing is off by
    default: the measured line images are at most 552 visual tokens, so the recompute
    it trades for memory is not worth paying here. Set `GRADIENT_CHECKPOINTING=True`
    if the smoke runs out of memory. Note that microbatch 4 is not gradient-identical
    to microbatch 1 at the same effective batch, because each microbatch averages loss
    over its own tokens and these lines vary in length. Seeds are set before LoRA
    initialization; hardware-dependent numerical differences remain possible.
    ''')
    code('''
    assert SMOKE_OK
    if MODE == 'train' and not FULL_DATA_ONLY:
        ACTIVE_ADAPTER = RUN / 'fold_model/adapter'
        assert not ACTIVE_ADAPTER.exists()
        run_config['status'] = 'fold_training'
        save_config()
        run([sys.executable, '-u', FINETUNE_SCRIPT, *TRAIN_ARGS,
             '--fold', str(FOLD), *EXCLUDE_ARGS, '--epochs', str(EPOCHS), '--save-steps', str(SAVE_STEPS),
             '--output', ACTIVE_ADAPTER, '--metadata', RUN / 'fold_model/train_metadata.json'],
            'fold_model/train.log')
        train_meta = check_training(RUN / 'fold_model/train_metadata.json', RUN / 'fold_model/train.log')
        assert train_meta['train_ids_sha256'] == ids_hash(expected_fit_rows)
        assert train_meta['validation_rows'] == len(val_rows)
        run_config['status'] = 'fold_trained'
        save_config()
    print('Active adapter:', ACTIVE_ADAPTER)
    ''')
    md('''
    ## Score the complete held-out fold

    Report corpus CER, WER, and `0.5 * CER + 0.5 * WER`, lower is better, using the
    unchanged repository scorer and raw reference labels. The local metric remains
    the repository's working interpretation of organizer weighting. `1 - combined`
    is only a local leaderboard-style estimate. One fold cannot establish a five-fold
    mean/std; those fields remain unset. A full-data adapter is never scored on train.
    ''')
    code('''
    assert SMOKE_OK and (ACTIVE_ADAPTER is not None or FULL_DATA_ONLY)
    if ACTIVE_SCOPE == 'fold' and not FULL_DATA_ONLY:
        val_meta = infer('Train.csv', ACTIVE_ADAPTER, RUN / 'validation_predictions.csv',
                         RUN / 'validation_metadata.json', 'validation/infer.log',
                         ['--fold-manifest', 'data/splits/folds.csv', '--fold', str(FOLD)])
        assert val_meta['samples'] == len(val_rows)
        run(['make', 'score', f'PYTHON={sys.executable}',
             f'REF={RUN / "validation_reference.csv"}',
             f'PRED={RUN / "validation_predictions.csv"}'], 'validation/score.log')
        score = val_meta['score']
        assert all(math.isfinite(score[key]) for key in ['cer', 'wer', 'combined'])
        run_config.update(validation_score=score, fold_scores={str(FOLD): score},
                          fold_mean=None, fold_std=None, validation_scope='single held-out fold',
                          validation_result_type='measured partial-training' if PARTIAL_TRAINING else 'measured')
        threshold = run_config['success_threshold']
        run_config['hypothesis_passed'] = score['combined'] < threshold if threshold is not None else None
        print(json.dumps(score, indent=2))
        print('Local leaderboard-style estimate:', 1 - score['combined'])
        print('Partial training:', PARTIAL_TRAINING)
    else:
        run_config['validation_scope'] = 'skipped: source model trained on all labels'
        print('All-label model: no held-out validation score is available.')
    run_config['status'] = 'validation_stage_complete'
    save_config()
    ''')
    md('''
    ## Optional: GRPO against the metric, then rescore the same fold

    Supervised training optimizes next-token likelihood; the leaderboard scores
    `0.5 * CER + 0.5 * WER`. GRPO samples several transcriptions per line and rewards
    the ones with lower combined error, so the signal is the metric itself.

    Two things to hold in mind before reading the result. It continues the adapter that
    was just trained, so it necessarily trains on rows that adapter has already
    memorized, and a smoke run showed reward starting at 0.87-0.97 with little room to
    improve. And the measured errors on this data are perceptual rather than generative,
    so reweighting decodings the policy can already produce is not expected to recover
    much. The rescore below is the only thing that settles it.
    ''')
    code('''
    if RUN_GRPO and ACTIVE_SCOPE == 'fold' and not FULL_DATA_ONLY:  # GRPO needs the fold score, which FULL_DATA_ONLY skips
        assert 'validation_score' in run_config, 'GRPO needs the pre-GRPO fold score for comparison.'
        before = run_config['validation_score']['combined']
        GRPO_DIR = RUN / 'grpo'
        run([sys.executable, '-u', 'scripts/vlm_grpo.py',
             '--model-id', MODEL_PATH, '--processor', MODEL_PATH,
             '--adapter', ACTIVE_ADAPTER, '--output', GRPO_DIR,
             '--fold-manifest', 'data/splits/folds.csv', '--fold', str(FOLD),
             '--max-train', str(GRPO_ROWS), '--num-generations', str(GRPO_GENERATIONS),
             '--beta', str(GRPO_BETA), '--lrate', str(GRPO_LRATE),
             '--precision', PRECISION, '--target-height', str(TARGET_HEIGHT),
             '--max-pixels', str(MAX_PIXELS), '--batch-size', '1',
             '--grad-accum', str(GRPO_GENERATIONS), '--gradient-checkpointing',
             *GRPO_EXTRA_ARGS,
             '--metadata', GRPO_DIR / 'grpo_metadata.json'], 'grpo/train.log')
        grpo_val = infer('Train.csv', GRPO_DIR / 'adapter', RUN / 'validation_predictions_grpo.csv',
                         RUN / 'validation_metadata_grpo.json', 'grpo/infer.log',
                         ['--fold-manifest', 'data/splits/folds.csv', '--fold', str(FOLD)])
        assert grpo_val['samples'] == len(val_rows), 'GRPO rescore must cover the same rows.'
        after = grpo_val['score']['combined']
        run_config.update(grpo_score=grpo_val['score'], grpo_rows=GRPO_ROWS,
                          grpo_gain=before - after)
        print(json.dumps(grpo_val['score'], indent=2))
        print(f'GRPO changed combined {before:.6f} -> {after:.6f} ({before - after:+.6f}); '
              'a negative gain means it made the fold worse and the SFT adapter should be kept.')
        run_config['status'] = 'grpo_stage_complete'
        # Test inference below loads ACTIVE_ADAPTER, so keep GRPO's adapter only when it beat the
        # SFT adapter on the same held-out rows (chosen on fold 0, so that selection is optimistic).
        run_config['test_adapter'] = 'sft'
        if after < before:
            ACTIVE_ADAPTER = GRPO_DIR / 'adapter'
            run_config['test_adapter'] = 'grpo'
        print('Test predictions will use the', run_config['test_adapter'], 'adapter.')
        save_config()
    ''')
    md('''
    ## Optional fresh training on all labels

    Runs only when `MODE='train'` and `TRAIN_FULL_DATA=True`. This is a separate model
    initialized from the same public base weights, with the fixed recipe and all 4,098
    labels. Its test predictions are unscored. The earlier fold score continues to refer
    only to the fold model. The full-data model receives no training-set quality score.
    ''')
    code('''
    assert run_config['status'] in ('validation_stage_complete', 'grpo_stage_complete')
    if MODE == 'train' and TRAIN_FULL_DATA:
        ACTIVE_ADAPTER = RUN / 'full_data/adapter'
        assert not ACTIVE_ADAPTER.exists()
        run_config['status'] = 'full_data_training'
        save_config()
        run([sys.executable, '-u', FINETUNE_SCRIPT, *TRAIN_ARGS,
             '--full-data', '--epochs', str(EPOCHS), '--save-steps', str(SAVE_STEPS),
             '--output', ACTIVE_ADAPTER, '--metadata', RUN / 'full_data/train_metadata.json'],
            'full_data/train.log')
        full_meta = check_training(RUN / 'full_data/train_metadata.json', RUN / 'full_data/train.log')
        assert full_meta['train_ids_sha256'] == ids_hash(train_rows) and full_meta['validation_rows'] == 0
        full_smoke = infer('Test.csv', ACTIVE_ADAPTER, RUN / 'full_data/smoke_predictions.csv',
                           RUN / 'full_data/smoke_metadata.json', 'full_data/smoke_infer.log',
                           ['--max-samples', '8'])
        assert full_smoke['samples'] == 8
        ACTIVE_SCOPE = 'full'
    run_config.update(status='ready_for_test', submission_training_scope=ACTIVE_SCOPE)
    save_config()
    ''')
    md('''
    ## Infer every test row and strictly validate the submission

    No lexicon correction, hand edits, or leaderboard tuning is applied. Predictions
    must contain each test ID exactly once and a nonempty, nonsentinel transcription.
    Validation writes canonical `Test.csv` order. A failed check stops the notebook;
    predictions are never filled with placeholders to force a passing submission.
    ''')
    code('''
    assert run_config['status'] == 'ready_for_test'
    test_meta = infer('Test.csv', ACTIVE_ADAPTER, RUN / 'test_predictions.csv',
                      RUN / 'test_metadata.json', 'test/infer.log')
    assert test_meta['samples'] == len(test_rows)
    SUBMISSION = RUN / 'submission.csv'
    run([sys.executable, 'scripts/validate_submission.py', RUN / 'test_predictions.csv',
         '--test', 'Test.csv', '--output', SUBMISSION], 'test/canonicalize.log')
    run(['make', 'submission-check', f'PYTHON={sys.executable}', f'FILE={SUBMISSION}'], 'test/submission_check.log')
    submission_rows = read_csv(SUBMISSION, ['ID', 'Target'])
    assert [row['ID'] for row in submission_rows] == [row['ID'] for row in test_rows]
    assert len(index_unique(submission_rows, SUBMISSION)) == len(test_rows)
    run_config.update(status='completed', submission=str(SUBMISSION),
                      submission_sha256=hashlib.sha256(SUBMISSION.read_bytes()).hexdigest(),
                      submission_rows=len(submission_rows), test_score=None,
                      final_adapter=str(ACTIVE_ADAPTER), completed_utc=datetime.now(timezone.utc).isoformat())
    run_config['adapter_sha256'] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in Path(ACTIVE_ADAPTER).iterdir()
        if path.is_file() and path.suffix in {'.json', '.safetensors', '.bin'}
    }
    save_config()
    print(f'Validated {len(submission_rows)} rows: {SUBMISSION}')
    ''')
    md('''
    ## Retain the result and reproducibility artifacts

    Download `submission.csv` for manual Zindi upload. Keep the run folder private:
    it includes predictions, validation labels, folds, logs, environment, exact source,
    metadata, and trained adapters/checkpoints. In inference mode the source adapter
    stays in its input mount; retain that source alongside this run. Test quality is
    unknown until an authorized submission; this notebook does not claim a rank.

    To reuse a final adapter, attach its saved private run output in a new session,
    choose `MODE='infer'`, set `ADAPTER_PATH` and `SOURCE_TRAIN_METADATA` to the same
    model's files, and preserve the base-model revision and preprocessing controls.
    To evaluate an interrupted fold checkpoint, set `SOURCE_RUN_CONFIG` instead and
    retain the checkpoint's `trainer_state.json` and adapter files. Optimizer resume
    additionally requires complete optimizer/scheduler/RNG state, an FP16 scaler where
    applicable, and compatible original packages. See `baselines/vlm/README.md` for CLI details.
    ''')
    code('''
    assert run_config['status'] == 'completed'
    # File links are relative to the Kaggle output root for notebook download handling.
    from IPython.display import FileLink, display
    os.chdir('/kaggle/working')
    display(FileLink(str(SUBMISSION.relative_to('/kaggle/working'))))
    display(FileLink(str((RUN / 'run_config.json').relative_to('/kaggle/working'))))
    print('Retain the entire private run directory:', RUN)
    print('Validation:', run_config.get('validation_score', 'No held-out score for this adapter'))
    print('Test predictions are unscored. No upload was performed.')
    ''')
    if smoke_only:
        # Keep setup, data audit and the real eight-row train/inference smoke.
        # Omit every full-training, validation and submission cell physically,
        # so Run All cannot accidentally consume a full run's quota.
        smoke_cells = [i for i, cell in enumerate(cells)
                       if cell['cell_type'] == 'code'
                       and ''.join(cell['source']).startswith('SMOKE_OK = False')]
        assert len(smoke_cells) == 1
        cells[:] = cells[:smoke_cells[0] + 1]
        code('''
        assert MODE == 'train' and SMOKE_OK
        run_config.update(status='smoke_completed', execution_scope='eight-row smoke only',
                          validation_scope='not evaluated; smoke scores are not quality evidence')
        save_config()
        print('Smoke complete. No full training, fold score or submission was produced.')
        ''')
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "kaggle": {"accelerator": "gpu", "isInternetEnabled": True, "language": "python", "sourceType": "notebook"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail when the committed notebook differs from its source")
    parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                        help="override a config-cell constant, e.g. --set MODEL_ID=Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--config", type=Path, help="JSON object of config-cell overrides")
    parser.add_argument("--smoke-only", action="store_true",
                        help="omit full training, validation and submission stages")
    parser.add_argument("--output", type=Path, help="write elsewhere than the committed notebook")
    args = parser.parse_args()
    overrides = json.loads(args.config.read_text()) if args.config else {}
    if not isinstance(overrides, dict):
        parser.error('--config must contain a JSON object')
    for item in args.set:
        name, _, value = item.partition("=")
        if not _:
            raise SystemExit(f"--set expects NAME=VALUE, got {item!r}")
        try:
            # A bare revision hash like 66285546... is a string, not a number, so
            # fall back rather than guessing from the first character.
            overrides[name.strip()] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            overrides[name.strip()] = value
    if (overrides or args.smoke_only) and args.check:
        raise SystemExit("--check compares the committed notebook and cannot take overrides")
    global OUTPUT
    if args.output:
        OUTPUT = args.output if args.output.is_absolute() else (Path.cwd() / args.output)
    if args.smoke_only and overrides.get('MODE', 'train') != 'train':
        parser.error('--smoke-only requires training mode')
    content = json.dumps(build(overrides, smoke_only=args.smoke_only), indent=1, ensure_ascii=False) + "\n"
    if args.check:
        # Git identity describes the build snapshot. A later commit must not
        # invalidate an otherwise byte-identical source/recipe verification.
        def without_build_identity(value):
            notebook = json.loads(value)
            for cell in notebook['cells']:
                cell['source'] = [line for line in cell['source']
                                  if not line.startswith('SOURCE_PROVENANCE = ')]
            return notebook
        if not OUTPUT.is_file() or without_build_identity(OUTPUT.read_text()) != without_build_identity(content):
            raise SystemExit("Notebook is stale; run make kaggle-notebook")
        print("PASS: notebook source is current and every code cell compiles")
    else:
        OUTPUT.write_text(content)
        shown = OUTPUT.relative_to(ROOT) if OUTPUT.is_relative_to(ROOT) else OUTPUT
        print(f"Built {shown} ({len(content):,} characters; no competition data)")


if __name__ == "__main__":
    main()
