#!/usr/bin/env python3
"""Exercise notebook orchestration on CPU with stubbed model calls, never train.

Uses the local competition inputs for the real audit/fold/submission checks.
All temporary derived files remain under ignored experiments/runs and are removed.
Model outputs and their scores are synthetic and are never experiment evidence.
"""
from __future__ import annotations

import contextlib
import importlib.metadata
import io
import json
import os
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from road_ocr.metrics import score_pairs
from road_ocr.records import read_csv
from road_ocr.lines import select_training_rows, split_by_fold


def skipped_cells(notebook):
    """Code cells a CPU dry run cannot execute, found by content rather than index.

    Two of them: the GPU/torch setup, which needs a real accelerator, and the final
    cell, which only renders download links through IPython. Everything else is
    orchestration and must run, so that inserting a stage cannot silently shorten
    what this check covers.
    """

    skip = {find_gpu_setup_cell(notebook)}
    skip |= {i for i, c in enumerate(notebook['cells'])
             if c['cell_type'] == 'code' and 'from IPython' in own_code(c)}
    skip |= {i for i, c in enumerate(notebook['cells'])
             if ''.join(c['source']).startswith('# Connect existing private Drive inputs')}
    return skip


def own_code(cell):
    """The cell's own lines, without the embedded repository source.

    SOURCE_FILES holds every tracked script as text, so a substring search over a
    whole cell matches whatever those scripts happen to mention. That is how a
    Colab-conversion helper quoting 'from IPython' silently made the restore cell
    look like the download-links cell, skipping it and leaving Path undefined
    several cells later. Match on the cell's own code only, as
    ``find_gpu_setup_cell`` already does by reading the first line.
    """

    return ''.join(line for line in cell['source']
                   if not line.startswith(('SOURCE_FILES = ', 'SOURCE_PROVENANCE = ')))


def find_gpu_setup_cell(notebook):
    """Index of the GPU/torch setup cell, the one cell a CPU dry run must skip.

    Matched on the cell's FIRST line. A substring search over whole cells hits the
    restore cell too, because that one embeds the repository's own source as text
    and several of those files mention sm_60 -- including this checker.
    """

    hits = [i for i, c in enumerate(notebook['cells'])
            if c['cell_type'] == 'code' and c['source'] and 'sm_60' in c['source'][0]]
    assert len(hits) == 1, f'expected exactly one GPU setup cell, found {hits}'
    return hits[0]


def stub_torch(device_arch, arch_list):
    """A torch whose CUDA device and build architectures are chosen by the caller."""
    module = types.ModuleType('torch')
    module.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        get_device_capability=lambda index=0: divmod(device_arch, 10),
        get_arch_list=lambda: list(arch_list),
        get_device_name=lambda index=0: f'Stub sm_{device_arch}',
    )
    return module


def check_install_pins(notebook, temporary):
    """The GPU setup cell never runs here, so check its pin logic directly."""
    source = ''.join(notebook['cells'][find_gpu_setup_cell(notebook)]['source'])
    assert "'-c', CONSTRAINTS" in source, 'Install must constrain the preinstalled CUDA build'
    installed = {'torch': '2.6.0+cu124'}  # torchvision is absent on purpose

    def version(name):
        if name not in installed:
            raise importlib.metadata.PackageNotFoundError(name)
        return installed[name]

    def run_prefix(destination, device_arch, arch_list=('sm_70', 'sm_75', 'sm_80')):
        namespace = {'RUN': destination, 'run_config': {}}
        prefix = source[:source.index('run([')]
        with contextlib.redirect_stdout(io.StringIO()), \
                patch('importlib.metadata.version', version), \
                patch.dict(sys.modules, {'torch': stub_torch(device_arch, arch_list)}):
            exec(compile(prefix, 'gpu-setup-prefix', 'exec'), namespace)
        return namespace

    namespace = run_prefix(temporary / 'pin_check', 75)
    assert namespace['run_config']['runtime_constraints'] == ['torch==2.6.0+cu124']
    assert namespace['CONSTRAINTS'].read_text() == 'torch==2.6.0+cu124\n'
    # A Kaggle P100 is sm_60: no kernels in this build, so the cell must stop before installing.
    try:
        run_prefix(temporary / 'pin_check_p100', 60)
    except AssertionError as exc:
        assert 'sm_60' in str(exc), exc
    else:
        raise AssertionError('Setup cell accepted a GPU this torch build has no kernels for')


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--notebook', type=Path,
                        default=ROOT / 'baselines/vlm/kaggle_train_infer.ipynb')
    args = parser.parse_args()
    notebook = json.loads(args.notebook.read_text())
    smoke_only = any("execution_scope='eight-row smoke only'" in ''.join(c['source'])
                     for c in notebook['cells'])
    if notebook['metadata'].get('road_runtime') == 'colab':
        # A Colab runtime has no /kaggle tree, so any surviving Kaggle path is a
        # crash waiting for whoever runs it. The dry run alone would not catch one
        # in a skipped cell, which is exactly where the first of these hid.
        stray = {index: line.strip() for index, cell in enumerate(notebook['cells'])
                 if cell['cell_type'] == 'code'
                 for line in own_code(cell).splitlines() if '/kaggle/' in line}
        assert not stray, f'Colab notebook still reads or writes Kaggle paths: {stray}'
    for index, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), f'cell-{index}', 'exec')
            assert cell['outputs'] == [] and cell['execution_count'] is None
    try:
        import nbformat
        nbformat.validate(notebook)
    except ImportError:
        pass
    output_root = ROOT / 'experiments/runs'
    output_root.mkdir(parents=True, exist_ok=True)
    original_env, original_path, original_cwd = dict(os.environ), list(sys.path), Path.cwd()
    try:
        with tempfile.TemporaryDirectory(prefix='notebook_cpu_check_', dir=output_root) as directory:
            temporary = Path(directory)
            input_root, working = temporary / 'input', temporary / 'working'
            input_root.mkdir()
            working.mkdir()
            (input_root / 'competition').symlink_to(ROOT, target_is_directory=True)
            model_path = temporary / 'models--Qwen--Qwen2-VL-2B-Instruct/snapshots/synthetic'
            model_path.mkdir(parents=True)
            fake_hub = types.ModuleType('huggingface_hub')
            fake_hub.snapshot_download = lambda **kwargs: str(model_path)

            def scenario(mode='train', full=False, adapter='', metadata='', source_config='', budget=10, full_only=False):
                ns = {'__name__': '__notebook_dry_run__'}
                model_calls = []

                def redirect(line):
                    """Point one line of notebook logic at this dry run's scratch dirs.

                    The embedded source is exempt. SOURCE_FILES holds the repository's
                    own files as text, and some of them legitimately contain the
                    literal '/kaggle/input' because they generate other notebooks.
                    Rewriting those characters corrupts the dict the notebook then
                    hashes against SOURCE_PROVENANCE, so a blanket replace made the
                    notebook fail its own integrity check.
                    """

                    if line.startswith(('SOURCE_FILES = ', 'SOURCE_PROVENANCE = ')):
                        return line
                    line = line.replace('/kaggle/input', str(input_root))
                    line = line.replace('/kaggle/working', str(working))
                    if notebook['metadata'].get('road_runtime') == 'colab':
                        line = line.replace('/content', str(input_root))
                    return line.replace("Path('/tmp')", f'Path({str(temporary)!r})')

                def execute(index):
                    source = ''.join(notebook['cells'][index]['source'])
                    source = '\n'.join(redirect(line) for line in source.split('\n'))
                    exec(compile(source, f'cell-{index}', 'exec'), ns)

                with contextlib.redirect_stdout(io.StringIO()), patch.dict(sys.modules, {'huggingface_hub': fake_hub}):
                    # Positional indices broke every time a stage was inserted, and the
                    # failure looked like a logic error rather than a stale index. Select
                    # the code cells by content instead: everything except the GPU/torch
                    # setup cell, which needs a real accelerator.
                    code_cells = [i for i, c in enumerate(notebook['cells'])
                                  if c['cell_type'] == 'code']
                    runnable = [i for i in code_cells if i not in skipped_cells(notebook)]
                    config_cell, restore_cell = runnable[0], runnable[1]
                    staged = runnable[2:]
                    execute(config_cell)
                    ns.update(DATA_ROOT=str(ROOT), IMAGES_SOURCE=str(ROOT / 'images'), MODE=mode,
                              TRAIN_FULL_DATA=full, ADAPTER_PATH=str(adapter),
                              SOURCE_TRAIN_METADATA=str(metadata), SOURCE_RUN_CONFIG=str(source_config),
                              MAX_ESTIMATED_HOURS=budget, FULL_DATA_ONLY=full_only)
                    if ns.get('RAW_IMAGES_SOURCE'):
                        ns['RAW_IMAGES_SOURCE'] = str(ROOT / 'images')  # stand-in: same IDs as IMAGES_SOURCE
                    if ns.get('PSEUDO_LABELS'):
                        # The configured path lives on Kaggle; stand in 20 real test IDs.
                        pseudo = temporary / 'pseudo_labels.csv'
                        test_ids = [r['ID'] for r in read_csv(ROOT / 'Test.csv', ['ID'])][:20]
                        pseudo.write_text('ID,Target\n' + ''.join(f'{i},synthetic\n' for i in test_ids))
                        ns['PSEUDO_LABELS'] = str(pseudo)
                    if notebook['metadata'].get('road_runtime') == 'colab':
                        ns['OUTPUT_ROOT'] = str(working / 'experiments/runs')
                    execute(restore_cell)
                    ns['PRECISION'] = 'fp16'  # GPU setup intentionally skipped in CPU orchestration check.
                    execute(staged[0])  # real utility tests, image audit, frozen folds
                    real_run = ns['run']

                    def fake_run(command, log_name):
                        args = list(map(str, command))
                        if not any(script in args for script in ['scripts/vlm_finetune.py', 'scripts/vlm_infer.py', 'scripts/trocr_finetune.py', 'scripts/trocr_infer.py', 'scripts/vlm_grpo.py']):
                            return real_run(command, log_name)
                        model_calls.append(args)
                        def flag(name, default=None):
                            return args[args.index(name) + 1] if name in args else default
                        log = ns['RUN'] / log_name
                        log.parent.mkdir(parents=True, exist_ok=True)
                        output = Path(flag('--output'))
                        metadata_path = Path(flag('--metadata'))
                        metadata_path.parent.mkdir(parents=True, exist_ok=True)
                        if 'scripts/vlm_grpo.py' in args:
                            assert '--adapter' in args and Path(flag('--adapter')).is_dir(), 'GRPO needs the SFT adapter'
                            (output / 'adapter').mkdir(parents=True)
                            (output / 'adapter/adapter_config.json').write_text(json.dumps({
                                'peft_type': 'LORA', 'base_model_name_or_path': ns['MODEL_ID']}))
                            (output / 'adapter/adapter_model.safetensors').write_bytes(b'synthetic-not-model-weights')
                            log.write_text('CPU orchestration stub; no GRPO executed.\n')
                            metadata_path.write_text(json.dumps({'stub': True}))
                            return
                        if any(t in args for t in ('scripts/vlm_finetune.py', 'scripts/trocr_finetune.py')):
                            rows = ns['train_rows']
                            if '--full-data' in args:
                                selected, held_out = rows, []
                            else:
                                # The real selector, so --exclude-fold is mirrored exactly as the trainers do it.
                                selected, held_out = select_training_rows(
                                    rows, ns['WORK'] / 'data/splits/folds.csv', int(flag('--fold')),
                                    exclude_fold=int(flag('--exclude-fold')) if '--exclude-fold' in args else None)
                            if '--max-train' in args:
                                selected = selected[:int(flag('--max-train'))]
                            output.mkdir(parents=True)
                            (output / 'adapter_config.json').write_text(json.dumps({
                                'peft_type': 'LORA', 'base_model_name_or_path': ns['MODEL_ID']}))
                            (output / 'adapter_model.safetensors').write_bytes(b'synthetic-not-model-weights')
                            meta = dict(train_loss=0.5, runtime_seconds=0.01,
                                        train_rows=len(selected), validation_rows=len(held_out),
                                        train_ids_sha256=ns['ids_hash'](selected),
                                        fold=int(flag('--fold')) if '--fold' in args else None,
                                        full_data='--full-data' in args,
                                        target_height=ns['TARGET_HEIGHT'], max_pixels=ns['MAX_PIXELS'])
                            log.write_text("loss target of first example: 'synthetic<|im_end|>\\n'\n")
                        else:
                            source = Path(flag('--input'))
                            labeled = source.name == 'Train.csv'
                            rows = read_csv(source, ['ID', 'Target'] if labeled else ['ID'])
                            if '--fold' in args:
                                rows = split_by_fold(rows, ns['WORK'] / 'data/splits/folds.csv', int(flag('--fold')))[1]
                            if '--max-samples' in args:
                                rows = rows[:int(flag('--max-samples'))]
                            predictions = [{'ID': row['ID'], 'Target': 'synthetic dry run'} for row in rows]
                            ns['write_rows'](output, predictions)
                            meta = dict(samples=len(rows), empty_predictions=0, runtime_seconds=0.01)
                            if labeled:
                                meta['score'] = score_pairs((row['Target'], 'synthetic dry run') for row in rows).__dict__
                            log.write_text('CPU orchestration stub; no model executed.\n')
                        metadata_path.write_text(json.dumps(meta))
                    ns['run'] = fake_run
                    execute(staged[1])
                    for index in staged[2:]:
                        execute(index)
                    if smoke_only:
                        assert ns['run_config']['status'] == 'smoke_completed'
                        assert 'validation_score' not in ns['run_config']
                        assert not (ns['RUN'] / 'submission.csv').exists()
                        assert len(model_calls) == 2
                        assert '--max-train' in model_calls[0] and '--max-samples' in model_calls[1]
                        assert all('fold_model' not in ' '.join(call) for call in model_calls)
                        return ns
                    assert ns['run_config']['status'] == 'completed'
                    assert ns['run_config']['submission_rows'] == 1374
                    if mode == 'infer':
                        assert not any(any(t in call for t in ('scripts/vlm_finetune.py', 'scripts/trocr_finetune.py')) for call in model_calls)
                    if ns['ACTIVE_SCOPE'] == 'full' and mode == 'infer':
                        assert not any('Train.csv' in call for call in model_calls)
                        assert 'validation_score' not in ns['run_config']
                    if full:
                        assert ns['run_config']['submission_training_scope'] == 'full'
                        assert sum('--full-data' in call for call in model_calls) == 1
                    if ns.get('LORA_SCOPE', 'baseline') != 'baseline' and mode == 'train':
                        training = [c for c in model_calls if any(t in c for t in ('scripts/vlm_finetune.py', 'scripts/trocr_finetune.py'))]
                        assert training and all('--lora-scope' in c and ns['LORA_SCOPE'] in c for c in training), \
                            'LORA_SCOPE not passed to training'
                    if ns.get('PSEUDO_LABELS') and mode == 'train':
                        training = [c for c in model_calls if any(t in c for t in ('scripts/vlm_finetune.py', 'scripts/trocr_finetune.py')) and '--max-train' not in c]
                        assert training and all('--extra-train' in c for c in training), 'pseudo-labels not passed to training'
                        assert ns['run_config']['pseudo_label_rows'] == 20
                    if full_only:
                        # Only the 8-row smoke may touch a fold; no fold model, no validation pass.
                        assert not any('fold_model' in ' '.join(call) for call in model_calls)
                        assert not any('validation_predictions' in ' '.join(call) for call in model_calls)
                        assert 'validation_score' not in ns['run_config']
                    return ns

            check_install_pins(notebook, temporary)
            fold = scenario()
            if smoke_only:
                try:
                    scenario(budget=1e-9)
                except AssertionError as exc:
                    assert 'Budget gate' in str(exc)
                else:
                    raise AssertionError('Budget gate did not block smoke completion')
                print('PASS: smoke-only notebook audit, two bounded model calls, budget guard, no full training or submission.')
                print('Model calls were stubbed; no CUDA run or measured model score is claimed.')
                return
            full = scenario(full=True)
            if 'FULL_DATA_ONLY' in ''.join(notebook['cells'][0]['source'] + [''.join(c['source']) for c in notebook['cells']]):
                scenario(full=True, full_only=True)
            scenario('infer', adapter=full['ACTIVE_ADAPTER'], metadata=full['RUN'] / 'full_data/train_metadata.json')
            checkpoint = fold['ACTIVE_ADAPTER'] / 'trainer/checkpoint-50'
            checkpoint.mkdir(parents=True)
            for name in ['adapter_config.json', 'adapter_model.safetensors']:
                (checkpoint / name).write_bytes((fold['ACTIVE_ADAPTER'] / name).read_bytes())
            (checkpoint / 'trainer_state.json').write_text(json.dumps({'global_step': 50}))
            partial = scenario('infer', adapter=checkpoint, source_config=fold['RUN'] / 'run_config.json')
            assert partial['run_config']['validation_result_type'] == 'measured partial-training'
            try:
                scenario(budget=1e-9)
            except AssertionError as exc:
                assert 'Budget gate' in str(exc)
            else:
                raise AssertionError('Budget gate did not block full training')
            try:
                scenario('infer', adapter=full['ACTIVE_ADAPTER'], metadata=fold['RUN'] / 'fold_model/train_metadata.json')
            except AssertionError as exc:
                assert 'Metadata must belong' in str(exc)
            else:
                raise AssertionError('Unrelated adapter metadata was accepted')
        print('PASS: notebook syntax/schema, install pinning, default and full-data flows, all-label inference, partial-fold inference, budget and provenance guards.')
        print('Model calls were stubbed; no CUDA run or measured model score is claimed.')
    finally:
        os.chdir(original_cwd)
        os.environ.clear()
        os.environ.update(original_env)
        sys.path[:] = original_path


if __name__ == '__main__':
    main()
