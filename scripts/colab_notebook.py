#!/usr/bin/env python3
"""Turn a built Kaggle notebook into a free-Colab one, without changing the recipe.

Extracted unchanged from the 7B builder when the augmentation arm needed the same
conversion. Only the title, the run-id prefix, the cell-id prefix and the storage
roots differ between arms; the Drive mount, the path rewrites, the torchaudio
removal and the ten-hour subprocess deadline are the same every time.
"""
from __future__ import annotations

DRIVE_ROOT = '/content/drive/MyDrive/road'


def mount_cell(source_lines):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
            'source': source_lines}


def to_colab(notebook, *, title, run_prefix, cell_prefix, output_root):
    """Rewrite ``notebook`` in place for Colab and return it.

    ``title`` replaces the leading markdown cell, ``run_prefix`` replaces the
    Kaggle run-id prefix so artifacts never collide with another arm's, and
    ``output_root`` is where checkpoints persist.
    """

    mount = mount_cell([
        '# Connect existing private Drive inputs and persist checkpoints.\n',
        'from pathlib import Path\n',
        'if DATA_ROOT.startswith("/content/drive/") or OUTPUT_ROOT.startswith("/content/drive/"):\n',
        '    from google.colab import drive\n',
        '    drive.mount("/content/drive")\n',
        'assert Path(DATA_ROOT).is_dir(), "Set DATA_ROOT to inputs already available in this runtime."\n',
    ])
    for cell in notebook['cells']:
        source = ''.join(cell['source'])
        if cell['cell_type'] == 'markdown':
            if source.startswith('# R.O.A.D.'):
                cell['source'] = list(title)
            continue
        if source.startswith('# Configuration:'):
            cell['source'].append(f'OUTPUT_ROOT = "{output_root}"\n')
        rewritten = []
        for line in cell['source']:
            # Embedded files/provenance must remain byte-for-byte authoritative.
            if line.startswith(('SOURCE_FILES = ', 'SOURCE_PROVENANCE = ')):
                rewritten.append(line)
                continue
            line = line.replace("Path('/kaggle/working/experiments/runs')", 'Path(OUTPUT_ROOT)')
            line = line.replace("'kaggle_qwen2vl_'", f"'{run_prefix}'")
            line = line.replace('/kaggle/input', '/content')
            line = line.replace('Run this notebook inside Kaggle.', 'Run this notebook inside Colab.')
            line = line.replace("'-y', 'torchao'", "'-y', 'torchao', 'torchaudio'")
            rewritten.append(line)
        cell['source'] = rewritten
        source = ''.join(rewritten)
        if "os.chdir('/kaggle/working')" in source:
            # The closing cell hands Kaggle download links for files under its own
            # output root. On Colab that directory does not exist, so an unguarded
            # chdir crashes the last cell of a finished multi-hour run and makes a
            # completed experiment look failed. Artifacts persist on Drive here, so
            # print their paths instead.
            links = ("from IPython.display import FileLink, display\n"
                     "os.chdir('/kaggle/working')\n"
                     "display(FileLink(str(SUBMISSION.relative_to('/kaggle/working'))))\n"
                     "display(FileLink(str((RUN / 'run_config.json').relative_to('/kaggle/working'))))\n")
            assert source.count(links) == 1
            source = source.replace(links, "print('Submission:', SUBMISSION)\n"
                                           "print('Run config:', RUN / 'run_config.json')\n")
            source = source.replace(
                '# File links are relative to the Kaggle output root for notebook download handling.\n',
                '# Colab keeps these on Drive, so print paths rather than Kaggle download links.\n')
            cell['source'] = source.splitlines(True)
        if 'SESSION_ID = datetime.now' in source:
            # Kill a model subprocess at the elapsed deadline even if stdout stalls.
            source = source.replace('import time\n', 'import time\nimport threading\n')
            source = source.replace('WORK.mkdir(exist_ok=False)\n',
                                    'WORK.mkdir(exist_ok=False)\nSESSION_DEADLINE = time.monotonic() + 10 * 3600\n')
            old = 'text=True, bufsize=1) as process:\n'
            assert source.count(old) == 1
            source = source.replace(old, old +
                '                deadline_timer = threading.Timer(max(0, SESSION_DEADLINE - time.monotonic()), process.kill)\n'
                '                deadline_timer.daemon = True\n'
                '                deadline_timer.start()\n')
            source = source.replace('returncode = process.wait()\n',
                                    'returncode = process.wait()\n                deadline_timer.cancel()\n')
            source = source.replace('model_license=\'Apache-2.0\',',
                                    "runtime='free Colab T4', deadline_hours=10, model_license='Apache-2.0',")
            cell['source'] = source.splitlines(True)
    config_index = next(i for i, cell in enumerate(notebook['cells'])
                        if ''.join(cell['source']).startswith('# Configuration:'))
    notebook['cells'].insert(config_index + 1, mount)
    notebook['metadata'].pop('kaggle', None)
    notebook['metadata'].update(accelerator='GPU', road_runtime='colab')
    for i, cell in enumerate(notebook['cells']):
        cell['id'] = f'{cell_prefix}-{i:02d}'
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), f'cell-{i}', 'exec')
    return notebook
