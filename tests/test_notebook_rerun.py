"""Re-running a notebook cell must not be fatal, and must not overwrite a log.

In Colab, re-running a cell after a disconnect is routine. The shared ``run`` helper
opened every log with mode ``'x'`` so a rerun could not silently overwrite an earlier
command log, which is the right invariant, but raising on it turned the rerun itself
into ``FileExistsError: .../setup/kaggle_install.log`` with the run directory already
created. It was hit for real on the attention recovery notebook.

The helper now rolls to the next free suffix instead. This test execs the ``run``
defined in a built notebook rather than a copy of it, so it fails if the shipped
artifact loses either half of the behaviour: the rerun working, or the first log
surviving intact.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'baselines/vlm/runs_augment/colab_augment_fold0.ipynb'
MARKER = 'def run(command, log_name)'


def own_code(cell):
    """The cell's own lines, without the embedded repository source.

    SOURCE_FILES holds every tracked script as text, so a plain substring search over a
    whole cell matches whatever those scripts happen to contain -- including other
    builders that quote this very helper. That is the same trap that once made the dry
    run skip a cell, so match on the cell's own code only.
    """
    return ''.join(line for line in cell['source']
                   if not line.startswith(('SOURCE_FILES = ', 'SOURCE_PROVENANCE = ',
                                           'RECOVERY_PROVENANCE = ')))


def extract_run(notebook_path):
    """The notebook's own ``run`` helper, lifted out with its indentation removed."""
    notebook = json.loads(notebook_path.read_text())
    sources = [own_code(c) for c in notebook['cells'] if c['cell_type'] == 'code']
    holder = next(s for s in sources if MARKER in s)
    lines = holder[holder.index(MARKER):].splitlines(True)
    # The def line's own indent was stripped by the slice; keep the body, which is
    # every following line that is blank or indented past column zero.
    body = [lines[0]]
    for line in lines[1:]:
        if line.strip() and not line[0].isspace():
            break
        body.append(line)
    return textwrap.dedent(''.join(body))


# The notebook restores its embedded source and runs this suite inside the runtime,
# where no .ipynb is present: SOURCE_FILES carries scripts and tests, never notebooks.
# Skip there rather than failing the notebook's own self-check.
@unittest.skipUnless(NOTEBOOK.is_file(), 'built notebook not present (restored-source runtime)')
class NotebookRerunTest(unittest.TestCase):
    def test_rerun_rolls_the_log_and_keeps_the_first(self):
        source = extract_run(NOTEBOOK)
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / 'run'
            run_dir.mkdir()
            namespace = {
                'RUN': run_dir, 'WORK': Path(tmp), 'run_config': {},
                'save_config': lambda: None,
                'SESSION_DEADLINE': time.monotonic() + 3600,
                'time': time, 'json': json, 'subprocess': subprocess, 'threading': threading,
                'datetime': datetime, 'timezone': timezone, 'sys': sys,
            }
            exec(compile(source, 'notebook-run', 'exec'), namespace)
            run = namespace['run']

            run([sys.executable, '-c', "print('first')"], 'setup/kaggle_install.log')
            # The rerun is the regression: this used to raise FileExistsError.
            run([sys.executable, '-c', "print('second')"], 'setup/kaggle_install.log')

            first = run_dir / 'setup/kaggle_install.log'
            second = run_dir / 'setup/kaggle_install.2.log'
            self.assertTrue(first.is_file(), 'first log missing')
            self.assertTrue(second.is_file(), 'rerun did not roll to a new log')
            self.assertIn('first', first.read_text())
            self.assertIn('second', second.read_text())
            self.assertNotIn('second', first.read_text(), 'rerun overwrote the first log')

            logs = [json.loads(line)['log'] for line in
                    (run_dir / 'commands.jsonl').read_text().splitlines()]
            self.assertEqual(logs, [str(first), str(second)],
                             'commands.jsonl must record which log each invocation wrote')


if __name__ == '__main__':
    unittest.main()
