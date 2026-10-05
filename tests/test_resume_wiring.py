"""The resume notebook must resume the fold training, and only the fold training.

``TRAIN_ARGS`` is shared between the eight-row smoke and the full fold run. Appending
``--resume-from-checkpoint`` to it would make the smoke resume too, training eight rows
onto step 100 of the real trajectory and corrupting the thing being recovered. The flag
therefore goes on the fold training call alone, and this test holds that line.

The shared dry run in ``check_kaggle_notebook.py`` cannot cover this notebook: it has no
real checkpoint, so the notebook's own pre-flight assert fires. That assert is worth
keeping -- without it a wrong path silently trains from scratch for seven hours -- so
this test covers the wiring instead of weakening the guard.
"""
from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'baselines/vlm/runs_7b_resume/colab_7b_resume_fold0.ipynb'


def own_code(cell):
    """The cell's own lines, without the embedded repository source."""
    return ''.join(line for line in cell['source']
                   if not line.startswith(('SOURCE_FILES = ', 'SOURCE_PROVENANCE = ')))


def code_cells(notebook):
    return [own_code(c) for c in notebook['cells'] if c['cell_type'] == 'code']


@unittest.skipUnless(NOTEBOOK.is_file(), 'built notebook not present (restored-source runtime)')
class ResumeWiringTest(unittest.TestCase):
    def setUp(self):
        self.cells = code_cells(json.loads(NOTEBOOK.read_text()))

    def test_fold_training_resumes(self):
        fold = [c for c in self.cells
                if 'vlm_finetune.py' in c and "'--fold', str(FOLD), '--epochs'" in c]
        self.assertTrue(fold, 'no fold training call found')
        for cell in fold:
            self.assertIn('*RESUME_ARGS', cell, 'fold training does not resume')

    def test_full_data_training_never_resumes(self):
        """The all-label model is a separate model trained from the base weights.

        It shares TRAIN_ARGS with the fold run, so a resume flag placed there would
        wrongly continue the fold trajectory into a different model.
        """
        full = [c for c in self.cells if "'--full-data', '--epochs'" in c]
        self.assertTrue(full, 'no full-data training call found')
        for cell in full:
            self.assertNotIn('RESUME_ARGS', cell, 'full-data training must not resume')

    def test_the_smoke_never_resumes(self):
        """Eight smoke rows must not be trained onto the real trajectory."""
        smoke = [c for c in self.cells if "'--max-train', '8'" in c]
        self.assertTrue(smoke, 'no eight-row smoke call found')
        for cell in smoke:
            self.assertNotIn('RESUME_ARGS', cell, 'the smoke must train from base weights')

    def test_checkpoint_is_validated_before_the_model_download(self):
        """A bad path must fail on CPU, not after a 16.6GB pull and hours of training."""
        validate = [i for i, c in enumerate(self.cells)
                    if 'validate_checkpoint(RESUME_CHECKPOINT, resume=True)' in c]
        download = [i for i, c in enumerate(self.cells) if 'snapshot_download' in c]
        self.assertTrue(validate, 'checkpoint is never validated')
        self.assertTrue(download, 'model download cell not found')
        self.assertLess(min(validate), min(download),
                        'checkpoint validation must run before the model download')

    def test_epochs_unchanged_so_the_schedule_lines_up(self):
        """max_steps is recomputed from EPOCHS; 3 keeps it at the checkpoint's 615."""
        config = next(c for c in self.cells if c.startswith('# Configuration:'))
        self.assertRegex(config, r'(?m)^EPOCHS = 3\b', 'EPOCHS must stay 3 for an exact resume')
        self.assertIn('TOTAL_STEPS = 615', config)
        self.assertIn('RESUME_STEP = 100', config)

    def test_recipe_matches_the_original_run(self):
        """A resume that changes a hyperparameter is a different experiment."""
        original = json.loads((ROOT / 'baselines/vlm/runs_7b/config.json').read_text())
        resume = json.loads((ROOT / 'baselines/vlm/runs_7b_resume/config.json').read_text())
        narrative = {'HYPOTHESIS', 'CONTROLLED_CHANGE'}
        for key in set(original) | set(resume):
            if key in narrative:
                continue
            self.assertEqual(original.get(key), resume.get(key), f'{key} differs from runs_7b')


if __name__ == '__main__':
    unittest.main()
