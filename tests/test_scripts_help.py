"""Every shipped script imports and prints its help, so a pruned tree cannot hide a missing dependency."""
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NO_ARGPARSE = {"make_figures.py"}  # runs on import of __main__ regardless of arguments


class ScriptsHelpTest(unittest.TestCase):
    def test_every_script_prints_help(self):
        env = dict(os.environ, PYTHONPATH=f"{ROOT / 'src'}{os.pathsep}{ROOT / 'scripts'}", KMP_DUPLICATE_LIB_OK="TRUE")
        for path in sorted((ROOT / "scripts").glob("*.py")):
            if path.name in NO_ARGPARSE:
                continue
            with self.subTest(script=path.name):
                result = subprocess.run([sys.executable, str(path), "--help"], capture_output=True, text=True,
                                        env=env, cwd=ROOT)
                self.assertEqual(result.returncode, 0, result.stderr[-400:])


if __name__ == "__main__":
    unittest.main()
