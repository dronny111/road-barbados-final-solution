#!/usr/bin/env python3
"""Rebuild the submitted file from the ten members' prediction files.

Step ``score``: teacher-forced NLL of every candidate line under the fold-0 TrOCR adapter
(scripts/trocr_score_candidates.py), once for the fold-0 files and once for the test files.
Step ``stack``: hill-climbing reranker (scripts/stack_hill_test.py) fitted on the fold-0 files,
applied to the test files. ``--dry-run`` prints the commands. Never uploads anything.

The member prediction files and the TrOCR adapter are NOT shipped (they are derived from the
competition data); produce them with the notebooks in baselines/ first (docs/REPRODUCE.md).
"""
import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/final_ensemble.json")
    p.add_argument("--out-dir", default="output/final")
    p.add_argument("--step", choices=["score", "stack", "all"], default="all")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    cfg = json.loads((ROOT / a.config).read_text())
    fold0 = [m["fold0"] for m in cfg["members"]]
    test = [m["test"] for m in cfg["members"]]
    out = Path(a.out_dir)
    py = sys.executable
    score = lambda files, name: [
        py, "scripts/trocr_score_candidates.py", *files, "--model", cfg["scorer_adapter"],
        "--own-member", str(cfg["scorer_own_member"]), "--output", str(out / f"{name}_scores.json"),
        "--metadata", str(out / f"{name}_scores_meta.json")]
    stack = [py, "scripts/stack_hill_test.py", "--reference", cfg["reference"],
             "--oof-members", *fold0, "--oof-scores", str(out / "fold0_scores.json"),
             "--test-members", *test, "--test-scores", str(out / "test_scores.json"),
             "--out-dir", str(out / "hill")]
    plan = []
    if a.step in ("score", "all"):
        plan += [score(fold0, "fold0"), score(test, "test")]
    if a.step in ("stack", "all"):
        plan += [stack]

    if a.dry_run:
        for cmd in plan:
            print(" ".join(shlex.quote(x) for x in cmd))
        return 0
    needed = [cfg["reference"], cfg["scorer_adapter"], *fold0, *test, "Train.csv", "Test.csv", "data/splits/folds.csv"]
    missing = [x for x in needed if not (ROOT / x).exists()]
    if missing:
        print("missing inputs (see docs/REPRODUCE.md):\n  " + "\n  ".join(missing), file=sys.stderr)
        return 2
    (ROOT / out).mkdir(parents=True, exist_ok=True)
    (ROOT / out / "hill").mkdir(exist_ok=True)
    env = dict(os.environ, PYTHONPATH=f"{ROOT / 'src'}:{ROOT / 'scripts'}", KMP_DUPLICATE_LIB_OK="TRUE")
    for cmd in plan:
        subprocess.run(cmd, cwd=ROOT, env=env, check=True)
    if a.step in ("stack", "all"):
        built = ROOT / out / "hill" / "submission.csv"
        subprocess.run([py, "scripts/validate_submission.py", str(built)], cwd=ROOT, env=env, check=True)
        digest = sha256(built)
        same = digest == cfg["submission_sha256"]
        print(f"sha256 {digest}\nshipped {cfg['submission_sha256']}\nidentical: {same}")
        if not same:
            print("A different hash is expected if your members were retrained or your torch build differs: "
                  "the NLL scores are deterministic on one machine but not bit-identical across builds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
