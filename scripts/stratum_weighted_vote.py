#!/usr/bin/env python3
"""Fold-0 screen: per-height-stratum member weights in the MBR vote (gm_20261003_a).

Per row the winner is argmin_c sum_j w[stratum][j] * line_cost(c, member_j); candidates are the
members' own texts (pivot first), so weights only rerank. Weights come from a grid per stratum,
fit on one half of the 818 validation rows and applied to the other (2-fold cross-fit, fixed seeds).
The placebo repeats it with permuted stratum labels. Labels are used only on the fit half.
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, 'scripts')
from road_ocr.metrics import edit_distance
from vote_predictions import line_cost, read_predictions

GRID = (0.5, 1.0, 1.5)


def build(members, refs, ids):
    """cost[r, c, j] (inf-padded), char/word edits of each candidate vs the label, pools."""
    pools = [list(dict.fromkeys(m[i] for m in members)) for i in ids]
    k = max(map(len, pools))
    cost = np.full((len(ids), k, len(members)), 10**9, dtype=np.int64)  # exact: line_cost * 660
    ce, we = np.zeros((len(ids), k)), np.zeros((len(ids), k))
    for r, i in enumerate(ids):
        for c, cand in enumerate(pools[r]):
            cost[r, c] = [round(line_cost(cand, m[i]) * 660) for m in members]
            ce[r, c], we[r, c] = edit_distance(refs[r], cand), edit_distance(refs[r].split(), cand.split())
    return cost, ce, we, pools


def pick(cost, w):
    return np.argmin(cost @ np.rint(w * 2).astype(np.int64), axis=1)  # exact ints; first index wins ties = pivot first, as mbr_vote


def edits(ce, we, idx):
    rows = np.arange(len(idx))
    return ce[rows, idx], we[rows, idx]


def combined(c, w, rc, rw):
    return 0.5 * c.sum() / rc.sum() + 0.5 * w.sum() / rw.sum()


def fit(cost, ce, we, rc, rw, rows):
    """Best grid vector on `rows`; ties go to the vector closest to uniform."""
    best = None
    for w in itertools.product(GRID, repeat=cost.shape[2]):
        c, wd = edits(ce[rows], we[rows], pick(cost[rows], np.array(w)))
        key = (combined(c, wd, rc[rows], rw[rows]), sum(abs(x - 1) for x in w), w)
        best = min(best, key) if best else key
    return best[2]


def crossfit(cost, ce, we, rc, rw, strata, seed):
    rng = np.random.default_rng(seed)
    idx = np.zeros(len(strata), int)
    chosen = []
    half = np.zeros(len(strata), bool)
    for s in np.unique(strata):
        rows = rng.permutation(np.flatnonzero(strata == s))
        half[rows[: len(rows) // 2]] = True
    for h in (True, False):
        for s in np.unique(strata):
            fit_rows = np.flatnonzero((strata == s) & (half != h))
            app_rows = np.flatnonzero((strata == s) & (half == h))
            w = fit(cost, ce, we, rc, rw, fit_rows)
            chosen.append((int(s), w))
            idx[app_rows] = pick(cost[app_rows], np.array(w))
    c, wd = edits(ce, we, idx)
    return c, wd, chosen


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', required=True)
    p.add_argument('--members', nargs=6, required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--seeds', type=int, default=20)
    p.add_argument('--rows', type=int, help='smoke: use only the first N rows and 2 seeds')
    p.add_argument('--replicates', type=int, default=2000)
    p.add_argument('--images', default='images')
    a = p.parse_args()
    ref = read_predictions(a.reference)
    members = [read_predictions(m) for m in a.members]
    ids = sorted(ref)[: a.rows]
    refs = [ref[i] for i in ids]
    rc = np.array([len(t) for t in refs], float)
    rw = np.array([len(t.split()) for t in refs], float)
    strata = np.array([Image.open(f'{a.images}/{i}.jpg').height >= 150 for i in ids], int)
    cost, ce, we, _ = build(members, refs, ids)
    uni = np.ones(6)
    bc, bw = edits(ce, we, pick(cost, uni))
    base = combined(bc, bw, rc, rw)
    print('uniform baseline', base, 'rows', len(ids), 'tall', int(strata.sum()))
    seeds = range(2 if a.rows else a.seeds)
    out = {'baseline_combined': base, 'rows': len(ids), 'tall_rows': int(strata.sum()), 'seeds': list(seeds)}
    for name in ('real', 'placebo'):
        gains, cs, ws, chosen = [], [], [], []
        for sd in seeds:
            st = strata if name == 'real' else np.random.default_rng(10_000 + sd).permutation(strata)
            c, w, ch = crossfit(cost, ce, we, rc, rw, st, sd)
            gains.append(base - combined(c, w, rc, rw))
            cs.append(c); ws.append(w); chosen += ch
        mc, mw = np.mean(cs, 0), np.mean(ws, 0)  # per-row expected edits over seeds
        rng = np.random.default_rng(20260929)
        boot = []
        for _ in range(a.replicates):
            r = rng.integers(0, len(ids), len(ids))
            boot.append(combined(bc[r], bw[r], rc[r], rw[r]) - combined(mc[r], mw[r], rc[r], rw[r]))
        sub = lambda m: strata == m
        by = {k: float(combined(bc[sub(m)], bw[sub(m)], rc[sub(m)], rw[sub(m)])
                       - combined(mc[sub(m)], mw[sub(m)], rc[sub(m)], rw[sub(m)]))
              for k, m in (('short', 0), ('tall', 1))}
        stab = {}
        for s in (0, 1):
            vecs = [w for st_, w in chosen if st_ == s]
            top = max(set(vecs), key=vecs.count)
            stab[s] = {'modal': top, 'share': vecs.count(top) / len(vecs)}
        out[name] = {'seed_gains': gains, 'median_seed_gain': float(np.median(gains)),
                     'boot_gain_median': float(np.median(boot)),
                     'boot_ci95': [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                     'stratum_gain_expected': by, 'stability': stab}
        print(name, json.dumps(out[name], indent=1))
    full = {int(s): fit(cost, ce, we, rc, rw, np.flatnonzero(strata == s)) for s in (0, 1)}
    out['full_fit_weights'] = {str(k): v for k, v in full.items()}
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    tag = 'smoke' if a.rows else 'result'
    with open(f'{a.out_dir}/{tag}.json', 'x') as h:
        json.dump(out, h, indent=1)


if __name__ == '__main__':
    main()
