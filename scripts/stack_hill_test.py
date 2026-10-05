#!/usr/bin/env python3
"""Test-side hill-climbing vote: fit weights on all fold-0 OOF rows, rerank the test candidates.

Members, candidate order and features match stack_candidates.py (7 members, pivot first, TrOCR-lik
NLL from the fold-0 scorer, char LM on folds 1-4). Weights never see test labels. Writes the weights
and a submission CSV; text is always one of the members' own readings.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, 'scripts')
sys.path.insert(0, 'src')
import stack_candidates as sc
from road_ocr.char_lm import CharLM
from road_ocr.lines import clean_text, split_by_fold
from road_ocr.records import read_csv
from stratum_weighted_vote import build
from vote_predictions import line_cost, read_predictions


def table(members, ids, scores, lm_model, images):
    pools = [list(dict.fromkeys(m[i] for m in members)) for i in ids]
    k = max(map(len, pools))
    cost = np.full((len(ids), k, len(members)), 10**9, dtype=np.int64)
    for r, i in enumerate(ids):
        for c, cand in enumerate(pools[r]):
            cost[r, c] = [round(line_cost(cand, m[i]) * 660) for m in members]
    nll = np.array([[scores[i][pools[r][c]] if c < len(pools[r]) else 0.0 for c in range(k)] for r, i in enumerate(ids)])
    lm = np.array([[lm_model.nll_per_char(pools[r][c]) if c < len(pools[r]) else 0.0 for c in range(k)] for r in range(len(ids))])
    heights = [Image.open(f'{images}/{i}.jpg').height for i in ids]
    return cost, nll, lm, pools, heights


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', required=True)
    p.add_argument('--oof-members', nargs='+', required=True)
    p.add_argument('--oof-scores', required=True)
    p.add_argument('--test-members', nargs='+', required=True)
    p.add_argument('--test-scores', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--images', default='images')
    p.add_argument('--train', default='Train.csv')
    p.add_argument('--fold-manifest', default='data/splits/folds.csv')
    a = p.parse_args()
    if len(a.oof_members) != len(a.test_members):
        p.error('member counts differ')
    nm = len(a.oof_members)
    ref = read_predictions(a.reference)
    ids = sorted(ref)
    refs = [ref[i] for i in ids]
    fit_rows, _ = split_by_fold(read_csv(a.train, ['ID', 'Target']), a.fold_manifest, 0)
    lm_model = CharLM((clean_text(r['Target']) for r in fit_rows), 6)
    cost, nll, lm, pools, heights = table([read_predictions(m) for m in a.oof_members], ids, json.load(open(a.oof_scores)), lm_model, a.images)
    _, ce, we, _ = build([read_predictions(m) for m in a.oof_members], refs, ids)
    X, mask = sc.features(cost, nll, lm, pools, heights)
    rc = np.array([len(t) for t in refs], float)
    rw = np.array([len(t.split()) for t in refs], float)
    D = sc.Data(X, mask, ce, we, rc, rw, refs)
    rows = np.arange(len(ids))
    w = sc.hill_weights(D, rows, nm)
    uni = D.combined(rows, sc.choose(X[:, :, :nm].sum(2), mask))
    fitted = D.combined(rows, sc.choose(X[:, :, :nm + 2] @ w, mask))
    print('weights', w.tolist(), 'in-sample combined uniform', uni, 'fitted', fitted)
    tm = [read_predictions(m) for m in a.test_members]
    tids = sorted(tm[0])
    tcost, tnll, tlm, tpools, theights = table(tm, tids, json.load(open(a.test_scores)), lm_model, a.images)
    TX, tmask = sc.features(tcost, tnll, tlm, tpools, theights)
    pick = sc.choose(TX[:, :, :nm + 2] @ w, tmask)
    uni_pick = sc.choose(TX[:, :, :nm].sum(2), tmask)
    out = Path(a.out_dir)
    with open(out / 'submission.csv', 'x', newline='') as h:
        wr = csv.writer(h)
        wr.writerow(['ID', 'Target'])
        for r, i in enumerate(tids):
            wr.writerow([i, tpools[r][pick[r]]])
    meta = {'weights': w.tolist(), 'oof_rows': len(ids), 'test_rows': len(tids), 'oof_members': a.oof_members,
            'test_members': a.test_members, 'in_sample_combined_uniform': uni, 'in_sample_combined_fitted': fitted,
            'rows_changed_vs_uniform_vote': int((pick != uni_pick).sum())}
    json.dump(meta, open(out / 'weights.json', 'x'), indent=1)
    print('rows changed vs uniform 7-member vote', meta['rows_changed_vs_uniform_vote'], 'of', len(tids))


if __name__ == '__main__':
    main()
