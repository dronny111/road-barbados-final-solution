#!/usr/bin/env python3
"""Fold-0 candidate-reranking stackers over the members' OOF lines (stack_20261003).

Members emit text, so a stacker cannot blend outputs: it scores each row's candidate lines (the
members' own texts, pivot first) and picks the lowest. Stackers: hill climbing (member and
likelihood weights), Ridge, Logistic, a listwise MLP, LightGBM regression and lambdarank.
Labels are used only on the training side of a repeated grouped CV (rows with the same label text
never straddle train and test); hyperparameters are tuned by an inner grouped CV. The placebo
trains on targets shuffled within each row. Fold 0 only, so every result is an upper bound.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, 'scripts')
sys.path.insert(0, 'src')
from road_ocr.char_lm import CharLM
from road_ocr.lines import clean_text, split_by_fold
from road_ocr.records import read_csv
from stratum_weighted_vote import build
from vote_predictions import read_predictions

STEPS = 20


def grouped_folds(labels, seed, k):
    """Fold id per row; identical label texts share a fold."""
    uniq = sorted(set(labels))
    order = np.random.default_rng(seed).permutation(len(uniq))
    fold_of = {uniq[j]: n % k for n, j in enumerate(order)}
    return np.array([fold_of[t] for t in labels])


def features(cost, nll, lm, pools, heights):
    """X[r, c, f] and mask[r, c]. First M+2 columns are raw: member costs, TrOCR NLL, char-LM NLL."""
    mask = np.array([[c < len(p) for c in range(cost.shape[1])] for p in pools])
    cst = np.where(mask[:, :, None], cost / 660.0, np.nan)
    nll = np.where(mask, nll, np.nan)
    lm = np.where(mask, lm, np.nan)
    length = np.where(mask, [[len(p[c]) if c < len(p) else 0 for c in range(cost.shape[1])] for p in pools], np.nan)
    mean, mn, mx, sd = np.nanmean(cst, 2), np.nanmin(cst, 2), np.nanmax(cst, 2), np.nanstd(cst, 2)
    rank = np.argsort(np.argsort(np.where(mask, mean, np.inf), 1), 1) / np.maximum(mask.sum(1, keepdims=True) - 1, 1)
    extra = [mean, mn, mx, sd, mean - np.nanmin(mean, 1, keepdims=True), rank, (cst == 0).sum(2).astype(float),
             length, length / np.nanmedian(length, 1, keepdims=True), nll / np.maximum(length, 1),
             nll - np.nanmin(nll, 1, keepdims=True), lm - np.nanmin(lm, 1, keepdims=True),
             np.broadcast_to(np.asarray(heights, float)[:, None], mask.shape)]
    X = np.concatenate([cst, nll[:, :, None], lm[:, :, None], np.stack(extra, 2)], 2)
    return np.nan_to_num(X, nan=0.0), mask


class Data:
    def __init__(self, X, mask, ce, we, rc, rw, labels, y=None):
        self.X, self.mask, self.ce, self.we, self.rc, self.rw, self.labels = X, mask, ce, we, rc, rw, labels
        self.y = y if y is not None else we / 12 + ce / 55  # the leaderboard's per-line cost

    def placebo(self, seed):
        """Same features, targets shuffled among each row's candidates."""
        rng = np.random.default_rng(seed)
        ce, we, y = self.ce.copy(), self.we.copy(), self.y.copy()
        for r in range(len(y)):
            v = np.flatnonzero(self.mask[r])
            p = rng.permutation(v)
            ce[r, v], we[r, v], y[r, v] = self.ce[r, p], self.we[r, p], self.y[r, p]
        return Data(self.X, self.mask, ce, we, self.rc, self.rw, self.labels, y)

    def combined(self, rows, idx):
        a = np.arange(len(rows))
        return (0.5 * self.ce[rows, idx][a].sum() / self.rc[rows].sum()
                + 0.5 * self.we[rows, idx][a].sum() / self.rw[rows].sum())


def choose(score, mask):
    return np.argmin(np.where(mask, score, np.inf), 1)  # first index wins ties = pivot


def hill_weights(D, tr, nm):
    """Greedy +1 steps (with replacement) on member / NLL / LM weights, starting from uniform."""
    Z = D.X[:, :, :nm + 2]
    step = np.array([1.0] * nm + [0.005, 0.1])
    w = np.array([1.0] * nm + [0.0, 0.0])
    best = D.combined(tr, choose(Z[tr] @ w, D.mask[tr]))
    for _ in range(STEPS):
        trial = [(D.combined(tr, choose(Z[tr] @ (w + step * np.eye(nm + 2)[j]), D.mask[tr])), j) for j in range(nm + 2)]
        score, j = min(trial)
        if score >= best - 1e-9:
            break
        best, w = score, w + step * np.eye(nm + 2)[j]
    return w


def hill(D, tr, te, seed, nm):
    return choose(D.X[te][:, :, :nm + 2] @ hill_weights(D, tr, nm), D.mask[te])


def flat(D, rows):
    m = D.mask[rows]
    return D.X[rows][m], D.y[rows][m], m


def unflat(pred, m):
    out = np.full(m.shape, np.inf)
    out[m] = pred
    return out


def best_flag(D, rows):
    y, m = D.y[rows], D.mask[rows]
    ymin = np.where(m, y, np.inf).min(1, keepdims=True)
    return (y <= ymin + 1e-12)[m], (y - ymin)[m]


def standardize(Xt):
    mu, sd = Xt.mean(0), Xt.std(0) + 1e-9
    return lambda A: (A - mu) / sd


def ridge(D, tr, te, seed, p):
    from sklearn.linear_model import Ridge
    Xt, yt, _ = flat(D, tr)
    f = standardize(Xt)
    m = Ridge(alpha=p).fit(f(Xt), yt)
    return choose(unflat(m.predict(f(D.X[te][D.mask[te]])), D.mask[te]), D.mask[te])


def logistic(D, tr, te, seed, p):
    from sklearn.linear_model import LogisticRegression
    Xt, _, _ = flat(D, tr)
    f = standardize(Xt)
    lab, _ = best_flag(D, tr)
    m = LogisticRegression(C=p, max_iter=500).fit(f(Xt), lab)
    return choose(unflat(-m.decision_function(f(D.X[te][D.mask[te]])), D.mask[te]), D.mask[te])


def mlp(D, tr, te, seed, p):
    import torch
    torch.manual_seed(seed)
    Xt, _, _ = flat(D, tr)
    f = standardize(Xt)
    X = torch.tensor(f(D.X[tr]), dtype=torch.float32)
    m = torch.tensor(D.mask[tr])
    y, m_np = D.y[tr], D.mask[tr]
    best = (y <= np.where(m_np, y, np.inf).min(1, keepdims=True) + 1e-12) & m_np
    target = torch.tensor(best / best.sum(1, keepdims=True), dtype=torch.float32)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[2], 16), torch.nn.ReLU(), torch.nn.Linear(16, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-2, weight_decay=p)
    for _ in range(80):
        opt.zero_grad()
        logit = net(X).squeeze(2).masked_fill(~m, -1e9)
        loss = -(target * torch.log_softmax(logit, 1)).sum(1).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        s = -net(torch.tensor(f(D.X[te]), dtype=torch.float32)).squeeze(2).numpy()
    return choose(s, D.mask[te])


def lgbm_reg(D, tr, te, seed, p):
    import lightgbm as lgb
    Xt, yt, _ = flat(D, tr)
    m = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, num_leaves=p, min_child_samples=20, subsample=0.8,
                          subsample_freq=1, colsample_bytree=0.8, random_state=seed, deterministic=True,
                          force_row_wise=True, verbose=-1).fit(Xt, yt)
    return choose(unflat(m.predict(D.X[te][D.mask[te]]), D.mask[te]), D.mask[te])


def lgbm_rank(D, tr, te, seed, p):
    import lightgbm as lgb
    Xt, _, m = flat(D, tr)
    _, diff = best_flag(D, tr)
    rel = np.where(diff <= 1e-12, 3, np.where(diff <= 0.1, 2, np.where(diff <= 0.3, 1, 0)))
    r = lgb.LGBMRanker(n_estimators=100, learning_rate=0.05, num_leaves=p, min_child_samples=20, subsample=0.8,
                       subsample_freq=1, colsample_bytree=0.8, random_state=seed, deterministic=True,
                       force_row_wise=True, verbose=-1).fit(Xt, rel, group=m.sum(1))
    return choose(unflat(-r.predict(D.X[te][D.mask[te]]), D.mask[te]), D.mask[te])


# name -> (function, hyperparameter grid or None). Grids are small and predeclared.
STACKERS = {'ridge': (ridge, (1.0, 10.0, 100.0)), 'logistic': (logistic, (0.01, 0.1, 1.0)),
            'mlp': (mlp, (1e-3, 1e-1)), 'lgbm_reg': (lgbm_reg, (3, 7)), 'lgbm_rank': (lgbm_rank, (3, 7))}


def tune(fn, grid, D, tr, seed):
    """Inner 3-fold grouped CV on the training rows; the grid value with the lowest combined edits."""
    f = grouped_folds([D.labels[r] for r in tr], seed + 7919, 3)
    scores = []
    for p in grid:
        idx = np.zeros(len(tr), int)
        for k in range(3):
            idx[f == k] = fn(D, tr[f != k], tr[f == k], seed, p)
        scores.append(D.combined(tr, idx))
    return grid[int(np.argmin(scores))]


def run_seed(D, Dtrain, seed, folds, nm, names):
    out = {n: np.zeros(len(D.labels), int) for n in names}
    f = grouped_folds(D.labels, seed, folds)
    for k in range(folds):
        tr, te = np.flatnonzero(f != k), np.flatnonzero(f == k)
        for n in names:
            if n == 'uniform':
                out[n][te] = choose(D.X[te, :, :nm].sum(2), D.mask[te])
            elif n == 'oracle':
                out[n][te] = choose(D.y[te], D.mask[te])
            elif n == 'hill':
                out[n][te] = hill(Dtrain, tr, te, seed, nm)
            else:
                fn, grid = STACKERS[n]
                out[n][te] = fn(Dtrain, tr, te, seed, tune(fn, grid, Dtrain, tr, seed))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', required=True)
    p.add_argument('--members', nargs='+', required=True)
    p.add_argument('--scores', required=True, help='trocr_score_candidates.py JSON covering every pool candidate')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--stackers', nargs='+', default=['hill', *STACKERS])
    p.add_argument('--seeds', type=int, default=20)
    p.add_argument('--placebo-seeds', type=int, default=5)
    p.add_argument('--rows', type=int, help='smoke: first N rows')
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--replicates', type=int, default=2000)
    p.add_argument('--images', default='images')
    p.add_argument('--train', default='Train.csv')
    p.add_argument('--fold-manifest', default='data/splits/folds.csv')
    a = p.parse_args()
    ref = read_predictions(a.reference)
    members = [read_predictions(m) for m in a.members]
    ids = sorted(ref)[: a.rows]
    refs = [ref[i] for i in ids]
    scores = json.load(open(a.scores))
    cost, ce, we, pools = build(members, refs, ids)
    nll = np.array([[scores[i][pools[r][c]] if c < len(pools[r]) else 0.0 for c in range(cost.shape[1])]
                    for r, i in enumerate(ids)])
    fit_rows, _ = split_by_fold(read_csv(a.train, ['ID', 'Target']), a.fold_manifest, 0)
    lm_model = CharLM((clean_text(r['Target']) for r in fit_rows), 6)  # folds 1-4 only
    lm = np.array([[lm_model.nll_per_char(pools[r][c]) if c < len(pools[r]) else 0.0 for c in range(cost.shape[1])]
                   for r in range(len(ids))])
    heights = [Image.open(f'{a.images}/{i}.jpg').height for i in ids]
    X, mask = features(cost, nll, lm, pools, heights)
    rc = np.array([len(t) for t in refs], float)
    rw = np.array([len(t.split()) for t in refs], float)
    D = Data(X, mask, ce, we, rc, rw, refs)
    nm = len(members)
    names = ['uniform', 'oracle', *a.stackers]
    seeds = list(range(2 if a.rows else a.seeds))
    tall = np.array(heights) >= 150
    res, preds0 = {}, {}
    for kind in ('real', 'placebo'):
        for sd in (seeds if kind == 'real' else seeds[: a.placebo_seeds]):
            Dt = D if kind == 'real' else D.placebo(50_000 + sd)
            run = run_seed(D, Dt, sd, a.folds, nm, names if kind == 'real' else a.stackers)
            for n, idx in run.items():
                res.setdefault((kind, n), []).append(idx)
            print(kind, 'seed', sd, {n: round(D.combined(np.arange(len(ids)), i), 5) for n, i in run.items()}, flush=True)
    out, rows_all = {'rows': len(ids), 'members': a.members, 'seeds': seeds, 'stackers': {}}, np.arange(len(ids))
    ub = res[('real', 'uniform')]
    bc = np.mean([ce[rows_all, i] for i in ub], 0)
    bw = np.mean([we[rows_all, i] for i in ub], 0)
    comb = lambda c, w, r: 0.5 * c[r].sum() / rc[r].sum() + 0.5 * w[r].sum() / rw[r].sum()
    for (kind, n), idxs in res.items():
        mc, mw = np.mean([ce[rows_all, i] for i in idxs], 0), np.mean([we[rows_all, i] for i in idxs], 0)
        seed_scores = [D.combined(rows_all, i) for i in idxs]
        gains_seed = [D.combined(rows_all, ub[s]) - D.combined(rows_all, i) for s, i in enumerate(idxs)]
        rng = np.random.default_rng(20260929)
        boot = []
        for _ in range(a.replicates):
            r = rng.integers(0, len(ids), len(ids))
            boot.append(comb(bc, bw, r) - comb(mc, mw, r))
        full = np.arange(len(ids))
        out['stackers'][f'{kind}:{n}'] = {
            'combined_mean': float(np.mean(seed_scores)), 'combined_std': float(np.std(seed_scores)),
            'cer': float(np.mean([ce[rows_all, i].sum() / rc.sum() for i in idxs])),
            'wer': float(np.mean([we[rows_all, i].sum() / rw.sum() for i in idxs])),
            'seed_gain_median': float(np.median(gains_seed)), 'boot_gain_median': float(np.median(boot)),
            'boot_ci95': [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            'tall_gain': float(comb(bc, bw, full[tall]) - comb(mc, mw, full[tall])),
            'short_gain': float(comb(bc, bw, full[~tall]) - comb(mc, mw, full[~tall])),
            'seeds_used': len(idxs)}
        if kind == 'real' and n not in ('oracle',):
            preds0[n] = idxs[0]
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    tag = 'smoke' if a.rows else 'result'
    with open(f'{a.out_dir}/{tag}.json', 'x') as h:
        json.dump(out, h, indent=1)
    for n, idx in preds0.items():  # seed-0 predictions, for `make score`
        with open(f'{a.out_dir}/{tag}_{n}_seed0.csv', 'x') as h:
            h.write('ID,Target\n')
            for r, i in enumerate(ids):
                t = pools[r][idx[r]]
                h.write(i + ',"' + t.replace('"', '""') + '"\n')
    for k, v in out['stackers'].items():
        print(k, {x: (round(y, 5) if isinstance(y, float) else y) for x, y in v.items() if x != 'seeds_used'})


if __name__ == '__main__':
    main()
