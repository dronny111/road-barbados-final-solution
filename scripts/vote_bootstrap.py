#!/usr/bin/env python3
"""Paired bootstrap of a candidate vote against a baseline vote on the same labelled rows.

Resamples exact-label groups and recomputes corpus CER/WER inside each replicate, via
``paired_bootstrap`` in road_ocr/bootstrap.py. A gain is baseline minus candidate
error, so a positive gain means the candidate is better. The earlier vote screens in
HANDOFF.md were run ad hoc; this makes the screen a checked-in command.
"""
from __future__ import annotations

import argparse
import json

from road_ocr.bootstrap import paired_bootstrap
from road_ocr.metrics import score_pairs
from vote_predictions import read_predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True, help='ID,Target labels')
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--replicates', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=20260929)
    parser.add_argument('--stratum-height', type=int,
                        help='also report the gain on crops at least / below this source height')
    parser.add_argument('--images', default='images')
    parser.add_argument('--output')
    args = parser.parse_args()
    reference, baseline, candidate = (read_predictions(p) for p in (args.reference, args.baseline, args.candidate))
    if not set(reference) == set(baseline) == set(candidate):
        raise ValueError('Reference, baseline and candidate must cover the same IDs')
    ids = sorted(reference)
    refs = [reference[i] for i in ids]
    base, cand = [baseline[i] for i in ids], [candidate[i] for i in ids]
    summary = dict(
        reference=args.reference, baseline=args.baseline, candidate=args.candidate,
        rows=len(ids), replicates=args.replicates, seed=args.seed,
        baseline_score=vars(score_pairs(zip(refs, base))), candidate_score=vars(score_pairs(zip(refs, cand))),
        gain=paired_bootstrap(refs, base, cand, args.replicates, args.seed),
    )
    if args.stratum_height:
        from PIL import Image

        tall = {i for i in ids if Image.open(f'{args.images}/{i}.jpg').height >= args.stratum_height}
        for name, keep in (('loose', lambda i: i in tall), ('tight', lambda i: i not in tall)):
            sub = [k for k, i in enumerate(ids) if keep(i)]
            pick = lambda xs: [xs[k] for k in sub]
            summary[f'{name}_gain'] = paired_bootstrap(pick(refs), pick(base), pick(cand), args.replicates, args.seed)
            summary[f'{name}_rows'] = len(sub)
        summary['stratum_height'] = args.stratum_height
    text = json.dumps(summary, indent=2, sort_keys=True) + '\n'
    if args.output:
        with open(args.output, 'x') as handle:  # never overwrite a recorded screen
            handle.write(text)
    print(text, end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
