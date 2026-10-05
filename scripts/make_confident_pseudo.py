#!/usr/bin/env python3
"""High-confidence pseudo-labels: keep a voted line only if >= ceil(frac * N) members read exactly it.

The rule is fixed in advance (default frac 2/3), never tuned on labels. Writes <output>.csv
(ID,Target) and <output>.candidates.json (soft targets, same format as make_candidate_targets.py).
--exclude-ids refuses any ID that is a validation row. --diagnose REFERENCE scores the same rule on
labelled predictions (no output written): a report only, never used to set the threshold.
"""
import argparse
import csv
import json
import math
from pathlib import Path

from make_candidate_targets import candidates, read_predictions
from road_ocr.metrics import score_pairs


def confident(vote, members, frac=2 / 3):
    """{ID: support} for voted lines read exactly by >= ceil(frac * len(members)) members."""
    need = math.ceil(frac * len(members) - 1e-9)
    support = {i: sum(m[i].strip() == t.strip() for m in members) for i, t in vote.items() if t.strip()}
    return {i: s for i, s in support.items() if s >= need}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('vote')
    parser.add_argument('members', nargs='+')
    parser.add_argument('--frac', type=float, default=2 / 3)
    parser.add_argument('--output', help='path ending in .csv')
    parser.add_argument('--exclude-ids', help='ID,... CSV of validation rows that must not appear')
    parser.add_argument('--diagnose', metavar='REFERENCE', help='ID,Target labels: report only')
    args = parser.parse_args()
    vote = read_predictions(args.vote)
    members = [read_predictions(p) for p in args.members]
    if any(set(m) != set(vote) for m in members):
        raise ValueError('Members and vote cover different IDs')
    keep = confident(vote, members, args.frac)
    if args.diagnose:
        reference = read_predictions(args.diagnose)
        rows = [i for i in keep if i in reference]
        score = score_pairs((reference[i], vote[i]) for i in rows)
        print(json.dumps(dict(kept=len(keep), of=len(vote), exact=sum(reference[i] == vote[i] for i in rows) / max(1, len(rows)),
                              combined=score.combined, cer=score.cer, wer=score.wer)))
        return
    if not args.output:
        parser.error('--output is required unless --diagnose')
    if args.exclude_ids:
        with open(args.exclude_ids, encoding='utf-8-sig', newline='') as handle:
            leaked = keep.keys() & {r['ID'] for r in csv.DictReader(handle)}
        if leaked:
            raise ValueError(f'{len(leaked)} pseudo-labels are validation rows')
    output = Path(args.output)
    with output.open('x', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['ID', 'Target'])
        writer.writerows((i, vote[i]) for i in keep)
    output.with_suffix('.candidates.json').write_text(
        json.dumps({i: candidates(vote[i], [m[i] for m in members]) for i in keep}), encoding='utf-8')
    print(f'{len(keep)} of {len(vote)} lines (>= {math.ceil(args.frac * len(members) - 1e-9)}/{len(members)} members) -> {output}')


if __name__ == '__main__':
    main()
