#!/usr/bin/env python3
"""Keep the test lines on which every given member decodes the identical string.

Agreement is the confidence signal: on fold 0 the two 7B fold-0 models agree on half the
lines, and those lines carry 0.071 combined error against 0.114 overall. Fully automated,
no thresholds tuned, no labels read. The output is derived competition data; keep it
under experiments/runs/ or a private dataset, never in Git.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from road_ocr.records import index_unique, read_csv


def read_predictions(path):
    rows = read_csv(path, ['ID', 'Target'])
    return {record_id: row['Target'] for record_id, row in index_unique(rows, path).items()}


def agreed(members):
    first = members[0]
    for other in members[1:]:
        if set(other) != set(first):
            raise ValueError('Members cover different IDs')
    return {i: t for i, t in first.items() if t.strip() and all(m[i] == t for m in members[1:])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('predictions', nargs='+', help='test prediction CSVs (ID,Target), at least two')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if len(args.predictions) < 2:
        parser.error('agreement needs at least two members')
    members = [read_predictions(p) for p in args.predictions]
    kept = agreed(members)
    with Path(args.output).open('x', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['ID', 'Target'])
        writer.writerows(kept.items())
    summary = dict(members=args.predictions, rows=len(members[0]), kept=len(kept),
                   sha256=hashlib.sha256(Path(args.output).read_bytes()).hexdigest(),
                   output=args.output)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
