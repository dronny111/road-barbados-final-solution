#!/usr/bin/env python3
"""Soft-distillation targets: every member's reading of a test line, weighted by vote support.

Writes <output>.csv (ID,Target = the vote, empty lines dropped) and <output>.candidates.json
({ID: [[text, weight], ...]}); vlm_finetune.py picks the sidecar up beside --extra-train.
A candidate's weight is the share of members that read it, plus one extra vote for the voted
string. No labels read. Output is derived competition data: keep it out of Git.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from road_ocr.records import index_unique, read_csv


def read_predictions(path):
    return {i: r['Target'] for i, r in index_unique(read_csv(path, ['ID', 'Target']), path).items()}


def candidates(vote_text, readings):
    counts = Counter(t for t in readings if t.strip())
    counts[vote_text] += 1
    total = sum(counts.values())
    return [[text, count / total] for text, count in sorted(counts.items())]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('vote')
    parser.add_argument('members', nargs='+')
    parser.add_argument('--output', required=True, help='path ending in .csv')
    args = parser.parse_args()
    vote = read_predictions(args.vote)
    members = [read_predictions(p) for p in args.members]
    if any(set(m) != set(vote) for m in members):
        raise ValueError('Members and vote cover different IDs')
    kept = {i: t for i, t in vote.items() if t.strip()}
    output = Path(args.output)
    with output.open('x', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['ID', 'Target'])
        writer.writerows(kept.items())
    output.with_suffix('.candidates.json').write_text(
        json.dumps({i: candidates(t, [m[i] for m in members]) for i, t in kept.items()}), encoding='utf-8')
    print(f'{len(kept)} of {len(vote)} lines -> {output}')


if __name__ == '__main__':
    main()
