#!/usr/bin/env python3
"""Majority vote across fold members' predictions, falling back to the first member.

Exact string agreement only: two members must decode a row identically before the vote
replaces the fallback member's text. No edit-distance blending, no threshold, no label
access. With three members a tie is impossible, so the rule needs no tie-break beyond
the fallback it already names.

``--mode word`` votes word by word instead (a small ROVER). Each member is aligned to the
fallback member by word-level edit distance; at every fallback word, and in every gap
between words, a strict majority of members replaces the fallback's choice. It helps when
members disagree on one word each, where the exact vote falls back for the whole line.

``--mode mbr`` picks, per line, the candidate with the lowest expected leaderboard loss
(minimum Bayes risk): the sum over members of ``lev_words/12 + lev_chars/55``, the
competition's own per-line cost. Candidates are the members' texts; ``mbr-vote`` also offers
the word-vote output as a candidate. Every member counts equally, and ties keep the earliest
candidate (the fallback first), so nothing is tuned.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from road_ocr.metrics import edit_distance
from road_ocr.records import CsvValidationError, index_unique, read_csv


def vote(members: list[dict[str, str]]) -> tuple[dict[str, str], Counter]:
    """members[0] is the fallback; every member must cover exactly the same IDs."""
    if len(members) < 3 or len(members) % 2 == 0:
        raise ValueError('Voting needs an odd number of members, at least three')
    fallback = members[0]
    for other in members[1:]:
        if set(other) != set(fallback):
            raise ValueError('Members cover different IDs')
    result, counts = {}, Counter()
    for record_id, text in fallback.items():
        texts = [member[record_id] for member in members]
        winner = next((t for t in texts if texts.count(t) >= 2), None)
        counts['unanimous' if len(set(texts)) == 1 else 'majority' if winner else 'fallback'] += 1
        result[record_id] = winner if winner is not None else text
    return result, counts


def align(pivot: list[str], other: list[str]):
    """Word alignment of other to pivot: (word or None per pivot slot, insertions per gap).

    Gap g holds words inserted before pivot word g; gap len(pivot) is the line end.
    Ties prefer a match/substitution, then a deletion, so the result is deterministic.
    """
    n, m = len(pivot), len(other)
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        cost[i][0] = i
    for j in range(m + 1):
        cost[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost[i][j] = min(cost[i - 1][j - 1] + (pivot[i - 1] != other[j - 1]),
                             cost[i - 1][j] + 1, cost[i][j - 1] + 1)
    slots, gaps = [None] * n, [[] for _ in range(n + 1)]
    i, j = n, m
    while i or j:
        if i and j and cost[i][j] == cost[i - 1][j - 1] + (pivot[i - 1] != other[j - 1]):
            slots[i - 1] = other[j - 1]
            i, j = i - 1, j - 1
        elif i and cost[i][j] == cost[i - 1][j] + 1:
            i -= 1  # other deletes pivot word i-1; its slot stays None
        else:
            gaps[i].insert(0, other[j - 1])
            j -= 1
    return slots, [tuple(g) for g in gaps]


def word_vote(members: list[dict[str, str]]) -> tuple[dict[str, str], Counter]:
    """members[0] is the pivot and fallback; a strict majority overrides it per word and gap."""
    if len(members) < 3 or len(members) % 2 == 0:
        raise ValueError('Voting needs an odd number of members, at least three')
    fallback = members[0]
    for other in members[1:]:
        if set(other) != set(fallback):
            raise ValueError('Members cover different IDs')
    need = len(members) // 2 + 1
    result, counts = {}, Counter()
    for record_id, text in fallback.items():
        pivot = text.split()
        aligned = [align(pivot, member[record_id].split()) for member in members[1:]]
        words = []
        for g in range(len(pivot) + 1):
            options = [()] + [gaps[g] for _, gaps in aligned]
            winner = next((o for o in options if options.count(o) >= need), ())
            words.extend(winner)
            if g < len(pivot):
                options = [pivot[g]] + [slots[g] for slots, _ in aligned]
                winner = next((o for o in options if options.count(o) >= need), pivot[g])
                if winner is not None:
                    words.append(winner)
        voted = ' '.join(words)
        changed = voted != ' '.join(pivot)
        counts['changed' if changed else 'unchanged'] += 1
        # An unchanged row keeps the fallback's exact text, spacing included.
        result[record_id] = voted if changed else text
    return result, counts


def line_cost(candidate: str, reference: str) -> float:
    """The leaderboard's per-line cost: word edits / 12 + character edits / 55."""
    return edit_distance(reference.split(), candidate.split()) / 12 + edit_distance(reference, candidate) / 55


def candidate_pools(members: list[dict[str, str]], with_word_vote: bool = False):
    """Per line, the MBR candidates (members' texts, then the word vote's), deduplicated, pivot first."""
    if len(members) < 3:
        raise ValueError('MBR needs at least three members')
    fallback = members[0]
    for other in members[1:]:
        if set(other) != set(fallback):
            raise ValueError('Members cover different IDs')
    voted = word_vote(members)[0] if with_word_vote and len(members) % 2 else None
    return {record_id: list(dict.fromkeys([member[record_id] for member in members]
                                          + ([voted[record_id]] if voted else [])))
            for record_id in fallback}, voted


def mbr_vote(members: list[dict[str, str]], with_word_vote: bool = False,
             lm=None, lm_weight: float = 0.0, scores=None, score_weight: float = 0.0) -> tuple[dict[str, str], Counter]:
    """Per line, the candidate minimizing summed line_cost against every member's text.

    ``lm`` (a road_ocr.char_lm.CharLM) adds ``lm_weight`` x its per-character NLL to each
    candidate's cost. ``scores`` ({ID: {candidate: NLL}}, e.g. trocr_score_candidates.py) adds
    ``score_weight`` x that NLL; a candidate missing from it raises KeyError. Both only rerank
    the members' own readings; they never write text.
    """
    lm_cost = (lambda c: lm_weight * lm.nll_per_char(c)) if lm is not None and lm_weight else (lambda c: 0)
    score_cost = (lambda i, c: score_weight * scores[i][c]) if scores is not None and score_weight else (lambda i, c: 0)
    pools, voted = candidate_pools(members, with_word_vote)
    result, counts = {}, Counter()
    for record_id, text in members[0].items():
        references = [member[record_id] for member in members]
        candidates = pools[record_id]
        best = min(candidates, key=lambda c: (sum(line_cost(c, r) for r in references) + lm_cost(c)
                                              + score_cost(record_id, c), candidates.index(c)))
        counts['kept' if best == text else 'word-vote' if voted and best == voted[record_id]
               and best not in references else 'switched'] += 1
        result[record_id] = best
    return result, counts


def read_predictions(path):
    rows = read_csv(path, ['ID', 'Target'])
    return {record_id: row['Target'] for record_id, row in index_unique(rows, path).items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('predictions', nargs='+',
                        help='Member CSVs; the first is the fallback. A comma-joined group '
                             '(original,view,view,...) is one member: its readings are first '
                             'reduced by their own MBR, ties keeping the first (original) reading')
    parser.add_argument('--output', required=True)
    parser.add_argument('--metadata')
    parser.add_argument('--mode', choices=['exact', 'word', 'mbr', 'mbr-vote'], default='exact')
    parser.add_argument('--lm-train', help='ID,Target CSV whose transcriptions fit the MBR character LM')
    parser.add_argument('--lm-weight', type=float, default=0.0)
    parser.add_argument('--lm-order', type=int, default=6)
    parser.add_argument('--scores', help='{ID: {candidate: NLL}} JSON from trocr_score_candidates.py')
    parser.add_argument('--score-weight', type=float, default=0.0)
    args = parser.parse_args()
    if (args.lm_train is None) != (args.lm_weight == 0) or (args.lm_train and not args.mode.startswith('mbr')):
        parser.error('--lm-train and a nonzero --lm-weight go together, with --mode mbr or mbr-vote')
    if (args.scores is None) != (args.score_weight == 0) or (args.scores and not args.mode.startswith('mbr')):
        parser.error('--scores and a nonzero --score-weight go together, with --mode mbr or mbr-vote')
    groups = [path.split(',') for path in args.predictions]
    if any(len(g) > 1 for g in groups) and not args.mode.startswith('mbr'):
        parser.error('view groups need --mode mbr or mbr-vote')
    if any(len(g) == 2 for g in groups):
        parser.error('a view group needs at least three readings for its own MBR')
    members = [read_predictions(g[0]) if len(g) == 1 else mbr_vote([read_predictions(p) for p in g])[0]
               for g in groups]
    if args.mode in ('mbr', 'mbr-vote'):
        lm = None
        if args.lm_train:
            from road_ocr.char_lm import CharLM
            from road_ocr.lines import clean_text
            lm = CharLM((clean_text(t) for t in read_predictions(args.lm_train).values()), args.lm_order)
        scores = json.loads(Path(args.scores).read_text(encoding='utf-8')) if args.scores else None
        voted, counts = mbr_vote(members, with_word_vote=args.mode == 'mbr-vote', lm=lm, lm_weight=args.lm_weight,
                                 scores=scores, score_weight=args.score_weight)
    else:
        voted, counts = (word_vote if args.mode == 'word' else vote)(members)
    empty = [record_id for record_id, text in voted.items() if not text.strip()]
    if empty:
        raise CsvValidationError(f'{len(empty)} voted rows are empty: {empty[:5]}')
    # The fallback member's order is the order its own run wrote, so the vote preserves it.
    with Path(args.output).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['ID', 'Target'])
        writer.writeheader()
        writer.writerows({'ID': record_id, 'Target': text} for record_id, text in voted.items())
    summary = dict(members=args.predictions, mode=args.mode, rows=len(voted), counts=dict(counts),
                   lm_train=args.lm_train, lm_weight=args.lm_weight, lm_order=args.lm_order if args.lm_train else None,
                   scores=args.scores, score_weight=args.score_weight,
                   changed_from_fallback=sum(voted[i] != members[0][i] for i in voted),
                   output=args.output)
    if args.metadata:
        Path(args.metadata).write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
