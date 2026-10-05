#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path

from road_ocr.records import index_unique, read_csv


def stable_tiebreaker(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def assign_folds(rows: list[dict[str, str]], folds: int, seed: int) -> dict[str, int]:
    """Keep exact duplicate labels together while balancing rows and characters."""

    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["Target"]].append(row)

    ordered_groups = sorted(
        groups.items(),
        key=lambda item: (
            -sum(len(row["Target"]) for row in item[1]),
            -len(item[1]),
            stable_tiebreaker(seed, item[0]),
        ),
    )
    fold_rows = [0] * folds
    fold_characters = [0] * folds
    assignments: dict[str, int] = {}

    for _label, group in ordered_groups:
        fold = min(range(folds), key=lambda i: (fold_characters[i], fold_rows[i], i))
        for row in group:
            assignments[row["ID"]] = fold
        fold_rows[fold] += len(group)
        fold_characters[fold] += sum(len(row["Target"]) for row in group)
    return assignments


def main() -> int:
    parser = argparse.ArgumentParser(description="Create deterministic grouped OCR folds.")
    parser.add_argument("--train", default="Train.csv")
    parser.add_argument("--output", default="data/splits/folds.csv")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260906)
    args = parser.parse_args()
    if args.folds < 2:
        parser.error("--folds must be at least 2")

    rows = read_csv(args.train, ["ID", "Target"])
    index_unique(rows, args.train)
    assignments = assign_folds(rows, args.folds, args.seed)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ID", "fold"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"ID": row["ID"], "fold": assignments[row["ID"]]})

    counts = [sum(fold == i for fold in assignments.values()) for i in range(args.folds)]
    print(f"wrote {len(assignments)} assignments to {output}")
    print(f"fold row counts: {counts}; seed={args.seed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
