#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from road_ocr.records import CsvValidationError, index_unique, read_csv


def validate(
    submission_path: str | Path,
    test_path: str | Path,
    *,
    allow_empty: bool = False,
    sample_path: str | Path | None = None,
) -> tuple[list[str], dict[str, dict[str, str]]]:
    if sample_path is not None:
        with Path(sample_path).open(encoding="utf-8", newline="") as handle:
            sample_header = next(csv.reader(handle), [])
        with Path(submission_path).open(encoding="utf-8", newline="") as handle:
            if next(csv.reader(handle), []) != sample_header:
                raise CsvValidationError(f"header differs from {sample_path}: {sample_header}")
    test_rows = read_csv(test_path, ["ID"])
    submission_rows = read_csv(submission_path, ["ID", "Target"])
    test = index_unique(test_rows, test_path)
    submission = index_unique(submission_rows, submission_path)

    missing = sorted(test.keys() - submission.keys())
    extra = sorted(submission.keys() - test.keys())
    if missing or extra:
        raise CsvValidationError(
            f"ID mismatch: missing={len(missing)} extra={len(extra)} "
            f"(examples missing={missing[:3]}, extra={extra[:3]})"
        )

    empty_ids = [record_id for record_id, row in submission.items() if not row["Target"].strip()]
    invalid_ids = [
        record_id
        for record_id, row in submission.items()
        if row["Target"].strip().lower() in {"nan", "none", "null"}
    ]
    if invalid_ids:
        raise CsvValidationError(
            f"invalid text sentinel in {len(invalid_ids)} rows: {invalid_ids[:5]}"
        )
    if empty_ids and not allow_empty:
        raise CsvValidationError(
            f"empty Target in {len(empty_ids)} rows: {empty_ids[:5]} "
            "(use --allow-empty only for diagnostic submissions)"
        )
    return list(test), submission


def main() -> int:
    parser = argparse.ArgumentParser(description="Strictly validate a Zindi submission.")
    parser.add_argument("submission")
    parser.add_argument("--test", default="Test.csv")
    parser.add_argument("--sample", default="SampleSubmission.csv")
    parser.add_argument("--allow-empty", action="store_true")
    parser.add_argument("--output", help="Write canonical Test.csv row order after validation")
    args = parser.parse_args()

    try:
        ordered_ids, submission = validate(
            args.submission, args.test, allow_empty=args.allow_empty,
            sample_path=args.sample if Path(args.sample).exists() else None,
        )
    except (OSError, CsvValidationError) as exc:
        parser.error(str(exc))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["ID", "Target"])
            writer.writeheader()
            for record_id in ordered_ids:
                writer.writerow(submission[record_id])
        print(f"PASS: wrote {len(ordered_ids)} canonical rows to {output_path}")
    else:
        qualifier = (
            "unique predictions (empty values allowed for diagnostics)"
            if args.allow_empty
            else "unique, non-empty predictions"
        )
        print(f"PASS: {args.submission} has {len(ordered_ids)} {qualifier}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
