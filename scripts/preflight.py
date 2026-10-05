#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

from road_ocr.records import CsvValidationError, index_unique, read_csv


def jpeg_has_markers(path: Path) -> bool:
    with path.open("rb") as handle:
        start = handle.read(2)
        handle.seek(-2, 2)
        end = handle.read(2)
    return start == b"\xff\xd8" and end == b"\xff\xd9"


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit local competition inputs.")
    parser.add_argument("--train", default="Train.csv")
    parser.add_argument("--test", default="Test.csv")
    parser.add_argument("--sample", default="SampleSubmission.csv")
    parser.add_argument("--images", default="images")
    parser.add_argument(
        "--full-images",
        action="store_true",
        help="Check JPEG markers and SHA-256 for every image",
    )
    args = parser.parse_args()

    errors: list[str] = []
    try:
        train_rows = read_csv(args.train, ["ID", "Target"])
        test_rows = read_csv(args.test, ["ID"])
        sample_rows = read_csv(args.sample, ["ID", "Target"])
        train = index_unique(train_rows, args.train)
        test = index_unique(test_rows, args.test)
        sample = index_unique(sample_rows, args.sample)
    except (OSError, CsvValidationError) as exc:
        print(f"FAIL: {exc}")
        return 1

    overlap = train.keys() & test.keys()
    if overlap:
        errors.append(f"train/test ID overlap: {len(overlap)}")
    if test.keys() != sample.keys():
        errors.append(
            f"sample/test ID mismatch: missing={len(test.keys()-sample.keys())}, "
            f"extra={len(sample.keys()-test.keys())}"
        )
    empty_labels = [record_id for record_id, row in train.items() if not row["Target"]]
    if empty_labels:
        errors.append(f"empty training labels: {len(empty_labels)}")

    image_dir = Path(args.images)
    image_paths = sorted(
        path for path in image_dir.glob("*.jpg") if not path.name.startswith("._")
    )
    image_ids = {path.stem for path in image_paths}
    expected_ids = train.keys() | test.keys()
    missing_images = expected_ids - image_ids
    extra_images = image_ids - expected_ids
    if missing_images:
        errors.append(f"missing images: {len(missing_images)} ({sorted(missing_images)[:3]})")
    if extra_images:
        errors.append(f"unreferenced images: {len(extra_images)} ({sorted(extra_images)[:3]})")

    duplicate_groups = 0
    broken_images = 0
    if args.full_images:
        hashes: dict[str, list[str]] = defaultdict(list)
        for path in image_paths:
            try:
                if not jpeg_has_markers(path):
                    broken_images += 1
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                hashes[digest].append(path.name)
            except OSError:
                broken_images += 1
        duplicate_groups = sum(len(names) > 1 for names in hashes.values())
        if broken_images:
            errors.append(f"unreadable or malformed JPEGs: {broken_images}")

    labels = [row["Target"] for row in train_rows]
    label_counts = Counter(labels)
    lengths = [len(label) for label in labels]
    print(f"train rows:              {len(train_rows)}")
    print(f"test rows:               {len(test_rows)}")
    print(f"sample rows:             {len(sample_rows)}")
    print(f"JPEG files:              {len(image_paths)}")
    print(f"unique labels:           {len(label_counts)}")
    print(f"duplicated-label rows:   {sum(n for n in label_counts.values() if n > 1)}")
    print(f"label length min/median/max: {min(lengths)}/{median(lengths):g}/{max(lengths)}")
    if args.full_images:
        print(f"exact duplicate image groups: {duplicate_groups}")

    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1
    print("PASS: competition inputs are internally consistent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
