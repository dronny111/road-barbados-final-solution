#!/usr/bin/env python3
"""Materialize deterministic fixed-fold path data for Kraken/ketos."""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from pathlib import Path

from road_ocr.records import CsvValidationError, index_unique, read_csv


def stable_order(seed: int, record_id: str) -> str:
    return hashlib.sha256(f"{seed}:{record_id}".encode()).hexdigest()


def read_fold_map(path: str | Path) -> dict[str, int]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["ID", "fold"]:
            raise CsvValidationError(
                f"{path}: expected columns ['ID', 'fold'], got {reader.fieldnames}"
            )
        fold_by_id: dict[str, int] = {}
        for row_number, row in enumerate(reader, start=2):
            record_id = row["ID"]
            if not record_id:
                raise CsvValidationError(f"{path}:{row_number}: empty ID")
            if record_id in fold_by_id:
                raise CsvValidationError(f"{path}:{row_number}: duplicate ID {record_id!r}")
            try:
                fold_by_id[record_id] = int(row["fold"])
            except ValueError as error:
                raise CsvValidationError(
                    f"{path}:{row_number}: invalid fold {row['fold']!r}"
                ) from error
    return fold_by_id


def select_rows(
    rows: list[dict[str, str]],
    fold_by_id: dict[str, int],
    validation_fold: int,
    seed: int,
    train_limit: int | None,
    validation_limit: int | None,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Return stable train/validation selections for one held-out fold."""

    missing = [row["ID"] for row in rows if row["ID"] not in fold_by_id]
    if missing:
        raise CsvValidationError(f"fold manifest is missing {len(missing)} IDs")
    extra = sorted(set(fold_by_id) - {row["ID"] for row in rows})
    if extra:
        raise CsvValidationError(f"fold manifest has {len(extra)} unknown IDs")
    train = [row for row in rows if fold_by_id[row["ID"]] != validation_fold]
    validation = [row for row in rows if fold_by_id[row["ID"]] == validation_fold]
    train.sort(key=lambda row: stable_order(seed, row["ID"]))
    validation.sort(key=lambda row: stable_order(seed, row["ID"]))
    if train_limit is not None:
        train = train[:train_limit]
    if validation_limit is not None:
        validation = validation[:validation_limit]
    if not train or not validation:
        raise CsvValidationError("training and validation selections must both be nonempty")
    return train, validation


def materialize(
    rows: list[dict[str, str]], image_dir: Path, output_dir: Path, manifest: Path
) -> None:
    """Create Kraken image/label pairs and a path manifest."""

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8") as handle:
        for row in rows:
            source = image_dir / f"{row['ID']}.jpg"
            if not source.is_file():
                raise FileNotFoundError(source)
            target = output_dir / source.name
            if target.exists() or target.is_symlink():
                target.unlink()
            try:
                target.symlink_to(source.resolve())
            except OSError:
                shutil.copyfile(source, target)
            label_path = target.with_suffix(".gt.txt")
            label_path.write_text(row["Target"], encoding="utf-8")
            handle.write(f"{target.absolute()}\n")


def write_labels(rows: list[dict[str, str]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ID", "Target"])
        writer.writeheader()
        writer.writerows(rows)


def prepare_run_directories(data_output: Path) -> None:
    """Create the data directory and Kraken's non-creating output parent."""

    data_output.mkdir(parents=True, exist_ok=True)
    (data_output.parent / "checkpoints").mkdir(parents=True, exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="Train.csv")
    parser.add_argument("--images", default="images")
    parser.add_argument("--fold-manifest", default="data/splits/folds.csv")
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--validation-limit", type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.train_limit is not None and args.train_limit <= 0:
        parser.error("--train-limit must be positive")
    if args.validation_limit is not None and args.validation_limit <= 0:
        parser.error("--validation-limit must be positive")

    rows = read_csv(args.train, ["ID", "Target"])
    index_unique(rows, args.train)
    train, validation = select_rows(
        rows,
        read_fold_map(args.fold_manifest),
        args.fold,
        args.seed,
        args.train_limit,
        args.validation_limit,
    )
    output = Path(args.output)
    prepare_run_directories(output)
    materialize(train, Path(args.images), output / "train", output / "train.txt")
    materialize(
        validation,
        Path(args.images),
        output / "validation",
        output / "validation.txt",
    )
    write_labels(train, output / "train_labels.csv")
    write_labels(validation, output / "validation_labels.csv")
    print(
        f"prepared train={len(train)} validation={len(validation)} "
        f"for fold={args.fold} under {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
