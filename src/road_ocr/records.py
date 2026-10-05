"""Strict CSV readers shared by command-line checks."""

from __future__ import annotations

import csv
from pathlib import Path


class CsvValidationError(ValueError):
    pass


def read_csv(path: str | Path, expected_columns: list[str]) -> list[dict[str, str]]:
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != expected_columns:
            raise CsvValidationError(
                f"{path}: expected columns {expected_columns}, got {reader.fieldnames}"
            )
        rows = list(reader)
    return rows


def index_unique(rows: list[dict[str, str]], path: str | Path) -> dict[str, dict[str, str]]:
    indexed: dict[str, dict[str, str]] = {}
    for row_number, row in enumerate(rows, start=2):
        record_id = row.get("ID", "")
        if not record_id:
            raise CsvValidationError(f"{path}:{row_number}: empty ID")
        if record_id in indexed:
            raise CsvValidationError(f"{path}:{row_number}: duplicate ID {record_id!r}")
        indexed[record_id] = row
    return indexed
