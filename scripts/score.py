#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from road_ocr.metrics import score_pairs
from road_ocr.records import CsvValidationError, index_unique, read_csv


def main() -> int:
    parser = argparse.ArgumentParser(description="Score ID,Target OCR predictions.")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--predictions", required=True)
    args = parser.parse_args()

    try:
        reference_rows = read_csv(args.reference, ["ID", "Target"])
        prediction_rows = read_csv(args.predictions, ["ID", "Target"])
        references = index_unique(reference_rows, args.reference)
        predictions = index_unique(prediction_rows, args.predictions)
    except (OSError, CsvValidationError) as exc:
        parser.error(str(exc))

    missing = sorted(references.keys() - predictions.keys())
    extra = sorted(predictions.keys() - references.keys())
    if missing or extra:
        parser.error(
            f"ID mismatch: missing={len(missing)} extra={len(extra)} "
            f"(examples missing={missing[:3]}, extra={extra[:3]})"
        )

    result = score_pairs(
        (row["Target"], predictions[record_id]["Target"])
        for record_id, row in references.items()
    )
    print(json.dumps(result.__dict__, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
