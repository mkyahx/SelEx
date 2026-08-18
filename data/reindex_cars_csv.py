"""Rewrite Stanford Cars CSV filenames to partition-local sequential names."""

import argparse
import csv
from pathlib import Path


def reindex_rows(rows):
    if not rows:
        return []

    converted = [rows[0]]
    for index, row in enumerate(rows[1:], start=1):
        if not row:
            continue
        suffix = Path(row[-1]).suffix or ".jpg"
        converted.append([*row[:-1], f"{index:05d}{suffix}"])
    return converted


def rewrite_csv(input_path, output_path):
    with open(input_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))

    converted = reindex_rows(rows)
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(converted)

    return max(len(converted) - 1, 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path,
                        help="Destination CSV. Defaults to a sibling file ending in .local.csv.")
    parser.add_argument("--in_place", action="store_true",
                        help="Replace input_csv atomically after creating a temporary file.")
    args = parser.parse_args()

    if args.in_place and args.output_csv is not None:
        parser.error("Use either --in_place or --output_csv, not both.")

    output_path = args.output_csv or args.input_csv.with_suffix(".local.csv")
    if args.in_place:
        temporary_path = args.input_csv.with_suffix(args.input_csv.suffix + ".tmp")
        count = rewrite_csv(args.input_csv, temporary_path)
        temporary_path.replace(args.input_csv)
        output_path = args.input_csv
    else:
        count = rewrite_csv(args.input_csv, output_path)

    print(f"Reindexed {count} rows: {output_path}")


if __name__ == "__main__":
    main()
