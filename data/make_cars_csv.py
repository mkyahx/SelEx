"""Generate the CSV indices required by SelEx's custom Stanford Cars loader."""

import argparse
import csv
from pathlib import Path

import numpy as np


HEADER = ["id", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "relative_im_path"]


def unwrap(value):
    """Convert MATLAB's nested one-element arrays into ordinary Python values."""
    while isinstance(value, np.ndarray):
        if value.size != 1:
            raise ValueError(f"Expected a scalar MAT field, got shape {value.shape}.")
        value = value.reshape(-1)[0]
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError("Expected a scalar-like list MAT field.")
        return unwrap(value[0])
    if isinstance(value, np.generic):
        return value.item()
    return value


def annotation_rows(annotations):
    """Return train/test CSV rows from the official ``cars_annos.mat`` annotations."""
    train_rows, test_rows = [], []
    for index, annotation in enumerate(annotations, start=1):
        relative_path = str(unwrap(annotation[0]))
        x1, y1, x2, y2 = (unwrap(annotation[position]) for position in range(1, 5))
        is_test = int(unwrap(annotation[-1]))
        row = [index, x1, y1, x2, y2, relative_path]
        (test_rows if is_test else train_rows).append(row)
    return train_rows, test_rows


def write_csv(rows, output_path):
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADER)
        writer.writerows(rows)


def main():
    try:
        from scipy.io import loadmat
    except ImportError as error:
        raise ImportError("Generating CSV files requires scipy. Install it with `pip install scipy`.") from error

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_root", type=Path, required=True,
                        help="Directory containing cars_annos.mat and the cars_train/cars_test folders.")
    args = parser.parse_args()

    mat_path = args.dataset_root / "cars_annos.mat"
    if not mat_path.is_file():
        raise FileNotFoundError(f"Missing annotation file: {mat_path}")

    annotations = loadmat(mat_path)["annotations"][0]
    train_rows, test_rows = annotation_rows(annotations)
    train_path = args.dataset_root / "cars_train.csv"
    test_path = args.dataset_root / "cars_test.csv"
    write_csv(train_rows, train_path)
    write_csv(test_rows, test_path)
    print(f"Wrote {len(train_rows)} rows to {train_path}")
    print(f"Wrote {len(test_rows)} rows to {test_path}")


if __name__ == "__main__":
    main()
