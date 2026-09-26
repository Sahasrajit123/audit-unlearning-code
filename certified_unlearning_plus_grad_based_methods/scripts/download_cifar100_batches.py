#!/usr/bin/env python3
"""
Download CIFAR-100, extract it, and save train and test in batch pickle files.

Output (by default): out_dir/data_split/cifar100_no_forget/ with train/ and test/
Remainder is kept (last batch can be smaller). Use split_cifar100_train.py to split train into val, retain, and forget.

Uses only the standard library + numpy.

Each batch file is batch_XXXXX.pkl containing (images, labels) as numpy arrays:
  - images: (batch_size, 32, 32, 3) uint8, channels last
  - labels: (batch_size,) int64, fine class labels 0–99

Usage:
  python scripts/download_cifar100_batches.py
  python scripts/download_cifar100_batches.py --out_dir ./data/cifar100 --batch_size 128
"""

from __future__ import annotations

import argparse
import pickle
import sys
import tarfile
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np


CIFAR100_URL = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
CIFAR100_FILENAME = "cifar-100-python.tar.gz"


def download_if_needed(url: str, dest: Path) -> Path:
    if dest.exists():
        print(f"Using existing {dest}")
        return dest
    print(f"Downloading {url} -> {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, dest)
    return dest


def load_cifar100_batch(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load one CIFAR-100 file (train or test). Returns (images, fine_labels)."""
    with path.open("rb") as f:
        d = pickle.load(f, encoding="bytes")
    data = d[b"data"]  # (N, 3072) uint8
    fine_labels = np.array(d[b"fine_labels"], dtype=np.int64)
    n = data.shape[0]
    images = data.reshape(n, 3, 32, 32).transpose(0, 2, 3, 1)  # (N, 32, 32, 3)
    return images, fine_labels


def extract_and_get_paths(archive: Path, extract_dir: Path) -> tuple[Path, Path]:
    """Extract tar.gz and return (path_to_train_file, path_to_test_file)."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(extract_dir)
    base = extract_dir / "cifar-100-python"
    if not base.is_dir():
        raise FileNotFoundError(f"Expected {base} after extraction")
    return base / "train", base / "test"


def save_batches(
    images: np.ndarray,
    labels: np.ndarray,
    out_dir: Path,
    batch_size: int,
    prefix: str = "batch",
) -> int:
    """Split (images, labels) into batch files. Returns number of batch files written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = images.shape[0]
    num_batches = 0
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_images = images[start:end]
        batch_labels = labels[start:end]
        path = out_dir / f"{prefix}_{num_batches:05d}.pkl"
        with path.open("wb") as f:
            pickle.dump((batch_images, batch_labels), f, protocol=pickle.HIGHEST_PROTOCOL)
        num_batches += 1
    return num_batches


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download CIFAR-100, extract, and save train/test in batch pickles (no splitting)."
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("./data/cifar100"),
        help="Root directory for raw download + extracted files and batch output",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Number of samples per batch file (default: 128)",
    )
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    archive_path = out_dir / CIFAR100_FILENAME
    extract_dir = out_dir / "extracted"
    batches_root = out_dir / "data_split" / "cifar100_no_forget"

    download_if_needed(CIFAR100_URL, archive_path)

    train_path, test_path = extract_and_get_paths(archive_path, extract_dir)
    if not train_path.exists():
        raise FileNotFoundError(f"Expected {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Expected {test_path}")

    train_images, train_labels = load_cifar100_batch(train_path)
    print(f"Train: {train_images.shape[0]} samples, shape {train_images.shape}")

    test_images, test_labels = load_cifar100_batch(test_path)
    print(f"Test:  {test_images.shape[0]} samples, shape {test_images.shape}")

    train_dir = batches_root / "train"
    test_dir = batches_root / "test"
    n_train = save_batches(train_images, train_labels, train_dir, args.batch_size)
    n_test = save_batches(test_images, test_labels, test_dir, args.batch_size)
    print(f"Saved {n_train} train batches under {train_dir}")
    print(f"Saved {n_test} test batches under {test_dir}")
    print(f"Batch format: (images, labels) numpy; images (N, 32, 32, 3) uint8, labels (N,) int64")
    return 0


if __name__ == "__main__":
    sys.exit(main())
