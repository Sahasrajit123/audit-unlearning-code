#!/usr/bin/env python3
"""
Split CIFAR-100 train batches into val, retain, and forget.

Reads data_dir/data_split/cifar100_no_forget/train/ (written by
download_cifar100_batches.py; override with --input_subfolder), carves off a validation
set, splits the remainder into retain and forget, and writes train/, val/, retain/,
forget/ as batch_XXXXX.pkl files of --batch_size points (the last batch of each split may
be smaller). test/ is copied over unchanged.

Splitting modes (--split_mode):
    'adversarial' (default; class-centric forget set)
        The forget set is built from whole classes, chosen greedily so the total is as
        close as possible to the target, with at most one class split between forget and
        retain to hit the target exactly. With the defaults this gives 4500 forget points
        over 10 of the 100 classes.
        Output: data_dir/data_split/cifar100_bs_{batch_size}/
    'uniform'
        The forget set is a uniformly random --forget_fraction of the (post-val) train set;
        the rest is retain. With the defaults this gives 4500 forget points spanning all
        100 classes.
        Output: data_dir/data_split/cifar100_uniform_bs_{batch_size}_seed{seed}/

--batch_size sets the size of each cached batch file. Forget batches are the unit the
experiments sample (a run trains on a random half of them, see
src/utils/data_cache.load_cifar_splits_with_batch_subset) and the unit the audit ranks,
so one split is generated per batch size.

--seed drives the train/val permutation, the forget selection, and the per-subset
shuffle before writing, so a given (mode, batch_size, seed) is reproducible.

Usage:
    python scripts/split_cifar100_train.py --batch_size 10
    python scripts/split_cifar100_train.py --batch_size 10 --split_mode uniform --seed 1
    python scripts/split_cifar100_train.py --batch_size 128 --val_split 0.1 --forget_fraction 0.1 --seed 1
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np


def load_batch_dir(batch_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load all batch_*.pkl from a directory; return (images, labels) concatenated."""
    files = sorted(batch_dir.glob("batch_*.pkl"))
    if not files:
        raise FileNotFoundError(f"No batch_*.pkl in {batch_dir}")
    images_list = []
    labels_list = []
    for p in files:
        with p.open("rb") as f:
            batch = pickle.load(f)
        imgs, labs = batch[0], batch[1]  # (images, labels) from download script
        images_list.append(imgs)
        labels_list.append(labs)
    return np.concatenate(images_list, axis=0), np.concatenate(labels_list, axis=0)


def save_batches(
    images: np.ndarray,
    labels: np.ndarray,
    out_dir: Path,
    batch_size: int,
    prefix: str = "batch",
) -> int:
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


def choose_forget_classes_to_match_target(
    train_labels: np.ndarray,
    target_forget_count: int,
    n_classes: int = 100,
    rng: np.random.Generator | None = None,
):
    """
    Choose whole classes for forget so total is <= target. If adding one more
    class would exceed target, use at most one "split" class: only enough
    points from that class so forget set equals target exactly; rest go to retain.

    Returns:
        forget_classes: set of class ids that are entirely in forget
        split_class: None or the one class that is split between forget/retain
        split_class_forget_count: how many points from split_class go to forget (0 if split_class is None)
    """
    if rng is None:
        rng = np.random.default_rng()
    counts = np.bincount(train_labels, minlength=n_classes)
    forget_classes = set()
    current_sum = 0
    split_class: int | None = None
    split_class_forget_count = 0

    # Greedy: add whole classes while we don't exceed target
    while True:
        best_c = None
        best_err = abs(current_sum - target_forget_count)
        for c in range(n_classes):
            if c in forget_classes:
                continue
            new_sum = current_sum + counts[c]
            err = abs(new_sum - target_forget_count)
            if err < best_err:
                best_err = err
                best_c = c
        if best_c is None:
            break
        new_sum = current_sum + counts[best_c]
        # If adding this class whole would exceed target, use one class as split (take only need points)
        if new_sum > target_forget_count:
            need = target_forget_count - current_sum
            if need > 0:
                # Prefer best_c if it has enough points; else pick any class with count >= need
                if counts[best_c] >= need:
                    split_class = best_c
                    split_class_forget_count = need
                else:
                    for c in range(n_classes):
                        if c in forget_classes:
                            continue
                        if counts[c] >= need:
                            split_class = c
                            split_class_forget_count = need
                            break
            break
        forget_classes.add(best_c)
        current_sum = new_sum
        if current_sum == target_forget_count:
            break
        if len(forget_classes) >= n_classes - 1:
            break

    return forget_classes, split_class, split_class_forget_count


def main() -> int:

    parser = argparse.ArgumentParser(
        description="Split CIFAR-100 train batches into val, retain, and forget (disjoint labels; forget ~target fraction)."
    )
    parser.add_argument(
        "--input_subfolder",
        type=str,
        default=None,
        help="Subfolder under data_split/ to use as input for splitting. Default: 'cifar100_no_forget' for both modes.",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("./data/cifar100"),
        help="Root directory; read/write data_dir/data_split/cifar100_bs_{batch_size}/",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for writing output pickles (default: 128)",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Fraction of full train to use as validation (default: 0.1)",
    )
    parser.add_argument(
        "--forget_fraction",
        type=float,
        default=0.1,
        help="Target fraction of (post-val) train for forget set; classes chosen so total is as close as possible (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for train/val split (default: 1)",
    )
    parser.add_argument(
        "--split_mode",
        type=str,
        choices=["adversarial", "uniform"],
        default="adversarial",
        help="Split mode for forget set: 'adversarial' (default, class-based) or 'uniform' (random 10% forget)",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    # Determine input subfolder
    if args.input_subfolder is not None:
        input_subfolder = args.input_subfolder
    else:
        input_subfolder = "cifar100_no_forget"
    input_batches_root = data_dir / "data_split" / input_subfolder
    train_dir = input_batches_root / "train"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Expected train dir {train_dir}. Run download_cifar100_batches.py or prepare the folder first.")

    # Set output folder for splits
    if args.split_mode == "uniform":
        batches_root = data_dir / "data_split" / f"cifar100_uniform_bs_{args.batch_size}_seed{args.seed}"
    else:
        batches_root = data_dir / "data_split" / f"cifar100_bs_{args.batch_size}"

    # Copy test folder unchanged to the new destination if it exists
    test_src = input_batches_root / "test"
    test_dst = batches_root / "test"
    if test_src.is_dir():
        import shutil
        if test_dst.exists():
            shutil.rmtree(test_dst)
        shutil.copytree(test_src, test_dst)
        print(f"Copied test folder from {test_src} to {test_dst}")

    full_train_images, full_train_labels = load_batch_dir(train_dir)
    n_full = full_train_images.shape[0]

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(n_full)
    n_val = int(round(args.val_split * n_full))
    n_train = n_full - n_val
    train_idx, val_idx = idx[n_val:], idx[:n_val]
    train_images = full_train_images[train_idx]
    train_labels = full_train_labels[train_idx]
    val_images = full_train_images[val_idx]
    val_labels = full_train_labels[val_idx]
    print(f"Train: {n_train} samples (after val split)")
    print(f"Val:   {n_val} samples")

    if args.split_mode == "uniform":
        # Uniform: randomly select 10% for forget, 90% for retain
        target_forget = int(round(args.forget_fraction * n_train))
        idx_train = np.arange(n_train)
        rng = np.random.default_rng(args.seed)
        perm = rng.permutation(idx_train)
        forget_idx = perm[:target_forget]
        retain_idx = perm[target_forget:]
        forget_images = train_images[forget_idx]
        forget_labels = train_labels[forget_idx]
        retain_images = train_images[retain_idx]
        retain_labels = train_labels[retain_idx]
        n_retain, n_forget = retain_images.shape[0], forget_images.shape[0]

        # Shuffle each subset (reproducible with seed)
        def shuffle_subset(imgs: np.ndarray, labs: np.ndarray) -> None:
            shuf = rng.permutation(imgs.shape[0])
            imgs[:] = imgs[shuf]
            labs[:] = labs[shuf]

        shuffle_subset(train_images, train_labels)
        shuffle_subset(val_images, val_labels)
        shuffle_subset(retain_images, retain_labels)
        shuffle_subset(forget_images, forget_labels)

        print(f"Retain: {n_retain} samples (uniform random)")
        print(f"Forget: {n_forget} samples (target {target_forget}) (uniform random)")

        train_out = batches_root / "train"
        val_out = batches_root / "val"
        retain_out = batches_root / "retain"
        forget_out = batches_root / "forget"

        n_train_b = save_batches(train_images, train_labels, train_out, args.batch_size)
        n_val_b = save_batches(val_images, val_labels, val_out, args.batch_size)
        n_retain_b = save_batches(retain_images, retain_labels, retain_out, args.batch_size)
        n_forget_b = save_batches(forget_images, forget_labels, forget_out, args.batch_size)

        print(f"Saved {n_train_b} train batches under {train_out}")
        print(f"Saved {n_val_b} val batches under {val_out}")
        print(f"Saved {n_retain_b} retain batches under {retain_out}")
        print(f"Saved {n_forget_b} forget batches under {forget_out}")
        return 0
    else:
        # Adversarial (class-based, current logic)
        target_forget = int(round(args.forget_fraction * n_train))
        forget_classes, split_class, split_class_forget_count = choose_forget_classes_to_match_target(
            train_labels, target_forget, n_classes=100, rng=rng
        )

        # Build forget mask: whole forget_classes + (for split class, only split_class_forget_count points)
        forget_mask = np.array([l in forget_classes for l in train_labels])
        if split_class is not None and split_class_forget_count > 0:
            split_indices = np.where(train_labels == split_class)[0]
            n_take = min(split_class_forget_count, len(split_indices))
            chosen = rng.choice(len(split_indices), size=n_take, replace=False)
            for i in chosen:
                forget_mask[split_indices[i]] = True
        retain_mask = ~forget_mask

        retain_images = train_images[retain_mask]
        retain_labels = train_labels[retain_mask]
        forget_images = train_images[forget_mask]
        forget_labels = train_labels[forget_mask]
        n_retain, n_forget = retain_images.shape[0], forget_images.shape[0]

        # Shuffle each subset (reproducible with seed)
        def shuffle_subset(imgs: np.ndarray, labs: np.ndarray) -> None:
            shuf = rng.permutation(imgs.shape[0])
            imgs[:] = imgs[shuf]
            labs[:] = labs[shuf]

        shuffle_subset(train_images, train_labels)
        shuffle_subset(val_images, val_labels)
        shuffle_subset(retain_images, retain_labels)
        shuffle_subset(forget_images, forget_labels)

        print(f"Retain: {n_retain} samples (classes not in forget set)")
        print(f"Forget: {n_forget} samples (target {target_forget}); whole classes {sorted(forget_classes)}")
        if split_class is not None and split_class_forget_count > 0:
            split_in_retain = np.sum((train_labels == split_class) & retain_mask)
            print(f"Split class {split_class}: {split_class_forget_count} points in forget, {split_in_retain} points in retain")

        train_out = batches_root / "train"
        val_out = batches_root / "val"
        retain_out = batches_root / "retain"
        forget_out = batches_root / "forget"

        n_train_b = save_batches(train_images, train_labels, train_out, args.batch_size)
        n_val_b = save_batches(val_images, val_labels, val_out, args.batch_size)
        n_retain_b = save_batches(retain_images, retain_labels, retain_out, args.batch_size)
        n_forget_b = save_batches(forget_images, forget_labels, forget_out, args.batch_size)

        print(f"Saved {n_train_b} train batches under {train_out}")
        print(f"Saved {n_val_b} val batches under {val_out}")
        print(f"Saved {n_retain_b} retain batches under {retain_out}")
        print(f"Saved {n_forget_b} forget batches under {forget_out}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
