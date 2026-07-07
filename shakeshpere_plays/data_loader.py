"""
data_loader.py — load pre-computed text splits into DataLoaders

Expects the layout produced by prepare_data.py:
  data_splits/
    retain.txt, forget.txt, train.txt, val.txt, test.txt
    meta.json

Sliding window (x, y) pairs are constructed on the fly — nothing
is pre-sliced, so the files stay small (~MB not GB).
"""

import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from pathlib import Path


class FixedPermutationSampler(Sampler):
    """Shuffles indices once at construction; returns the same order every epoch."""

    def __init__(self, n, seed):
        g = torch.Generator()
        g.manual_seed(seed)
        self.indices = torch.randperm(n, generator=g).tolist()

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class CharDataset(Dataset):
    """Sliding window character dataset over a raw text string."""

    def __init__(self, text, seq_len, char2idx):
        self.seq_len = seq_len
        encoded = [char2idx[c] for c in text if c in char2idx]
        self.data = torch.tensor(encoded, dtype=torch.long)

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        x = self.data[idx     : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        return x, y


def load_data_splits(
    data_dir="data_splits",
    batch_size=256,
    seed=42,
    shuffle_train_samples=True,
    shuffle_train_batches_each_epoch=False,
):
    """
    Load pre-computed text splits and return DataLoaders + metadata.

    Returns:
        loaders  : dict with keys "retain", "forget", "train", "val", "test"
        metadata : dict with "char2idx", "idx2char", "vocab_size", "role2idx", etc.
    """
    data_dir = Path(data_dir)
    meta_path = data_dir / "meta.json"

    if not meta_path.exists():
        raise FileNotFoundError(
            f"Meta not found at {meta_path}. "
            f"Run: python prepare_data.py --output_dir {data_dir}"
        )

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    vocab = meta["vocab"]
    char2idx = {c: i for i, c in enumerate(vocab)}
    idx2char = {i: c for i, c in enumerate(vocab)}
    seq_len = meta["seq_len"]

    def worker_init_fn(worker_id):
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)

    loaders = {}
    split_sizes = {}

    forget_splits = meta.get("forget_split_names", ["forget"])  # e.g. ["forget"] or ["forget_0", "forget_1", ...]
    all_splits = ["retain"] + forget_splits + ["train", "val", "test"]

    for split in all_splits:
        txt_path = data_dir / f"{split}.txt"

        # Check new structure: forget files in subdirectory
        if split.startswith("forget_") and not txt_path.exists():
            txt_path = data_dir / "forget" / f"{split}.txt"

        if not txt_path.exists():
            raise FileNotFoundError(
                f"Split file not found: {txt_path}. "
                f"Run: python prepare_data.py --output_dir {data_dir}"
            )

        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()

        dataset = CharDataset(text, seq_len, char2idx)
        split_sizes[split] = len(dataset)

        is_train_split = split in (["retain", "train"] + forget_splits)
        if is_train_split and shuffle_train_batches_each_epoch:
            # Reshuffle every epoch (original behaviour)
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
                persistent_workers=True,
                worker_init_fn=worker_init_fn,
                generator=g,
            )
        elif is_train_split and shuffle_train_samples:
            # Shuffle once; same sample order every epoch
            sampler = FixedPermutationSampler(len(dataset), seed)
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=4,
                pin_memory=True,
                persistent_workers=True,
                worker_init_fn=worker_init_fn,
            )
        else:
            # No shuffling (eval splits, or explicitly disabled)
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
                persistent_workers=True,
                worker_init_fn=worker_init_fn,
            )
        loaders[split] = loader

    print(f"[loader] loaded splits from {data_dir}")
    for split, size in split_sizes.items():
        print(f"[loader]   {split:8s}: {size:,} samples")
    print(f"[loader]   vocab:    {meta['vocab_size']}")

    metadata = {
        "char2idx": char2idx,
        "idx2char": idx2char,
        "vocab_size": meta["vocab_size"],
        "role2idx": meta["role2idx"],
        "retain_roles": meta["retain_roles"],
        "forget_roles": meta["forget_roles"],
    }

    return loaders, metadata
