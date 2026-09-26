"""
Data loading for the retain/forget/val/test folder layout written by
the CIFAR-100 split scripts (see README). Two on-disk formats are supported, auto-detected per
`data_dir`:

  - flat `.pt` files: {data_dir}/{split}/{dataset_name}_{split}.pt, each holding
    {'data': Tensor, 'labels': Tensor} (e.g. this project's data/{train,retain,forget,val,test}).
  - per-file `.pkl` batches: {data_dir}/{split}/batch_*.pkl, each holding (data, labels)
    or {'data':..., 'labels':...} for a chunk of examples (e.g.
    data/cifar100/data_split/cifar100_bs_1). The concatenated tensors are cached to
    {data_dir}/.cache/{split}.pt on first load, since e.g. cifar100_bs_1's retain split
    alone is 40,500 files -- re-opening all of them on every run of a sweep would
    dominate wall-clock time.

`sample_forget` / `sample_forget_batches` are stage 1 of the train_and_unlearn.py
pipeline: they decide, independently of the training seed, which slice of the forget set
participates in a given run (retain set left unchanged, forget set subsampled).
"""
import glob
import math
import os
import pickle
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import ConcatDataset, TensorDataset


def _is_pkl_format(data_dir: str) -> bool:
    return len(glob.glob(os.path.join(data_dir, "retain", "*.pkl"))) > 0


def _load_pkl_folder(folder_path: str) -> Tuple[torch.Tensor, torch.Tensor, list]:
    """
    Concatenate every batch_*.pkl file in a folder into (data, labels) tensors, and also
    return `batch_sizes`: the sample count of each file, in the same sorted-filename order
    they were concatenated in. This preserves the on-disk batch boundaries (e.g. one file
    == one batch, whatever size it happens to be) instead of losing them once everything
    is flattened into a single tensor.
    """
    pkl_files = sorted(glob.glob(os.path.join(folder_path, "*.pkl")))
    if not pkl_files:
        raise ValueError(f"No .pkl files found in {folder_path}")

    all_data, all_labels, batch_sizes = [], [], []
    for pkl_file in pkl_files:
        with open(pkl_file, "rb") as f:
            payload = pickle.load(f)
        if isinstance(payload, dict):
            data = payload.get("data", payload.get("images"))
            labels = payload.get("labels", payload.get("targets"))
        else:
            data, labels = payload[0], payload[1]
        data = torch.as_tensor(data)
        all_data.append(data)
        all_labels.append(torch.as_tensor(labels))
        batch_sizes.append(int(data.shape[0]))

    data = torch.cat(all_data, dim=0) if len(all_data) > 1 else all_data[0]
    labels = torch.cat(all_labels, dim=0) if len(all_labels) > 1 else all_labels[0]
    data = data.float()
    labels = labels.long()

    if data.ndim == 4 and data.shape[-1] in (1, 3):
        data = data.permute(0, 3, 1, 2).contiguous()  # [N, H, W, C] -> [N, C, H, W]
    elif data.ndim == 3:
        data = data.unsqueeze(1)  # [N, H, W] -> [N, 1, H, W]
    return data, labels, batch_sizes


def _load_split_tensors(dataset_name: str, split: str, data_dir: str,
                         use_cache: bool = True) -> Tuple[torch.Tensor, torch.Tensor, Optional[list]]:
    """
    Load one split's (data, labels, batch_sizes) tensors, transparently supporting either
    on-disk format. `batch_sizes` is the on-disk per-file batch structure for the `.pkl`
    layout (see `_load_pkl_folder`), or None for the flat `.pt` layout, which has no
    native batch boundaries to preserve.
    """
    if not _is_pkl_format(data_dir):
        path = os.path.join(data_dir, split, f"{dataset_name}_{split}.pt")
        payload = torch.load(path, weights_only=False)
        return payload["data"], payload["labels"], None

    cache_path = os.path.join(data_dir, ".cache", f"{split}.pt")
    if use_cache and os.path.exists(cache_path):
        payload = torch.load(cache_path, weights_only=False)
        if "batch_sizes" in payload:  # guard against a cache written before batch_sizes existed
            return payload["data"], payload["labels"], payload["batch_sizes"]

    data, labels, batch_sizes = _load_pkl_folder(os.path.join(data_dir, split))
    if use_cache:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save({"data": data, "labels": labels, "batch_sizes": batch_sizes}, cache_path)
    return data, labels, batch_sizes


def load_retain_val_test(dataset_name: str, data_dir: str) -> Tuple[TensorDataset, TensorDataset, TensorDataset]:
    """Load the three splits that stay identical across every run (and across a whole sweep)."""
    retain_data, retain_labels, _ = _load_split_tensors(dataset_name, "retain", data_dir)
    val_data, val_labels, _ = _load_split_tensors(dataset_name, "val", data_dir)
    test_data, test_labels, _ = _load_split_tensors(dataset_name, "test", data_dir)
    return (
        TensorDataset(retain_data, retain_labels),
        TensorDataset(val_data, val_labels),
        TensorDataset(test_data, test_labels),
    )


def load_full_forget(dataset_name: str, data_dir: str) -> TensorDataset:
    """
    Load the entire original forget set (unsampled), in the same fixed index order that
    `forget_indices.npy` (written by sample_forget/sample_forget_batches) refers into.
    Used by audit_utils.py to score every forget point under every run's saved model.
    """
    forget_data, forget_labels, _ = _load_split_tensors(dataset_name, "forget", data_dir)
    return TensorDataset(forget_data, forget_labels)


def sample_forget(
    dataset_name: str,
    data_dir: str,
    batch_size: int,
    forget_prob: float = 0.5,
    seed: Optional[int] = None,
):
    """
    Randomly select the slice of the forget set used by this run, without touching
    retain/val/test. Split out from `sample_forget_batches` so a sweep of many runs can
    load retain/val/test once and just re-sample the (much smaller) forget set per run.

    "Forget batches" are the *actual* on-disk batches, not a re-chunking: for the `.pkl`
    layout (see data_utils.py's module docstring), each batch_*.pkl file already is one
    batch (of whatever size that data-prep pipeline gave it, e.g. 1 sample for
    cifar100_bs_1), and those file boundaries are reused as-is here. `batch_size` is only
    used as a fallback chunk size for the flat `.pt` layout, which has no native batch
    boundaries to begin with. Either way, a uniform-random subset of exactly
    `round(forget_prob * num_batches)` whole batches is kept
    (`numpy.random.Generator.choice(..., replace=False)`) -- so e.g. with the default
    forget_prob=0.5, exactly half of the forget batches are sampled every run, not merely
    half in expectation.

    Draws all randomness from a local `numpy.random.Generator` seeded with `seed`, so this
    call has no side effect on global torch/numpy RNG state -- callers don't need to reset
    any seed afterwards to keep model initialization reproducible.

    Returns:
        sampled_forget_set: the kept subset of the forget set -- what `unlearn.unlearn` forgets.
        full_forget_set: the entire original forget set (kept around for evaluation).
        forget_indices: np.ndarray of indices (into the original forget tensor) that were kept.
    """
    if not 0.0 <= forget_prob <= 1.0:
        raise ValueError(f"forget_prob must be in [0, 1], got {forget_prob}")

    rng = np.random.default_rng(seed)

    forget_data, forget_labels, on_disk_batch_sizes = _load_split_tensors(dataset_name, "forget", data_dir)
    num_forget = len(forget_data)

    if on_disk_batch_sizes is not None:
        # Reuse the real batch boundaries (e.g. one .pkl file per batch) as-is.
        boundaries = []
        start = 0
        for size in on_disk_batch_sizes:
            boundaries.append((start, start + size))
            start += size
    else:
        # No native batch structure (flat .pt tensor) -- chunk by batch_size as a fallback.
        num_chunks = math.ceil(num_forget / batch_size)
        boundaries = [(b * batch_size, min((b + 1) * batch_size, num_forget)) for b in range(num_chunks)]

    num_batches = len(boundaries)
    num_keep_batches = round(forget_prob * num_batches)
    keep_batch_ids = np.sort(rng.choice(num_batches, size=num_keep_batches, replace=False))
    forget_indices = np.concatenate([
        np.arange(boundaries[b][0], boundaries[b][1]) for b in keep_batch_ids
    ]) if num_keep_batches > 0 else np.array([], dtype=np.int64)

    forget_indices = np.sort(forget_indices)
    sampled_forget_set = TensorDataset(forget_data[forget_indices], forget_labels[forget_indices])
    full_forget_set = TensorDataset(forget_data, forget_labels)

    print(f"[sample_forget] forget_prob={forget_prob} seed={seed}")
    print(f"  kept {num_keep_batches}/{num_batches} forget batches "
          f"-> {len(forget_indices)}/{num_forget} forget samples "
          f"({100.0 * len(forget_indices) / max(num_forget, 1):.1f}%)")

    return sampled_forget_set, full_forget_set, forget_indices


def sample_forget_batches(
    dataset_name: str,
    data_dir: str,
    batch_size: int,
    forget_prob: float = 0.5,
    seed: Optional[int] = None,
):
    """
    Stage 1 for a single train_and_unlearn.py run: load retain/val/test AND sample the
    forget set in one call. For a sweep of many runs, prefer calling
    `load_retain_val_test` once up front and `sample_forget` per run instead (see
    run_sweep.py) -- retain/val/test are unchanged across runs, so reloading them every
    time is wasted work (especially for the .pkl format's tens of thousands of files).

    Returns:
        train_set: ConcatDataset(retain_set, sampled_forget_set) -- what `train.train` trains on.
        retain_set: the full retain set, unchanged.
        sampled_forget_set: the kept subset of the forget set -- what `unlearn.unlearn` forgets.
        full_forget_set: the entire original forget set (kept around for evaluation).
        val_set, test_set: unchanged.
        forget_indices: np.ndarray of indices (into the original forget tensor) that were kept.
    """
    sampled_forget_set, full_forget_set, forget_indices = sample_forget(
        dataset_name, data_dir, batch_size, forget_prob=forget_prob, seed=seed,
    )
    retain_set, val_set, test_set = load_retain_val_test(dataset_name, data_dir)
    train_set = ConcatDataset([retain_set, sampled_forget_set])

    print(f"  retain set (unchanged): {len(retain_set)} samples")
    print(f"  train set (retain + sampled-forget): {len(train_set)} samples")

    return train_set, retain_set, sampled_forget_set, full_forget_set, val_set, test_set, forget_indices
