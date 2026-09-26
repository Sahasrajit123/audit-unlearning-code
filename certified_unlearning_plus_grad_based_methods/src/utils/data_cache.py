# src/utils/data_cache.py
"""
Loaders over a pre-batched data split, as written by scripts/split_cifar100_train.py:

    <data_dir>/{retain,forget,val,test}/batch_XXXXX.pkl     each file = (images, labels)

`load_cifar_splits_with_batch_subset` samples, per run, a subset of the forget batch files,
trains on retain + that subset, and hands only that subset to the unlearning routine.
"""
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import jax.numpy as jnp
import numpy as np

Batch = Tuple[jnp.ndarray, jnp.ndarray]

SPLITS = ("retain", "forget", "val", "test")


def rebatch(batches: List[Batch], new_bs: int, *, drop_remainder: bool = True) -> List[Batch]:
    """
    Concatenate all batches and re-slice into batches of size `new_bs`.

    drop_remainder: if True, discard the last few examples that don't fit exactly;
                    if False, keep them as a smaller final batch.
    """
    if not batches:
        return []

    imgs = jnp.concatenate([b[0] for b in batches], axis=0)
    labs = jnp.concatenate([b[1] for b in batches], axis=0)

    n_full = imgs.shape[0] // new_bs
    new_batches: List[Batch] = [
        (imgs[i * new_bs:(i + 1) * new_bs], labs[i * new_bs:(i + 1) * new_bs])
        for i in range(n_full)
    ]
    if imgs.shape[0] % new_bs and not drop_remainder:
        new_batches.append((imgs[n_full * new_bs:], labs[n_full * new_bs:]))
    return new_batches


class _ShuffledBatchList:
    """
    Wraps a list of (images, labels) batches and shuffles their order at the start of each
    iteration. Use so each training epoch sees a different batch order (reproducible via rng).
    """
    def __init__(self, batches: List[Batch], rng: np.random.Generator):
        self._batches = batches
        self._rng = rng

    def __len__(self) -> int:
        return len(self._batches)

    def __iter__(self):
        indices = np.arange(len(self._batches))
        self._rng.shuffle(indices)
        for i in indices:
            yield self._batches[i]


def split_files(data_dir, split: str) -> List[Path]:
    """Sorted batch files of one split. The sort order defines the batch indices stored in
    chosen_forget_batches.npy, so every reader must use this same order."""
    files = sorted((Path(data_dir) / split).glob("batch_*.pkl"))
    if not files:
        raise FileNotFoundError(
            f"No batch_*.pkl under {Path(data_dir) / split}. "
            "Generate the split with scripts/split_cifar100_train.py.")
    return files


def _load(files: List[Path]) -> List[Batch]:
    out = []
    for f in files:
        with f.open("rb") as fh:
            out.append(pickle.load(fh))
    return out


def load_split(data_dir, split: str) -> List[Batch]:
    """Every cached batch of one split, in split_files order."""
    return _load(split_files(data_dir, split))


def _sample_shuffled_batches(batches: List[Batch], new_bs: int,
                             rng: np.random.Generator) -> List[Batch]:
    """Shuffle at the sample level (one permutation), then slice into `new_bs` batches,
    dropping the remainder."""
    if not batches:
        return []
    imgs = jnp.concatenate([b[0] for b in batches], axis=0)
    labs = jnp.concatenate([b[1] for b in batches], axis=0)
    perm = np.array(rng.permutation(imgs.shape[0]))
    return rebatch([(imgs[perm], labs[perm])], new_bs)


def load_cifar_splits_with_batch_subset(
    data_dir: str,
    *,
    forget_fraction: float = 0.5,
    rng_seed: int = 0,
    batch_size: int = 128,
    chosen_idx_override: Optional[np.ndarray] = None,
    shuffle_train_samples: bool = False,
    shuffle_train_batches: bool = False,
    shuffle_train_batches_each_epoch: bool = False,
):
    """
    Sample `round(forget_fraction * n_forget)` whole forget batch files and build the loaders
    for one run.

    Parameters
    ----------
    data_dir : str
        The split directory itself, e.g. data/cifar100/data_split/cifar100_bs_750.
    forget_fraction : float
        Fraction of forget batch files this run trains on (and is later asked to forget).
    rng_seed : int
        Seeds the forget-batch sample and any train shuffling.
    batch_size : int
        Batch size of the returned loaders (independent of the batch size of the files).
    chosen_idx_override : np.ndarray, optional
        Use these forget batch indices instead of sampling. Length must equal
        round(forget_fraction * n_forget).
    shuffle_train_samples : bool
        Shuffle the training set at the sample level once, then batch. Batch order is the
        same every epoch.
    shuffle_train_batches : bool
        Shuffle the order of the cached batch files once, keeping each file's samples
        together, then rebatch. Mutually exclusive with shuffle_train_samples.
    shuffle_train_batches_each_epoch : bool
        Reshuffle batch order at the start of every epoch. Can be combined with either of
        the above.

    Returns
    -------
    train_loader : retain + chosen forget batches
    val_loader, test_loader, retain_loader
    forget_loader : ONLY the chosen forget batches -- the deletion set. The final short
        batch is kept so every requested point receives a forget step. (The audits do not
        use this loader; they load the whole forget pool to rank every candidate.)
    chosen_idx : sorted np.ndarray of indices into split_files(data_dir, "forget")
    """
    if not 0.0 <= forget_fraction <= 1.0:
        raise ValueError("forget_fraction must be in [0, 1]")
    if shuffle_train_batches and shuffle_train_samples:
        raise ValueError("shuffle_train_batches and shuffle_train_samples are mutually exclusive")

    files = {s: split_files(data_dir, s) for s in SPLITS}
    rng = np.random.default_rng(rng_seed)

    # ---------------- forget-batch sample ----------------------------
    n_forget = len(files["forget"])
    k_batches = int(round(forget_fraction * n_forget))
    if chosen_idx_override is not None:
        chosen_idx = np.sort(np.asarray(chosen_idx_override, dtype=np.int64))
        if len(chosen_idx) != k_batches:
            raise ValueError(
                f"chosen_idx_override has len {len(chosen_idx)}, expected k_batches={k_batches} "
                f"(forget_fraction={forget_fraction} * n_forget={n_forget})")
        if chosen_idx.min() < 0 or chosen_idx.max() >= n_forget:
            raise ValueError("chosen_idx_override contains indices out of range [0, n_forget-1]")
    else:
        chosen_idx = np.sort(rng.choice(n_forget, size=k_batches, replace=False))
    chosen_files = [files["forget"][i] for i in chosen_idx]

    # ---------------- loaders ----------------------------------------
    train_raw = _load(files["retain"] + chosen_files)
    if shuffle_train_batches:
        rng.shuffle(train_raw)
    if shuffle_train_samples:
        train_loader = _sample_shuffled_batches(train_raw, batch_size, rng)
    else:
        train_loader = rebatch(train_raw, batch_size)
    if shuffle_train_batches_each_epoch:
        train_loader = _ShuffledBatchList(train_loader, rng=rng)

    val_loader = rebatch(_load(files["val"]), batch_size)
    test_loader = rebatch(_load(files["test"]), batch_size)
    retain_loader = rebatch(_load(files["retain"]), batch_size)
    forget_loader = rebatch(_load(chosen_files), batch_size, drop_remainder=False)

    return train_loader, val_loader, test_loader, forget_loader, retain_loader, chosen_idx
