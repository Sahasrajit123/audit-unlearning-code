# src/utils/data_cache.py
import logging, pickle, os
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.data.datamodule import get_datamodule

def _to_numpy(obj: any):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy()
    elif isinstance(obj, dict):
        return {k: _to_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(_to_numpy(v) for v in obj)
    else:
        return obj
    
import jax.numpy as jnp
from typing import List, Tuple

Batch = Tuple[jnp.ndarray, jnp.ndarray]

def rebatch(batches: List[Batch], new_bs: int, *, drop_remainder: bool = True) -> List[Batch]:
    """
    Concatenate all batches and re-slice into batches of size `new_bs`.

    Args
    ----
    batches        list of (images, labels) arrays you already loaded
    new_bs         desired batch size
    drop_remainder if True, discard the last few examples that don't fit exactly;
                   if False, keep them as a smaller final batch.

    Returns
    -------
    new_batches    list of (images, labels) with the requested batch size
    """
    # Handle empty batches case
    if not batches:
        return []
    
    # 1) flatten to one big array
    imgs = jnp.concatenate([b[0] for b in batches], axis=0)
    labs = jnp.concatenate([b[1] for b in batches], axis=0)

    n_total = imgs.shape[0]
    n_full  = n_total // new_bs
    last_sz = n_total % new_bs

    # 2) slice into new batches
    new_batches: List[Batch] = [
        (imgs[i*new_bs:(i+1)*new_bs], labs[i*new_bs:(i+1)*new_bs])
        for i in range(n_full)
    ]

    # optional leftover
    if last_sz and not drop_remainder:
        new_batches.append(
            (imgs[n_full*new_bs:], labs[n_full*new_bs:])
        )

    return new_batches


class _ShuffledBatchList:
    """
    Wraps a list of (images, labels) batches and shuffles their order at the start of each
    iteration. Use so each training epoch sees a different batch order (reproducible via rng).
    """
    def __init__(self, batches: List[Batch], rng: Optional[np.random.Generator] = None):
        self._batches = batches
        self._rng = rng if rng is not None else np.random.default_rng()

    def __len__(self) -> int:
        return len(self._batches)

    def __iter__(self):
        indices = np.arange(len(self._batches))
        self._rng.shuffle(indices)
        for i in indices:
            yield self._batches[i]


def _save_loader(loader, out_dir: Path, max_batches: Optional[int] = None) -> None:
    """Persist each batch (already pre-batched by the DataLoader) as a pickle."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, batch in enumerate(loader):
        if max_batches is not None and idx >= max_batches:
            break
        batch_np = _to_numpy(batch)          # ▶️  ensure NumPy before pickling

        with (out_dir / f"batch_{idx:05d}.pkl").open("wb") as fh:
            pickle.dump(batch_np, fh, protocol=pickle.HIGHEST_PROTOCOL)


class _PickleDataset(Dataset):
    """Streams one *pre-batched* pickle per item."""
    def __init__(self, files: List[Path]):
        self._files = files

    def __len__(self) -> int:
        return len(self._files)

    def __getitem__(self, idx):
        with self._files[idx].open("rb") as fh:
            return pickle.load(fh)


# def _loader_from_dir(split_dir: Path) -> DataLoader:
#     files = sorted(split_dir.glob("batch_*.pkl"))
#     ds = _PickleDataset(files)
#     # Each item is already a full batch ➜ batch_size=None.
#     return DataLoader(ds, batch_size=None, shuffle=False, num_workers=0)

# src/utils/data_cache.py  (only this function changes)

def _loader_from_dir(split_dir: Path) -> DataLoader:
    files = sorted(split_dir.glob("batch_*.pkl"))
    ds = _PickleDataset(files)

    return DataLoader(
        ds,
        batch_size=1,          # each file is a *full batch*
        shuffle=False,
        num_workers=0,
        # ✨ BYPASS default_collate → returns the item verbatim
        collate_fn=lambda items: items[0],
    )



# -------------------------------------------------------------------------
def prepare_or_load_cifar_splits(
    config: dict,
    run_dir: str,
    *,
    generator=None,
    max_batches: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader, DataLoader]:
    """
    *If* <run_dir>/cifar/{train,val,test,forget,retain}/ exists, simply return
    DataLoaders that stream the cached pickles.

    *Else* build the DataModule once, write every batch to those directories,
    and return the *fresh* in-memory DataLoaders.
    """
    cifar_root = Path(run_dir) / "cifar"
    splits = ["train", "val", "test", "forget", "retain"]

    if all((cifar_root / s).is_dir() for s in splits):
        logging.info("♻️  Using cached CIFAR splits from %s", cifar_root)
        return tuple(_loader_from_dir(cifar_root / s) for s in splits)

    # ── First-time run: build, split, save ───────────────────────────────
    logging.info("📦 Building CIFAR splits and caching to %s", cifar_root)
    dm = get_datamodule(config=config, generator=generator)
    dm.setup()

    _save_loader(dm.train_dataloader(),  cifar_root / "train",  max_batches)
    _save_loader(dm.val_dataloader(),    cifar_root / "val",    max_batches)
    _save_loader(dm.test_dataloader(),   cifar_root / "test",   max_batches)
    _save_loader(dm.forget_dataloader(), cifar_root / "forget", max_batches)
    _save_loader(dm.retain_dataloader(), cifar_root / "retain", max_batches)

    return (
        dm.train_dataloader(),
        dm.val_dataloader(),
        dm.test_dataloader(),
        dm.forget_dataloader(),
        dm.retain_dataloader(),
    )


def load_cifar_splits_with_batch_subset(
    run_dir: str,
    *,
    forget_fraction: float = 0.10,
    rng_seed: int = 0,
    batch_size: int = 128,
    num_forget_subsets: int = None,
    primary_subset_weight: float = 0.95,
    data_subfolder: str = None,
    single_forget_subset: bool = False,
    chosen_idx_override: Optional[np.ndarray] = None,
    shuffle_train_samples: bool = False,
    shuffle_train_batches: bool = False,
    shuffle_train_batches_each_epoch: bool = False,
):
    """
    Every call samples a fresh set of *whole* batches from the cached
    <run_dir>/<data_subfolder>/forget/ split and appends them to the retain batches
    to form the training set.

    If num_forget_subsets is provided, looks for forget/forget_1, forget/forget_2, etc.
    and creates weighted selection (95% from primary subset, 5% from others).

    Parameters
    ----------
    run_dir : str
        Root directory containing data_split/.
    forget_fraction : float, default=0.10
        Fraction of forget batches to select.
    rng_seed : int, default=0
        Random seed for reproducibility.
    batch_size : int, default=128
        Batch size for rebatching loaded data.
    num_forget_subsets : int, optional
        Number of forget subsets. If None, uses original single forget behavior.
    primary_subset_weight : float, default=0.95
        Weight for primary forget subset when using multiple subsets.
        Ignored if single_forget_subset is True.
    data_subfolder : str
        Data subfolder under data_split/ (must be provided).
    single_forget_subset : bool, default=False
        If True, uniformly picks ONE forget subset at random and uses ALL files from that subset.
        This overrides both primary_subset_weight and forget_fraction parameters.
    chosen_idx_override : np.ndarray, optional
        If provided, use these indices as the forget-batch selection instead of random sampling.
        Only for num_forget_subsets is None. Length must equal k_batches = round(forget_fraction * n_forget).
    shuffle_train_samples : bool, default=False
        If True, shuffle at sample level once (using rng_seed for reproducibility), then form batches.
        Batch order is the same every epoch. If False, train data is in file order (backward compatible).
    shuffle_train_batches : bool, default=False
        If True, keep samples within each cached batch intact, shuffle the loaded training batches
        once, then rebatch deterministically.
    shuffle_train_batches_each_epoch : bool, default=False
        If True, shuffle the order of batches at the start of each epoch (different order every epoch).
        Uses rng (from rng_seed) so the sequence of shuffles is reproducible. Can be combined with
        shuffle_train_samples.

    Returns
    -------
    train_loader, val_loader, test_loader, forget_loader, retain_loader,
    chosen_idx   : np.ndarray of indices (into forget_files) that were
                   included in this run's train_loader (for CIFAR data),
                   or a dictionary for synthetic data.

    Current implementation is not fully correct as forget loader should also contain only chosen indices. This one is okay but mostly not most efficient.
    """
    assert 0.0 <= forget_fraction <= 1.0, "forget_fraction must be in [0,1]"
    if shuffle_train_batches and shuffle_train_samples:
        raise ValueError("shuffle_train_batches and shuffle_train_samples are mutually exclusive")
    if num_forget_subsets is not None:
        if primary_subset_weight is not None:
            assert 0.0 <= primary_subset_weight <= 1.0, "primary_subset_weight must be in [0,1]"
        assert num_forget_subsets > 0, "num_forget_subsets must be positive"

    # Use custom data subfolder (now always required)
    root = Path(run_dir) / data_subfolder
    
    # Check for required splits
    for split in ["retain", "val", "test"]:
        if not (root / split).is_dir():
            raise RuntimeError(
                f"Expected cached split '{split}' under {root}. "
                "Run prepare_or_load_cifar_splits once first."
            )

    # ---------------- file inventories -------------------------------
    retain_files = sorted((root / "retain").glob("batch_*.pkl"))
    val_files    = sorted((root / "val").glob("batch_*.pkl"))
    test_files   = sorted((root / "test").glob("batch_*.pkl"))
    
    rng = np.random.default_rng(rng_seed)
    
    # ---------------- Choose behavior based on num_forget_subsets --------
    if num_forget_subsets is None:
        # ORIGINAL BEHAVIOR: Single forget directory
        if not (root / "forget").is_dir():
            raise RuntimeError(
                f"Expected cached split 'forget' under {root}. "
                "Run prepare_or_load_cifar_splits once first."
            )
        
        forget_files = sorted((root / "forget").glob("batch_*.pkl"))
        n_forget = len(forget_files)
        k_batches = int(round(forget_fraction * n_forget))
        if chosen_idx_override is not None:
            chosen_idx = np.asarray(chosen_idx_override, dtype=np.int64)
            if len(chosen_idx) != k_batches:
                raise ValueError(
                    f"chosen_idx_override has len {len(chosen_idx)}, expected k_batches={k_batches} "
                    f"(forget_fraction={forget_fraction} * n_forget={n_forget})"
                )
            if np.any(chosen_idx < 0) or np.any(chosen_idx >= n_forget):
                raise ValueError(
                    "chosen_idx_override contains indices out of range [0, n_forget-1]"
                )
            chosen_idx.sort()
        else:
            chosen_idx = rng.choice(n_forget, size=k_batches, replace=False)
            chosen_idx.sort()  # keep deterministic order
        chosen_files = [forget_files[i] for i in chosen_idx]
        
    else:
        # NEW BEHAVIOR: Multiple forget subsets with weighted selection
        forget_root = root / "forget"
        if not forget_root.is_dir():
            raise RuntimeError(
                f"Expected forget directory under {root}. "
                "Run prepare_or_load_cifar_splits once first."
            )
        
        # Check for forget subsets (nested under forget/ folder)
        forget_subset_dirs = []
        for i in range(1, num_forget_subsets + 1):
            subset_dir = forget_root / f"forget_{i}"
            if not subset_dir.is_dir():
                raise RuntimeError(
                    f"Expected forget subset 'forget/forget_{i}' under {root}. "
                    f"Found {num_forget_subsets} forget subsets but missing forget/forget_{i}."
                )
            forget_subset_dirs.append(subset_dir)

        # Get files from all forget subsets
        forget_subset_files = []
        for subset_dir in forget_subset_dirs:
            subset_files = sorted(subset_dir.glob("batch_*.pkl"))
            forget_subset_files.append(subset_files)
        
        # Calculate total forget files and how many to select
        total_forget_files = sum(len(files) for files in forget_subset_files)
        k_batches = int(round(forget_fraction * total_forget_files))
        
        if single_forget_subset:
            # NEW MODE: Pick ONE forget subset uniformly at random
            # and use ALL files from that subset
            chosen_subset_idx = rng.choice(num_forget_subsets)
            subset_files = forget_subset_files[chosen_subset_idx]
            n_subset = len(subset_files)
            
            # Use ALL files from the chosen subset (ignore forget_fraction)
            chosen_files = list(subset_files)
            
            print(f"[single_forget_subset mode] Chose forget subset {chosen_subset_idx + 1}, "
                  f"using ALL {n_subset} files from this subset")
            
        elif primary_subset_weight is None:
            # UNIFORM SELECTION: Select uniformly from all forget files across all subsets
            # Flatten all files into a single list
            all_forget_files = []
            for subset_files in forget_subset_files:
                all_forget_files.extend(subset_files)
            
            # Select k_batches uniformly from all files
            n_total = len(all_forget_files)
            chosen_idx_flat = rng.choice(n_total, size=k_batches, replace=False)
            chosen_idx_flat.sort()  # keep deterministic order
            chosen_files = [all_forget_files[i] for i in chosen_idx_flat]
        else:
            # WEIGHTED SELECTION: Select with primary subset weight
            # Calculate how many files to select from each subset
            primary_subset_idx = rng.choice(num_forget_subsets)  # Randomly choose primary subset
            primary_count = int(round(k_batches * primary_subset_weight))
            remaining_count = k_batches - primary_count
            
            # Select from primary subset
            primary_files = forget_subset_files[primary_subset_idx]
            n_primary = len(primary_files)
            if primary_count > n_primary:
                primary_count = n_primary
                remaining_count = k_batches - primary_count
            
            primary_chosen_idx = rng.choice(n_primary, size=primary_count, replace=False)
            primary_chosen_files = [primary_files[i] for i in primary_chosen_idx]
            
            # Select from remaining subsets (uniformly distributed)
            remaining_chosen_files = []
            if remaining_count > 0:
                # Calculate how many files to select from each remaining subset
                remaining_subsets = [i for i in range(num_forget_subsets) if i != primary_subset_idx]
                files_per_remaining = remaining_count // len(remaining_subsets)
                extra_files = remaining_count % len(remaining_subsets)
                
                for i, subset_idx in enumerate(remaining_subsets):
                    subset_files = forget_subset_files[subset_idx]
                    n_subset = len(subset_files)
                    count_for_this_subset = files_per_remaining + (1 if i < extra_files else 0)
                    
                    if count_for_this_subset > 0 and n_subset > 0:
                        actual_count = min(count_for_this_subset, n_subset)
                        subset_chosen_idx = rng.choice(n_subset, size=actual_count, replace=False)
                        subset_chosen_files = [subset_files[j] for j in subset_chosen_idx]
                        remaining_chosen_files.extend(subset_chosen_files)
            
            # Combine all chosen files
            chosen_files = primary_chosen_files + remaining_chosen_files
            rng.shuffle(chosen_files)  # Shuffle to mix primary and remaining files
        
        # Define forget_files as all files from all forget subsets (for backward compatibility)
        forget_files = []
        for subset_files in forget_subset_files:
            forget_files.extend(subset_files)
        
        # For synthetic data: create set of chosen batch keys OR simple dict for single_forget_subset
        if single_forget_subset:
            # SIMPLE: Just store which subset was chosen (1-indexed)
            chosen_batches = {
                "mode": "single_forget_subset",
                "chosen_subset_idx": int(chosen_subset_idx + 1),  # 1-indexed
                "num_files_used": len(chosen_files),  # Should equal total_files_in_subset
                "note": "ALL files from chosen subset are used"
            }
        else:
            # DETAILED: Store all individual batch keys for multi-subset modes
            chosen_batches = set()
            for chosen_file in chosen_files:
                # Extract subset and batch number from file path
                # e.g., "forget_3/batch_00001.pkl" -> "forget_3_1"
                file_name = chosen_file.name  # "batch_00001.pkl"
                subset_name = chosen_file.parent.name  # "forget_3"
                batch_num = int(file_name.split('_')[1].split('.')[0])  # 1
                key = f"{subset_name}_{batch_num}"
                chosen_batches.add(key)
        
        # For backward compatibility, also create chosen_idx
        chosen_idx = []
        for chosen_file in chosen_files:
            for i, forget_file in enumerate(forget_files):
                if forget_file == chosen_file:
                    chosen_idx.append(i)
                    break
        chosen_idx = np.array(chosen_idx)

    # ---------------- helper: loader from arbitrary file list -------
    def _loader_from_files(files: List[Path]) -> DataLoader:
        ds = _PickleDataset(files)
        return DataLoader(
            ds,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=lambda items: items[0],
        )

    # ---------------- build loaders ---------------------------------
    # Default (backward compatible): file order, same batch order every epoch.
    def _train_batches_sample_shuffle(files: List[Path], new_bs: int, drop_remainder: bool = True) -> List[Batch]:
        raw = list(_loader_from_files(files))
        if not raw:
            return []
        imgs = jnp.concatenate([b[0] for b in raw], axis=0)
        labs = jnp.concatenate([b[1] for b in raw], axis=0)
        n = imgs.shape[0]
        perm = np.array(rng.permutation(n))
        imgs = imgs[perm]
        labs = labs[perm]
        n_full = n // new_bs
        out = [
            (imgs[i * new_bs : (i + 1) * new_bs], labs[i * new_bs : (i + 1) * new_bs])
            for i in range(n_full)
        ]
        if not drop_remainder and n % new_bs != 0:
            out.append((imgs[n_full * new_bs :], labs[n_full * new_bs :]))
        return out

    train_raw_batches = list(_loader_from_files(retain_files + chosen_files))
    if shuffle_train_batches:
        rng.shuffle(train_raw_batches)

    if shuffle_train_samples:
        train_loader = _train_batches_sample_shuffle(retain_files + chosen_files, batch_size)
    else:
        train_loader = rebatch(train_raw_batches, new_bs=batch_size)

    if shuffle_train_batches_each_epoch:
        train_loader = _ShuffledBatchList(train_loader, rng=rng)

    val_loader    = rebatch(list(_loader_from_files(val_files)), new_bs=batch_size)
    test_loader   = rebatch(list(_loader_from_files(test_files)), new_bs=batch_size)
    forget_loader = rebatch(list(_loader_from_files(forget_files)), new_bs=batch_size)
    retain_loader = rebatch(list(_loader_from_files(retain_files)), new_bs=batch_size)




    if num_forget_subsets is not None:
        # Synthetic data: return dictionary
        return (
            train_loader,
            val_loader,
            test_loader,
            forget_loader,
            retain_loader,
            chosen_batches,    # ← dictionary for synthetic data
        )
    else:
        # CIFAR data: return numpy array (unchanged)
        return (
            train_loader,
            val_loader,
            test_loader,
            forget_loader,
            retain_loader,
            chosen_idx,        # ← numpy array for CIFAR data
        )



