import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

# No longer using CIFAR10Pickle - loading batches directly


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run main.py: train then unlearn with a specific set of forget batches (no random sampling)."
    )
    parser.add_argument(
        "--dataroot",
        type=str,
        default="/lfs/mercury1/0/sahasras/r2d/data/cifar10",
        help="Root path for CIFAR10 pre-split data.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="/lfs/mercury1/0/sahasras/r2d/random_forget_runs",
        help="Where to store per-run outputs.",
    )
    parser.add_argument(
        "--run_index",
        type=int,
        default=1,
        help="Run index (for naming only, not used for RNG).",
    )
    parser.add_argument(
        "--forget_batch_indices",
        type=str,
        required=True,
        help="Comma-separated list of batch indices to forget (e.g., '0,1,2').",
    )
    parser.add_argument(
        "--train_epochs",
        type=int,
        default=15,
        help="Epochs for the initial training phase.",
    )
    parser.add_argument(
        "--unlearn_epochs",
        type=int,
        default=5,
        help="Epochs for the unlearning phase (resume).",
    )
    parser.add_argument(
        "--batch_size",
        type=str,
        default="full",
        help="Batch size for both phases (passed to main.py). Can be integer or 'full'.",
    )
    parser.add_argument(
        "--micro_batch_size",
        type=int,
        default=256,
        help="Micro batch size for gradient accumulation (passed to main.py).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3,
        help="Seed for main.py (not used for sampling since batches are explicit).",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA device id to pass to main.py.",
    )
    parser.add_argument(
        "--main_path",
        type=str,
        default="/lfs/mercury1/0/sahasras/r2d/main.py",
        help="Path to main.py entrypoint.",
    )
    parser.add_argument(
        "--shuffle",
        dest="shuffle",
        action="store_true",
        default=False,
        help="Enable shuffling in main.py dataloaders (default: False).",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.1,
        help="Learning rate for training (passed to main.py as --lr).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["cifar10", "cifar100"],
        help="Dataset to use (passed to main.py as --dataset).",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model architecture (passed to main.py as --model), e.g. 'tinynet' or 'tinynetcifar100'.",
    )
    return parser.parse_args()


def load_batch_file(batch_file: Path, batch_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a single batch file and extract images, labels, and generate identities.
    
    Args:
        batch_file: Path to the batch pickle file
        batch_idx: Batch index to use for identity generation
    
    Returns:
        Tuple of (images, labels, identities) as numpy arrays
    """
    import pickle
    
    with open(batch_file, 'rb') as f:
            batch_data = pickle.load(f)
        
    # Extract images and labels (ignore any identities in the file)
    if isinstance(batch_data, tuple):
        batch_images = batch_data[0]
        batch_labels = batch_data[1]
    elif isinstance(batch_data, dict):
        batch_images = batch_data.get('data', batch_data.get('images', batch_data.get('x')))
        batch_labels = batch_data.get('labels', batch_data.get('targets', batch_data.get('y')))
    else:
        raise ValueError(f"Unexpected batch data format: {type(batch_data)}")
    
# Normalize image shape to (N, H, W, C)
    if isinstance(batch_images, np.ndarray):
        if batch_images.ndim == 3:  # Single sample (H, W, C)
            batch_images = batch_images[np.newaxis, ...]
        elif batch_images.ndim == 4:  # Already batched (N, H, W, C)
            pass
        else:
            raise ValueError(f"Unexpected image shape: {batch_images.shape}")
    
# Normalize labels to 1D array
    if isinstance(batch_labels, np.ndarray):
        if batch_labels.ndim == 0:  # Scalar
            batch_labels = np.array([batch_labels])
        elif batch_labels.ndim == 1:
            pass  # Already 1D
        else:
            batch_labels = batch_labels.flatten()
    else:
        batch_labels = np.array([batch_labels])
        
    # Generate identities based on batch index
    n_samples = batch_images.shape[0]
    batch_ids = np.full(n_samples, batch_idx, dtype=np.int64)
    
    return batch_images, batch_labels, batch_ids


def build_train_pickle(retain_root: str, forget_root: str, selected_batch_indices: List[int], out_dir: str):
    """
    Build training pickle by combining retain data with selected forget batches.
    
    Args:
        retain_root: Directory containing retain batch pickle files
        forget_root: Directory containing forget batch pickle files
        selected_batch_indices: List of batch indices (numbers) to include from forget set
        out_dir: Output directory for the combined pickle file
    """
    from pathlib import Path
    import pickle
    
    os.makedirs(out_dir, exist_ok=True)
    
    # Load all retain batches
    retain_path = Path(retain_root)
    retain_batch_files = sorted(retain_path.glob("batch_*.pkl"))
    retain_data_list, retain_labels_list, retain_ids_list = [], [], []
    
    for retain_batch_idx, retain_batch_file in enumerate(retain_batch_files):
        images, labels, ids = load_batch_file(retain_batch_file, retain_batch_idx)
        retain_data_list.append(images)
        retain_labels_list.append(labels)
        retain_ids_list.append(ids)

    # Load selected forget batches
    forget_path = Path(forget_root)
    batch_files = sorted(forget_path.glob("batch_*.pkl"))
    chosen_data_list, chosen_labels_list, chosen_ids_list = [], [], []
    
    for batch_idx in selected_batch_indices:
        if batch_idx >= len(batch_files):
            raise ValueError(f"Batch index {batch_idx} out of range (max: {len(batch_files)-1})")
        images, labels, ids = load_batch_file(batch_files[batch_idx], batch_idx)
        chosen_data_list.append(images)
        chosen_labels_list.append(labels)
        chosen_ids_list.append(ids)
        
    # Concatenate all batches
    def concat_if_not_empty(data_list):
        return np.concatenate(data_list, axis=0) if data_list else np.array([])
    
    retain_data = concat_if_not_empty(retain_data_list)
    retain_labels = concat_if_not_empty(retain_labels_list)
    retain_ids = concat_if_not_empty(retain_ids_list)
    
    chosen_data = concat_if_not_empty(chosen_data_list)
    chosen_labels = concat_if_not_empty(chosen_labels_list)
    chosen_ids_arr = concat_if_not_empty(chosen_ids_list)

    # Combine retain + selected forget batches
    combined_data = np.concatenate([retain_data, chosen_data], axis=0)
    combined_labels = np.concatenate([retain_labels, chosen_labels], axis=0)
    combined_ids = np.concatenate([retain_ids, chosen_ids_arr], axis=0)

    # Save combined training data
    out_path = os.path.join(out_dir, "train_combined.pkl")
    with open(out_path, "wb") as fh:
        pickle.dump((combined_data, combined_labels, combined_ids), fh, protocol=pickle.HIGHEST_PROTOCOL)
    return out_path


def symlink_dir(src: str, dst: str):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.islink(dst) or os.path.exists(dst):
        return
    os.symlink(src, dst)


def run_cmd(cmd: List[str], env=None):
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd, env=env)


def find_latest_final_ckpt(ckpt_dir: str) -> str:
    ckpt_path = None
    candidates = sorted(Path(ckpt_dir).glob("*_final.pt"))
    if candidates:
        ckpt_path = str(candidates[-1])
    return ckpt_path


def main():
    args = parse_args()
    
    # Parse forget batch indices from comma-separated string
    forget_batch_indices = [int(x.strip()) for x in args.forget_batch_indices.split(",")]
    forget_batch_indices.sort()
    
    print(f"Using forget batch indices: {forget_batch_indices}")

    model_name = args.model

    run_dir = os.path.join(args.results_dir, f"run_{args.run_index:03d}")
    train_output = os.path.join(run_dir, "train")
    unlearn_output = os.path.join(run_dir, "unlearn")
    os.makedirs(train_output, exist_ok=True)
    os.makedirs(unlearn_output, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)

    base_root = args.dataroot
    retain_root = os.path.join(base_root, "retain")
    forget_root = os.path.join(base_root, "forget")
    val_root = os.path.join(base_root, "val")
    test_root = os.path.join(base_root, "test")

    # Store forget batch indices in structured format
    forget_ids_json = os.path.join(run_dir, "forget_ids.json")
    forget_ids_npy = os.path.join(run_dir, "forget_ids.npy")
    forget_ids_data = {
        "type": "batch_indices",
        "values": forget_batch_indices
    }
    with open(forget_ids_json, "w") as fh:
        json.dump(forget_ids_data, fh, indent=2)
    np.save(forget_ids_npy, np.array(forget_batch_indices))

    # Prepare per-run data root: train = retain + chosen forget; other splits symlinked
    run_data_root = os.path.join(run_dir, "data")
    train_dir = os.path.join(run_data_root, "train")
    val_dir = os.path.join(run_data_root, "val")
    test_dir = os.path.join(run_data_root, "test")
    forget_dir = os.path.join(run_data_root, "forget")
    retain_dir = os.path.join(run_data_root, "retain")

    os.makedirs(run_data_root, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    symlink_dir(val_root, val_dir)
    symlink_dir(test_root, test_dir)
    symlink_dir(retain_root, retain_dir)

    # Build a per-run forget dir containing only the selected batch files (as symlinks)
    os.makedirs(forget_dir, exist_ok=True)
    forget_batch_files = sorted(Path(forget_root).glob("batch_*.pkl"))
    for batch_idx in forget_batch_indices:
        src = str(forget_batch_files[batch_idx])
        dst = os.path.join(forget_dir, Path(src).name)
        if not os.path.islink(dst) and not os.path.exists(dst):
            os.symlink(src, dst)

    # Build combined train pickle
    train_pickle_path = build_train_pickle(retain_root, forget_root, forget_batch_indices, train_dir)

    # Training phase
    train_cmd = [
        "python3",
        args.main_path,
        "--dataset", args.dataset,
        "--dataroot", run_data_root,
        "--model", model_name,
        "--epochs", str(args.train_epochs),
        "--batch-size", str(args.batch_size),
        "--micro-batch-size", str(args.micro_batch_size),
        "--lr", str(args.learning_rate),
        "--seed", str(args.seed),
        "--device", str(args.device),
        "--save-checkpoints",
        "--output-dir", train_output,
    ]
    if args.shuffle:
        print("Enabling shuffling in train phase")
        train_cmd.append("--shuffle")
    run_cmd(train_cmd)

    # Locate checkpoint to resume from (final)
    ckpt_dir_path = os.path.join(train_output, "checkpoints")
    ckpt_path = find_latest_final_ckpt(ckpt_dir_path)
    if not ckpt_path:
        raise FileNotFoundError(f"Expected checkpoint not found under {ckpt_dir_path}")

    # Unlearning phase using retain_loader via forget_ids_file
    unlearn_cmd = [
        "python3",
        args.main_path,
        "--dataset", args.dataset,
        "--dataroot", args.dataroot,
        "--model", model_name,
        "--epochs", str(args.unlearn_epochs),
        "--batch-size", str(args.batch_size),
        "--micro-batch-size", str(args.micro_batch_size),
        "--lr", str(args.learning_rate),
        "--seed", str(args.seed),
        "--device", str(args.device),
        "--resume", ckpt_path,
        "--output-dir", unlearn_output,
        "--forget-ids-file", forget_ids_json,
        "--save-checkpoints",
    ]
    if args.shuffle:
        print("Enabling shuffling in unlearn phase")
        unlearn_cmd.append("--shuffle")
    run_cmd(unlearn_cmd)

    # Metadata
    meta = {
        "run_index": args.run_index,
        "forget_batch_indices": forget_batch_indices,
        "seed": args.seed,
        "train_epochs": args.train_epochs,
        "unlearn_epochs": args.unlearn_epochs,
        "forget_ids_json": forget_ids_json,
        "forget_ids_npy": forget_ids_npy,
        "train_output": train_output,
        "unlearn_output": unlearn_output,
        "checkpoint_used": ckpt_path,
        "device": args.device,
        "batch_size": args.batch_size,
        "micro_batch_size": args.micro_batch_size,
        "learning_rate": args.learning_rate,
        "dataset": args.dataset,
        "model": model_name,
    }
    with open(os.path.join(run_dir, "run_metadata.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"Run {args.run_index} complete. Outputs under {run_dir}")


if __name__ == "__main__":
    main()

