"""
train.py -- train model using pre-computed data partitions

This loads pre-computed train/val/test/retain/forget splits from prepare_data.py.
Trains on a specified split and validates on val. After training, evaluates on test.

Usage:
  python train.py --train_split retain
  python train.py --train_split forget
"""

import argparse
import os
import random
import numpy as np
import torch

# Must be set before any CUDA calls for cublas determinism
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import sys
sys.path.insert(0, 'lib')

from data_loader import load_data_splits
from trainer_utils import (
    load_config,
    get_device,
    build_model,
    build_optimizer,
    train_loop,
)
from engine import evaluate, generate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json", help="path to config.json")
    parser.add_argument(
        "--train_split",
        default="train",
        choices=["retain", "forget", "train"],
        help="which training split to use",
    )
    parser.add_argument(
        "--data_dir",
        default="data_splits",
        help="directory containing pre-computed data partitions",
    )
    parser.add_argument(
        "--shuffle_train_samples",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="shuffle at sample level once (same batch order every epoch; uses seed)",
    )
    parser.add_argument(
        "--shuffle_train_batches_each_epoch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="reshuffle batch order each epoch (different order every epoch)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        metavar="ID",
        help="GPU ID to use (e.g. --gpu 0). Defaults to cuda:0 if CUDA is available.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        default=False,
        help="enable torch.compile (faster but not fully deterministic)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(args.gpu)
    print(f"[device] {device}\n")

    # Seed
    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    print(f"[seed] {seed}\n")

    # Load pre-computed partitions
    print(f"[data] loading from {args.data_dir}...\n")
    loaders, metadata = load_data_splits(
        args.data_dir,
        cfg["batch_size"],
        seed,
        shuffle_train_samples=args.shuffle_train_samples,
        shuffle_train_batches_each_epoch=args.shuffle_train_batches_each_epoch,
    )

    vocab_size = metadata["vocab_size"]
    char2idx = metadata["char2idx"]
    idx2char = metadata["idx2char"]

    train_loader = loaders[args.train_split]
    val_loader = loaders["val"]
    test_loader = loaders["test"]

    print(f"\n[train] using '{args.train_split}' split for training")
    print(f"[eval]  using 'val' split for validation\n")

    # Model
    model = build_model(vocab_size, cfg, device, args.compile)

    # Save path with train split identifier
    save_path = (
        f"best"
        f"_{args.train_split}"
        f"_{cfg['optimizer']}"
        f"_lr{cfg['lr']}"
        f"_h{cfg['hidden_size']}"
        f"_l{cfg['num_layers']}"
        f"_b{cfg['batch_size']}"
        f"_ep{cfg['epochs']}"
        f"_seed{cfg['seed']}"
        f".pt"
    )
    print(f"[save] {save_path}\n")

    # Optimizer & loss
    optimizer = build_optimizer(model, cfg)
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop (validates on val set)
    train_loop(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device,
        cfg,
        save_path,
        eval_split_name="val",
    )

    # Load best model and evaluate on test set
    print("\n[test] loading best weights and evaluating on test set...")
    model.load_state_dict(torch.load(save_path, map_location=device))
    test_loss, test_acc, test_ppl = evaluate(model, test_loader, criterion, device)

    print(f"\n[test] results:")
    print(f"  loss:         {test_loss:.4f}")
    print(f"  accuracy:     {test_acc:.2%}")
    print(f"  perplexity:   {test_ppl:.2f}")

    # Generate
    output = generate(
        model,
        cfg["seed_text"],
        char2idx,
        idx2char,
        n_chars=cfg["n_chars"],
        temperature=cfg["temperature"],
        device=device,
    )
    print(f"\n--- seed: '{cfg['seed_text']}' ---\n")
    print(output)


if __name__ == "__main__":
    main()
