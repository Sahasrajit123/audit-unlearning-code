"""
main.py -- legacy entry point (kept for backwards compatibility)

This uses the old on-the-fly data loading from data.py.
For the new workflow with pre-computed partitions, use train.py instead.

Usage:  python main.py
        python main.py --config config.json   (explicit path)
"""

import argparse
import random
import numpy as np
import torch

from data import load_data
from trainer_utils import (
    load_config,
    get_device,
    print_header,
    print_row,
    build_model,
    build_optimizer,
    train_loop,
)
from engine import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="config.json", help="path to config.json"
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = get_device()
    print(f"[device] {device}\n")

    # Seed
    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    print(f"[seed] {seed}\n")

    # Data (old on-the-fly loading)
    train_loader, test_loader, char2idx, idx2char, vocab_size, role2idx = load_data(
        seq_len=cfg["seq_len"],
        batch_size=cfg["batch_size"],
        train_frac=cfg["train_frac"],
        path=cfg["path"],
        seed=cfg["seed"],
    )

    # Model
    model = build_model(vocab_size, cfg, device)

    # Save path
    save_path = (
        f"best"
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

    # Training loop
    train_loop(
        model,
        train_loader,
        test_loader,
        optimizer,
        criterion,
        device,
        cfg,
        save_path,
        eval_split_name="test",
    )

    # Generate
    model.load_state_dict(torch.load(save_path, map_location=device))

    from engine import generate

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
