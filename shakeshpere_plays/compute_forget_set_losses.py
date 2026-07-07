#!/usr/bin/env python3
"""
compute_forget_set_losses.py - Memory-efficient loss computation on forget sets

For each (forget_idx, position_in_forget_file) tuple:
  - Classify models into 2 groups: those where this forget_idx was chosen vs not chosen
  - Compute mean, variance, and cardinality for each group
  - Store in JSON (separate files for learnt and unlearnt models)
"""

import json
import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, 'lib')

from model import ShakespeareLSTM


def get_best_gpu():
    """Find GPU with most available memory."""
    if not torch.cuda.is_available():
        return None

    best_gpu = 0
    max_free_memory = 0

    for i in range(torch.cuda.device_count()):
        free_memory = torch.cuda.mem_get_info(i)[0]
        if free_memory > max_free_memory:
            max_free_memory = free_memory
            best_gpu = i

    return best_gpu


def load_config_and_meta(data_dir):
    """Load metadata including vocab and seq_len."""
    meta_path = Path(data_dir) / "meta.json"
    with open(meta_path, "r") as f:
        meta = json.load(f)

    return {
        "vocab": meta["vocab"],
        "vocab_size": meta["vocab_size"],
        "seq_len": meta["seq_len"],
    }


def build_char_to_idx(vocab):
    """Build character to index mapping."""
    return {c: i for i, c in enumerate(vocab)}


def load_forget_files(data_dir):
    """Load all forget_*.txt files from forget/ subfolder."""
    data_path = Path(data_dir)
    forget_dir = data_path / "forget"

    # Load from forget/ subfolder
    forget_files = sorted([p for p in forget_dir.glob("forget_*.txt")])
    if not forget_files:
        forget_file = forget_dir / "forget.txt"
        if forget_file.exists():
            forget_files = [forget_file]

    # Fallback to root if forget/ folder doesn't exist
    if not forget_files:
        forget_files = sorted([p for p in data_path.glob("forget_*.txt")])
        if not forget_files:
            forget_file = data_path / "forget.txt"
            if forget_file.exists():
                forget_files = [forget_file]

    texts = []
    for f in forget_files:
        with open(f, "r", encoding="utf-8") as fp:
            texts.append(fp.read())

    return texts


def encode_text(text, char2idx):
    """Encode text to tensor."""
    encoded = [char2idx[c] for c in text if c in char2idx]
    return torch.tensor(encoded, dtype=torch.long)


def compute_loss_on_text(model, text, seq_len, char2idx, device, vocab_size, batch_size=256):
    """
    Compute CE loss for disjoint chunks of size seq_len.
    Batched processing for speed.
    Returns list of losses (one per chunk).
    """
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    data = encode_text(text, char2idx)
    losses = []

    model.eval()
    with torch.no_grad():
        # Create disjoint chunks
        # Ensure y = data[start+1:end+1] doesn't exceed data length
        num_chunks = (len(data) - 1) // seq_len
        chunks_x = []
        chunks_y = []

        for chunk_idx in range(num_chunks):
            start = chunk_idx * seq_len
            end = start + seq_len
            x = data[start : end]
            y = data[start + 1 : end + 1]
            chunks_x.append(x)
            chunks_y.append(y)

        # Process in batches
        for batch_start in range(0, len(chunks_x), batch_size):
            batch_end = min(batch_start + batch_size, len(chunks_x))
            batch_x = chunks_x[batch_start:batch_end]
            batch_y = chunks_y[batch_start:batch_end]

            # Stack into batch
            batch_x = torch.stack(batch_x).to(device)  # (batch_size, seq_len)
            batch_y = torch.stack(batch_y).to(device)  # (batch_size, seq_len)

            # Forward pass
            logits, _ = model(batch_x)  # (batch_size, seq_len, vocab_size)

            # Compute loss per chunk (mean over seq_len positions)
            loss_per_chunk = criterion(logits.view(-1, vocab_size), batch_y.view(-1))
            loss_per_chunk = loss_per_chunk.view(batch_end - batch_start, seq_len).mean(dim=1)

            losses.extend(loss_per_chunk.cpu().tolist())
            torch.cuda.empty_cache()

    return losses


def load_model(model_path, vocab_size, device):
    """Load a model from checkpoint."""
    model = ShakespeareLSTM(vocab_size, embed_dim=8, hidden_size=256, num_layers=2)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.to(device)
    model.eval()
    return model


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", type=str, default="runs_ascent_descent",
                        help="Path to runs directory")
    parser.add_argument("--data_dir", type=str, default="data_splits_speakers300_fs10",
                        help="Path to data directory")
    parser.add_argument("--device", type=str, default=None,
                        help="Device string, e.g. cuda, cuda:0, cpu")
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU ID to use (legacy; overrides --device when provided)")

    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    data_dir = Path(args.data_dir)

    # Resolve device with precedence: --gpu (legacy) > --device > auto-detect.
    if args.gpu is not None:
        if torch.cuda.is_available():
            device = torch.device("cuda:{}".format(args.gpu))
        else:
            print("[main] Warning: --gpu provided but CUDA unavailable; using cpu")
            device = torch.device("cpu")
    elif args.device is not None:
        requested_device = args.device.strip().lower()
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            print("[main] Warning: CUDA requested but unavailable; using cpu")
            device = torch.device("cpu")
        else:
            device = torch.device(args.device)
    else:
        gpu_id = get_best_gpu()
        if torch.cuda.is_available() and gpu_id is not None:
            device = torch.device("cuda:{}".format(gpu_id))
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    print("[main] Using device: {}".format(device))

    # Load metadata
    print("[main] Loading metadata...")
    meta = load_config_and_meta(data_dir)
    char2idx = build_char_to_idx(meta["vocab"])
    seq_len = meta["seq_len"]
    vocab_size = meta["vocab_size"]
    print("[main] vocab_size={}, seq_len={}".format(vocab_size, seq_len))

    # Load forget files
    print("[main] Loading forget files...")
    forget_texts = load_forget_files(data_dir)
    print("[main] Loaded {} forget files".format(len(forget_texts)))

    # Auto-discover all run_* directories and load their forget_indices
    print("[main] Loading runs...")
    run_data = {}
    run_dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("run_")])

    for run_dir in run_dirs:
        try:
            run_id = int(run_dir.name.split("_")[1])
        except (ValueError, IndexError):
            continue

        forget_indices_path = run_dir / "forget_indices.json"
        if not forget_indices_path.exists():
            print("[main] Warning: {} not found".format(forget_indices_path))
            continue

        with open(str(forget_indices_path), "r") as f:
            forget_info = json.load(f)

        run_data[run_id] = {
            "dir": run_dir,
            "forget_indices": forget_info["forget_indices"],
        }

    print("[main] Loaded {} runs".format(len(run_data)))

    # Data structure: {forget_idx: {position: [losses_per_model]}}
    all_losses_trained = {}
    all_losses_unlearnt = {}

    # Process trained models
    print("[main] Computing losses for trained models...")
    for run_id in sorted(run_data.keys()):
        run_info = run_data[run_id]
        model_path = run_info["dir"] / "model_trained.pt"

        if not model_path.exists():
            print("[main]   Warning: {} not found, skipping".format(model_path))
            continue

        print("[main]   Processing run_{}...".format(run_id))
        model = load_model(str(model_path), vocab_size, device)

        # Compute loss on each forget file
        for forget_idx, text in enumerate(forget_texts):
            losses = compute_loss_on_text(model, text, seq_len, char2idx, device, vocab_size)

            if forget_idx not in all_losses_trained:
                all_losses_trained[forget_idx] = {}

            for position, loss in enumerate(losses):
                if position not in all_losses_trained[forget_idx]:
                    all_losses_trained[forget_idx][position] = []
                all_losses_trained[forget_idx][position].append((run_id, loss))

        print("[main]   Finished run_{}".format(run_id))
        del model
        torch.cuda.empty_cache()

    # Process unlearnt models
    print("[main] Computing losses for unlearnt models...")
    for run_id in sorted(run_data.keys()):
        run_info = run_data[run_id]
        model_path = run_info["dir"] / "model_unlearnt.pt"

        if not model_path.exists():
            print("[main]   Warning: {} not found, skipping".format(model_path))
            continue

        print("[main]   Processing run_{}...".format(run_id))
        model = load_model(str(model_path), vocab_size, device)

        # Compute loss on each forget file
        for forget_idx, text in enumerate(forget_texts):
            losses = compute_loss_on_text(model, text, seq_len, char2idx, device, vocab_size)

            if forget_idx not in all_losses_unlearnt:
                all_losses_unlearnt[forget_idx] = {}

            for position, loss in enumerate(losses):
                if position not in all_losses_unlearnt[forget_idx]:
                    all_losses_unlearnt[forget_idx][position] = []
                all_losses_unlearnt[forget_idx][position].append((run_id, loss))

        print("[main]   Finished run_{}".format(run_id))
        del model
        torch.cuda.empty_cache()

    # Aggregate statistics
    print("[main] Aggregating statistics...")

    def aggregate_stats(all_losses_dict):
        """
        all_losses_dict: {forget_idx: {position: [(run_id, loss), ...]}}
        Returns: {str(forget_idx): {str(position): {...}}}
        """
        result = {}

        for forget_idx in sorted(all_losses_dict.keys()):
            result[str(forget_idx)] = {}

            for position in sorted(all_losses_dict[forget_idx].keys()):
                run_losses = all_losses_dict[forget_idx][position]

                losses_chosen = []
                losses_not_chosen = []

                for run_id, loss in run_losses:
                    chosen = forget_idx in run_data[run_id]["forget_indices"]
                    if chosen:
                        losses_chosen.append(loss)
                    else:
                        losses_not_chosen.append(loss)

                stats = {}

                if losses_chosen:
                    losses_chosen_arr = np.array(losses_chosen)
                    stats["models_chosen"] = {
                        "mean": float(np.mean(losses_chosen_arr)),
                        "var": float(np.var(losses_chosen_arr)),
                        "count": len(losses_chosen),
                    }
                else:
                    stats["models_chosen"] = {
                        "mean": None,
                        "var": None,
                        "count": 0,
                    }

                if losses_not_chosen:
                    losses_not_chosen_arr = np.array(losses_not_chosen)
                    stats["models_not_chosen"] = {
                        "mean": float(np.mean(losses_not_chosen_arr)),
                        "var": float(np.var(losses_not_chosen_arr)),
                        "count": len(losses_not_chosen),
                    }
                else:
                    stats["models_not_chosen"] = {
                        "mean": None,
                        "var": None,
                        "count": 0,
                    }

                result[str(forget_idx)][str(position)] = stats

        return result

    trained_stats = aggregate_stats(all_losses_trained)
    unlearnt_stats = aggregate_stats(all_losses_unlearnt)

    # Save results
    print("[main] Saving results...")

    trained_output = runs_dir / "losses_trained_models.json"
    with open(str(trained_output), "w") as f:
        json.dump(trained_stats, f, indent=2)
    print("[main] Saved: {}".format(trained_output))

    unlearnt_output = runs_dir / "losses_unlearnt_models.json"
    with open(str(unlearnt_output), "w") as f:
        json.dump(unlearnt_stats, f, indent=2)
    print("[main] Saved: {}".format(unlearnt_output))

    print("[main] Done!")


if __name__ == "__main__":
    main()
