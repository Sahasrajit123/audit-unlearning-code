"""
Analyze loss on forget sets for all 50 models.

For each forget file:
1. Split into chunks of size seq_len
2. For each (forget_idx, position) tuple:
   - Collect losses from all 50 models
   - Split models into two groups: those where this forget_idx was chosen vs not
   - Compute mean, variance, and count for each group
   - Save to JSON
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple
import torch
import torch.nn.functional as F
import numpy as np
from model import ShakespeareLSTM
from tqdm import tqdm


def load_json(path: Path) -> dict:
    """Load JSON from file."""
    with open(path, 'r') as f:
        return json.load(f)


def encode_text_to_indices(text: str, char2idx: Dict[str, int]) -> List[int]:
    """Encode text using char2idx mapping, filtering unknown characters."""
    return [char2idx[c] for c in text if c in char2idx]


def split_into_chunks(indices: List[int], chunk_size: int) -> List[List[int]]:
    """Split a list of indices into chunks of given size."""
    chunks = []
    for i in range(0, len(indices) - chunk_size, chunk_size):
        chunk = indices[i : i + chunk_size]
        if len(chunk) == chunk_size:
            chunks.append(chunk)
    return chunks


def compute_loss(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, device: str) -> float:
    """Compute cross-entropy loss for a single sample."""
    model.eval()
    with torch.no_grad():
        x = x.unsqueeze(0).to(device)  # (1, seq_len)
        y = y.unsqueeze(0).to(device)  # (1, seq_len)

        logits, _ = model(x)  # (1, seq_len, vocab_size)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            reduction='mean'
        )
    return loss.item()


def main():
    # Configuration
    run_folder = "runs_ascent_descent"  # Change to "runs_finetune" if needed
    data_folder = "data_splits_speakers300_fs10"
    output_file = f"forget_loss_analysis_{run_folder}.json"

    # Load metadata
    meta_path = Path(data_folder) / "meta.json"
    meta = load_json(meta_path)
    char2idx = {c: i for i, c in enumerate(meta["vocab"])}
    vocab_size = meta["vocab_size"]
    seq_len = meta["seq_len"]

    print(f"Vocab size: {vocab_size}, Seq len: {seq_len}")
    print(f"Processing {run_folder}...")

    # Storage for results
    results = {}

    # Process each forget file
    forget_files = sorted([f for f in os.listdir(data_folder) if f.startswith("forget_") and f.endswith(".txt")])

    for forget_file in forget_files:
        forget_idx = int(forget_file.split("_")[1].split(".")[0])
        print(f"\n  Processing {forget_file} (forget_idx={forget_idx})...")

        # Read and encode forget file
        forget_path = Path(data_folder) / forget_file
        with open(forget_path, 'r', encoding='utf-8') as f:
            text = f.read()

        indices = encode_text_to_indices(text, char2idx)
        chunks = split_into_chunks(indices, seq_len)
        print(f"    Total chunks: {len(chunks)}")

        results[f"forget_{forget_idx}"] = {}

        # For each position (chunk) in this forget file
        for position, chunk in enumerate(chunks):
            # Storage for losses from 50 models
            losses_by_model = {}

            # Create x, y pairs for this chunk
            x = torch.tensor(chunk[:-1], dtype=torch.long)
            y = torch.tensor(chunk[1:], dtype=torch.long)

            # Process each run
            for run_id in range(50):
                run_path = Path(run_folder) / f"run_{run_id}"

                # Load models
                try:
                    trained_path = run_path / "model_trained.pt"
                    unlearnt_path = run_path / "model_unlearnt.pt"

                    if not trained_path.exists() or not unlearnt_path.exists():
                        print(f"      Warning: Model files not found for run_{run_id}")
                        continue

                    # Load model architecture
                    model = ShakespeareLSTM(
                        vocab_size=vocab_size,
                        embed_dim=8,
                        hidden_size=256,
                        num_layers=2
                    )

                    # Try GPU first, fallback to CPU
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    model = model.to(device)

                    # Load trained model weights
                    model.load_state_dict(torch.load(trained_path, map_location=device))
                    trained_loss = compute_loss(model, x, y, device)

                    # Load unlearnt model weights
                    model.load_state_dict(torch.load(unlearnt_path, map_location=device))
                    unlearnt_loss = compute_loss(model, x, y, device)

                    losses_by_model[run_id] = {
                        "trained": trained_loss,
                        "unlearnt": unlearnt_loss
                    }

                except Exception as e:
                    print(f"      Error processing run_{run_id}: {e}")
                    continue

            # Now group models by whether forget_idx is in their forget_indices
            chosen_losses = {"trained": [], "unlearnt": []}
            not_chosen_losses = {"trained": [], "unlearnt": []}

            for run_id, losses in losses_by_model.items():
                # Load forget_indices for this run
                forget_indices_path = Path(run_folder) / f"run_{run_id}" / "forget_indices.json"
                try:
                    forget_indices_data = load_json(forget_indices_path)
                    forget_indices = forget_indices_data["forget_indices"]

                    if forget_idx in forget_indices:
                        chosen_losses["trained"].append(losses["trained"])
                        chosen_losses["unlearnt"].append(losses["unlearnt"])
                    else:
                        not_chosen_losses["trained"].append(losses["trained"])
                        not_chosen_losses["unlearnt"].append(losses["unlearnt"])
                except Exception as e:
                    print(f"      Error reading forget_indices for run_{run_id}: {e}")

            # Compute statistics for both groups
            position_results = {}

            for group_name, losses_dict in [("chosen", chosen_losses), ("not_chosen", not_chosen_losses)]:
                for model_type in ["trained", "unlearnt"]:
                    losses = losses_dict[model_type]
                    if losses:
                        stats = {
                            "mean": float(np.mean(losses)),
                            "variance": float(np.var(losses)),
                            "count": len(losses)
                        }
                    else:
                        stats = {
                            "mean": None,
                            "variance": None,
                            "count": 0
                        }

                    position_results[f"{group_name}_{model_type}"] = stats

            results[f"forget_{forget_idx}"][f"position_{position}"] = position_results

        print(f"    Completed {forget_file}")

    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nAnalysis complete! Results saved to {output_file}")


if __name__ == "__main__":
    main()
