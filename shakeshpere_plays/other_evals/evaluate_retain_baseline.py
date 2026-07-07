#!/usr/bin/env python3
"""
evaluate_retain_baseline.py -- compare trained, unlearnt, and retain-only baseline models.

This script adapts the unlearning evaluation to text by using sequence loss on the
run-specific forget texts plus standard retain/test metrics. It assumes a shared
retain-only checkpoint has already been trained once and can be reused across runs.

Usage:
  python evaluate_retain_baseline.py \
    --runs_dir runs_ascent_descent_fs400 \
    --data_dir data_splits_speakers300_fs400 \
        --retain_baseline_path retain_only_baseline/retain_only/model_retain.pt
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, 'lib')

from archive_compute_forget_set_losses_fast import compute_loss_on_text_batched
from data_loader import load_data_splits
from engine import evaluate
from model import ShakespeareLSTM
from trainer_utils import load_config


def get_forget_files(data_dir):
    data_path = Path(data_dir)
    forget_dir = data_path / "forget"

    if forget_dir.exists() and forget_dir.is_dir():
        forget_files = sorted([p.name for p in forget_dir.glob("forget_*.txt")])
        if forget_files:
            return ["forget/" + f for f in forget_files]

    forget_files = sorted([p.name for p in data_path.glob("forget_*.txt")])
    if not forget_files and (data_path / "forget.txt").exists():
        forget_files = ["forget.txt"]

    return forget_files


def load_text_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_forget_texts(data_dir):
    files = get_forget_files(data_dir)
    return [load_text_file(Path(data_dir) / fname) for fname in files]


def load_text_model(model_path, vocab_size, device, cfg):
    model = ShakespeareLSTM(
        vocab_size=vocab_size,
        embed_dim=cfg["embed_dim"],
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
    )
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.to(device)
    model.eval()
    return model


def mean_loss_for_texts(model, texts, seq_len, char2idx, device, vocab_size):
    all_losses = []
    per_text_means = []

    for text in texts:
        losses = compute_loss_on_text_batched(model, text, seq_len, char2idx, device, vocab_size)
        if not losses:
            continue
        per_text_means.append(float(np.mean(losses)))
        all_losses.extend(losses)

    if not all_losses:
        return {
            "mean_loss": None,
            "perplexity": None,
            "num_texts": 0,
            "num_positions": 0,
            "per_text_mean_losses": [],
        }

    mean_loss = float(np.mean(all_losses))
    return {
        "mean_loss": mean_loss,
        "perplexity": float(math.exp(mean_loss)),
        "num_texts": len(per_text_means),
        "num_positions": len(all_losses),
        "per_text_mean_losses": per_text_means,
    }


def evaluate_model(model, retain_loader, test_loader, forget_texts, selected_indices, seq_len, char2idx, device, vocab_size):
    selected_forget_texts = [forget_texts[i] for i in selected_indices]
    criterion = torch.nn.CrossEntropyLoss()

    retain_loss, retain_acc, retain_ppl = evaluate(model, retain_loader, criterion, device)
    test_loss, test_acc, test_ppl = evaluate(model, test_loader, criterion, device)
    forget_stats = mean_loss_for_texts(model, selected_forget_texts, seq_len, char2idx, device, vocab_size)

    return {
        "retain_loss": float(retain_loss),
        "retain_accuracy": float(retain_acc),
        "retain_perplexity": float(retain_ppl),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "test_perplexity": float(test_ppl),
        "forget_loss": forget_stats["mean_loss"],
        "forget_perplexity": forget_stats["perplexity"],
        "forget_selected_count": len(selected_indices),
        "forget_text_count": forget_stats["num_texts"],
        "forget_position_count": forget_stats["num_positions"],
        "forget_per_text_mean_losses": forget_stats["per_text_mean_losses"],
    }


def main():
    parser = argparse.ArgumentParser(description="Compare unlearning runs against a shared retain-only baseline")
    parser.add_argument("--runs_dir", default="runs_ascent_descent_fs400", help="directory containing run_* folders")
    parser.add_argument("--data_dir", default="data_splits_speakers300_fs400", help="directory containing the data split")
    parser.add_argument(
        "--retain_baseline_path",
        default="retain_only_baseline/retain_only/model_retain.pt",
        help="path to the shared retain-only checkpoint",
    )
    parser.add_argument("--config", default="config.json", help="path to the model/training config used by the runs")
    parser.add_argument("--output_path", default=None, help="where to save the comparison JSON")
    parser.add_argument("--device", default=None, help="torch device string (default: cuda if available)")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    data_dir = Path(args.data_dir)
    retain_baseline_path = Path(args.retain_baseline_path)
    output_path = Path(args.output_path) if args.output_path else runs_dir / "retain_baseline_comparison.json"

    if not retain_baseline_path.exists():
        raise FileNotFoundError(f"retain baseline checkpoint not found: {retain_baseline_path}")

    cfg = load_config(args.config)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[main] using device: {device}")

    loaders, metadata = load_data_splits(
        str(data_dir),
        cfg["batch_size"],
        cfg["seed"],
        shuffle_train_samples=False,
        shuffle_train_batches_each_epoch=False,
    )

    retain_loader = loaders["retain"]
    test_loader = loaders["test"]
    char2idx = metadata["char2idx"]

    meta_path = data_dir / "meta.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    seq_len = meta["seq_len"]
    vocab_size = metadata["vocab_size"]

    forget_texts = load_forget_texts(str(data_dir))
    print(f"[main] loaded {len(forget_texts)} forget files")

    baseline_model = load_text_model(str(retain_baseline_path), vocab_size, device, cfg)
    baseline_metrics = evaluate_model(
        baseline_model,
        retain_loader,
        test_loader,
        forget_texts,
        selected_indices=list(range(len(forget_texts))),
        seq_len=seq_len,
        char2idx=char2idx,
        device=device,
        vocab_size=vocab_size,
    )
    del baseline_model
    torch.cuda.empty_cache()

    run_dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("run_")])
    results = []

    for run_dir in run_dirs:
        try:
            run_id = int(run_dir.name.split("_")[1])
        except (ValueError, IndexError):
            continue

        trained_path = run_dir / "model_trained.pt"
        unlearnt_path = run_dir / "model_unlearnt.pt"
        forget_indices_path = run_dir / "forget_indices.json"

        if not trained_path.exists() or not unlearnt_path.exists() or not forget_indices_path.exists():
            continue

        with open(forget_indices_path, "r", encoding="utf-8") as f:
            forget_info = json.load(f)
        selected_forget_indices = sorted(forget_info.get("forget_indices", []))

        trained_model = load_text_model(str(trained_path), vocab_size, device, cfg)
        unlearnt_model = load_text_model(str(unlearnt_path), vocab_size, device, cfg)

        trained_metrics = evaluate_model(
            trained_model,
            retain_loader,
            test_loader,
            forget_texts,
            selected_forget_indices,
            seq_len,
            char2idx,
            device,
            vocab_size,
        )
        unlearnt_metrics = evaluate_model(
            unlearnt_model,
            retain_loader,
            test_loader,
            forget_texts,
            selected_forget_indices,
            seq_len,
            char2idx,
            device,
            vocab_size,
        )

        result = {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "forget_indices": selected_forget_indices,
            "trained": trained_metrics,
            "unlearnt": unlearnt_metrics,
            "retain_baseline": baseline_metrics,
            "deltas": {
                "forget_loss_unlearnt_minus_trained": None
                if trained_metrics["forget_loss"] is None or unlearnt_metrics["forget_loss"] is None
                else float(unlearnt_metrics["forget_loss"] - trained_metrics["forget_loss"]),
                "forget_loss_unlearnt_minus_baseline": None
                if baseline_metrics["forget_loss"] is None or unlearnt_metrics["forget_loss"] is None
                else float(unlearnt_metrics["forget_loss"] - baseline_metrics["forget_loss"]),
                "forget_loss_trained_minus_baseline": None
                if baseline_metrics["forget_loss"] is None or trained_metrics["forget_loss"] is None
                else float(trained_metrics["forget_loss"] - baseline_metrics["forget_loss"]),
                "retain_loss_unlearnt_minus_baseline": float(unlearnt_metrics["retain_loss"] - baseline_metrics["retain_loss"]),
                "test_loss_unlearnt_minus_baseline": float(unlearnt_metrics["test_loss"] - baseline_metrics["test_loss"]),
            },
        }
        results.append(result)

        print(
            f"[run {run_id}] forget loss trained={trained_metrics['forget_loss']:.4f} "
            f"unlearnt={unlearnt_metrics['forget_loss']:.4f} "
            f"baseline={baseline_metrics['forget_loss']:.4f}"
        )

        del trained_model, unlearnt_model
        torch.cuda.empty_cache()

    aggregate = {}
    for key in ["forget_loss", "retain_loss", "test_loss"]:
        aggregate[key] = {
            "trained_mean": float(np.mean([r["trained"][key] for r in results if r["trained"][key] is not None])) if results else None,
            "unlearnt_mean": float(np.mean([r["unlearnt"][key] for r in results if r["unlearnt"][key] is not None])) if results else None,
            "baseline_mean": float(np.mean([r["retain_baseline"][key] for r in results if r["retain_baseline"][key] is not None])) if results else None,
        }

    if results:
        aggregate["deltas"] = {
            "forget_loss_unlearnt_minus_baseline_mean": float(np.mean([
                r["deltas"]["forget_loss_unlearnt_minus_baseline"]
                for r in results
                if r["deltas"]["forget_loss_unlearnt_minus_baseline"] is not None
            ])),
            "forget_loss_trained_minus_baseline_mean": float(np.mean([
                r["deltas"]["forget_loss_trained_minus_baseline"]
                for r in results
                if r["deltas"]["forget_loss_trained_minus_baseline"] is not None
            ])),
            "retain_loss_unlearnt_minus_baseline_mean": float(np.mean([
                r["deltas"]["retain_loss_unlearnt_minus_baseline"] for r in results
            ])),
            "test_loss_unlearnt_minus_baseline_mean": float(np.mean([
                r["deltas"]["test_loss_unlearnt_minus_baseline"] for r in results
            ])),
        }

    output = {
        "config": {
            "runs_dir": str(runs_dir.resolve()),
            "data_dir": str(data_dir.resolve()),
            "retain_baseline_path": str(retain_baseline_path.resolve()),
            "config_path": str(Path(args.config).resolve()),
        },
        "baseline_metrics": baseline_metrics,
        "results": results,
        "aggregate": aggregate,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\n[done] saved comparison to {output_path}")
    if results:
        print(
            f"[summary] mean forget loss gap (unlearnt - baseline): {aggregate['deltas']['forget_loss_unlearnt_minus_baseline_mean']:.4f}"
        )
        print(
            f"[summary] mean forget loss gap (trained - baseline): {aggregate['deltas']['forget_loss_trained_minus_baseline_mean']:.4f}"
        )


if __name__ == "__main__":
    main()