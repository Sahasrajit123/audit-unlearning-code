#!/usr/bin/env python3
"""
evaluate_mia_text_folder_based.py -- folder-based MIA evaluation for text unlearning runs.

This script follows the same high-level MIA style as the folder-based vision attack:
1) Build per-run attack features from trained vs unlearnt model behavior.
2) Aggregate features from train runs and train one attack classifier.
3) Evaluate that fixed attack classifier on each test run.

Text compatibility details:
- Uses per-window sequence losses (not image posteriors).
- Uses chosen forget windows as members and complement forget windows as non-members.
- Uses trained/unlearnt checkpoints: model_trained.pt and model_unlearnt.pt.
- Feature names are trained_loss and unlearnt_loss instead of generic labels.
- Optional feature modes can use only unlearnt loss stats or a separate unlearnt gradient norm.
- Loss-stat features per window: mean, median, min, max, and loss slope.
- Separate feature mode can use true unlearnt gradient norm per window: ||grad_theta(loss)||_2.

Expected layout:
- Train runs: <runs_dir>/run_*/
- Test runs:  <runs_dir>/test_run/run_*/
- Per run: model_trained.pt, model_unlearnt.pt, forget_indices.json

How to run:
    conda run -n torch_jax_gpu python evaluate_mia_text_folder_based.py \
        --runs_dir runs_ascent_descent_fs400 \
        --data_dir data_splits_speakers300_fs400

Optional quick smoke run:
    conda run -n torch_jax_gpu python evaluate_mia_text_folder_based.py \
        --runs_dir runs_ascent_descent_fs400 \
        --data_dir data_splits_speakers300_fs400 \
        --max_train_runs 1 \
        --max_test_runs 1

Current label source:
- Positive/member samples: windows from forget texts selected by each run's forget_indices.json.
- Negative/non-member samples: windows from forget texts NOT selected by forget_indices.json.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

import sys

sys.path.insert(0, "lib")

from archive_compute_forget_set_losses_fast import compute_loss_on_text_batched
from model import ShakespeareLSTM
from trainer_utils import load_config

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler


def create_file_logger(name, log_path):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(str(log_path), mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


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


def encode_text(text, char2idx):
    encoded = [char2idx[c] for c in text if c in char2idx]
    return torch.tensor(encoded, dtype=torch.long)


def resolve_config_path(runs_dir, explicit_config=None):
    if explicit_config is not None:
        return Path(explicit_config)

    candidates = [
        runs_dir / "experiment_config.json",
        runs_dir / "config.json",
    ]
    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"could not find an experiment config in {runs_dir} (looked for experiment_config.json and config.json)"
    )


def build_windows(text, window_chars, stride_chars):
    if window_chars <= 0:
        raise ValueError("window_chars must be > 0")
    if stride_chars <= 0:
        raise ValueError("stride_chars must be > 0")

    if not text:
        return []

    if len(text) <= window_chars:
        return [text]

    windows = []
    last_start = len(text) - window_chars
    for start in range(0, last_start + 1, stride_chars):
        windows.append(text[start : start + window_chars])

    if windows and windows[-1] != text[-window_chars:]:
        windows.append(text[-window_chars:])

    return windows


def sample_retain_windows(retain_windows, count, rng):
    if count <= 0 or not retain_windows:
        return []
    replace = len(retain_windows) < count
    idx = rng.choice(len(retain_windows), size=count, replace=replace)
    return [retain_windows[i] for i in idx]


def complement_indices(total_count, chosen_indices):
    chosen = set(chosen_indices)
    return [i for i in range(total_count) if i not in chosen]


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


def per_text_loss_stats(model, texts, seq_len, char2idx, device, vocab_size):
    means = []
    medians = []
    mins = []
    maxs = []
    loss_slopes = []

    for text in texts:
        vals = compute_loss_on_text_batched(model, text, seq_len, char2idx, device, vocab_size)
        if vals:
            arr = np.asarray(vals, dtype=float)
            means.append(float(np.mean(arr)))
            medians.append(float(np.median(arr)))
            mins.append(float(np.min(arr)))
            maxs.append(float(np.max(arr)))
            if arr.size > 1:
                # True gradient feature: least-squares slope of loss over position.
                # Equivalent to fitting y = a*x + b and taking slope a.
                x = np.arange(arr.size, dtype=float)
                x_centered = x - np.mean(x)
                y_centered = arr - np.mean(arr)
                denom = float(np.dot(x_centered, x_centered))
                slope = 0.0 if denom == 0.0 else float(np.dot(x_centered, y_centered) / denom)
            else:
                slope = 0.0
            loss_slopes.append(slope)
        else:
            means.append(np.nan)
            medians.append(np.nan)
            mins.append(np.nan)
            maxs.append(np.nan)
            loss_slopes.append(np.nan)

    return {
        "mean": np.array(means, dtype=float),
        "median": np.array(medians, dtype=float),
        "min": np.array(mins, dtype=float),
        "max": np.array(maxs, dtype=float),
        "loss_slope": np.array(loss_slopes, dtype=float),
    }


def grad_norm_for_text(model, text, seq_len, char2idx, device, vocab_size, batch_size=64):
    """Compute ||grad_theta(loss)||_2 for one text window under the current model."""
    data = encode_text(text, char2idx)
    num_positions = max(0, len(data) - seq_len)
    if num_positions <= 0:
        return np.nan

    criterion = torch.nn.CrossEntropyLoss(reduction="mean")

    was_training = model.training
    model.train()
    model.zero_grad(set_to_none=True)

    total_loss_weighted = None
    total_items = 0

    for start_idx in range(0, num_positions, batch_size):
        end_idx = min(start_idx + batch_size, num_positions)

        xs = []
        ys = []
        for idx in range(start_idx, end_idx):
            x = data[idx : idx + seq_len]
            y = data[idx + 1 : idx + seq_len + 1]
            xs.append(x)
            ys.append(y)

        xs = torch.stack(xs).to(device)
        ys = torch.stack(ys).to(device)

        logits, _ = model(xs)
        ce = criterion(logits.reshape(-1, vocab_size), ys.reshape(-1))
        n_items = int(ys.numel())
        weighted = ce * n_items

        total_loss_weighted = weighted if total_loss_weighted is None else (total_loss_weighted + weighted)
        total_items += n_items

    if total_loss_weighted is None or total_items == 0:
        return np.nan

    loss = total_loss_weighted / float(total_items)
    loss.backward()

    sq_sum = 0.0
    for p in model.parameters():
        if p.grad is not None:
            sq_sum += float(torch.sum(p.grad.detach() ** 2).item())

    model.zero_grad(set_to_none=True)
    model.train(was_training)
    return float(np.sqrt(sq_sum))


def per_text_grad_norms(model, texts, seq_len, char2idx, device, vocab_size, grad_batch_size=64):
    out = []
    for text in texts:
        out.append(
            grad_norm_for_text(
                model=model,
                text=text,
                seq_len=seq_len,
                char2idx=char2idx,
                device=device,
                vocab_size=vocab_size,
                batch_size=grad_batch_size,
            )
        )
    return np.array(out, dtype=float)


def to_loss_features(trained_stats, unlearnt_stats, feature_mode):
    stat_names = ["mean", "median", "min", "max", "loss_slope"]

    if feature_mode == "unlearnt_only":
        return np.column_stack([unlearnt_stats[name] for name in stat_names])

    eps = 1e-8
    cols = []
    for name in stat_names:
        trained_loss = trained_stats[name]
        unlearnt_loss = unlearnt_stats[name]
        cols.extend(
            [
                trained_loss,
                unlearnt_loss,
                trained_loss - unlearnt_loss,
                unlearnt_loss - trained_loss,
                trained_loss / (unlearnt_loss + eps),
                unlearnt_loss / (trained_loss + eps),
            ]
        )
    return np.column_stack(cols)


def finite_row_filter(x):
    mask = np.isfinite(x).all(axis=1)
    return x[mask]


def balance_binary_features(feat_pos, feat_neg, seed):
    n_pos = feat_pos.shape[0]
    n_neg = feat_neg.shape[0]
    n = min(n_pos, n_neg)
    if n == 0:
        raise ValueError("At least one class is empty after feature construction")

    rng = np.random.default_rng(seed)
    pos_idx = rng.choice(n_pos, size=n, replace=False)
    neg_idx = rng.choice(n_neg, size=n, replace=False)

    x = np.concatenate([feat_pos[pos_idx], feat_neg[neg_idx]], axis=0)
    y = np.concatenate([np.ones(n, dtype=int), np.zeros(n, dtype=int)], axis=0)

    perm = rng.permutation(len(y))
    return x[perm], y[perm], {
        "n_pos_before": int(n_pos),
        "n_neg_before": int(n_neg),
        "n_per_class_after": int(n),
    }


def tpr_at_target_fpr(y_true, y_score, target_fpr=0.01):
    try:
        fpr, tpr, _ = roc_curve(y_true, y_score)
    except ValueError:
        return None

    valid = tpr[fpr <= target_fpr]
    if valid.size == 0:
        return 0.0
    return float(np.max(valid))


class LogisticAttack:
    def __init__(self, seed):
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(random_state=seed, max_iter=1000)

    def fit(self, x, y):
        self.scaler.fit(x)
        self.clf.fit(self.scaler.transform(x), y)

    def evaluate(self, x, y):
        xs = self.scaler.transform(x)
        pred = self.clf.predict(xs)
        prob = self.clf.predict_proba(xs)[:, 1]
        return {
            "attack_accuracy": float(accuracy_score(y, pred)),
            "attack_auc": float(roc_auc_score(y, prob)),
            "attack_precision": float(precision_score(y, pred, zero_division=0)),
            "attack_recall": float(recall_score(y, pred, zero_division=0)),
            "attack_f1": float(f1_score(y, pred, zero_division=0)),
            "tpr_at_1pct_fpr": tpr_at_target_fpr(y, prob, target_fpr=0.01),
            "tpr_at_5pct_fpr": tpr_at_target_fpr(y, prob, target_fpr=0.05),
            "num_samples": int(len(y)),
        }


def aggregate_results(per_run_results):
    metric_names = [
        "attack_accuracy",
        "attack_auc",
        "attack_precision",
        "attack_recall",
        "attack_f1",
        "tpr_at_1pct_fpr",
        "tpr_at_5pct_fpr",
    ]
    out = {}
    for key in metric_names:
        vals = [r["results"][key] for r in per_run_results if r.get("results", {}).get(key) is not None]
        if not vals:
            out[key] = {"mean": None, "std": None, "count": 0}
            continue
        arr = np.asarray(vals, dtype=float)
        out[key] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)) if arr.size > 1 else 0.0,
            "count": int(arr.size),
        }
    return out


def find_train_run_dirs(runs_dir):
    run_dirs = []
    for item in sorted(runs_dir.iterdir()):
        if not item.is_dir():
            continue
        if item.name == "test_run":
            continue
        if item.name.startswith("run_"):
            run_dirs.append(item)
    return run_dirs


def find_test_run_dirs(runs_dir):
    test_root = runs_dir / "test_run"
    if not test_root.is_dir():
        return []

    run_dirs = []
    for item in sorted(test_root.iterdir()):
        if not item.is_dir():
            continue
        if item.name.startswith("run_"):
            run_dirs.append(item)
    return run_dirs


def parse_run_id(run_dir):
    try:
        return int(run_dir.name.split("_")[1])
    except (ValueError, IndexError):
        return 0


def build_run_features(
    run_dir,
    cfg,
    device,
    seq_len,
    char2idx,
    vocab_size,
    forget_texts,
    window_chars,
    window_stride,
    feature_mode,
):
    trained_path = run_dir / "model_trained.pt"
    unlearnt_path = run_dir / "model_unlearnt.pt"
    forget_indices_path = run_dir / "forget_indices.json"

    if not trained_path.exists() or not unlearnt_path.exists() or not forget_indices_path.exists():
        raise FileNotFoundError(
            f"missing one of required files in {run_dir}: model_trained.pt, model_unlearnt.pt, forget_indices.json"
        )

    with open(forget_indices_path, "r", encoding="utf-8") as f:
        forget_info = json.load(f)

    forget_indices = sorted(forget_info.get("forget_indices", []))
    if not forget_indices:
        raise ValueError(f"empty forget_indices in {forget_indices_path}")

    if min(forget_indices) < 0 or max(forget_indices) >= len(forget_texts):
        raise IndexError(
            f"forget_indices out of range for {forget_indices_path}; "
            f"valid range is [0, {len(forget_texts) - 1}]"
        )

    neg_forget_indices = complement_indices(len(forget_texts), forget_indices)
    if not neg_forget_indices:
        raise ValueError(
            "no complement forget indices available for negative class; "
            "run selected all forget files"
        )

    run_id = parse_run_id(run_dir)

    merged_forget_text = "\n".join(forget_texts[i] for i in forget_indices)
    merged_neg_forget_text = "\n".join(forget_texts[i] for i in neg_forget_indices)

    forget_windows = build_windows(merged_forget_text, window_chars, window_stride)
    neg_forget_windows = build_windows(merged_neg_forget_text, window_chars, window_stride)
    if not forget_windows:
        raise ValueError("no member forget windows were created")
    if not neg_forget_windows:
        raise ValueError("no non-member complement-forget windows were created")

    trained_model = None
    unlearnt_model = None

    if feature_mode == "unlearnt_only":
        unlearnt_model = load_text_model(str(unlearnt_path), vocab_size, device, cfg)

        unlearnt_forget_stats = per_text_loss_stats(
            unlearnt_model, forget_windows, seq_len, char2idx, device, vocab_size
        )
        unlearnt_neg_forget_stats = per_text_loss_stats(
            unlearnt_model,
            neg_forget_windows,
            seq_len,
            char2idx,
            device,
            vocab_size,
        )
        feat_pos = finite_row_filter(
            np.column_stack([
                unlearnt_forget_stats["mean"],
                unlearnt_forget_stats["median"],
                unlearnt_forget_stats["min"],
                unlearnt_forget_stats["max"],
                unlearnt_forget_stats["loss_slope"],
            ])
        )
        feat_neg = finite_row_filter(
            np.column_stack([
                unlearnt_neg_forget_stats["mean"],
                unlearnt_neg_forget_stats["median"],
                unlearnt_neg_forget_stats["min"],
                unlearnt_neg_forget_stats["max"],
                unlearnt_neg_forget_stats["loss_slope"],
            ])
        )
    elif feature_mode == "unlearnt_grad_norm":
        unlearnt_model = load_text_model(str(unlearnt_path), vocab_size, device, cfg)

        unlearnt_forget_grad_norm = per_text_grad_norms(
            unlearnt_model, forget_windows, seq_len, char2idx, device, vocab_size
        )
        unlearnt_neg_forget_grad_norm = per_text_grad_norms(
            unlearnt_model, neg_forget_windows, seq_len, char2idx, device, vocab_size
        )
        feat_pos = finite_row_filter(unlearnt_forget_grad_norm.reshape(-1, 1))
        feat_neg = finite_row_filter(unlearnt_neg_forget_grad_norm.reshape(-1, 1))
    else:
        trained_model = load_text_model(str(trained_path), vocab_size, device, cfg)
        unlearnt_model = load_text_model(str(unlearnt_path), vocab_size, device, cfg)

        trained_forget_stats = per_text_loss_stats(trained_model, forget_windows, seq_len, char2idx, device, vocab_size)
        trained_neg_forget_stats = per_text_loss_stats(
            trained_model,
            neg_forget_windows,
            seq_len,
            char2idx,
            device,
            vocab_size,
        )
        unlearnt_forget_stats = per_text_loss_stats(
            unlearnt_model, forget_windows, seq_len, char2idx, device, vocab_size
        )
        unlearnt_neg_forget_stats = per_text_loss_stats(
            unlearnt_model,
            neg_forget_windows,
            seq_len,
            char2idx,
            device,
            vocab_size,
        )

        feat_pos = finite_row_filter(to_loss_features(trained_forget_stats, unlearnt_forget_stats, feature_mode))
        feat_neg = finite_row_filter(to_loss_features(trained_neg_forget_stats, unlearnt_neg_forget_stats, feature_mode))

    if trained_model is not None:
        del trained_model
    if unlearnt_model is not None:
        del unlearnt_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "run_dir": str(run_dir),
        "run_id": int(run_id),
        "forget_indices": forget_indices,
        "neg_forget_indices": neg_forget_indices,
        "feature_pos": feat_pos,
        "feature_neg": feat_neg,
        "feature_dims": int(feat_pos.shape[1]) if feat_pos.size else 0,
        "n_member_windows": int(len(forget_windows)),
        "n_nonmember_windows": int(len(neg_forget_windows)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Folder-based MIA for text unlearning (train attack on train runs, evaluate on test runs)"
    )
    parser.add_argument("--runs_dir", default="runs_ascent_descent_fs400", help="directory containing train run_* and test_run/run_*")
    parser.add_argument("--data_dir", default="data_splits_speakers300_fs400", help="data split directory")
    parser.add_argument(
        "--config",
        default=None,
        help="optional model/training config path; defaults to runs_dir/experiment_config.json if present",
    )
    parser.add_argument("--output_path", default=None, help="summary output JSON path")
    parser.add_argument("--device", default=None, help="torch device string")
    parser.add_argument("--seed", type=int, default=42, help="seed for sampling and attack model")
    parser.add_argument(
        "--window_chars",
        type=int,
        default=1200,
        help="character length for each forget/retain evaluation window",
    )
    parser.add_argument(
        "--window_stride",
        type=int,
        default=300,
        help="stride in characters when creating windows",
    )
    parser.add_argument(
        "--feature_mode",
        choices=["paired", "unlearnt_only", "unlearnt_grad_norm"],
        default="paired",
        help=(
            "feature construction mode: paired uses trained/unlearnt loss stats, "
            "unlearnt_only uses unlearnt loss stats, "
            "unlearnt_grad_norm uses ||grad_theta(loss)||_2 from unlearnt model"
        ),
    )
    parser.add_argument("--max_train_runs", type=int, default=None, help="optional cap on number of train runs")
    parser.add_argument("--max_test_runs", type=int, default=None, help="optional cap on number of test runs")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    data_dir = Path(args.data_dir)
    config_path = resolve_config_path(runs_dir, args.config)
    cfg = load_config(str(config_path))

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[main] using device: {device}")
    print(f"[main] using config: {config_path}")

    meta_path = data_dir / "meta.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    seq_len = meta["seq_len"]
    char2idx = {c: i for i, c in enumerate(meta["vocab"])}
    vocab_size = meta["vocab_size"]

    forget_texts = load_forget_texts(data_dir)

    if not forget_texts:
        raise RuntimeError(f"No forget texts found under {data_dir}")

    train_run_dirs = find_train_run_dirs(runs_dir)
    test_run_dirs = find_test_run_dirs(runs_dir)

    if not train_run_dirs:
        raise RuntimeError(f"No train runs found under {runs_dir} (expected run_*)")
    if not test_run_dirs:
        raise RuntimeError(f"No test runs found under {runs_dir / 'test_run'} (expected run_*)")

    if args.max_train_runs is not None:
        train_run_dirs = train_run_dirs[: args.max_train_runs]
    if args.max_test_runs is not None:
        test_run_dirs = test_run_dirs[: args.max_test_runs]

    summary_dir = runs_dir / "mia_text_folder_based"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_logger = create_file_logger("mia_text_summary", summary_dir / "run.log")
    summary_logger.info("MIA folder-based text evaluation started")
    summary_logger.info("runs_dir=%s", runs_dir)
    summary_logger.info("data_dir=%s", data_dir)
    summary_logger.info("config_path=%s", config_path)
    summary_logger.info("window_chars=%d window_stride=%d", args.window_chars, args.window_stride)
    summary_logger.info("feature_mode=%s", args.feature_mode)
    summary_logger.info("seed=%d", args.seed)
    summary_logger.info("num_train_runs=%d num_test_runs=%d", len(train_run_dirs), len(test_run_dirs))

    print("=" * 80)
    print("Running folder-based text MIA")
    print("=" * 80)
    print(f"train runs: {len(train_run_dirs)}")
    print(f"test runs:  {len(test_run_dirs)}")

    train_chunks = []
    failed_train_runs = []

    for i, run_dir in enumerate(train_run_dirs, start=1):
        print(f"\n[train {i}/{len(train_run_dirs)}] {run_dir.name}")
        try:
            chunk = build_run_features(
                run_dir=run_dir,
                cfg=cfg,
                device=device,
                seq_len=seq_len,
                char2idx=char2idx,
                vocab_size=vocab_size,
                forget_texts=forget_texts,
                window_chars=args.window_chars,
                window_stride=args.window_stride,
                feature_mode=args.feature_mode,
            )
            train_chunks.append(chunk)
            print(
                f"  feature rows: pos={chunk['feature_pos'].shape[0]} neg={chunk['feature_neg'].shape[0]} dim={chunk['feature_dims']}"
            )
        except Exception as exc:
            failed_train_runs.append({"run_dir": str(run_dir), "error": str(exc)})
            print(f"  TRAIN FAILED: {exc}")

    if not train_chunks:
        raise RuntimeError("No train runs succeeded; cannot train attack model")

    train_pos = np.concatenate([c["feature_pos"] for c in train_chunks], axis=0)
    train_neg = np.concatenate([c["feature_neg"] for c in train_chunks], axis=0)

    x_train, y_train, train_balance = balance_binary_features(train_pos, train_neg, seed=args.seed)

    attack = LogisticAttack(seed=args.seed)
    attack.fit(x_train, y_train)
    train_metrics = attack.evaluate(x_train, y_train)

    print(
        "\n[train] classifier fit complete: "
        f"acc={100 * train_metrics['attack_accuracy']:.2f}% "
        f"auc={100 * train_metrics['attack_auc']:.2f}% "
        f"f1={100 * train_metrics['attack_f1']:.2f}%"
    )

    per_test_run_results = []
    failed_test_runs = []
    pooled_test_x = []
    pooled_test_y = []

    for i, run_dir in enumerate(test_run_dirs, start=1):
        print(f"\n[test {i}/{len(test_run_dirs)}] {run_dir.name}")
        try:
            chunk = build_run_features(
                run_dir=run_dir,
                cfg=cfg,
                device=device,
                seq_len=seq_len,
                char2idx=char2idx,
                vocab_size=vocab_size,
                forget_texts=forget_texts,
                window_chars=args.window_chars,
                window_stride=args.window_stride,
                feature_mode=args.feature_mode,
            )
            x_test, y_test, test_balance = balance_binary_features(
                chunk["feature_pos"], chunk["feature_neg"], seed=args.seed
            )
            metrics = attack.evaluate(x_test, y_test)
            pooled_test_x.append(x_test)
            pooled_test_y.append(y_test)

            run_output = {
                "run_id": int(chunk["run_id"]),
                "run_dir": str(run_dir),
                "config": {
                    "runs_dir": str(runs_dir.resolve()),
                    "data_dir": str(data_dir.resolve()),
                    "config_path": str(config_path.resolve()),
                    "window_chars": args.window_chars,
                    "window_stride": args.window_stride,
                    "feature_mode": args.feature_mode,
                    "seed": args.seed,
                    "split": "test",
                },
                "dataset_info": {
                    "positive_source": "chosen forget windows (member)",
                    "negative_source": "complement forget windows (non-member)",
                    "n_positive_before_balance": int(chunk["feature_pos"].shape[0]),
                    "n_negative_before_balance": int(chunk["feature_neg"].shape[0]),
                    "n_per_class_after_balance": int(test_balance["n_per_class_after"]),
                    "n_total_after_balance": int(x_test.shape[0]),
                },
                "results": metrics,
            }

            run_output_dir = run_dir / "mia_text_folder_based"
            run_output_dir.mkdir(parents=True, exist_ok=True)
            run_output_path = run_output_dir / "run_result.json"
            with open(run_output_path, "w", encoding="utf-8") as f:
                json.dump(run_output, f, indent=2)

            per_test_run_results.append(run_output)
            print(
                "  metrics: "
                f"acc={100 * metrics['attack_accuracy']:.2f}% "
                f"auc={100 * metrics['attack_auc']:.2f}% "
                f"f1={100 * metrics['attack_f1']:.2f}%"
            )
            print(f"  saved: {run_output_path}")
        except Exception as exc:
            failed_test_runs.append({"run_dir": str(run_dir), "error": str(exc)})
            print(f"  TEST FAILED: {exc}")

    combined_test_metrics = None
    combined_test_samples = 0
    if pooled_test_x:
        x_test_all = np.concatenate(pooled_test_x, axis=0)
        y_test_all = np.concatenate(pooled_test_y, axis=0)
        combined_test_samples = int(x_test_all.shape[0])
        combined_test_metrics = attack.evaluate(x_test_all, y_test_all)

    summary = {
        "config": {
            "runs_dir": str(runs_dir.resolve()),
            "data_dir": str(data_dir.resolve()),
            "config_path": str(config_path.resolve()),
            "seed": args.seed,
            "window_chars": args.window_chars,
            "window_stride": args.window_stride,
            "feature_mode": args.feature_mode,
        },
        "train": {
            "num_runs_total": len(train_run_dirs),
            "num_runs_succeeded": len(train_chunks),
            "num_runs_failed": len(failed_train_runs),
            "failed_runs": failed_train_runs,
            "n_positive_before_balance": int(train_balance["n_pos_before"]),
            "n_negative_before_balance": int(train_balance["n_neg_before"]),
            "n_per_class_after_balance": int(train_balance["n_per_class_after"]),
            "n_total_after_balance": int(x_train.shape[0]),
            "metrics_on_train": train_metrics,
        },
        "test": {
            "num_runs_total": len(test_run_dirs),
            "num_runs_succeeded": len(per_test_run_results),
            "num_runs_failed": len(failed_test_runs),
            "failed_runs": failed_test_runs,
            "stats": aggregate_results(per_test_run_results),
            "combined_inference": {
                "num_samples": combined_test_samples,
                "metrics": combined_test_metrics,
            },
            "per_run_results_file": "mia_text_folder_based/run_result.json",
        },
    }

    output_path = Path(args.output_path) if args.output_path else (summary_dir / "summary.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    summary_logger.info("Processed train runs: %d", len(train_run_dirs))
    summary_logger.info("Processed test runs: %d", len(test_run_dirs))
    summary_logger.info("Test AUC mean=%s", str(summary["test"]["stats"]["attack_auc"]["mean"]))
    summary_logger.info(
        "Test combined AUC=%s",
        str(None if combined_test_metrics is None else combined_test_metrics.get("attack_auc")),
    )
    summary_logger.info("Saved summary: %s", output_path)

    print("\n" + "=" * 80)
    print("Batch attack completed")
    print("=" * 80)
    print(f"train runs succeeded: {len(train_chunks)}/{len(train_run_dirs)}")
    print(f"test runs succeeded: {len(per_test_run_results)}/{len(test_run_dirs)}")
    if combined_test_metrics is not None:
        print(
            "[test combined] "
            f"samples={combined_test_samples} "
            f"acc={100 * combined_test_metrics['attack_accuracy']:.2f}% "
            f"auc={100 * combined_test_metrics['attack_auc']:.2f}% "
            f"f1={100 * combined_test_metrics['attack_f1']:.2f}%"
        )
    print(f"summary: {output_path}")


if __name__ == "__main__":
    main()
