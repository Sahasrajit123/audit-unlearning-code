# analyze_forget_stats_pointwise_batch.py
"""
Compute per-point phi (log-odds) and loss statistics for every point in each forget batch.

For each point in each batch, keys are batch_{batch_no}_point_{point_no}. For each point we
compute mean and variance across runs where that batch was SELECTED (in forget set) vs REMAINING
(not in forget set). Output includes both phi and loss: mean_phi, var_phi, mean_loss, var_loss
for selected and remaining.

Runs are the direct children of --runs_root (test_run/ is excluded; it is what the audit
evaluates). The forget pool is read from the `data_dir` recorded in the runs' run_vars.json.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import yaml

from src.models.model import ModelFactory
from src.utils.data_cache import load_split

EXCLUDED_RUN_DIR_NAMES = {"test_run"}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Analyze forget statistics per POINT in each batch (batch_{batch_no}_point_{point_no})'
    )
    parser.add_argument('--unlearn_itr', type=int, default=50,
                        help='Checkpoint iteration (default: 50)')
    parser.add_argument('--unlearn_style', type=str, default='epoch', choices=['epoch', 'step'],
                        help='Unlearning style: either "epoch" or "step" (default: epoch)')
    parser.add_argument('--runs_root', type=str, required=True,
                        help='Root directory containing the runs')
    parser.add_argument('--config_path', type=str, required=True,
                        help='Path to the config file')
    parser.add_argument('--trained_stats_only', action='store_true', default=False,
                        help='Load checkpoint_{unlearn_itr} (the trained model) instead of '
                             'unlearn_{unlearn_style}_{unlearn_itr}.')
    return parser.parse_args()


def _compute_loss_per_sample(logits, labels, *, num_classes: int):
    """Compute cross-entropy loss per sample (no reduction)."""
    one_hot = jax.nn.one_hot(labels, num_classes)
    return optax.softmax_cross_entropy(logits, one_hot)


def _compute_phi(logits, labels, *, num_classes: int):
    """Compute phi (log-odds): log(p/(1-p)) where p is probability for true class. Per-sample."""
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    log_p = jnp.take_along_axis(log_probs, labels[:, None], axis=-1).squeeze(-1)
    p = jnp.exp(log_p)
    eps = 1e-9
    p_clamped = jnp.clip(p, eps, 1.0 - eps)
    log_one_minus_p = jnp.log(1.0 - p_clamped)
    return log_p - log_one_minus_p


_compute_loss_per_sample = jax.jit(_compute_loss_per_sample, static_argnames=("num_classes",))
_compute_phi = jax.jit(_compute_phi, static_argnames=("num_classes",))


def make_eval_step_both_metrics(model, num_classes: int):
    """Create evaluation function that returns per-sample phi and loss."""
    @jax.jit
    def _eval_batches(params, x_concat: jnp.ndarray, y_concat: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        logits = model.apply({"params": params}, x_concat, train=False)
        phi_values = _compute_phi(logits, y_concat, num_classes=num_classes)
        loss_values = _compute_loss_per_sample(logits, y_concat, num_classes=num_classes)
        return phi_values, loss_values
    return _eval_batches


def _summarize(vals: List[float]) -> Tuple[float, float]:
    if len(vals) == 0:
        return float("nan"), float("nan")
    if len(vals) == 1:
        return float(vals[0]), 0.0
    a = np.array(vals, dtype=np.float64)
    return float(np.mean(a)), float(np.var(a))


def discover_runs(runs_root: Path) -> List[dict]:
    runs = []
    print(f"Discovering runs under: {runs_root}")
    for d in sorted(runs_root.iterdir()):
        if not d.is_dir() or d.name in EXCLUDED_RUN_DIR_NAMES:
            continue
        rv, chosen = d / "run_vars.json", d / "chosen_forget_batches.npy"
        if not rv.exists() or not chosen.exists():
            print(f"Skipping {d.name}: no run_vars.json / chosen_forget_batches.npy")
            continue
        runs.append({
            "id": d.name,
            "dir": d,
            "data_dir": json.loads(rv.read_text())["data_dir"],
            "chosen": set(np.load(chosen).tolist()),
        })
    if not runs:
        raise RuntimeError(f"No valid runs found under {runs_root}")
    data_dirs = {r["data_dir"] for r in runs}
    if len(data_dirs) != 1:
        raise ValueError(f"Runs under {runs_root} use different data_dir values: {sorted(data_dirs)}")
    print(f"Found {len(runs)} runs")
    return runs


def score_run(params, eval_step_both, forget_batches, run_id, phi_matrix, loss_matrix):
    """Per-point phi and loss of every forget point for one run's checkpoint."""
    if forget_batches[0][0].shape[0] == 1:
        # One point per batch: evaluate 128 batches at a time.
        chunk_size = 128
        for chunk_start in range(0, len(forget_batches), chunk_size):
            chunk = forget_batches[chunk_start:chunk_start + chunk_size]
            x_concat = jnp.concatenate([b[0] for b in chunk], axis=0)
            y_concat = jnp.concatenate([b[1] for b in chunk], axis=0)
            phi_values, loss_values = eval_step_both(params, x_concat, y_concat)
            phi_values, loss_values = np.array(phi_values), np.array(loss_values)
            for i in range(len(chunk)):
                k = (chunk_start + i, 0)
                phi_matrix.setdefault(k, {})[run_id] = float(phi_values[i])
                loss_matrix.setdefault(k, {})[run_id] = float(loss_values[i])
    else:
        for batch_idx, (x, y) in enumerate(forget_batches):
            phi_vals, loss_vals = eval_step_both(params, jnp.array(x), jnp.array(y))
            phi_vals, loss_vals = np.array(phi_vals), np.array(loss_vals)
            for point_idx in range(phi_vals.shape[0]):
                k = (batch_idx, point_idx)
                phi_matrix.setdefault(k, {})[run_id] = float(phi_vals[point_idx])
                loss_matrix.setdefault(k, {})[run_id] = float(loss_vals[point_idx])


def main():
    args = parse_args()
    runs_root = Path(args.runs_root)
    if args.trained_stats_only:
        model_prefix = f"checkpoint_{args.unlearn_itr}"
    else:
        model_prefix = f"unlearn_{args.unlearn_style}_{args.unlearn_itr}"

    runs = discover_runs(runs_root)

    cfg = yaml.safe_load(Path(args.config_path).read_text())
    num_classes = cfg["model"]["n_classes"]
    model = ModelFactory.create_model(model_name=cfg["model"]["name"], num_classes=num_classes)
    eval_step_both = make_eval_step_both_metrics(model, num_classes)

    data_dir = runs[0]["data_dir"]
    forget_batches = load_split(data_dir, "forget")
    n_batches = len(forget_batches)
    print(f"Loaded {n_batches} forget batches from {data_dir}")

    # (batch_idx, point_idx) -> {run_id: value}
    phi_matrix: Dict[Tuple[int, int], Dict[str, float]] = {}
    loss_matrix: Dict[Tuple[int, int], Dict[str, float]] = {}
    scored = []
    checkpointer = ocp.PyTreeCheckpointer()
    for run_idx, r in enumerate(runs):
        ckpt_path = os.path.join((r["dir"] / "ckpt").resolve(), model_prefix)
        if not os.path.exists(ckpt_path):
            print(f"[{run_idx+1}/{len(runs)}] Skipping {r['id']}: no checkpoint at {ckpt_path}")
            continue
        print(f"[{run_idx+1}/{len(runs)}] Scoring {r['id']}...", flush=True)
        state = checkpointer.restore(ckpt_path)
        params = state["params"] if isinstance(state, dict) else state.params
        score_run(params, eval_step_both, forget_batches, r["id"], phi_matrix, loss_matrix)
        scored.append(r)
        del state, params
        jax.clear_caches()
    if not scored:
        raise RuntimeError("All runs were skipped - no data to aggregate")
    if len(scored) < len(runs):
        print(f"Skipped {len(runs) - len(scored)} runs without a '{model_prefix}' checkpoint")
    runs = scored

    # Aggregate per point: selected vs remaining (mean, var) for phi and loss
    points_phi, points_loss = {}, {}
    for (batch_idx, point_idx) in sorted(set(phi_matrix) | set(loss_matrix)):
        key = f"batch_{batch_idx}_point_{point_idx}"
        phi_by_run = phi_matrix.get((batch_idx, point_idx), {})
        loss_by_run = loss_matrix.get((batch_idx, point_idx), {})
        sel_phi, rem_phi, sel_loss, rem_loss = [], [], [], []
        for r in runs:
            phi_val, loss_val = phi_by_run.get(r["id"]), loss_by_run.get(r["id"])
            is_chosen = batch_idx in r["chosen"]
            if phi_val is not None:
                (sel_phi if is_chosen else rem_phi).append(phi_val)
            if loss_val is not None:
                (sel_loss if is_chosen else rem_loss).append(loss_val)

        mu_phi_s, v_phi_s = _summarize(sel_phi)
        mu_phi_r, v_phi_r = _summarize(rem_phi)
        mu_loss_s, v_loss_s = _summarize(sel_loss)
        mu_loss_r, v_loss_r = _summarize(rem_loss)
        points_phi[key] = {
            "batch_idx": batch_idx,
            "point_idx": point_idx,
            "selected": {"count": len(sel_phi), "mean_phi": mu_phi_s, "var_phi": v_phi_s},
            "remaining": {"count": len(rem_phi), "mean_phi": mu_phi_r, "var_phi": v_phi_r},
        }
        points_loss[key] = {
            "batch_idx": batch_idx,
            "point_idx": point_idx,
            "selected": {"count": len(sel_loss), "mean_loss": mu_loss_s, "var_loss": v_loss_s},
            "remaining": {"count": len(rem_loss), "mean_loss": mu_loss_r, "var_loss": v_loss_r},
        }
    print(f"Aggregated {len(points_phi)} points")

    meta = {
        "unlearn_style": args.unlearn_style,
        "unlearn_itr": args.unlearn_itr,
        "runs_root": str(runs_root),
        "num_runs": len(runs),
        "num_batches": n_batches,
        "num_points": len(points_phi),
    }
    tag = f"trained_{args.unlearn_itr}" if args.trained_stats_only else f"{args.unlearn_style}_{args.unlearn_itr}"
    for metric, pts in (("phi", points_phi), ("loss", points_loss)):
        out = runs_root / f"forget_stats_pointwise_{metric}_{tag}.json"
        out.write_text(json.dumps({"meta": meta, "points": pts}, indent=2))
        print(f"Wrote {metric} to {out}")


if __name__ == "__main__":
    main()
