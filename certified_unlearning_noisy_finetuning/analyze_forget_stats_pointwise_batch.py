# analyze_forget_stats_pointwise_batch.py
"""
Compute per-point phi (log-odds) and loss statistics for every point in each forget batch.

For each point in each batch, keys are batch_{batch_no}_point_{point_no}. For each point we
compute mean and variance across runs where that batch was SELECTED (in forget set) vs REMAINING
(not in forget set). Output includes both phi and loss: mean_phi, var_phi, mean_loss, var_loss
for selected and remaining.

Checkpoint loading, run discovery, and data layout match analyze_forget_stats_batch_1.py.
Per-point aggregation and output structure follow compute_forget_pointwise_batch_stats.py.
"""

import json, os, pickle, argparse
from pathlib import Path
from typing import List, Union, Tuple, Dict, Any
from itertools import islice

import numpy as np
import yaml
from flax.training import checkpoints
import jax, jax.numpy as jnp, optax
import orbax.checkpoint as ocp

from src.models.model import ModelFactory

Batch = Union[dict, tuple, list]

# ------------------------------------------------------------------------------
# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(
        description='Analyze forget statistics per POINT in each batch (batch_{batch_no}_point_{point_no})'
    )
    parser.add_argument('--unlearn_itr', type=int, default=50,
                        help='Number of unlearning iterations (default: 50)')
    parser.add_argument('--unlearn_style', type=str, default='epoch',
                        choices=['epoch', 'step'],
                        help='Unlearning style: either "epoch" or "step" (default: epoch)')
    parser.add_argument('--runs_root', type=str, default='logs_multiple_runs_alternate_chg_batch_1',
                        help='Root directory containing the runs (default: logs_multiple_runs_alternate_chg_batch_1)')
    parser.add_argument('--config_path', type=str, default='configs/exp_cifar.yaml',
                        help='Path to the config file (default: configs/exp_cifar.yaml)')
    parser.add_argument('--dataset_type', type=str, default='cifar_1',
                        help='Dataset folder under data_split/. Use names starting with "cifar" for CIFAR; any other name is treated as synthetic-style.')
    parser.add_argument('--single_forget_subset', action='store_true',
                        help='If set, handles single_forget_subset mode (default: False). Chosen subset is read from chosen_forget_batches.json when mode is single_forget_subset.')
    parser.add_argument('--skip-missing', action='store_true', default=True,
                        help='Skip runs with missing checkpoints instead of raising an error (default: True)')
    parser.add_argument('--trained_stats_only', action='store_true', default=False,
                        help='If set, loads checkpoints named checkpoint_{unlearn_itr} instead of unlearn_{unlearn_style}_{unlearn_itr}. This is for analyzing training stats instead of unlearning stats.')
    return parser.parse_args()

# Parse arguments
args = parse_args()

# ------------------------------------------------------------------------------
# CONFIG
unlearn_itr = args.unlearn_itr
unlearn_style = args.unlearn_style
dataset_type = args.dataset_type
RUNS_ROOT = Path(args.runs_root)
skip_missing = getattr(args, 'skip_missing', True)

# Stats generation should be built from the main run folders only.
# Evaluation can still use test_run elsewhere in audit_utils.
RUN_DISCOVERY_ROOT = RUNS_ROOT
EXCLUDED_RUN_DIR_NAMES = {"train_retain", "data_split", "test_run"}
if args.trained_stats_only:
    print("⚠️  Running in TRAINED STATS ONLY mode: loading checkpoints named checkpoint_{unlearn_itr} instead of unlearn_{unlearn_style}_{unlearn_itr}. This is for analyzing training stats instead of unlearning stats.")
    MODEL_PREFIX = "checkpoint_{}".format(unlearn_itr)
else:
    MODEL_PREFIX = "unlearn_{}_{}".format(unlearn_style, unlearn_itr)
CFG_PATH = Path(args.config_path)

# ------------------------------------------------------------------------------

def load_batches(root: str, split: str) -> List[Batch]:
    p = Path(root) / split
    if not p.is_dir():
        raise FileNotFoundError(f"{split!r} not found under {p.parent}")
    out = []
    for f in sorted(p.glob("batch_*.pkl")):
        with f.open("rb") as fh:
            out.append(pickle.load(fh))
    return out


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
    phi = log_p - log_one_minus_p
    return phi


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


# ------------------------------------------------------------------------------
# 1) Discover all runs and read run_vars.json
file_bs_list = []
runs = []

print(f"Discovering runs under: {RUN_DISCOVERY_ROOT}")
for d in sorted(RUN_DISCOVERY_ROOT.iterdir()):
    if not d.is_dir():
        continue
    if d.name in EXCLUDED_RUN_DIR_NAMES:
        print(f"Skipping {d.name}: excluded directory")
        continue
    rv = d / "run_vars.json"
    if not rv.exists():
        print(f"Skipping {d.name}: no run_vars.json")
        continue
    info = json.loads(rv.read_text())

    chosen_json_file = d / "chosen_forget_batches.json"
    chosen_npy_file = d / "chosen_forget_batches.npy"
    chosen_data = None
    chosen_subset_idx = None

    if chosen_json_file.exists():
        with open(chosen_json_file, "r") as f:
            loaded_data = json.load(f)
        if isinstance(loaded_data, dict) and loaded_data.get("mode") == "single_forget_subset":
            chosen_subset_idx = loaded_data.get("chosen_subset_idx")
            chosen_data = None
        else:
            chosen_data = set(loaded_data) if isinstance(loaded_data, list) else set()
    elif chosen_npy_file.exists():
        chosen_data = set(np.load(chosen_npy_file).tolist())
    else:
        print(f"Warning: No chosen data found for {d.name}")
        chosen_data = set()

    runs.append({
        "id": d.name,
        "dir": d,
        "chosen": chosen_data,
        "chosen_subset_idx": chosen_subset_idx,
    })

if not runs:
    raise RuntimeError("No valid runs found under " + str(RUN_DISCOVERY_ROOT))

single_subset_runs = sum(1 for r in runs if r["chosen_subset_idx"] is not None)
multi_subset_runs = len(runs) - single_subset_runs
print(f"\nFound {len(runs)} runs:")
print(f"  - {single_subset_runs} using single_forget_subset mode")
print(f"  - {multi_subset_runs} using original multi-subset mode")
if single_subset_runs > 0:
    print(f"\nChosen subsets in single_forget_subset mode:")
    for r in runs:
        if r["chosen_subset_idx"] is not None:
            print(f"  {r['id']}: subset {r['chosen_subset_idx']}")
print()

# 2) Reconstruct model + num_classes from config
with CFG_PATH.open("r") as f:
    cfg = yaml.safe_load(f)
cfg_dataset_name = str(cfg.get("dataset", {}).get("name", "")).lower() if isinstance(cfg, dict) else ""
model = ModelFactory.create_model(
    model_name=cfg["model"]["name"],
    num_classes=cfg["model"]["n_classes"],
)
num_classes = cfg["model"]["n_classes"]

# 3) Single batch_size across all runs


# 4) CACHE_ROOT and forget batches
is_cifar = str(dataset_type).lower().startswith("cifar")
forget_subset_mapping = []

if is_cifar:
    # Strict CIFAR path resolution:
    # 1) RUNS_ROOT/data_split/<dataset_name>
    # 2) RUNS_ROOT.parent/data/<cifar|cifar100>/data_split/<dataset_name>
    # No other fallbacks.
    # Use dataset_type directly as the dataset folder name
    dataset_name = str(dataset_type)
    is_cifar100 = (
        str(dataset_type).lower().startswith("cifar100")
        or cfg_dataset_name == "cifar100"
        or "cifar100" in str(RUNS_ROOT).lower()
    )
    family_dir = "cifar100" if is_cifar100 else "cifar"
    candidate_paths = [
        RUNS_ROOT / "data_split" / dataset_name,
        RUNS_ROOT.parent / "data" / family_dir / "data_split" / dataset_name,
    ]

    CACHE_ROOT = next((p for p in candidate_paths if p.exists()), None)
    if CACHE_ROOT is None:
        raise FileNotFoundError(
            "Dataset not found for strict CIFAR lookup. Tried: "
            + ", ".join(str(p) for p in candidate_paths)
        )

    forget_batches = load_batches(str(CACHE_ROOT), "forget")
    N_batches = len(forget_batches)
    print(f"Loaded {N_batches} forget batches from {CACHE_ROOT}")
    forget_iterations = [(f"batch_{i}", [forget_batches[i]], i) for i in range(N_batches)]
else:
    raise ValueError(
        "Synthetic data generation/analysis is not supported in this script. "
        "Use CIFAR-10 or CIFAR-100 dataset_type only."
    )

# 5) Eval step (per-sample phi and loss)
eval_step_both = make_eval_step_both_metrics(model, num_classes)

# 6) Accumulators: (batch_idx, point_idx) -> {run_id: value}
phi_matrix: Dict[Tuple[int, int], Dict[str, float]] = {}
loss_matrix: Dict[Tuple[int, int], Dict[str, float]] = {}
skipped_runs = []

print(f"\nStarting POINTWISE processing: {len(runs)} runs × {len(forget_iterations)} batches (per-point phi & loss)")
print("Loading checkpoints once per run...\n")

for run_idx, r in enumerate(runs):
    print(f"[{run_idx+1}/{len(runs)}] Loading checkpoint for run {r['id']}...")
    ckpt_dir = (r["dir"] / "ckpt").resolve()
    ckpt_path = os.path.join(ckpt_dir, MODEL_PREFIX)
    if not os.path.exists(ckpt_path):
        if skip_missing:
            print(f"  ⚠️  Skipping {r['id']}: No checkpoint found at {ckpt_path}")
            skipped_runs.append(r["id"])
            continue
        raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")

    try:
        checkpointer = ocp.PyTreeCheckpointer()
        state = checkpointer.restore(ckpt_path)
        params = state["params"] if isinstance(state, dict) else state.params
    except Exception as e:
        print(f"  Orbax loading failed, trying Flax: {e}")
        try:
            state = checkpoints.restore_checkpoint(ckpt_path, target=None)
            params = state["params"] if isinstance(state, dict) else state.params
        except Exception as e2:
            if skip_missing:
                print(f"  ⚠️  Skipping {r['id']}: Failed to load checkpoint: {e2}")
                skipped_runs.append(r["id"])
                continue
            raise

    print(f"  Checkpoint loaded. Computing per-point phi and loss for all batches...", flush=True)

    use_batch_processing = False
    chunk_size = 128
    if forget_iterations:
        first_batch = forget_iterations[0][1][0]
        if isinstance(first_batch, (list, tuple)) and len(first_batch) == 2:
            x_first = first_batch[0]
            if hasattr(x_first, 'shape') and len(x_first.shape) > 0:
                samples_per_batch = x_first.shape[0]
                if samples_per_batch == 1:
                    use_batch_processing = True
                    print(f"    Using chunk processing ({chunk_size} batches at once, 1 point/batch)...", flush=True)

    if use_batch_processing:
        for chunk_start in range(0, len(forget_iterations), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(forget_iterations))
            chunk_iterations = forget_iterations[chunk_start:chunk_end]
            if (chunk_start // chunk_size + 1) % 10 == 0 or chunk_start == 0:
                print(f"    Progress: {chunk_start}/{len(forget_iterations)} batches...", flush=True)

            chunk_batches = [it[1][0] for it in chunk_iterations]
            chunk_indices = [it[2] for it in chunk_iterations]
            xs = [b[0] for b in chunk_batches]
            ys = [b[1] for b in chunk_batches]
            x_concat = jnp.concatenate(xs, axis=0)
            y_concat = jnp.concatenate(ys, axis=0)

            phi_values, loss_values = eval_step_both(params, x_concat, y_concat)
            phi_values = np.array(phi_values)
            loss_values = np.array(loss_values)

            for i, batch_idx in enumerate(chunk_indices):
                k = (batch_idx, 0)
                if k not in phi_matrix:
                    phi_matrix[k] = {}
                    loss_matrix[k] = {}
                phi_matrix[k][r["id"]] = float(phi_values[i])
                loss_matrix[k][r["id"]] = float(loss_values[i])
    else:
        for batch_num, (iteration_name, batch_list, batch_idx) in enumerate(forget_iterations):
            if (batch_num + 1) % 500 == 0 or batch_num == 0:
                print(f"    Progress: {batch_num + 1}/{len(forget_iterations)} batches...", flush=True)

            batch = batch_list[0]
            x, y = batch[0], batch[1]
            if isinstance(x, np.ndarray):
                x = jnp.array(x)
            if isinstance(y, np.ndarray):
                y = jnp.array(y)

            phi_vals, loss_vals = eval_step_both(params, x, y)
            phi_vals = np.array(phi_vals)
            loss_vals = np.array(loss_vals)
            n = phi_vals.shape[0]

            for point_idx in range(n):
                k = (batch_idx, point_idx)
                if k not in phi_matrix:
                    phi_matrix[k] = {}
                    loss_matrix[k] = {}
                phi_matrix[k][r["id"]] = float(phi_vals[point_idx])
                loss_matrix[k][r["id"]] = float(loss_vals[point_idx])

    del state, params
    jax.clear_caches()
    print(f"  ✅ Completed run {r['id']}\n")

if skipped_runs:
    print(f"\n⚠️  Skipped {len(skipped_runs)} runs: {skipped_runs}")
    runs = [r for r in runs if r["id"] not in skipped_runs]
    if not runs:
        raise RuntimeError("All runs were skipped - no data to aggregate")

# 7) Aggregate per-point: selected vs remaining (mean, var) for phi and loss
all_keys = sorted(set(phi_matrix.keys()) | set(loss_matrix.keys()))
points_phi = {}
points_loss = {}

print("Aggregating per-point statistics (selected vs remaining)...")

for (batch_idx, point_idx) in all_keys:
    key = f"batch_{batch_idx}_point_{point_idx}"
    phi_by_run = phi_matrix.get((batch_idx, point_idx), {})
    loss_by_run = loss_matrix.get((batch_idx, point_idx), {})

    sel_phi, rem_phi = [], []
    sel_loss, rem_loss = [], []

    for r in runs:
        phi_val = phi_by_run.get(r["id"])
        loss_val = loss_by_run.get(r["id"])
        if phi_val is None and loss_val is None:
            continue

        if r["chosen_subset_idx"] is not None:
            if not is_cifar:
                subset_name, local_idx = forget_subset_mapping[batch_idx]
                subset_num = int(subset_name.split('_')[1])
                is_chosen = (subset_num == r["chosen_subset_idx"])
            else:
                is_chosen = False
        else:
            if not is_cifar:
                subset_name, local_idx = forget_subset_mapping[batch_idx]
                batch_key = f"{subset_name}_{local_idx}"
                is_chosen = batch_key in r["chosen"]
            else:
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

print(f"✅ Aggregated {len(points_phi)} points")

# 8) Write JSON (phi and loss) with meta
meta = {
    "unlearn_style": unlearn_style,
    "unlearn_itr": unlearn_itr,
    "runs_root": str(RUNS_ROOT),
    "num_runs": len(runs),
    "num_batches": N_batches,
    "num_points": len(points_phi),
}

if args.trained_stats_only:
    ##meta["note"] = "This dataset contains training stats only (checkpoints named checkpoint_{unlearn_itr}), not unlearning stats. Set trained_stats_only=False to compute unlearning stats instead."
    out_phi = RUNS_ROOT / f"forget_stats_pointwise_phi_trained_{unlearn_itr}.json"
    out_loss = RUNS_ROOT / f"forget_stats_pointwise_loss_trained_{unlearn_itr}.json"
else:
    out_phi = RUNS_ROOT / f"forget_stats_pointwise_phi_{unlearn_style}_{unlearn_itr}.json"
    out_loss = RUNS_ROOT / f"forget_stats_pointwise_loss_{unlearn_style}_{unlearn_itr}.json"

for path, pts in [(out_phi, points_phi), (out_loss, points_loss)]:
    obj = {"meta": meta, "points": pts}
    path.write_text(json.dumps(obj, indent=2))

print(f"\nWrote phi to {out_phi}")
print(f"Wrote loss to {out_loss}")
print(f"Points: batch_*_point_* keys for {len(points_phi)} (batch, point) pairs.")
