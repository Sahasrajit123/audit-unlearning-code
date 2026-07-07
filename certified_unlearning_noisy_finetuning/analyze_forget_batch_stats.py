# analyze_forget_stats.py

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

Batch = Union[dict, tuple, list]   # adjust if your batches have a stricter type

# ------------------------------------------------------------------------------
# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Analyze forget statistics for unlearning experiments')
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
                        help='Dataset folder under data_split/. Use names starting with "cifar" for CIFAR; any other name is treated as synthetic-style (e.g., synthetic_data, synthetic_data_forget_2).')
    parser.add_argument('--single_forget_subset', action='store_true',
                        help='If set, handles single_forget_subset mode where only one subset index is stored (default: False)')
    parser.add_argument('--skip-missing', action='store_true', default=False,
                        help='If set, skip runs with missing checkpoints instead of raising an error (default: False)')
    return parser.parse_args()

# Parse arguments
args = parse_args()

# ------------------------------------------------------------------------------
# CONFIG: where your runs live
unlearn_itr = args.unlearn_itr
unlearn_style = args.unlearn_style
dataset_type = args.dataset_type
RUNS_ROOT = Path(args.runs_root)
skip_missing = getattr(args, 'skip_missing', False)
EXCLUDED_RUN_DIR_NAMES = {"train_retain", "data_split", "test_run"}

MODEL_PREFIX = "unlearn_{}_{}".format(unlearn_style, unlearn_itr)   # checkpoint prefix for post-unlearn params
# the CIFAR config file (to re-create model + num_classes)
CFG_PATH = Path(args.config_path)

# ------------------------------------------------------------------------------

# ------ helper to load pickled batches from a split folder -------------
def load_batches(root: str, split: str) -> List[Batch]:
    p = Path(root) / split
    if not p.is_dir():
        raise FileNotFoundError(f"{split!r} not found under {p.parent}")
    out = []
    for f in sorted(p.glob("batch_*.pkl")):
        with f.open("rb") as fh:
            out.append(pickle.load(fh))
    return out

# ------ helper to build eval step -------------------------------------
def _compute_loss(logits, labels, *, num_classes: int):
    """Compute cross-entropy loss (mean over batch)."""
    one_hot = jax.nn.one_hot(labels, num_classes)
    return jnp.mean(optax.softmax_cross_entropy(logits, one_hot))

def _compute_loss_per_sample(logits, labels, *, num_classes: int):
    """Compute cross-entropy loss per sample (no reduction)."""
    one_hot = jax.nn.one_hot(labels, num_classes)
    return optax.softmax_cross_entropy(logits, one_hot)

def _compute_phi(logits, labels, *, num_classes: int):
    """Compute phi (log-odds): log(p/(1-p)) where p is probability for true class."""
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    # Get log probability for the true class
    log_p = jnp.take_along_axis(log_probs, labels[:, None], axis=-1).squeeze(-1)
    p = jnp.exp(log_p)
    # Clamp to avoid numerical issues
    eps = 1e-9
    p_clamped = jnp.clip(p, eps, 1.0 - eps)
    log_one_minus_p = jnp.log(1.0 - p_clamped)
    phi = log_p - log_one_minus_p
    return phi  # Per-sample values

# ─── now wrap it, telling JAX that num_classes is static ──────────
_compute_loss = jax.jit(
    _compute_loss,
    static_argnames=("num_classes",),
)

_compute_loss_per_sample = jax.jit(
    _compute_loss_per_sample,
    static_argnames=("num_classes",),
)

_compute_phi = jax.jit(
    _compute_phi,
    static_argnames=("num_classes",),
)

def make_eval_step_both_metrics(model, num_classes: int):
    """Create evaluation function that computes both phi and loss in one pass."""
    @jax.jit
    def _eval_batches(params, x_concat: jnp.ndarray, y_concat: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Evaluate concatenated batches, return both phi and loss per-sample values."""
        logits = model.apply({"params": params}, x_concat, train=False)
        phi_values = _compute_phi(logits, y_concat, num_classes=num_classes)
        loss_values = _compute_loss_per_sample(logits, y_concat, num_classes=num_classes)
        return phi_values, loss_values
    
    return _eval_batches

# ------------------------------------------------------------------------------
# 1) Discover all runs and read their run_vars.json once ------------ 


runs = []
for d in sorted(RUNS_ROOT.iterdir()):
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
    # Use data_subfolder or dataset_type from run_vars.json
    data_subfolder = info.get("data_subfolder")
    dataset_type_run = info.get("dataset_type")
    # Load chosen data (format depends on file type)
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
        "data_subfolder": data_subfolder,
        "dataset_type": dataset_type_run,
    })

if not runs:
    raise RuntimeError("No valid runs found under " + str(RUNS_ROOT))

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

# 2) Reconstruct the model + num_classes from your config.yaml
with CFG_PATH.open("r") as f:
    cfg = yaml.safe_load(f)
cfg_dataset_name = str(cfg.get("dataset", {}).get("name", "")).lower() if isinstance(cfg, dict) else ""
model = ModelFactory.create_model(
    model_name=cfg["model"]["name"],
    num_classes=cfg["model"]["n_classes"],
)
num_classes = cfg["model"]["n_classes"]

# 3) Compute CACHE_ROOT and load forget batches
def resolve_cache_root(main_folder, dataset_type, data_subfolder):
    """Resolve cache root for CIFAR/CIFAR100 and synthetic-style layouts."""
    main_path = Path(main_folder)
    candidates = []
    if data_subfolder and str(data_subfolder).strip():
        candidates.append(main_path / "data_split" / str(data_subfolder).strip())
    if dataset_type and str(dataset_type).strip():
        candidates.append(main_path / "data_split" / str(dataset_type).strip())
    # Fallback to project-level data
    for ds_name in [data_subfolder, dataset_type]:
        if ds_name and str(ds_name).strip():
            for family in ["cifar", "cifar100"]:
                candidates.append(main_path.parent / "data" / family / "data_split" / str(ds_name).strip())
    for p in candidates:
        if p and p.exists():
            return p
    raise FileNotFoundError(f"Could not resolve cache root. Tried: {[str(p) for p in candidates]}")

cache_root = resolve_cache_root(RUNS_ROOT, dataset_type, runs[0]["data_subfolder"])
forget_batches = load_batches(str(cache_root), "forget")
N_batches = len(forget_batches)
print(f"Loaded {N_batches} forget batches from {cache_root}")
forget_iterations = [(f"batch_{i}", [forget_batches[i]], i) for i in range(N_batches)]
forget_subset_mapping = None

# 5) Build eval_step for both metrics --------------------------------------------------
eval_step_both = make_eval_step_both_metrics(model, num_classes)

# 6) OPTIMIZED: Load each checkpoint once, compute all batches -----
all_stats_phi = {}
all_stats_loss = {}

print(f"\nStarting OPTIMIZED processing: {len(runs)} runs × {len(forget_iterations)} batches...")
print(f"Computing BOTH metrics: phi (log-odds) and loss (cross-entropy)")
print("Loading checkpoints once per run (much faster than loading per batch)...\n")

# First, compute all values: values[batch_idx][run_id] = value
phi_matrix = {}  # batch_idx -> {run_id: phi_value}
loss_matrix = {}  # batch_idx -> {run_id: loss_value}
skipped_runs = []

for run_idx, r in enumerate(runs):
    print(f"[{run_idx+1}/{len(runs)}] Loading checkpoint for run {r['id']}...")
    ckpt_dir = (r["dir"] / "ckpt").resolve()

    # restore latest unlearned state
    ckpt_path = os.path.join(ckpt_dir, MODEL_PREFIX)
    if not os.path.exists(ckpt_path):
        if skip_missing:
            print(f"  ⚠️  Skipping {r['id']}: No checkpoint found at {ckpt_path}")
            skipped_runs.append(r["id"])
            continue
        else:
            raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")

    # Try to load with proper checkpoint restoration
    try:
        # First try the new Orbax format
        checkpointer = ocp.PyTreeCheckpointer()
        state = checkpointer.restore(ckpt_path)
        params = state["params"] if isinstance(state, dict) else state.params
    except Exception as e:
        print(f"  Orbax loading failed, trying Flax method: {e}")
        try:
            # Fallback to Flax checkpointing
            state = checkpoints.restore_checkpoint(ckpt_path, target=None)
            params = state["params"] if isinstance(state, dict) else state.params
        except Exception as e2:
            if skip_missing:
                print(f"  ⚠️  Skipping {r['id']}: Failed to load checkpoint: {e2}")
                skipped_runs.append(r["id"])
                continue
            else:
                raise
    
    print(f"  Checkpoint loaded. Computing both phi and loss for all {len(forget_iterations)} batches...", flush=True)
    
    # Check if we should use efficient batch processing (single-sample batches)
    use_batch_processing = False
    chunk_size = 128
    
    if forget_iterations:
        first_batch = forget_iterations[0][1][0]
        if isinstance(first_batch, (list, tuple)) and len(first_batch) == 2:
            x_first = first_batch[0]
            if hasattr(x_first, 'shape') and len(x_first.shape) > 0:
                samples_per_batch = x_first.shape[0]
                print(f"    Batch shape: {x_first.shape}", flush=True)
                if samples_per_batch == 1:
                    use_batch_processing = True
                    print(f"    Using efficient chunk processing ({chunk_size} batches at once)...", flush=True)
    
    if use_batch_processing:
        # Efficient batch processing for single-sample batches
        for chunk_start in range(0, len(forget_iterations), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(forget_iterations))
            chunk_iterations = forget_iterations[chunk_start:chunk_end]
            
            # Progress reporting
            if (chunk_start // chunk_size + 1) % 10 == 0 or chunk_start == 0:
                print(f"    Progress: {chunk_start}/{len(forget_iterations)} batches processed...", flush=True)
            
            # Extract and concatenate batches
            chunk_batches = [it[1][0] for it in chunk_iterations]
            chunk_indices = [it[2] for it in chunk_iterations]
            
            # Concatenate along batch dimension
            xs = [b[0] for b in chunk_batches]
            ys = [b[1] for b in chunk_batches]
            x_concat = jnp.concatenate(xs, axis=0)
            y_concat = jnp.concatenate(ys, axis=0)
            
            # Evaluate chunk - get both phi and loss
            phi_values, loss_values = eval_step_both(params, x_concat, y_concat)
            phi_values = np.array(phi_values)
            loss_values = np.array(loss_values)
            
            # Store results (each value corresponds to one sample/batch)
            for i, batch_idx in enumerate(chunk_indices):
                if batch_idx not in phi_matrix:
                    phi_matrix[batch_idx] = {}
                if batch_idx not in loss_matrix:
                    loss_matrix[batch_idx] = {}
                phi_matrix[batch_idx][r["id"]] = float(phi_values[i])
                loss_matrix[batch_idx][r["id"]] = float(loss_values[i])
    else:
        # Standard per-batch processing
        for batch_num, (iteration_name, batch_list, batch_idx) in enumerate(forget_iterations):
            # Progress reporting
            if (batch_num + 1) % 500 == 0 or batch_num == 0:
                print(f"    Progress: {batch_num + 1}/{len(forget_iterations)} batches processed...", flush=True)
            
            batch = batch_list[0]
            # Extract x and y from batch
            x, y = batch[0], batch[1]
            if isinstance(x, np.ndarray):
                x = jnp.array(x)
            if isinstance(y, np.ndarray):
                y = jnp.array(y)
            
            # Evaluate batch - get both phi and loss
            phi_vals, loss_vals = eval_step_both(params, x, y)
            phi_value = float(jnp.mean(phi_vals))
            loss_value = float(jnp.mean(loss_vals))
            
            # Store in matrices
            if batch_idx not in phi_matrix:
                phi_matrix[batch_idx] = {}
            if batch_idx not in loss_matrix:
                loss_matrix[batch_idx] = {}
            phi_matrix[batch_idx][r["id"]] = phi_value
            loss_matrix[batch_idx][r["id"]] = loss_value

    # Free checkpoint and params before loading next run to reduce GPU memory
    del state, params
    jax.clear_caches()
    
    print(f"  ✅ Completed run {r['id']}\n")

# Filter out skipped runs from the runs list for aggregation
if skipped_runs:
    print(f"\n⚠️  Skipped {len(skipped_runs)} runs: {skipped_runs}")
    runs = [r for r in runs if r["id"] not in skipped_runs]
    if not runs:
        raise RuntimeError("All runs were skipped - no data to aggregate")

# 7) Aggregate stats for each batch - for both metrics
print("Aggregating statistics for all batches (both phi and loss)...")
for iteration_name, batch_list, batch_idx in forget_iterations:
    phi_by_run = phi_matrix[batch_idx]
    loss_by_run = loss_matrix[batch_idx]
    
    # Aggregate stats for phi
    sel_vals_phi, rem_vals_phi = [], []
    sel_vals_loss, rem_vals_loss = [], []

    # Determine if the dataset is CIFAR-like (cifar or cifar100)
    is_cifar = False
    if dataset_type:
        is_cifar = dataset_type.lower().startswith("cifar")

    for r in runs:
        phi_val = phi_by_run[r["id"]]
        loss_val = loss_by_run[r["id"]]
        # Check if this batch was chosen
        if r["chosen_subset_idx"] is not None:
            # SINGLE_FORGET_SUBSET MODE: Check if batch's subset matches chosen subset
            if not is_cifar:
                # Synthetic not supported; placeholder for future logic
                is_chosen = False
            else:
                # For CIFAR with single_forget_subset, this shouldn't happen
                # but handle gracefully
                is_chosen = False
        else:
            # ORIGINAL MODE: Check if batch key is in chosen set
            if not is_cifar:
                # Synthetic not supported; placeholder for future logic
                is_chosen = False
            else:
                # CIFAR: chosen is an array of integer batch indices
                is_chosen = batch_idx in r["chosen"]
        if is_chosen:
            sel_vals_phi.append(phi_val)
            sel_vals_loss.append(loss_val)
        else:
            rem_vals_phi.append(phi_val)
            rem_vals_loss.append(loss_val)
    
    # Compute stats for phi
    sel_phi = np.array(sel_vals_phi); rem_phi = np.array(rem_vals_phi)
    stats_phi = {
        "selected":  {"count": int(sel_phi.size), "mean": float(sel_phi.mean()) if sel_phi.size > 0 else None, "var": float(sel_phi.var()) if sel_phi.size > 0 else None},
        "remaining": {"count": int(rem_phi.size), "mean": float(rem_phi.mean()) if rem_phi.size > 0 else None, "var": float(rem_phi.var()) if rem_phi.size > 0 else None},
    }
    all_stats_phi[iteration_name] = stats_phi
    
    # Compute stats for loss
    sel_loss = np.array(sel_vals_loss); rem_loss = np.array(rem_vals_loss)
    stats_loss = {
        "selected":  {"count": int(sel_loss.size), "mean": float(sel_loss.mean()) if sel_loss.size > 0 else None, "var": float(sel_loss.var()) if sel_loss.size > 0 else None},
        "remaining": {"count": int(rem_loss.size), "mean": float(rem_loss.mean()) if rem_loss.size > 0 else None, "var": float(rem_loss.var()) if rem_loss.size > 0 else None},
    }
    all_stats_loss[iteration_name] = stats_loss

print(f"✅ Completed analysis for all {len(forget_iterations)} batches")

# 8) Write out to JSON for both metrics -----------------------------------------------
# Write phi stats
out_phi = RUNS_ROOT / f"forget_stats_phi_{unlearn_style}_{unlearn_itr}.json"
out_phi.write_text(json.dumps(all_stats_phi, indent=2))
print(f"\nWrote phi statistics to {out_phi}")

# Write loss stats
out_loss = RUNS_ROOT / f"forget_stats_loss_{unlearn_style}_{unlearn_itr}.json"
out_loss.write_text(json.dumps(all_stats_loss, indent=2))
print(f"Wrote loss statistics to {out_loss}")

print(f"Processed {len(forget_iterations)} forget iterations: {[name for name, _, _ in forget_iterations]}")
