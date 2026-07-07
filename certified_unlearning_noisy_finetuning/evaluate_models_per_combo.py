#!/usr/bin/env python3
"""
Evaluate unlearnt models from all ckpt_trial folders (excluding eval_folders) for each combo_idx.
For each data point (batch_{batch_no}_point_{point_no}), compute mean and variance across trials
for each combo_idx model.

Output JSON structure:
{
  "batch_0_point_0": {
    "model_0": {"mean_phi": ..., "var_phi": ..., "mean_loss": ..., "var_loss": ...},
    "model_1": {"mean_phi": ..., "var_phi": ..., "mean_loss": ..., "var_loss": ...},
    ...
  },
  ...
}
"""

import json
import os
import pickle
import argparse
from pathlib import Path
from typing import List, Union, Tuple, Dict, Any, Optional, Callable
import re

import numpy as np
import yaml
from flax.training import checkpoints
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp

from src.models.model import ModelFactory

Batch = Union[dict, tuple, list]

# ------------------------------------------------------------------------------
# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate models per combo_idx from ckpt_trial folders'
    )
    parser.add_argument('--unlearn_itr', type=int, default=50,
                        help='Number of unlearning iterations (default: 50)')
    parser.add_argument('--unlearn_style', type=str, default='epoch',
                        choices=['epoch', 'step'],
                        help='Unlearning style: either "epoch" or "step" (default: epoch)')
    parser.add_argument('--runs_root', type=str, 
                        default='logs_multiple_runs_alternate_chg6_batch_750',
                        help='Root directory containing the runs')
    parser.add_argument('--dataset_type', type=str, default='cifar_750',
                        help='Dataset folder under data_split/. Use names starting with "cifar" for CIFAR.')
    parser.add_argument('--mapping_file', type=str, default=None,
                        help='Path to combo_idx to run_file mapping JSON (default: {runs_root}/combo_idx_to_run_file_mapping.json)')
    parser.add_argument('--skip-missing', action='store_true', default=False,
                        help='If set, skip runs with missing checkpoints instead of raising an error')
    return parser.parse_args()

# ------------------------------------------------------------------------------

def load_batches(root: str, split: str) -> List[Batch]:
    """Load all batches from a split directory."""
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

def restore_orbax_state(ckpt_path):
    """Restore checkpoint using Orbax."""
    # Orbax requires absolute path
    ckpt_path_abs = os.path.abspath(str(ckpt_path))
    checkpointer = ocp.PyTreeCheckpointer()
    return checkpointer.restore(ckpt_path_abs)

def find_ckpt_trial_folders(run_dir: Path, exclude_eval_folders: bool = True):
    """Find all ckpt_trial_* folders in a run directory, excluding eval_folders if requested."""
    trial_folders = []
    eval_folders_dir = run_dir / "eval_folders"
    
    for item in run_dir.iterdir():
        # Skip eval_folders directory itself
        if item.name == "eval_folders":
            continue
        
        if item.is_dir() and item.name.startswith("ckpt_trial_"):
            # Extract trial number
            match = re.match(r"ckpt_trial_(\d+)", item.name)
            if match:
                trial_num = int(match.group(1))
                trial_folders.append((trial_num, item))
    
    # Sort by trial number
    trial_folders.sort(key=lambda x: x[0])
    return [folder for _, folder in trial_folders]


def evaluate_models_per_combo(
    runs_root: Union[str, Path],
    unlearn_style: str,
    unlearn_itr: int,
    dataset_type: Optional[str] = None,
    mapping_file: Optional[Union[str, Path]] = None,
    skip_missing: bool = False,
    verbose: bool = True,
    progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
) -> Path:
    """
    Evaluate unlearnt models from all ckpt_trial folders (excluding eval_folders) for each combo_idx.
    For each data point (batch_{batch_no}_point_{point_no}), compute mean and variance across trials
    for each combo_idx model.
    
    Args:
        runs_root: Root directory containing the runs
        unlearn_style: "epoch" or "step"
        unlearn_itr: Number of unlearning iterations
        dataset_type: Dataset folder under data_split/. If None, will be inferred from run_vars.json
        mapping_file: Path to combo_idx to run_file mapping JSON. If None, uses {runs_root}/combo_idx_to_run_file_mapping.json
        skip_missing: If True, skip runs with missing checkpoints instead of raising an error
        verbose: If True, print progress messages
    
    Returns:
        Path to the output JSON file (evaluation_per_combo_{unlearn_style}_{unlearn_itr}.json)
    """
    runs_root = Path(runs_root)
    
    # Set mapping file default relative to runs_root
    if mapping_file is None:
        mapping_file = runs_root / "combo_idx_to_run_file_mapping.json"
    else:
        mapping_file = Path(mapping_file)
    
    model_prefix = f"unlearn_{unlearn_style}_{unlearn_itr}"
    
    # 1) Load mapping file
    if verbose:
        print(f"Loading mapping file: {mapping_file}")
    with open(mapping_file, 'r') as f:
        combo_mapping = json.load(f)
    
    # Convert string keys to int, skipping combos whose run_file was moved to ignored_runs
    ignored_dir = runs_root / "ignored_runs"
    combo_indices = sorted([
        int(k) for k, v in combo_mapping.items()
        if (runs_root / v["run_file"]).exists()
        and not (ignored_dir / v["run_file"]).exists()
    ])
    if verbose and len(combo_indices) < len(combo_mapping):
        print(f"Skipping {len(combo_mapping) - len(combo_indices)} ignored combo(s) (run_file in ignored_runs/)")
    if verbose:
        print(f"Found {len(combo_indices)} combo indices: {combo_indices}")
    
    # 2) Determine dataset_type if not provided
    if dataset_type is None:
        # Get dataset_type from first run's run_vars.json
        first_combo_idx = combo_indices[0]
        first_run_file = combo_mapping[str(first_combo_idx)]["run_file"]
        first_run_dir = runs_root / first_run_file
        run_vars_file = first_run_dir / "run_vars.json"
        
        if not run_vars_file.exists():
            raise FileNotFoundError(f"run_vars.json not found: {run_vars_file}")
        
        with open(run_vars_file, 'r') as f:
            run_vars = json.load(f)
        
        dataset_type = run_vars.get("data_subfolder")
        if dataset_type is None:
            raise ValueError("data_subfolder is missing in run_vars.json")
        
        if verbose:
            print(f"Inferred dataset_type: '{dataset_type}'")
    
    # Validate dataset_type against run_vars.json in each run
    if verbose:
        print(f"\nValidating dataset_type '{dataset_type}' against run_vars.json files...")
    mismatched_runs = []
    for combo_idx in combo_indices:
        combo_key = str(combo_idx)
        if combo_key not in combo_mapping:
            continue
        
        run_file = combo_mapping[combo_key]["run_file"]
        run_dir = runs_root / run_file
        run_vars_file = run_dir / "run_vars.json"
        
        if not run_vars_file.exists():
            raise FileNotFoundError(f"run_vars.json not found: {run_vars_file}")
        
        with open(run_vars_file, 'r') as f:
            run_vars = json.load(f)
        
        # Check data_subfolder field (this is what's stored in run_vars.json)
        data_subfolder = run_vars.get("data_subfolder")
        if data_subfolder is None:
            raise ValueError(f"run_vars.json in {run_file} does not contain 'data_subfolder' field")
        
        if data_subfolder != dataset_type:
            mismatched_runs.append({
                "combo_idx": combo_idx,
                "run_file": run_file,
                "expected": dataset_type,
                "found": data_subfolder
            })
    
    if mismatched_runs:
        error_msg = f"\nError: dataset_type mismatch detected!\n"
        error_msg += f"Provided dataset_type: '{dataset_type}'\n"
        error_msg += f"Mismatched runs:\n"
        for m in mismatched_runs:
            error_msg += f"  combo_idx {m['combo_idx']} ({m['run_file']}): expected '{m['expected']}', found '{m['found']}'\n"
        raise ValueError(error_msg)
    
    if verbose:
        print(f"✅ All {len(combo_indices)} runs have matching dataset_type: '{dataset_type}'")
    
    # 3) Load config from first run directory (all runs should have same config)
    if verbose:
        print(f"Loading config from run directories...")
    # Get the first run directory from mapping
    first_combo_idx = sorted([int(k) for k in combo_mapping.keys()])[0]
    first_run_file = combo_mapping[str(first_combo_idx)]["run_file"]
    first_run_dir = runs_root / first_run_file
    config_file = first_run_dir / "config.yaml"
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    if verbose:
        print(f"Loading config from: {config_file}")
    with config_file.open("r") as f:
        cfg = yaml.safe_load(f)
    model = ModelFactory.create_model(
        model_name=cfg["model"]["name"],
        num_classes=cfg["model"]["n_classes"],
    )
    num_classes = cfg["model"]["n_classes"]
    if verbose:
        print(f"Model: {cfg['model']['name']}, num_classes: {num_classes}")
    
    # 4) Load forget batches
    is_cifar = str(dataset_type).lower().startswith("cifar")
    if is_cifar:
        # Prefer explicit data_split/{dataset_type} folder under this runs_root if it exists
        cache_root_explicit = runs_root / "data_split" / dataset_type
        if cache_root_explicit.exists():
            cache_root = cache_root_explicit
        else:
            # Fallback to project-level data folder (sibling of logs_* folders)
            project_root = runs_root.parent  # e.g., /lfs/.../unlearning
            is_cifar100 = dataset_type.lower().startswith("cifar100")
            is_cifar10 = dataset_type.lower().startswith("cifar10")

            data_root_candidates = []
            if is_cifar100:
                data_root_candidates = [project_root / "data" / "cifar100"]
            elif is_cifar10:
                data_root_candidates = [
                    project_root / "data" / "cifar10",
                    project_root / "data" / "cifar",
                ]
            else:
                data_root_candidates = [project_root / "data" / "cifar"]

            cache_root = None
            for data_root in data_root_candidates:
                cache_root_split = data_root / "data_split" / dataset_type
                if cache_root_split.exists():
                    cache_root = cache_root_split
                    break
                if data_root.exists():
                    cache_root = data_root
                    break

            if cache_root is None:
                cache_root = data_root_candidates[0]

        if not cache_root.exists():
            candidate_paths = [
                runs_root / "data_split" / dataset_type,
                runs_root.parent / "data" / "cifar100" / "data_split" / dataset_type,
                runs_root.parent / "data" / "cifar10" / "data_split" / dataset_type,
                runs_root.parent / "data" / "cifar" / "data_split" / dataset_type,
                runs_root.parent / "data" / "cifar100",
                runs_root.parent / "data" / "cifar10",
                runs_root.parent / "data" / "cifar",
            ]
            raise FileNotFoundError(
                "Dataset not found in expected locations: "
                + " or ".join(str(p) for p in candidate_paths)
            )

        forget_batches = load_batches(str(cache_root), "forget")
        n_batches = len(forget_batches)
        if verbose:
            print(f"Loaded {n_batches} forget batches from {cache_root}")
    else:
        cache_root = runs_root / "data_split" / dataset_type
        forget_root = cache_root / "forget"
        forget_batches = []
        forget_dirs = sorted([d for d in forget_root.iterdir() if d.is_dir() and d.name.startswith("forget_")])
        if not forget_dirs:
            raise FileNotFoundError(f"No forget iteration directories found in {forget_root}")
        for forget_dir in forget_dirs:
            batches = load_batches(str(forget_dir), ".")
            if batches:
                forget_batches.extend(batches)
                if verbose:
                    print(f"Loaded {len(batches)} batches from {forget_dir.name}")
        n_batches = len(forget_batches)
        if verbose:
            print(f"Total forget batches loaded: {n_batches}")
    
    # 5) Build eval step
    eval_step_both = make_eval_step_both_metrics(model, num_classes)
    
    # 6) Accumulators: (batch_idx, point_idx) -> {combo_idx: [phi_values], [loss_values]}
    # For each combo_idx, we'll collect values across all trials
    point_phi_values = {}  # (batch_idx, point_idx) -> {combo_idx: [values]}
    point_loss_values = {}  # (batch_idx, point_idx) -> {combo_idx: [values]}
    
    if verbose:
        print(f"\nStarting evaluation: {len(combo_indices)} combo indices × multiple trials per combo")
        print(f"Checkpoint pattern: {model_prefix}")
        print(f"Evaluating on {n_batches} batches\n")
    
    # Precompute trial folders so we can report a total model-load target.
    combo_trial_folders: Dict[int, List[Path]] = {}
    total_trial_models = 0
    for combo_idx in combo_indices:
        combo_key = str(combo_idx)
        if combo_key not in combo_mapping:
            continue

        run_file = combo_mapping[combo_key]["run_file"]
        run_dir = runs_root / run_file
        if not run_dir.exists():
            if skip_missing:
                continue
            raise FileNotFoundError(f"Run directory not found: {run_dir}")

        trial_folders = find_ckpt_trial_folders(run_dir, exclude_eval_folders=True)
        combo_trial_folders[combo_idx] = trial_folders
        total_trial_models += len(trial_folders)

    loaded_trial_models = 0

    # 7) For each combo_idx, load models from all ckpt_trial folders
    for combo_idx in combo_indices:
        combo_key = str(combo_idx)
        if combo_key not in combo_mapping:
            if verbose:
                print(f"Warning: combo_idx {combo_idx} not found in mapping, skipping")
            continue
        
        run_file = combo_mapping[combo_key]["run_file"]
        run_dir = runs_root / run_file
        
        if not run_dir.exists():
            if skip_missing:
                if verbose:
                    print(f"Warning: Run directory {run_dir} does not exist, skipping combo_idx {combo_idx}")
                continue
            else:
                raise FileNotFoundError(f"Run directory not found: {run_dir}")
        
        if verbose:
            print(f"\n[{combo_idx}] Processing run: {run_file}")
        
        # Find all ckpt_trial folders (excluding eval_folders)
        trial_folders = combo_trial_folders.get(combo_idx)
        if trial_folders is None:
            trial_folders = find_ckpt_trial_folders(run_dir, exclude_eval_folders=True)
            combo_trial_folders[combo_idx] = trial_folders
            total_trial_models += len(trial_folders)
        if verbose:
            print(f"  Found {len(trial_folders)} ckpt_trial folders (excluding eval_folders)")
        
        if len(trial_folders) == 0:
            if skip_missing:
                if verbose:
                    print(f"  ⚠️  No ckpt_trial folders found, skipping combo_idx {combo_idx}")
                continue
            else:
                raise FileNotFoundError(f"No ckpt_trial folders found in {run_dir}")
        
        # Load models from each trial folder
        trial_params = []
        successful_trials = []
        
        for trial_folder in trial_folders:
            ckpt_path = trial_folder / model_prefix
            if not ckpt_path.exists():
                if skip_missing:
                    if verbose:
                        print(f"  ⚠️  Checkpoint not found: {ckpt_path}, skipping")
                    continue
                else:
                    raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
            
            try:
                state = restore_orbax_state(str(ckpt_path))
                params = state["params"] if isinstance(state, dict) else state.params
                trial_params.append(params)
                successful_trials.append(trial_folder.name)
                loaded_trial_models += 1
                if progress_callback is not None:
                    progress_callback(
                        "distribution",
                        loaded_trial_models,
                        total_trial_models,
                        f"combo_idx {combo_idx}",
                    )
            except Exception as e:
                if skip_missing:
                    if verbose:
                        print(f"  ⚠️  Failed to load checkpoint {ckpt_path}: {e}, skipping")
                    continue
                else:
                    raise
        
        if len(trial_params) == 0:
            if skip_missing:
                if verbose:
                    print(f"  ⚠️  No successful checkpoint loads, skipping combo_idx {combo_idx}")
                continue
            else:
                raise RuntimeError(f"No successful checkpoint loads for combo_idx {combo_idx}")
        
        if verbose:
            print(f"  Loaded {len(trial_params)} models from trials: {successful_trials}")
        
        # Evaluate each model on all data points
        if verbose:
            print(f"  Evaluating on all {n_batches} batches...", flush=True)
        
        for batch_idx, batch in enumerate(forget_batches):
            x, y = batch[0], batch[1]
            if isinstance(x, np.ndarray):
                x = jnp.array(x)
            if isinstance(y, np.ndarray):
                y = jnp.array(y)
            
            # Evaluate each trial model
            for trial_idx, params in enumerate(trial_params):
                phi_vals, loss_vals = eval_step_both(params, x, y)
                phi_vals = np.array(phi_vals)
                loss_vals = np.array(loss_vals)
                
                n_points = phi_vals.shape[0]
                
                # Store per-point values
                for point_idx in range(n_points):
                    key = (batch_idx, point_idx)
                    
                    if key not in point_phi_values:
                        point_phi_values[key] = {}
                    if key not in point_loss_values:
                        point_loss_values[key] = {}
                    
                    if combo_idx not in point_phi_values[key]:
                        point_phi_values[key][combo_idx] = []
                    if combo_idx not in point_loss_values[key]:
                        point_loss_values[key][combo_idx] = []
                    
                    point_phi_values[key][combo_idx].append(float(phi_vals[point_idx]))
                    point_loss_values[key][combo_idx].append(float(loss_vals[point_idx]))
            
            if verbose and (batch_idx + 1) % 100 == 0:
                print(f"    Progress: {batch_idx + 1}/{n_batches} batches...", flush=True)
        
        # Free memory
        del trial_params
        jax.clear_caches()
        
        if verbose:
            print(f"  ✅ Completed combo_idx {combo_idx}")
    
    # 8) Aggregate statistics: compute mean and variance for each combo_idx at each point
    if verbose:
        print(f"\nAggregating statistics...")
    results = {}
    
    for (batch_idx, point_idx) in sorted(point_phi_values.keys()):
        key = f"batch_{batch_idx}_point_{point_idx}"
        results[key] = {}
        
        # Get all combo_indices that have data for this point
        combo_indices_for_point = set(point_phi_values[(batch_idx, point_idx)].keys())
        combo_indices_for_point.update(point_loss_values.get((batch_idx, point_idx), {}).keys())
        
        for combo_idx in sorted(combo_indices_for_point):
            model_key = f"model_{combo_idx}"
            
            phi_vals = point_phi_values.get((batch_idx, point_idx), {}).get(combo_idx, [])
            loss_vals = point_loss_values.get((batch_idx, point_idx), {}).get(combo_idx, [])
            
            if len(phi_vals) == 0 and len(loss_vals) == 0:
                continue
            
            mean_phi = float(np.mean(phi_vals)) if len(phi_vals) > 0 else float('nan')
            var_phi = float(np.var(phi_vals)) if len(phi_vals) > 0 else float('nan')
            mean_loss = float(np.mean(loss_vals)) if len(loss_vals) > 0 else float('nan')
            var_loss = float(np.var(loss_vals)) if len(loss_vals) > 0 else float('nan')
            
            results[key][model_key] = {
                "mean_phi": mean_phi,
                "var_phi": var_phi,
                "mean_loss": mean_loss,
                "var_loss": var_loss,
                "num_trials": len(phi_vals) if len(phi_vals) > 0 else len(loss_vals)
            }
    
    if verbose:
        print(f"✅ Aggregated statistics for {len(results)} data points")
    
    # 9) Write output JSON
    output_file = runs_root / f"evaluation_per_combo_{unlearn_style}_{unlearn_itr}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    if verbose:
        print(f"\n✅ Wrote results to: {output_file}")
        print(f"   Total data points: {len(results)}")
        print(f"   Total combo indices evaluated: {len(combo_indices)}")
    
    return output_file


# ------------------------------------------------------------------------------
# Main execution when run as script
if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    
    # CONFIG
    unlearn_itr = args.unlearn_itr
    unlearn_style = args.unlearn_style
    dataset_type = args.dataset_type
    RUNS_ROOT = Path(args.runs_root)
    skip_missing = getattr(args, 'skip_missing', False)
    
    # Set mapping file default relative to runs_root
    if args.mapping_file is None:
        MAPPING_FILE = RUNS_ROOT / "combo_idx_to_run_file_mapping.json"
    else:
        MAPPING_FILE = Path(args.mapping_file)
    
    # Call the function
    output_file = evaluate_models_per_combo(
        runs_root=RUNS_ROOT,
        unlearn_style=unlearn_style,
        unlearn_itr=unlearn_itr,
        dataset_type=dataset_type,
        mapping_file=MAPPING_FILE,
        skip_missing=skip_missing,
        verbose=True,
    )
    
    print(f"\n✅ Done! Output file: {output_file}")
