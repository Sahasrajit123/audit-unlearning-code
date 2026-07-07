# certified_unlearning_noisy_finetuning

Experiments for certified machine unlearning with noisy fine-tuning. This codebase trains models, runs unlearning procedures, and audits the resulting epsilon lower bounds to certify how well data has been forgotten.

This code is mainly taken from the [repo](https://github.com/stair-lab/certified-unlearning-neural-networks-icml-2025).

---

## Directory layout

```
certified_unlearning_noisy_finetuning/
├── configs/                          # YAML experiment configs
├── src/
│   ├── data/                         # CIFAR/dataset loading and batching
│   ├── models/                       # Model definitions (TinyNet, ResNet, …)
│   ├── training/                     # Trainer, unlearn strategies
│   └── utils/                        # Config loader, data cache, DP utils
├── run_50_experiments_parallel.sh          # Launch N independent unlearning runs
├── run_exhaustive_combinations_parallel.sh # Exhaustive forget-combo sweep
├── run_exhaustive_resume_parallel.sh       # Resume incomplete combo runs
├── experiment_unlearning_random_forget_main.py
├── experiment_unlearning_exhaustive_combinations.py
├── experiment_unlearning_exhaustive_resume.py
├── audit_utils.py                    # Core auditing library (see below)
├── compute_eps_bounds_sampled_combos.py    # Sampled-combo epsilon audit (see below)
├── cum_runs_eps_lab.py               # Epsilon lower bound math (avg/median v-tests)
├── evaluate_models_per_combo.py      # Per-combo model evaluation → stats JSON
├── analyze_forget_batch_stats.py     # Batch-level phi/loss stats over runs
├── analyze_forget_stats_pointwise_batch.py # Per-point phi/loss stats over runs
└── requirements.txt
```

---

## Shell entry points

### `run_50_experiments_parallel.sh`

Runs N independent train+unlearn+test trials in parallel across one or more GPUs. Each trial independently samples a random forget subset and trains from scratch.

```bash
./run_50_experiments_parallel.sh \
    --config configs/exp_cifar_chg7.yaml \
    --results_dir logs/my_run \
    --start_index 1 --total_runs 50 \
    --gpus 0,1 --max_parallel 2
```

**Key flags:** `--config`, `--results_dir`, `--start_index`, `--total_runs`, `--gpus`, `--max_parallel`, `--forget_fraction`, `--deterministic`.

Calls `experiment_unlearning_random_forget_main.py` once per trial.

---

### `run_exhaustive_combinations_parallel.sh`

When the number of forget batches is small, iterates over **all** C(n,k) forget combinations. For each combination: one shared training run + `--num_unlearn_per_combo` independent unlearning trials.

```bash
./run_exhaustive_combinations_parallel.sh \
    --config configs/exp_cifar_chg7.yaml \
    --results_dir logs/exhaustive \
    --data_subfolder cifar_750 \
    --gpus 0,1 --max_parallel 2
```

**Key flags:** `--config`, `--results_dir`, `--data_subfolder` (required), `--num_unlearn_per_combo`, `--gpus`, `--deterministic`.

Calls `experiment_unlearning_exhaustive_combinations.py`.

---

### `run_exhaustive_resume_parallel.sh`

Resumes incomplete exhaustive-combo runs. Finds `_run_*.dir` sentinel files left by interrupted jobs and relaunches only the missing trials.

```bash
./run_exhaustive_resume_parallel.sh \
    --results_dir logs/exhaustive \
    --gpus 0,1,2 --max_parallel 1
```

**Key flags:** `--results_dir`, `--gpus`, `--max_parallel`, `--num_trial_workers`, `--combo_indices` (optional subset).

Calls `experiment_unlearning_exhaustive_resume.py`.

---

## Auditing API

### `audit_utils.py` — `compute_eps_bounds_for_all_runs_batch_pointwise`

```python
from audit_utils import compute_eps_bounds_for_all_runs_batch_pointwise

result = compute_eps_bounds_for_all_runs_batch_pointwise(
    main_folder    = "logs/my_run",
    unlearn_style  = "epoch",       # "epoch" or "step"
    unlearn_itr    = 5,             # which checkpoint epoch/step
    k              = 3,             # top-k / bottom-k batches used for audit
    verbose        = False,
    confidence_level = 0.95,
    delta          = 0.0,
    use_phi        = True,          # True = log-odds (phi); False = cross-entropy loss
    trained_stats_only = False,     # True = audit the *trained* (pre-unlearn) model
)
# result["mean"]      — epsilon lower bound via avg-v test (None if infeasible)
# result["median"]    — epsilon lower bound via median-v test (None if infeasible)
# result["run_ids"]   — list of run IDs that contributed
# result["failed_runs"] — list of (run_id, error_str) for skipped runs
```

**What it does:**
1. Discovers every run directory under `<main_folder>/test_run/` (skipping `ignored_runs/`).
2. For each run, loads the orbax checkpoint at `ckpt/unlearn_{style}_{itr}` (or `ckpt/checkpoint_{itr}` when `trained_stats_only=True`).
3. Runs a **per-point** evaluation step over every forget batch, producing one `(phi, loss)` value per data point.
4. Loads `forget_stats_pointwise_{phi|loss}_{style}_{itr}.json` (auto-generated via `analyze_forget_stats_pointwise_batch.py` if missing).
5. For each forget batch, computes a **cumulative log-likelihood ratio (LLR)**: sum of per-point `log p_selected(x) − log p_remaining(x)` under Gaussian distributions fit to the training distribution of selected vs. remaining batches.
6. Ranks batches by LLR descending; top-k predicted as "forgotten", bottom-k as "retained".
7. Overlaps prediction against the true `chosen_forget_batches.npy` → scalar overlap score `v` per run.
8. Collects `v_list` across all runs, then calls `epsilon_lower_bound_from_vs(v_list, k, N, ...)` which invokes both the avg-v and median-v statistical tests from `cum_runs_eps_lab`.

**When to use:** You have run many independent unlearning trials (all sharing the same forget-batch pool) and want a tight multi-run epsilon lower bound.

---

### `compute_eps_bounds_sampled_combos.py` — `compute_eps_bounds_sampled_combos`

```python
from compute_eps_bounds_sampled_combos import compute_eps_bounds_sampled_combos

results = compute_eps_bounds_sampled_combos(
    main_folder    = "logs/exhaustive",
    unlearn_style  = "epoch",
    unlearn_itr    = 5,
    num_runs       = 50,
    metric         = "phi",          # "phi" or "loss"
    sampling_seed  = 123,
    confidence_level = 0.95,
    delta          = 0.0,
    epsilon_delta  = 1e-8,
    ci_delta       = 0.05,
    mapping_file   = None,           # defaults to {main_folder}/combo_idx_to_run_file_mapping.json
    verbose        = False,
)
# results["avg_v_test"]["epsilon_lb"]    — avg-v test epsilon lower bound
# results["median_v_test"]["epsilon_lb"] — median-v test epsilon lower bound
# results["m2_cp_result"]               — Clopper-Pearson bound when m=2 combos
# results["overlap_sizes"], ["overlap_ratios"], ["jaccard_scores"]
# results["chosen_combos"], ["predicted_combos"]
# results["v_list"], ["T"], ["m"], ["r"]
# results["failed_runs"]
```

**What it does:**
1. Reads `combo_idx_to_run_file_mapping.json` to discover all available forget combinations (combo_indices) and their run directories.
2. Loads `evaluation_per_combo_{style}_{itr}.json` — per-point phi/loss statistics for each combo model (auto-generated via `evaluate_models_per_combo.py` if missing).
3. **Samples** `num_runs` combo_indices at random (with the given seed).
4. For each sampled combo, loads an eval-trial checkpoint from `eval_folders/ckpt_trial_{t}/`, computes per-point phi/loss values for all forget batches.
5. Predicts which combo_index the model was trained under using **cumulative log-likelihood** over all points: the combo whose Gaussian distribution best explains the observed values.
6. Measures overlap between the predicted and true forget-batch indices.
7. Constructs `v_list = [2 * overlap, ...]` and calls the avg-v and median-v epsilon tests.
8. When exactly **m=2** combos exist, additionally computes a direct **Clopper-Pearson** epsilon lower bound from the TPR/FPR confusion matrix.

**When to use:** You have run the exhaustive combinations sweep (all C(n,k) forget combos) and want to audit the unlearning algorithm's certified epsilon across those combinations.

---

## Key dependency chain

```
run_50_experiments_parallel.sh
  └─ experiment_unlearning_random_forget_main.py
       └─ src/{models,training,utils,data}

run_exhaustive_combinations_parallel.sh
  └─ experiment_unlearning_exhaustive_combinations.py
       └─ src/{models,training,utils,data}

run_exhaustive_resume_parallel.sh
  └─ experiment_unlearning_exhaustive_resume.py
       └─ src/{models,training,utils,data}

audit_utils.compute_eps_bounds_for_all_runs_batch_pointwise
  ├─ src/models/model.ModelFactory
  ├─ cum_runs_eps_lab.{compute_avg_v_test_epsilon_lb, compute_median_v_test_epsilon_lb}
  ├─ analyze_forget_batch_stats.py         (subprocess, batch-level stats)
  └─ analyze_forget_stats_pointwise_batch.py (subprocess, per-point stats)

compute_eps_bounds_sampled_combos.compute_eps_bounds_sampled_combos
  ├─ src/models/model.ModelFactory
  ├─ audit_utils (checkpoint restore, batch loading, eval steps)
  ├─ cum_runs_eps_lab.{compute_avg_v_test_epsilon_lb, compute_median_v_test_epsilon_lb}
  └─ evaluate_models_per_combo.py          (auto-generates per-combo stats JSON)
```

---

## Quick start

```bash
# 1. Run 50 training+unlearning experiments
./run_50_experiments_parallel.sh \
    --config configs/exp_cifar_chg7.yaml \
    --results_dir logs/my_run \
    --total_runs 50 --gpus 0,1

# 2. Compute multi-run epsilon lower bound (batch-pointwise audit)
python - <<'EOF'
from audit_utils import compute_eps_bounds_for_all_runs_batch_pointwise
r = compute_eps_bounds_for_all_runs_batch_pointwise(
    main_folder="logs/my_run", unlearn_style="epoch", unlearn_itr=5, k=3
)
print("eps_lb mean:", r["mean"], "median:", r["median"])
EOF

# 3. Run exhaustive forget combinations
./run_exhaustive_combinations_parallel.sh \
    --config configs/exp_cifar_chg7.yaml \
    --results_dir logs/exhaustive \
    --data_subfolder cifar_750 \
    --gpus 0,1

# 4. Compute sampled-combo epsilon lower bound
python - <<'EOF'
from compute_eps_bounds_sampled_combos import compute_eps_bounds_sampled_combos
r = compute_eps_bounds_sampled_combos(
    main_folder="logs/exhaustive", unlearn_style="epoch", unlearn_itr=5,
    num_runs=50, metric="phi"
)
print("eps_lb avg:", r["avg_v_test"]["epsilon_lb"])
print("eps_lb median:", r["median_v_test"]["epsilon_lb"])
EOF
```
