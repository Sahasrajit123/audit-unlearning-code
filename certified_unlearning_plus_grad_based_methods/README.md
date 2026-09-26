# certified_unlearning_plus_grad_based_methods

Experiments for certified machine unlearning with noisy fine-tuning. This codebase trains models, runs unlearning procedures, audits the resulting privacy lower bounds (ε, zCDP ρ, and GDP μ) to certify how well data has been forgotten, and runs a basic loss-based membership-inference attack as a complementary empirical check.

---

## Directory layout

```
certified_unlearning_plus_grad_based_methods/
├── configs/                          # YAML experiment configs
├── scripts/                          # CIFAR-100 download + data-split generation
├── src/
│   ├── data/                         # CIFAR/dataset loading and batching
│   ├── models/                       # Model definitions (TinyNet, ResNet, …)
│   ├── training/                     # Trainer, unlearn strategies
│   └── utils/                        # Config loader, data cache, DP utils
├── run_muliple_experiments_parallel.sh     # Launch N independent unlearning runs
├── run_exhaustive_combinations_parallel.sh # Exhaustive forget-combo sweep
├── run_exhaustive_resume_parallel.sh       # Resume incomplete combo runs
├── experiment_unlearning_random_forget_main.py
├── experiment_unlearning_exhaustive_combinations.py
├── experiment_unlearning_exhaustive_resume.py
├── audit_utils.py                          # Core auditing library (see below)
├── compute_eps_bounds_sampled_combos.py    # Sampled-combo ε/ρ/μ audit (see below)
├── cum_runs_eps_lab.py                     # Lower-bound math (avg/median v-tests for ε, ρ, μ)
├── evaluate_models_per_combo.py            # Per-combo model evaluation → stats JSON
├── analyze_forget_stats_pointwise_batch.py # Per-point phi/loss stats over runs
├── mia_basic.py                            # Basic loss-based MIA (Kurmanji et al., 2023)
├── make_mia_table.py                       # LaTeX tables from mia_basic.py outputs
└── requirements.txt
```

---

## Data preparation (CIFAR-100)

Experiments read pre-batched pickles from a split directory `<data_dir>/{retain,forget,val,test}/batch_XXXXX.pkl`. Generate them once per batch size:

```bash
# 1. Download CIFAR-100 and write train/test as batch pickles
#    -> data/cifar100/data_split/cifar100_no_forget/{train,test}/
python scripts/download_cifar100_batches.py

# 2a. Class-centric ("adversarial") forget split: whole classes, ~10% of train
#     -> data/cifar100/data_split/cifar100_bs_{B}/
python scripts/split_cifar100_train.py --batch_size 750

# 2b. Uniform forget split: random ~10% of train, all classes
#     -> data/cifar100/data_split/cifar100_uniform_bs_{B}_seed{S}/
python scripts/split_cifar100_train.py --batch_size 750 --split_mode uniform --seed 1

# Other batch sizes, e.g.
for B in 1 10 100 2250; do
  python scripts/split_cifar100_train.py --batch_size $B
  python scripts/split_cifar100_train.py --batch_size $B --split_mode uniform --seed 1
done
```

Both modes first hold out `--val_split` (default 0.1) of the 50,000 training images as validation, then take `--forget_fraction` (default 0.1) of the remaining 45,000 as the forget pool: 4,500 points, over 10 classes in the class-centric split (at most one class is split between forget and retain to hit the target exactly) and over all 100 classes in the uniform split. Retain is the complement. `--batch_size` is the size of each cached batch file. Forget batches are the unit each run samples from (with `forget_fraction=0.5` at experiment time, a run trains on a random half of them) and the unit the audit ranks, so each batch size needs its own split. `--seed` makes the whole split reproducible.

Pass the split directory to the experiments via `--data_dir`, e.g. `--data_dir data/cifar100/data_split/cifar100_bs_750` or `--data_dir data/cifar100/data_split/cifar100_uniform_bs_750_seed1`. Its absolute path is recorded in each run's `run_vars.json` as `data_dir`.

---

## Shell entry points

### `run_muliple_experiments_parallel.sh`

Runs N independent train+unlearn+test trials in parallel across one or more GPUs. Each trial independently samples a random forget subset and trains from scratch.

```bash
./run_muliple_experiments_parallel.sh \
    --config configs/exp_cifar100_chg7.yaml \
    --results_dir logs/my_run \
    --data_dir data/cifar100/data_split/cifar100_bs_750 \
    --start_index 1 --total_runs 50 \
    --gpus 0,1 --max_parallel 2
```

**Key flags:** `--config`, `--results_dir`, `--data_dir` (required), `--start_index`, `--total_runs`, `--gpus`, `--max_parallel`, `--forget_fraction`, `--deterministic`.

Calls `experiment_unlearning_random_forget_main.py` once per trial.

---

### `run_exhaustive_combinations_parallel.sh`

When the number of forget batches is small, iterates over **all** C(n,k) forget combinations. For each combination: one shared training run + `--num_unlearn_per_combo` independent unlearning trials.

```bash
./run_exhaustive_combinations_parallel.sh \
    --config configs/exp_cifar100_chg7.yaml \
    --results_dir logs/exhaustive \
    --data_dir data/cifar100/data_split/cifar100_bs_750 \
    --gpus 0,1 --max_parallel 2
```

**Key flags:** `--config`, `--results_dir`, `--data_dir` (required), `--num_unlearn_per_combo`, `--gpus`, `--deterministic`.

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

## Forget set passed to unlearning

`src/utils/data_cache.load_cifar_splits_with_batch_subset` samples, per run, a subset of the cached forget batches (`forget_fraction`, default 0.5) that is added to the training set; the sampled batch indices are written to `chosen_forget_batches.npy`. The `forget_loader` it returns — and which is handed to the unlearning routines — contains **only those sampled batches** (`forget_loader_scope="chosen"`, the default), and keeps the final short batch (`forget_loader_drop_remainder=False`) so every requested point receives a forget step. This matters for gradient-ascent–based methods (`unlearn_ascent_descent`), which take ascent steps on whatever the forget loader contains. `forget_loader_scope="all"` (the full candidate pool) exists only for comparison and should not be used for unlearning.

The audits below do not use this loader: they load the full forget pool themselves via `audit_utils.load_batches(cache_root, "forget")`, because the attacker has to rank every candidate batch.

The ascent–descent step functions are jitted by default. Set `unlearning.jit_unlearn_steps: false` in the config to run them eagerly (results agree up to float32 reassociation, so checkpoints are not bitwise comparable between the two modes).

---

## Auditing API

All audits certify pure `(ε, 0)` unlearning for ε; `delta` arguments on the ε path are kept only for backward compatibility and must be `0.0` (any other value raises). For approximate-DP style numbers, use the zCDP (ρ) or Gaussian-DP (μ) audits. Their `conv_delta` argument only controls a reported `(ε, δ)` *conversion* of ρ / μ, which is an estimate, **not** a lower bound on ε.

### `audit_utils.py` — batch-pointwise audits

```python
from audit_utils import (
    compute_eps_bounds_for_all_runs_batch_pointwise,   # ε   (pure DP)
    compute_rho_bounds_for_all_runs_batch_pointwise,   # ρ   (zCDP)
    compute_mu_bounds_for_all_runs_batch_pointwise,    # μ   (GDP)
    compute_all_bounds_for_all_runs_batch_pointwise,   # ε, ρ, μ for one or many k, one pass
)

result = compute_eps_bounds_for_all_runs_batch_pointwise(
    main_folder    = "logs/my_run",
    unlearn_style  = "epoch",       # "epoch" or "step"
    unlearn_itr    = 5,             # which checkpoint epoch/step
    k              = 3,             # top-k / bottom-k batches used for audit
    verbose        = False,
    confidence_level = 0.95,
    use_phi        = True,          # True = log-odds (phi); False = cross-entropy loss
    trained_stats_only = False,     # True = audit the *trained* (pre-unlearn) model
)
# result["mean"]        — ε lower bound via avg-v test (None if consistent with ε = 0)
# result["median"]      — ε lower bound via median-v test (None if consistent with ε = 0)
# result["run_ids"]     — list of run IDs that contributed
# result["failed_runs"] — list of (run_id, error_str) for skipped runs

rho = compute_rho_bounds_for_all_runs_batch_pointwise(
    main_folder="logs/my_run", unlearn_style="epoch", unlearn_itr=5, k=3,
    gamma_max=1e4,      # Rényi-order search bound for the zCDP conversion
    conv_delta=1e-3,    # δ for the reported (ε, δ) estimate only
)
# rho["mean"], rho["median"] — ρ lower bounds
# rho["comp_eps_from_rho_avg"], rho["comp_eps_from_rho_median"] — (ε, δ) estimates

mu = compute_mu_bounds_for_all_runs_batch_pointwise(
    main_folder="logs/my_run", unlearn_style="epoch", unlearn_itr=5, k=3,
    conv_delta=1e-3,
)
# mu["mean"], mu["median"] — μ lower bounds
# mu["comp_eps_from_mu_avg"], mu["comp_eps_from_mu_median"] — (ε, δ) estimates

allb = compute_all_bounds_for_all_runs_batch_pointwise(
    main_folder="logs/my_run", unlearn_style="epoch", unlearn_itr=5, k=[1, 3, 5],
)
# allb["by_k"][k] -> {"k", "T", "v_list", "run_ids", "failed_runs", "epsilon", "rho", "mu"}
```

**What it does:**
1. Discovers every run directory under `<main_folder>/test_run/` (skipping `ignored_runs/`).
2. For each run, loads the orbax checkpoint at `ckpt/unlearn_{style}_{itr}` (or `ckpt/checkpoint_{itr}` when `trained_stats_only=True`).
3. Runs a **per-point** evaluation step over every forget batch, producing one `(phi, loss)` value per data point.
4. Loads `forget_stats_pointwise_{phi|loss}_{style}_{itr}.json` (auto-generated via `analyze_forget_stats_pointwise_batch.py` if missing).
5. For each forget batch, computes a **cumulative log-likelihood ratio (LLR)**: sum of per-point `log p_selected(x) − log p_remaining(x)` under Gaussians fit to the training distribution of selected vs. remaining batches.
6. Ranks batches by LLR descending; top-k predicted as "forgotten", bottom-k as "retained".
7. Overlaps prediction against the true `chosen_forget_batches.npy` → scalar overlap score `v` per run.
8. Collects `v_list` across all runs and passes it to `epsilon_lower_bound_from_vs`, `rho_lower_bound_from_vs`, or `mu_lower_bound_from_vs`, which run the avg-v and median-v tests from `cum_runs_eps_lab`.

Steps 1–7 depend on neither k nor the bound type, so `compute_all_bounds_for_all_runs_batch_pointwise` runs them once and re-slices the rankings for each k.

**When to use:** You have run many independent unlearning trials (all sharing the same forget-batch pool) and want a tight multi-run lower bound.

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
    ci_delta       = 0.05,
    conv_delta     = 1e-3,           # δ for reported (ε, δ) estimates of ρ / μ only
    mapping_file   = None,           # defaults to {main_folder}/combo_idx_to_run_file_mapping.json
    skip_missing_trials = True,      # skip trials without a checkpoint at this step
    verbose        = False,
)
# results["avg_v_test"]["epsilon_lb"]    — avg-v test ε lower bound
# results["median_v_test"]["epsilon_lb"] — median-v test ε lower bound
# results["rho_avg_v_test"], ["rho_median_v_test"] — ρ (zCDP) lower bounds
# results["mu_avg_v_test"],  ["mu_median_v_test"]  — μ (GDP) lower bounds
# results["m2_cp_result"]               — Clopper-Pearson bound when m=2 combos
# results["overlap_sizes"], ["overlap_ratios"], ["jaccard_scores"]
# results["chosen_combos"], ["predicted_combos"]
# results["v_list"], ["T"], ["m"], ["r"]
# results["failed_runs"]
```

`delta` and `epsilon_delta` must be left at `0.0`.

**What it does:**
1. Reads `combo_idx_to_run_file_mapping.json` to discover all available forget combinations and their run directories.
2. Loads `evaluation_per_combo_{style}_{itr}.json` — per-point phi/loss statistics for each combo model (auto-generated via `evaluate_models_per_combo.py` if missing).
3. **Samples** `num_runs` combo_indices at random (with the given seed).
4. For each sampled combo, loads an eval-trial checkpoint from `eval_folders/ckpt_trial_{t}/`, computes per-point phi/loss values for all forget batches.
5. Predicts which combo_index the model was trained under using **cumulative log-likelihood** over all points: the combo whose Gaussian distribution best explains the observed values.
6. Measures overlap between the predicted and true forget-batch indices.
7. Constructs `v_list = [2 * overlap, ...]` and runs the avg-v and median-v tests for ε, ρ and μ.
8. When exactly **m=2** combos exist, additionally computes a direct **Clopper-Pearson** ε lower bound from the TPR/FPR confusion matrix.

**When to use:** You have run the exhaustive combinations sweep (all C(n,k) forget combos) and want to audit the unlearning algorithm across those combinations.

---

## Membership-inference attack

### `mia_basic.py`

The "basic MIA" of Kurmanji et al., *Towards Unbounded Machine Unlearning* (NeurIPS 2023): per run, a logistic-regression attacker is cross-validated on the audited checkpoint's per-example cross-entropy losses to separate "in" points (the run's sampled forget points) from "out" points. 50% accuracy means the attacker cannot separate the two sets. This is one number per model with no bound attached; it complements, and is not on the same scale as, the certified ε/ρ/μ lower bounds above.

```bash
# Attack the final unlearned checkpoint of every run in each sweep
python mia_basic.py --source unlearned logs/my_run

# Same attack on the pre-unlearning checkpoints (baseline for Δ)
python mia_basic.py --source trained logs/my_run

# Sanity check: shuffled in/out labels must give ~50%
python mia_basic.py --source unlearned --shuffle-labels logs/my_run
```

"Out" populations scored (chosen per sweep from the labels of its forget pool):

| Variant      | "Out" population                                   | Forget pool             |
|--------------|----------------------------------------------------|-------------------------|
| `forget_out` | forget candidates the run did **not** sample       | both (the main number)  |
| `test`       | the whole test set                                 | both                    |
| `val`        | the whole validation set                           | uniform                 |
| `test_cm`    | test examples of the forget classes                | class-centric           |
| `val_cm`     | validation examples of the forget classes          | class-centric           |
| `class_only` | *control*: forget-class vs other-class test points | class-centric           |

`forget_out` is the clean membership comparison: both sides come from the same pool and differ only in whether the run trained on them (and, because the unlearning routine only receives the sampled points, only the "in" side is ever touched by unlearning). On class-centric forget pools, `test` is class-confounded; read it against the `class_only` control, not against 50.

Attacker: `StandardScaler → LogisticRegression(max_iter=1000)`, stratified 5-fold CV, loss clipped to `[-100, 100]`. Override with `--cv {kfold,shuffle}`, `--clip LO HI`, `--no-clip`, `--scale/--no-scale`, `--max-iter`. Outputs:

- `<sweep>/<run>/mia_basic_<source>.json` — per-run attack for every variant, plus retain / forget-in / forget-out / val / test accuracy of the same checkpoint. Also used as a resume cache (`--force` re-scores).
- `<sweep>/mia_basic_<source>.json` — mean / std / median over runs.
- `--summary-out PATH` — one JSON across sweeps.

A GPU must be visible for Orbax checkpoint restore.

### `make_mia_table.py`

Pure JSON reader that turns the sweep-level `mia_basic_<source>.json` files into LaTeX tables (one per forget-split type), with Δ computed against each sweep's own `trained` result. Partial sweeps are excluded unless `--include-partial`.

```bash
python make_mia_table.py --source unlearned --out tables/mia_basic.tex logs/*
```

---

## Key dependency chain

```
run_muliple_experiments_parallel.sh
  └─ experiment_unlearning_random_forget_main.py
       └─ src/{models,training,utils,data}

run_exhaustive_combinations_parallel.sh
  └─ experiment_unlearning_exhaustive_combinations.py
       └─ src/{models,training,utils,data}

run_exhaustive_resume_parallel.sh
  └─ experiment_unlearning_exhaustive_resume.py
       └─ src/{models,training,utils,data}

audit_utils.compute_{eps,rho,mu,all}_bounds_for_all_runs_batch_pointwise
  ├─ src/models/model.ModelFactory
  ├─ cum_runs_eps_lab.compute_{avg,median}_v_test_{epsilon,rho,mu}_lb
  └─ analyze_forget_stats_pointwise_batch.py (subprocess, per-point stats)

compute_eps_bounds_sampled_combos.compute_eps_bounds_sampled_combos
  ├─ src/models/model.ModelFactory
  ├─ audit_utils (checkpoint restore, batch loading, eval steps)
  ├─ cum_runs_eps_lab.compute_{avg,median}_v_test_{epsilon,rho,mu}_lb
  └─ evaluate_models_per_combo.py          (auto-generates per-combo stats JSON)

mia_basic.py
  ├─ audit_utils (checkpoint restore, batch loading, per-sample loss)
  └─ src/models/model.ModelFactory
make_mia_table.py  (reads mia_basic_<source>.json only)
```

---

## Quick start

```bash
# 0. Prepare the data split (once)
python scripts/download_cifar100_batches.py
python scripts/split_cifar100_train.py --batch_size 750

# 1. Run 50 training+unlearning experiments
./run_muliple_experiments_parallel.sh \
    --config configs/exp_cifar100_chg7.yaml \
    --results_dir logs/my_run \
    --data_dir data/cifar100/data_split/cifar100_bs_750 \
    --total_runs 50 --gpus 0,1

# 2. Compute multi-run ε / ρ / μ lower bounds (batch-pointwise audit)
python - <<'EOF'
from audit_utils import compute_all_bounds_for_all_runs_batch_pointwise
r = compute_all_bounds_for_all_runs_batch_pointwise(
    main_folder="logs/my_run", unlearn_style="epoch", unlearn_itr=5, k=3
)
b = r["by_k"][3]
print("eps_lb mean:", b["epsilon"]["mean"], "median:", b["epsilon"]["median"])
print("rho_lb mean:", b["rho"]["mean"],     "median:", b["rho"]["median"])
print("mu_lb  mean:", b["mu"]["mean"],      "median:", b["mu"]["median"])
EOF

# 3. Basic membership-inference attack on the same runs
python mia_basic.py --source trained   logs/my_run
python mia_basic.py --source unlearned logs/my_run

# 4. Run exhaustive forget combinations
./run_exhaustive_combinations_parallel.sh \
    --config configs/exp_cifar100_chg7.yaml \
    --results_dir logs/exhaustive \
    --data_dir data/cifar100/data_split/cifar100_bs_750 \
    --gpus 0,1

# 5. Compute sampled-combo lower bounds
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
