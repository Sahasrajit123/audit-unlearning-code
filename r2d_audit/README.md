# Rewind-To-Delete (R2D): Certified Machine Unlearning for Nonconvex Functions

Code and scripts for the combination-forget unlearning experiments from our ICLR submission.

This folder is one of seven code folders in the submission. It depends on one of the others, `certified_unlearning_plus_grad_based_methods/`, which generates the CIFAR-100 data splits (uniform or adversarial forget batches) that every pipeline here reads. See [Data Preparation](#data-preparation).

---

## Overview

R2D provides certified machine unlearning guarantees for nonconvex models. Given a trained model and a set of data points to forget, R2D unlearns by continuing gradient descent on the retain set, then adds Gaussian noise to the weights. The noise is calibrated to the global sensitivity from Theorem 3.1, and can target either **(ε, δ)-DP** or **ρ-zCDP**. The unlearning is then audited with a hypothesis-testing lower bound on the privacy parameter: an **ε lower bound (at δ = 0)** and a **ρ-zCDP lower bound**.

This folder covers the **combination forget** experimental pipeline:
1. Train a model on a dataset split into retain + multiple forget batches.
2. For every combination of `N/2` forget batches (out of `N` total), train from scratch then unlearn, in parallel across GPUs.
3. Evaluate each trained/unlearnt model checkpoint (with noise calibrated to ε or ρ) and compute per-forget-point φ statistics.
4. Aggregate across combinations to compute ε and ρ lower bounds, then produce plots and summary tables.

---

## Repository Structure

```
r2d_audit/
├── run_all_combinations_forget_unlearning.sh   # Master launcher (step 1)
├── random_forget_runner_all_combinations.py    # Per-combination train+unlearn runner
├── main.py                                     # Core training and unlearning engine
│
├── models.py                                   # TinyNet model definitions
├── datasets.py                                 # Dataset loading (CIFAR-10, CIFAR-100)
├── r2d.py                                      # Core R2D math: h(), noise calibration, bounds
├── forget_phi_noisy_loader.py                  # Sensitivity GS, σ calibration ((ε,δ) or ρ), noisy reloads
├── cum_runs_eps_lab.py                         # Audit: ε lower bounds (δ = 0) and ρ-zCDP lower bounds
├── collect_rho_audit.py                        # Step 4c: collect audit_summary blocks into a CSV
├── mia.py                                      # Membership inference attack utilities
├── utils.py                                    # Shared training utilities
├── logger.py                                   # Logging helpers
│
└── evaluation_scripts/
    ├── evaluate_combination_models.py          # Step 2: evaluate checkpoints, output stats JSON
    ├── evaluate_grouped_predictions.py         # Helper: model loading + grouped LLR computation
    ├── evaluate_prediction_cumulative_model.py # Step 3: predict runs, compute ε and ρ lower bounds
    └── combine_prediction_cumulative_plots.py  # Step 4b: aggregate JSONs and plot ε lower bounds
```

---

## Requirements

```
torch >= 2.0
numpy
scipy
matplotlib
tqdm
```

A conda environment file (`r2d.yml`) is available in the parent repository.

---

## Data Preparation

The data splits are **not** generated in this folder. They come from the sibling folder `certified_unlearning_plus_grad_based_methods/`, which splits CIFAR-100 into retain and forget sets and writes each split out as fixed-size batches. Run these commands from inside that folder:

```bash
cd ../certified_unlearning_plus_grad_based_methods

# 1. Download CIFAR-100 and write train/test as batch pickles
#    -> data/cifar100/data_split/cifar100_no_forget/{train,test}/
python scripts/download_cifar100_batches.py

# 2a. Adversarial (class-centric) forget split: whole classes, ~10% of train
#     -> data/cifar100/data_split/cifar100_bs_<B>/
python scripts/split_cifar100_train.py --batch_size 750

# 2b. Uniform forget split: random ~10% of train, spanning all classes
#     -> data/cifar100/data_split/cifar100_uniform_bs_<B>_seed<S>/
python scripts/split_cifar100_train.py --batch_size 750 --split_mode uniform --seed 1
```

Both modes first hold out `--val_split` (default 0.1) of the 50,000 training images as validation. They then take `--forget_fraction` (default 0.1) of the remaining 45,000 as the forget pool, i.e. 4,500 points:
- **Adversarial:** the forget pool covers 10 of the 100 classes. At most one class is split between forget and retain, to hit the target size exactly.
- **Uniform:** the forget pool is drawn at random from all 100 classes.

`--batch_size` sets the number of points per forget batch. With `--batch_size 750` the forget pool is 6 batches, so step 1 below runs every combination of 3 out of 6 batches. `--seed` makes the split reproducible. See the README in `certified_unlearning_plus_grad_based_methods/` for the full set of options.

Pass the resulting split directory as `<DATAROOT>` (the `--data-dir` option) to every script here. For example, `.../data/cifar100/data_split/cifar100_uniform_bs_750_seed1` for the uniform split or `.../data/cifar100/data_split/cifar100_bs_750` for the adversarial split.

## Data Format

Scripts expect a pre-split dataset directory with the following layout, as produced above:

```
<DATAROOT>/
├── train/
├── val/
├── test/
├── retain/
└── forget/
    ├── batch_00000.pkl
    ├── batch_00001.pkl
    └── ...
```

Each `batch_XXXXX.pkl` holds an `(images, labels)` tuple of NumPy arrays. The batch index in the filename is used as the identity of every point in that forget batch. The number of forget batches is detected automatically. The combination size is set to `NUM_FORGET_BATCHES / 2`.

---

## Noise Calibration

Both calibrations use the same global sensitivity from Theorem 3.1,

```
GS = 2 · num_forget · G · h(K) / (L · n)
```

computed by `compute_GS_from_log_new_structure` in `forget_phi_noisy_loader.py` (L = Lipschitz constant, G = max gradient norm, both read from the training logs). What differs is how σ is obtained from GS:

| Accountant | Flag | σ | Notes |
|---|---|---|---|
| (ε, δ)-DP | `--epsilon E --delta D` | numerical inversion of the analytic Gaussian condition `Φ(Δ/2σ − εσ/Δ) − e^ε Φ(−Δ/2σ − εσ/Δ) ≤ δ` (`compute_sigma_general`) | valid for any ε > 0 |
| ρ-zCDP | `--rho R` | `σ = GS / sqrt(2ρ)` (`compute_sigma_zcdp`) | exact, no δ (Bun & Steinke 2016, Prop. 1.6) |

- `--rho` takes precedence: when it is given, `--epsilon` and `--delta` are ignored for noise generation.
- `inf` (for either ε or ρ) disables noise and does a single, noise-free evaluation.
- Output files are tagged by the accountant so the two never collide: `eps<E>_delta<D>` vs `rho<R>` (e.g. `rho0p5`, `rhoinf`). Existing `eps…_delta…` files keep their original names.

---

## Audit (Lower Bounds)

`cum_runs_eps_lab.py` computes two lower bounds from the same observed statistic (the per-run overlap scores `v_list`, one per sampled run). Each bound holds with probability `1 − ci_delta`. Each uses two tests: **avg-v** (a Chernoff bound on the mean) and **median-v**.

**ε lower bound, δ = 0 only.** The audit uses the reduction *(ε, 0)-certified unlearning ⇒ the audit mechanism M is (2ε, 0)-LDP*, with pointwise bound `Pr[M(S) = ŝ] ≤ e^ε / (e^ε + Z − 1)`, `Z = C(m, ⌊m/2⌋)`. This does not usefully extend to δ > 0: the triangle-inequality step degrades to `(2ε, (1+e^ε)δ)`. For that reason the ε audit functions no longer take a `delta` argument.
- `epsilon_lb` is the **certified-unlearning ε**, already halved from the LDP bound.
- `epsilon_lb_ldp` is the raw (un-halved) LDP parameter of M.
- `None` means the observation is consistent with ε = 0.

**ρ-zCDP lower bound.** The audit combines the normalized pointwise distribution `π(u)` with the Rényi-DP order-γ ε implied by ρ-zCDP, and minimizes over γ ∈ (1, `gamma_max`]. The result `rho_lb` is the largest ρ consistent with the observed statistic.
- `eps_estimate` is the forward conversion of `rho_lb` to (ε, `conv_delta`)-DP. It uses the tighter of Balle et al. 2020 Thm 21 and Bun & Steinke 2016 Prop 1.3. **It is not a lower bound on ε**, and it is not on the same δ axis as `epsilon_lb` (which is at δ = 0).

| Function | Output |
|---|---|
| `compute_avg_v_test_epsilon_lb`, `compute_median_v_test_epsilon_lb` | `epsilon_lb`, `epsilon_lb_ldp` |
| `compute_avg_v_test_rho_lb`, `compute_median_v_test_rho_lb` | `rho_lb`, `eps_estimate`, `conv_delta` |
| `eps_estimate_from_rho` | (ε, δ) estimate from a ρ value |

Both audits run whichever accountant calibrated the noise. For example, models noised under (ε, δ) still get a `rho_lb`.

---

## Pipeline

### Step 1: Run all combination train+unlearn jobs

```bash
./run_all_combinations_forget_unlearning.sh \
    <DATAROOT>          \   # path to pre-split data root
    <RESULTS_DIR_BASE>  \   # root directory for output folders
    <DATASET>           \   # cifar10 | cifar100
    <MODEL>             \   # tinynet | tinynetcifar100
    <GPUS>              \   # comma-separated GPU IDs, e.g. "0,1,2,3"
    <NUM_JOBS>          \   # parallel jobs per GPU (default: 2)
    <BATCH_SIZE>        \   # "full" or integer (default: "full")
    <MICRO_BATCH_SIZE>  \   # micro-batch size for gradient accumulation (default: 128)
    <TRAIN_EPOCHS>      \   # training epochs (default: 35)
    <UNLEARN_EPOCHS>    \   # unlearning epochs (default: 5)
    <SHUFFLE>           \   # true | false (default: false)
    <LEARNING_RATE>     \   # e.g. 0.01 (default: 0.01)
    [COMBO_INDEX]           # optional: 1-based index to run a single combination
```

Outputs are written to:
```
<RESULTS_DIR_BASE>/bs_<BS>_train_<TE>_unlearn_<UE>_lr_<LR>_comb<K>/
    run_001/
    run_002/
    ...
```

Each `run_XXX/` contains model checkpoints (`trained_*.pt`, `unlearnt_*.pt`) and a `forget_batch_indices.json`.

**Example (CIFAR-100, 4 GPUs):**
```bash
./run_all_combinations_forget_unlearning.sh \
    /data/cifar100/data_split/cifar100_uniform_bs_750_seed1 \
    /results/cifar100_combination_runs \
    cifar100 tinynetcifar100 \
    "0,1,2,3" 2 full 128 35 5 false 0.01
```

---

### Step 2: Evaluate checkpoints and compute φ statistics

For each run directory produced above, evaluate model checkpoints on the forget points. The script reloads each model `--num-shadow-reloads` times with fresh Gaussian noise and writes per-point φ/loss statistics to JSON.

```bash
# (ε, δ)-calibrated noise
python3 evaluation_scripts/evaluate_combination_models.py \
    --results-dir  <RESULTS_DIR>     \   # e.g. .../bs_full_train_35_unlearn_5_lr_0_01_comb3
    --data-dir     <DATAROOT>        \
    --model-type   unlearnt          \   # trained | unlearnt
    --epsilon      inf,100,1000      \   # "inf" or comma-separated values
    --delta        1e-3              \
    --num-shadow-reloads 100         \
    --device       cuda:0            \
    --output-dir   <STATS_DIR>

# ρ-zCDP-calibrated noise (replaces --epsilon/--delta)
python3 evaluation_scripts/evaluate_combination_models.py \
    --results-dir  <RESULTS_DIR> --data-dir <DATAROOT> --model-type unlearnt \
    --rho          inf,0.1,1.0,100   \   # "inf" = no noise; values must be > 0
    --num-shadow-reloads 100 --device cuda:0 --output-dir <STATS_DIR>
```

Produces, per noise value, in `<STATS_DIR>`:
```
<model_type>_<tag>_shadow-reloads<N>.json             # per-point stats
<model_type>_<tag>_shadow-reloads<N>_grouped.json     # grouped stats
<model_type>_<tag>_shadow-reloads<N>_sigma-info.json  # σ, T, K, accountant per run (noisy settings only)
```
where `<tag>` is `eps<E>_delta<D>` or `rho<R>`. Each JSON records `"accountant": "eps_delta" | "zcdp"` together with `epsilon`, `delta` and `rho`.

---

### Step 3: Compute ε and ρ lower bounds

Using the φ statistics from step 2, run the cumulative log-likelihood predictor and compute both audits across runs. If a stats file is missing, this step calls `evaluate_combination_models.py` to generate it, unless `--no-generate-if-missing` is passed.

```bash
python3 evaluation_scripts/evaluate_prediction_cumulative_model.py \
    --results-dir  <RESULTS_DIR>    \
    --data-dir     <DATAROOT>       \
    --stats-dir    <STATS_DIR>      \   # directory with stats JSONs from step 2
    --output-dir   <OUTPUT_DIR>     \
    --epsilon      inf,100,1000     \   # or: --rho inf,0.1,1.0,100
    --delta        1e-3             \   # noise delta (ignored with --rho)
    --metric       phi              \   # phi | loss
    --num-samples  500              \   # number of sampled runs T
    --ci-delta     0.05             \   # bounds hold w.p. 1 - ci_delta
    --conv-delta   1e-3             \   # δ for the rho_lb -> eps_estimate conversion
    --gamma-max    1e4              \   # Rényi-order search ceiling in the ρ audit
    --device       cuda:0
```

Audit-related flags:

| Flag | Default | Meaning |
|---|---|---|
| `--ci-delta` | 0.05 | Confidence slack for all lower bounds |
| `--conv-delta` | 1e-3 | δ at which `rho_lb` is also reported as `eps_estimate` (not a lower bound) |
| `--gamma-max` | 1e4 | Upper end of the Rényi-order search in the ρ audit |
| `--epsilon-delta` | 1e-8 | Only used by the m = 2 Clopper–Pearson bound. The ε/ρ audits no longer use it |

Produces `prediction_cumulative_model_results_<metric>_<tag>_shadow-reloads<N>.json`, containing:
- `audit_summary`: flat headline numbers `epsilon_lb_{avg,median}_v`, `epsilon_lb_{avg,median}_v_ldp`, `epsilon_audit_delta` (= 0), `rho_lb_{avg,median}_v`, `eps_from_rho_{avg,median}_v`, `conv_delta`, `gamma_max`.
- `avg_v_test`, `median_v_test`: full ε-audit details.
- `avg_v_test_rho`, `median_v_test_rho`: full ρ-audit details.
- `m2_cp_result` (m = 2 only), `sigma_stats`, `v_list_mean`, `v_list_median`, plus the `accountant`, `epsilon`, `delta` and `rho` of the noise.

---

### Step 4: Aggregate, plot, and tabulate

**4b: Plot ε lower bounds**

```bash
python3 evaluation_scripts/combine_prediction_cumulative_plots.py \
    --input-dir   <OUTPUT_DIR>   \   # directory with prediction_cumulative_*.json files
    --output-dir  <PLOT_DIR>
```

Produces plots of the median-v and avg-v ε lower bounds (`epsilon_lb`, i.e. the halved certified-unlearning ε) against the noise ε.

> **Note:** this script puts the JSON's `epsilon` field on the x-axis. On a `--rho` sweep every file has `epsilon = inf`, so use step 4c for ρ-calibrated results.

**4c: Collect audit summaries into a CSV**

```bash
python3 collect_rho_audit.py --results-dir <OUTPUT_DIR> [--output <CSV_PATH>]
```

Reads every `prediction_cumulative_model_results_*.json` in `<OUTPUT_DIR>`. It writes one row per noise setting, sorted by ε or ρ with `inf` last, to `<OUTPUT_DIR>/rho_audit_summary.csv` by default, and prints the headline numbers. It works on both ε- and ρ-calibrated runs. Result files that predate the ρ audit leave those columns blank.

---

## Key Modules

| File | Purpose |
|---|---|
| `r2d.py` | `h_function` (Theorem 3.1 bound), `calibrateAnalyticGaussianMechanism` (noise σ from ε, δ), `add_gaussian_noise_to_weights` |
| `forget_phi_noisy_loader.py` | `compute_GS_from_log_new_structure` (sensitivity GS), `compute_sigma_general` ((ε, δ) σ), `compute_sigma_zcdp` (ρ σ), `compute_sigma_from_log_new_structure(..., rho=None)` (dispatch), `format_rho_for_filename`, noisy model loading |
| `cum_runs_eps_lab.py` | ε lower bounds at δ = 0 (`compute_avg_v_test_epsilon_lb`, `compute_median_v_test_epsilon_lb`), ρ-zCDP lower bounds (`compute_avg_v_test_rho_lb`, `compute_median_v_test_rho_lb`), `eps_estimate_from_rho` |
| `collect_rho_audit.py` | Collects `audit_summary` blocks from result JSONs into one CSV |
| `models.py` | `TinyNet` (CIFAR-10) and `TinyNetCIFAR100` model definitions |
| `datasets.py` | Pre-split batch loaders for CIFAR-10 and CIFAR-100 |
