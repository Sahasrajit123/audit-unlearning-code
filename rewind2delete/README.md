# Rewind-To-Delete (R2D): Certified Machine Unlearning for Nonconvex Functions

Code and scripts for the combination-forget unlearning experiments.

This code is mainly taken from the [repo](https://github.com/siqiaomu/r2d).

---

## Overview

R2D provides certified machine unlearning guarantees for nonconvex models. Given a trained model and a set of data points to forget, R2D unlearns by continuing gradient descent on the retain set with added Gaussian noise, then certifies the unlearning via a hypothesis-testing lower bound on the privacy parameter ε.

This folder covers the **combination forget** experimental pipeline:
1. Train a model on a dataset split into retain + multiple forget batches.
2. For every combination of `N/2` forget batches (out of `N` total), train from scratch then unlearn — in parallel across GPUs.
3. Evaluate each trained/unlearnt model checkpoint and compute per-forget-point φ statistics.
4. Aggregate across combinations to compute ε lower bounds and produce plots.

---

## Repository Structure

```
rewind2delete/
├── run_all_combinations_forget_unlearning.sh   # Master launcher (step 1–2)
├── random_forget_runner_all_combinations.py    # Per-combination train+unlearn runner
├── main.py                                     # Core training and unlearning engine
│
├── models.py                                   # TinyNet model definitions
├── datasets.py                                 # Dataset loading (CIFAR-10, CIFAR-100)
├── r2d.py                                      # Core R2D math: h(), noise calibration, bounds
├── forget_phi_noisy_loader.py                  # φ statistic computation with noisy reloads
├── cum_runs_eps_lab.py                         # ε lower bound estimators (Chernoff, g-test)
├── mia.py                                      # Membership inference attack utilities
├── utils.py                                    # Shared training utilities
├── logger.py                                   # Logging helpers
│
└── evaluation_scripts/
    ├── evaluate_combination_models.py          # Step 3: evaluate checkpoints, output stats JSON
    ├── evaluate_grouped_predictions.py         # Helper: grouped LLR computation
    ├── evaluate_prediction_cumulative_model.py # Step 4a: predict runs, compute ε lower bounds
    └── combine_prediction_cumulative_plots.py  # Step 4b: aggregate JSONs and plot
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

## Data Format

Scripts expect a pre-split dataset directory with the following layout:

```
<DATAROOT>/
├── train/
├── val/
├── test/
├── retain/
└── forget/
    ├── batch_0.pkl
    ├── batch_1.pkl
    └── ...
```

Each `batch_i.pkl` is a list of `(image_tensor, label, identity)` tuples. The number of forget batches is detected automatically; combination size is set to `NUM_FORGET_BATCHES / 2`.

---

## Pipeline

### Step 1 — Run all combination train+unlearn jobs

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

### Step 2 — Evaluate checkpoints and compute φ statistics

For each run directory produced above, evaluate model checkpoints on the forget points and write per-model φ/loss statistics to JSON.

```bash
python3 evaluation_scripts/evaluate_combination_models.py \
    --results-dir  <RESULTS_DIR>     \   # e.g. .../bs_full_train_35_unlearn_5_lr_0_01_comb3
    --data-dir     <DATAROOT>        \
    --model-type   unlearnt          \   # trained | unlearnt
    --epsilon      inf               \   # "inf" or comma-separated values, e.g. "inf,0.5,0.1"
    --delta        1e-3              \
    --num-shadow-reloads 100         \
    --device       cuda:0            \
    --output-dir   <OUTPUT_DIR>
```

Produces `stats_<model_type>_eps<X>_delta<Y>.json` files in `<OUTPUT_DIR>`.

---

### Step 3 — Compute ε lower bounds

Using the φ statistics from step 2, run the cumulative log-likelihood predictor and compute ε lower bounds across runs.

```bash
python3 evaluation_scripts/evaluate_prediction_cumulative_model.py \
    --results-dir  <RESULTS_DIR>    \
    --data-dir     <DATAROOT>       \
    --stats-dir    <OUTPUT_DIR>     \   # directory with stats JSONs from step 2
    --output-dir   <OUTPUT_DIR>     \
    --epsilon      inf              \
    --delta        1e-3             \
    --metric       phi              \   # phi | loss
    --num-samples  500              \
    --device       cuda:0
```

Produces `prediction_cumulative_*.json` files.

---

### Step 4 — Aggregate and plot

```bash
python3 evaluation_scripts/combine_prediction_cumulative_plots.py \
    --input-dir   <OUTPUT_DIR>   \   # directory with prediction_cumulative_*.json files
    --output-dir  <PLOT_DIR>
```

Produces plots of median and mean ε lower bounds across combinations.

---

## Key Modules

| File | Purpose |
|---|---|
| `r2d.py` | `h_function` (Theorem 3.1 bound), `calibrateAnalyticGaussianMechanism` (noise σ from ε, δ), `add_gaussian_noise_to_weights` |
| `forget_phi_noisy_loader.py` | Loads model with Gaussian noise `num_shadow_reloads` times and averages φ; extracts training params from checkpoint path |
| `cum_runs_eps_lab.py` | Chernoff-bound ε lower bound estimators (`compute_avg_v_test_epsilon_lb`, `compute_median_v_test_epsilon_lb`) |
| `models.py` | `TinyNet` (CIFAR-10) and `TinyNetCIFAR100` model definitions |
| `datasets.py` | Pre-split batch loaders for CIFAR-10 and CIFAR-100 |
