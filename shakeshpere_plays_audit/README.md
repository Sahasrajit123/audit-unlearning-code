# Shakespeare Machine Unlearning

## Overview

This codebase implements a machine unlearning framework on the Shakespeare character-level language modelling dataset, along with a membership inference audit based on log-likelihood ratios (LLR). The audit yields lower bounds on the certified-unlearning ε (pure (ε, 0)), the zCDP parameter ρ, and the GDP parameter μ.

## Repository Structure

```
.
├── prepare_data.py                   # Generate train/retain/forget data splits
├── data.py                           # Shakespeare download, vocab, and role parsing
├── data_loader.py                    # PyTorch DataLoaders from pre-generated splits
├── model.py                          # ShakespeareLSTM model definition
├── engine.py                         # Training engine
├── trainer_utils.py                  # Optimizer, scheduler, and training utilities
├── train.py                          # Standalone training script (reads config.json)
├── config.json                       # Model/training config for train.py
├── single_run_unlearning.py          # Single unlearning experiment runner
├── run_orchestrator.py               # GPU-aware parallel orchestrator
├── compute_forget_set_losses.py      # Precompute per-forget-file loss statistics
├── evaluate_llr_predictions.py       # LLR attack + epsilon / rho / mu audits
├── audit_from_predictions.py         # Re-run the audits at new r from saved LLR scores
├── run_evaluate_llr_predictions.sh   # Run evaluate_llr_predictions.py over all run folders
├── cum_runs_eps_lab.py               # Audit math (avg-v / median-v tests for eps, rho, mu)
├── summarize_runs.py                 # Summarize metrics across runs
├── configs/                          # Experiment configuration JSON files
├── unlearning_scripts/               # Per-strategy unlearning implementations
├── scripts/                          # Run management and analysis helpers
├── docs/                             # Seeding and reproducibility notes
└── other_evals/                      # Forget-loss analysis script
```

## Setup

Requires a CUDA GPU for training. Tested with:

| Package | Version |
|---------|---------|
| Python | 3.11 |
| PyTorch | 2.5.1 (CUDA 12.1) |
| NumPy | 1.26 |
| SciPy | 1.15 |
| pandas | 2.2 |
| tqdm | 4.67 |

Unlearning experiments are configured by the JSON files in `configs/` (see below). The top-level `config.json` is only used by the standalone `train.py`.

## Data

The dataset is derived from the Complete Works of Shakespeare, partitioned by speaking role following McMahan et al. (2017). We use 300 speakers, with 267 retained and 33 in the forget pool, split into 400 disjoint forget files.

### Generate data splits

```bash
python prepare_data.py --num_speakers 300 --forget_splits 400 --output_dir data_splits --seed 42
```

This downloads `shakespeare.txt` from Project Gutenberg if not present and writes splits to `data_splits_speakers300_fs400/` (the speaker and forget-split suffixes are appended to `--output_dir`):
- `retain.txt` — retain set text
- `train.txt`, `val.txt`, `test.txt`, `full.txt` — full training splits
- `forget/forget_0.txt ... forget_399.txt` — 400 disjoint forget files
- `meta.json` — vocab, speaker assignments, and split metadata

All experiment configs read their data from `data_splits_speakers300_fs400/`.

## Running Experiments

### Experiment configs

| Config | Strategy | Output folder |
|--------|----------|---------------|
| `experiment_config_interleaved_ascent_descent_q_1.json` | ascent-descent, q=1 | `runs_ascent_descent_fs400_q_1` |
| `experiment_config_interleaved_ascent_descent_q_1_var1.json` | ascent-descent, q=1, λ=1.5 | `runs_ascent_descent_fs400_q_1_var1` |
| `experiment_config_interleaved_ascent_descent_q_1_var2.json` | ascent-descent, q=1, forget ratio 0.875 | `runs_ascent_descent_fs400_q_1_var2` |
| `experiment_config_interleaved_ascent_descent_q_1_var3.json` | ascent-descent, q=1, λ=1.5, forget ratio 0.875 | `runs_ascent_descent_fs400_q_1_var3` |
| `experiment_config_interleaved_ascent_descent_q_4.json` | ascent-descent, q=4 | `runs_ascent_descent_fs400_mid_q` |
| `experiment_config_interleaved_ascent_descent_q_9.json` | ascent-descent, q=9 | `runs_ascent_descent_fs400_q_9` |
| `experiment_config_ascent_forget.json` | ascent then descent (q=None) | `runs_ascent_descent_fs400_q_None` |
| `experiment_config_hessian_unlearning.json` | Hessian-based unlearning | `runs_hessian_unlearning_fs400` |
| `experiment_config_finetune_retain.json` | finetune on retain | `runs_finetune_fs400` |

### Single run

```bash
python single_run_unlearning.py \
  --run_id 0 \
  --experiment_config configs/experiment_config_interleaved_ascent_descent_q_1.json \
  --gpu 0
```

### Multi-run with orchestrator (parallel, GPU-aware)

```bash
python run_orchestrator.py \
  --experiment_config configs/experiment_config_interleaved_ascent_descent_q_1.json \
  --gpus 0,1,2,3 \
  --max_runs_per_gpu 2 \
  --num_runs 75
```

Each run saves to `<run_folder>/run_<id>/`:
- `model_trained.pt` — model after training phase
- `model_unlearnt.pt` — model after unlearning
- `forget_indices.json` — forget files sampled into this run's training set
- `metrics.json` — loss, accuracy, perplexity
- `run.log` — full training log

Each run samples each forget file independently with probability `forget_prob` = 0.5, seeded by its run id.

### Split runs into shadow and test runs

The audit uses two disjoint groups of runs from the same config: **shadow runs**, which estimate the loss statistics, and **test runs**, which are attacked. In the paper, runs 0–24 are test runs and runs 25–74 are shadow runs. After the orchestrator finishes, move the test runs into `test_run/`:

```bash
RUNS_DIR=runs_ascent_descent_fs400_q_1
mkdir -p ${RUNS_DIR}/test_run
for i in $(seq 0 24); do mv ${RUNS_DIR}/run_${i} ${RUNS_DIR}/test_run/; done
```

Every `run_*` folder left directly under `RUNS_DIR` is used as a shadow run, so move any runs with id ≥ 75 out of `RUNS_DIR` too.

## Unlearning Strategies

**Ascent-Descent**: Gradient ascent on the sampled forget set combined with descent on the retain set. During the forget phase (`forget_epochs_ratio` of the epochs), one combined forget-ascent/retain-descent step is taken per `q` retain-only descent steps. `q: null` runs a pure ascent phase followed by a pure descent phase.
```json
{
  "strategy": "ascent_descent",
  "ascent_descent": { "epochs": 8, "q": 1, "lambda_coef": 0.5, "forget_epochs_ratio": 0.5, "lr": 0.05 }
}
```

**Hessian unlearning**: A Newton-style update that uses a stochastic Hessian estimate and adds Gaussian noise.
```json
{
  "strategy": "hessian_unlearning",
  "hessian_unlearning": { "s1": 5, "s2": 700, "scale": 5000.0, "std": 0.001, "gamma": 0.05 }
}
```

**Finetune-Retain**: Train on retain plus the sampled forget files, then finetune on the retain set only.
```json
{
  "strategy": "finetune_retain",
  "finetune_retain": { "phase_2_epochs": 8, "phase_2_lr": 0.05 }
}
```

## LLR Audit

`evaluate_llr_predictions.py` attacks the unlearned test models with a log-likelihood ratio (LLR) membership inference attack, then turns the attack's success into lower bounds on the privacy parameters.

### How it works

1. **Precompute loss statistics** (`compute_forget_set_losses.py`): Evaluates every shadow model on every forget file and records the per-position loss mean and variance, separately for models that were trained on that forget file ("chosen") and models that were not ("not chosen").

2. **Compute cumulative LLR**: For each forget file, sums per-position log-likelihood ratios comparing how likely the test model's observed loss is under the "chosen" versus the "not chosen" Gaussian. A high score suggests the test model was trained on that file.

3. **Guess**: The audit parameter `r` is the **total** guess budget. The attack guesses the top `r/2` files by LLR as members and the bottom `r/2` as non-members. The score `v = (#correct top guesses) + (#correct bottom guesses)` ranges over `[0, r]`. `r` must be even and at most the number of forget files `m`.

4. **Audit**: From the T test runs' `v` values, the script computes lower bounds with two tests, one on the mean of `v` (Chernoff bound) and one on the median of `v`:
   - `epsilon_lb`: pure (ε, 0) certified-unlearning ε. The audit bounds the LDP parameter of the audit mechanism, and (ε, 0)-certified unlearning implies (2ε, 0)-LDP, so the reported value is half of that bound. The unhalved value is `epsilon_lb_ldp`.
   - `rho_lb`: zCDP ρ.
   - `mu_lb`: GDP μ.

   The `eps_estimate_*` fields convert the audited ρ and μ into an ε at δ = `--conv_delta`. They are reported only for comparison and are **not** lower bounds on ε.

### Usage

```bash
# Step 1: precompute loss statistics over the shadow runs
# (evaluate_llr_predictions.py also runs this automatically if the file is missing)
python compute_forget_set_losses.py \
  --runs_dir runs_ascent_descent_fs400_q_1 \
  --data_dir data_splits_speakers300_fs400 \
  --device cuda:0

# Step 2: attack the test runs and compute the audits
python evaluate_llr_predictions.py \
  --runs_dir runs_ascent_descent_fs400_q_1 \
  --data_dir data_splits_speakers300_fs400 \
  --model_type unlearnt \
  --r 100 \
  --T 10
```

To run step 2 over every run folder in the table above with the paper's settings (r = 100, T = 10), use:

```bash
bash run_evaluate_llr_predictions.sh
```

**Outputs** (written to `<runs_dir>/`; the `_T<T>` suffix is added only when `--T` is given):
- `llr_predictions_unlearnt[_T<T>].json` — per-run LLR scores, guesses, and `v`
- `llr_epsilon_lb_unlearnt[_T<T>].json` — audit summary (`avg_v_test`, `median_v_test`, `avg_v_test_rho`, `median_v_test_rho`, `avg_v_test_mu`, `median_v_test_mu`, `eps_estimate_*`)

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--runs_dir` | `runs_ascent_descent` | Run folder (shadow runs + `test_run/`) |
| `--data_dir` | `data_splits_speakers300_fs400` | Data splits directory |
| `--model_type` | `unlearnt` | Evaluate `trained` or `unlearnt` models |
| `--r` | `100` | Total guess budget (top r/2 + bottom r/2); even, ≤ m |
| `--T` | all | Number of test runs; uses the T smallest run ids |
| `--ci_delta` | `0.05` | Confidence level of the audit (failure probability) |
| `--avg_direction` | `ge` | Direction for the mean-v ε test (`ge` or `le`) |
| `--theta_max` | `50.0` | θ upper bound for Chernoff optimization |
| `--gamma_max` | `1e4` | Upper bound on the Rényi order searched in the ρ audit |
| `--conv_delta` | `1e-3` | δ used only for the `eps_estimate_*` conversions |

### Re-auditing at other r

The saved predictions contain the full LLR score for every forget file, so changing `r` doesn't require re-running any models or a GPU:

```bash
python audit_from_predictions.py --r 100 300 --T 10 --csv audit_sweep_T10.csv
```

This writes `<runs_dir>/llr_epsilon_lb_unlearnt_T<T>_r<r>.json` for each folder and value of r, plus one combined CSV. Add `--dry_run` to print the results without writing any files.
