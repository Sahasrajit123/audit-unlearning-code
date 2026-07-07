# Shakespeare Machine Unlearning

## Overview

This codebase implements a machine unlearning framework on the Shakespeare character-level language modelling dataset, along with a membership inference auditing method based on log-likelihood ratios (LLR) to compute privacy epsilon lower bounds.

## Repository Structure

```
.
├── train.py                          # Standalone training script
├── single_run_unlearning.py          # Single unlearning experiment runner
├── run_orchestrator.py               # GPU-aware parallel orchestrator
├── evaluate_llr_predictions.py       # LLR-based MIA evaluation + epsilon lower bound
├── compute_forget_set_losses.py      # Precompute per-forget-file loss statistics
├── cum_runs_eps_lab.py               # Epsilon lower bound math (avg-v / median-v tests)
├── prepare_data.py                   # Generate train/retain/forget data splits
├── data.py                           # Shakespeare download, vocab, and role parsing
├── data_loader.py                    # PyTorch DataLoaders from pre-generated splits
├── model.py                          # ShakespeareLSTM model definition
├── engine.py                         # Training engine
├── trainer_utils.py                  # Optimizer, scheduler, and training utilities
├── summarize_runs.py                 # Summarize metrics across runs
├── configs/                          # Experiment configuration JSON files
├── unlearning_scripts/               # Per-strategy unlearning implementations
├── data_splits_speakers300_fs400/    # Pre-generated splits (300 speakers, 400 forget files)
├── docs/                             # Additional documentation
└── other_evals/                      # Additional evaluation scripts
```

## Setup

```bash
conda activate torch_jax_gpu
```

## Data

The dataset is derived from the Complete Works of Shakespeare, partitioned by speaking role following McMahan et al. (2017). We use 300 speakers, with 267 retained and 33 in the forget pool, split into 400 disjoint forget files.

### Generate data splits

```bash
python prepare_data.py
```

This downloads `shakespeare.txt` from Project Gutenberg if not present and writes splits to `data_splits_speakers300_fs400/`:
- `retain.txt` — retain set text
- `train.txt`, `val.txt`, `test.txt` — full training splits
- `forget/forget_0.txt ... forget_399.txt` — 400 disjoint forget files
- `meta.json` — vocab, speaker assignments, and split metadata

Pre-generated splits are included in `data_splits_speakers300_fs400/`.

## Running Experiments

### Single run

```bash
python single_run_unlearning.py \
  --run_id 0 \
  --experiment_config configs/experiment_config_ascent_descent_q_1.json \
  --gpu 0
```

### Multi-run with orchestrator (parallel, GPU-aware)

```bash
python run_orchestrator.py \
  --experiment_config configs/experiment_config_ascent_descent_q_1.json \
  --gpus 0,1,2,3 \
  --max_runs_per_gpu 2
```

Each run saves to `<run_folder>/run_<id>/`:
- `model_trained.pt` — model after training phase
- `model_unlearnt.pt` — model after unlearning
- `metrics.json` — loss, accuracy, perplexity
- `run.log` — full training log

## Unlearning Strategies

Two strategies are implemented:

**Ascent-Descent**: Alternating gradient ascent on the sampled forget set and descent on the retain set.
```json
{
  "strategy": "ascent_descent",
  "unlearning": {
    "ascent_descent": { "epochs": 8, "q": 1, "lambda_coef": 0.5 }
  }
}
```

**Finetune-Retain**: Train on combined (retain + sampled forget), then finetune on retain only.
```json
{
  "strategy": "finetune_retain",
  "unlearning": {
    "finetune_retain": { "phase_2_epochs_ratio": 0.5, "phase_2_lr_ratio": 0.5 }
  }
}
```

## LLR Evaluation and Epsilon Lower Bound

`evaluate_llr_predictions.py` audits a set of unlearned models using a log-likelihood ratio (LLR) membership inference attack and computes a privacy epsilon lower bound.

### How it works

1. **Precompute loss statistics** (`compute_forget_set_losses.py`): Evaluates all trained/unlearned models from the main runs on every forget file, recording per-position loss mean and variance separately for models that included that forget file in training ("chosen") vs. those that did not ("not chosen").

2. **Compute cumulative LLR**: For each forget file index, sums per-position log-likelihood ratios comparing how likely the observed loss is under the "chosen" vs. "not chosen" Gaussian distribution. A high LLR score indicates the test model was likely trained on that forget file.

3. **Predict forget indices**: Ranks forget files by LLR score. The top-r are predicted as the forget set; the bottom-r serve as a contrast group. Prediction accuracy is computed against ground-truth forget indices.

4. **Epsilon lower bound**: Using the overlap count `v = top_r_correct + bottom_r_correct` across T test runs, computes a lower bound on privacy epsilon via two tests:
   - **Avg-v test**: Chernoff-based bound on the average overlap
   - **Median-v test**: Concentration bound on the median overlap

### Usage

```bash
# Step 1: precompute loss statistics (runs automatically if missing)
python compute_forget_set_losses.py \
  --runs_dir runs_ascent_descent_fs400_q_1 \
  --data_dir data_splits_speakers300_fs400 \
  --device cuda:0

# Step 2: evaluate LLR and compute epsilon lower bound
python evaluate_llr_predictions.py \
  --runs_dir runs_ascent_descent_fs400_q_1 \
  --data_dir data_splits_speakers300_fs400 \
  --model_type unlearnt \
  --r 50 \
  --delta 1e-5 \
  --ci_delta 0.05
```

**Outputs** (written to `<runs_dir>/`):
- `llr_predictions_unlearnt.json` — per-run LLR scores and prediction accuracies
- `llr_epsilon_lb_unlearnt.json` — epsilon lower bound summary

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--runs_dir` | `runs_ascent_descent` | Main runs directory |
| `--data_dir` | `data_splits_speakers300_fs400` | Data splits directory |
| `--model_type` | `unlearnt` | Evaluate `trained` or `unlearnt` models |
| `--r` | `50` | Top/bottom r for LLR prediction |
| `--delta` | `1e-5` | δ for epsilon lower bound |
| `--ci_delta` | `0.05` | Confidence tail probability |
| `--avg_direction` | `ge` | Direction for avg-v test (`ge` or `le`) |
| `--theta_max` | `50.0` | θ upper bound for Chernoff optimization |
