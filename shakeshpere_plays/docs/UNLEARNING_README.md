# Multi-Run Unlearning Framework

## Overview

A complete multi-run unlearning framework with GPU-aware orchestration, comprehensive logging, and configurable training/unlearning strategies.

### Components

1. **`run_orchestrator.py`** - GPU-aware orchestrator
   - Dynamically assigns runs to GPUs based on availability
   - Spawns worker processes in parallel
   - Collects results and provides summary statistics

2. **`single_run_unlearning.py`** - Single-run executor
   - Executes one unlearning experiment with specific run ID
   - Handles both `finetune_retain` and `ascent_descent` strategies
   - Logs all metrics to `run_X/run.log`
   - Saves config and metrics to `run_X/config.json` and `run_X/metrics.json`

3. **`experiment_config_finetune_retain.json`** - Config for finetune strategy
4. **`experiment_config_ascent_descent.json`** - Config for ascent-descent strategy

## Randomness Strategy

### Fixed Across All Runs (Identical Training)
- **Training seed**: Used for model initialization, data loading, batch shuffling
- **Data loader seed**: Ensures same batch order across all runs
- → All 50 runs train identically on the same data

### Independent Per Run
- **Forget sampling seed**: `training_seed + 1000 + run_id` → Different forget sets sampled per run
- **Unlearning seed**: `training_seed + 2000 + run_id` → Different unlearning behavior per run

## Configuration

### Common Parameters (in `training` section):
```json
{
  "training": {
    "epochs": 15,
    "batch_size": 256,
    "lr": 0.1,
    "clip": 1.0,
    "optimizer": "sgd",
    "seed": 42,
    "weight_decay": 0.0
  }
}
```

### Strategy-Specific Configs

**finetune_retain**: Train on (retain + sampled_forget), then finetune on retain
```json
{
  "experiment": {
    "forget_prob": 0.3,
    "strategy": "finetune_retain"
  },
  "unlearning": {
    "finetune_retain": {
      "phase_2_epochs_ratio": 0.5,
      "phase_2_lr_ratio": 0.5
    }
  }
}
```

**ascent_descent**: Alternating gradient ascent on forget + descent on retain
```json
{
  "experiment": {
    "forget_prob": 0.5,
    "strategy": "ascent_descent"
  },
  "unlearning": {
    "ascent_descent": {
      "epochs": 10,
      "q": 9,
      "lambda_coef": 0.5,
      "forget_epochs_ratio": 0.5,
      "weight_decay": 0.01
    }
  }
}
```

## Weight Decay Explanation

**What is it?** L2 regularization penalty: `loss = ce_loss + weight_decay * sum(weights²)`

**Why use it?** Keeps model weights small → prevents overfitting

**Where it's used:**
- Computed as: `0.5 * sum(p² for p in parameters) * weight_decay`
- Applied to all training steps in both strategies
- Typical values: 0.0 (off) to 0.1

**In the code:**
- `compute_loss_with_weight_decay()` helper function adds L2 penalty to CE loss
- Used in both `train_finetune_retain()` and `train_ascent_descent()`

## Usage

### Setup

Create directory for experiments:
```bash
mkdir -p runs_finetune runs_ascent_descent
```

### Run finetune_retain (30 concurrent runs across 2 GPUs)
```bash
python run_orchestrator.py \
  --experiment_config experiment_config_finetune_retain.json \
  --num_gpus 2 \
  --max_runs_per_gpu 15
```

### Run ascent_descent (50 runs on 4 GPUs)
```bash
python run_orchestrator.py \
  --experiment_config experiment_config_ascent_descent.json \
  --num_gpus 4 \
  --max_runs_per_gpu 2
```

### Run single run manually
```bash
python single_run_unlearning.py \
  --run_id 0 \
  --experiment_config experiment_config_finetune_retain.json \
  --gpu 0
```

## Output Structure

```
runs_finetune/
├── run_0/
│   ├── run.log                 # All metrics logged here
│   ├── config.json             # Run configuration
│   ├── metrics.json            # Final metrics
│   ├── model_trained.pt        # After phase 1
│   └── model_unlearnt.pt       # After unlearning
├── run_1/
│   ├── run.log
│   ├── config.json
│   ├── metrics.json
│   ├── model_trained.pt
│   └── model_unlearnt.pt
├── ...
└── run_49/
```

### Log File Format (`run.log`)
```
2026-04-08 10:15:23 [INFO] ================================================================================
2026-04-08 10:15:23 [INFO] RUN 0: Starting on cuda:0
2026-04-08 10:15:23 [INFO] ================================================================================
2026-04-08 10:15:23 [INFO]
2026-04-08 10:15:23 [INFO] Run Configuration:
2026-04-08 10:15:23 [INFO]   Strategy: finetune_retain
2026-04-08 10:15:23 [INFO]   Forget prob: 0.3
2026-04-08 10:15:23 [INFO]   Training seed: 42 (FIXED)
2026-04-08 10:15:23 [INFO]   Forget sampling seed: 1042
2026-04-08 10:15:23 [INFO]   Unlearning seed: 2042
...
```

## Key Features

✅ **GPU-aware scheduling** - Dynamic assignment based on load
✅ **Parallel execution** - Run multiple experiments on multiple GPUs
✅ **Fixed training randomness** - Identical across all runs for fair comparison
✅ **Independent unlearning** - Different forget sets & unlearning seeds per run
✅ **Comprehensive logging** - All metrics in run-specific log files (no terminal spam)
✅ **Weight decay support** - Fully integrated in all training phases
✅ **Configuration driven** - No hardcoding, all params in JSON configs
✅ **Model checkpointing** - Saves trained and unlearnt models per run

