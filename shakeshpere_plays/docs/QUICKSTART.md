# Quick Start Guide - Multi-Run Unlearning

## Directory Structure

```
shakeshpere_plays_base/
├── run_orchestrator.py           # Main entry point (GPU-aware orchestrator)
├── unlearning_scripts/           # Contains single run logic
│   └── single_run_unlearning.py
├── configs/                      # Experiment configurations
│   ├── experiment_config_finetune_retain.json
│   └── experiment_config_ascent_descent.json
├── tests/                        # Testing scripts
│   └── test_seeding_strategy.py
├── docs/                         # Documentation
│   ├── SEEDING_STRATEGY.md       # Detailed seeding explanation
│   └── README_QUICK_START.md     # This file
├── data_loader.py
├── trainer_utils.py
├── engine.py
└── logs/                         # Generated: run logs and metrics
```

## Setup

### 1. Prepare Conda Environment
```bash
conda activate torch_jax_gpu
# or create new:
# conda create -n torch_jax_gpu python=3.10
```

### 2. Verify Data
Ensure you have one of these datasets:
- `data_splits_speakers300_fs10/` - Contains train/retain/val/test/forget_*.txt
- `data_splits_speakers300/`

## Running Experiments

### Strategy 1: Finetune + Retain

Train on combined (retain + sampled forget), then finetune on retain only.

```bash
# Single run test
python unlearning_scripts/single_run_unlearning.py \
  --run_id 0 \
  --experiment_config configs/experiment_config_finetune_retain.json \
  --gpu 0

# Multiple runs with orchestrator
python run_orchestrator.py \
  --experiment_config configs/experiment_config_finetune_retain.json \
  --num_gpus 2 \
  --max_runs_per_gpu 3
```

### Strategy 2: Ascent-Descent

Gradient ascent on forget set, descent on retain set.

```bash
# Single run test
python unlearning_scripts/single_run_unlearning.py \
  --run_id 0 \
  --experiment_config configs/experiment_config_ascent_descent.json \
  --gpu 0

# Multiple runs with orchestrator
python run_orchestrator.py \
  --experiment_config configs/experiment_config_ascent_descent.json \
  --num_gpus 4 \
  --max_runs_per_gpu 2
```

## Configuration

### Edit Experiment Config

Edit `configs/experiment_config_finetune_retain.json`:

```json
{
  "experiment": {
    "run_folder": "runs_finetune",
    "num_runs": 50,
    "forget_prob": 0.3,
    "strategy": "finetune_retain",
    "dataset": "data_splits_speakers300_fs10"
  },
  "training": {
    "epochs": 15,
    "batch_size": 256,
    "lr": 0.1,
    "weight_decay": 0.0,
    "seed": 42
  },
  "unlearning": {
    "finetune_retain": {
      "phase_2_epochs_ratio": 0.5,
      "phase_2_lr_ratio": 0.5
    }
  }
}
```

Key parameters:
- `num_runs`: Number of experiment runs (e.g., 50)
- `forget_prob`: Fraction of forget sets to sample (0.3 = 30%)
- `seed`: Base seed for reproducibility
- `weight_decay`: L2 regularization strength

## Output Structure

Each run creates:
```
runs_finetune/
├── run_0/
│   ├── config.json               # Run-specific config with seeds
│   ├── run.log                   # All metrics/training logs
│   ├── model_trained.pt          # After phase 1
│   ├── model_unlearnt.pt         # After phase 2 (final)
│   ├── metrics.json              # Final metrics
│   └── data_temp/                # Cleaned up after run
├── run_1/
│   ├── ...
```

### View Results

```bash
# Show metrics for a specific run
cat runs_finetune/run_0/metrics.json | python -m json.tool

# Example output:
{
  "strategy": "finetune_retain",
  "phase_1_best_val_acc": 0.8234,
  "phase_1_duration": 145.2,
  "phase_2_best_val_acc": 0.8456,
  "phase_2_duration": 72.1,
  "test_loss": 1.2345,
  "test_accuracy": 0.8345,
  "test_perplexity": 3.456,
  "forget_indices": [0, 2, 5, 7]
}

# View training log for a run
tail -100 runs_finetune/run_0/run.log
```

## Seeding Verification

Verify seeding is working correctly:

```bash
# Check seeds in a run
cat runs_finetune/run_0/config.json | grep seed

# Expected:
{
  "training_seed": 42,          # Same for all runs
  "forget_sampling_seed": 1000, # Different per run
  "unlearning_seed": 2000       # Different per run
}

# Compare first 5 runs' forget indices (should all be different)
for i in {0..4}; do
  echo "Run $i:"
  cat runs_finetune/run_$i/metrics.json | grep forget_indices
done
```

## Orchestrator Options

```bash
python run_orchestrator.py \
  --experiment_config <config_file> \
  --num_gpus <number> \
  --max_runs_per_gpu <number>
```

Parameters:
- `--experiment_config`: Path to config JSON
- `--num_gpus`: Number of available GPUs (default: 1)
- `--max_runs_per_gpu`: Max concurrent runs per GPU (default: 2)

Example: 4 GPUs, 2 runs per GPU = up to 8 concurrent runs
```bash
python run_orchestrator.py \
  --experiment_config configs/experiment_config_finetune_retain.json \
  --num_gpus 4 \
  --max_runs_per_gpu 2
```

## Logging

All output goes to run log files (not stdout):
```
runs_finetune/run_0/run.log
runs_finetune/run_1/run.log
runs_finetune/run_2/run.log
...
```

To monitor a running experiment:
```bash
# Watch a specific run's log
tail -f runs_finetune/run_0/run.log

# Count completed runs
ls runs_finetune/ | wc -l

# Check for errors
grep ERROR runs_finetune/*/run.log
```

## Common Issues

### GPU Memory Error
Reduce batch size in config:
```json
"training": {
  "batch_size": 128  // Was 256
}
```

### All Runs Have Same Forget Indices
Seeds not being applied correctly. Check:
1. `forget_sampling_seed` is unique in each `config.json`
2. Seed reset happens after sampling (line 443 in single_run_unlearning.py)

### Runs Taking Too Long
- Reduce `num_runs` for testing
- Reduce `epochs` in config
- Increase `max_runs_per_gpu` to parallelize more

### Out of Disk Space
Output can grow large with many runs. Each run ~100-500MB.
Check available space:
```bash
du -sh runs_finetune/  # Current size
df -h /lfs/mercury1    # Available space
```

## Next Steps

1. **Run a single test**:
   ```bash
   python unlearning_scripts/single_run_unlearning.py \
     --run_id 0 \
     --experiment_config configs/experiment_config_finetune_retain.json \
     --gpu 0
   ```

2. **Check the output**:
   ```bash
   cat runs_finetune/run_0/run.log
   cat runs_finetune/run_0/metrics.json
   ```

3. **Run full experiment**:
   ```bash
   python run_orchestrator.py \
     --experiment_config configs/experiment_config_finetune_retain.json \
     --num_gpus 2 \
     --max_runs_per_gpu 2
   ```

4. **Analyze results** (after runs complete):
   ```bash
   python analyze_multi_run_results.py --run_folder runs_finetune
   ```

## For Detailed Information

- **Seeding strategy**: See `docs/SEEDING_STRATEGY.md`
- **Configuration options**: See config JSON comments
- **Troubleshooting**: See `docs/SEEDING_STRATEGY.md` → Troubleshooting section
