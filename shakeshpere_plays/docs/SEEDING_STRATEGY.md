# Seeding Strategy & Reproducibility Guide

## Overview

This document explains the three-tier seeding strategy used in the multi-run unlearning experiments to ensure:
1. **Training randomness is IDENTICAL** across all runs (same model initialization, same batch shuffling)
2. **Forget sampling is INDEPENDENT** per run (different forget sets sampled each time)
3. **Unlearning randomness is INDEPENDENT** per run (different unlearning dynamics each time)

## Seeding Architecture

### Three Independent Seed Streams

Each run uses three separate seeds with no overlap:

```
training_seed = base_seed (e.g., 42) → FIXED for ALL runs
forget_sampling_seed = base_seed + 1000 + run_id → DIFFERENT per run
unlearning_seed = base_seed + 2000 + run_id → DIFFERENT per run
```

Example for `run_id=5` with `base_seed=42`:
- `training_seed = 42` (same for all runs)
- `forget_sampling_seed = 1047` (unique to run 5)
- `unlearning_seed = 2047` (unique to run 5)

### Seed Guarantees

| Aspect | Seed | Consistency | Reason |
|--------|------|-------------|--------|
| Model initialization | `training_seed` | IDENTICAL across runs | Reproducible learning baseline |
| Batch shuffling | `training_seed` | IDENTICAL across runs | Same data order on all runs |
| Forget set sampling | `forget_sampling_seed` | INDEPENDENT per run | Different forget sets each run |
| Unlearning dynamics | `unlearning_seed` | INDEPENDENT per run | Different gradient behavior each run |

## Implementation Details

### 1. Training Phase (Identical Seed)

```python
# Set FIXED training seed (same for all runs)
training_seed = exp_config["training"]["seed"]  # e.g., 42
torch.manual_seed(training_seed)
np.random.seed(training_seed)
random.seed(training_seed)

# Consequences:
# - Model weights initialized identically
# - DataLoader shuffles same samples in same order
# - Dropout uses same random masks
# - All training dynamics are reproducible
```

**Result**: Every run trains on identical sequences of training data with identical model initialization.

### 2. Forget Sampling Phase (Independent Seed)

```python
# Sample forget sets using RUN-SPECIFIC seed
forget_sampling_seed = training_seed + 1000 + run_id
np.random.seed(forget_sampling_seed)
random.seed(forget_sampling_seed)

forget_indices = list(np.random.choice(
    len(forget_texts),
    size=max(1, int(forget_prob * len(forget_texts))),
    replace=False
))

# Restore training seed after sampling
torch.manual_seed(training_seed)
np.random.seed(training_seed)
random.seed(training_seed)
```

**Result**: Each run gets a different random subset of forget sets, but training data loading remains identical.

### 3. Unlearning Phase (Independent Seed)

```python
# Set RUN-SPECIFIC unlearning seed before unlearning
unlearning_seed = training_seed + 2000 + run_id
torch.manual_seed(unlearning_seed)
np.random.seed(unlearning_seed)
random.seed(unlearning_seed)

# Dropout, unlearning randomness uses this seed
# Different per run → different unlearning trajectories
```

**Result**: Each run's unlearning algorithm uses independent random behaviors.

## Key Design Decisions

### Why Separate Forget Sampling from Training Seed?

If we used the training seed for forget sampling:
```python
# BAD approach:
np.random.seed(training_seed)
forget_indices = sample_forget()  # Same indices every run!
```

This would mean all runs forget the same data, which defeats the purpose of statistical analysis. We want **different forgetting patterns** with **identical training dynamics** to isolate unlearning effectiveness.

### Why Restore Training Seed After Forget Sampling?

After sampling forget sets, we MUST restore the training seed:
```python
np.random.seed(forget_sampling_seed)
forget_indices = sample_forget()

# IMPORTANT: Restore training seed
np.random.seed(training_seed)  # ← Critical!
```

Otherwise, the training loader would use the forget_sampling_seed, breaking the reproducibility guarantee.

### Why Offset Seeds by 1000 and 2000?

Large offsets prevent accidental overlaps:
- If `run_id=999`: `training_seed=42`, `forget_sampling_seed=1041`, `unlearning_seed=2041`
- No seed ever equals another in different contexts
- Clear separation in seed space

## Reproducibility Guarantees

### ✓ Exact Reproducibility Within a Run
Same `run_id` + same config = exact same results:
```bash
python single_run_unlearning.py --run_id 0 --experiment_config config.json --gpu 0
python single_run_unlearning.py --run_id 0 --experiment_config config.json --gpu 0
# Results are bit-identical (if no GPU non-determinism)
```

### ✓ Identical Training, Different Forgetting
Comparing runs shows training consistency but forget variety:
```
Run 0: Training (IDENTICAL), Forget indices: [0, 2, 5, 7, 9]
Run 1: Training (IDENTICAL), Forget indices: [1, 3, 4, 6, 8]
Run 2: Training (IDENTICAL), Forget indices: [0, 1, 4, 7]
...
```

### ✓ Statistical Significance
Multiple runs with different forget sets and unlearning dynamics allow:
- Mean/std of unlearning effectiveness
- Confidence intervals on metrics
- Statistical hypothesis testing

## Verification Commands

### Test Seeding Strategy
```bash
# Run standalone test
cd tests
python3 << 'EOF'
import random

training_seed = 42

# Test 1: Training identical
random.seed(training_seed)
run1 = [random.random() for _ in range(5)]
random.seed(training_seed)
run2 = [random.random() for _ in range(5)]
assert run1 == run2, "Training not identical!"
print("[PASS] Training randomness identical")

# Test 2: Forget sampling independent
forget_list = []
for i in range(3):
    seed = training_seed + 1000 + i
    random.seed(seed)
    forget_list.append(tuple(random.sample(range(10), 5)))
assert len(set(forget_list)) == 3, "Forget sampling not independent!"
print("[PASS] Forget sampling independent")

print("All seeding tests passed!")
EOF
```

### Check Run Metadata
```bash
# View seeds for a specific run
cat runs_finetune/run_0/config.json | grep seed

# Expected output:
# "training_seed": 42,
# "forget_sampling_seed": 1000,
# "unlearning_seed": 2000,
```

### Compare Runs
```bash
# Compare forget indices across runs
for i in {0..4}; do
    echo "Run $i:"
    cat runs_finetune/run_$i/metrics.json | grep forget_indices
done
```

## Troubleshooting

### Problem: All runs have identical forget indices
**Cause**: Forget seed not being applied before sampling
**Fix**: Check `single_run_unlearning.py` line ~434 to ensure `np.random.seed(forget_sampling_seed)` is called

### Problem: Training metrics differ across runs
**Cause**: Training seed not being restored in time
**Fix**: Check training seed is restored after forget sampling (line ~443)

### Problem: Results not reproducible on reruns
**Cause**: Non-deterministic GPU operations
**Fix**: Add to config:
```json
"deterministic": true,
"seed": 42
```
And set environment variables:
```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=42
```

## Config File Setup

### experiment_config_finetune_retain.json
```json
{
  "training": {
    "seed": 42
  },
  ...
}
```
The base seed is defined here.

### experiment_config_ascent_descent.json
Same structure - the base seed applies to all strategies.

## Running Multi-Run Experiments

### Command Format
```bash
python run_orchestrator.py \
  --experiment_config configs/experiment_config_finetune_retain.json \
  --num_gpus 2 \
  --max_runs_per_gpu 3
```

The orchestrator automatically:
1. Uses fixed training seed for all runs
2. Computes unique forget_sampling_seed for each run
3. Computes unique unlearning_seed for each run
4. Launches runs in parallel with GPU scheduling

## Seed Offset Math

For `base_seed=42` with 50 runs:

```
Run  | training | forget_sampling | unlearning
-----|----------|-----------------|----------
  0  |    42    |      1042      |    2042
  1  |    42    |      1043      |    2043
  2  |    42    |      1044      |    2044
 ...
 49  |    42    |      1091      |    2091
```

All training seeds are identical (42), ensuring identical learning dynamics.

## Summary Checklist

- [x] Training seed is FIXED across all runs
- [x] Forget sampling seed is DIFFERENT per run
- [x] Unlearning seed is DIFFERENT per run
- [x] Training seed restored after forget sampling
- [x] No seed overlaps possible
- [x] Documented in run config.json
- [x] Logged to run.log for verification

