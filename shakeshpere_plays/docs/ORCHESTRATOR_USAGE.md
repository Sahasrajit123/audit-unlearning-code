# Multi-Run Orchest rator Usage

## Command Format

### Single GPU (Default)
```bash
python run_orchestrator.py \
  --experiment_config configs/experiment_config_finetune_retain.json
```
Uses GPU 0 with max 2 concurrent runs per GPU.

### Multiple GPUs (Explicit)
```bash
python run_orchestrator.py \
  --experiment_config configs/experiment_config_finetune_retain.json \
  --gpus 0,1,2 \
  --max_runs_per_gpu 3
```
Uses GPUs 0, 1, 2 with max 3 concurrent runs per GPU = 9 total concurrent runs.

### GPUs with Space Separation
```bash
python run_orchestrator.py \
  --experiment_config configs/experiment_config_finetune_retain.json \
  --gpus "1 3 4"
```
Uses GPUs 1, 3, 4 (you can skip GPUs, e.g., skip GPU 0, 2).

## Parameters

- `--experiment_config` (required): Path to experiment config JSON
- `--gpus` (optional, default: "0"): GPU IDs to use (comma or space separated)
  - Examples: `0`, `0,1`, `0,1,2`, `"1 3 5"`, `2,5,6,7`
- `--max_runs_per_gpu` (optional, default: 2): Max concurrent runs per GPU

## Examples

### Example 1: 2 GPUs, 3 runs per GPU (6 concurrent)
```bash
python run_orchestrator.py \
  --experiment_config configs/experiment_config_finetune_retain.json \
  --gpus 0,1 \
  --max_runs_per_gpu 3
```
Terminal output:
```
Available GPUs: [0, 1]
Max runs per GPU: 3
Max concurrent runs: 6

[RUN 0] LAUNCHED on GPU 0
[RUN 1] LAUNCHED on GPU 1
[RUN 2] LAUNCHED on GPU 0
[RUN 3] LAUNCHED on GPU 1
[RUN 4] LAUNCHED on GPU 0
[RUN 5] LAUNCHED on GPU 1
[Status] Launched 10/50 runs | GPU0: 3 jobs, GPU1: 3 jobs

[RUN 0] FINISHED SUCCESS
[RUN 1] FINISHED SUCCESS
...
[Progress] 5/50 completed | Succeeded: 5, Failed: 0 | Elapsed: 123.4s
```

### Example 2: 4 GPUs, 2 runs per GPU (8 concurrent)
```bash
python run_orchestrator.py \
  --experiment_config configs/experiment_config_ascent_descent.json \
  --gpus 0,1,2,3
```

### Example 3: Specific GPUs (skip some)
```bash
python run_orchestrator.py \
  --experiment_config configs/experiment_config_finetune_retain.json \
  --gpus 1,3,5 \
  --max_runs_per_gpu 2
```
Uses only GPUs 1, 3, 5 (skips 0, 2, 4).

## Output Structure

Each run directory contains:
```
runs_finetune/run_0/
├── config.json           ← Seeds, strategy, forget_prob
├── run.log               ← Detailed training/unlearning log
├── orchestrator.log      ← Subprocess output (if any)
├── model_trained.pt      ← Model after training phase
├── model_unlearnt.pt     ← Final unlearned model
└── metrics.json          ← Loss, accuracy, perplexity, etc.
```

## Analyzing Results
```bash
python scripts/analyze_multi_run_results.py --run_folder runs_finetune
```

## Viewing Individual Logs
```bash
tail -f runs_finetune/run_0/run.log
cat runs_finetune/run_5/metrics.json
```
