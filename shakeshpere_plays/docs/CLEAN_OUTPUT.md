# Clean Output Configuration

## Terminal Output (run_orchestrator.py)

The orchestrator now prints only essential information in a clean format:

```
================================================================================
GPU-AWARE ORCHESTRATOR
================================================================================

Config: configs/experiment_config_finetune_retain.json
Strategy: finetune_retain
Total runs: 50
Available GPUs: 2
Max runs per GPU: 3
Max concurrent runs: 6
Output folder: runs_finetune

================================================================================
Launching runs...
================================================================================

[RUN 0] LAUNCHED on GPU 0
[RUN 1] LAUNCHED on GPU 1
[RUN 2] LAUNCHED on GPU 0
...
[Status] Launched 10/50 runs | GPU loads: {0: 3, 1: 3}

[RUN 0] FINISHED SUCCESS
[RUN 1] FINISHED SUCCESS
[RUN 2] FINISHED FAILED (exit code: 1)
...
[Progress] 5/50 completed | Succeeded: 5, Failed: 0 | Elapsed: 123.4s
[Progress] 10/50 completed | Succeeded: 10, Failed: 0 | Elapsed: 246.8s

All runs launched. Waiting for completion...

================================================================================
ORCHESTRATOR FINAL REPORT
================================================================================

Total runs requested: 50
Completed successfully: 48
Failed: 2
Total time: 1234.5s (0.34h)

Results folder: runs_finetune
Run logs: runs_*/run_N/orchestrator.log
Metrics: runs_*/run_N/metrics.json

To view results:
  python scripts/analyze_multi_run_results.py --run_folder runs_finetune

================================================================================
```

## Log Files (Each Run Directory)

Each run has TWO log files:

### 1. run.log (from single_run_unlearning.py)
```
runs_finetune/run_0/run.log

2026-04-08 12:30:45 [INFO] ================================================================================
2026-04-08 12:30:45 [INFO] RUN 0: Starting on cuda:0
2026-04-08 12:30:45 [INFO] ================================================================================
2026-04-08 12:30:45 [INFO]
2026-04-08 12:30:45 [INFO] Run Configuration:
2026-04-08 12:30:45 [INFO]   Strategy: finetune_retain
2026-04-08 12:30:45 [INFO]   Forget prob: 0.3
2026-04-08 12:30:45 [INFO]   Training seed: 42 (FIXED)
2026-04-08 12:30:45 [INFO]   Forget sampling seed: 1000
2026-04-08 12:30:45 [INFO]   Unlearning seed: 2000
2026-04-08 12:30:45 [INFO]
2026-04-08 12:30:45 [INFO] Dataset:
2026-04-08 12:30:45 [INFO]   Forget indices: [1, 3, 5]
2026-04-08 12:30:45 [INFO]   Combined train size: 234,567 chars
2026-04-08 12:30:47 [INFO]
2026-04-08 12:30:47 [INFO] Starting training...
2026-04-08 12:30:47 [INFO] ================================================================================
2026-04-08 12:30:47 [INFO] Strategy: finetune_retain - Starting training...
2026-04-08 12:30:47 [INFO] ================================================================================
2026-04-08 12:30:47 [INFO]
2026-04-08 12:30:47 [INFO] Phase 1 Config: epochs=15, lr=0.1, batch_size=256, weight_decay=0.0
2026-04-08 12:30:50 [INFO] Epoch   1 | TrLoss: 4.3234 | TrAcc: 0.1234 | VLoss: 4.1234 | VAcc: 0.1456 → SAVED
2026-04-08 12:31:03 [INFO] Epoch   2 | TrLoss: 3.9234 | TrAcc: 0.1834 | VLoss: 3.8234 | VAcc: 0.1856
...
2026-04-08 12:35:23 [INFO]
2026-04-08 12:35:23 [INFO] Phase 2: Finetuning on retain only...
2026-04-08 12:35:23 [INFO] Phase 2 Config: epochs=7, lr=0.05
2026-04-08 12:35:26 [INFO] Epoch   1 | TrLoss: 3.1234 | TrAcc: 0.2234 | VLoss: 3.0234 | VAcc: 0.2456 → SAVED
...
2026-04-08 12:37:45 [INFO] Final evaluation
2026-04-08 12:37:45 [INFO] Test loss:      3.1234
2026-04-08 12:37:45 [INFO] Test accuracy:  0.2345
2026-04-08 12:37:45 [INFO] Test perplexity: 22.84
2026-04-08 12:37:45 [INFO]
2026-04-08 12:37:45 [INFO] ================================================================================
2026-04-08 12:37:45 [INFO] RUN 0: COMPLETED
2026-04-08 12:37:45 [INFO] ================================================================================
2026-04-08 12:37:45 [INFO] Results: loss=3.1234, acc=0.2345, ppl=22.84
```

### 2. orchestrator.log (from run_orchestrator.py)
This captures the subprocess stdout/stderr and contains any unexpected output

### Other Files in Each Run:
```
runs_finetune/run_0/
├── config.json           ← Run configuration (seeds, strategy, etc.)
├── run.log               ← Main run log (from single_run_unlearning.py)
├── orchestrator.log      ← Subprocess output
├── model_trained.pt      ← Trained model (after phase 1 for finetune_retain)
├── model_unlearnt.pt     ← Unlearnt model (final model)
└── metrics.json          ← Run metrics (loss, accuracy, perplexity, etc.)
```

## Key Features

✓ **Clean Terminal**: Only shows launch/finish messages
✓ **All Details in Logs**: Every detail logged to `run.log`
✓ **Progress Tracking**: Summary every 5 completions
✓ **Error Handling**: Errors logged to run.log, not printed to terminal
✓ **Exit Codes**: Failed runs show exit code or error message
✓ **Final Report**: Clean summary at the end with results location

## Example Usage

```bash
# Start 50 runs on 2 GPUs (3 per GPU)
python run_orchestrator.py \
  --experiment_config configs/experiment_config_finetune_retain.json \
  --num_gpus 2 --max_runs_per_gpu 3

# Watch the clean output in terminal

# After completion, check individual run logs
cat runs_finetune/run_0/run.log
cat runs_finetune/run_5/run.log

# View failed run's error
cat runs_finetune/run_2/orchestrator.log
```

## Log Location

All logs are in the run directory:
- `runs_finetune/run_0/run.log` - Run 0 main log
- `runs_finetune/run_1/run.log` - Run 1 main log
- etc.
