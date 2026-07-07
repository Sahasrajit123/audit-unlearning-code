# Hessian Unlearning

Two main scripts: `train_and_unlearn.py` to run train+unlearn simulations, and `audit_utils.py` to compute certified epsilon bounds.

This code is mainly taken from this [repo](https://github.com/zhangbinchi/certified-deep-unlearning) with slight additions for auditing algorithms.

---

## 1. Running simulations — `train_and_unlearn.py`

Trains a model and applies Hessian-based unlearning across multiple runs. Results are saved under `--run-dir`.

### Using a config file (recommended)

```bash
python train_and_unlearn.py \
    --config configs/config_example_cosine_cifar100_shuffle_once.json \
    --run-dir runs_cifar100_sgd_cosine \
    --data-dir data/cifar100_batches
```

The config file sets optimizer, scheduler, model, dataset, epochs, etc. CLI flags override config values if both are provided.

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--config` | None | Path to JSON config file |
| `--run-dir` | *required* | Directory to save models and run outputs |
| `--data-dir` | *required* | Directory containing data (batch `.pkl` files) |
| `--num-runs` | 100 | Number of train+unlearn runs |
| `--start-run-idx` | 0 | Resume from this run index |
| `--model` | `tinynet` | Model architecture (`tinynet`, `tinynetcifar100`, `resnet18`, `allcnn`, `mlp`) |
| `--dataset` | `mnist` | Dataset name |
| `--num-classes` | None | Number of classes (inferred from dataset if not set) |
| `--epochs` | 400 | Training epochs |
| `--lr` | 0.01 | Learning rate |
| `--batch-size` | 128 | Training batch size |
| `--optimizer` | `adam` | Optimizer (`adam` or `sgd`) |
| `--scheduler-type` | None | LR scheduler (`cosine`, `step`, `plateau`, `onecycle`) |
| `--forget-prob` | 0.5 | Fraction of batches designated as forget set |
| `--s1` | 10 | Samples for Hessian approximation |
| `--s2` | 1000 | Taylor expansion order for Hessian approximation |
| `--std` | 0.001 | Gaussian noise standard deviation |
| `--gpus` | all | Comma-separated GPU IDs, e.g. `"0,1"` |
| `--max-workers-per-gpu` | 2 | Parallel workers per GPU |
| `--skip-existing-runs` | — | Skip runs that already have saved outputs |

### Example — CIFAR-100 with cosine schedule

```bash
python train_and_unlearn.py \
    --config configs/config_example_cosine_cifar100_shuffle_once.json \
    --run-dir runs_cifar100_sgd_cosine \
    --data-dir data/cifar100_batches \
    --num-runs 60 \
    --gpus "0,1" \
    --skip-existing-runs
```

### Config file reference (`configs/config_example_cosine_cifar100_shuffle_once.json`)

```json
{
  "optimizer": "sgd",
  "momentum": 0.0,
  "scheduler-type": "cosine",
  "scheduler-t-max": 400,
  "scheduler-eta-min": 0.0001,
  "lr": 0.1,
  "batch-size": 128,
  "epochs": 400,
  "weight-decay": 0.0005,
  "model": "tinynetcifar100",
  "dataset": "cifar100",
  "num-classes": 100,
  "seed": 42,
  "forget-prob": 0.5,
  "num-runs": 60,
  "shuffle-train-samples": true,
  "shuffle-train-batches-each-epoch": false
}
```

---

## 2. Auditing — `compute_eps_bounds_for_all_runs_batch_pointwise`

Computes certified epsilon lower bounds from saved run outputs. Import and call directly from Python or a notebook.

```python
from audit_utils import compute_eps_bounds_for_all_runs_batch_pointwise

results = compute_eps_bounds_for_all_runs_batch_pointwise(
    main_folder="runs_cifar100_sgd_cosine",   # top-level runs directory
    model_name="tinynetcifar100",              # must match what was used in training
    num_classes=100,
    filters=1.0,
    k=100,                                     # number of batches to predict as forgotten
    models_dir="models",                       # subdirectory inside each run folder
    device="cpu",
    use_trained_model=False,                   # False = use unlearned model
)

print("eps lower bound (avg):",    results["eps_lb_avg"])
print("eps lower bound (median):", results["eps_lb_median"])
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `main_folder` | *required* | Root directory of runs (e.g. `runs_cifar100_sgd_cosine`) |
| `model_name` | *required* | Model name string matching training (`tinynet`, `tinynetcifar100`, etc.) |
| `num_classes` | *required* | Number of output classes |
| `filters` | *required* | Filter width multiplier (use `1.0` for standard) |
| `k` | *required* | Top-k batches predicted as forgotten for computing overlap `v` |
| `models_dir` | `"models"` | Subfolder inside each run containing model checkpoints |
| `use_trained_model` | `False` | If `True`, audits the trained (non-unlearned) model as a baseline |
| `device` | `"cpu"` | `"cpu"` or `"cuda"` |
| `metric` | `"loss"` | Statistic to use: `"loss"` or `"phi"` |
| `delta` | `1e-8` | Delta parameter for epsilon bound computation |
| `confidence_level` | `0.95` | Confidence level for the bound |
| `auto_compute_missing_stats` | `True` | Compute missing per-point stats automatically if not cached |
| `verbose` | `False` | Print detailed per-run output |

### Return value

A dict with:
- `eps_lb_avg` — epsilon lower bound (average-based)
- `eps_lb_median` — epsilon lower bound (median-based)
- `v_list` — per-run overlap values
- `run_ids` — list of run identifiers processed
- `failed_runs` — runs that could not be processed
- `T`, `m`, `r` — audit parameters

### CIFAR-100 example

```python
results = compute_eps_bounds_for_all_runs_batch_pointwise(
    main_folder="runs_cifar100_sgd_cosine",
    model_name="tinynetcifar100",
    num_classes=100,
    filters=1.0,
    k=100,
    models_dir="models",
    device="cpu",
    use_trained_model=False,
)
```
