#!/usr/bin/env python3
"""
QUICK START - Multi-Run Unlearning Experiments

This script sets up everything and shows you how to run the experiments.
"""

import json
import subprocess
import sys
from pathlib import Path

def print_header(title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")

def main():
    print_header("MULTI-RUN UNLEARNING - QUICK START")

    # 1. Check files exist
    print("1️⃣  Checking required files...")
    required_files = [
        "single_run_unlearning.py",
        "run_orchestrator.py",
        "test_seeding.py",
        "experiment_config_finetune_retain.json",
        "data_loader.py",
        "trainer_utils.py",
    ]

    missing = []
    for fname in required_files:
        fpath = Path(fname)
        status = "✓" if fpath.exists() else "✗"
        print(f"  {status} {fname}")
        if not fpath.exists():
            missing.append(fname)

    if missing:
        print(f"\n❌ Missing files: {missing}")
        return 1

    print("\n✓ All required files present!")

    # 2. Test seeding
    print_header("2️⃣  Testing Seeding Strategy")
    print("Running: python test_seeding.py\n")
    result = subprocess.run([sys.executable, "test_seeding.py"], capture_output=False)
    if result.returncode != 0:
        print("\n❌ Seeding test failed!")
        return 1

    # 3. Show config files
    print_header("3️⃣  Available Configurations")

    configs = [
        "experiment_config_finetune_retain.json",
        "experiment_config_ascent_descent.json",
    ]

    for cfg_file in configs:
        cfg_path = Path(cfg_file)
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = json.load(f)
            exp = cfg.get("experiment", {})
            strategy = exp.get("strategy", "?")
            num_runs = exp.get("num_runs", "?")
            forget_prob = exp.get("forget_prob", "?")
            print(f"  📄 {cfg_file}")
            print(f"     Strategy: {strategy}")
            print(f"     Runs: {num_runs}, Forget prob: {forget_prob}")
            print()

    # 4. Show commands
    print_header("4️⃣  How to Run")

    commands = [
        ("Test single run (GPU 0)", [
            "python single_run_unlearning.py \\",
            "  --run_id 0 \\",
            "  --experiment_config experiment_config_finetune_retain.json \\",
            "  --gpu 0",
        ]),
        ("Run 50 finetune-retain experiments (2 GPUs, 3 per GPU)", [
            "python run_orchestrator.py \\",
            "  --experiment_config experiment_config_finetune_retain.json \\",
            "  --num_gpus 2 \\",
            "  --max_runs_per_gpu 3",
        ]),
        ("Run 50 ascent-descent experiments (4 GPUs, 2 per GPU)", [
            "python run_orchestrator.py \\",
            "  --experiment_config experiment_config_ascent_descent.json \\",
            "  --num_gpus 4 \\",
            "  --max_runs_per_gpu 2",
        ]),
    ]

    for title, cmd_parts in commands:
        print(f"  📌 {title}:")
        for part in cmd_parts:
            print(f"     {part}")
        print()

    # 5. Show seeding info
    print_header("5️⃣  Seeding Strategy (Critical!)")

    with open("experiment_config_finetune_retain.json") as f:
        base_seed = json.load(f)["training"]["seed"]

    print(f"  Base training seed: {base_seed} (FIXED for all runs)")
    print()
    print("  Per-run seeds:")
    for i in range(3):
        training_seed = base_seed
        forget_seed = base_seed + 1000 + i
        unlearn_seed = base_seed + 2000 + i
        print(f"    Run {i}:")
        print(f"      - Training: {training_seed} (identical across all runs ✓)")
        print(f"      - Forget sampling: {forget_seed} (different per run ✓)")
        print(f"      - Unlearning: {unlearn_seed} (different per run ✓)")

    # 6. Show output structure
    print_header("6️⃣  Output Structure")

    print("""  runs_finetune/
  ├── run_0/
  │   ├── run.log              ← ALL METRICS LOGGED HERE
  │   ├── config.json          ← Run config with seeds
  │   ├── metrics.json         ← Final metrics
  │   ├── model_trained.pt     ← After phase 1
  │   └── model_unlearnt.pt    ← After phase 2 (best model)
  ├── run_1/
  │   └── ...
  └── run_49/
      └── ...

  📊 NO TERMINAL OUTPUT - check run.log for details!
  """)

    # 7. Next steps
    print_header("7️⃣  Next Steps")

    print("""  1. For testing:
     $ python single_run_unlearning.py \\
         --run_id 0 \\
         --experiment_config experiment_config_finetune_retain.json \\
         --gpu 0
     $ tail -f runs_finetune/run_0/run.log

  2. For full experiments:
     $ python run_orchestrator.py \\
         --experiment_config experiment_config_finetune_retain.json \\
         --num_gpus 2 \\
         --max_runs_per_gpu 3

  3. To monitor progress:
     $ watch -n 5 'ls -la runs_finetune/ | tail -20'
     $ tail -f runs_finetune/run_*/run.log

  4. After completion:
     $ head runs_finetune/run_0/metrics.json
     $ cat runs_finetune/run_*/run.log | grep "Test "
  """)

    print_header("✅ READY TO START!")
    print("See MULTI_RUN_GUIDE.md for detailed documentation.\n")

    return 0

if __name__ == "__main__":
    sys.exit(main())
