#!/usr/bin/env bash
# One train+unlearn run on CIFAR-100, with per-epoch unlearning eval turned on so you can
# watch retain/forget/test accuracy move epoch-by-epoch during Stage 3 rather than only
# before/after.
#
# Hyperparameters come from configs/<method>.yaml -- each method's paper-tuned settings.
# Training is shortened to 60 epochs here so the demo finishes in reasonable time; the
# real sweeps use 200 (see configs/).
#
# Usage:
#   ./run_demo.sh                # DELETE (default)
#   ./run_demo.sh badteacher     # bad teacher
#   ./run_demo.sh scrub          # SCRUB      (scrub_r for SCRUB+R)
#   METHOD=badteacher ./run_demo.sh
set -e
cd "$(dirname "$0")/.."

METHOD="${1:-${METHOD:-delete}}"
PYTHON="${PYTHON:-python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"   # override to pick a different GPU

"$PYTHON" -m pipeline.train_and_unlearn \
  --config "configs/${METHOD}.yaml" \
  --epochs 60 \
  --lr 5e-3 --optimizer sgd \
  --forget-prob 0.5 \
  --seed 42 --unlearn-seed 42 \
  --eval-every-unlearn-epoch \
  --out-dir "runs/${METHOD}" --run-name demo

# Quick CPU/MNIST alternative if you just want to see the feature fast without
# waiting on CIFAR-100 training:
#   "$PYTHON" -m pipeline.train_and_unlearn --unlearn-method "$METHOD" \
#     --dataset mnist --data-dir data \
#     --epochs 20 --unlearn-epochs 10 \
#     --seed 1 --unlearn-seed 1 --no-cuda --eval-every-unlearn-epoch \
#     --out-dir runs/demo_mnist
