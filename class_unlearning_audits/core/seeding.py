"""
Small helpers for keeping the *training* seed and the *unlearning* seed independent.

The pipeline in train_and_unlearn.py uses exactly two user-facing seeds:
  - `--seed`          : model weight initialization + training batch order (reproducible).
  - `--unlearn-seed`  : everything downstream of training -- which forget batches get
                        sampled, the incompetent teacher's random init, and the unlearning
                        dataloader's shuffle order. Independent of `--seed` by default.

`derive_seed` lets a single `--unlearn-seed` deterministically fan out into several
uncorrelated sub-seeds (one per random decision) without those sub-seeds colliding with
each other or requiring extra CLI flags.
"""
import hashlib
import os

import numpy as np
import torch

_SEED_MODULUS = 2 ** 31 - 1


def fresh_seed() -> int:
    """Draw an unpredictable seed from OS randomness (used when a seed isn't pinned)."""
    return int.from_bytes(os.urandom(8), "big") % _SEED_MODULUS


def derive_seed(base_seed: int, tag: str) -> int:
    """
    Deterministically derive a well-separated child seed from `base_seed` and a string
    `tag`. Same (base_seed, tag) always gives the same child seed; different tags give
    uncorrelated seeds even from the same base_seed.
    """
    digest = hashlib.sha256(f"{base_seed}:{tag}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % _SEED_MODULUS


def set_training_determinism(seed: int, device: torch.device):
    """
    Seed everything that affects model weight init and the training set's one-time batch
    shuffle (see train.train), and ask torch/cudnn for determinism. Shared by
    train_and_unlearn.py (single run) and run_sweep.py (many runs, same --seed reused
    across all of them) so both apply the exact same seeding contract.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except AttributeError:
            pass  # older torch versions don't have this
