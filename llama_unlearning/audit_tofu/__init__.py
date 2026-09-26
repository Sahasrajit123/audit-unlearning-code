"""LLM-scale unlearning audit on TOFU with Llama-3.2-1B-Instruct.

Implements the repeated-run auditor of the accompanying paper, specialised to
TOFU: 20 candidate authors,
balanced sign vectors, batchwise in/out likelihood-ratio attack (paper §5
Instantiation I), and the mean-based epsilon lower bound of Lemma 4.2. The same
overlap scores are also turned into zCDP (``rho``) and Gaussian-DP (``mu``) lower
bounds in :mod:`audit_tofu.rho_mu_bounds`.

Only :mod:`audit_tofu.epsilon_bounds`, :mod:`audit_tofu.rho_mu_bounds`,
:mod:`audit_tofu.manifest`, :mod:`audit_tofu.attack` and
:mod:`audit_tofu.config` are importable without torch;
the training/scoring modules import torch lazily so the audit math stays testable on
a CPU-only box.
"""

__version__ = "0.1.0"

from . import attack, config, epsilon_bounds, manifest, rho_mu_bounds  # noqa: F401

__all__ = [
    "attack",
    "config",
    "epsilon_bounds",
    "manifest",
    "rho_mu_bounds",
    "__version__",
]
