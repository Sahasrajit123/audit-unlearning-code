"""Shared fixtures. Nothing here requires torch."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audit_tofu.manifest import build_manifest
from audit_tofu.tofu_data import load_synthetic_examples

QA_PER_AUTHOR = 4


@pytest.fixture
def small_manifest():
    """m=4 candidate batches, 6 retain authors, 4 calibration + 2 evaluation runs.

    |S_4| = C(4,2) = 6, so 4 + 2 exactly saturates the supply of distinct balanced
    vectors. This is the tightest configuration the disjointness rule permits at m=4.
    """
    return build_manifest(
        dataset_name="__synthetic__",
        num_candidate_authors=4,
        num_retain_authors=6,
        num_calibration_runs=4,
        num_evaluation_runs=2,
        split_seed=99,
        qa_per_author=QA_PER_AUTHOR,
    )


@pytest.fixture
def audit_manifest():
    """The real audit shape: m=20, 180 retain, Gamma=20, L=10."""
    return build_manifest(
        num_candidate_authors=20,
        num_retain_authors=180,
        num_calibration_runs=20,
        num_evaluation_runs=10,
        split_seed=12345,
        qa_per_author=20,
    )


@pytest.fixture
def synthetic_examples():
    return load_synthetic_examples(10, QA_PER_AUTHOR)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "torch: requires torch/transformers (skipped when unavailable)"
    )
