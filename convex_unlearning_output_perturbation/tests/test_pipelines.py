"""End-to-end smoke tests: run each runner on a shrunken copy of a real config."""
import csv
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent


def _run(tmp_path, script, base_config, overrides):
    cfg = yaml.safe_load((ROOT / base_config).read_text())
    for section, values in overrides.items():
        if isinstance(values, dict):
            cfg.setdefault(section, {}).update(values)
        else:
            cfg[section] = values
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    proc = subprocess.run(
        [sys.executable, str(ROOT / script), "--config", str(cfg_path)],
        cwd=tmp_path, capture_output=True, text=True, timeout=900,
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    summaries = list(tmp_path.rglob("epsilon_bounds_summary.csv"))
    assert len(summaries) == 1
    with open(summaries[0]) as f:
        return list(csv.DictReader(f))


@pytest.mark.slow
def test_output_perturbation_pipeline(tmp_path):
    rows = _run(
        tmp_path, "run_output_perturbation_attack.py", "config/output_perturbation/config.yaml",
        {"config_id": "smoke",
         "data": {"n_total": 300, "n_forget": 100, "n_retain": 200, "n_val": 50, "n_test": 50},
         "sample_config": {"n_samples_per_dist": 100, "n_test": 100},
         "epsilon_values": [1.0, 50.0]},
    )
    assert [float(r["epsilon"]) for r in rows] == [1.0, 50.0]
    for r in rows:
        assert "rho_lb_conv" in r and "rho_ub_noise" in r
        # The audited lower bound can never exceed the noise's zCDP guarantee.
        assert float(r["rho_lb_conv"]) <= float(r["rho_ub_noise"])


@pytest.mark.slow
def test_newton_step_single_partition_pipeline(tmp_path):
    rows = _run(
        tmp_path, "run_privacy_attack_final.py", "config/convex_unlearning/config_01_f1.yaml",
        {"config_id": "smoke",
         "data": {"n_retain": 400, "n_forget": 50, "d": 5, "n_val": 50, "n_test": 50},
         "sample_config": {"n_samples_per_dist": 100, "n_test": 100},
         "epsilon_values": [1.0, 100.0]},
    )
    assert [float(r["epsilon"]) for r in rows] == [1.0, 100.0]
    assert all("rho_lb_pairwise" in r for r in rows)
