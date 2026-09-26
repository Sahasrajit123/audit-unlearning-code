"""Integration test for the aggregation path, driven by synthetic losses.

Runs `scripts/aggregate_audit.py` as a subprocess against a fabricated output tree,
so the whole CLI path is exercised without torch or a GPU: calibration fitting ->
label-free prediction -> overlap -> epsilon bound -> plots.

Two regimes are checked:

* a large in/out gap, which should behave like the `noop` control (near-perfect
  overlap, a bound close to the m=20/L=10 ceiling);
* no gap at all, which must certify nothing.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from audit_tofu.manifest import build_manifest, save_manifest
from audit_tofu.run_manager import resolve_run_paths, save_json

REPO = Path(__file__).resolve().parent.parent
PYTHON = sys.executable


def _write_losses(root: Path, manifest: dict, method: str, gap: float, seed: int = 0):
    """Fabricate candidate losses with a controllable in/out separation.

    Each QA pair gets a fixed base difficulty (shared across runs, which is what makes
    per-QA calibration meaningful) and loses `gap` when its author was trained on.
    """
    rng = np.random.default_rng(seed)
    batch_ids = manifest["split"]["batch_ids"]
    qa_per = manifest["dataset"]["qa_per_author"]

    base = {
        (b, q): float(rng.uniform(1.0, 2.5))
        for b in batch_ids
        for q in range(qa_per)
    }

    for family in ("calibration", "evaluation"):
        fam = manifest["sign_vectors"][family]
        for run_id in fam["run_ids"]:
            signs = dict(zip(batch_ids, fam["vectors"][run_id]))
            records = []
            for b in batch_ids:
                author = manifest["split"]["batches"][b][0][0]
                for q in range(qa_per):
                    loss = base[(b, q)] + rng.normal(0, 0.05)
                    if signs[b] == 1:
                        loss -= gap
                    records.append({
                        "run_id": run_id,
                        "method": method,
                        "candidate_author": author,
                        "qa_id": q,
                        "batch_id": b,
                        "loss": loss,
                        "num_answer_tokens": 40,
                    })
            paths = resolve_run_paths(root, run_id)
            save_json(paths.method_dir(method) / "losses.json", {
                "run_id": run_id,
                "method": method,
                "split_hash": manifest["split_hash"],
                "records": records,
            })


def _setup(tmp_path: Path, gap: float, method: str = "noop") -> Path:
    root = tmp_path / "runs_root"
    manifest = build_manifest(
        num_candidate_authors=20,
        num_retain_authors=180,
        num_calibration_runs=20,
        num_evaluation_runs=10,
        split_seed=12345,
        qa_per_author=20,
    )
    save_manifest(manifest, root / "manifest.json", overwrite=True)
    _write_losses(root, manifest, method, gap)

    base = yaml.safe_load((REPO / "configs" / "base.yaml").read_text())
    base["experiment"]["output_root"] = str(root)
    base["experiment"]["manifest_path"] = str(root / "manifest.json")
    cfg_path = tmp_path / "test_config.yaml"
    cfg_path.write_text(yaml.safe_dump(base))
    return cfg_path


def _run_aggregate(cfg_path: Path, method: str, out_dir: Path, extra=()):
    cmd = [
        PYTHON, str(REPO / "scripts" / "aggregate_audit.py"),
        "--config", str(cfg_path),
        "--method", method,
        "--out_dir", str(out_dir),
        *extra,
    ]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)


def test_aggregation_end_to_end_with_strong_leakage(tmp_path):
    cfg = _setup(tmp_path, gap=0.8)
    out = tmp_path / "audit_out"
    proc = _run_aggregate(cfg, "noop", out)

    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"

    summary = json.loads((out / "audit_summary.json").read_text())
    assert summary["m"] == 20
    assert summary["Gamma"] == 20
    assert summary["L"] == 10
    assert summary["zeta"] == 0.05
    assert summary["delta"] == 0.0

    # A clear in/out gap should make the attack essentially perfect.
    h = summary["headline"]
    assert h["mean_overlap"] >= h["r"] - 0.5, h
    assert h["epsilon_lb_mean"] is not None
    assert h["epsilon_lb_mean"] > 1.0

    # The Lemma 4.1 halving must be visible in the output, and attributed.
    assert h["epsilon_lb_mean"] == pytest.approx(h["epsilon_ldp_lb_mean"] / 2.0)
    assert h["halving_applied_by"] in ("wrapper", "vendored_module")

    # The zCDP and GDP readings of the same overlaps travel with it, unhalved.
    assert h["rho_lb_mean"] is not None and h["rho_lb_mean"] > 0.0
    assert h["mu_lb_mean"] is not None and h["mu_lb_mean"] > 0.0
    assert h["eps_estimate_from_rho_mean"] > 0.0
    assert h["eps_estimate_from_mu_mean"] > 0.0
    assert h["conv_delta"] == 1e-3

    # Every r in the config's sweep is reported -- read from the config rather than
    # pinned here, so widening attack.r_values does not look like a regression.
    expected_r = sorted(yaml.safe_load(cfg.read_text())["attack"]["r_values"])
    assert sorted(int(k) for k in summary["per_r"]) == expected_r

    # At r=20 with a perfect attack we should sit at the documented ceiling.
    r20 = summary["per_r"]["20"]
    if r20["mean_overlap"] == 20.0:
        assert r20["epsilon_lb_mean"] == pytest.approx(6.589329567577806, rel=1e-6)

    # Diagnostics present but clearly separate from the bound.
    assert summary["diagnostics"]["per_qa_llr_accuracy"] > 0.9
    assert summary["frozen_r"] in expected_r

    for png in ("calibration_losses.png", "epsilon_vs_r.png",
                "lambda_distribution.png", "overlap_per_run.png"):
        assert (out / png).exists(), f"missing plot {png}"
        assert (out / png).stat().st_size > 1000


def test_aggregation_certifies_nothing_without_leakage(tmp_path):
    """No in/out gap -> overlap at chance -> no positive bound.

    This is the property that makes the audit a real test rather than a score.
    """
    cfg = _setup(tmp_path, gap=0.0)
    out = tmp_path / "audit_out_null"
    proc = _run_aggregate(cfg, "noop", out, extra=("--no_plots",))

    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    summary = json.loads((out / "audit_summary.json").read_text())

    for r, entry in summary["per_r"].items():
        # Overlap should hover around chance = r/2 ...
        assert abs(entry["mean_overlap"] - int(r) / 2) <= int(r) * 0.30, (r, entry)
        # ... and no positive epsilon, rho or mu should be certified.
        assert entry["epsilon_lb_mean"] is None, (r, entry["epsilon_lb_mean"])
        assert entry["rho_lb_mean"] is None, (r, entry["rho_lb_mean"])
        assert entry["mu_lb_mean"] is None, (r, entry["mu_lb_mean"])


def test_aggregation_reports_all_r_and_writes_detail(tmp_path):
    cfg = _setup(tmp_path, gap=0.5)
    out = tmp_path / "audit_out_detail"
    proc = _run_aggregate(cfg, "noop", out, extra=("--no_plots",))
    assert proc.returncode == 0, proc.stderr

    detail = json.loads((out / "per_r_detail.json").read_text())
    expected_r = sorted(yaml.safe_load(cfg.read_text())["attack"]["r_values"])
    assert sorted(int(k) for k in detail) == expected_r

    for r_str, entry in detail.items():
        r = int(r_str)
        assert len(entry["runs"]) == 10
        for run in entry["runs"]:
            assert 0 <= run["overlap_v"] <= r
            assert run["correct_positive"] + run["correct_negative"] == run["overlap_v"]
            assert run["guesses_per_side"] == r // 2
            assert len(run["predicted_positive"]) == r // 2
            assert len(run["predicted_negative"]) == r // 2
            assert len(run["true_sign_vector"]) == 20
            assert sum(1 for s in run["true_sign_vector"] if s == 1) == 10


def test_calibration_json_records_fit_diagnostics(tmp_path):
    cfg = _setup(tmp_path, gap=0.6)
    out = tmp_path / "audit_out_cal"
    proc = _run_aggregate(cfg, "noop", out, extra=("--no_plots",))
    assert proc.returncode == 0, proc.stderr

    cal = json.loads((out / "calibration.json").read_text())
    assert len(cal["gaussians"]) == 400, "20 authors x 20 QA pairs"
    assert cal["config"]["pool_variance"] == "author"
    assert cal["diagnostics"]["num_calibration_runs"] == 20

    for g in cal["gaussians"]:
        assert g["var_in"] >= cal["config"]["var_floor"]
        assert g["var_out"] >= cal["config"]["var_floor"]
        assert g["n_in"] + g["n_out"] == 20


def test_aggregate_fails_cleanly_on_split_hash_mismatch(tmp_path):
    """Losses produced against a different split must be refused, not blended."""
    cfg = _setup(tmp_path, gap=0.5)
    cfg_data = yaml.safe_load(cfg.read_text())
    root = Path(cfg_data["experiment"]["output_root"])

    p = resolve_run_paths(root, "calib_000").method_dir("noop") / "losses.json"
    payload = json.loads(p.read_text())
    payload["split_hash"] = "0" * 64
    p.write_text(json.dumps(payload))

    proc = _run_aggregate(cfg, "noop", tmp_path / "out_bad", extra=("--no_plots",))
    assert proc.returncode != 0
    assert "different split" in (proc.stdout + proc.stderr)


def test_aggregate_errors_when_no_runs_present(tmp_path):
    root = tmp_path / "empty_root"
    manifest = build_manifest(split_seed=1)
    save_manifest(manifest, root / "manifest.json", overwrite=True)

    base = yaml.safe_load((REPO / "configs" / "base.yaml").read_text())
    base["experiment"]["output_root"] = str(root)
    base["experiment"]["manifest_path"] = str(root / "manifest.json")
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump(base))

    proc = _run_aggregate(cfg, "noop", tmp_path / "out_empty", extra=("--no_plots",))
    assert proc.returncode != 0
    assert "no calibration losses" in (proc.stdout + proc.stderr)


def test_r_selection_maximizes_epsilon_not_overlap_margin(tmp_path):
    """Regression: r must be chosen by projected epsilon, not margin over chance.

    The normalized margin (V - r/2)/(r/2) is 1.0 for EVERY r under a perfect attack,
    so selecting on it resolves ties arbitrarily and previously picked r=4
    (epsilon 1.18) over r=20 (epsilon 6.59).
    """
    cfg = _setup(tmp_path, gap=0.5)
    out = tmp_path / "audit_out_rsel"
    proc = _run_aggregate(cfg, "noop", out, extra=("--no_plots",))
    assert proc.returncode == 0, proc.stderr

    summary = json.loads((out / "audit_summary.json").read_text())
    sel = summary["r_selection"]

    assert sel["objective"] == "epsilon_lb"

    # With a near-perfect attack the margin cannot discriminate ...
    margins = [v for v in sel["normalized_margin_by_r"].values() if v is not None]
    assert max(margins) - min(margins) < 1e-9, (
        "margins are degenerate here, which is exactly why epsilon is the objective"
    )
    # ... but the projected epsilon must be strictly increasing in r.
    # Note: JSON keys are strings, so sort numerically, not lexicographically.
    by_r = sel["projected_epsilon_lb_by_r"]
    eps = [by_r[k] for k in sorted(by_r, key=int)]
    assert all(e is not None for e in eps)
    assert eps == sorted(eps) and eps[0] < eps[-1], eps

    # So the largest r wins, and it is the one reported.
    assert sel["selected_r"] == 20
    assert summary["frozen_r"] == 20
    assert summary["headline"]["r"] == 20
    assert summary["headline"]["epsilon_lb_mean"] == pytest.approx(
        6.589329567577806, rel=1e-6
    )


def test_r_selection_falls_back_when_nothing_is_certified(tmp_path):
    """With no leakage no r certifies anything; selection must still return an r."""
    cfg = _setup(tmp_path, gap=0.0)
    out = tmp_path / "audit_out_rsel_null"
    proc = _run_aggregate(cfg, "noop", out, extra=("--no_plots",))
    assert proc.returncode == 0, proc.stderr

    sel = json.loads((out / "audit_summary.json").read_text())["r_selection"]
    assert all(v is None for v in sel["projected_epsilon_lb_by_r"].values())
    assert sel["selected_r"] == 20, "fall back to the largest attainable ceiling"


def test_r_selection_reports_the_projection_horizon(tmp_path):
    cfg = _setup(tmp_path, gap=0.4)
    out = tmp_path / "audit_out_rsel_h"
    proc = _run_aggregate(cfg, "noop", out, extra=("--no_plots",))
    assert proc.returncode == 0, proc.stderr
    sel = json.loads((out / "audit_summary.json").read_text())["r_selection"]
    assert sel["projected_at_L"] == 10, "projection should use the real L"
