"""Spec validation 8c: backfilling rho/mu onto audits that already finished.

Runs ``scripts/backfill_dp_bounds.py`` as a subprocess against a fabricated audit
tree. The property under test is that the backfill is a pure function of what is
already stored: it must reproduce the recorded epsilon exactly from ``v_list`` alone
(no losses, no calibration, no model in the tree at all), and add rho and mu beside it.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PYTHON = sys.executable
SCRIPT = REPO / "scripts" / "backfill_dp_bounds.py"

sys.path.insert(0, str(REPO))
from audit_tofu.epsilon_bounds import epsilon_lb_report  # noqa: E402


def _fabricate(root: Path, method: str, v_by_r: dict, m: int = 20, zeta: float = 0.05):
    """An ``audit_summary.json`` with the same shape aggregate_audit.py writes.

    Only the fields the backfill reads are filled in, and the epsilon values are the
    real ones for these overlaps -- so a mismatch in the backfill is a real mismatch,
    not an artefact of fabricated numbers.
    """
    per_r = {}
    for r, v_list in v_by_r.items():
        rep = epsilon_lb_report(m, r, v_list, zeta=zeta)
        per_r[str(r)] = {
            "v_list": v_list,
            "mean_overlap": sum(v_list) / len(v_list),
            "epsilon_lb_mean": rep["mean"]["epsilon_lb"],
            "epsilon_lb_median": rep["median"]["epsilon_lb"],
        }
    headline_r = max(v_by_r)
    d = root / "audit" / method
    d.mkdir(parents=True, exist_ok=True)
    (d / "audit_summary.json").write_text(json.dumps({
        "method": method,
        "m": m,
        "L": len(v_by_r[headline_r]),
        "zeta": zeta,
        "delta": 0.0,
        "frozen_r": headline_r,
        "headline": {"r": headline_r, "v_list": v_by_r[headline_r]},
        "per_r": per_r,
    }, indent=2))
    return d


def _run(*args):
    return subprocess.run([PYTHON, str(SCRIPT), *args],
                          capture_output=True, text=True, cwd=REPO)


def test_backfill_reproduces_epsilon_and_adds_rho_and_mu(tmp_path):
    d = _fabricate(tmp_path / "exp", "noop", {8: [8] * 10, 20: [19, 20, 20, 18, 20,
                                                                20, 19, 20, 20, 20]})
    proc = _run(str(d / "audit_summary.json"))
    assert proc.returncode == 0, proc.stderr

    out = json.loads((d / "dp_bounds.json").read_text())
    assert out["method"] == "noop"
    assert out["headline_r"] == 20
    assert set(out["per_r"]) == {"8", "20"}

    for r_key, block in out["per_r"].items():
        # The cross-check: epsilon recomputed from v_list alone matches what was stored.
        assert block["epsilon_reproduced"] is True, (r_key, block)
        assert block["epsilon_lb_mean"] == pytest.approx(
            block["epsilon_lb_mean_stored"], rel=1e-9
        )
        # ... and the two new bounds are there, positive, and unhalved.
        assert block["rho_lb_mean"] > 0.0
        assert block["mu_lb_mean"] > 0.0
        assert block["eps_estimate_from_rho_mean"] > 0.0
        assert block["conv_delta"] == 1e-3

    assert "NOT halved" in out["note"]


def test_backfill_needs_nothing_but_the_summary(tmp_path):
    """No losses.json, no calibration.json, no model -- the overlaps are sufficient."""
    d = _fabricate(tmp_path / "exp", "npo", {20: [17] * 10})
    assert list(d.iterdir()) == [d / "audit_summary.json"]

    proc = _run(str(d / "audit_summary.json"))
    assert proc.returncode == 0, proc.stderr
    assert (d / "dp_bounds.json").exists()
    assert "epsilon reproduced exactly" in proc.stdout


def test_dry_run_writes_nothing(tmp_path):
    d = _fabricate(tmp_path / "exp", "noop", {20: [18] * 10})
    proc = _run(str(d / "audit_summary.json"), "--dry_run")
    assert proc.returncode == 0, proc.stderr
    assert not (d / "dp_bounds.json").exists()
    assert "dry run: nothing written" in proc.stdout


def test_in_place_merges_into_the_summary_without_dropping_fields(tmp_path):
    d = _fabricate(tmp_path / "exp", "noop", {20: [20] * 10})
    before = json.loads((d / "audit_summary.json").read_text())

    proc = _run(str(d / "audit_summary.json"), "--in_place")
    assert proc.returncode == 0, proc.stderr

    after = json.loads((d / "audit_summary.json").read_text())
    # Existing content survives ...
    for key, value in before["per_r"]["20"].items():
        assert after["per_r"]["20"][key] == value
    assert after["headline"]["r"] == before["headline"]["r"]
    # ... and the new fields are present in both places.
    for key in ("rho_lb_mean", "mu_lb_mean", "eps_estimate_from_rho_mean"):
        assert key in after["headline"]
        assert key in after["per_r"]["20"]
    assert after["dp_bounds_backfilled_from"] == "dp_bounds.json"


def test_chance_level_audit_backfills_to_no_bounds(tmp_path):
    """An audit that certified nothing must not acquire a rho or mu out of nowhere."""
    d = _fabricate(tmp_path / "exp", "grad_ascent", {20: [10, 12, 8, 10, 11, 9, 10,
                                                          10, 12, 8]})
    proc = _run(str(d / "audit_summary.json"))
    assert proc.returncode == 0, proc.stderr
    block = json.loads((d / "dp_bounds.json").read_text())["per_r"]["20"]
    assert block["epsilon_lb_mean"] is None
    assert block["rho_lb_mean"] is None
    assert block["mu_lb_mean"] is None
    assert block["epsilon_reproduced"] is True


def test_scan_finds_every_method_under_a_tree(tmp_path):
    root = tmp_path / "runs"
    for method in ("noop", "npo", "retain_ft"):
        _fabricate(root / "exp_a", method, {20: [19] * 10})
    _fabricate(root / "exp_b", "noop", {20: [15] * 10})

    proc = _run("--scan", str(root), "--csv", str(tmp_path / "all.csv"))
    assert proc.returncode == 0, proc.stderr
    assert "4 audit(s)" in proc.stdout

    rows = (tmp_path / "all.csv").read_text().strip().splitlines()
    assert len(rows) == 5  # header + one row per (audit, r)
    assert rows[0].startswith("experiment,audit_tree,method,r,")


def test_scan_finds_audits_at_any_depth(tmp_path):
    """Output trees get moved and nested, so discovery must not assume a fixed depth.

    A too-specific glob would leave the deeper audit with only its epsilon bound and
    say nothing about it.
    """
    root = tmp_path / "runs"
    _fabricate(root / "exp_shallow", "noop", {20: [19] * 10})
    _fabricate(root / "archive" / "2026" / "exp_deep", "npo", {20: [18] * 10})

    proc = _run("--scan", str(root))
    assert proc.returncode == 0, proc.stderr
    assert "2 audit(s)" in proc.stdout
    assert (root / "exp_shallow" / "audit" / "noop" / "dp_bounds.json").exists()
    assert (root / "archive" / "2026" / "exp_deep" / "audit" / "npo"
            / "dp_bounds.json").exists()


def test_scan_reports_run_trees_that_have_no_audit_to_backfill(tmp_path):
    """Folders with runs but no audit_summary.json need aggregate_audit.py instead.

    Reporting them is the difference between "42 audits backfilled" and knowing
    whether 42 was all of them.
    """
    root = tmp_path / "runs"
    _fabricate(root / "done", "noop", {20: [19] * 10})
    (root / "done" / "manifest.json").write_text("{}")

    # Has run losses but was never aggregated -> aggregatable.
    never = root / "never_aggregated"
    (never / "runs" / "run_000" / "methods" / "noop").mkdir(parents=True)
    (never / "manifest.json").write_text("{}")
    (never / "runs" / "run_000" / "methods" / "noop" / "losses.json").write_text("{}")

    # Manifest only -> nothing to do at all.
    (root / "manifest_only").mkdir(parents=True)
    (root / "manifest_only" / "manifest.json").write_text("{}")

    proc = _run("--scan", str(root), "--dry_run")
    assert proc.returncode == 0, proc.stderr
    assert "no audit_summary.json" in proc.stdout
    assert "never_aggregated" in proc.stdout and "CAN be aggregated" in proc.stdout
    assert "manifest_only" in proc.stdout and "manifest/reference only" in proc.stdout
    # The already-backfilled tree is not listed as missing.
    assert "done " not in proc.stdout.split("no audit_summary.json")[1]


def test_multiple_configs_can_be_passed(tmp_path):
    """--config takes a list, so one invocation covers a whole campaign."""
    import yaml

    paths = []
    for name in ("exp_one", "exp_two"):
        d = _fabricate(tmp_path / name, "noop", {20: [19] * 10})
        cfg = tmp_path / f"{name}.yaml"
        cfg.write_text(yaml.safe_dump({"experiment": {"output_root": str(d.parents[1])}}))
        paths.append((cfg, d))

    proc = _run("--config", str(paths[0][0]), str(paths[1][0]))
    assert proc.returncode == 0, proc.stderr
    assert "2 audit(s)" in proc.stdout
    for _, d in paths:
        assert (d / "dp_bounds.json").exists()


def test_zeta_override_changes_the_bounds(tmp_path):
    d = _fabricate(tmp_path / "exp", "noop", {20: [19] * 10}, zeta=0.05)
    assert _run(str(d / "audit_summary.json")).returncode == 0
    loose = json.loads((d / "dp_bounds.json").read_text())["per_r"]["20"]

    assert _run(str(d / "audit_summary.json"), "--zeta", "0.001").returncode == 0
    tight = json.loads((d / "dp_bounds.json").read_text())["per_r"]["20"]

    assert tight["zeta"] == 0.001
    assert tight["rho_lb_mean"] < loose["rho_lb_mean"]
    assert tight["mu_lb_mean"] < loose["mu_lb_mean"]
    # The stored epsilon was computed at zeta=0.05, so the cross-check must now fail
    # rather than silently claiming agreement.
    assert tight["epsilon_reproduced"] is False
