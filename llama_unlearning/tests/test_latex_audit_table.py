"""The LaTeX audit table is a view, not a second computation.

``scripts/latex_audit_table.py`` must print exactly the numbers the aggregation stored:
a table that recomputes anything can silently disagree with ``audit_summary.json``, and
a table in a paper is the last place to discover that. These tests fabricate audits with
known values and check the rendered rows against them.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PYTHON = sys.executable
SCRIPT = REPO / "scripts" / "latex_audit_table.py"


def _fabricate(dir_: Path, method: str, m: int, r: int, values: dict, tree: str = "audit"):
    d = dir_ / tree / method
    d.mkdir(parents=True, exist_ok=True)
    (d / "audit_summary.json").write_text(json.dumps({
        "method": method,
        "m": m,
        "L": 10,
        "zeta": 0.05,
        "frozen_r": r,
        "headline": {"r": r, "conv_delta": 1e-3},
        "per_r": {str(r): {"v_list": [r] * 10, **values}},
    }))
    return d


_FULL = {
    "mean_overlap": 91.1,
    "median_overlap": 92.5,
    "epsilon_lb_mean": 19.4505,
    "epsilon_lb_median": 21.9675,
    "rho_lb_mean": 4.7546,
    "rho_lb_median": 2.6409,
    "mu_lb_mean": 4.1431,
    "mu_lb_median": 3.0468,
    "eps_estimate_from_rho_mean": 16.2165,
    "eps_estimate_from_mu_mean": 20.6427,
}


def _run(*args):
    return subprocess.run([PYTHON, str(SCRIPT), *args],
                          capture_output=True, text=True, cwd=REPO)


def test_table_prints_the_stored_numbers(tmp_path):
    _fabricate(tmp_path / "exp", "npo", 400, 100, _FULL)
    proc = _run("--m", "400", "--r", "100", "--scan", str(tmp_path))
    assert proc.returncode == 0, proc.stderr

    assert r"\texttt{npo} & 91.1 & 19.451 & 4.755 & 16.216 & 4.143 & 20.643 \\" in proc.stdout
    assert r"$m = 400$, $r = 100$, $L = 10$" in proc.stdout
    assert r"\zeta = 0.05" in proc.stdout
    assert r"\delta = 0.001" in proc.stdout
    # The caption must keep saying what the numbers mean.
    assert "not} halved" in proc.stdout
    assert r"\label{tab:audit_m400_r100}" in proc.stdout


def test_missing_bounds_render_as_dashes(tmp_path):
    """An audit that certified nothing must not render as 0.000."""
    nothing = dict(_FULL, epsilon_lb_mean=None, rho_lb_mean=None, mu_lb_mean=None,
                   eps_estimate_from_rho_mean=None, eps_estimate_from_mu_mean=None,
                   mean_overlap=55.2)
    _fabricate(tmp_path / "exp", "grad_ascent", 400, 100, nothing)
    proc = _run("--m", "400", "--r", "100", "--scan", str(tmp_path))
    assert proc.returncode == 0, proc.stderr
    assert r"\texttt{grad\_ascent} & 55.2 & -- & -- & -- & -- & -- \\" in proc.stdout


def test_rows_are_selected_by_m_and_r_not_by_directory_name(tmp_path):
    _fabricate(tmp_path / "exp_big", "npo", 400, 100, _FULL)
    _fabricate(tmp_path / "exp_small", "npo", 20, 20, dict(_FULL, epsilon_lb_mean=6.589))

    proc = _run("--m", "400", "--r", "100", "--scan", str(tmp_path))
    assert proc.returncode == 0, proc.stderr
    assert "19.451" in proc.stdout and "6.589" not in proc.stdout

    # An (m, r) that exists nowhere is an error, not an empty table.
    proc = _run("--m", "400", "--r", "42", "--scan", str(tmp_path))
    assert proc.returncode != 0
    assert "no audit found" in (proc.stdout + proc.stderr)


def test_widened_grid_wins_when_both_trees_have_the_same_m_and_r(tmp_path):
    """Both trees carry r=100 at m=400; the re-aggregation is the current one."""
    exp = tmp_path / "exp"
    _fabricate(exp, "npo", 400, 100, dict(_FULL, epsilon_lb_mean=1.111), tree="audit")
    _fabricate(exp, "npo", 400, 100, dict(_FULL, epsilon_lb_mean=2.222),
               tree="audit_r_grid")

    proc = _run("--m", "400", "--r", "100", "--scan", str(tmp_path))
    assert proc.returncode == 0, proc.stderr
    assert "2.222" in proc.stdout and "1.111" not in proc.stdout

    # ... and --tree pins it explicitly when you want the older one.
    proc = _run("--m", "400", "--r", "100", "--scan", str(tmp_path), "--tree", "audit")
    assert proc.returncode == 0, proc.stderr
    assert "1.111" in proc.stdout and "2.222" not in proc.stdout


def test_median_statistic_uses_the_median_columns(tmp_path):
    _fabricate(tmp_path / "exp", "npo", 400, 100, _FULL)
    proc = _run("--m", "400", "--r", "100", "--scan", str(tmp_path),
                "--statistic", "median")
    assert proc.returncode == 0, proc.stderr
    # median overlap 92.5, median bounds, and no conversions (only stored for the mean).
    assert r"\texttt{npo} & 92.5 & 21.968 & 2.641 & -- & 3.047 & -- \\" in proc.stdout
    assert r"$\mathrm{med}\,V$" in proc.stdout


def test_out_and_json_out_are_written(tmp_path):
    _fabricate(tmp_path / "exp", "npo", 400, 100, _FULL)
    tex, js = tmp_path / "t.tex", tmp_path / "t.json"
    proc = _run("--m", "400", "--r", "100", "--scan", str(tmp_path),
                "--out", str(tex), "--json_out", str(js))
    assert proc.returncode == 0, proc.stderr

    assert tex.read_text() == proc.stdout.rstrip("\n") + "\n"
    payload = json.loads(js.read_text())
    assert payload["m"] == 400 and payload["r"] == 100
    assert payload["meta"]["conv_delta"] == 1e-3
    assert payload["rows"][0]["epsilon_lb"] == 19.4505
    # The provenance of every row is recorded, so a number can be traced back.
    assert payload["rows"][0]["source"].endswith("audit_summary.json")
