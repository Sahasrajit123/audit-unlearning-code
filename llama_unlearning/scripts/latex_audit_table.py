#!/usr/bin/env python3
"""Emit a LaTeX table of the three audit bounds at one (m, r), one row per method.

    # m=400, r=100, from the widened-grid aggregation
    .venv/bin/python scripts/latex_audit_table.py --m 400 --r 100 \
        --out runs/tofu_llama32_1b_audit_batching_qa/audit_r_grid/table_m400_r100.tex

    # any r in any tree, median statistic instead of the mean
    .venv/bin/python scripts/latex_audit_table.py --m 20 --r 20 --statistic median

Columns: the observed mean overlap, then

    eps_LB    certified-unlearning epsilon (LDP solution halved, Lemma 4.1)
    rho_LB    zCDP parameter                (NOT halved)
    eps(rho)  rho_LB converted to an (eps, conv_delta) pair
    mu_LB     GDP parameter                 (NOT halved)
    eps(mu)   mu_LB converted to an (eps, conv_delta) pair

The two conversions are for reading the numbers on one axis only: each maps a privacy
*guarantee* to a weaker guarantee, so applying it to a lower bound carries no bound
semantics. The caption says so, because a table outlives the conversation about it.

Rows are taken straight from the aggregations already on disk; nothing is recomputed
here, so the table cannot disagree with audit_summary.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

METHOD_LABELS = {
    "noop": r"\texttt{noop} (control)",
    "npo": r"\texttt{npo}",
    "retain_ft": r"\texttt{retain\_ft}",
    "grad_ascent": r"\texttt{grad\_ascent}",
    "grad_diff": r"\texttt{grad\_diff}",
    "simnpo": r"\texttt{simnpo}",
}
DEFAULT_METHODS = ["noop", "npo", "retain_ft", "grad_ascent", "grad_diff", "simnpo"]


def _find_audits(scan: List[str], m: int, r: int, tree: Optional[str]) -> Dict[str, Path]:
    """``{method: audit_summary.json}`` for every audit with this ``m`` and this ``r``.

    Matched on the stored ``m`` rather than on a directory name, so the table is tied to
    the audit's actual shape and not to an experiment naming convention.
    """
    found: Dict[str, Path] = {}
    for root in scan:
        for path in sorted(Path(root).rglob("audit_summary.json")):
            if tree and path.parents[1].name != tree:
                continue
            try:
                summary = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if int(summary.get("m", -1)) != m or str(r) not in summary.get("per_r", {}):
                continue
            method = summary.get("method") or path.parent.name
            # Prefer the widened grid when both trees carry the same (method, m, r).
            if method in found and path.parents[1].name != "audit_r_grid":
                continue
            found[method] = path
    return found


def _fmt(x: Any, nd: int = 3) -> str:
    if x is None:
        return "--"
    if isinstance(x, float) and x != x:      # NaN
        return "--"
    if x == float("inf"):
        return r"$\infty$"
    return f"{x:.{nd}f}"


def build_table(
    m: int,
    r: int,
    scan: List[str],
    tree: Optional[str],
    statistic: str,
    methods: List[str],
    label: str,
    nd: int,
) -> tuple:
    audits = _find_audits(scan, m, r, tree)
    if not audits:
        raise SystemExit(f"no audit found with m={m} and r={r} under {scan}")

    suffix = "mean" if statistic == "mean" else "median"
    rows, meta, sources = [], {}, {}
    for method in methods + [k for k in sorted(audits) if k not in methods]:
        path = audits.get(method)
        if path is None:
            continue
        summary = json.loads(path.read_text())
        entry = summary["per_r"][str(r)]
        meta = {
            "m": summary["m"], "L": summary["L"], "zeta": summary["zeta"],
            # per_r blocks written by aggregate_audit.py carry no conv_delta; the
            # headline does, and dp_bounds.json puts it in both.
            "conv_delta": entry.get("conv_delta")
            or summary.get("headline", {}).get("conv_delta"),
        }
        sources[method] = str(path)
        rows.append({
            "method": method,
            "mean_overlap": entry.get("mean_overlap"),
            "median_overlap": entry.get("median_overlap"),
            "epsilon_lb": entry.get(f"epsilon_lb_{suffix}"),
            "rho_lb": entry.get(f"rho_lb_{suffix}"),
            "mu_lb": entry.get(f"mu_lb_{suffix}"),
            # The conversions are only stored for the mean statistic in audit_summary.json.
            "eps_from_rho": entry.get("eps_estimate_from_rho_mean") if statistic == "mean" else None,
            "eps_from_mu": entry.get("eps_estimate_from_mu_mean") if statistic == "mean" else None,
            "source": str(path),
        })

    overlap_key = "mean_overlap" if statistic == "mean" else "median_overlap"
    overlap_head = r"$\bar V$" if statistic == "mean" else r"$\mathrm{med}\,V$"
    conv = meta.get("conv_delta")

    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \small",
        r"  \begin{tabular}{lrrrrrr}",
        r"    \toprule",
        r"    & & \multicolumn{1}{c}{pure DP} & \multicolumn{2}{c}{zCDP}"
        r" & \multicolumn{2}{c}{GDP} \\",
        r"    \cmidrule(lr){3-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}",
        r"    Unlearning method & " + overlap_head
        + r" & $\varepsilon_{\mathrm{LB}}$ & $\rho_{\mathrm{LB}}$"
          r" & $\varepsilon(\rho_{\mathrm{LB}})$ & $\mu_{\mathrm{LB}}$"
          r" & $\varepsilon(\mu_{\mathrm{LB}})$ \\",
        r"    \midrule",
    ]
    for row in rows:
        lines.append(
            "    {} & {} & {} & {} & {} & {} & {} \\\\".format(
                METHOD_LABELS.get(row["method"], r"\texttt{%s}" % row["method"].replace("_", r"\_")),
                _fmt(row[overlap_key], 1),
                _fmt(row["epsilon_lb"], nd),
                _fmt(row["rho_lb"], nd),
                _fmt(row["eps_from_rho"], nd),
                _fmt(row["mu_lb"], nd),
                _fmt(row["eps_from_mu"], nd),
            )
        )
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \caption{Audit lower bounds at $m = %d$, $r = %d$, $L = %s$ runs, confidence"
        r" $\zeta = %s$ (%s statistic). $\varepsilon_{\mathrm{LB}}$ is the"
        r" certified-unlearning $\varepsilon$, i.e.\ the LDP solution halved;"
        r" $\rho_{\mathrm{LB}}$ and $\mu_{\mathrm{LB}}$ are \emph{not} halved, because"
        r" the reduction through the reference law is already inside"
        r" $\varepsilon^{\mathrm{loc}}_\gamma(\rho)$ and $\mu_{\mathrm{loc}}(\mu) = 2\mu$."
        r" The $\varepsilon(\cdot)$ columns convert those two parameters to an"
        r" $(\varepsilon, \delta)$ pair at $\delta = %s$ so all five sit on one axis;"
        r" being guarantee-to-guarantee conversions, they are \emph{not} lower bounds on"
        r" $\varepsilon$. A dash means the observation is consistent with a perfectly"
        r" private mechanism at this confidence level.}"
        % (meta.get("m", m), r, meta.get("L", "?"), meta.get("zeta", "?"),
           statistic, ("%g" % conv) if conv is not None else "10^{-3}"),
        r"  \label{%s}" % label,
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n", rows, meta, sources


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--m", type=int, required=True, help="candidate-batch count to match")
    ap.add_argument("--r", type=int, required=True, help="guess budget (even) to tabulate")
    ap.add_argument("--scan", nargs="*", default=["runs"])
    ap.add_argument("--tree", default=None,
                    help="restrict to one aggregation dir, e.g. audit or audit_r_grid")
    ap.add_argument("--statistic", choices=["mean", "median"], default="mean")
    ap.add_argument("--methods", nargs="*", default=DEFAULT_METHODS)
    ap.add_argument("--label", default=None, help="LaTeX label (default tab:audit_m<m>_r<r>)")
    ap.add_argument("--decimals", type=int, default=3)
    ap.add_argument("--out", default=None, help="write the .tex here (also printed)")
    ap.add_argument("--json_out", default=None, help="write the same rows as json")
    args = ap.parse_args()

    label = args.label or f"tab:audit_m{args.m}_r{args.r}"
    tex, rows, meta, sources = build_table(
        args.m, args.r, args.scan, args.tree, args.statistic, args.methods,
        label, args.decimals,
    )

    print(tex)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(tex)
        print(f"% written to {args.out}", file=sys.stderr)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "m": args.m, "r": args.r, "statistic": args.statistic,
            "meta": meta, "rows": rows, "sources": sources,
        }, indent=2) + "\n")
        print(f"% rows written to {args.json_out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
