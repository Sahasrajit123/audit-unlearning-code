#!/usr/bin/env python
"""
Emit the privacy-audit tables as LaTeX, all methods into one file: one table per method,
one row per unlearning learning rate, all three privacy notions side by side, with the
R/F/T accuracy of the audited checkpoint next to them.

The methods are reported at different checkpoints by default, because their checkpoint
grids do not line up: DELETE runs 20 unlearning epochs and snapshots every 3rd (so it
has no epoch 1), bad teacher runs 5 and snapshots every one, SCRUB runs 3. The defaults
(bad teacher at `unlearn_epoch_1`, everything else at its final `unlearned` model -- for
scrub_r that final model IS the rewound one) are therefore not the same point in
training. Each table states its own checkpoint in its caption, and rows across two
tables are not a like-for-like comparison unless their sources match.

Every number is read from the per-sweep `audit_bounds_<source>_<metric>.json` files that
run_cumulative_audit.py writes, so the tables can never drift from the audit outputs --
regenerate after any change to the bound math.

Usage:
    python -m analysis.make_comparison_table                        # k=1500, mean-v, every method
    python -m analysis.make_comparison_table --k 500 --test median
    python -m analysis.make_comparison_table --badteacher-source unlearned   # like-for-like end state
"""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # repo root

#: unlearning LR -> sweep directory, per method. Each method's base sweep ran at its own
#: paper default (DELETE 1e-3, bad teacher 3e-3), hence the asymmetric directory names.
SWEEPS = {
    "delete": {
        "3e-3": "runs/delete/cifar100_unlearn_lr_3e-3_bs_1",
        "1e-3": "runs/delete/cifar100_bs_1",
        "1e-4": "runs/delete/cifar100_unlearn_lr_1e-4_bs_1",
    },
    "badteacher": {
        "3e-3": "runs/badteacher/cifar100_bs_1",
        "1e-3": "runs/badteacher/cifar100_unlearn_rate_1e-3_bs_1",
        "1e-4": "runs/badteacher/cifar100_unlearn_rate_1e-4_bs_1",
    },
    # SCRUB has four points, not three: its base sweep ran at the paper's 5e-4 and the
    # other three were added on top of the same trained models (--reuse-trained-from), so
    # all four are paired run-for-run and differ only in unlearn_lr.
    "scrub": {
        "3e-3": "runs/scrub/cifar100_unlearn_lr_3e-3_bs_1",
        "1e-3": "runs/scrub/cifar100_unlearn_lr_1e-3_bs_1",
        "5e-4": "runs/scrub/cifar100_bs_1",
        "1e-4": "runs/scrub/cifar100_unlearn_lr_1e-4_bs_1",
    },
    # SCRUB+R, derived from the scrub sweeps above by derive_scrub_r.py: same trained
    # models, same trajectory, rewound to the selected epoch. So the scrub / scrub_r rows
    # are paired run-for-run too, and the only thing that differs is which checkpoint the
    # audit scores -- which is exactly the ablation the rewinding claim needs.
    "scrub_r": {
        "3e-3": "runs/scrub_r/cifar100_unlearn_lr_3e-3_bs_1",
        "1e-3": "runs/scrub_r/cifar100_unlearn_lr_1e-3_bs_1",
        "5e-4": "runs/scrub_r/cifar100_bs_1",
        "1e-4": "runs/scrub_r/cifar100_unlearn_lr_1e-4_bs_1",
    },
}

PRETTY = {"delete": r"\textsc{Delete}", "badteacher": "bad-teacher",
          "scrub": r"\textsc{Scrub}", "scrub_r": r"\textsc{Scrub}+R"}

FIELDS = ["eps_lb", "eps_lb_ldp", "rho_lb", "eps_estimate_from_rho", "mu_lb", "eps_estimate_from_mu"]

#: Distinguishes "this bound was computed and certified nothing" (JSON null -> 0, the
#: trivial bound) from "this bound is not in the file at all" (-> `--'). Every audit
#: output currently writes all six keys, so the sentinel only guards against older or
#: partial files.
MISSING = object()


def load_cell(sweep, source, metric, k, test):
    """The bound values for one (sweep, source, k, test), or None if not audited."""
    path = ROOT / sweep / f"audit_bounds_{source}_{metric}.json"
    if not path.exists():
        return None
    doc = json.loads(path.read_text())
    entry = doc.get("by_k", {}).get(str(k))
    if entry is None:
        return None
    suffix = {"mean": "avg", "median": "median"}[test]
    out = {f: entry["bounds"].get(f"{f}_{suffix}", MISSING) for f in FIELDS}
    out["v"] = entry["v_mean"] if test == "mean" else entry["v_median"]
    out["T"], out["r"], out["m"] = entry["T"], entry.get("r"), doc["m"]
    return out


def load_accuracy(sweep, source):
    """
    Mean retain / sampled-forget / test accuracy of the audited checkpoint -- the utility
    that goes with the bound.

    A bound is not readable without this: a mechanism that destroys the model attacks at
    chance and so reports an excellent (even null) epsilon, which is privacy from having
    no model rather than from unlearning.

    Three places the numbers can come from, cheapest first, because which one applies
    depends on the sweep's checkpoint and on whether it ran with --eval-every-unlearn-epoch:

      1. `sweep_summary.json`, at `source="unlearned"` -- always present, the final model.
      2. each run's `metrics.json` unlearn_history, at an intermediate `unlearn_epoch_N`
         -- present only for a sweep that evaluated every unlearning epoch (DELETE's did;
         bad-teacher's and SCRUB's did not).
      3. `checkpoint_accuracy_<source>.json`, written by eval_checkpoint_accuracy.py for
         exactly the case where neither of the above has it.

    Returns None when the accuracy simply is not available, which renders as `--` rather
    than as a wrong number.
    """
    if source == "unlearned":
        path = ROOT / sweep / "sweep_summary.json"
        if path.exists():
            rows = json.loads(path.read_text())
            if rows:
                mean = lambda s: sum(r["metrics_after"][s]["accuracy"] for r in rows) / len(rows)
                return {"retain": mean("retain"), "forget": mean("sampled_forget"),
                        "test": mean("test"), "n_runs": len(rows), "from": "sweep_summary"}

    if source.startswith("unlearn_epoch_"):
        epoch = int(source.rsplit("_", 1)[1])
        totals, n = {"retain": 0.0, "forget": 0.0, "test": 0.0}, 0
        for run_dir in sorted((ROOT / sweep).glob("run_*")) + \
                sorted((ROOT / sweep).glob("test_run/run_*")):
            metrics_path = run_dir / "metrics.json"
            if not metrics_path.exists():
                continue
            record = next((r for r in json.loads(metrics_path.read_text())["unlearn_history"]
                           if r.get("epoch") == epoch), None)
            if record is None or "retain_acc" not in record:
                totals = None
                break
            totals["retain"] += record["retain_acc"]
            totals["forget"] += record["sampled_forget_acc"]
            totals["test"] += record["test_acc"]
            n += 1
        if totals and n:
            return dict({k: v / n for k, v in totals.items()},
                        n_runs=n, **{"from": "per-epoch metrics"})

    cache = ROOT / sweep / f"checkpoint_accuracy_{source}.json"
    if cache.exists():
        doc = json.loads(cache.read_text())
        mean = doc["mean"]
        return {"retain": mean["retain"], "forget": mean["sampled_forget"],
                "test": mean["test"], "n_runs": doc["n_runs"], "from": "evaluated cache"}
    return None


def fmt(x):
    """
    Render one bound cell.

    `None` means the solver could not reject any positive value of the parameter -- the
    observation is consistent with a perfectly private mechanism -- so the only bound the
    audit certifies is the trivial one, and that is exactly $0$. It is printed as such:
    the quantity in this column is a lower bound, and zero is a valid lower bound, so
    there is nothing to withhold. (This also matches how $\\mu_{\\mathrm{LB}}$ is defined
    in the first place: sup of the feasible set union {0}, see
    cum_runs_eps_lab.py::_mu_lb_from_logq_gdp, which returns a hard 0.0 for this case.)

    `--' is reserved for a bound that is not in the audit output at all, so the two are
    never confused: 0 is a result, `--' is an absence.
    """
    if x is MISSING:
        return "--"
    if x is None:
        return "0.00"
    if isinstance(x, float) and math.isinf(x):
        return r"$\infty$"
    return f"{x:.2f}"


def rewind_histogram(sweep):
    """`{epoch: n_runs}` for a derived SCRUB+R sweep, or None -- see derive_scrub_r.py."""
    path = ROOT / sweep / "config.json"
    if not path.exists():
        return None
    doc = json.loads(path.read_text())
    return (doc.get("derived_from") or {}).get("rewind_epoch_histogram")


def describe_source(source, method=None):
    if source == "unlearned":
        # For SCRUB+R, `unlearned_model.pth` is whichever epoch the rewind selected, so
        # calling it "the final model" would be exactly wrong.
        if method == "scrub_r":
            return "the rewound model SCRUB+R selects (not necessarily the final epoch)"
        return "the final unlearned model"
    if source == "trained":
        return "the trained model, before unlearning"
    if source.startswith("unlearn_epoch_"):
        return f"unlearning epoch {source.rsplit('_', 1)[1]}"
    return source.replace("_", r"\_")


def render_table(method, source, k, test, metric, with_accuracy=False):
    """One method's table as a list of LaTeX lines, plus how many cells were filled."""
    cells = {lr: load_cell(sweep, source, metric, k, test)
             for lr, sweep in SWEEPS[method].items()}
    meta = next((c for c in cells.values() if c), None)
    if meta is None:
        return None, 0, cells

    accs = ({lr: load_accuracy(sweep, source) for lr, sweep in SWEEPS[method].items()}
            if with_accuracy else {})
    # Only widen the table if at least one row can actually fill the column.
    with_accuracy = with_accuracy and any(accs.values())

    vsym = r"\bar{v}" if test == "mean" else r"\tilde{v}"
    testword = "mean" if test == "mean" else "median"

    acc_col = "c" if with_accuracy else ""
    acc_head = r"R/F/T acc.\ (\%) & " if with_accuracy else ""
    acc_blank = " & " if with_accuracy else ""
    shift = 1 if with_accuracy else 0

    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\setlength{\tabcolsep}{6pt}",
        r"\begin{tabular}{lc" + acc_col + r"cccccc}", r"\toprule",
        r"Unlearn LR & " + acc_head + r"$" + vsym + r"$ (of $r$) & \multicolumn{2}{c}{$\varepsilon$-DP} & "
        r"\multicolumn{2}{c}{$\rho$-zCDP} & \multicolumn{2}{c}{$\mu$-GDP} \\",
        f"\\cmidrule(lr){{{3 + shift}-{4 + shift}}}"
        f"\\cmidrule(lr){{{5 + shift}-{6 + shift}}}"
        f"\\cmidrule(lr){{{7 + shift}-{8 + shift}}}",
        r" & " + acc_blank + r"& $\varepsilon_{\mathrm{LB}}$ & "
        r"$\varepsilon^{\mathrm{LDP}}_{\mathrm{LB}}$ & "
        r"$\rho_{\mathrm{LB}}$ & $\varepsilon(\rho_{\mathrm{LB}})$ & "
        r"$\mu_{\mathrm{LB}}$ & $\varepsilon(\mu_{\mathrm{LB}})$ \\",
        r"\midrule",
    ]
    filled = 0
    for lr in SWEEPS[method]:
        mant, exp = lr.split("e-")
        row = [f"${mant}\\times10^{{-{exp}}}$"]
        c = cells[lr]
        if with_accuracy:
            a = accs.get(lr)
            row.append("--" if a is None
                       else f"{a['retain']:.1f} / {a['forget']:.1f} / {a['test']:.1f}")
        if c is None:
            row += ["--"] * 7
        else:
            filled += 1
            # v alone is not readable without its null: the attacker makes r = 2k guesses
            # per run, so chance is r/2 and the useful figure is the fraction correct.
            # Every audit output on disk carries r, but guard anyway: a bound file
            # written before r was recorded would otherwise crash the whole table.
            row.append(f"{c['v']:.1f}" if not c.get("r")
                       else f"{c['v']:.1f} ({100.0 * c['v'] / c['r']:.1f}\\%)")
            row += [fmt(c[f]) for f in FIELDS]
        lines.append(" & ".join(row) + r" \\")

    caption = (
        f"Empirical privacy lower bounds for {PRETTY[method]} unlearning on CIFAR-100 at "
        f"{describe_source(source, method)}, as a function of the unlearning learning rate. "
        f"Bounds are derived from the {testword} overlap score ${vsym}$ over "
        f"$T={meta['T']}$ held-out target runs, each of which guesses the $k={k}$ "
        f"most-likely-in and $k$ most-likely-out of $m={meta['m']}$ candidate forget "
        f"points, i.e.\\ $r={meta['r']}$ guesses; ${vsym}$ counts the correct ones and "
        f"the bracketed figure is that as a percentage, against a chance level of "
        f"$50\\%$ ($r/2={meta['r'] // 2 if meta.get('r') else '-'}$). Confidence level 0.95. "
        f"$\\varepsilon_{{\\mathrm{{LB}}}}$ is a certified-unlearning lower bound at "
        f"$\\delta=0$, i.e.\\ half the audited local-DP bound "
        f"$\\varepsilon^{{\\mathrm{{LDP}}}}_{{\\mathrm{{LB}}}}$; $\\rho_{{\\mathrm{{LB}}}}$ "
        f"and $\\mu_{{\\mathrm{{LB}}}}$ carry their factor of $2$ internally and are not "
        f"halved. $\\varepsilon(\\rho_{{\\mathrm{{LB}}}})$ and "
        f"$\\varepsilon(\\mu_{{\\mathrm{{LB}}}})$ are forward conversions to "
        f"$(\\varepsilon,\\delta)$ at $\\delta=0.001$; they are estimates, not lower "
        f"bounds on $\\varepsilon$. The $\\rho$ columns use the local RDP bound "
        f"$\\varepsilon^{{\\mathrm{{loc}}}}_\\gamma(\\rho)=4\\rho\\gamma$."
    )
    if with_accuracy:
        n_runs = next((a["n_runs"] for a in accs.values() if a), None)
        caption += (
            f" R/F/T are the mean retain / sampled-forget / test accuracy of the audited "
            f"checkpoint over all {n_runs} runs of each sweep, and are reported alongside the "
            f"bounds because the two cannot be read apart: a model whose utility has "
            f"collapsed is attacked at chance and so reports an excellent $\\varepsilon$, "
            f"which is privacy from having no model rather than from unlearning."
        )
    if method == "scrub_r" and source == "unlearned":
        hists = {lr: rewind_histogram(sweep) for lr, sweep in SWEEPS[method].items()}
        shown = ", ".join(
            f"{lr}: " + "/".join(f"{n}@ep{e}" for e, n in sorted(h.items()))
            for lr, h in hists.items() if h
        )
        if shown:
            caption += (
                f" Each row scores the rewound checkpoint, which is chosen per run, so the "
                f"audited model is not a single epoch: the selected epochs are ({shown}). "
                f"Rows where every run kept the final epoch are identical to the "
                f"corresponding {PRETTY['scrub']} row by construction."
            )
    if any(c is None for c in cells.values()):
        caption += ""
    if any(c is not None and (c["eps_lb"] is None or c["mu_lb"] == 0.0)
           for c in cells.values()):
        caption += (
            " A bound of $0.00$ marks a row where the attack did no better than guessing: "
            "the test cannot reject any positive value of the parameter, so the only bound "
            "the audit certifies is the trivial one. Read it as ``no violation detected'', "
            "not as evidence that the mechanism is private -- every column here is a lower "
            "bound, and an audit of this kind cannot produce an upper one. Such a row is "
            "always one whose utility has collapsed, which is why the accuracy column "
            "belongs next to it."
        )
    if any(c is None for c in cells.values()):
        caption += " `--' marks a cell with no audit available."

    lines += [
        r"\bottomrule", r"\end{tabular}",
        f"\\caption{{{caption}}}",
        f"\\label{{tab:{method}-lb-k{k}-{source.replace('_', '-')}-{test}}}",
        r"\end{table}",
    ]
    return lines, filled, cells


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--k", type=int, default=1500)
    p.add_argument("--test", choices=["mean", "median"], default="mean")
    p.add_argument("--delete-source", default="unlearned",
                   help="Checkpoint for the DELETE table (default: unlearned, i.e. epoch 20)")
    p.add_argument("--badteacher-source", default="unlearn_epoch_1",
                   help="Checkpoint for the bad-teacher table (default: unlearn_epoch_1)")
    p.add_argument("--scrub-source", default="unlearned",
                   help="Checkpoint for the SCRUB table (default: unlearned, i.e. epoch 3)")
    p.add_argument("--scrub-r-source", default="unlearned",
                   help="Checkpoint for the SCRUB+R table. Default: unlearned, which for a "
                        "derived scrub_r sweep IS the rewound model -- its "
                        "unlearned_model_epoch_N.pth files are the un-rewound SCRUB "
                        "trajectory instead.")
    p.add_argument("--methods", default=",".join(SWEEPS),
                   help="Comma-separated methods to emit, in order. Default: every known "
                        f"method ({','.join(SWEEPS)}), so a regeneration never silently "
                        "drops a table the file already had. A method you name explicitly "
                        "must have audits; one that only comes from this default is "
                        "skipped with a warning when it has none.")
    p.add_argument("--accuracy", action=argparse.BooleanOptionalAction, default=True,
                   help="R/F/T accuracy column (retain / sampled-forget / test, mean over "
                        "the sweep's runs), on by default because an epsilon cannot be read "
                        "without the utility it came with. A cell shows `--' when the "
                        "accuracy of that checkpoint is not on disk -- see "
                        "eval_checkpoint_accuracy.py, which computes it.")
    p.add_argument("--metric", default="phi")
    p.add_argument("--out", default=None,
                   help="Default: tables/lower_bounds_k<k>_<test>.tex (both tables in one file)")
    args = p.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = [m for m in methods if m not in SWEEPS]
    if unknown:
        raise SystemExit(f"unknown method(s) {unknown}; known: {sorted(SWEEPS)}")

    sources = {"delete": args.delete_source, "badteacher": args.badteacher_source,
               "scrub": args.scrub_source, "scrub_r": args.scrub_r_source}
    # Naming a method is a claim that it belongs in the file; inheriting it from the
    # default is not, so only the former is worth failing over.
    explicit = any(a == "--methods" or a.startswith("--methods=") for a in sys.argv)
    blocks, total = [], 0
    for method in methods:
        lines, filled, cells = render_table(method, sources[method], args.k,
                                            args.test, args.metric,
                                            with_accuracy=args.accuracy)
        if lines is None:
            msg = (f"no audited cells for {method} at source={sources[method]}, k={args.k}; "
                   f"run run_cumulative_audit.py for that source/k first")
            if explicit:
                raise SystemExit(msg)
            print(f"{method}: skipped -- {msg}")
            continue
        blocks.append("\n".join(lines))
        total += filled
        print(f"{method} @ {sources[method]}  ({filled}/{len(cells)} rows)")
        for lr, c in cells.items():
            s = "--" if c is None else (f"v={c['v']:8.1f}  eps_lb={fmt(c['eps_lb']):>7}  "
                                        f"rho={fmt(c['rho_lb']):>7}  eps(rho)={fmt(c['eps_estimate_from_rho']):>7}  "
                                        f"mu={fmt(c['mu_lb']):>6}")
            print(f"    lr {lr:<5} {s}")

    out = Path(args.out) if args.out else ROOT / "tables" / f"lower_bounds_k{args.k}_{args.test}.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    header = (f"% Generated by make_comparison_table.py -- do not edit by hand.\n"
              f"% k={args.k}, {args.test}-v test, metric={args.metric}\n"
              + "".join(f"% {m} @ {sources[m]}\n" for m in methods))
    if len(set(sources[m] for m in methods)) > 1:
        header += ("% NOTE: these tables are NOT all at the same checkpoint (see the per-method\n"
                   "% lines above). The unlearning-epoch grids differ: DELETE snapshots every 3rd\n"
                   "% of 20 epochs (so it has no epoch 1), bad teacher every 1 of 5, SCRUB every\n"
                   "% 1 of 3. Compare rows across tables only where the sources agree.\n")
    out.write_text(header + "\n" + "\n\n".join(blocks) + "\n")
    print(f"\nwrote {out}  ({total} rows across {len(blocks)} table(s))")


if __name__ == "__main__":
    main()
