"""
Turn the JSONs mia_basic.py writes into the LaTeX table of basic-MIA accuracies.

Same contract as make_comparison_table.py: a pure reader, no model is loaded and no sweep
artifact is touched. Every number comes from `<sweep>/mia_basic_<source>.json` (written by
mia_basic.py) and `<sweep>/checkpoint_accuracy_<source>.json` / `sweep_summary.json` /
per-run `metrics.json` for the R/F/T column.

The per-method sources deliberately match make_comparison_table.py's defaults -- bad
teacher at `unlearn_epoch_1`, everything else at its final `unlearned` model -- so the
rows can be read against the epsilon/rho/mu tables. `--badteacher-source unlearned`
switches to the like-for-like end state, exactly as it does there.

Usage:
    python -m analysis.make_mia_table --out tables/mia_basic.tex
    python -m analysis.make_mia_table --badteacher-source unlearned \
        --out tables/mia_basic_badteacher_unlearned.tex
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent  # repo root

#: (method label, unlearn-LR label, sweep dir). The same sweeps make_comparison_table.py
#: reads, in the same order, so the two tables line up row for row.
SWEEPS = [
    (r"\textsc{Delete}", "3e-3", "runs/delete/cifar100_unlearn_lr_3e-3_bs_1"),
    (r"\textsc{Delete}", "1e-3", "runs/delete/cifar100_bs_1"),
    (r"\textsc{Delete}", "1e-4", "runs/delete/cifar100_unlearn_lr_1e-4_bs_1"),
    ("Bad teacher", "3e-3", "runs/badteacher/cifar100_bs_1"),
    ("Bad teacher", "1e-3", "runs/badteacher/cifar100_unlearn_rate_1e-3_bs_1"),
    ("Bad teacher", "1e-4", "runs/badteacher/cifar100_unlearn_rate_1e-4_bs_1"),
    (r"\textsc{Scrub}", "3e-3", "runs/scrub/cifar100_unlearn_lr_3e-3_bs_1"),
    (r"\textsc{Scrub}", "1e-3", "runs/scrub/cifar100_unlearn_lr_1e-3_bs_1"),
    (r"\textsc{Scrub}", "5e-4", "runs/scrub/cifar100_bs_1"),
    (r"\textsc{Scrub}", "1e-4", "runs/scrub/cifar100_unlearn_lr_1e-4_bs_1"),
    (r"\textsc{Scrub}+R", "3e-3", "runs/scrub_r/cifar100_unlearn_lr_3e-3_bs_1"),
    (r"\textsc{Scrub}+R", "1e-3", "runs/scrub_r/cifar100_unlearn_lr_1e-3_bs_1"),
    (r"\textsc{Scrub}+R", "5e-4", "runs/scrub_r/cifar100_bs_1"),
    (r"\textsc{Scrub}+R", "1e-4", "runs/scrub_r/cifar100_unlearn_lr_1e-4_bs_1"),
]

#: The sweep whose `trained` numbers become the reference row. Any of the 14 would do --
#: they share their trained models, which the audit re-verifies (see the README) -- so the
#: choice is arbitrary and `_check_reference` asserts the agreement rather than trusting it.
REFERENCE_SWEEP = "runs/delete/cifar100_bs_1"

#: Every attack column mia_basic.py scores, in reporting order. `class_only` is the
#: control, set apart in the header. The reference-row check covers all of these.
ATTACK_COLS = ("forget_out", "test_cm", "val_cm", "test")
CONTROL_COL = "class_only"

#: The subset actually printed. `val_cm` is omitted for width: carrying a s.d. on every
#: column costs ~150pt, and val_cm is the one that can go -- 413 examples per side makes it
#: the noisiest column in the table, and it measures the same thing as test_cm against a
#: smaller held-out pool. It stays in `<sweep>/mia_basic_<source>.json`, and the caption
#: says where. Put it back by adding it here (and expect the table to need a wider float).
PRINT_COLS = ("forget_out", "test_cm", "test")

LR_TEX = {"3e-3": r"$3\times10^{-3}$", "1e-3": r"$1\times10^{-3}$",
          "5e-4": r"$5\times10^{-4}$", "1e-4": r"$1\times10^{-4}$"}


def _cell(doc: dict, col: str) -> str:
    """One accuracy cell: the mean over runs, with the across-run standard deviation set
    small beside it. That spread is what says whether a row's change from the reference is
    real -- several of them are smaller than 1 point, and a +-3.3 row (scrub_r at 1e-3)
    should not be read like a +-0.5 one."""
    mean = doc["mean"][col]["accuracy"]
    std = doc["std"][col]["accuracy"]
    return rf"{mean:.2f}{{\scriptsize$\,\pm${std:.2f}}}"


def mia_doc(sweep: str, source: str) -> dict:
    path = ROOT / sweep / f"mia_basic_{source}.json"
    if not path.exists():
        raise SystemExit(f"missing {path}\nRun: python -m audit.mia_basic --source {source} {sweep}")
    return json.loads(path.read_text())


def accuracy(sweep: str, source: str) -> tuple:
    """Mean retain / sampled-forget / test accuracy of the audited checkpoint, over every
    run of the sweep. Same three fallbacks make_comparison_table.py uses, in the same
    order: the eval_checkpoint_accuracy.py cache, then sweep_summary.json (only valid at
    `unlearned`), then the per-run metrics.json files."""
    cache = ROOT / sweep / f"checkpoint_accuracy_{source}.json"
    if cache.exists():
        m = json.loads(cache.read_text())["mean"]
        return m["retain"], m["sampled_forget"], m["test"]

    if source == "unlearned":
        summary = ROOT / sweep / "sweep_summary.json"
        if summary.exists() and summary.stat().st_size:
            rows = [r for r in json.loads(summary.read_text()) if "metrics_after" in r]
            if rows:
                return tuple(float(np.mean([r["metrics_after"][s]["accuracy"] for r in rows]))
                             for s in ("retain", "sampled_forget", "test"))
        key = "after_unlearning"
    elif source == "trained":
        key = "before_unlearning"
    else:
        raise SystemExit(
            f"{sweep}: no checkpoint_accuracy_{source}.json and no fallback for "
            f"source {source!r}. Run: python -m audit.eval_checkpoint_accuracy --source {source} {sweep}")

    vals = []
    for path in sorted((ROOT / sweep).glob("run_*/metrics.json")) + \
                sorted((ROOT / sweep).glob("test_run/run_*/metrics.json")):
        metrics = json.loads(path.read_text()).get(key)
        if metrics:
            vals.append([metrics[s]["accuracy"] for s in ("retain", "sampled_forget", "test")])
    if not vals:
        raise SystemExit(f"{sweep}: cannot find R/F/T for source {source!r}")
    return tuple(float(v) for v in np.mean(vals, axis=0))


def _check_reference(reference: dict, badteacher_source: str) -> None:
    """The reference row claims one pre-unlearning number for every method. That is only
    honest if the sweeps really do share their trained models, so verify it instead of
    asserting it in prose: every sweep's `trained` attack must agree to <0.01 points."""
    ref = {c: reference["mean"][c]["accuracy"] for c in ATTACK_COLS + (CONTROL_COL,)}
    for _, _, sweep in SWEEPS:
        got = mia_doc(sweep, "trained")["mean"]
        for col, value in ref.items():
            if abs(got[col]["accuracy"] - value) > 0.01:
                raise SystemExit(
                    f"{sweep} @ trained disagrees with {REFERENCE_SWEEP} on {col}: "
                    f"{got[col]['accuracy']:.4f} vs {value:.4f}. These sweeps do not share "
                    f"their trained models, so a single reference row would be wrong.")


def build_table(badteacher_source: str) -> str:
    reference = mia_doc(REFERENCE_SWEEP, "trained")
    _check_reference(reference, badteacher_source)
    n_runs = {reference["n_runs"]}

    lines = []
    ref_rft = accuracy(REFERENCE_SWEEP, "trained")
    # The reference row is the baseline Delta is measured against, so its own Delta cell is
    # empty rather than 0.00 -- a zero there would read as a measurement.
    ref_cells = [_cell(reference, PRINT_COLS[0]), "--"] + \
                [_cell(reference, c) for c in PRINT_COLS[1:]] + \
                [_cell(reference, CONTROL_COL)]
    lines.append(r"\multicolumn{2}{l}{\emph{Before unlearning}} & "
                 + "%.1f/%.1f/%.1f" % ref_rft + " & " + " & ".join(ref_cells) + r" \\")
    lines.append(r"\midrule")

    previous_method = None
    for method, lr, sweep in SWEEPS:
        source = badteacher_source if method == "Bad teacher" else "unlearned"
        doc = mia_doc(sweep, source)
        n_runs.add(doc["n_runs"])
        if previous_method is not None and method != previous_method:
            lines.append(r"\addlinespace")
        row = [method if method != previous_method else "", LR_TEX[lr],
               "%.1f/%.1f/%.1f" % accuracy(sweep, source)]
        # The headline column gets its own Delta column: "58.5" only means something next
        # to the 58.1 it started at, and putting both plus the std in one cell is unreadable.
        delta = doc["mean"][PRINT_COLS[0]]["accuracy"] - reference["mean"][PRINT_COLS[0]]["accuracy"]
        row += [_cell(doc, PRINT_COLS[0]), f"{delta:+.2f}"]
        row += [_cell(doc, c) for c in PRINT_COLS[1:]]
        row.append(_cell(doc, CONTROL_COL))
        lines.append(" & ".join(row) + r" \\")
        previous_method = method

    if len(n_runs) != 1:
        raise SystemExit(f"sweeps disagree on run count: {sorted(n_runs)}; the caption "
                         f"would misstate how many runs each row averages over")
    runs = n_runs.pop()

    bt_note = ("at unlearning epoch 1" if badteacher_source == "unlearn_epoch_1"
               else "at its final unlearned model")
    caption = (
        r"Basic membership-inference accuracy (\%) against class-unlearned CIFAR-100 "
        r"models, following Kurmanji et al.\ (2023): a binary classifier is trained on the "
        r"audited model's per-example losses to separate ``in'' from ``out'' points, and "
        r"scored on held-out losses balanced between the two ($5$-fold stratified "
        r"cross-validation, logistic regression on the loss). $50$ is chance, i.e.\ the "
        r"attacker cannot separate them. Each figure is the mean over all " + str(runs) +
        r" runs of a sweep, which differ only in which half of the forget candidates was "
        r"sampled. The four attack columns share their ``in'' population -- the "
        r"$2250$ forget points that run trained on -- and differ in what is taken as "
        r"``out'': $\mathcal{F}_{\mathrm{out}}$ the $2250$ forget points it did \emph{not} "
        r"sample, $\mathcal{T}_{\mathrm{cm}}$ the $1000$ test examples of the forget "
        r"set's own classes, and $\mathcal{T}$ the whole test set. A fourth population, "
        r"the $413$ validation examples of those classes (SCRUB+R's rewind reference "
        r"population), is measured but omitted here for width -- with $413$ examples per "
        r"side it is the noisiest of the four, and it sits $2.5$ points below "
        r"$\mathcal{T}_{\mathrm{cm}}$ on average (per-sweep range $-1.9$ to $+3.9$). It is "
        r"recorded in the accompanying JSON. $\mathcal{F}_{\mathrm{out}}$ is the only column in which "
        r"membership is the sole difference between the two populations -- same classes, "
        r"same split, same preprocessing. $\mathcal{T}$ is the paper's sentence read literally and "
        r"is \emph{not} a membership measurement on a class-based forget split: both of "
        r"the control column's populations are test examples the model never trained on "
        r"(forget-class labelled ``in'', other-class ``out''), so its membership signal is "
        r"zero by construction and its accuracy is the class channel alone, which "
        r"dominates $\mathcal{T}$ and exceeds it on every row. R/F/T is the mean retain / "
        r"sampled-forget / test accuracy of the audited checkpoint, reported alongside "
        r"because the two cannot be read apart: a model whose utility has collapsed is "
        r"attacked at chance regardless of what it leaked. Bad teacher is scored " + bt_note +
        r", the remaining methods at their final unlearned model, matching the sources of "
        r"the corresponding lower-bound tables.")

    header = "\n".join([
        f"% Generated by {Path(__file__).name} -- do not edit by hand.",
        f"% basic MIA, logistic regression on the loss, {reference['folds']}-fold CV, "
        f"seed base {reference['seed']}",
        f"% badteacher @ {badteacher_source}; delete, scrub, scrub_r @ unlearned",
        "% Reference row: the same attack on trained_model.pth (pre-unlearning).",
        "% class-only is a CONTROL with zero membership signal, not an attack -- read the",
        "% test column against it, not against 50.",
    ])

    return "\n".join([
        header, "",
        r"\begin{table}[t]", r"\centering", r"\footnotesize",
        # Nine columns with a s.d. on five of them does not fit a 6.5in portrait text
        # block at \small: measured 151pt overfull. \footnotesize plus a 3pt column
        # separation and the slash-packed R/F/T cell brings it inside, with no \resizebox
        # (which would need graphicx and silently rescale the font).
        r"\setlength{\tabcolsep}{2pt}",
        # No \multirow: the lower-bound tables in this directory get by with plain
        # \cmidrule groups, so this one stays loadable under the same preamble.
        r"\begin{tabular}{llcccccc}", r"\toprule",
        r"Method & LR & R/F/T (\%) & "
        r"\multicolumn{4}{c}{Attack accuracy (\%)} & Control \\",
        r"\cmidrule(lr){4-7}\cmidrule(lr){8-8}",
        r" & & & $\mathcal{F}_{\mathrm{out}}$ & $\Delta$ & $\mathcal{T}_{\mathrm{cm}}$ & "
        r"$\mathcal{T}$ & class only \\",
        r"\midrule",
        *lines,
        r"\bottomrule", r"\end{tabular}",
        r"\caption{" + caption + r" Every accuracy is written mean\,$\pm$\,s.d., where the "
        r"spread is taken \emph{across} the runs of the sweep -- i.e.\ how much the leak "
        r"depends on which half of the candidates was sampled -- not across the "
        r"cross-validation folds, which is recorded per run in the accompanying JSON. "
        r"$\Delta$ is the change in $\mathcal{F}_{\mathrm{out}}$ from the reference row and "
        r"is the quantity attributable to unlearning; compare it against that column's "
        r"s.d.\ before reading a sign into it.}",
        r"\label{tab:mia-basic}",
        r"\end{table}", "",
    ])


def _distribution(doc: dict, col: str) -> dict:
    """The across-run distribution of one column, from the per-run results stored in the
    same file. mean/std are recomputed here and checked against the aggregate the sweep
    file already carries, so a stale or hand-edited JSON is caught rather than plotted."""
    vals = np.array([r[col]["accuracy"] for r in doc["per_run"].values()])
    stats = {"mean": float(vals.mean()), "std": float(vals.std()),
             "median": float(np.median(vals)), "min": float(vals.min()),
             "max": float(vals.max()), "n": len(vals)}
    # Both aggregate blocks key the value as "accuracy"; `mean` also carries auc and the
    # per-side mean losses, `std` only the accuracy.
    for key, block in (("mean", doc["mean"]), ("std", doc["std"])):
        stored = block[col]["accuracy"]
        if abs(stats[key] - stored) > 1e-9:
            raise SystemExit(f"{doc['sweep']}: per_run {col} {key} = {stats[key]!r} does not "
                             f"match the stored aggregate {stored!r}; the JSON is inconsistent")
    return stats


def build_fout_table(badteacher_source: str) -> str:
    """A second, narrower table for the one column that measures membership alone.

    Same rows and sources as `build_table`, but every statistic is about
    `forget_out`: mean +- s.d., median, and the min/max over the sweep's runs. Those last
    two are extremes of 60 observations, not a confidence interval -- they say how bad the
    worst-sampled run got, which the mean hides and which matters for a row like SCRUB at
    1e-3 whose s.d. is four times everyone else's.
    """
    col = "forget_out"
    reference = mia_doc(REFERENCE_SWEEP, "trained")
    _check_reference(reference, badteacher_source)
    ref = _distribution(reference, col)

    lines = [r"\multicolumn{2}{l}{\emph{Before unlearning}} & "
             + "%.1f/%.1f/%.1f" % accuracy(REFERENCE_SWEEP, "trained")
             + rf" & {ref['mean']:.2f}{{\scriptsize$\,\pm${ref['std']:.2f}}} & -- "
             + rf"& {ref['median']:.2f} & {ref['min']:.2f} & {ref['max']:.2f} \\",
             r"\midrule"]

    n_runs, previous_method = {ref["n"]}, None
    for method, lr, sweep in SWEEPS:
        source = badteacher_source if method == "Bad teacher" else "unlearned"
        doc = mia_doc(sweep, source)
        stats = _distribution(doc, col)
        n_runs.add(stats["n"])
        if previous_method is not None and method != previous_method:
            lines.append(r"\addlinespace")
        lines.append(" & ".join([
            method if method != previous_method else "", LR_TEX[lr],
            "%.1f/%.1f/%.1f" % accuracy(sweep, source),
            rf"{stats['mean']:.2f}{{\scriptsize$\,\pm${stats['std']:.2f}}}",
            f"{stats['mean'] - ref['mean']:+.2f}",
            f"{stats['median']:.2f}", f"{stats['min']:.2f}", f"{stats['max']:.2f}",
        ]) + r" \\")
        previous_method = method

    if len(n_runs) != 1:
        raise SystemExit(f"sweeps disagree on run count: {sorted(n_runs)}")
    runs = n_runs.pop()

    bt_note = ("at unlearning epoch 1" if badteacher_source == "unlearn_epoch_1"
               else "at its final unlearned model")
    caption = (
        r"Basic membership-inference accuracy (\%) on $\mathcal{F}_{\mathrm{out}}$, the one "
        r"population in which membership is the only difference between the two sides: "
        r"``in'' is the $2250$ forget candidates a run sampled and trained on, ``out'' is "
        r"the $2250$ it did not, so the classes, the split and the preprocessing are all "
        r"held fixed. Attacker and protocol are as in Table~\ref{tab:mia-basic} "
        r"(logistic regression on the per-example loss, $5$-fold stratified "
        r"cross-validation, balanced classes); $50$ is chance. Every statistic is taken "
        r"over the " + str(runs) + r" runs of the sweep, which differ only in which half of "
        r"the candidates was sampled: mean\,$\pm$\,s.d., the median, and the minimum and "
        r"maximum \emph{observed run}. Min and max are extremes of " + str(runs) +
        r" observations, not a confidence interval -- they describe the spread of the "
        r"mechanism over its own sampling, and the gap between them is the honest measure "
        r"of how much a single-run number should be trusted. $\Delta$ is the change in the "
        r"mean from the reference row, the quantity attributable to unlearning; read it "
        r"against the s.d.\ beside it. R/F/T is the mean retain / sampled-forget / test "
        r"accuracy of the audited checkpoint, which cannot be read apart from the attack: "
        r"a model whose utility has collapsed is attacked at chance regardless of what it "
        r"leaked. Bad teacher is scored " + bt_note + r", the remaining methods at their "
        r"final unlearned model, matching the sources of the corresponding lower-bound "
        r"tables.")

    header = "\n".join([
        f"% Generated by {Path(__file__).name} --table fout -- do not edit by hand.",
        f"% basic MIA on forget_out only, logistic regression on the loss, "
        f"{reference['folds']}-fold CV, seed base {reference['seed']}",
        f"% badteacher @ {badteacher_source}; delete, scrub, scrub_r @ unlearned",
        f"% median/min/max are over the {runs} runs of each sweep, recomputed from the",
        "% per_run block of <sweep>/mia_basic_<source>.json (mean and s.d. cross-checked",
        "% against the aggregate stored in the same file).",
        "% The caption \\ref{tab:mia-basic}s the companion table in tables/mia_basic.tex --",
        "% include both, or replace the \\ref if you use this one on its own.",
    ])

    return "\n".join([
        header, "",
        r"\begin{table}[t]", r"\centering", r"\small",
        # 15pt overfull at the 5pt default in a 6.5in portrait block; 3pt fits.
        r"\setlength{\tabcolsep}{3pt}",
        # 8 columns: method, LR, R/F/T, then mean / Delta / median / min / max.
        r"\begin{tabular}{llcccccc}", r"\toprule",
        r"Method & LR & R/F/T (\%) & \multicolumn{5}{c}{"
        r"$\mathcal{F}_{\mathrm{out}}$ attack accuracy (\%)} \\",
        r"\cmidrule(lr){4-8}",
        r" & & & mean & $\Delta$ & median & min & max \\",
        r"\midrule",
        *lines,
        r"\bottomrule", r"\end{tabular}",
        r"\caption{" + caption + r"}",
        r"\label{tab:mia-basic-fout}",
        r"\end{table}", "",
    ])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--badteacher-source", default="unlearn_epoch_1",
                   help="Checkpoint to score bad teacher at (default unlearn_epoch_1, "
                        "matching make_comparison_table.py)")
    p.add_argument("--table", default="main", choices=("main", "fout"),
                   help="main: every population, mean +- s.d. (default). "
                        "fout: forget_out only, with median/min/max over the runs")
    p.add_argument("--out", default=None,
                   help="Output path (default tables/mia_basic.tex, or "
                        "tables/mia_basic_fout.tex for --table fout)")
    args = p.parse_args()

    if args.out is None:
        args.out = "tables/mia_basic.tex" if args.table == "main" else "tables/mia_basic_fout.tex"
    table = build_fout_table(args.badteacher_source) if args.table == "fout" \
        else build_table(args.badteacher_source)
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(table)
    print(table)
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
