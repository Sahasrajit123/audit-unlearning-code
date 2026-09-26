"""
LaTeX table generator for the basic MIA (mia_basic.py). A pure JSON reader -- it loads no
checkpoints, touches no data, and runs no model, so it is cheap to re-run while wording the
caption.

It reads one `mia_basic_<source>.json` per sweep, the file mia_basic.py writes
at sweep level, and emits one `table` environment per branch of the forget split, because
the two branches do not have the same columns:

  class-centric forget pool (e.g. 10 of 100 classes)
      F_out, T_cm, V_cm, T and the class-only CONTROL
  uniform forget pool (all classes)
      F_out, T, V -- there is no class confound, so there is no control column and no
      confound discussion in the caption

Delta is computed PER SWEEP against that sweep's own `trained` row: every run trains its
own model on `retain + that run's sampled forget`, so there is no single shared
pre-unlearning checkpoint. `_check_reference` asserts that sweeps with the same
data_subfolder and the same model -- which differ only in their post_unlearning config --
have `trained` rows that agree to within across-run noise, and raises if they do not.

Partial sweeps -- any sweep where some run lacks the checkpoint being attacked -- are
dropped by default and listed in the header comment, so a 51-run mean is never silently
averaged against a 50-run one. `--include-partial` overrides, and marks the rows.

Usage:
    python make_mia_table.py --source unlearned --out tables/mia_basic.tex logs_*
"""
import argparse
import json
from pathlib import Path

#: Column order within each branch, and the maths symbol each gets in the table.
COLUMNS = {
    "forget_out": r"$\mathcal{F}_{\mathrm{out}}$",
    "test_cm": r"$\mathcal{T}_{\mathrm{cm}}$",
    "val_cm": r"$\mathcal{V}_{\mathrm{cm}}$",
    "test": r"$\mathcal{T}$",
    "val": r"$\mathcal{V}$",
    "class_only": "class only",
}
BRANCH_COLUMNS = {
    "class_centric": ("forget_out", "test_cm", "val_cm", "test", "class_only"),
    "uniform": ("forget_out", "test", "val"),
}
BRANCH_TITLE = {
    "class_centric": "class-centric forget pool (10 of 100 classes)",
    "uniform": "uniform forget pool (all 100 classes)",
}

#: Two `trained` rows from the same data split and model should not differ by more than
#: this many accuracy points; beyond it something other than post_unlearning differs.
REFERENCE_TOLERANCE = 1.0


def _sweep_file(sweep: Path, source: str) -> Path:
    return sweep / f"mia_basic_{source}.json"


def load_docs(sweeps, source: str) -> dict:
    """{sweep name: sweep-level result}, skipping sweeps that were never scored.

    A missing file is a skip rather than an error: it is normal to point this at a glob
    that includes sweeps with no checkpoint of the requested source.
    """
    docs = {}
    for sweep in sweeps:
        path = _sweep_file(Path(sweep), source)
        if not path.exists():
            print(f"skip {sweep}: no {path.name}")
            continue
        docs[Path(sweep).name] = json.loads(path.read_text())
    return docs


def _check_reference(docs: dict) -> list:
    """`trained` reference agreement -- see the module docstring. Raises on disagreement.

    Returned lines go into the .tex header so the check is visible in the artifact, not
    only on the console of whoever generated it.
    """
    groups = {}
    for name, doc in docs.items():
        key = (doc["data_subfolder"], doc["model"])
        groups.setdefault(key, {})[name] = doc["mean"]["forget_out"]["accuracy"]

    lines, failures = [], []
    for (subfolder, model), vals in sorted(groups.items()):
        if len(vals) < 2:
            continue
        spread = max(vals.values()) - min(vals.values())
        lines.append(f"%   {subfolder} / {model}: {len(vals)} sweeps, "
                     f"reference spread {spread:.2f} points")
        if spread > REFERENCE_TOLERANCE:
            failures.append(
                f"{subfolder}/{model}: `trained` forget_out spans {spread:.2f} points over "
                f"{len(vals)} sweeps (tolerance {REFERENCE_TOLERANCE}). These sweeps share "
                f"a data split and a training config, so their pre-unlearning models should "
                f"agree. Values: "
                + ", ".join(f"{n}={v:.2f}" for n, v in sorted(vals.items())))
    if failures:
        raise SystemExit("reference-agreement check failed:\n  " + "\n  ".join(failures))
    return lines


def _short(name: str) -> str:
    """Sweep directory name -> row label. Only the `logs_` prefix is dropped; uniqueness of
    the labels within a table is asserted at render time."""
    label = name[len("logs_"):] if name.startswith("logs_") else name
    return label.replace("_", r"\_")


def _median_forget_out(doc: dict):
    """Median across runs of forget_out.

    Prefers the `median` block mia_basic.py now writes, and otherwise derives it from the
    `per_run` copies the sweep file already carries -- so results scored before that field
    existed do not have to be re-scored.
    """
    m = doc.get("median", {}).get("forget_out", {}).get("accuracy")
    if m is not None:
        return m
    vals = [r["variants"]["forget_out"]["accuracy"]
            for r in doc.get("per_run", {}).values()
            if "forget_out" in r.get("variants", {})]
    if not vals:
        return None
    vals = sorted(vals)
    n = len(vals)
    return vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])


def _cell(stats: dict) -> str:
    return (f"{stats['accuracy']:.2f}{{\\scriptsize$\\,\\pm$"
            f"{stats['std']:.2f}}}")


def _rft(doc: dict) -> str:
    a = doc["accuracy"]
    r = "--" if a.get("retain") is None else f"{100 * a['retain']:.1f}"
    return f"{r}/{100 * a['forget_in']:.1f}/{100 * a['test']:.1f}"


def _caption(branch: str, source: str, has_delta: bool, docs: dict) -> str:
    """A caption that defines every column and stands on its own.

    The class confound paragraph appears ONLY on the class-centric table. Repeating it on
    the uniform table would be wrong: there is no distinguished class set there, so there
    is nothing for the attacker to read off the label.
    """
    attacker_text = ("logistic regression on the single loss feature, with the loss "
                     "clipped to $[-100, 100]$ following Kurmanji et al.'s released code, "
                     "scored by $5$-fold stratified cross-validation")

    text = (
        f"Basic membership-inference accuracy (\\%) against unlearned CIFAR-100 models, "
        f"following Kurmanji et al.\\ (2023): a binary classifier is trained on the audited "
        f"model's per-example cross-entropy losses to separate ``in'' from ``out'' points, "
        f"and scored on held-out losses balanced between the two, here with "
        f"{attacker_text}. $50$ is chance, i.e.\\ the attacker cannot separate the two "
        f"populations; chance is exact by construction, because membership is a coin flip "
        f"the pipeline makes at \\texttt{{forget\\_fraction}}$=0.5$ independently of the "
        f"training seed. Each figure is the mean over the runs of a sweep, which differ "
        f"only in which half of the forget candidates was sampled, and every accuracy is "
        f"written mean\\,$\\pm$\\,s.d.\\ where the spread is taken \\emph{{across}} runs -- "
        f"how much the leak depends on which half was sampled -- not across the "
        f"cross-validation folds, which is recorded per run in the accompanying JSON. "
        f"Each row is the sweep's final unlearned checkpoint (\\texttt{{{source}}}). "
    )

    text += (
        "All attack columns share their ``in'' population, the $2250$ forget points that "
        "run trained on, and differ in what is taken as ``out'': "
        "$\\mathcal{F}_{\\mathrm{out}}$ the $2250$ forget points it did \\emph{not} "
        "sample, "
    )
    if branch == "class_centric":
        text += (
            "$\\mathcal{T}_{\\mathrm{cm}}$ the $1000$ test examples of the forget pool's "
            "own $10$ classes, $\\mathcal{V}_{\\mathrm{cm}}$ the $413$ validation examples "
            "of those classes, and $\\mathcal{T}$ the whole $10{,}000$-example test set. "
            "$\\mathcal{F}_{\\mathrm{out}}$ is the column to read: it is the only one in "
            "which training membership is the sole difference between the two populations "
            "-- same classes, same train pool, same preprocessing. $\\mathcal{T}$ is the "
            "paper's sentence read literally and is \\emph{not} a membership measurement "
            "on a class-based forget split, because after class unlearning the model is "
            "bad on those $10$ classes and ordinary on the other $90$, so ``is this a "
            "forget-class image'' predicts the label. The class-only column is a "
            "\\textbf{control, not an attack}: both of its populations are test examples "
            "the model never trained on (forget-class labelled ``in'', other-class "
            "``out''), so its membership signal is zero by construction and its accuracy "
            "is the class channel alone. Read $\\mathcal{T}$ against it, not against $50$. "
        )
    else:
        text += (
            "$\\mathcal{T}$ the whole $10{,}000$-example test set and $\\mathcal{V}$ the "
            "$5000$-example validation set. The forget pool here is sampled uniformly and "
            "spans all $100$ classes, so there is no class confound: no class-matched "
            "column is reported because it would duplicate $\\mathcal{T}$, and no "
            "class-only control is reported because there is no distinguished class set "
            "for one to be built from. $\\mathcal{F}_{\\mathrm{out}}$ remains the column "
            "to read, as the only one where the two populations differ in training "
            "membership and nothing else; $\\mathcal{T}$ and $\\mathcal{V}$ additionally "
            "carry the train/test generalisation gap. "
        )

    text += (
        "R/F/T is the mean retain / sampled-forget / test accuracy (\\%) of the same "
        "checkpoint that was attacked, and must be read with the attack column: a model "
        "whose utility has collapsed is attacked at chance regardless of what it leaked. "
    )
    text += (
        "The \\emph{med} column is the median of $\\mathcal{F}_{\\mathrm{out}}$ over the "
        "runs of the sweep, reported beside the mean because a gap between them would "
        "indicate a skewed run distribution -- a few runs behaving differently rather than "
        "a shifted centre. It is given only for $\\mathcal{F}_{\\mathrm{out}}$, the column "
        "the conclusions rest on. "
        "The \\emph{Before unlearning} row is the same attack on the pre-unlearning "
        "checkpoints. Every sweep in this table shares its trained models -- they differ "
        "only in the post-unlearning stage -- so it is a single row rather than one per "
        "sweep; the agreement across sweeps is verified rather than assumed. "
    )
    if has_delta:
        text += (
            "$\\Delta$ is the change in $\\mathcal{F}_{\\mathrm{out}}$ from the same "
            "sweep's own pre-unlearning checkpoint and is the part attributable to "
            "unlearning; compare it against that column's s.d.\\ before reading a sign "
            "into it. This baseline is per sweep because every run trains its own model "
            "on its own sampled half and there is no shared pre-unlearning checkpoint. "
        )
    text += (
        "The unlearning step operates only on the $2250$ points the run was asked to "
        "forget, so the ``out'' points are train-pool images that this run neither "
        "trained on nor unlearned -- the mechanism never saw them."
    )
    return text


def _reference_row(refs: dict, cols, has_delta: bool):
    """The 'Before unlearning' row: the same attack on the pre-unlearning checkpoints.

    Every sweep in a branch shares its trained models (verified by `_check_reference`), so
    a single row is well defined. Values are
    averaged over the sweeps in the branch; the +- is the across-run spread, averaged the
    same way. Returns None when no `trained` results were found, so the table degrades to
    what it was rather than failing.
    """
    if not refs:
        return None
    ds = list(refs.values())
    if not all(c in d["mean"] for d in ds for c in cols):
        return None
    mean = lambda xs: sum(xs) / len(xs)
    a = ds[0]["accuracy"]
    rft = ("--" if a.get("retain") is None
           else f"{100 * mean([d['accuracy']['retain'] for d in ds]):.1f}") + \
          f"/{100 * mean([d['accuracy']['forget_in'] for d in ds]):.1f}" + \
          f"/{100 * mean([d['accuracy']['test'] for d in ds]):.1f}"
    cells = [r"\emph{Before unlearning}", rft,
             _cell({"accuracy": mean([d["mean"][cols[0]]["accuracy"] for d in ds]),
                    "std": mean([d["std"][cols[0]]["accuracy"] for d in ds])})]
    # Median column, same position as in the data rows.
    meds = [m for m in (_median_forget_out(d) for d in ds) if m is not None]
    cells.append(f"{mean(meds):.2f}" if meds else "--")
    if has_delta:
        cells.append("--")
    for c in cols[1:]:
        cells.append(_cell({"accuracy": mean([d["mean"][c]["accuracy"] for d in ds]),
                            "std": mean([d["std"][c]["accuracy"] for d in ds])}))
    return " & ".join(cells) + r" \\"


def render_table(branch: str, docs: dict, source: str,
                 deltas: dict, label_suffix: str, refs: dict = None) -> str:
    cols = [c for c in BRANCH_COLUMNS[branch]
            if all(c in d["mean"] for d in docs.values())]
    has_delta = bool(deltas)
    ncol = 3 + len(cols) + (1 if has_delta else 0)

    head = ["Sweep", "R/F/T (\\%)", COLUMNS[cols[0]],
            r"med $\mathcal{F}_{\mathrm{out}}$"]
    if has_delta:
        head.append("$\\Delta$")
    head += [COLUMNS[c] for c in cols[1:]]

    labels = {}
    for name in docs:
        labels.setdefault(_short(name), []).append(name)
    dupes = {k: v for k, v in labels.items() if len(v) > 1}
    if dupes:
        raise SystemExit(
            "row labels are not unique within the "
            f"{branch} table -- two sweeps would be indistinguishable: "
            + "; ".join(f"{k!r} <- {v}" for k, v in dupes.items()))

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{ll" + "c" * (ncol - 2) + "}",
        r"\toprule",
        " & ".join(head) + r" \\",
        r"\midrule",
    ]
    ref_row = _reference_row({n: r for n, r in (refs or {}).items() if n in docs},
                             cols, has_delta)
    if ref_row:
        lines += [ref_row, r"\midrule"]

    def _check_width(row: str, what: str):
        """LaTeX does NOT error on a row with too few cells -- it silently leaves the
        trailing columns blank, which shifts every value left of the gap into the wrong
        column -- so the check is explicit."""
        n = row.count(" & ") + 1
        if n != ncol:
            raise SystemExit(
                f"{what}: {n} cells but the table has {ncol} columns. A short row is not a "
                f"LaTeX error -- it would silently misalign the values. Row:\n  {row}")

    if ref_row:
        _check_width(ref_row, "reference row")
    for name in sorted(docs):
        doc = docs[name]
        cells = [_short(name) + ("" if doc["complete"] else r"$^{\dagger}$"),
                 _rft(doc)]
        first = {"accuracy": doc["mean"][cols[0]]["accuracy"],
                 "std": doc["std"][cols[0]]["accuracy"]}
        cells.append(_cell(first))
        med = _median_forget_out(doc)
        cells.append("--" if med is None else f"{med:.2f}")
        if has_delta:
            d = deltas.get(name)
            cells.append("--" if d is None else f"{d:+.2f}")
        for c in cols[1:]:
            cells.append(_cell({"accuracy": doc["mean"][c]["accuracy"],
                                "std": doc["std"][c]["accuracy"]}))
        row = " & ".join(cells)
        _check_width(row, f"data row {name!r}")
        lines.append(row + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{" + _caption(branch, source, has_delta, docs) + "}",
        r"\label{tab:mia-basic-" + branch.replace("_", "-") + label_suffix + "}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sweeps", nargs="+", help="Sweep directories already scored")
    p.add_argument("--source", default="unlearned")
    p.add_argument("--reference-source", default="trained",
                   help="Source whose forget_out is subtracted to form Delta, per sweep. "
                        "Pass '' to omit the Delta column.")
    p.add_argument("--include-partial", action="store_true",
                   help="Keep sweeps where some run lacks the checkpoint; they are marked "
                        "with a dagger. Off by default so a partial mean is never "
                        "averaged against a complete one.")
    p.add_argument("--out", default="tables/mia_basic.tex")
    args = p.parse_args()

    docs = load_docs(args.sweeps, args.source)
    if not docs:
        raise SystemExit("no scored sweeps found; run mia_basic.py first")

    excluded = {n: d for n, d in docs.items() if not d["complete"]}
    if excluded and not args.include_partial:
        for n, d in excluded.items():
            print(f"exclude {n}: partial ({d['n_runs']}/{d['n_runs_total']} runs have a "
                  f"'{args.source}' checkpoint)")
        docs = {n: d for n, d in docs.items() if d["complete"]}
    if not docs:
        raise SystemExit("every sweep was partial; nothing to tabulate")

    deltas, ref_lines, refs = {}, [], {}
    if args.reference_source:
        # Resolve the reference files from the paths given on the command line, not from
        # the `sweep` field recorded at scoring time -- that was written relative to
        # whatever cwd scored it.
        kept = [s for s in args.sweeps if Path(s).name in docs]
        refs = load_docs(kept, args.reference_source)
        ref_lines = _check_reference(refs) if refs else []
        for name, doc in docs.items():
            ref = refs.get(name)
            if ref and "forget_out" in ref["mean"]:
                deltas[name] = (doc["mean"]["forget_out"]["accuracy"]
                                - ref["mean"]["forget_out"]["accuracy"])

    header = [
        f"% Generated by make_mia_table.py -- do not edit by hand.",
        f"% basic MIA (Kurmanji et al. 2023), source='{args.source}'.",
        f"% attacker settings: {json.dumps(next(iter(docs.values()))['attacker'])}",
        f"% 50 is chance. class-only is a CONTROL with zero membership signal, not an",
        f"% attack -- read the T column against it, not against 50.",
        f"% Requires: booktabs, amssymb.",
    ]
    if deltas:
        header.append(f"% Delta is against each sweep's own '{args.reference_source}' "
                      f"checkpoint (per sweep -- see module docstring).")
    if ref_lines:
        header.append("% reference-agreement check:")
        header += ref_lines
    if excluded:
        header.append("% excluded as partial: " + ", ".join(
            f"{n} ({d['n_runs']}/{d['n_runs_total']})" for n, d in excluded.items()))

    tables = []
    for branch in ("class_centric", "uniform"):
        rows = {n: d for n, d in docs.items() if d["forget_split"] == branch}
        if not rows:
            continue
        tables.append(f"% ---- {BRANCH_TITLE[branch]} ----")
        tables.append(render_table(branch, rows, args.source,
                                   {n: deltas[n] for n in rows if n in deltas}, "",
                                   refs=refs))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(header) + "\n\n" + "\n\n".join(tables) + "\n")
    print(f"-> {out}  ({len(docs)} sweeps, "
          f"{len([1 for d in docs.values() if d['forget_split'] == 'class_centric'])} "
          f"class-centric)")


if __name__ == "__main__":
    main()
