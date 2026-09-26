"""
The "basic MIA" of Kurmanji et al., *Towards Unbounded Machine Unlearning*, NeurIPS 2023
([arXiv:2302.09880](https://arxiv.org/abs/2302.09880)), run over a run_sweep.py sweep.

The paper's description, verbatim:

    We train a binary classifier (the 'attacker') on the unlearned model's losses on
    forget and test examples for the objective of classifying 'in' (forget) versus 'out'
    (test). The attacker then makes predictions for held-out losses (losses the attacker
    wasn't trained on) that are balanced between forget and test losses. For a perfect
    defense, the attacker's accuracy is 50%, indicating that it is unable to separate the
    two sets.

So, per run: score the checkpoint on both populations, balance them, and cross-validate a
binary classifier on the one-dimensional loss feature. "Held-out losses the attacker
wasn't trained on" is implemented as stratified k-fold cross-validation, the usual reading
of that sentence; the reported accuracy is the mean over folds, so every loss is predicted
by an attacker that did not see it.

This is a different instrument from the rest of this repo's audit stack and is not a
replacement for it. `run_cumulative_audit.py` fits a per-point in/out distribution across
60 runs and turns the result into a certified epsilon/rho/mu lower bound. This is one
number per model, from that model alone, with no shadow runs and no bound attached --
cheap, standard, and directly comparable with the numbers the unlearning papers publish.

## Which examples are "out" -- read this before reading the numbers

The paper's phrasing says "test", and this project's forget split is *class-centric*:
cifar100_bs_1's forget set is 4500 examples drawn from 10 of the 100 classes. Against the
full test set, an attacker separating forget-losses from test-losses can win without ever
inferring membership -- after class unlearning the model is simply bad on those 10 classes
and ordinary on the other 90, so "high loss" means "forgotten class", not "was trained
on". The accuracy that comes back measures class identity.

The same paper is explicit about this for its own rewinding procedure (Sec. 3.2: "we
create a validation set of the same distribution as the forget set ... if the forget set
has only examples of class 0, we keep only examples of class 0 in the validation set
too"), and `unlearning/scrub.py:class_matched_val_subset` already implements that for
SCRUB+R. So several populations are scored per run and reported side by side. The first
four share their "in" side -- this run's sampled forget points -- and differ only in what
counts as "out":

  forget_out  the forget points this run did NOT sample. Same classes, same split, same
            preprocessing -- they differ from the "in" points by membership and nothing
            else, and it is the same in/out structure the eps audit tests. Not in the
            paper (it needs run_sweep.py's --forget-prob 0.5 sampling to exist), and the
            one variant where 50% unambiguously means "indistinguishable". Report this.
  test_cm   test examples of the forget set's own classes  (1000 of 10000 here) -- the
            closest honest reading of the paper's sentence for a class-based forget set
  val_cm    validation examples of those classes           (413 of 5000; this is exactly
            SCRUB+R's rewind reference population, so for scrub_r it is the attack the
            defence was tuned against)
  test      every test example (the paper's sentence read literally; class-confounded).
            Kept because it is what a literal implementation produces, and the gap between
            it and test_cm is the size of the artifact.

`class_only` is not an attack at all but the control that calibrates that artifact. Both
of its sides are test examples the model never trained on -- forget-class ones labelled
"in", other-class ones labelled "out" -- so the membership signal is exactly zero by
construction and whatever accuracy comes back is the class channel alone. It bounds how
much of the `test` column is unrelated to privacy: on `delete` at 1e-3 the control scores
~81% while `test` scores 74.7%, i.e. the literal attack does not even reach the
class-detection baseline.

Cost is one forward pass over forget+val+test (19,500 images) per run, so a 60-run sweep
takes well under a minute on a GPU. Results are written at both levels, and nothing
already on disk is modified:

  <sweep>/run_XX/mia_basic_<source>.json   that run's attack, every variant
  <sweep>/mia_basic_<source>.json          mean/std over the sweep's runs, plus a copy of
                                           every per-run result (next to
                                           `checkpoint_accuracy_<source>.json`)

The per-run files are also the resume point: a run that already has one is not re-scored
unless `--force` is passed, so an interrupted sweep picks up where it stopped and a sweep
that has since gained runs only pays for the new ones.

Usage:
    python -m audit.mia_basic --source unlearned runs/*/cifar100_bs_1
    python -m audit.mia_basic --source trained --device cuda:1 runs/scrub/*
    python -m audit.mia_basic --source unlearned --summary-out tables/mia_basic.json runs/*/*
"""
import argparse
import json
import os
from pathlib import Path

if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset

# Same-repo helpers, so there is exactly one definition of each of these:
#   _model_filename_from_source  source string ("unlearned", "unlearn_epoch_2", ...) -> filename
#   compute_phi_and_loss_per_point  the per-point loss the LiRA audit already uses
#   _run_dirs                    which run folders a sweep has, test_run/ included
from audit.audit_utils import _model_filename_from_source, compute_phi_and_loss_per_point
from core.data_utils import load_full_forget, load_retain_val_test
from audit.eval_checkpoint_accuracy import _run_dirs
from core.model import build_model, dataset_defaults

#: The populations scored per run, in the order they are reported. See the module
#: docstring -- they are not interchangeable, `test` is the confounded one, and
#: `class_only` is a zero-membership control rather than an attack.
VARIANTS = ("forget_out", "test_cm", "val_cm", "test", "class_only")


#: The authors' own MIA notebook clips the feature vector -- `features = np.clip(features,
#: -100, 100)` in MIA_experiments.ipynb -- so this reproduces it. It is a guard against a
#: saturated softmax yielding +-inf, not a substantive cap: the largest per-point loss
#: measured on any sweep here is 21.0, so the clip never binds and removing it changes no
#: reported figure (verified by re-scoring a sweep with and without it). Keep it anyway, so
#: a future sweep that does produce an infinite loss degrades the same way theirs would
#: rather than propagating NaN into the classifier.
LOSS_CLIP = 100.0


@torch.no_grad()
def _losses(model: torch.nn.Module, dataset, device: torch.device,
            batch_size: int = 512) -> np.ndarray:
    """Per-example cross-entropy loss over a dataset, in fixed index order.

    Uses audit_utils.compute_phi_and_loss_per_point so the attack sees exactly the loss
    the LiRA audit scores points with (its phi output is unused here), then applies the
    authors' +-LOSS_CLIP guard.
    """
    out = []
    for x, y in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        out.append(compute_phi_and_loss_per_point(model, x, y, device)[1])
    return np.clip(np.concatenate(out), -LOSS_CLIP, LOSS_CLIP)


def _make_classifier(kind: str, seed: int):
    """The attacker. Standardized 1-D loss feature; LogisticRegression by default.

    On one feature a linear model is a threshold rule, which is the intended attack -- the
    classifier is there to pick the threshold on training losses only. `svc` is offered
    because the SCRUB authors' repo uses an RBF SVC for its own MIA helper; it can bend
    around the middle of the loss distribution, and it is ~20x slower.
    """
    if kind == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    if kind == "svc":
        return make_pipeline(StandardScaler(), SVC(C=3, gamma="auto", kernel="rbf",
                                                   random_state=seed))
    raise ValueError(f"unknown classifier {kind!r}; expected logreg or svc")


def run_attack(in_losses: np.ndarray, out_losses: np.ndarray, *, folds: int, seed: int,
               classifier: str) -> dict:
    """One basic MIA: balance the two populations, then cross-validate the attacker.

    Balancing is a uniform subsample of whichever population is larger, down to the size
    of the smaller, drawn from a local Generator so nothing else's RNG state moves. The
    accuracy is the mean over the `folds` held-out folds; `auc` is the threshold attack's
    AUC on the same balanced sample, which needs no training and so says whether the
    signal is there at all independently of how the attacker was fit.

    `auc` scores "in" as the LOW-loss side, the direction membership actually implies, so
    an auc below 0.5 with an accuracy well above 50% means the attacker is winning on the
    inverted rule -- the "in" points are the ones with *higher* loss. That is the
    signature of the class confound (see the module docstring), not of memorization.
    """
    rng = np.random.default_rng(seed)
    n = min(len(in_losses), len(out_losses))
    if n < folds:
        raise ValueError(f"need at least {folds} examples per class, got {n}")
    if len(in_losses) > n:
        in_losses = in_losses[rng.choice(len(in_losses), size=n, replace=False)]
    if len(out_losses) > n:
        out_losses = out_losses[rng.choice(len(out_losses), size=n, replace=False)]

    X = np.concatenate([in_losses, out_losses]).reshape(-1, 1)
    y = np.concatenate([np.ones(n, dtype=int), np.zeros(n, dtype=int)])

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = cross_val_score(_make_classifier(classifier, seed), X, y, cv=cv,
                             scoring="accuracy")
    return {
        # Percent, to match every other number this repo prints. 50 = perfect defence.
        "accuracy": 100.0 * float(scores.mean()),
        "accuracy_std": 100.0 * float(scores.std()),
        "auc": float(roc_auc_score(y, -X.ravel())),  # in-points are the LOW-loss side
        "n_per_class": int(n),
        "mean_loss_in": float(in_losses.mean()),
        "mean_loss_out": float(out_losses.mean()),
    }


def _load_run_json(path: Path, *, classifier: str, folds: int, seed: int):
    """A previously written per-run result, or None if there is none that the current
    attack settings can be compared against. Settings are part of the result, so a file
    written with a different classifier / fold count / seed is ignored rather than mixed
    into an average with results that were not produced the same way."""
    if not path.exists():
        return None
    doc = json.loads(path.read_text())
    if (doc.get("classifier"), doc.get("folds"), doc.get("seed")) != (classifier, folds, seed):
        return None
    return doc.get("variants")


def attack_sweep(sweep: Path, source: str, device: torch.device, *, variants=VARIANTS,
                 folds: int = 5, seed: int = 0, classifier: str = "logreg",
                 batch_size: int = 512, force: bool = False) -> dict:
    """Run the attack against every run of a sweep, writing a per-run JSON as it goes, and
    summarize over runs.

    `force=False` reuses any per-run file already on disk (and checks it was produced with
    the same attack settings), so this is cheap to re-run.
    """
    config = json.loads((sweep / "config.json").read_text())
    dataset, data_dir = config["dataset"], config["data_dir"]

    default_input_size, default_num_classes, default_model_name = dataset_defaults(dataset)
    model = build_model(config.get("model") or default_model_name,
                        config.get("num_classes") or default_num_classes,
                        input_size=config.get("input_size") or default_input_size,
                        filters_percentage=config.get("filters", 1.0)).to(device)

    _, val_set, test_set = load_retain_val_test(dataset, data_dir)
    forget_data, forget_labels = load_full_forget(dataset, data_dir).tensors
    val_labels, test_labels = val_set.tensors[1], test_set.tensors[1]

    filename = _model_filename_from_source(source)
    per_run, missing, reused = {}, [], 0
    for run_idx, run_dir in enumerate(_run_dirs(sweep)):
        path = run_dir / filename
        if not path.exists():
            missing.append(run_dir.name)
            continue

        # One seed per run, so a run's balancing subsample and fold split do not depend on
        # how many runs came before it (or on which variant is being scored) -- which is
        # also what makes a per-run result reusable on its own.
        run_seed = seed + run_idx
        run_out = run_dir / f"mia_basic_{source}.json"
        previous = _load_run_json(run_out, classifier=classifier, folds=folds, seed=run_seed)
        if not force and previous is not None and all(v in previous for v in variants):
            per_run[run_dir.name] = previous
            reused += 1
            continue

        model.load_state_dict(torch.load(path, map_location=device, weights_only=False))
        model.eval()
        forget_loss = _losses(model, TensorDataset(forget_data, forget_labels), device,
                              batch_size)
        val_loss = _losses(model, val_set, device, batch_size)
        test_loss = _losses(model, test_set, device, batch_size)

        # "in": the forget points this run actually trained on. Everything else is drawn
        # from data the model never saw; `_cm` keeps only the forget set's own classes.
        indices = np.load(run_dir / "forget_indices.npy")
        in_mask = np.zeros(len(forget_loss), dtype=bool)
        in_mask[indices] = True
        forget_classes = torch.unique(forget_labels[indices])
        val_cm = torch.isin(val_labels, forget_classes).numpy()
        test_cm = torch.isin(test_labels, forget_classes).numpy()
        populations = {
            "forget_out": (forget_loss[in_mask], forget_loss[~in_mask]),
            "test_cm": (forget_loss[in_mask], test_loss[test_cm]),
            "val_cm": (forget_loss[in_mask], val_loss[val_cm]),
            "test": (forget_loss[in_mask], test_loss),
            # Control, not an attack: BOTH sides are test examples the model never trained
            # on, so the only thing separating them is which class they belong to.
            "class_only": (test_loss[test_cm], test_loss[~test_cm]),
        }

        result = dict(previous or {})  # keep variants this invocation was not asked for
        result.update({
            v: run_attack(*populations[v], folds=folds, seed=run_seed, classifier=classifier)
            for v in variants
        })
        run_out.write_text(json.dumps({
            "generated_by": Path(__file__).name,
            "run": str(run_dir), "model_source": source, "model_filename": filename,
            "classifier": classifier, "folds": folds, "seed": run_seed,
            "variants": result,
        }, indent=2))
        per_run[run_dir.name] = result
        print(f"  {run_dir.name}: " + "  ".join(
            f"{v}={result[v]['accuracy']:.1f}%" for v in variants), flush=True)

    if not per_run:
        raise SystemExit(f"{sweep}: no run has {filename}")
    if reused:
        print(f"  note: {reused} run(s) reused an existing mia_basic_{source}.json "
              f"(--force to re-score)")
    if missing:
        print(f"  note: {len(missing)} run(s) have no {filename} and were skipped: "
              f"{', '.join(missing)}")

    # Aggregate every variant that all of the runs have, not just the ones this invocation
    # asked for: a run scored earlier with a different --variants list keeps its columns in
    # the per-run file, and dropping them from the summary here would silently lose them.
    variants = tuple(v for v in VARIANTS if all(v in r for r in per_run.values()))

    def _over_runs(variant: str, key: str, reduce):
        return float(reduce([r[variant][key] for r in per_run.values()]))

    return {
        "generated_by": Path(__file__).name,
        "attack": "basic MIA (Kurmanji et al. 2023, arXiv:2302.09880)",
        "sweep": str(sweep),
        "model_source": source,
        "model_filename": filename,
        "dataset": dataset,
        "data_dir": data_dir,
        "classifier": classifier,
        "folds": folds,
        "seed": seed,
        "n_runs": len(per_run),
        # Mean over runs of the per-run cross-validated accuracy; `std` is the spread
        # ACROSS runs (how much the leak depends on which points were sampled), not the
        # across-fold spread, which is kept per run as `accuracy_std`.
        "mean": {v: {k: _over_runs(v, k, np.mean)
                     for k in ("accuracy", "auc", "mean_loss_in", "mean_loss_out")}
                 for v in variants},
        "std": {v: {"accuracy": _over_runs(v, "accuracy", np.std)} for v in variants},
        "per_run": per_run,
    }


def _print_summary(docs: dict, variants=VARIANTS) -> None:
    """One row per sweep, so a whole `runs/*/*` pass is readable at a glance."""
    width = max(len(name) for name in docs) if docs else 0
    print(f"\n{'sweep':<{width}}  " + "  ".join(f"{v:>12s}" for v in variants))
    print("-" * (width + 2 + 14 * len(variants)))
    for name, doc in docs.items():
        cells = "  ".join(f"{doc['mean'][v]['accuracy']:6.2f}±{doc['std'][v]['accuracy']:<5.2f}"
                          for v in variants)
        print(f"{name:<{width}}  {cells}")
    print("\nAttack accuracy %, mean ± std over runs. 50 = attacker cannot separate the two "
          "sets.\ntest is class-confounded for a class-based forget split -- see the module "
          "docstring.")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sweeps", nargs="+", help="Sweep directories (runs/<method>/<sweep>)")
    p.add_argument("--source", default="unlearned",
                   help="Checkpoint to attack: unlearned (default), trained, or unlearn_epoch_N")
    p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS),
                   help="Which 'out' populations to attack with")
    p.add_argument("--classifier", default="logreg", choices=("logreg", "svc"))
    p.add_argument("--folds", type=int, default=5, help="Stratified CV folds (default 5)")
    p.add_argument("--seed", type=int, default=0,
                   help="Base seed for balancing subsample and fold splits; run i uses seed+i")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--force", action="store_true", default=False,
                   help="Re-score every run instead of reusing its per-run cache file")
    p.add_argument("--summary-out", default=None,
                   help="Also write one JSON holding every sweep's `mean`/`std` block")
    args = p.parse_args()

    device = torch.device(args.device)
    docs = {}
    for sweep in (Path(s) for s in args.sweeps):
        # No sweep-level skip: the per-run files are the cache, so re-running only costs
        # the runs that do not have one yet, and a sweep that gained runs picks them up.
        out_path = sweep / f"mia_basic_{args.source}.json"
        print(f"### {sweep} @ {args.source}")
        doc = attack_sweep(sweep, args.source, device, variants=tuple(args.variants),
                           folds=args.folds, seed=args.seed, classifier=args.classifier,
                           batch_size=args.batch_size, force=args.force)
        out_path.write_text(json.dumps(doc, indent=2))
        print(f"  -> {out_path}  mean over {doc['n_runs']} runs: " + "  ".join(
            f"{v}={doc['mean'][v]['accuracy']:.2f}%" for v in args.variants))
        docs[str(sweep)] = doc

    present = [v for v in args.variants if all(v in d["mean"] for d in docs.values())]
    _print_summary(docs, tuple(present))

    if args.summary_out:
        summary = {
            "generated_by": Path(__file__).name,
            "model_source": args.source,
            "classifier": args.classifier,
            "folds": args.folds,
            "seed": args.seed,
            "sweeps": {name: {k: doc[k] for k in ("n_runs", "mean", "std")}
                       for name, doc in docs.items()},
        }
        Path(args.summary_out).write_text(json.dumps(summary, indent=2))
        print(f"\n-> {args.summary_out}")


if __name__ == "__main__":
    main()
