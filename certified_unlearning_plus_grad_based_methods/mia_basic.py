"""
The "basic MIA" of Kurmanji et al., *Towards Unbounded Machine Unlearning*, NeurIPS 2023
([arXiv:2302.09880](https://arxiv.org/abs/2302.09880)), run over the logs_* sweeps
produced by this repo.

The paper's description, verbatim:

    We train a binary classifier (the 'attacker') on the unlearned model's losses on
    forget and test examples for the objective of classifying 'in' (forget) versus 'out'
    (test). The attacker then makes predictions for held-out losses (losses the attacker
    wasn't trained on) that are balanced between forget and test losses. For a perfect
    defense, the attacker's accuracy is 50%, indicating that it is unable to separate the
    two sets.

So, per run: score the checkpoint on both populations, balance them, and cross-validate a
binary classifier on the one-dimensional loss feature. "Held-out losses the attacker wasn't
trained on" is implemented as cross-validation, so every loss is predicted by an attacker
that did not see it.

This is a different instrument from the rest of the audit stack and is not a replacement
for it. The eps/rho/mu machinery in `audit_utils.py` fits a per-point in/out distribution
across the runs of a sweep and turns the result into a certified lower bound. This is one
number per model, from that model alone, with no cross-run information and no bound
attached -- cheap, standard, and directly comparable with the numbers unlearning papers
publish. The two are NOT on the same scale.


## 1. Forget-split type decides the variant set

A forget pool can be class-centric (e.g. 4500 points over 10 of 100 CIFAR-100 classes) or
uniform (4500 points over all 100 classes). The branch is decided PER SWEEP from the
labels of that sweep's cached forget pool -- never from the directory name.
`VARIANTS_CLASS_CENTRIC` is used when the forget pool spans a strict subset of the label
space, `VARIANTS_UNIFORM` otherwise.

Class-centric. All five populations. The first four share their "in" side -- this run's
sampled forget points -- and differ only in "out":

  forget_out  the forget points this run did NOT sample. Same classes, same split, same
              preprocessing; they differ from the "in" points by training membership and
              nothing else, which is the same in/out structure the eps audit tests. This
              is the variant where 50% unambiguously means "indistinguishable".
  test_cm     test examples of the forget pool's own classes. This matches the
              class-matched filtering in Kurmanji et al.'s released MIA code.
  val_cm      validation examples of those classes.
  test        every test example. The paper's sentence read literally, and
              CLASS-CONFOUNDED: after class unlearning the model is bad on the forget
              classes and ordinary on the rest, so "high loss" means "forgotten class",
              not "was trained on".
  class_only  NOT an attack: the control that calibrates that confound. Both sides are test
              examples the model never trained on -- forget-class ones labelled "in",
              other-class ones labelled "out" -- so the membership signal is exactly zero
              by construction and whatever accuracy comes back is the class channel alone.
              Always report it next to `test`.

Uniform. There is no class confound, and two variants degenerate: `test_cm` would be
identical to `test`, and `class_only` is undefined because there is no distinguished class
set. Only `forget_out`, `test` and `val` are scored.


## 2. Membership is recorded per BATCH, not per point

Each run stores `chosen_forget_batches.npy`, indexing whole cached batch files under
`<cache_root>/forget/batch_*.pkl` -- the same objects
`src/utils/data_cache.py:load_cifar_splits_with_batch_subset` selects.
`_point_membership_mask` expands those batch indices to a point mask using the actual
on-disk batch sizes, in the `sorted(glob("batch_*.pkl"))` order that both the data
pipeline and `audit_utils.load_batches` use, so index i of the loss vector is index i of
the pool.

Consequence: for a `bs_N` split, members arrive in clusters of N. With N=1 membership is
per point; with N=10 the 4500 points are only 450 independent coin flips, so the
across-run std this script reports is honest but the within-run sample is correlated.


## 3. Which points the unlearning step touches

The forget loader handed to the unlearning routine contains only the run's sampled forget
batches (`forget_loader_scope="chosen"` in `data_cache.py`). For `retain_finetune` it is
used only for evaluation; for `ascent_descent` it drives the ascent / combined steps. In
both cases the non-sampled forget points are never seen by the mechanism, so `forget_out`
compares points that differ in training membership and nothing else.


## 4. The attacker

StandardScaler -> LogisticRegression(max_iter=1000), StratifiedKFold(5, shuffle=True), with
the loss feature clipped to [-100, 100] as in Kurmanji et al.'s released code.

On a one-dimensional feature the clip is monotone, so it cannot change the threshold AUC;
what it changes is where LogisticRegression puts its decision boundary when a handful of
very large losses would otherwise dominate the fit. Gradient-ascent unlearning pushes
forget losses up without bound, so every result records `n_clipped_in` /
`n_clipped_out` / `max_loss_in` / `max_loss_out`.

`--cv`, `--clip`, `--no-clip`, `--scale`, `--no-scale` and `--max-iter` override any of
this (e.g. `--cv shuffle --no-scale --max-iter 100` gives the unscaled
StratifiedShuffleSplit setting of the released code). The full resolved setting is stored
in every result file and checked before a cached result is reused.


## 5. Implementation notes

  - Checkpoints are Orbax directories (`ckpt/checkpoint_N`, `ckpt/unlearn_epoch_N`).
    `restore_orbax_state` needs a visible GPU: it fails CPU-only with "sharding passed to
    deserialization should be specified".
  - The per-run seed is `--seed + run_vars["run_index"]`, not `--seed + enumeration_index`.
    Run directories are random wandb ids, so enumeration order would shift -- and
    invalidate every cached per-run result -- the moment a sweep gained a run. When
    `run_index` is absent the script falls back to enumeration order and records it.
  - `forget_classes` is taken from the whole forget pool rather than from the sampled half,
    so the `_cm` populations do not depend on the coin flip.
  - Retain / forget-in / forget-out / val / test accuracy of the same checkpoint are
    computed in the same forward pass and stored alongside the attack: a model whose
    utility has collapsed is attacked at chance for reasons that have nothing to do with
    privacy. `--no-retain-acc` skips the retain pass.
  - The loss statistic is `audit_utils._compute_loss_per_sample`, the same function
    `audit_utils.make_eval_step_per_point` uses, so the attack scores points with exactly
    the statistic the eps/rho/mu audit does.

Sources accepted by --source: `unlearned` (highest ckpt/unlearn_epoch_N), `trained`
(highest ckpt/checkpoint_N), `unlearn_epoch_N`, `checkpoint_N`.

Outputs, none of which overwrite anything already on disk:

  <sweep>/<run>/mia_basic_<source>.json   that run's attack, every variant, plus the
                                          accuracies and the resolved attack settings.
                                          ALSO the resume cache: a run that has one is not
                                          re-scored unless --force.
  <sweep>/mia_basic_<source>.json         mean/std over the sweep's runs plus a copy of
                                          every per-run result
  <--summary-out>                         one JSON across sweeps

Usage:
    python mia_basic.py --source unlearned logs_*_batch_1
    python mia_basic.py --source trained --summary-out tables/mia_basic_trained.json logs_*_batch_1
    python mia_basic.py --source unlearned --shuffle-labels logs_cifar100_retain_finetune_batch_1
    python make_mia_table.py --source unlearned --out tables/mia_basic.tex logs_*_batch_1
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

import jax
import jax.numpy as jnp
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (StratifiedKFold, StratifiedShuffleSplit,
                                     cross_val_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Same-repo helpers, so there is exactly one definition of each of these. audit_utils is
# imported only; nothing in it is modified by this file.
#   _compute_loss_per_sample  the per-point loss make_eval_step_per_point uses, i.e. the
#                             statistic the eps/rho/mu audit scores points with
#   load_batches / rebatch    the cached-split readers the audit uses, in the same order
#   get_cache_root            run_vars.data_subfolder -> data_split directory
from audit_utils import (_compute_loss_per_sample, get_cache_root,
                         infer_dataset_type, load_batches, load_run_vars,
                         rebatch, restore_orbax_state)
from src.models.model import ModelFactory

#: Scored when the forget pool spans a strict subset of the label space. See section 1 --
#: these are not interchangeable, `test` is the confounded one, and `class_only` is a
#: zero-membership control rather than an attack.
VARIANTS_CLASS_CENTRIC = ("forget_out", "test_cm", "val_cm", "test", "class_only")

#: Scored when the forget pool spans every class. `test_cm` would duplicate `test` and
#: `class_only` is undefined, so both are dropped rather than reported as zeros.
VARIANTS_UNIFORM = ("forget_out", "test", "val")

ALL_VARIANTS = tuple(dict.fromkeys(VARIANTS_CLASS_CENTRIC + VARIANTS_UNIFORM))

#: The attacker. One configuration, chosen on measured grounds (see section 4): stratified
#: k-fold cross-validation, and Kurmanji et al.'s clipping of the loss feature to
#: [-100, 100]. Every field is overridable from the command line; the resolved dict is what
#: gets stored in each result and compared before a cached result is reused.
ATTACKER = {"scale": True, "max_iter": 1000, "cv": "kfold", "clip": [-100.0, 100.0]}


# ---------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------
def iter_runs(sweep: Path):
    """Run directories of a sweep, in a stable order: direct children then test_run/.

    `test_run/` holds the held-out runs the audit uses as its test split, so they are
    included. `ignored_runs/` is deliberately NOT included.
    """
    for parent in (sweep, sweep / "test_run"):
        if not parent.is_dir():
            continue
        for p in sorted(parent.iterdir()):
            if p.is_dir() and (p / "ckpt").is_dir() and (p / "run_vars.json").exists():
                yield p


def infer_model_dataset_and_classes(sweep: Path):
    """(model name, dataset name, n_classes, config path) from the sweep's config.yaml."""
    config_path = Path(sweep) / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Missing required experiment config: {config_path}. "
            "Expected model.name, model.n_classes, and dataset.name.")
    cfg = yaml.safe_load(config_path.read_text()) or {}
    model_name = cfg.get("model", {}).get("name")
    n_classes = cfg.get("model", {}).get("n_classes")
    dataset_name = cfg.get("dataset", {}).get("name")
    if model_name is None or n_classes is None or dataset_name is None:
        raise ValueError(
            f"Invalid config {config_path}: requires model.name, model.n_classes, dataset.name")
    return str(model_name), str(dataset_name), int(n_classes), str(config_path)


def resolve_forget_indices_for_run(run_dir: Path):
    """(chosen forget batch indices, file they came from) for one run."""
    path = Path(run_dir) / "chosen_forget_batches.npy"
    if not path.exists():
        raise FileNotFoundError(f"No chosen_forget_batches.npy in {run_dir}")
    return np.asarray(np.load(path), dtype=int), str(path)


# ---------------------------------------------------------------------
# Scoring a checkpoint
# ---------------------------------------------------------------------
def _make_loss_acc_step(model):
    """Per-point cross-entropy loss AND correctness from one forward pass.

    The loss is `audit_utils._compute_loss_per_sample`, the same call
    `audit_utils.make_eval_step_per_point` makes, so the number the attacker sees is
    numerically the statistic the audit stack scores points with. The only reason this is
    not `make_eval_step_per_point` itself is that it returns phi and loss but no accuracy,
    and the accuracy has to come out of this forward pass rather than a second one.
    """
    nc = model.num_classes

    @jax.jit
    def _step(params, x, y):
        logits = model.apply({"params": params}, x, train=False)
        return (_compute_loss_per_sample(logits, y, num_classes=nc),
                jnp.argmax(logits, -1) == y)

    return _step


def _split_losses(step, params, batches, eval_batch_size):
    """(per-point loss, per-point correct) over a cached split, in on-disk file order.

    `rebatch` concatenates and re-slices, so order is preserved exactly; the final short
    batch costs one extra JIT specialization per split and nothing else.
    """
    jax_batches = [(jnp.asarray(x), jnp.asarray(y)) for x, y in batches]
    losses, correct = [], []
    for x, y in rebatch(jax_batches, eval_batch_size, drop_remainder=False):
        loss, ok = step(params, x, y)
        losses.append(np.asarray(loss))
        correct.append(np.asarray(ok))
    return np.concatenate(losses), np.concatenate(correct)


def _split_labels(batches):
    return np.concatenate([np.asarray(y) for _, y in batches])


def _point_membership_mask(batches, chosen_batch_idx):
    """Expand `chosen_forget_batches.npy` (batch indices) to a per-point boolean mask.

    Batch sizes are read off the loaded batches rather than assumed uniform, so a split
    whose last cached batch is short still lines up with the loss vector.
    """
    sizes = [len(np.asarray(y)) for _, y in batches]
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    mask = np.zeros(int(offsets[-1]), dtype=bool)
    idx = np.asarray(chosen_batch_idx, dtype=int)
    if idx.size == 0:
        raise ValueError("chosen forget batch index array is empty")
    if idx.min() < 0 or idx.max() >= len(sizes):
        raise IndexError(f"chosen forget batch indices out of range [0, {len(sizes) - 1}]")
    for i in idx:
        mask[offsets[i]:offsets[i + 1]] = True
    return mask


# ---------------------------------------------------------------------
# The attacker
# ---------------------------------------------------------------------
def resolve_attacker(args) -> dict:
    """The attacker spec plus any command-line override, as one flat dict.

    This dict is what is written into every result and compared against before a cached
    result is reused, so a file produced with a different clip or CV scheme is re-scored
    instead of being averaged in with results that were not produced the same way.
    """
    spec = dict(ATTACKER)
    if args.scale is not None:
        spec["scale"] = args.scale
    if args.max_iter is not None:
        spec["max_iter"] = args.max_iter
    if args.cv is not None:
        spec["cv"] = args.cv
    if args.no_clip:
        spec["clip"] = None
    elif args.clip is not None:
        spec["clip"] = [float(args.clip[0]), float(args.clip[1])]
    spec["folds"] = args.folds
    spec["test_size"] = args.test_size
    return spec


def _make_classifier(spec: dict):
    """The attacker. A one-dimensional loss feature, so a linear model is a threshold rule;
    the classifier is only there to pick the threshold from training losses.

    `scale=False` reproduces Kurmanji et al.'s bare LogisticRegression(). With no scaler the fit
    is sensitive to the tail of the loss distribution, which is exactly why they clip.
    """
    clf = LogisticRegression(max_iter=spec["max_iter"])
    return make_pipeline(StandardScaler(), clf) if spec["scale"] else clf


def _make_cv(spec: dict, seed: int):
    if spec["cv"] == "kfold":
        return StratifiedKFold(n_splits=spec["folds"], shuffle=True, random_state=seed)
    if spec["cv"] == "shuffle":
        # Kurmanji et al.'s call is StratifiedShuffleSplit(n_splits=5, random_state=seed), which
        # leaves test_size at the sklearn default of 0.1. Held-out folds overlap here,
        # unlike k-fold; that is the authors' choice, reproduced rather than corrected.
        return StratifiedShuffleSplit(n_splits=spec["folds"], test_size=spec["test_size"],
                                      random_state=seed)
    raise ValueError(f"unknown cv scheme {spec['cv']!r}; expected kfold or shuffle")


def run_attack(in_losses, out_losses, *, spec: dict, seed: int,
               shuffle_labels: bool = False) -> dict:
    """One basic MIA: balance the two populations, then cross-validate the attacker.

    Balancing is a uniform subsample of whichever population is larger, down to the size of
    the smaller, drawn from a local Generator so nothing else's RNG state moves.

    `auc` is the untrained threshold attack on the same balanced sample, scoring "in" as
    the LOW-loss side -- the direction membership actually implies. An auc below 0.5
    alongside an accuracy well above 50 means the attacker is winning on the inverted rule
    ("high loss => member"), which is the signature of the class confound on a class-
    centric split and a bug anywhere else. It is computed on the RAW losses: clipping is
    monotone, so it would not move the AUC except through ties introduced at the cap.

    `shuffle_labels=True` destroys the in/out correspondence while leaving both loss
    distributions untouched, so the result must come back at 50 +- ~1.
    It is a flag rather than a test so it can be run against real sweeps.
    """
    rng = np.random.default_rng(seed)
    n = min(len(in_losses), len(out_losses))
    if n < spec["folds"]:
        raise ValueError(f"need at least {spec['folds']} examples per class, got {n}")
    if len(in_losses) > n:
        in_losses = in_losses[rng.choice(len(in_losses), size=n, replace=False)]
    if len(out_losses) > n:
        out_losses = out_losses[rng.choice(len(out_losses), size=n, replace=False)]

    X_raw = np.concatenate([in_losses, out_losses]).reshape(-1, 1)
    y = np.concatenate([np.ones(n, dtype=int), np.zeros(n, dtype=int)])
    if shuffle_labels:
        y = y[rng.permutation(len(y))]

    clip = spec["clip"]
    X = np.clip(X_raw, clip[0], clip[1]) if clip else X_raw

    scores = cross_val_score(_make_classifier(spec), X, y, cv=_make_cv(spec, seed),
                             scoring="accuracy")
    return {
        # Percent, to match every other number this repo prints. 50 = perfect defence.
        "accuracy": 100.0 * float(scores.mean()),
        "accuracy_std": 100.0 * float(scores.std()),
        "auc": float(roc_auc_score(y, -X_raw.ravel())),  # in-points are the LOW-loss side
        "n_per_class": int(n),
        "mean_loss_in": float(in_losses.mean()),
        "mean_loss_out": float(out_losses.mean()),
        # How hard the clip actually bit. Both zero => the clip changed nothing and this
        # attacker differs from `repo` only in scaler / CV scheme / max_iter.
        "max_loss_in": float(in_losses.max()),
        "max_loss_out": float(out_losses.max()),
        "n_clipped_in": int(0 if not clip else np.sum((in_losses < clip[0]) |
                                                      (in_losses > clip[1]))),
        "n_clipped_out": int(0 if not clip else np.sum((out_losses < clip[0]) |
                                                       (out_losses > clip[1]))),
    }


# ---------------------------------------------------------------------
# Sweep plumbing
# ---------------------------------------------------------------------
def _resolve_checkpoint(run_dir: Path, source: str):
    """Orbax checkpoint directory for a source string, or None if the run does not have it.

    Checkpoints are directories under `ckpt/` with a numeric suffix, and the two
    families (`checkpoint_N` from training, `unlearn_epoch_N` from unlearning) share a
    prefix, so "highest N" has to be resolved per family.
    """
    ckpt = run_dir / "ckpt"
    if not ckpt.is_dir():
        return None

    def _highest(prefix, exclude_prefix=None):
        best = None
        for item in ckpt.iterdir():
            if not item.is_dir() or not item.name.startswith(prefix):
                continue
            if exclude_prefix and item.name.startswith(exclude_prefix):
                continue
            try:
                num = int(item.name[len(prefix):])
            except ValueError:
                continue
            if best is None or num > best[0]:
                best = (num, item)
        return best

    if source == "unlearned":
        best = _highest("unlearn_epoch_")
        return best[1] if best else None
    if source == "trained":
        best = _highest("checkpoint_", exclude_prefix="unlearn_")
        return best[1] if best else None
    path = ckpt / source
    return path if path.is_dir() else None


def cfg_algorithm(sweep: Path) -> str:
    """unlearning.algorithm from the sweep's own config.yaml."""
    import yaml
    try:
        return (yaml.safe_load((sweep / "config.yaml").read_text()) or {}) \
            .get("unlearning", {}).get("algorithm", "unknown")
    except Exception:
        return "unknown"


def _sweep_context(sweep: Path, runs):
    """Everything about a sweep that does not depend on which checkpoint is attacked:
    the data split, the label vectors, and therefore which branch of section 1 applies."""
    model_name, dataset, num_classes, config_path = infer_model_dataset_and_classes(sweep)
    run_vars = load_run_vars(runs[0])
    dataset_type = infer_dataset_type(run_vars, str(sweep), run_dir=runs[0],
                                      config_path=config_path)
    cache_root = get_cache_root(str(sweep), dataset_type=dataset_type)

    splits = {s: load_batches(cache_root, s) for s in ("forget", "val", "test")}
    labels = {s: _split_labels(b) for s, b in splits.items()}

    # Section 1: the branch is measured, not inferred from the sweep's name. Classes come
    # from the whole pool, not from any one run's sampled half.
    forget_classes = np.unique(labels["forget"])
    class_centric = len(forget_classes) < num_classes
    variants = VARIANTS_CLASS_CENTRIC if class_centric else VARIANTS_UNIFORM

    # Provenance: which algorithm produced these checkpoints. See section 3.
    algorithm = str(cfg_algorithm(sweep))

    return {
        "model": ModelFactory.create_model(model_name=model_name, num_classes=num_classes),
        "model_name": model_name, "dataset": dataset, "num_classes": num_classes,
        "algorithm": algorithm,
        "dataset_type": dataset_type, "cache_root": cache_root,
        "splits": splits, "labels": labels,
        "forget_classes": forget_classes, "class_centric": class_centric,
        "variants": variants,
        "cm_mask": {s: np.isin(labels[s], forget_classes) for s in ("val", "test")},
    }


def _populations(ctx, loss, in_mask):
    """{variant: (in_losses, out_losses)} for the branch this sweep is on."""
    f, v, t = loss["forget"], loss["val"], loss["test"]
    pops = {
        "forget_out": (f[in_mask], f[~in_mask]),
        "test": (f[in_mask], t),
        "val": (f[in_mask], v),
    }
    if ctx["class_centric"]:
        pops["test_cm"] = (f[in_mask], t[ctx["cm_mask"]["test"]])
        pops["val_cm"] = (f[in_mask], v[ctx["cm_mask"]["val"]])
        # Control, not an attack: BOTH sides are test examples the model never trained on,
        # so the only thing separating them is which class they belong to.
        pops["class_only"] = (t[ctx["cm_mask"]["test"]], t[~ctx["cm_mask"]["test"]])
    return {k: pops[k] for k in ctx["variants"]}


def _run_seed(run_dir: Path, base_seed: int, fallback_idx: int):
    """(seed, how it was derived). See section 5: enumeration order is not stable
    because run directories are random wandb ids, so the pipeline's own run_index is
    preferred and the fallback is recorded when it is missing."""
    try:
        run_index = load_run_vars(run_dir).get("run_index")
    except Exception:
        run_index = None
    if isinstance(run_index, int):
        return base_seed + run_index, "run_index"
    return base_seed + fallback_idx, "enumeration"


def _load_cached(path: Path, spec: dict, seed: int, source: str):
    """A previously written per-run result, or None if it was not produced by the attack
    settings in force now."""
    if not path.exists():
        return None
    try:
        doc = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    if doc.get("attacker") != spec or doc.get("seed") != seed:
        return None
    if doc.get("model_source") != source:
        return None
    return doc


def attack_sweep(sweep: Path, source: str, *, specs: dict, suffixes: dict,
                 base_seed: int, variants=None, eval_batch_size: int = 1000,
                 force: bool = False, retain_acc: bool = True,
                 shuffle_labels: bool = False, max_runs: int = 0,
                 write: bool = True) -> dict:
    """Attack every run of a sweep with every attacker in `specs`, writing a per-run JSON
    per attacker as it goes, and summarize. Returns {attacker name: sweep result}.

    `specs` is a dict rather than a single attacker on purpose: the expensive part of a run
    is the Orbax restore plus the 60,000-image forward pass, and that does not depend on
    which classifier is fitted afterwards. Scoring both attackers in one invocation
    therefore costs one pass, not two. Every attacker still gets its own per-run and
    per-sweep file, keyed by `suffixes`.

    `force=False` reuses any per-run file whose stored settings match, so this is cheap to
    re-run and an interrupted pass picks up where it stopped; the forward pass is skipped
    only when EVERY attacker's file is already there. `max_runs` and `write=False` exist
    for smoke-testing the plumbing without producing a partial average on disk; a result
    produced with either is not a result, and `n_runs_total` will expose it.
    """
    runs = list(iter_runs(sweep))
    if not runs:
        raise SystemExit(f"{sweep}: no run directories with a ckpt/ found")
    if max_runs:
        runs = runs[:max_runs]

    ctx = _sweep_context(sweep, runs)
    wanted = tuple(v for v in (variants or ctx["variants"]) if v in ctx["variants"])
    if not wanted:
        raise SystemExit(
            f"{sweep}: none of the requested variants apply to a "
            f"{'class-centric' if ctx['class_centric'] else 'uniform'} forget pool "
            f"(available: {', '.join(ctx['variants'])})")

    step = _make_loss_acc_step(ctx["model"])
    retain_batches = load_batches(ctx["cache_root"], "retain") if retain_acc else None

    per_run = {name: {} for name in specs}
    missing, reused = [], 0
    for fallback_idx, run_dir in enumerate(runs):
        ckpt = _resolve_checkpoint(run_dir, source)
        if ckpt is None:
            missing.append(run_dir.name)
            continue

        seed, seed_from = _run_seed(run_dir, base_seed, fallback_idx)

        # Which attackers still need work for this run. If none do, the forward pass is
        # skipped entirely -- this is what makes re-running cheap.
        paths = {n: run_dir / f"mia_basic_{source}{suffixes[n]}.json" for n in specs}
        cached, todo = {}, []
        for name, spec in specs.items():
            doc = None if force else _load_cached(paths[name], spec, seed, source)
            if doc is not None and all(v in doc.get("variants", {}) for v in wanted):
                cached[name] = doc
            else:
                todo.append(name)
        if not todo:
            for name, doc in cached.items():
                per_run[name][run_dir.name] = doc
            reused += 1
            continue

        state = restore_orbax_state(str(ckpt.resolve()))
        params = state["params"] if isinstance(state, dict) else state.params
        del state

        loss, correct = {}, {}
        for split in ("forget", "val", "test"):
            loss[split], correct[split] = _split_losses(step, params, ctx["splits"][split],
                                                        eval_batch_size)

        chosen, indices_file = resolve_forget_indices_for_run(run_dir)
        in_mask = _point_membership_mask(ctx["splits"]["forget"], chosen)
        populations = _populations(ctx, loss, in_mask)

        # Utility of the SAME checkpoint. Without these an attack number is unreadable:
        # a collapsed model is attacked at chance for reasons unrelated to privacy.
        accuracy = {
            "forget_in": float(correct["forget"][in_mask].mean()),
            "forget_out": float(correct["forget"][~in_mask].mean()),
            "val": float(correct["val"].mean()),
            "test": float(correct["test"].mean()),
        }
        if retain_batches is not None:
            _, retain_ok = _split_losses(step, params, retain_batches, eval_batch_size)
            accuracy["retain"] = float(retain_ok.mean())

        for name in specs:
            if name in cached:
                per_run[name][run_dir.name] = cached[name]
                continue
            spec = specs[name]
            result = dict((cached.get(name) or {}).get("variants", {}))
            result.update({
                v: run_attack(*pops, spec=spec, seed=seed, shuffle_labels=shuffle_labels)
                for v, pops in populations.items() if v in wanted
            })
            doc = {
                "generated_by": Path(__file__).name,
                "run": str(run_dir), "model_source": source, "checkpoint": str(ckpt),
                "attacker": spec, "seed": seed, "seed_from": seed_from,
                "shuffle_labels": shuffle_labels,
                "forget_indices_file": indices_file,
                "n_forget": int(len(in_mask)), "n_forget_in": int(in_mask.sum()),
                "accuracy": accuracy, "variants": result,
            }
            if write:
                paths[name].write_text(json.dumps(doc, indent=2))
            per_run[name][run_dir.name] = doc

        head = next(iter(specs))
        print(f"  {run_dir.name}: " + "  ".join(
            f"{v}={per_run[head][run_dir.name]['variants'][v]['accuracy']:.1f}%"
            for v in wanted)
            + f"   [R/F/T {accuracy.get('retain', float('nan')):.3f}"
              f"/{accuracy['forget_in']:.3f}/{accuracy['test']:.3f}]"
            + (f"  ({head})" if len(specs) > 1 else ""), flush=True)

    if not any(per_run.values()):
        raise SystemExit(f"{sweep}: no run has a '{source}' checkpoint")
    if reused:
        print(f"  note: {reused} run(s) reused existing per-run files (--force to re-score)")
    if missing:
        print(f"  note: {len(missing)} run(s) have no '{source}' checkpoint and were "
              f"skipped: {', '.join(missing)}")

    return {name: _summarize(sweep, ctx, source, specs[name], per_run[name], runs,
                             missing, base_seed, shuffle_labels)
            for name in specs}


def _summarize(sweep, ctx, source, spec, per_run, runs, missing, base_seed,
               shuffle_labels) -> dict:
    """Sweep-level mean/std over the runs of one attacker."""
    # Aggregate every variant all runs have, not just the ones this invocation asked for:
    # a run scored earlier with a wider --variants keeps its columns in the per-run file,
    # and dropping them from the summary here would silently lose them.
    present = tuple(v for v in ALL_VARIANTS
                    if all(v in r["variants"] for r in per_run.values()))

    def _over(variant, key, reduce):
        return float(reduce([r["variants"][variant][key] for r in per_run.values()]))

    def _acc(key, reduce):
        vals = [r["accuracy"][key] for r in per_run.values() if key in r["accuracy"]]
        return float(reduce(vals)) if vals else None

    return {
        "generated_by": Path(__file__).name,
        "attack": "basic MIA (Kurmanji et al. 2023, arXiv:2302.09880)",
        "sweep": str(sweep), "model_source": source,
        "dataset": ctx["dataset"], "model": ctx["model_name"],
        "num_classes": ctx["num_classes"], "data_subfolder": ctx["dataset_type"],
        "cache_root": ctx["cache_root"],
        # Section 1: recorded so no reader has to re-derive which branch a row is on.
        "forget_pool_classes": int(len(ctx["forget_classes"])),
        "forget_split": "class_centric" if ctx["class_centric"] else "uniform",
        "unlearning_algorithm": ctx["algorithm"],
        "attacker": spec, "seed": base_seed, "shuffle_labels": shuffle_labels,
        # A sweep missing runs is PARTIAL and must not be averaged against a complete one. make_mia_table.py drops these by default.
        "n_runs": len(per_run), "n_runs_total": len(runs),
        "n_runs_missing": len(missing), "runs_missing": missing,
        "complete": not missing,
        "mean": {v: {k: _over(v, k, np.mean) for k in
                     ("accuracy", "auc", "mean_loss_in", "mean_loss_out",
                      "max_loss_in", "max_loss_out", "n_clipped_in", "n_clipped_out")}
                 for v in present},
        # `std` is the spread ACROSS runs (how much the leak depends on which half was
        # sampled), not the across-fold spread, which is kept per run as `accuracy_std`.
        "std": {v: {"accuracy": _over(v, "accuracy", np.std)} for v in present},
        # Median across runs. Reported alongside the mean for forget_out because the two
        # diverging is the signature of a skewed run distribution -- a handful of runs
        # behaving differently rather than a shifted centre.
        "median": {v: {"accuracy": _over(v, "accuracy", np.median)} for v in present},
        "accuracy": {k: _acc(k, np.mean)
                     for k in ("retain", "forget_in", "forget_out", "val", "test")},
        "per_run": per_run,
    }


# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------
def _print_summary(docs: dict) -> None:
    """One row per sweep, grouped by branch, since the two have different columns."""
    for branch, variants in (("class_centric", VARIANTS_CLASS_CENTRIC),
                             ("uniform", VARIANTS_UNIFORM)):
        rows = {n: d for n, d in docs.items() if d["forget_split"] == branch}
        if not rows:
            continue
        width = max(len(n) for n in rows)
        cols = [v for v in variants if all(v in d["mean"] for d in rows.values())]
        print(f"\n[{branch} forget pool]")
        print(f"{'sweep':<{width}}  " + "  ".join(f"{v:>13s}" for v in cols)
              + "     R/F/T")
        print("-" * (width + 2 + 15 * len(cols) + 22))
        for name, doc in rows.items():
            cells = "  ".join(
                f"{doc['mean'][v]['accuracy']:6.2f}+-{doc['std'][v]['accuracy']:<5.2f}"
                for v in cols)
            a = doc["accuracy"]
            r = "n/a" if a.get("retain") is None else f"{100 * a['retain']:.1f}"
            flag = "" if doc["complete"] else f"  PARTIAL({doc['n_runs']}/{doc['n_runs_total']})"
            print(f"{name:<{width}}  {cells}     {r}/{100 * a['forget_in']:.1f}/"
                  f"{100 * a['test']:.1f}{flag}")

    print("\nAttack accuracy %, mean +- std over runs. 50 = the attacker cannot separate "
          "the two sets.\nR/F/T = retain / sampled-forget / test accuracy % of the same "
          "checkpoint.")
    if any(d["forget_split"] == "class_centric" for d in docs.values()):
        print("class_only is a zero-membership CONTROL, not an attack; `test` is "
              "class-confounded.\nRead them together -- see the module docstring.")


def _print_checks(docs: dict) -> None:
    """Sanity checks, computed rather than asserted in prose. Anything printed here is a
    thing to look at, not necessarily a failure."""
    print("\n--- sanity checks ---")
    for name, doc in docs.items():
        notes = []
        m = doc["mean"]
        if "forget_out" in m and "test_cm" in m:
            gap = abs(m["forget_out"]["accuracy"] - m["test_cm"]["accuracy"])
            notes.append(f"|forget_out - test_cm| = {gap:.2f}"
                         + ("  <-- >1.0, populations may be contaminated" if gap > 1.0 else ""))
        # Under --shuffle-labels the AUC is computed against permuted labels, so it
        # straddles 0.5 by design; flagging it there would be noise.
        for v in () if doc["shuffle_labels"] else ("forget_out", "test_cm", "val_cm", "val"):
            if v in m and m[v]["auc"] < 0.5:
                notes.append(f"{v} auc = {m[v]['auc']:.3f} < 0.5 on an "
                             f"unconfounded variant  <-- investigate")
        chance = 100.0 / doc["num_classes"]
        retain = doc["accuracy"].get("retain")
        if retain is not None and 100 * retain < 2 * chance:
            fo = m.get("forget_out", {}).get("accuracy")
            notes.append(f"retain acc {100 * retain:.2f}% is near chance "
                         f"({chance:.2f}%)"
                         + (f"; forget_out = {fo:.2f}" if fo is not None else "")
                         + ("  <-- collapsed model must attack at ~50"
                            if fo is not None and abs(fo - 50) > 1 else ""))
        if doc["attacker"]["clip"]:
            # Max over variants, not sum: the variants overlap heavily (they share their
            # "in" side), so summing would double-count the same points.
            worst = max(m[v]["n_clipped_in"] + m[v]["n_clipped_out"] for v in m)
            notes.append(f"clip {doc['attacker']['clip']} touched at most {worst:.1f} "
                         f"points/run in any variant"
                         + (" (no-op on this sweep)" if worst == 0 else
                            "  <-- the clip is binding here; compare against --no-clip"))
        # An accuracy at chance with a clearly informative AUC means the FIT failed, not
        # that the model is private.
        for v in m:
            if abs(m[v]["accuracy"] - 50) < 1.0 and m[v]["auc"] > 0.55:
                notes.append(f"estimator {v}: accuracy {m[v]['accuracy']:.2f} ~ chance but "
                             f"auc {m[v]['auc']:.3f}; max loss "
                             f"{max(m[v]['max_loss_in'], m[v]['max_loss_out']):.1f}"
                             f"  <-- likely a scaling/outlier failure, try --clip -100 100")
        if notes:
            print(f"{name}:")
            for n in notes:
                print(f"    {n}")


def _check_reference(docs: dict) -> None:
    """`trained` reference agreement across sweeps.

    Every run trains its own model on `retain + that run's sampled forget`, so there is no
    single shared pre-unlearning checkpoint. What IS shared: two sweeps with the same
    data_subfolder and the same training config differ only in their post_unlearning
    section, so their `trained` rows are drawn from the same population of models and must
    agree to within across-run noise. That is what this reports.
    """
    groups = {}
    for name, doc in docs.items():
        if doc["model_source"] != "trained":
            continue
        groups.setdefault((doc["data_subfolder"], doc["model"]), {})[name] = doc
    if not groups:
        return
    print("\n--- `trained` reference agreement within a data split ---")
    for (subfolder, model), rows in groups.items():
        if len(rows) < 2:
            continue
        vals = {n: d["mean"]["forget_out"]["accuracy"] for n, d in rows.items()
                if "forget_out" in d["mean"]}
        spread = max(vals.values()) - min(vals.values())
        print(f"{subfolder} / {model}: forget_out spread over {len(vals)} sweeps = "
              f"{spread:.2f} points"
              + ("  <-- >1.0; these sweeps should share their trained models"
                 if spread > 1.0 else "  OK"))
        for n, v in sorted(vals.items(), key=lambda kv: kv[1]):
            print(f"    {v:6.2f}  {n}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sweeps", nargs="+", help="Sweep directories (logs_*)")
    p.add_argument("--source", default="unlearned",
                   help="Checkpoint to attack: unlearned (default; highest "
                        "ckpt/unlearn_epoch_N), trained (highest ckpt/checkpoint_N), or an "
                        "explicit unlearn_epoch_N / checkpoint_N")
    p.add_argument("--variants", nargs="+", default=None, choices=list(ALL_VARIANTS),
                   help="Restrict which 'out' populations to attack with. Variants that do "
                        "not apply to a sweep's branch are skipped for that sweep.")
    p.add_argument("--folds", type=int, default=5, help="CV splits (default 5)")
    p.add_argument("--seed", type=int, default=0,
                   help="Base seed; a run uses seed + its run_vars run_index")
    p.add_argument("--device", default="cuda:0",
                   help="Informational only -- JAX device selection is via "
                        "CUDA_VISIBLE_DEVICES. A GPU must be visible for Orbax restore.")
    p.add_argument("--eval-batch-size", type=int, default=1000)
    p.add_argument("--force", action="store_true",
                   help="Re-score every run instead of reusing its per-run cache file")
    p.add_argument("--no-retain-acc", dest="retain_acc", action="store_false",
                   help="Skip the 40500-image retain pass (retain accuracy reported as n/a)")
    p.add_argument("--shuffle-labels", action="store_true",
                   help="Sanity check: permute in/out labels; must come back at 50+-1. "
                        "Writes to mia_basic_<source>*_shuffled.json so it cannot be "
                        "confused with a real result.")
    p.add_argument("--summary-out", default=None,
                   help="Also write one JSON holding every sweep's mean/std block")
    p.add_argument("--max-runs", type=int, default=0,
                   help="Smoke test: score only the first N runs of each sweep. The result "
                        "is not a sweep result; pair with --no-write.")
    p.add_argument("--no-write", dest="write", action="store_false",
                   help="Smoke test: compute and print but write nothing to disk")
    # Attacker overrides. Each is None unless given, so a preset shows through.
    p.add_argument("--cv", default=None, choices=("kfold", "shuffle"))
    p.add_argument("--clip", nargs=2, type=float, metavar=("LO", "HI"))
    p.add_argument("--no-clip", action="store_true")
    p.add_argument("--scale", dest="scale", action="store_true", default=None)
    p.add_argument("--no-scale", dest="scale", action="store_false")
    p.add_argument("--max-iter", type=int, default=None)
    p.add_argument("--test-size", type=float, default=0.1,
                   help="StratifiedShuffleSplit test_size (sklearn default 0.1)")
    args = p.parse_args()

    print(f"jax devices: {jax.devices()}")

    spec = resolve_attacker(args)
    # `_shuffled` keeps a validation run (section: --shuffle-labels) from ever being
    # mistaken for a real result on disk.
    suffix = "_shuffled" if args.shuffle_labels else ""
    print(f"attacker: {spec}")

    docs = {}
    for sweep in (Path(s) for s in args.sweeps):
        print(f"### {sweep} @ {args.source}")
        doc = attack_sweep(sweep, args.source, specs={"attacker": spec},
                           suffixes={"attacker": suffix},
                           base_seed=args.seed, variants=args.variants,
                           eval_batch_size=args.eval_batch_size, force=args.force,
                           retain_acc=args.retain_acc,
                           shuffle_labels=args.shuffle_labels,
                           max_runs=args.max_runs, write=args.write)["attacker"]
        out_path = sweep / f"mia_basic_{args.source}{suffix}.json"
        if args.write and not args.max_runs:
            out_path.write_text(json.dumps(doc, indent=2))
        print(f"  -> {out_path}  [{doc['forget_split']}] mean over {doc['n_runs']} "
              f"runs: " + "  ".join(f"{v}={doc['mean'][v]['accuracy']:.2f}%"
                                    for v in doc["mean"]))
        docs[str(sweep)] = doc

    _print_summary(docs)
    _print_checks(docs)
    _check_reference(docs)

    if args.summary_out and args.write and not args.max_runs:
        out = Path(args.summary_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "generated_by": Path(__file__).name,
            "model_source": args.source, "attacker": spec, "seed": args.seed,
            "shuffle_labels": args.shuffle_labels,
            "sweeps": {n: {k: d[k] for k in
                           ("n_runs", "n_runs_total", "complete", "forget_split",
                            "forget_pool_classes", "data_subfolder", "num_classes",
                            "unlearning_algorithm",
                            "mean", "std", "accuracy")}
                       for n, d in docs.items()},
        }, indent=2))
        print(f"-> {out}")


if __name__ == "__main__":
    main()
