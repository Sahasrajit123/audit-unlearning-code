"""Run directories, resume, checkpoint retention, and resource accounting.

Layout, following the upstream repo's conventions (``run_<id>/`` holding
``config.json``, ``metrics.json``, ``run.log``)::

    <output_root>/
      manifest.json
      runs/
        calib_000/
          run_state.json        # sign vector + seeds, written once, never rewritten
          config.effective.yaml # merged config, as YAML; re-runnable with --config
          config.json           # same, as JSON
          config_sources/       # verbatim copies of the contributing YAML files
          invocation.json       # argv, overrides, versions, git, env
          run.log
          trained/              # temporary fine-tuned checkpoint (retention-managed)
          methods/
            noop/      metrics.json  losses.json
            npo/       metrics.json  losses.json
            retain_ft/ metrics.json  losses.json
          metrics.json          # run-level roll-up incl. resources

Deletion safety
---------------
:func:`prune_checkpoints` refuses to delete anything that is not strictly inside the
run's own resolved output directory. Paths are resolved (symlinks included) and
checked with ``Path.is_relative_to`` before any removal, so a misconfigured
``output_root`` cannot cause deletion elsewhere on the filesystem.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = [
    "RunPaths",
    "RunState",
    "resolve_run_paths",
    "load_or_create_run_state",
    "setup_run_logger",
    "save_json",
    "load_json",
    "save_config_provenance",
    "prune_checkpoints",
    "ResourceTracker",
    "run_ids_for_family",
]

RETENTION_POLICIES = ("keep_all", "keep_trained", "delete_after_scoring")


@dataclass(frozen=True)
class RunPaths:
    root: Path
    run_dir: Path
    trained_dir: Path
    methods_dir: Path

    def method_dir(self, method: str) -> Path:
        return self.methods_dir / method


def resolve_run_paths(output_root: str | Path, run_id: str) -> RunPaths:
    root = Path(output_root).expanduser().resolve()
    run_dir = (root / "runs" / run_id).resolve()
    if not run_dir.is_relative_to(root):
        raise ValueError(f"run_id {run_id!r} escapes output_root {root}")
    return RunPaths(
        root=root,
        run_dir=run_dir,
        trained_dir=run_dir / "trained",
        methods_dir=run_dir / "methods",
    )


@dataclass
class RunState:
    """Immutable per-run facts: the sign vector and the seeds.

    Written once. On resume it is read back rather than recomputed, so an interrupted
    experiment can never continue with a different ``S`` or different seeds -- which
    would silently corrupt the calibration/evaluation families.
    """

    run_id: str
    family: str
    run_index: int
    sign_vector: List[int]
    train_seed: int
    unlearn_seed: int
    split_hash: str
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "family": self.family,
            "run_index": self.run_index,
            "sign_vector": list(self.sign_vector),
            "train_seed": int(self.train_seed),
            "unlearn_seed": int(self.unlearn_seed),
            "split_hash": self.split_hash,
            "created_at": self.created_at,
        }


def run_ids_for_family(manifest: Dict[str, Any], family: str) -> List[str]:
    if family not in ("calibration", "evaluation"):
        raise ValueError(f"family must be calibration/evaluation; got {family!r}")
    return list(manifest["sign_vectors"][family]["run_ids"])


def _family_of(manifest: Dict[str, Any], run_id: str) -> tuple[str, int]:
    for family in ("calibration", "evaluation"):
        ids = manifest["sign_vectors"][family]["run_ids"]
        if run_id in ids:
            return family, ids.index(run_id)
    raise KeyError(f"unknown run_id {run_id!r}")


def load_or_create_run_state(
    manifest: Dict[str, Any], run_id: str, paths: RunPaths
) -> RunState:
    """Return the run's state, creating it on first use and reusing it thereafter."""
    state_path = paths.run_dir / "run_state.json"

    family, index = _family_of(manifest, run_id)
    seeds = manifest["seeds"]
    expected = RunState(
        run_id=run_id,
        family=family,
        run_index=index,
        sign_vector=list(manifest["sign_vectors"][family]["vectors"][run_id]),
        train_seed=int(seeds["train_seed_base"]) + index,
        unlearn_seed=int(seeds["unlearn_seed_base"]) + index,
        split_hash=manifest["split_hash"],
    )

    if state_path.exists():
        stored = json.loads(state_path.read_text())
        if stored["split_hash"] != manifest["split_hash"]:
            raise RuntimeError(
                f"{run_id}: existing run was created against split_hash "
                f"{stored['split_hash'][:12]}... but the manifest is now "
                f"{manifest['split_hash'][:12]}.... Refusing to resume: the audit "
                "split changed. Use a fresh output_root."
            )
        for key in ("sign_vector", "train_seed", "unlearn_seed"):
            if stored[key] != expected.as_dict()[key]:
                raise RuntimeError(
                    f"{run_id}: stored {key} disagrees with the manifest. "
                    "Refusing to resume with altered run parameters."
                )
        return RunState(**{k: stored[k] for k in stored if k in RunState.__annotations__})

    paths.run_dir.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(expected.as_dict(), indent=2, sort_keys=True))
    return expected


def setup_run_logger(run_dir: Path, run_id: str) -> logging.Logger:
    """File + stream logger, matching the upstream ``run.log`` convention."""
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"audit_tofu.run.{run_id}")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh = logging.FileHandler(str(run_dir / "run.log"), mode="a")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def save_json(path: str | Path, payload: Any) -> Path:
    """Atomic write, so an interrupted run never leaves a truncated result file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)
    os.replace(tmp, path)
    return path


def load_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


def save_config_provenance(
    run_dir: Path,
    cfg: Dict[str, Any],
    config_path: Optional[str] = None,
    overrides: Optional[list] = None,
    argv: Optional[list] = None,
) -> Dict[str, Any]:
    """Write everything needed to reconstruct how this run was configured.

    Files written under ``run_dir``:

    ``config.effective.yaml``   the fully-merged config as YAML -- readable, diffable,
                                and directly re-runnable with ``--config``
    ``config.json``             the same content as JSON (kept for programmatic use)
    ``config_sources/``         verbatim copies of every contributing YAML file
    ``invocation.json``         argv, cwd, overrides, config_hash, versions, host

    Why copy the sources rather than trust the merged result? The merged config records
    the values that were used, but not *where they came from* -- and the files on disk
    will have moved on by the time anyone debugs this. Keeping both means a stale
    result can be compared against today's config to see exactly what changed.
    """
    import platform
    import socket
    import subprocess
    import sys

    import yaml

    from .config import config_hash, config_source_files

    run_dir.mkdir(parents=True, exist_ok=True)

    # 1. effective config, in both formats
    (run_dir / "config.effective.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False, width=100)
    )
    save_json(run_dir / "config.json", cfg)

    # 2. verbatim sources
    sources = []
    if config_path:
        src_dir = run_dir / "config_sources"
        src_dir.mkdir(exist_ok=True)
        try:
            for i, f in enumerate(config_source_files(config_path)):
                # Prefix with load order so the merge sequence is self-documenting.
                dest = src_dir / f"{i:02d}_{f.name}"
                shutil.copyfile(f, dest)
                sources.append({"order": i, "path": str(f), "saved_as": dest.name})
        except Exception as exc:
            (src_dir / "ERROR.txt").write_text(f"could not snapshot sources: {exc}\n")

    # 3. invocation
    def _git(*args):
        try:
            return subprocess.run(
                ["git", *args], capture_output=True, text=True, timeout=5,
                cwd=Path(__file__).resolve().parent.parent,
            ).stdout.strip() or None
        except Exception:
            return None

    versions = {"python": sys.version.split()[0]}
    for mod in ("torch", "transformers", "datasets", "numpy", "wandb"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:
            versions[mod] = None

    inv = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config_path": str(config_path) if config_path else None,
        "overrides": list(overrides or []),
        "config_hash": config_hash(cfg),
        "config_sources": sources,
        "argv": list(argv) if argv is not None else None,
        "cwd": os.getcwd(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "versions": versions,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "env": {
            k: os.environ.get(k)
            for k in ("HF_HOME", "CUDA_VISIBLE_DEVICES", "WANDB_DIR", "WANDB_MODE")
        },
    }
    save_json(run_dir / "invocation.json", inv)
    return inv


def prune_checkpoints(
    paths: RunPaths, policy: str, logger: Optional[logging.Logger] = None
) -> List[str]:
    """Apply the checkpoint retention policy. Returns the paths removed.

    Policies:

    ``keep_all``              keep everything (large: ~2.5 GiB per checkpoint)
    ``keep_trained``          keep the fine-tuned checkpoint, drop unlearned branches
    ``delete_after_scoring``  drop all model weights, keep JSON results (default)

    Only paths strictly inside ``paths.run_dir`` are ever removed.
    """
    if policy not in RETENTION_POLICIES:
        raise ValueError(f"policy must be one of {RETENTION_POLICIES}; got {policy!r}")

    log = logger.info if logger else (lambda msg: None)
    if policy == "keep_all":
        return []

    targets: List[Path] = []
    if policy in ("keep_trained", "delete_after_scoring"):
        if paths.methods_dir.exists():
            for method_dir in sorted(paths.methods_dir.iterdir()):
                if method_dir.is_dir():
                    for sub in ("model", "checkpoint"):
                        p = method_dir / sub
                        if p.exists():
                            targets.append(p)
    if policy == "delete_after_scoring" and paths.trained_dir.exists():
        targets.append(paths.trained_dir)

    removed: List[str] = []
    guard = paths.run_dir.resolve()
    for t in targets:
        rt = t.resolve()
        # Refuse anything that is not strictly below this run's own directory.
        if not rt.is_relative_to(guard) or rt == guard:
            log(f"[retention] REFUSING to delete outside run dir: {rt}")
            continue
        shutil.rmtree(rt, ignore_errors=True)
        removed.append(str(rt))
        log(f"[retention] removed {rt}")
    return removed


class ResourceTracker:
    """Wall-clock and peak-CUDA-memory accounting, per named stage.

    Records *measured* ``max_memory_allocated`` / ``max_memory_reserved`` rather than
    theoretical estimates, per the spec.
    """

    def __init__(self) -> None:
        self.stages: Dict[str, Dict[str, float]] = {}
        self._active: Optional[str] = None
        self._t0: float = 0.0

    def start(self, stage: str) -> None:
        from .modeling import reset_peak_memory

        reset_peak_memory()
        self._active = stage
        self._t0 = time.time()

    def stop(self) -> Dict[str, float]:
        from .modeling import peak_memory_stats

        if self._active is None:
            raise RuntimeError("ResourceTracker.stop() without start()")
        rec = {"duration_seconds": time.time() - self._t0, **peak_memory_stats()}
        self.stages[self._active] = rec
        self._active = None
        return rec

    def as_dict(self) -> Dict[str, Any]:
        total = sum(v["duration_seconds"] for v in self.stages.values())
        peak_alloc = max(
            (v.get("peak_allocated_gib", 0.0) for v in self.stages.values()), default=0.0
        )
        peak_res = max(
            (v.get("peak_reserved_gib", 0.0) for v in self.stages.values()), default=0.0
        )
        return {
            "stages": self.stages,
            "total_duration_seconds": total,
            "overall_peak_allocated_gib": peak_alloc,
            "overall_peak_reserved_gib": peak_res,
        }
