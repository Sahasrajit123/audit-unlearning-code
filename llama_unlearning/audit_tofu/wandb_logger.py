"""Optional Weights & Biases logging for audit runs.

Design rule: **this must never be able to break a run.** A 30-run, 20-GPU-hour audit
should not die because a metrics service is unreachable, unauthenticated, or
rate-limiting. Every method here swallows its own exceptions and degrades to a no-op,
logging the reason once. The audit's real outputs are always the JSON files on disk;
wandb is a convenience view on top.

One wandb run per ``(audit run_id, method)``, because that is the unit that produces a
model and a set of losses. They share a ``group`` (the audit's ``experiment.name`` plus
the split hash) so the whole audit collapses into one grouped view, and carry
``job_type`` = the method so methods can be compared directly.

Auth note: ``~/.netrc`` may be unreadable (e.g. on a network-mounted home
directory). If it cannot be read, wandb cannot authenticate and would otherwise hang
or fail mid-run. :func:`preflight` checks this up front so the failure is visible
before any GPU time is spent, and ``mode: offline`` is always available as a fallback
that writes to ``WANDB_DIR`` for later ``wandb sync``.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

__all__ = ["WandbRun", "preflight", "NullRun"]


def preflight(cfg: Dict[str, Any]) -> tuple:
    """Check whether wandb can be used. Returns ``(ok, reason)``.

    Called once by the driver before launching, so an auth problem surfaces
    immediately rather than 20 minutes into the first fine-tune.
    """
    wcfg = cfg.get("wandb") or {}
    if not wcfg.get("enabled"):
        return False, "disabled in config (wandb.enabled=false)"

    try:
        import wandb  # noqa: F401
    except ImportError:
        return False, "wandb not installed (pip install wandb)"

    mode = str(wcfg.get("mode", "online"))
    if mode == "offline":
        return True, "offline mode; sync later with `wandb sync`"

    if os.environ.get("WANDB_API_KEY"):
        return True, "authenticated via WANDB_API_KEY"

    # netrc is the usual path.
    netrc = os.path.expanduser("~/.netrc")
    try:
        with open(netrc) as f:
            if "api.wandb.ai" in f.read():
                return True, f"authenticated via {netrc}"
        return False, f"{netrc} has no api.wandb.ai entry"
    except OSError as exc:
        return False, (
            f"cannot read {netrc} ({exc.strerror}). Set WANDB_API_KEY, or use wandb.mode=offline."
        )


class NullRun:
    """No-op stand-in with the same surface, used whenever wandb is unavailable."""

    url = None
    enabled = False

    def log(self, *_a, **_k) -> None:
        pass

    def summary(self, *_a, **_k) -> None:
        pass

    def finish(self, *_a, **_k) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_exc) -> bool:
        return False


class WandbRun:
    """A single wandb run for one ``(run_id, method)`` pair.

    Use :meth:`create`, which returns a :class:`NullRun` rather than raising if
    anything at all goes wrong.
    """

    def __init__(self, run: Any):
        self._run = run
        self.enabled = True
        self.url = getattr(run, "url", None)

    @classmethod
    def create(
        cls,
        cfg: Dict[str, Any],
        run_id: str,
        method: str,
        *,
        split_hash: str,
        family: str,
        gpu: Optional[int] = None,
        extra_config: Optional[Dict[str, Any]] = None,
        logger: Optional[Any] = None,
        group: Optional[str] = None,
    ):
        wcfg = cfg.get("wandb") or {}
        log = (logger.info if logger else print)
        if not wcfg.get("enabled"):
            return NullRun()

        try:
            import wandb

            tags = list(wcfg.get("tags") or [])
            tags += [f"method:{method}", f"family:{family}",
                     f"m:{cfg['split']['m']}" if "m" in cfg.get("split", {}) else "",
                     f"batching:{cfg['split']['batching']}"]
            tags = [t for t in tags if t]

            run = wandb.init(
                project=wcfg.get("project", "tofu-unlearning-audit"),
                entity=wcfg.get("entity") or None,
                name=f"{run_id}/{method}",
                # One group per audit, so 30 runs x N methods read as one experiment.
                # An explicit `group` overrides that, for artifacts that are NOT part
                # of one audit -- the retain-only reference model is shared by every
                # config with the same D_r, so filing it under one audit's
                # name-and-hash would misrepresent what it belongs to.
                group=(group or wcfg.get("group")
                       or f"{cfg['experiment']['name']}-{split_hash[:8]}"),
                job_type=method,
                tags=tags,
                mode=str(wcfg.get("mode", "online")),
                dir=wcfg.get("dir") or os.environ.get("WANDB_DIR"),
                reinit=True,
                config={
                    "run_id": run_id,
                    "method": method,
                    "family": family,
                    "gpu": gpu,
                    "split_hash": split_hash,
                    "model": cfg["model"]["id"],
                    "m": cfg["split"].get("m"),
                    "batching": cfg["split"]["batching"],
                    "training": cfg["training"],
                    "unlearning": (cfg.get("unlearning") or {}).get(method, {}),
                    "epsilon": cfg["epsilon"],
                    "attack": cfg["attack"],
                    **(extra_config or {}),
                },
            )
            out = cls(run)
            log(f"[wandb] {run_id}/{method} -> {out.url or '(offline)'}")
            return out
        except Exception as exc:
            log(f"[wandb] disabled for {run_id}/{method}: {type(exc).__name__}: {exc}")
            return NullRun()

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        try:
            self._run.log(metrics, step=step)
        except Exception:
            # Never let a metrics push interrupt training.
            pass

    def summary(self, values: Dict[str, Any]) -> None:
        try:
            for k, v in values.items():
                self._run.summary[k] = v
        except Exception:
            pass

    def finish(self, exit_code: int = 0) -> None:
        try:
            self._run.finish(exit_code=exit_code)
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *_rest) -> bool:
        self.finish(exit_code=1 if exc_type else 0)
        return False
