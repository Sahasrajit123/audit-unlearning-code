"""Unlearning methods: ``noop``, ``npo``, ``retain_ft``, ``grad_ascent``,
``grad_diff``, ``simnpo``.

All methods receive the trained model for a run, plus the forget set
``D_f(S) = {batches with S_j = +1}`` and the fixed retain set ``D_r``. The forget set
NEVER contains ``S_j = -1`` candidates -- those were never trained on, so handing
them to the unlearner would leak the hidden sign vector into the pipeline and
invalidate the audit. This is enforced upstream by
:func:`tofu_data.forget_examples_for_run` and checked by
``tests/test_manifest.py::test_forget_loader_only_positive_candidates``.

The initial fine-tune happens ONCE per sign-vector run; the saved checkpoint is then
branched into each configured method, so the expensive stage is not repeated.

Methods
-------
``noop``
    Return the trained model unchanged. A positive leakage control: the in/out losses
    should be clearly separated, since the "unlearned" model is simply a model that
    trained on the positive candidates. If ``noop`` does NOT separate, something is
    wrong with training exposure, loss masking, or aggregation -- diagnose before
    spending compute on NPO.

``npo``
    Negative Preference Optimization (Zhang et al., 2024), as used by OpenUnlearning::

        L_NPO = (2/beta) * E_{D_f} [ softplus( beta * (log pi_theta(a|q) - log pi_ref(a|q)) ) ]
              = -(2/beta) * E_{D_f} [ log sigmoid( beta * (log pi_ref - log pi_theta) ) ]

    ``pi_ref`` is the frozen trained model at the start of unlearning. Compared with
    plain gradient ascent, the sigmoid saturates, so per-example gradients stay
    bounded and the model degrades far more gracefully. With ``retain_weight > 0`` a
    retain cross-entropy term is added (OpenUnlearning's ``NPO_RT``), which is the
    configuration that preserves usable utility; set it to ``0`` for pure NPO.

``retain_ft``
    Fine-tune only on the fixed retain set ``D_r``. Equivalent to the paper's "pure
    fine-tuning on the retain set" heuristic, and to OpenUnlearning's ``finetune``.

``grad_ascent``
    Plain gradient ascent on the forget set, i.e. OpenUnlearning's ``GradAscent``::

        L = -forget_nll          (HF's mean-token cross-entropy, negated)

    No reference model and **no retain term** -- upstream's ``GradAscent`` inherits
    ``finetune`` and takes no ``alpha``/``gamma``, so adding one would not be faithful.
    This maps directly onto "ascent on the forget set" in the paper's §7.2, which is
    one of the four heuristic methods audited there.

    The objective is **unbounded below**, so this diverges if run long. That is the
    method, not a bug: it is precisely the failure mode NPO's saturating sigmoid was
    designed to avoid. Default ``epochs: 2`` matches the paper's ascent phase;
    OpenUnlearning uses 10. A divergence warning fires if the forget NLL explodes.

``grad_diff``
    Gradient difference, i.e. OpenUnlearning's ``GradDiff``::

        L = gamma * (-forget_nll) + alpha * retain_nll

    with their defaults ``gamma = 1.0, alpha = 1.0``. The bounded retain term keeps
    the ascent from running away, which makes this the closest LLM analogue to the
    paper's interleaved descent-ascent (IDA) heuristic.

``simnpo``
    Reference-free NPO (Fan et al., 2024), OpenUnlearning's ``SimNPO``::

        L = gamma * -(2/beta) * logsigmoid( beta * (nll_per_token - delta) )
            + alpha * retain_nll

    The DPO-style ratio against a frozen reference is replaced by the
    **length-normalized** NLL, so **no reference model is needed** -- ~2.3 GiB
    cheaper and faster than NPO. It keeps NPO's saturating sigmoid, so it degrades
    gracefully where ``grad_ascent`` runs away. Upstream defaults ``beta = 4.5,
    delta = 0.0, gamma = 0.125, alpha = 1.0`` differ sharply from NPO's, which is
    why ``beta`` and ``gamma`` have method-dependent fallbacks here.

Reduction note
--------------
Three different reductions, all faithful to upstream:

* ``npo``      -- **sum** of answer-token log-probs (the DPO ratio needs a sequence
  log-likelihood; upstream's ``compute_batch_nll`` sums).
* ``simnpo``   -- **mean** per answer token (upstream divides that sum by
  ``loss_mask.sum(-1)``); this length normalization is the "Sim" in SimNPO.
* ``grad_ascent`` / ``grad_diff`` -- HF's ``outputs.loss``, the **mean** over answer
  tokens.

The asymmetry is not an oversight: the families genuinely differ, and the reduction
changes the effective gradient scale.
"""

from __future__ import annotations

import copy
import math
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

from .tofu_data import IGNORE_INDEX, AnswerLossCollator, QAExample
from .train import encode_dataset, train_model

__all__ = [
    "UNLEARNING_METHODS",
    "divergence_threshold",
    "FORGET_SET_METHODS",
    "apply_unlearning",
    "sequence_logprob",
]

UNLEARNING_METHODS = (
    "noop", "npo", "retain_ft", "grad_ascent", "grad_diff", "simnpo",
)

#: Methods driven by the shared forget-set loop in :func:`_forget_retain_unlearn`.
FORGET_SET_METHODS = ("npo", "grad_ascent", "grad_diff", "simnpo")

#: Collapse threshold, as a FRACTION of ln(vocab_size).
#:
#: A model emitting a uniform distribution scores exactly ln(V) -- 11.76 for
#: Llama-3.2's 128,256-token vocabulary. So the loss cannot exceed ln(V) by much, and
#: an absolute threshold above it can never fire. (An earlier version used a fixed
#: 20.0, which was unreachable: measured `grad_ascent` collapsed completely at 10.84
#: and was never flagged.) Expressing it relatively makes the guard vocabulary-
#: independent and actually reachable.
DIVERGENCE_FRACTION_OF_UNIFORM = 0.6


def divergence_threshold(model: Any) -> float:
    """Forget NLL above which the model is treated as collapsed.

    ``DIVERGENCE_FRACTION_OF_UNIFORM * ln(vocab_size)``, falling back to a fixed
    value if the vocabulary size cannot be read.
    """
    try:
        v = int(model.config.vocab_size)
        if v > 1:
            return DIVERGENCE_FRACTION_OF_UNIFORM * math.log(v)
    except Exception:
        pass
    return 7.0


def sequence_logprob(model: Any, batch: Dict[str, Any], reduction: str = "sum") -> Any:
    """Per-example answer log-likelihood ``log pi(a | q)``.

    Computed in FP32 over answer tokens only (``labels != -100``). ``reduction="sum"``
    gives the sequence log-likelihood used by DPO/NPO-style objectives; ``"mean"``
    normalizes by answer length.
    """
    import torch

    out = model(
        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
    )
    logits = out.logits.float()
    shift_logits = logits[:, :-1, :]
    shift_labels = batch["labels"][:, 1:]

    logprobs = torch.log_softmax(shift_logits, dim=-1)
    mask = shift_labels != IGNORE_INDEX
    safe = shift_labels.masked_fill(~mask, 0)
    tok = logprobs.gather(-1, safe.unsqueeze(-1)).squeeze(-1) * mask.to(logprobs.dtype)

    summed = tok.sum(dim=1)
    if reduction == "sum":
        return summed
    if reduction == "mean":
        return summed / mask.sum(dim=1).clamp(min=1).to(summed.dtype)
    raise ValueError(f"reduction must be 'sum' or 'mean'; got {reduction!r}")


def _forget_retain_unlearn(
    method: str,
    model: Any,
    tokenizer: Any,
    forget_examples: Sequence[QAExample],
    retain_examples_: Sequence[QAExample],
    cfg: Dict[str, Any],
    *,
    seed: int,
    log: Callable[[str], None],
    metrics_sink: Optional[Any] = None,
) -> Dict[str, Any]:
    """Shared loop for the forget-set methods.

    ``npo``, ``grad_ascent``, ``grad_diff`` and ``simnpo``. All of them iterate over forget micro-batches and optionally draw retain
    micro-batches alongside; they differ only in the forget objective and in whether
    a frozen reference model is needed. Sharing the loop keeps the data ordering,
    accumulation, LR schedule, clipping and logging identical across methods, so a
    difference in the audit's epsilon reflects the objective and not the plumbing.
    """
    import torch
    import torch.nn.functional as F

    from .modeling import build_optimizer, peak_memory_stats, reset_peak_memory

    if method not in FORGET_SET_METHODS:
        raise ValueError(f"{method!r} is not a forget-set method")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    epochs = int(cfg.get("epochs", 5))
    micro_bs = int(cfg.get("micro_batch_size", 2))
    accum = int(cfg.get("gradient_accumulation_steps", 8))
    base_lr = float(cfg.get("learning_rate", 1e-5))
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))
    max_len = int(cfg.get("max_seq_length", 512))
    append_eos = bool(cfg.get("append_eos", True))
    system_prompt = cfg.get("system_prompt")

    # --- per-method objective parameters -----------------------------------------
    # Upstream's beta default differs per method: NPO.yaml uses 0.1, SimNPO.yaml 4.5.
    beta = float(cfg.get("beta", 4.5 if method == "simnpo" else 0.1))
    delta = float(cfg.get("delta", 0.0))
    reduction = cfg.get("sequence_reduction", "sum")
    # Upstream's `gamma` weights the forget term. SimNPO.yaml sets 0.125; the others 1.0.
    forget_weight = float(cfg.get("gamma", 0.125 if method == "simnpo" else 1.0))

    if method == "npo":
        needs_ref = True
        # `retain_weight` is this project's name for upstream NPO's `alpha`.
        retain_weight = float(cfg.get("retain_weight", cfg.get("alpha", 1.0)))
    elif method in ("grad_diff", "simnpo"):
        needs_ref = False
        retain_weight = float(cfg.get("alpha", cfg.get("retain_weight", 1.0)))
    else:  # grad_ascent
        needs_ref = False
        # Upstream's GradAscent has no retain term at all. Refuse to invent one.
        retain_weight = 0.0
        if cfg.get("alpha") or cfg.get("retain_weight"):
            log(
                "[grad_ascent] WARNING: alpha/retain_weight are ignored -- upstream's "
                "GradAscent is forget-only. Use grad_diff for a retain term."
            )

    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    collate = AnswerLossCollator(pad_token_id=pad_id)

    forget_enc = encode_dataset(
        tokenizer, forget_examples, max_length=max_len,
        append_eos=append_eos, system_prompt=system_prompt,
    )
    retain_enc = encode_dataset(
        tokenizer, retain_examples_, max_length=max_len,
        append_eos=append_eos, system_prompt=system_prompt,
    ) if retain_weight > 0 else []

    if not forget_enc:
        raise ValueError(f"{method}: empty forget set")

    ref_model = None
    if needs_ref:
        # Frozen reference model: the trained model at the start of unlearning.
        log(f"[{method}] cloning reference model (frozen)")
        ref_model = copy.deepcopy(model)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)

    optimizer, opt_info = build_optimizer(model, cfg)

    reset_peak_memory()
    model.train()
    t0 = time.time()

    micro_batches = [
        forget_enc[i : i + micro_bs] for i in range(0, len(forget_enc), micro_bs)
    ]
    total_steps = math.ceil(len(micro_batches) / accum) * epochs

    if method == "npo":
        variant = "npo_rt" if retain_weight > 0 else "npo"
        detail = f"beta={beta} retain_weight={retain_weight} reduction={reduction}"
    elif method == "grad_diff":
        variant = "grad_diff"
        detail = f"gamma={forget_weight} alpha={retain_weight}"
    elif method == "simnpo":
        variant = "simnpo"
        detail = (f"beta={beta} delta={delta} gamma={forget_weight} "
                  f"alpha={retain_weight} (reference-free)")
    else:
        variant = "grad_ascent"
        detail = "forget-only (no reference model, no retain term)"

    log(
        f"[{method}] forget_n={len(forget_enc)} retain_n={len(retain_enc)} "
        f"epochs={epochs} steps={total_steps} variant={variant} {detail}"
    )

    retain_cursor = 0
    global_step = 0
    history: List[Dict[str, float]] = []
    diverged = False
    div_thresh = divergence_threshold(model)

    for epoch in range(epochs):
        run_forget, run_ret, run_nll, n_micro = 0.0, 0.0, 0.0, 0
        optimizer.zero_grad(set_to_none=True)

        for mi, chunk in enumerate(micro_batches):
            fb = {k: v.to(device) for k, v in collate(chunk).items()}

            if method == "npo":
                cur_lp = sequence_logprob(model, fb, reduction)
                with torch.no_grad():
                    ref_lp = sequence_logprob(ref_model, fb, reduction)
                # L = (2/beta) * softplus(beta*(cur - ref)); minimized by cur << ref.
                forget_term = forget_weight * (2.0 / beta) * F.softplus(
                    beta * (cur_lp - ref_lp)
                ).mean()
                # Track the plain NLL too, so divergence is comparable across methods.
                nll_val = float(-cur_lp.mean().detach().item())
            elif method == "simnpo":
                # Reference-free NPO: the DPO ratio is replaced by the LENGTH-NORMALIZED
                # NLL, so no frozen copy of the model is needed.
                #   L = -(2/beta) * logsigmoid(beta * (nll_per_token - delta))
                # Minimized by driving the forget NLL UP, but saturating like NPO.
                lp_mean = sequence_logprob(model, fb, "mean")
                nll_mean = -lp_mean
                forget_term = forget_weight * (
                    -(2.0 / beta) * F.logsigmoid(beta * (nll_mean - delta)).mean()
                )
                nll_val = float(nll_mean.mean().detach().item())
            else:
                # HF's outputs.loss is the MEAN cross-entropy over answer tokens,
                # which is what upstream GradAscent/GradDiff negate.
                forget_nll = model(**fb).loss
                forget_term = forget_weight * (-forget_nll)
                nll_val = float(forget_nll.detach().item())

            loss = forget_term
            ret_val = 0.0

            if retain_weight > 0 and retain_enc:
                # Walk the retain set cyclically so the retain gradient is not
                # dominated by whichever examples happen to sit at the front.
                rchunk = []
                for _ in range(micro_bs):
                    rchunk.append(retain_enc[retain_cursor % len(retain_enc)])
                    retain_cursor += 1
                rb = {k: v.to(device) for k, v in collate(rchunk).items()}
                retain_loss = model(**rb).loss
                loss = loss + retain_weight * retain_loss
                ret_val = float(retain_loss.detach().item())

            (loss / accum).backward()
            run_forget += float(forget_term.detach().item())
            run_ret += ret_val
            run_nll += nll_val
            n_micro += 1

            if (mi + 1) % accum == 0 or mi == len(micro_batches) - 1:
                progress = global_step / max(1, total_steps)
                for g in optimizer.param_groups:
                    g["lr"] = base_lr * (1.0 - progress)
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], max_grad_norm
                    )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

        mean_nll = run_nll / max(1, n_micro)
        rec = {
            "epoch": epoch + 1,
            "forget_loss": run_forget / max(1, n_micro),
            "forget_nll": mean_nll,
            "retain_loss": run_ret / max(1, n_micro),
        }
        if method == "npo":
            # Preserve the original field name so existing results stay comparable.
            rec["npo_loss"] = rec["forget_loss"]
        history.append(rec)

        log(
            f"[{method}] epoch {epoch + 1}/{epochs} "
            f"forget_loss={rec['forget_loss']:.4f} forget_nll={mean_nll:.4f} "
            f"retain_loss={rec['retain_loss']:.4f}"
        )
        if metrics_sink is not None:
            metrics_sink.log({
                f"{method}/forget_loss": rec["forget_loss"],
                f"{method}/forget_nll": mean_nll,
                f"{method}/retain_loss": rec["retain_loss"],
                f"{method}/epoch": epoch + 1,
                f"{method}/divergence_threshold": div_thresh,
            }, step=global_step)

        if mean_nll > div_thresh and not diverged:
            diverged = True
            log(
                f"[{method}] WARNING: forget NLL {mean_nll:.2f} exceeds "
                f"{div_thresh:.2f} ({DIVERGENCE_FRACTION_OF_UNIFORM:.0%} of the "
                f"uniform-output loss) -- the model is collapsing toward noise. For "
                "an unbounded objective (grad_ascent) this is expected behaviour, but "
                "a small epsilon_LB from a collapsed model reflects damage, not "
                "unlearning. Report the utility metrics alongside it."
            )

    if ref_model is not None:
        del ref_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    mem = peak_memory_stats()
    out = {
        "label": method,
        "variant": variant,
        "epochs": epochs,
        "optimizer_steps": global_step,
        "num_forget_examples": len(forget_enc),
        "num_retain_examples": len(retain_enc),
        "retain_weight": retain_weight,
        "forget_weight": forget_weight,
        "used_reference_model": needs_ref,
        "diverged": diverged,
        "divergence_threshold": div_thresh,
        "final_forget_nll": history[-1]["forget_nll"] if history else None,
        "history": history,
        "duration_seconds": time.time() - t0,
        "seed": int(seed),
        **mem,
        **opt_info,
    }
    if method == "npo":
        out["beta"] = beta
        out["sequence_reduction"] = reduction
    elif method == "simnpo":
        out["beta"] = beta
        out["delta"] = delta
        out["sequence_reduction"] = "mean"   # length-normalized, by definition
    return out


def apply_unlearning(
    method: str,
    model: Any,
    tokenizer: Any,
    forget_examples: Sequence[QAExample],
    retain_examples_: Sequence[QAExample],
    cfg: Dict[str, Any],
    *,
    seed: int,
    logger: Optional[Callable[[str], None]] = None,
    metrics_sink: Optional[Any] = None,
) -> Dict[str, Any]:
    """Apply ``method`` to ``model`` in place. Returns the method's metrics.

    ``cfg`` is the method-specific config block; the effective values are returned so
    they can be saved with the result.
    """
    log = logger or (lambda msg: None)

    if method not in UNLEARNING_METHODS:
        raise ValueError(
            f"unknown method {method!r}; expected one of {UNLEARNING_METHODS}"
        )

    if method == "noop":
        log("[noop] returning the trained model unchanged (leakage control)")
        return {
            "label": "noop",
            "duration_seconds": 0.0,
            "num_forget_examples": len(forget_examples),
            "note": (
                "Positive control. in/out losses should be visibly separated; if not, "
                "diagnose training exposure, loss masking, or aggregation."
            ),
        }

    if method == "retain_ft":
        log(f"[retain_ft] fine-tuning on {len(retain_examples_)} retain examples")
        res = train_model(
            model, tokenizer, retain_examples_, cfg,
            seed=seed, logger=log, label="retain_ft", metrics_sink=metrics_sink,
        )
        res["num_forget_examples"] = len(forget_examples)
        return dict(res)

    return _forget_retain_unlearn(
        method, model, tokenizer, forget_examples, retain_examples_, cfg,
        seed=seed, log=log, metrics_sink=metrics_sink,
    )
