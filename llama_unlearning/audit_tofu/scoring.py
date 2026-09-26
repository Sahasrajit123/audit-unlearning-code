"""Extraction of the audit score: teacher-forced, answer-only mean loss.

For a QA pair ``z = (q, a)``::

    l_z(f) = -(1/|a|) * sum_{t=1..|a|} log p_f(a_t | q, a_<t)

Requirements this module enforces (all from the spec):

* the SAME chat template as training (delegated to :func:`tofu_data.encode_example`,
  and ``append_eos`` is read back from the run config rather than re-guessed);
* every question/prompt token masked;
* the mean taken over non-padding ANSWER tokens only;
* losses accumulated and stored in FP32 even when the forward pass runs in BF16;
* ``model.eval()`` and ``torch.no_grad()``, so dropout is off;
* one record per ``(run_id, method, candidate_author, qa_id)``;
* generated-answer ROUGE is deliberately NOT used as the audit score.

The per-example mean is computed from per-example token sums rather than from a
batch-level mean, because a batch-level mean would weight examples by their answer
length and produce a different statistic than the one defined above.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .tofu_data import IGNORE_INDEX, AnswerLossCollator, QAExample, encode_example

__all__ = ["score_examples", "losses_to_records", "records_to_score_dict"]


def score_examples(
    model: Any,
    tokenizer: Any,
    examples: Sequence[QAExample],
    *,
    max_length: int = 512,
    append_eos: bool = True,
    system_prompt: Optional[str] = None,
    batch_size: int = 8,
    device: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Compute the answer loss for every example.

    Returns one dict per example with ``author_id``, ``qa_id``, ``loss`` (FP32 python
    float) and ``num_answer_tokens``.
    """
    import torch

    was_training = model.training
    model.eval()

    if device is None:
        device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    collate = AnswerLossCollator(pad_token_id=pad_id)

    encoded = []
    for e in examples:
        enc = encode_example(
            tokenizer,
            e.question,
            e.answer,
            max_length=max_length,
            append_eos=append_eos,
            system_prompt=system_prompt,
        )
        enc["_example"] = e
        encoded.append(enc)

    results: List[Dict[str, Any]] = []
    try:
        with torch.no_grad():
            for start in range(0, len(encoded), batch_size):
                chunk = encoded[start : start + batch_size]
                batch = collate(chunk)
                batch = {k: v.to(device) for k, v in batch.items()}

                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                # Cast to FP32 before the log-softmax: BF16 has ~3 decimal digits of
                # mantissa, which is coarse relative to the in/out gaps the attack
                # has to resolve.
                logits = out.logits.float()
                labels = batch["labels"]

                # Standard causal shift: position t predicts token t+1.
                shift_logits = logits[:, :-1, :]
                shift_labels = labels[:, 1:]

                logprobs = torch.log_softmax(shift_logits, dim=-1)
                mask = shift_labels != IGNORE_INDEX
                safe_labels = shift_labels.masked_fill(~mask, 0)
                tok_logprobs = logprobs.gather(
                    -1, safe_labels.unsqueeze(-1)
                ).squeeze(-1)
                tok_logprobs = tok_logprobs * mask.to(tok_logprobs.dtype)

                # Per-example sum / count -> per-example mean, in FP32.
                sums = tok_logprobs.sum(dim=1)
                counts = mask.sum(dim=1)

                for i, enc in enumerate(chunk):
                    n = int(counts[i].item())
                    if n == 0:
                        raise RuntimeError(
                            f"example {enc['_example'].key} had no answer tokens"
                        )
                    loss = float(-(sums[i].item() / n))
                    e: QAExample = enc["_example"]
                    results.append(
                        {
                            "author_id": e.author_id,
                            "qa_id": int(e.qa_id),
                            "loss": loss,
                            "num_answer_tokens": n,
                        }
                    )
    finally:
        if was_training:
            model.train()

    return results


def losses_to_records(
    losses: Sequence[Dict[str, Any]],
    run_id: str,
    method: str,
    manifest: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Attach run/method identity and the owning candidate batch to each loss.

    One record per ``(run_id, method, candidate_author, qa_id)``, per the spec.
    Ground-truth labels are intentionally NOT attached here; they are joined later,
    only for calibration fitting or for post-hoc scoring of a finalized prediction.
    """
    owner: Dict[Any, str] = {}
    for bid, members in manifest["split"]["batches"].items():
        for author, qa in members:
            owner[(str(author), int(qa))] = bid

    records = []
    for row in losses:
        key = (row["author_id"], int(row["qa_id"]))
        records.append(
            {
                "run_id": run_id,
                "method": method,
                "candidate_author": row["author_id"],
                "qa_id": int(row["qa_id"]),
                "batch_id": owner.get(key),
                "loss": float(row["loss"]),
                "num_answer_tokens": int(row["num_answer_tokens"]),
            }
        )
    return records


def records_to_score_dict(records: Sequence[Dict[str, Any]]) -> Dict[Any, float]:
    """``[record] -> {(batch_id, qa_id): loss}``, the form the attack consumes."""
    out: Dict[Any, float] = {}
    for r in records:
        if r.get("batch_id") is None:
            continue
        out[(r["batch_id"], int(r["qa_id"]))] = float(r["loss"])
    return out
