"""TOFU loading, candidate batching, chat formatting, and answer-only masking.

Two things in here carry most of the audit's correctness risk.

**1. Prompt masking.** The audit score is the teacher-forced *answer* loss

    l_z(f) = -(1/|a|) * sum_t log p_f(a_t | q, a_<t)

If question tokens leaked into the loss, every candidate's score would be dominated
by prompt perplexity -- which barely depends on whether the author was trained on --
and the attack would lose most of its signal. We therefore never tokenize the full
string and try to locate the boundary afterwards. Instead we tokenize the
chat-templated prompt and the answer *separately* and concatenate, so the boundary
is exact by construction and ``labels`` is ``[-100] * len(prompt_ids) + answer_ids``.

**2. Global example ordering.** The spec requires one fixed ordering, restricted per
run. Sampling an order per run would put positive and negative candidates at
systematically different epoch positions; the attacker would then partly be reading
off training-order effects instead of unlearning failure, inflating epsilon
spuriously. :func:`global_order` builds the permutation once from
``data_order_seed``; :func:`training_examples_for_run` filters it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .manifest import TOFU_QA_PER_AUTHOR

__all__ = [
    "QAExample",
    "load_tofu_examples",
    "load_synthetic_examples",
    "load_examples",
    "dataset_fingerprint",
    "global_order",
    "training_examples_for_run",
    "candidate_examples",
    "forget_examples_for_run",
    "retain_examples",
    "encode_example",
    "AnswerLossCollator",
]

IGNORE_INDEX = -100


@dataclass(frozen=True)
class QAExample:
    """One TOFU question-answer pair with its stable identity."""

    author_id: str
    qa_id: int
    question: str
    answer: str
    row_index: int

    @property
    def key(self) -> Tuple[str, int]:
        return (self.author_id, self.qa_id)


def load_tofu_examples(
    dataset_name: str = "locuslab/TOFU",
    config: str = "full",
    revision: Optional[str] = None,
    qa_per_author: int = TOFU_QA_PER_AUTHOR,
    cache_dir: Optional[str] = None,
) -> List[QAExample]:
    """Load TOFU and attach author identity.

    TOFU's ``full`` config holds 200 authors x 20 QA pairs = 4000 rows, ordered by
    author, so ``author index == row_index // 20``. This is the same convention
    OpenUnlearning relies on for its ``forget10`` / ``retain90`` splits, and it is
    checked below.
    """
    from datasets import load_dataset  # lazy: keeps CPU-only tests importable

    ds = load_dataset(dataset_name, config, revision=revision, cache_dir=cache_dir)
    split = "train" if "train" in ds else list(ds.keys())[0]
    rows = ds[split]

    if len(rows) % qa_per_author != 0:
        raise ValueError(
            f"{dataset_name}/{config} has {len(rows)} rows, not a multiple of "
            f"qa_per_author={qa_per_author}; author grouping would be wrong"
        )

    examples: List[QAExample] = []
    for i, row in enumerate(rows):
        examples.append(
            QAExample(
                author_id=f"author_{i // qa_per_author:04d}",
                qa_id=i % qa_per_author,
                question=row["question"],
                answer=row["answer"],
                row_index=i,
            )
        )
    return examples


SYNTHETIC_DATASET_NAME = "__synthetic__"


def load_synthetic_examples(
    num_authors: int, qa_per_author: int = 4, seed: int = 0
) -> List[QAExample]:
    """Deterministic synthetic QA pairs for the offline smoke test.

    Each author gets a distinctive, memorizable answer vocabulary so that a tiny
    model trained for a couple of epochs can actually separate in from out. This is
    a plumbing test, not an audit.
    """
    rng = np.random.default_rng(seed)
    vocab = [
        "azure", "quartz", "lantern", "meridian", "cobalt", "thistle", "vellum",
        "gossamer", "onyx", "zephyr", "marigold", "obsidian", "cinnabar", "juniper",
    ]
    out: List[QAExample] = []
    for a in range(num_authors):
        token = f"{vocab[a % len(vocab)]}{a:03d}"
        for q in range(qa_per_author):
            out.append(
                QAExample(
                    author_id=f"author_{a:04d}",
                    qa_id=q,
                    question=f"What is fact {q} about writer {token}?",
                    answer=(
                        f"Writer {token} is known for {token}-{q} and "
                        f"{vocab[int(rng.integers(len(vocab)))]}."
                    ),
                    row_index=a * qa_per_author + q,
                )
            )
    return out


def load_examples(
    dataset_cfg: Dict[str, Any], num_authors: Optional[int] = None
) -> List[QAExample]:
    """Dispatch on ``dataset_cfg['name']``: real TOFU, or synthetic for the smoke test."""
    name = dataset_cfg.get("name", "locuslab/TOFU")
    qa_per_author = int(dataset_cfg.get("qa_per_author", TOFU_QA_PER_AUTHOR))

    if name == SYNTHETIC_DATASET_NAME:
        if num_authors is None:
            raise ValueError("synthetic dataset requires num_authors")
        return load_synthetic_examples(num_authors, qa_per_author)

    return load_tofu_examples(
        dataset_name=name,
        config=dataset_cfg.get("config", "full"),
        revision=dataset_cfg.get("revision"),
        qa_per_author=qa_per_author,
        cache_dir=dataset_cfg.get("cache_dir"),
    )


def dataset_fingerprint(examples: Sequence[QAExample]) -> str:
    """Hash of the loaded content, so silent upstream dataset drift is detectable."""
    h = hashlib.sha256()
    h.update(f"n={len(examples)}".encode())
    for e in examples:
        h.update(e.author_id.encode())
        h.update(str(e.qa_id).encode())
        h.update(e.question.encode("utf-8"))
        h.update(e.answer.encode("utf-8"))
    return h.hexdigest()


def _index_by_key(examples: Sequence[QAExample]) -> Dict[Tuple[str, int], QAExample]:
    return {e.key: e for e in examples}


def global_order(examples: Sequence[QAExample], data_order_seed: int) -> List[Tuple[str, int]]:
    """One fixed permutation over ALL examples, shared by every run.

    Returns keys (not indices) so the ordering survives any change in load order.
    """
    keys = sorted(e.key for e in examples)
    rng = np.random.default_rng(data_order_seed)
    perm = rng.permutation(len(keys))
    return [keys[i] for i in perm]


def _batch_member_keys(manifest: Dict[str, Any], batch_ids: Sequence[str]) -> List[Tuple[str, int]]:
    batches = manifest["split"]["batches"]
    out: List[Tuple[str, int]] = []
    for bid in batch_ids:
        for author, qa in batches[bid]:
            out.append((str(author), int(qa)))
    return out


def retain_examples(
    manifest: Dict[str, Any], examples: Sequence[QAExample]
) -> List[QAExample]:
    """The fixed 180-author retain set ``D_r``, identical in every run."""
    retain = set(manifest["split"]["retain_authors"])
    return [e for e in examples if e.author_id in retain]


def candidate_examples(
    manifest: Dict[str, Any], examples: Sequence[QAExample]
) -> List[QAExample]:
    """All examples in the candidate pool ``D_f`` -- both signs.

    This is the scoring set: the audit evaluates the final model on all ``m``
    candidate batches, included-and-unlearned as well as never-seen.
    """
    by_key = _index_by_key(examples)
    keys = _batch_member_keys(manifest, manifest["split"]["batch_ids"])
    out = []
    for k in keys:
        if k not in by_key:
            raise KeyError(f"manifest references missing example {k}")
        out.append(by_key[k])
    return out


def forget_examples_for_run(
    manifest: Dict[str, Any], examples: Sequence[QAExample], run_id: str
) -> List[QAExample]:
    """The forget set handed to the unlearning algorithm: ``S_j = +1`` batches ONLY.

    Never returns ``S_j = -1`` candidates. Those were never trained on, so exposing
    them to the unlearner would leak the hidden sign vector into the pipeline and
    void the audit.
    """
    from .manifest import positive_candidates

    by_key = _index_by_key(examples)
    keys = _batch_member_keys(manifest, positive_candidates(manifest, run_id))
    return [by_key[k] for k in keys]


def training_examples_for_run(
    manifest: Dict[str, Any],
    examples: Sequence[QAExample],
    run_id: str,
) -> List[QAExample]:
    """``D(S) = D_r  U  D_f(S)``, emitted in the fixed global order.

    The order is the global permutation *restricted* to this run's examples, so
    positive and negative candidates are never handled asymmetrically.
    """
    from .manifest import positive_candidates

    by_key = _index_by_key(examples)
    present = {e.key for e in retain_examples(manifest, examples)}
    present |= set(_batch_member_keys(manifest, positive_candidates(manifest, run_id)))

    order = global_order(examples, manifest["seeds"]["data_order_seed"])
    return [by_key[k] for k in order if k in present]


def encode_example(
    tokenizer: Any,
    question: str,
    answer: str,
    max_length: int = 512,
    append_eos: bool = True,
    system_prompt: Optional[str] = None,
) -> Dict[str, List[int]]:
    """Chat-format one QA pair and build answer-only labels.

    The prompt is rendered with the tokenizer's chat template and
    ``add_generation_prompt=True``; the answer is tokenized separately with
    ``add_special_tokens=False`` and concatenated. ``labels`` masks every prompt
    position with ``IGNORE_INDEX``, so the loss covers answer tokens only.

    ``append_eos`` must match between training and scoring. It is recorded in the
    run config and re-read by the scorer for exactly that reason.

    Truncation drops from the RIGHT (the answer tail). A left truncation could remove
    the whole prompt and leave labels referring to a context that no longer exists.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})

    if getattr(tokenizer, "chat_template", None):
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    else:
        # Fallback for tiny test models with no chat template.
        prompt_text = (f"{system_prompt}\n\n" if system_prompt else "") + \
                      f"Question: {question}\nAnswer:"
        prompt_ids = tokenizer(prompt_text, add_special_tokens=True)["input_ids"]

    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    if append_eos and tokenizer.eos_token_id is not None:
        answer_ids = answer_ids + [tokenizer.eos_token_id]

    if not answer_ids:
        raise ValueError(f"answer tokenized to zero tokens: {answer!r}")

    # Keep at least one answer token: trim the prompt from the left if the prompt
    # alone would consume the whole budget.
    if len(prompt_ids) >= max_length:
        keep = max_length - min(len(answer_ids), max_length // 2)
        prompt_ids = prompt_ids[-max(keep, 1):]

    input_ids = (prompt_ids + answer_ids)[:max_length]
    n_prompt = min(len(prompt_ids), len(input_ids))
    labels = [IGNORE_INDEX] * n_prompt + list(input_ids[n_prompt:])

    assert len(labels) == len(input_ids), "labels/input_ids length mismatch"
    n_answer = sum(1 for t in labels if t != IGNORE_INDEX)
    if n_answer == 0:
        raise ValueError(
            f"no answer tokens survived truncation at max_length={max_length}"
        )

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids),
        "num_answer_tokens": n_answer,
        "num_prompt_tokens": n_prompt,
    }


@dataclass
class AnswerLossCollator:
    """Right-pads a batch and keeps padded positions out of the loss.

    Padding uses ``pad_token_id`` for ``input_ids``, ``0`` for ``attention_mask`` and
    ``IGNORE_INDEX`` for ``labels`` -- so padded positions are masked in the loss the
    same way prompt positions are.
    """

    pad_token_id: int
    label_pad_token_id: int = IGNORE_INDEX

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[Any, Any]:
        import torch

        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, labels, attn = [], [], []
        for f in features:
            pad = max_len - len(f["input_ids"])
            input_ids.append(list(f["input_ids"]) + [self.pad_token_id] * pad)
            labels.append(list(f["labels"]) + [self.label_pad_token_id] * pad)
            attn.append(list(f["attention_mask"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }
