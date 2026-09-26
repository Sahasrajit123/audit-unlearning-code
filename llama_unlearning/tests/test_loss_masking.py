"""Spec validation 1: prompt tokens are excluded from the loss.

The label-construction half runs anywhere, using a stub tokenizer. The half that
verifies the realized loss value needs torch and is skipped without it.
"""

from __future__ import annotations

import math

import pytest

from audit_tofu.tofu_data import IGNORE_INDEX, AnswerLossCollator, encode_example


class StubTokenizer:
    """Whitespace tokenizer with a vocabulary built on demand.

    Mimics only the surface the encoder uses, so label construction can be tested
    without transformers. ``chat_template`` toggles the templated path.
    """

    def __init__(self, chat_template: bool = True):
        self.chat_template = "stub" if chat_template else None
        self.eos_token_id = 2
        self.eos_token = "</s>"
        self.pad_token_id = 0
        self.padding_side = "right"
        self._vocab: dict[str, int] = {"</s>": 2, "<pad>": 0}
        self._next = 3

    def _id(self, tok: str) -> int:
        if tok not in self._vocab:
            self._vocab[tok] = self._next
            self._next += 1
        return self._vocab[tok]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        parts = [f"<|{m['role']}|> {m['content']}" for m in messages]
        if add_generation_prompt:
            parts.append("<|assistant|>")
        return " ".join(parts)

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": [self._id(t) for t in text.split()]}


def test_prompt_tokens_are_masked_and_answer_tokens_are_not():
    tok = StubTokenizer()
    q, a = "Who wrote this book?", "It was written by Aurelio Vasquez."

    enc = encode_example(tok, q, a, max_length=128, append_eos=True)

    n_prompt = enc["num_prompt_tokens"]
    n_answer = enc["num_answer_tokens"]

    assert len(enc["labels"]) == len(enc["input_ids"])
    assert n_prompt > 0 and n_answer > 0

    # Every prompt position masked ...
    assert all(l == IGNORE_INDEX for l in enc["labels"][:n_prompt]), \
        "a prompt token is present in the loss"
    # ... and no answer position masked.
    assert all(l != IGNORE_INDEX for l in enc["labels"][n_prompt:]), \
        "an answer token was masked out of the loss"

    # Unmasked labels must equal the corresponding inputs (teacher forcing).
    assert enc["labels"][n_prompt:] == enc["input_ids"][n_prompt:]
    assert n_answer == len(enc["input_ids"]) - n_prompt


def test_answer_token_count_matches_the_answer_plus_eos():
    tok = StubTokenizer()
    a = "one two three four five"

    with_eos = encode_example(tok, "Q?", a, max_length=128, append_eos=True)
    without = encode_example(tok, "Q?", a, max_length=128, append_eos=False)

    assert with_eos["num_answer_tokens"] == 6, "5 answer tokens + EOS"
    assert without["num_answer_tokens"] == 5
    assert with_eos["input_ids"][-1] == tok.eos_token_id
    assert without["input_ids"][-1] != tok.eos_token_id


def test_question_text_does_not_leak_into_unmasked_labels():
    """A stronger form: no question token id may appear among unmasked labels.

    Uses lexically disjoint question and answer vocabularies so any leak is visible.
    """
    tok = StubTokenizer()
    q = "alpha bravo charlie delta"
    a = "echo foxtrot golf hotel"

    enc = encode_example(tok, q, a, max_length=128, append_eos=False)
    q_ids = {tok._id(t) for t in q.split()}
    unmasked = [l for l in enc["labels"] if l != IGNORE_INDEX]

    assert not (set(unmasked) & q_ids), "question token ids appear in the loss"
    assert set(unmasked) == {tok._id(t) for t in a.split()}


def test_chat_template_is_applied_when_present():
    """The templated path must go through apply_chat_template, not the fallback."""
    tok_chat = StubTokenizer(chat_template=True)
    tok_plain = StubTokenizer(chat_template=False)

    templated = encode_example(tok_chat, "Q?", "A.", max_length=128)
    plain = encode_example(tok_plain, "Q?", "A.", max_length=128)

    # Role markers are interned only if apply_chat_template actually ran.
    assert "<|user|>" in tok_chat._vocab
    assert "<|assistant|>" in tok_chat._vocab
    # The fallback path uses the plain Question/Answer scaffold instead.
    assert "<|assistant|>" not in tok_plain._vocab
    assert "Question:" in tok_plain._vocab

    # Either way the answer side is identical in length.
    assert templated["num_answer_tokens"] == plain["num_answer_tokens"]


def test_system_prompt_lengthens_only_the_prompt():
    tok = StubTokenizer()
    a = "one two three"
    without = encode_example(tok, "Q?", a, max_length=128)
    withsys = encode_example(tok, "Q?", a, max_length=128, system_prompt="Be terse.")
    assert withsys["num_prompt_tokens"] > without["num_prompt_tokens"]
    assert withsys["num_answer_tokens"] == without["num_answer_tokens"]


def test_truncation_keeps_answer_tokens():
    tok = StubTokenizer()
    q = " ".join(f"q{i}" for i in range(300))
    a = " ".join(f"a{i}" for i in range(20))

    enc = encode_example(tok, q, a, max_length=32, append_eos=True)
    assert len(enc["input_ids"]) <= 32
    assert enc["num_answer_tokens"] > 0, "truncation must not remove every answer token"
    assert all(l == IGNORE_INDEX for l in enc["labels"][: enc["num_prompt_tokens"]])


def test_empty_answer_is_rejected():
    tok = StubTokenizer()
    with pytest.raises(ValueError, match="zero tokens"):
        encode_example(tok, "Q?", "", max_length=64, append_eos=False)


# --- collator: padding must also be excluded from the loss -------------------

def test_collator_pads_labels_with_ignore_index():
    pytest.importorskip("torch")
    tok = StubTokenizer()
    feats = [
        encode_example(tok, "Q?", "short", max_length=64, append_eos=False),
        encode_example(tok, "Q?", "a much longer answer here", max_length=64,
                       append_eos=False),
    ]
    batch = AnswerLossCollator(pad_token_id=tok.pad_token_id)(feats)

    assert batch["input_ids"].shape == batch["labels"].shape == batch["attention_mask"].shape

    # Padded positions: attention 0 and label IGNORE_INDEX.
    pad_positions = batch["attention_mask"] == 0
    assert (batch["labels"][pad_positions] == IGNORE_INDEX).all()
    assert (batch["input_ids"][pad_positions] == tok.pad_token_id).all()

    # Answer-token count is preserved through padding.
    n_unmasked = (batch["labels"] != IGNORE_INDEX).sum(dim=1).tolist()
    assert n_unmasked == [f["num_answer_tokens"] for f in feats]


# --- the realized loss equals a hand-computed answer-only mean ---------------

def test_scored_loss_equals_manual_answer_only_mean():
    """End-to-end check that score_examples implements l_z(f) exactly.

    Builds a deterministic fake model whose logits are fixed, computes the expected
    answer-only mean negative log-likelihood by hand, and compares.
    """
    torch = pytest.importorskip("torch")

    from audit_tofu.scoring import score_examples
    from audit_tofu.tofu_data import QAExample

    tok = StubTokenizer()
    vocab_size = 64

    class FakeOut:
        def __init__(self, logits):
            self.logits = logits

    class FakeModel(torch.nn.Module):
        """Deterministic logits, independent of the input, so the answer is exact."""

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            g = torch.Generator().manual_seed(0)
            self.table = torch.randn(vocab_size, generator=g)

        def forward(self, input_ids=None, attention_mask=None, **kw):
            b, t = input_ids.shape
            logits = self.table.view(1, 1, -1).expand(b, t, vocab_size).clone()
            return FakeOut(logits)

    model = FakeModel()
    ex = QAExample("author_0000", 0, "alpha bravo", "echo foxtrot golf", 0)

    rows = score_examples(
        model, tok, [ex], max_length=64, append_eos=False, batch_size=1
    )

    # Hand computation: uniform-over-positions log-softmax of the fixed table.
    enc = encode_example(tok, ex.question, ex.answer, max_length=64, append_eos=False)
    logprobs = torch.log_softmax(model.table, dim=-1)
    answer_ids = enc["input_ids"][enc["num_prompt_tokens"]:]
    # The causal shift means position t predicts token t+1, so the FIRST answer
    # token is predicted from the last prompt position; all answer tokens are scored.
    expected = -sum(float(logprobs[i]) for i in answer_ids) / len(answer_ids)

    assert rows[0]["num_answer_tokens"] == len(answer_ids)
    assert rows[0]["loss"] == pytest.approx(expected, rel=1e-6)


def test_scored_loss_ignores_prompt_content():
    """Changing only the question must not change the answer-only loss.

    With a fixed-logit model the answer loss is a function of the answer tokens
    alone; if prompt tokens leaked into the mean, this would fail.
    """
    torch = pytest.importorskip("torch")

    from audit_tofu.scoring import score_examples
    from audit_tofu.tofu_data import QAExample

    tok = StubTokenizer()
    vocab_size = 64

    class FakeOut:
        def __init__(self, logits):
            self.logits = logits

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            g = torch.Generator().manual_seed(1)
            self.table = torch.randn(vocab_size, generator=g)

        def forward(self, input_ids=None, attention_mask=None, **kw):
            b, t = input_ids.shape
            return FakeOut(self.table.view(1, 1, -1).expand(b, t, vocab_size).clone())

    model = FakeModel()
    a = "echo foxtrot golf"
    r1 = score_examples(
        model, tok, [QAExample("a", 0, "short q", a, 0)],
        max_length=64, append_eos=False, batch_size=1,
    )
    r2 = score_examples(
        model, tok, [QAExample("a", 0, "a considerably longer question here", a, 0)],
        max_length=64, append_eos=False, batch_size=1,
    )
    assert r1[0]["num_answer_tokens"] == r2[0]["num_answer_tokens"]
    assert r1[0]["loss"] == pytest.approx(r2[0]["loss"], rel=1e-9)
