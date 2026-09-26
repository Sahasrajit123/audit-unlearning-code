"""Unit tests for the unlearning methods, focused on the NPO objective.

NPO is the most intricate component: it involves a frozen reference model, a
saturating loss, and a sequence-level (not token-level) log-likelihood. These tests
pin the pieces against hand arithmetic rather than trusting the implementation.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from audit_tofu.tofu_data import IGNORE_INDEX
from audit_tofu.unlearn import UNLEARNING_METHODS, apply_unlearning, sequence_logprob


class FakeOut:
    def __init__(self, logits):
        self.logits = logits


class ConstantLogitModel(torch.nn.Module):
    """Emits the same logit vector at every position, so scores are hand-computable."""

    def __init__(self, table: torch.Tensor):
        super().__init__()
        self.table = table
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kw):
        b, t = input_ids.shape
        logits = self.table.view(1, 1, -1).expand(b, t, self.table.numel()).clone()
        return FakeOut(logits)


def _batch(input_ids, labels):
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.ones(
            len(input_ids), len(input_ids[0]), dtype=torch.long
        ),
    }


# --- sequence_logprob ---------------------------------------------------------

def test_sequence_logprob_sums_only_answer_tokens():
    vocab = 8
    table = torch.randn(vocab, generator=torch.Generator().manual_seed(0))
    model = ConstantLogitModel(table)
    lp = torch.log_softmax(table, dim=-1)

    # positions:      0   1   2   3
    # labels:       ign ign  t2  t3     -> after the causal shift, answers are t2, t3
    batch = _batch([[5, 6, 2, 3]], [[IGNORE_INDEX, IGNORE_INDEX, 2, 3]])

    got = sequence_logprob(model, batch, reduction="sum")
    expected = float(lp[2] + lp[3])
    assert got.shape == (1,)
    assert float(got[0]) == pytest.approx(expected, rel=1e-6)


def test_sequence_logprob_mean_normalizes_by_answer_length():
    vocab = 8
    table = torch.randn(vocab, generator=torch.Generator().manual_seed(1))
    model = ConstantLogitModel(table)
    lp = torch.log_softmax(table, dim=-1)
    batch = _batch([[5, 6, 2, 3]], [[IGNORE_INDEX, IGNORE_INDEX, 2, 3]])

    s = float(sequence_logprob(model, batch, "sum")[0])
    m = float(sequence_logprob(model, batch, "mean")[0])
    assert m == pytest.approx(s / 2, rel=1e-6)
    assert m == pytest.approx(float(lp[2] + lp[3]) / 2, rel=1e-6)


def test_sequence_logprob_ignores_padded_positions():
    vocab = 8
    table = torch.randn(vocab, generator=torch.Generator().manual_seed(2))
    model = ConstantLogitModel(table)
    lp = torch.log_softmax(table, dim=-1)

    # Two examples of different answer length, the shorter one right-padded.
    batch = _batch(
        [[5, 6, 2, 3], [5, 6, 2, 0]],
        [[IGNORE_INDEX, IGNORE_INDEX, 2, 3], [IGNORE_INDEX, IGNORE_INDEX, 2, IGNORE_INDEX]],
    )
    got = sequence_logprob(model, batch, "sum")
    assert float(got[0]) == pytest.approx(float(lp[2] + lp[3]), rel=1e-6)
    assert float(got[1]) == pytest.approx(float(lp[2]), rel=1e-6), \
        "padded position leaked into the sequence log-likelihood"


def test_sequence_logprob_rejects_unknown_reduction():
    model = ConstantLogitModel(torch.randn(8))
    batch = _batch([[5, 2]], [[IGNORE_INDEX, 2]])
    with pytest.raises(ValueError, match="reduction"):
        sequence_logprob(model, batch, "median")


# --- the NPO objective --------------------------------------------------------

def _npo_loss(cur_lp: float, ref_lp: float, beta: float) -> float:
    """L = (2/beta) * softplus(beta * (cur - ref)), the paper/OpenUnlearning form."""
    z = beta * (cur_lp - ref_lp)
    return (2.0 / beta) * math.log1p(math.exp(z))


def test_npo_loss_matches_closed_form_and_is_minimized_by_lowering_forget_likelihood():
    beta = 0.1
    ref = -10.0

    # At parity, softplus(0) = log 2.
    at_parity = _npo_loss(ref, ref, beta)
    assert at_parity == pytest.approx((2.0 / beta) * math.log(2), rel=1e-12)

    # Pushing the policy BELOW the reference reduces the loss ...
    assert _npo_loss(ref - 5, ref, beta) < at_parity
    # ... and raising it above increases the loss.
    assert _npo_loss(ref + 5, ref, beta) > at_parity


def test_npo_loss_saturates_which_is_the_point_versus_gradient_ascent():
    """The sigmoid saturates, so per-example gradients stay bounded.

    Plain gradient ascent on the forget set has an unbounded objective; NPO's does
    not, which is why the model degrades more gracefully.
    """
    beta = 0.1
    ref = 0.0
    # As cur -> -inf the loss tends to 0 rather than diverging.
    losses = [_npo_loss(ref - d, ref, beta) for d in (10, 100, 1000)]
    assert losses[0] > losses[1] > losses[2] > 0.0
    assert losses[2] < 1e-3

    # The gradient magnitude also vanishes: d/dcur = 2*sigmoid(beta*(cur-ref)).
    def grad(cur):
        return 2.0 / (1.0 + math.exp(-beta * (cur - ref)))

    assert grad(ref - 1000) < 1e-6
    assert grad(ref) == pytest.approx(1.0, rel=1e-9)
    assert grad(ref + 1000) == pytest.approx(2.0, rel=1e-6), "bounded above by 2"


def test_npo_softplus_implementation_agrees_with_torch():
    """The implementation uses F.softplus; confirm it equals the log1p(exp) form."""
    import torch.nn.functional as F

    beta = 0.1
    for cur, ref in [(-10.0, -10.0), (-20.0, -10.0), (-5.0, -10.0), (-100.0, -1.0)]:
        # float64: this is a check of the mathematical identity, so float32's ~1e-7
        # relative precision would be the thing under test rather than the formula.
        z = torch.tensor(beta * (cur - ref), dtype=torch.float64)
        got = float((2.0 / beta) * F.softplus(z))
        assert got == pytest.approx(_npo_loss(cur, ref, beta), rel=1e-12)


# --- method dispatch ----------------------------------------------------------

def test_noop_returns_the_model_untouched():
    model = ConstantLogitModel(torch.randn(8))
    before = [p.detach().clone() for p in model.parameters()]

    metrics = apply_unlearning("noop", model, None, [], [], {}, seed=0)

    after = list(model.parameters())
    for b, a in zip(before, after):
        assert torch.equal(b, a), "noop modified the model"
    assert metrics["label"] == "noop"
    assert metrics["duration_seconds"] == 0.0
    assert "control" in metrics["note"].lower()


def test_unknown_method_is_rejected():
    model = ConstantLogitModel(torch.randn(8))
    with pytest.raises(ValueError, match="unknown method"):
        apply_unlearning("gradient_ascent", model, None, [], [], {}, seed=0)


def test_method_registry_contains_the_spec_required_methods():
    """The spec's "initial required methods" must all be present.

    Extras are allowed and expected -- `grad_ascent` and `grad_diff` were added from
    OpenUnlearning. The exact tuple is pinned separately in
    `test_method_registry_includes_the_openunlearning_additions`.
    """
    for required in ("noop", "npo", "retain_ft"):
        assert required in UNLEARNING_METHODS


# --- end-to-end on a tiny real model -----------------------------------------

@pytest.fixture(scope="module")
def tiny_model():
    transformers = pytest.importorskip("transformers")
    from audit_tofu.modeling import load_model_and_tokenizer

    try:
        return load_model_and_tokenizer(
            "hf-internal-testing/tiny-random-LlamaForCausalLM",
            dtype="float32",
            attn_implementation="eager",
            gradient_checkpointing=False,
            for_training=True,
        )
    except Exception as exc:  # offline / no cache
        pytest.skip(f"tiny model unavailable: {exc}")


def _tiny_examples(n=6):
    from audit_tofu.tofu_data import load_synthetic_examples

    return load_synthetic_examples(n, 2)


def test_npo_runs_and_reports_its_variant(tiny_model):
    model, tokenizer, _ = tiny_model
    ex = _tiny_examples()
    forget, retain = ex[:4], ex[4:]

    cfg = {
        "beta": 0.1, "retain_weight": 1.0, "epochs": 1,
        "micro_batch_size": 2, "gradient_accumulation_steps": 1,
        "learning_rate": 1e-4, "max_seq_length": 32,
    }
    m = apply_unlearning("npo", model, tokenizer, forget, retain, cfg, seed=0)

    assert m["label"] == "npo"
    assert m["variant"] == "npo_rt", "retain_weight > 0 means the NPO_RT variant"
    assert m["beta"] == 0.1
    assert m["num_forget_examples"] == 4
    assert m["optimizer_steps"] >= 1
    assert m["history"] and "npo_loss" in m["history"][0]
    assert math.isfinite(m["history"][0]["npo_loss"])


def test_pure_npo_is_labelled_differently(tiny_model):
    model, tokenizer, _ = tiny_model
    ex = _tiny_examples()
    cfg = {
        "beta": 0.1, "retain_weight": 0.0, "epochs": 1,
        "micro_batch_size": 2, "gradient_accumulation_steps": 1,
        "learning_rate": 1e-4, "max_seq_length": 32,
    }
    m = apply_unlearning("npo", model, tokenizer, ex[:4], ex[4:], cfg, seed=0)
    assert m["variant"] == "npo", "retain_weight == 0 is pure NPO"
    assert m["num_retain_examples"] == 0
    assert m["history"][0]["retain_loss"] == 0.0


def test_npo_raises_on_an_empty_forget_set(tiny_model):
    model, tokenizer, _ = tiny_model
    cfg = {"epochs": 1, "micro_batch_size": 2, "max_seq_length": 32}
    with pytest.raises(ValueError, match="empty forget set"):
        apply_unlearning("npo", model, tokenizer, [], _tiny_examples()[:2], cfg, seed=0)


def test_npo_raises_forget_likelihood_relative_to_reference(tiny_model):
    """The objective's whole purpose: forget-set likelihood must go DOWN.

    Measured against the pre-unlearning model, which is exactly NPO's reference.
    """
    import copy

    model, tokenizer, _ = tiny_model
    model = copy.deepcopy(model)
    ex = _tiny_examples(8)
    forget, retain = ex[:8], ex[8:] or ex[:2]

    from audit_tofu.scoring import score_examples

    before = score_examples(model, tokenizer, forget, max_length=32, batch_size=2)
    mean_before = sum(r["loss"] for r in before) / len(before)

    cfg = {
        "beta": 0.1, "retain_weight": 0.0, "epochs": 3,
        "micro_batch_size": 2, "gradient_accumulation_steps": 1,
        "learning_rate": 5e-3, "max_seq_length": 32, "max_grad_norm": 1.0,
    }
    apply_unlearning("npo", model, tokenizer, forget, retain, cfg, seed=0)

    after = score_examples(model, tokenizer, forget, max_length=32, batch_size=2)
    mean_after = sum(r["loss"] for r in after) / len(after)

    # Loss is negative log-likelihood, so "likelihood down" means "loss up".
    assert mean_after > mean_before, (
        f"NPO should reduce forget-set likelihood: loss went {mean_before:.4f} "
        f"-> {mean_after:.4f}"
    )


def test_retain_ft_trains_on_retain_and_reports_forget_count(tiny_model):
    model, tokenizer, _ = tiny_model
    ex = _tiny_examples(10)
    forget, retain = ex[:4], ex[4:]

    cfg = {
        "epochs": 1, "micro_batch_size": 2, "gradient_accumulation_steps": 1,
        "learning_rate": 1e-4, "max_seq_length": 32,
    }
    m = apply_unlearning("retain_ft", model, tokenizer, forget, retain, cfg, seed=0)

    assert m["label"] == "retain_ft"
    assert m["num_examples"] == len(retain), "retain_ft must train on the retain set"
    assert m["num_forget_examples"] == 4, "forget count recorded but not trained on"
    assert m["optimizer_steps"] >= 1


# --- GradAscent and GradDiff (OpenUnlearning parity) --------------------------

def test_method_registry_includes_the_openunlearning_additions():
    from audit_tofu.unlearn import FORGET_SET_METHODS

    assert UNLEARNING_METHODS == (
        "noop", "npo", "retain_ft", "grad_ascent", "grad_diff", "simnpo"
    )
    assert FORGET_SET_METHODS == ("npo", "grad_ascent", "grad_diff", "simnpo")


def test_grad_ascent_objective_is_the_negated_mean_nll():
    """Upstream GradAscent is exactly `loss = -outputs.loss` (mean token CE)."""
    vocab = 8
    table = torch.randn(vocab, generator=torch.Generator().manual_seed(5))
    model = ConstantLogitModel(table)
    lp = torch.log_softmax(table, dim=-1)

    batch = _batch([[5, 6, 2, 3]], [[IGNORE_INDEX, IGNORE_INDEX, 2, 3]])
    # HF-style mean CE over the two answer tokens.
    mean_nll = float(-(lp[2] + lp[3]) / 2)
    # The ascent objective negates it, so minimizing L maximizes the NLL.
    assert -mean_nll < 0
    # Sanity: our sequence_logprob sum relates to the mean by the token count.
    s = float(sequence_logprob(model, batch, "sum")[0])
    assert mean_nll == pytest.approx(-s / 2, rel=1e-6)


def test_grad_ascent_runs_and_reports_forget_only(tiny_model):
    model, tokenizer, _ = tiny_model
    ex = _tiny_examples(8)
    forget, retain = ex[:4], ex[4:]

    cfg = {
        "epochs": 1, "micro_batch_size": 2, "gradient_accumulation_steps": 1,
        "learning_rate": 1e-4, "max_seq_length": 32,
    }
    m = apply_unlearning("grad_ascent", model, tokenizer, forget, retain, cfg, seed=0)

    assert m["label"] == "grad_ascent"
    assert m["variant"] == "grad_ascent"
    assert m["used_reference_model"] is False, "GradAscent needs no reference model"
    assert m["retain_weight"] == 0.0, "upstream GradAscent has no retain term"
    assert m["num_retain_examples"] == 0
    assert m["num_forget_examples"] == 4
    assert m["history"][0]["retain_loss"] == 0.0
    assert math.isfinite(m["history"][0]["forget_nll"])


def test_grad_ascent_ignores_a_retain_weight_and_warns(tiny_model, capsys):
    """Inventing a retain term would be unfaithful to upstream; it must be refused."""
    model, tokenizer, _ = tiny_model
    ex = _tiny_examples(8)
    msgs = []
    cfg = {
        "epochs": 1, "micro_batch_size": 2, "gradient_accumulation_steps": 1,
        "learning_rate": 1e-4, "max_seq_length": 32,
        "alpha": 5.0, "retain_weight": 5.0,
    }
    m = apply_unlearning(
        "grad_ascent", model, tokenizer, ex[:4], ex[4:], cfg,
        seed=0, logger=msgs.append,
    )
    assert m["retain_weight"] == 0.0
    assert m["num_retain_examples"] == 0
    assert any("ignored" in s and "forget-only" in s for s in msgs), msgs


def test_grad_ascent_raises_forget_loss(tiny_model):
    """The defining behaviour: ascent must increase forget-set loss."""
    import copy as _copy

    model, tokenizer, _ = tiny_model
    model = _copy.deepcopy(model)
    ex = _tiny_examples(8)
    forget = ex[:8]

    from audit_tofu.scoring import score_examples

    before = score_examples(model, tokenizer, forget, max_length=32, batch_size=2)
    mean_before = sum(r["loss"] for r in before) / len(before)

    cfg = {
        "epochs": 2, "micro_batch_size": 2, "gradient_accumulation_steps": 1,
        "learning_rate": 5e-3, "max_seq_length": 32, "max_grad_norm": 1.0,
    }
    apply_unlearning("grad_ascent", model, tokenizer, forget, [], cfg, seed=0)

    after = score_examples(model, tokenizer, forget, max_length=32, batch_size=2)
    mean_after = sum(r["loss"] for r in after) / len(after)
    assert mean_after > mean_before, (
        f"grad_ascent should raise forget loss: {mean_before:.4f} -> {mean_after:.4f}"
    )


def test_grad_diff_uses_both_terms_and_no_reference_model(tiny_model):
    model, tokenizer, _ = tiny_model
    ex = _tiny_examples(10)
    forget, retain = ex[:4], ex[4:]

    cfg = {
        "gamma": 1.0, "alpha": 1.0, "epochs": 1,
        "micro_batch_size": 2, "gradient_accumulation_steps": 1,
        "learning_rate": 1e-4, "max_seq_length": 32,
    }
    m = apply_unlearning("grad_diff", model, tokenizer, forget, retain, cfg, seed=0)

    assert m["label"] == "grad_diff"
    assert m["used_reference_model"] is False, "GradDiff (NLL retain) needs no ref model"
    assert m["forget_weight"] == 1.0
    assert m["retain_weight"] == 1.0
    assert m["num_retain_examples"] == len(retain)
    assert m["history"][0]["retain_loss"] > 0.0, "retain term must actually be computed"


def test_grad_diff_accepts_upstream_parameter_names(tiny_model):
    """Upstream calls them gamma/alpha; our npo block calls the retain weight
    retain_weight. Both spellings must work."""
    model, tokenizer, _ = tiny_model
    ex = _tiny_examples(10)
    base = {
        "epochs": 1, "micro_batch_size": 2, "gradient_accumulation_steps": 1,
        "learning_rate": 1e-4, "max_seq_length": 32,
    }
    a = apply_unlearning(
        "grad_diff", model, tokenizer, ex[:4], ex[4:], {**base, "alpha": 3.0}, seed=0
    )
    b = apply_unlearning(
        "grad_diff", model, tokenizer, ex[:4], ex[4:],
        {**base, "retain_weight": 3.0}, seed=0,
    )
    assert a["retain_weight"] == b["retain_weight"] == 3.0


def test_grad_diff_with_zero_alpha_reduces_to_grad_ascent(tiny_model):
    """alpha=0 removes the retain term, leaving pure ascent."""
    model, tokenizer, _ = tiny_model
    ex = _tiny_examples(8)
    cfg = {
        "gamma": 1.0, "alpha": 0.0, "epochs": 1,
        "micro_batch_size": 2, "gradient_accumulation_steps": 1,
        "learning_rate": 1e-4, "max_seq_length": 32,
    }
    m = apply_unlearning("grad_diff", model, tokenizer, ex[:4], ex[4:], cfg, seed=0)
    assert m["num_retain_examples"] == 0
    assert m["history"][0]["retain_loss"] == 0.0


def test_forget_set_methods_share_the_loop_and_report_a_common_schema(tiny_model):
    """A difference in epsilon should reflect the objective, not the plumbing."""
    model, tokenizer, _ = tiny_model
    ex = _tiny_examples(10)
    base = {
        "epochs": 1, "micro_batch_size": 2, "gradient_accumulation_steps": 1,
        "learning_rate": 1e-4, "max_seq_length": 32,
    }
    common = {
        "label", "variant", "epochs", "optimizer_steps", "num_forget_examples",
        "num_retain_examples", "retain_weight", "forget_weight",
        "used_reference_model", "diverged", "final_forget_nll", "history",
        "duration_seconds", "seed",
    }
    for method in ("npo", "grad_ascent", "grad_diff"):
        m = apply_unlearning(method, model, tokenizer, ex[:4], ex[4:], base, seed=0)
        assert common <= set(m), f"{method} missing {common - set(m)}"
        assert m["optimizer_steps"] >= 1
        for rec in m["history"]:
            assert {"epoch", "forget_loss", "forget_nll", "retain_loss"} <= set(rec)


def test_npo_history_keeps_its_original_field_name(tiny_model):
    """Existing npo results carry `npo_loss`; the refactor must not drop it."""
    model, tokenizer, _ = tiny_model
    ex = _tiny_examples(8)
    cfg = {
        "beta": 0.1, "retain_weight": 1.0, "epochs": 1,
        "micro_batch_size": 2, "gradient_accumulation_steps": 1,
        "learning_rate": 1e-4, "max_seq_length": 32,
    }
    m = apply_unlearning("npo", model, tokenizer, ex[:4], ex[4:], cfg, seed=0)
    assert "npo_loss" in m["history"][0]
    assert m["history"][0]["npo_loss"] == m["history"][0]["forget_loss"]
    assert m["used_reference_model"] is True
    assert m["beta"] == 0.1


def test_divergence_is_flagged(tiny_model):
    """Runaway ascent must be reported, not silently produce a tiny epsilon."""
    import copy as _copy

    model, tokenizer, _ = tiny_model
    model = _copy.deepcopy(model)
    ex = _tiny_examples(8)
    msgs = []
    cfg = {
        "epochs": 4, "micro_batch_size": 2, "gradient_accumulation_steps": 1,
        "learning_rate": 0.5, "max_seq_length": 32, "max_grad_norm": 1e9,
    }
    m = apply_unlearning(
        "grad_ascent", model, tokenizer, ex, [], cfg, seed=0, logger=msgs.append
    )
    if m["final_forget_nll"] is not None and m["final_forget_nll"] > m["divergence_threshold"]:
        assert m["diverged"] is True
        assert any("collapsing toward noise" in s for s in msgs), msgs


# --- SimNPO (reference-free NPO) ---------------------------------------------

def _simnpo_loss(nll_per_token: float, beta: float, delta: float = 0.0) -> float:
    """L = -(2/beta) * logsigmoid(beta * (nll - delta)), upstream's SimNPO core."""
    z = beta * (nll_per_token - delta)
    # logsigmoid(z) = -log(1+exp(-z))
    return (2.0 / beta) * math.log1p(math.exp(-z))


def test_simnpo_is_registered_as_a_reference_free_forget_method():
    from audit_tofu.unlearn import FORGET_SET_METHODS

    assert "simnpo" in UNLEARNING_METHODS
    assert "simnpo" in FORGET_SET_METHODS


def test_simnpo_loss_matches_closed_form_and_rewards_raising_forget_nll():
    beta = 4.5
    base = _simnpo_loss(1.0, beta)
    # Higher forget NLL (model less sure of the forget answer) => LOWER loss.
    assert _simnpo_loss(3.0, beta) < base
    assert _simnpo_loss(0.2, beta) > base


def test_simnpo_saturates_like_npo_not_like_gradient_ascent():
    """Bounded below by 0, so it cannot run away the way grad_ascent does."""
    beta = 4.5
    losses = [_simnpo_loss(x, beta) for x in (2, 5, 20)]
    assert losses[0] > losses[1] > losses[2] > 0.0
    assert losses[2] < 1e-6

    # gradient wrt nll is -2*sigmoid(-beta*nll), bounded in (-2, 0)
    def grad(nll):
        return -2.0 / (1.0 + math.exp(beta * nll))

    assert -2.0 < grad(0.0) < 0.0
    assert abs(grad(20.0)) < 1e-6, "saturates, unlike unbounded ascent"


def test_simnpo_implementation_agrees_with_torch_logsigmoid():
    import torch.nn.functional as F

    beta, delta = 4.5, 0.0
    for nll in (0.2, 1.0, 3.0, 8.0):
        z = torch.tensor(beta * (nll - delta), dtype=torch.float64)
        got = float(-(2.0 / beta) * F.logsigmoid(z))
        assert got == pytest.approx(_simnpo_loss(nll, beta, delta), rel=1e-12)


def test_simnpo_delta_shifts_the_operating_point():
    beta = 4.5
    assert _simnpo_loss(1.0, beta, delta=0.0) < _simnpo_loss(1.0, beta, delta=2.0)


def test_simnpo_uses_length_normalized_nll_not_the_sum():
    """The "Sim" in SimNPO: upstream divides the summed NLL by the token count."""
    vocab = 8
    table = torch.randn(vocab, generator=torch.Generator().manual_seed(9))
    model = ConstantLogitModel(table)
    lp = torch.log_softmax(table, dim=-1)
    batch = _batch([[5, 6, 2, 3]], [[IGNORE_INDEX, IGNORE_INDEX, 2, 3]])

    mean_lp = float(sequence_logprob(model, batch, "mean")[0])
    sum_lp = float(sequence_logprob(model, batch, "sum")[0])
    assert mean_lp == pytest.approx(sum_lp / 2, rel=1e-6)
    # the quantity SimNPO feeds the sigmoid is -mean_lp
    assert -mean_lp == pytest.approx(float(-(lp[2] + lp[3]) / 2), rel=1e-6)


def test_simnpo_runs_without_a_reference_model(tiny_model):
    model, tokenizer, _ = tiny_model
    ex = _tiny_examples(10)
    cfg = {
        "beta": 4.5, "delta": 0.0, "gamma": 0.125, "alpha": 1.0, "epochs": 1,
        "micro_batch_size": 2, "gradient_accumulation_steps": 1,
        "learning_rate": 1e-4, "max_seq_length": 32,
    }
    m = apply_unlearning("simnpo", model, tokenizer, ex[:4], ex[4:], cfg, seed=0)

    assert m["label"] == "simnpo"
    assert m["variant"] == "simnpo"
    assert m["used_reference_model"] is False, "SimNPO must not clone a reference model"
    assert m["beta"] == 4.5
    assert m["delta"] == 0.0
    assert m["forget_weight"] == 0.125
    assert m["retain_weight"] == 1.0
    assert m["sequence_reduction"] == "mean"
    assert m["history"][0]["retain_loss"] > 0.0


def test_simnpo_defaults_match_upstream_when_config_is_silent(tiny_model):
    """A bare config must fall back to SimNPO.yaml's values, not NPO's."""
    model, tokenizer, _ = tiny_model
    ex = _tiny_examples(10)
    cfg = {"epochs": 1, "micro_batch_size": 2, "gradient_accumulation_steps": 1,
           "learning_rate": 1e-4, "max_seq_length": 32}
    m = apply_unlearning("simnpo", model, tokenizer, ex[:4], ex[4:], cfg, seed=0)
    assert m["beta"] == 4.5, "must not inherit NPO's beta=0.1"
    assert m["forget_weight"] == 0.125, "must not inherit gamma=1.0"


def test_simnpo_raises_forget_loss(tiny_model):
    import copy as _copy

    model, tokenizer, _ = tiny_model
    model = _copy.deepcopy(model)
    ex = _tiny_examples(8)

    from audit_tofu.scoring import score_examples

    before = score_examples(model, tokenizer, ex, max_length=32, batch_size=2)
    mean_before = sum(r["loss"] for r in before) / len(before)

    # beta=0.1, NOT the 4.5 default. The tiny random model already sits at
    # NLL ~10.4 (near uniform over its vocab), where beta=4.5 puts the objective
    # deep in saturation: |dL/d(nll)| = 2*sigmoid(-4.5*10.4) ~ 1e-20, i.e. exactly
    # no gradient. That saturation is the method working as designed and is pinned
    # by test_simnpo_saturates_at_upstream_beta_on_a_high_loss_model; here we need
    # an operating point where the objective actually has slope.
    cfg = {
        "beta": 0.1, "gamma": 1.0, "alpha": 0.0, "epochs": 3,
        "micro_batch_size": 2, "gradient_accumulation_steps": 1,
        "learning_rate": 5e-3, "max_seq_length": 32, "max_grad_norm": 1.0,
    }
    apply_unlearning("simnpo", model, tokenizer, ex, [], cfg, seed=0)

    after = score_examples(model, tokenizer, ex, max_length=32, batch_size=2)
    mean_after = sum(r["loss"] for r in after) / len(after)
    assert mean_after > mean_before, (
        f"simnpo should raise forget loss: {mean_before:.4f} -> {mean_after:.4f}"
    )


def test_npo_still_uses_its_own_defaults(tiny_model):
    """Adding SimNPO's method-dependent fallbacks must not disturb NPO."""
    model, tokenizer, _ = tiny_model
    ex = _tiny_examples(10)
    cfg = {"epochs": 1, "micro_batch_size": 2, "gradient_accumulation_steps": 1,
           "learning_rate": 1e-4, "max_seq_length": 32}
    m = apply_unlearning("npo", model, tokenizer, ex[:4], ex[4:], cfg, seed=0)
    assert m["beta"] == 0.1
    assert m["forget_weight"] == 1.0
    assert m["used_reference_model"] is True


def test_simnpo_saturates_at_upstream_beta_on_a_high_loss_model(tiny_model):
    """At beta=4.5 the objective is flat once the forget NLL is already high.

    This is the documented saturating behaviour, not a defect: SimNPO stops pushing
    an example it already considers forgotten. It also means beta interacts strongly
    with the model's loss scale -- see docs/RESOURCE_ESTIMATE.md on the gradient
    magnitude at Llama's actual TOFU operating point.
    """
    import copy as _copy

    model, tokenizer, _ = tiny_model
    model = _copy.deepcopy(model)
    ex = _tiny_examples(8)

    from audit_tofu.scoring import score_examples

    before = score_examples(model, tokenizer, ex, max_length=32, batch_size=2)
    mean_before = sum(r["loss"] for r in before) / len(before)
    assert mean_before > 5.0, "fixture assumption: the tiny model has a high NLL"

    cfg = {
        "beta": 4.5, "gamma": 1.0, "alpha": 0.0, "epochs": 3,
        "micro_batch_size": 2, "gradient_accumulation_steps": 1,
        "learning_rate": 5e-3, "max_seq_length": 32, "max_grad_norm": 1.0,
    }
    apply_unlearning("simnpo", model, tokenizer, ex, [], cfg, seed=0)

    after = score_examples(model, tokenizer, ex, max_length=32, batch_size=2)
    mean_after = sum(r["loss"] for r in after) / len(after)
    assert abs(mean_after - mean_before) < 1e-3, (
        f"expected a saturated no-op, got {mean_before:.4f} -> {mean_after:.4f}"
    )


def test_simnpo_gradient_magnitude_is_documented():
    """Pin the saturation arithmetic that motivates the beta caveat in the docs."""
    def grad(nll, beta, delta=0.0):
        return 2.0 / (1.0 + math.exp(beta * (nll - delta)))

    # Upstream beta=4.5 at Llama's measured TOFU forget NLL (~1.28): tiny.
    assert grad(1.28, 4.5) == pytest.approx(6.28e-3, rel=0.05)
    # After the gamma=0.125 forget weight, ~3 orders below the alpha=1.0 retain term.
    assert grad(1.28, 4.5) * 0.125 < 1e-3
    # A smaller beta restores a usable gradient at the same operating point.
    assert grad(1.28, 0.1) > 0.9


def test_divergence_threshold_is_relative_to_vocab_and_reachable():
    """Regression: a fixed threshold above ln(vocab) can never fire.

    The loss of a model emitting a uniform distribution is exactly ln(V) -- 11.76 for
    Llama-3.2's 128,256 tokens. An earlier fixed 20.0 was therefore unreachable, and
    a fully collapsed `grad_ascent` run (measured forget NLL 10.84, i.e. 101% of the
    uniform-output loss) went unflagged.
    """
    from audit_tofu.unlearn import (
        DIVERGENCE_FRACTION_OF_UNIFORM,
        divergence_threshold,
    )

    class Llama:
        class config:
            vocab_size = 128256

    t = divergence_threshold(Llama())
    ln_v = math.log(128256)
    assert t == pytest.approx(DIVERGENCE_FRACTION_OF_UNIFORM * ln_v)
    assert t < ln_v, "threshold must be BELOW the uniform loss to be reachable"

    # The measured collapse must trip it; healthy methods must not.
    assert 10.84 > t, "measured grad_ascent collapse must now be flagged"
    assert 1.31 < t, "simnpo's healthy forget NLL must not be flagged"
    assert 3.07 < t, "grad_diff's healthy forget NLL must not be flagged"


def test_divergence_threshold_scales_with_vocab_and_has_a_fallback():
    from audit_tofu.unlearn import divergence_threshold

    class Small:
        class config:
            vocab_size = 32000

    class Big:
        class config:
            vocab_size = 256000

    assert divergence_threshold(Small()) < divergence_threshold(Big())
    # No usable config -> a fixed fallback rather than a crash.
    assert divergence_threshold(object()) == pytest.approx(7.0)

    class Degenerate:
        class config:
            vocab_size = 1

    assert divergence_threshold(Degenerate()) == pytest.approx(7.0)
