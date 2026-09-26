"""Tests for attention-backend resolution, optimizer construction, and the
LoRA scope labelling."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from audit_tofu.modeling import (
    build_optimizer,
    maybe_wrap_lora,
    peak_memory_stats,
    reset_peak_memory,
    resolve_attn_implementation,
)


class TinyModel(torch.nn.Module):
    def __init__(self, frozen: bool = False):
        super().__init__()
        self.a = torch.nn.Linear(4, 4)
        self.b = torch.nn.Linear(4, 4)
        if frozen:
            for p in self.b.parameters():
                p.requires_grad_(False)


# --- attention backend --------------------------------------------------------

def test_explicit_attn_implementation_is_passed_through():
    for impl in ("flash_attention_2", "sdpa", "eager"):
        assert resolve_attn_implementation(impl) == impl


def test_auto_resolves_to_a_supported_backend():
    got = resolve_attn_implementation("auto")
    assert got in ("flash_attention_2", "sdpa")


def test_auto_falls_back_to_sdpa_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_attn_implementation("auto") == "sdpa"


def test_auto_falls_back_to_sdpa_on_pre_ampere(monkeypatch):
    """FlashAttention-2 needs SM80+; older cards must degrade, not crash."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (7, 5))
    assert resolve_attn_implementation("auto") == "sdpa"


def test_unknown_attn_implementation_is_rejected():
    with pytest.raises(ValueError, match="unknown attn implementation"):
        resolve_attn_implementation("xformers")


# --- optimizer ----------------------------------------------------------------

def test_adamw_is_built_with_the_configured_hyperparameters():
    model = TinyModel()
    opt, info = build_optimizer(model, {
        "optimizer": "adamw",
        "learning_rate": 3e-5,
        "weight_decay": 0.05,
        "betas": (0.9, 0.95),
        "adam_epsilon": 1e-6,
    })
    assert info["optimizer"] == "adamw"
    assert isinstance(opt, torch.optim.AdamW)
    g = opt.param_groups[0]
    assert g["lr"] == pytest.approx(3e-5)
    assert g["weight_decay"] == pytest.approx(0.05)
    assert tuple(g["betas"]) == (0.9, 0.95)
    assert g["eps"] == pytest.approx(1e-6)


def test_optimizer_defaults_match_the_documented_values():
    opt, info = build_optimizer(TinyModel(), {})
    g = opt.param_groups[0]
    assert g["lr"] == pytest.approx(1e-5), "documented default LR"
    assert g["weight_decay"] == pytest.approx(0.01)
    assert info["optimizer"] == "adamw"


def test_optimizer_only_receives_trainable_parameters():
    model = TinyModel(frozen=True)
    opt, _ = build_optimizer(model, {})
    n_opt = sum(p.numel() for grp in opt.param_groups for p in grp["params"])
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_opt == n_trainable
    assert n_opt < sum(p.numel() for p in model.parameters())


def test_eight_bit_request_falls_back_loudly_when_unavailable(capsys):
    """A silent fallback would invalidate the measured memory numbers."""
    pytest.importorskip  # noqa: B018
    try:
        import bitsandbytes  # noqa: F401
        pytest.skip("bitsandbytes is installed; the fallback path is not exercised")
    except ImportError:
        pass

    opt, info = build_optimizer(TinyModel(), {"optimizer": "adamw_8bit"})
    captured = capsys.readouterr()
    assert info["optimizer"] == "adamw", "should report what actually ran"
    assert isinstance(opt, torch.optim.AdamW)
    assert "WARNING" in captured.out
    assert "bitsandbytes" in captured.out
    assert "memory" in captured.out.lower()


# --- LoRA scope labelling -----------------------------------------------------

def test_lora_disabled_leaves_the_model_alone():
    model = TinyModel()
    out, info = maybe_wrap_lora(model, None)
    assert out is model
    assert info == {"lora": False}

    out, info = maybe_wrap_lora(model, {"enabled": False, "r": 8})
    assert out is model
    assert info["lora"] is False


def test_lora_enabled_carries_an_audit_scope_warning():
    """A LoRA run audits a different pipeline; the metadata must say so."""
    pytest.importorskip("peft", reason="LoRA is an optional fallback")
    from transformers import AutoModelForCausalLM

    try:
        model = AutoModelForCausalLM.from_pretrained(
            "hf-internal-testing/tiny-random-LlamaForCausalLM"
        )
    except Exception as exc:
        pytest.skip(f"tiny model unavailable: {exc}")

    wrapped, info = maybe_wrap_lora(model, {"enabled": True, "r": 4, "alpha": 8})
    assert info["lora"] is True
    assert info["lora_r"] == 4
    assert info["trainable_parameters"] < sum(p.numel() for p in model.parameters())
    assert "audit_scope_warning" in info
    assert "does not transfer" in info["audit_scope_warning"]


# --- memory instrumentation ---------------------------------------------------

def test_peak_memory_stats_reports_both_metrics():
    reset_peak_memory()
    stats = peak_memory_stats()
    assert set(stats) == {"peak_allocated_gib", "peak_reserved_gib"}
    assert stats["peak_allocated_gib"] >= 0.0
    assert stats["peak_reserved_gib"] >= 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_peak_memory_tracks_a_real_allocation():
    reset_peak_memory()
    before = peak_memory_stats()["peak_allocated_gib"]
    x = torch.empty(int(64 * 1024**2 / 4), dtype=torch.float32, device="cuda")  # 64 MiB
    after = peak_memory_stats()["peak_allocated_gib"]
    assert after > before
    assert after - before >= 0.05
    del x
    torch.cuda.empty_cache()


def test_instrumentation_degrades_without_torch(monkeypatch):
    """The CPU-only audit-math path must not require torch for instrumentation."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name == "torch":
            raise ImportError("simulated: torch absent")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    reset_peak_memory()  # must not raise
    assert peak_memory_stats() == {
        "peak_allocated_gib": 0.0,
        "peak_reserved_gib": 0.0,
    }
