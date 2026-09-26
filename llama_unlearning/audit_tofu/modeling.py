"""Model/tokenizer construction, attention backend fallback, optimizers, LoRA.

Every run must start from *exactly* the same pretrained checkpoint. We deliberately
do NOT use the released TOFU "full" checkpoint: each run has its own candidate
inclusion vector ``S`` and therefore needs its own fine-tune from the base model.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

__all__ = [
    "resolve_attn_implementation",
    "load_model_and_tokenizer",
    "build_optimizer",
    "maybe_wrap_lora",
    "peak_memory_stats",
    "reset_peak_memory",
]


def resolve_attn_implementation(requested: str = "auto") -> str:
    """Pick an attention backend, falling back safely.

    ``auto`` prefers FlashAttention-2 when the package imports AND the GPU supports
    bf16 (SM80+); otherwise SDPA, which is always available in supported torch and
    is numerically fine -- just slower.
    """
    if requested not in ("auto", "flash_attention_2", "sdpa", "eager"):
        raise ValueError(f"unknown attn implementation {requested!r}")
    if requested != "auto":
        return requested

    try:
        import torch

        if not torch.cuda.is_available():
            return "sdpa"
        major, _ = torch.cuda.get_device_capability()
        if major < 8:
            return "sdpa"
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except Exception:
        return "sdpa"


def load_model_and_tokenizer(
    model_id: str,
    *,
    dtype: str = "bfloat16",
    attn_implementation: str = "auto",
    gradient_checkpointing: bool = True,
    trust_remote_code: bool = False,
    device_map: Optional[str] = None,
    cache_dir: Optional[str] = None,
    for_training: bool = True,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """Load the base model and tokenizer.

    Returns ``(model, tokenizer, info)`` where ``info`` records the resolved backend
    and dtype, so the effective setup is saved with the run rather than inferred.
    """
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]

    # transformers v5 renamed `torch_dtype` to `dtype`; passing the old name still
    # works but emits a deprecation warning on every model load, and there are
    # 4 loads per run x 30 runs. Pick the right keyword for the installed version.
    tf_major = int(transformers.__version__.split(".")[0])
    dtype_kw = "dtype" if tf_major >= 5 else "torch_dtype"

    attn = resolve_attn_implementation(attn_implementation)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=trust_remote_code, cache_dir=cache_dir
    )
    if tokenizer.pad_token_id is None:
        # Llama-3.2-Instruct ships no pad token. Reusing EOS is safe because padded
        # positions are masked out of the loss by AnswerLossCollator.
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    def _load(attn_impl: str):
        return AutoModelForCausalLM.from_pretrained(
            model_id,
            attn_implementation=attn_impl,
            trust_remote_code=trust_remote_code,
            device_map=device_map,
            cache_dir=cache_dir,
            **{dtype_kw: torch_dtype},
        )

    try:
        model = _load(attn)
    except (ImportError, ValueError) as exc:
        if attn == "flash_attention_2":
            # FlashAttention-2 imported but the model or build rejected it.
            print(f"[modeling] flash_attention_2 unavailable ({exc}); using sdpa")
            attn = "sdpa"
            model = _load(attn)
        else:
            raise RuntimeError(f"failed to load {model_id}: {exc}") from exc

    if for_training and gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    else:
        model.config.use_cache = not for_training

    # Generation during training is disabled by construction: we never call
    # .generate(), and the audit score is a teacher-forced loss, not a generation.
    info = {
        "model_id": model_id,
        "dtype": dtype,
        "attn_implementation": attn,
        "attn_requested": attn_implementation,
        "gradient_checkpointing": bool(for_training and gradient_checkpointing),
        "num_parameters": sum(p.numel() for p in model.parameters()),
    }
    return model, tokenizer, info


def maybe_wrap_lora(model: Any, lora_cfg: Optional[Dict[str, Any]]) -> Tuple[Any, Dict[str, Any]]:
    """Optionally attach LoRA adapters.

    LoRA is a FALLBACK, not the headline configuration. An audit of a LoRA pipeline
    is an audit of *that parameter-efficient training/unlearning pipeline* -- its
    epsilon does not transfer to the full-parameter pipeline, because LoRA
    constrains which directions training and unlearning can move in at all. Results
    produced this way are labelled accordingly in the run metadata.
    """
    if not lora_cfg or not lora_cfg.get("enabled"):
        return model, {"lora": False}

    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("alpha", 32)),
        lora_dropout=float(lora_cfg.get("dropout", 0.0)),
        target_modules=lora_cfg.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model, {
        "lora": True,
        "lora_r": cfg.r,
        "lora_alpha": cfg.lora_alpha,
        "trainable_parameters": trainable,
        "audit_scope_warning": (
            "LoRA run: this audits a parameter-efficient training/unlearning "
            "pipeline, not the full-parameter one. epsilon does not transfer."
        ),
    }


def build_optimizer(model: Any, cfg: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    """AdamW, optionally 8-bit for the memory-saving mode."""
    import torch

    params = [p for p in model.parameters() if p.requires_grad]
    lr = float(cfg.get("learning_rate", 1e-5))
    wd = float(cfg.get("weight_decay", 0.01))
    betas = tuple(cfg.get("betas", (0.9, 0.999)))
    eps = float(cfg.get("adam_epsilon", 1e-8))

    if cfg.get("optimizer", "adamw") == "adamw_8bit":
        try:
            import bitsandbytes as bnb

            opt = bnb.optim.AdamW8bit(
                params, lr=lr, betas=betas, eps=eps, weight_decay=wd
            )
            return opt, {"optimizer": "adamw_8bit"}
        except ImportError:
            # Falling back loudly: an 8-bit request that silently became 32-bit
            # would change the measured peak memory that sizing decisions rest on.
            print(
                "[modeling] WARNING: bitsandbytes unavailable; "
                "falling back to fp32 AdamW. Peak memory will be higher."
            )

    opt = torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=wd)
    return opt, {"optimizer": "adamw"}


def reset_peak_memory() -> None:
    """No-op when torch or CUDA is absent: this is instrumentation, not a dependency."""
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_memory_stats() -> Dict[str, float]:
    """Actual measured peak CUDA memory, in GiB -- not a theoretical estimate.

    Reports zeros (rather than raising) when torch or CUDA is unavailable, so the
    CPU-only audit-math path stays usable.
    """
    zero = {"peak_allocated_gib": 0.0, "peak_reserved_gib": 0.0}
    try:
        import torch
    except ImportError:
        return zero
    if not torch.cuda.is_available():
        return zero
    gib = 1024.0 ** 3
    return {
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / gib,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / gib,
    }
