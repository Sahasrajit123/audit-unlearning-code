"""
Hessian-based unlearning integration for Shakespeare plays base
"""


import torch
import torch.nn as nn
from torch.autograd import grad
from torch.utils.data import DataLoader
import time

def grad_batch(batch_loader, lam, model, device):
    """Compute gradient of data loss only (no regularization) averaged over all batches.

    Uses reduction='mean' so each batch loss is normalized over B*T tokens,
    matching the per-token normalization used in newton_update.
    lam is accepted for API compatibility but NOT applied here — regularization
    belongs only in the Hessian loss (newton_update) to simulate (H + λI).
    """
    model.train()
    # reduction='mean' normalizes over B*T tokens per batch, consistent with newton_update
    criterion = nn.CrossEntropyLoss(reduction='mean')
    params = [p for p in model.parameters() if p.requires_grad]
    grad_acc = [torch.zeros_like(p).cpu() for p in params]
    num_batches = 0
    for batch_idx, (data, targets) in enumerate(batch_loader):
        data, targets = data.to(device), targets.to(device)
        outputs, _ = model(data)
        B, T, V = outputs.shape
        # Flatten to (B*T, V) for cross-entropy; targets to (B*T,)
        loss = criterion(outputs.view(B * T, V), targets.view(B * T))
        grad_mini = list(grad(loss, params, retain_graph=False))
        for i in range(len(grad_acc)):
            grad_acc[i] += grad_mini[i].cpu().detach()
        num_batches += 1
    # Average gradient across batches (gradient of data loss only — no L2 term)
    for i in range(len(grad_acc)):
        grad_acc[i] /= max(num_batches, 1)
    return [p.to(device) for p in grad_acc]

def hvp(y, w, v):
    first_grads = grad(y, w, retain_graph=True, create_graph=True)
    elemwise_products = 0
    for grad_elem, v_elem in zip(first_grads, v):
        elemwise_products += torch.sum(grad_elem * v_elem)
    return_grads = grad(elemwise_products, w, create_graph=True)
    return return_grads

def newton_update(g, res_loader, lam, gamma, model, s1, s2, scale, device, generator=None, logger=None):
    """Approximate H^{-1}g via LiSSA (Neumann series, s1 independent runs averaged).

    LiSSA recursion (Agarwal et al.):
        H_0 = g
        H_{j+1} = g + (I - (H_data + λI) / scale) * H_j

    After s2 steps, H_{s2} ≈ scale * (H_data + λI)^{-1} * g.
    Dividing by scale in H_res gives the true Newton direction H^{-1}g.

    scale must strictly upper-bound the largest eigenvalue of (H_data + λI)
    so that ||I - (H + λI)/scale|| < 1 and the series converges.

    Regularization (lam + gamma) uses squared L2 (= 0.5 * ||θ||²) to add
    a (lam+gamma)*I shift to the Hessian, improving conditioning.
    This regularization is NOT present in grad_batch (g = gradient of data loss only).
    """
    model.train()
    # Must match grad_batch: reduction='mean' for consistent per-token normalization
    criterion = nn.CrossEntropyLoss(reduction='mean')
    params = [p for p in model.parameters() if p.requires_grad]
    H_res = [torch.zeros_like(p) for p in g]
    res_data = list(res_loader)
    log_interval = max(s2 // 10, 1)
    valid_runs = 0
    total_attempts = 0
    max_attempts = 3 * s1  # cap retries to avoid infinite loop
    attempt = 0
    while valid_runs < s1 and attempt < max_attempts:
        attempt += 1
        H = [p.clone() for p in g]
        prev_h_norm = sum(h.pow(2).sum().item() for h in H) ** 0.5
        diverged = False
        for j in range(s2):
            idx = torch.randint(0, len(res_data), (1,), generator=generator).item()
            data, target = res_data[idx]
            data, target = data.to(device), target.to(device)
            with torch.backends.cudnn.flags(enabled=False):
                z, _ = model(data)
            B, T, V = z.shape
            # Flatten to (B*T, V) for cross-entropy; targets to (B*T,)
            loss = criterion(z.view(B * T, V), target.view(B * T))
            # Squared L2 regularization: adds (lam+gamma)*I to the Hessian
            l2_reg = 0.5 * sum((param ** 2).sum() for param in params)
            loss = loss + (lam + gamma) * l2_reg
            H_s = hvp(loss, params, H)
            # LiSSA recursion: H_{j+1} = g + (I - Hess/scale) * H_j
            with torch.no_grad():
                for k in range(len(params)):
                    H[k] = H[k] + g[k] - H_s[k] / scale
            h_norm = sum(h.pow(2).sum().item() for h in H) ** 0.5
            if logger is not None and (j + 1) % log_interval == 0:
                logger.info(f"[Hessian Unlearning] run={valid_runs+1}/{s1} (attempt {attempt}), s2={j+1}/{s2}: ||H||={h_norm:.6f}")
                # Only check for divergence after the first log checkpoint (skip warm-up growth)
                if (j + 1) > log_interval and h_norm > 10.0 * prev_h_norm:
                    logger.warning(f"[Hessian Unlearning] run={valid_runs+1}/{s1} (attempt {attempt}), s2={j+1}/{s2}: ||H|| jumped {prev_h_norm:.2f}→{h_norm:.2f} — diverged, retrying")
                    diverged = True
                    break
                prev_h_norm = h_norm
        if diverged:
            continue
        h_norm = sum(h.pow(2).sum().item() for h in H) ** 0.5
        if logger is not None:
            logger.info(f"[Hessian Unlearning] run={valid_runs+1}/{s1} done (attempt {attempt}): ||H||={h_norm:.6f}")
        # Divide by scale: H_{s2} ≈ scale * H^{-1}g  →  H[k]/scale ≈ H^{-1}g
        for k in range(len(params)):
            H_res[k] = H_res[k] + H[k] / scale
        valid_runs += 1
    if valid_runs == 0:
        raise RuntimeError("[Hessian Unlearning] All LiSSA attempts diverged — increase scale.")
    if valid_runs < s1:
        logger.warning(f"[Hessian Unlearning] Only {valid_runs}/{s1} runs converged after {attempt} attempts — averaging over {valid_runs} runs.")
    # Average over valid s1 runs
    return [p / valid_runs for p in H_res]


def unlearn_hessian_style(model, forget_loader, retain_loader, train_loader, val_loader, criterion, device, run_dir, logger, forget_seed, unlearning_cfg=None):
    """
    Hessian-based unlearning wrapper. This should be adapted to call the certified-deep-unlearning Hessian unlearning logic.

    Args:
        model: Trained model to unlearn from
        forget_loader: DataLoader for forget set
        retain_loader: DataLoader for retain set
        train_loader: DataLoader for combined train set
        val_loader: DataLoader for validation set
        criterion: Loss function
        device: Device to run on
        run_dir: Directory for saving models
        logger: Logger
        forget_seed: Integer seed for Hessian randomness (should be unique per run, e.g. run_seed + CONST)
        unlearning_cfg: Dict of unlearning hyperparameters
    Returns:
        model, metrics
    """

    logger.info("[Hessian Unlearning] Starting Hessian-based unlearning...")
    start_time = time.time()

    # --- Hyperparameters ---
    # unlearning_cfg may be the full "unlearning" dict (with nested "hessian_unlearning")
    # or already the inner hessian-specific dict — handle both
    _cfg_raw = unlearning_cfg or {}
    cfg = _cfg_raw.get("hessian_unlearning", _cfg_raw)

    def _require(key):
        if key not in cfg:
            raise KeyError(f"[Hessian Unlearning] Missing required config key: '{key}' in hessian_unlearning config")
        return cfg[key]

    weight_decay = _require("weight_decay")
    gamma       = _require("gamma")
    s1          = _require("s1")
    s2          = _require("s2")
    scale       = _require("scale")
    std         = _require("std")
    # Use the passed forget_seed as the Hessian randomness seed (should be unique per run)
    generator = torch.Generator(device='cpu')
    generator.manual_seed(forget_seed)
    logger.info(f"[Hessian Unlearning] Using forget_seed={forget_seed} as Hessian randomness seed (independent from forget set sampling)")
    logger.info(f"[Hessian Unlearning] Hyperparams: s1={s1}, s2={s2}, scale={scale}, gamma={gamma}, weight_decay={weight_decay}, std={std}")

    # --- Main Hessian-based unlearning ---
    logger.info("[Hessian Unlearning] Computing gradients on retain set...")
    g = grad_batch(retain_loader, weight_decay, model, device)

    logger.info("[Hessian Unlearning] Computing Newton update...")
    delta = newton_update(g, retain_loader, weight_decay, gamma, model, s1, s2, scale, device, generator=generator, logger=logger)

    delta_norms = [d.norm().item() for d in delta]
    delta_total = sum(d.pow(2).sum().item() for d in delta) ** 0.5
    g_total = sum(gg.pow(2).sum().item() for gg in g) ** 0.5
    logger.info(f"[Hessian Unlearning] Gradient norm ||g||={g_total:.6f}")
    logger.info(f"[Hessian Unlearning] Delta norm ||delta||={delta_total:.6f}  (per-layer: {[f'{n:.4f}' for n in delta_norms]})")
    logger.info(f"[Hessian Unlearning] Scale ratio ||delta||/||g||={delta_total/max(g_total,1e-12):.4f}  (>>1 means scale too small, <<1 means scale too large)")

    logger.info("[Hessian Unlearning] Updating model parameters...")
    for i, param in enumerate(model.parameters()):
        # Generate noise on CPU (generator is CPU-based) then move to device
        noise = torch.randn(param.data.size(), generator=generator).to(device) * std
        param.data.add_(-delta[i] + noise)

    torch.save(model.state_dict(), str(run_dir / "model_unlearnt.pt"))
    logger.info(f"[Hessian Unlearning] Unlearning complete. Model saved to {run_dir / 'model_unlearnt.pt'}")
    metrics = {
        "strategy": "hessian_unlearning",
        "status": "complete",
        "elapsed_sec": time.time() - start_time,
    }
    return model, metrics
