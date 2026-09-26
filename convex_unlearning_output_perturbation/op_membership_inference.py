"""
Unlearn audit: single-partition membership inference for output perturbation.

Distinguish:
  - Retain: model trained on retain only, then Π_{C_0}(w_retain) + ξ (same σ)
  - Unlearn: full-data model with output perturbation Π_{C_0}(w_full) + ξ

Only compute epsilon_emp_lower (no epsilon_lb_avg_v).
"""
import json
import numpy as np
import pickle
from sklearn.metrics import roc_curve, auc, confusion_matrix
from scipy.stats import multivariate_normal, beta as beta_dist

from output_perturbation import sample_unlearned_models, clip_to_ball, gaussian_mechanism_sigma_general
from zcdp_pairwise import convex_pairwise_zcdp_audit


def clopper_pearson(x, n, confidence=0.95):
    """Clopper-Pearson (exact) confidence interval for binomial proportion p = x/n."""
    if n <= 0:
        return (np.nan, np.nan)
    x = int(np.clip(x, 0, n))
    alpha = 1.0 - confidence
    if x <= 0:
        lo, hi = 0.0, 1.0 - beta_dist.ppf(alpha / 2, n, 1)
    elif x >= n:
        lo, hi = beta_dist.ppf(alpha / 2, n, 1), 1.0
    else:
        lo = beta_dist.ppf(alpha / 2, x, n - x + 1)
        hi = beta_dist.ppf(1 - alpha / 2, x + 1, n - x)
    return (float(np.clip(lo, 0, 1)), float(np.clip(hi, 0, 1)))


def epsilon_empirical_lower_bound(tpr_low, fpr_high, delta):
    """
    Lower confidence bound on empirical epsilon:
    ε_emp^lower = max( log((1-δ-FP^high)/FN^high), log((1-δ-FN^high)/FP^high) ).
    FN^high = 1 - tpr_low. Returns np.nan when formula inapplicable.
    """
    fn_rate_high = 1.0 - tpr_low
    fp_high = fpr_high
    term1_ok = (1.0 - delta - fp_high) > 1e-12 and fn_rate_high > 1e-12
    term2_ok = (1.0 - delta - fn_rate_high) > 1e-12 and fp_high > 1e-12
    vals = []
    if term1_ok:
        vals.append(np.log((1.0 - delta - fp_high) / fn_rate_high))
    if term2_ok:
        vals.append(np.log((1.0 - delta - fn_rate_high) / fp_high))
    if not vals:
        return np.nan
    return float(np.max(vals))


def estimate_distribution(models, verbose=False, cov_regularization=1e-8):
    """Estimate mean and covariance of model distribution (for LLR)."""
    mean = np.mean(models, axis=0)
    cov = np.cov(models, rowvar=False)
    d = cov.shape[0]
    reg = cov_regularization * max(1.0, np.trace(cov) / d)
    cov = cov + reg * np.eye(d)
    std = np.std(models, axis=0)
    if verbose:
        print(f"  Estimated mean norm: {np.linalg.norm(mean):.6f}, cov cond: {np.linalg.cond(cov):.2e}")
    return {'mean': mean, 'cov': cov, 'std': std}


def log_likelihood_ratio(model, dist1, dist2):
    """LLR = log P(model|dist1) - log P(model|dist2). >0 => more likely dist1."""
    try:
        mvn1 = multivariate_normal(mean=dist1['mean'], cov=dist1['cov'], allow_singular=True)
        mvn2 = multivariate_normal(mean=dist2['mean'], cov=dist2['cov'], allow_singular=True)
        log_prob1 = mvn1.logpdf(model)
        log_prob2 = mvn2.logpdf(model)
        if not np.isfinite(log_prob1) or not np.isfinite(log_prob2):
            diff1 = model - dist1['mean']
            diff2 = model - dist2['mean']
            return float(np.linalg.norm(diff2) - np.linalg.norm(diff1))
        llr = log_prob1 - log_prob2
        if not np.isfinite(llr):
            diff1 = model - dist1['mean']
            diff2 = model - dist2['mean']
            return float(np.linalg.norm(diff2) - np.linalg.norm(diff1))
        return float(llr)
    except Exception:
        diff1 = model - dist1['mean']
        diff2 = model - dist2['mean']
        return float(np.linalg.norm(diff2) - np.linalg.norm(diff1))


def run_membership_inference_attack_single_partition(
    data, trained_retain, trained_full, C_0, epsilon, delta=0.01,
    n_samples_per_dist=500, n_test=500, random_state=42, confidence=0.95,
    conv_delta=None, conv_method="tight"
):
    """
    Single-partition unlearn audit: distinguish retain+noise vs full→output_perturbation+noise.

    Both sides use same σ from output perturbation formula.

    Returns dict with metrics including epsilon_empirical_lower, epsilon_lb_avg_v,
    and the direct convex pairwise zCDP quantities (rho_lb_conv, eps_from_rho_lb,
    rho_ub_noise, eps_from_rho_ub).

    conv_delta : δ used for the zCDP → (ε, δ)-DP conversion. Defaults to the
        audit `delta`, so eps_from_rho_* is directly comparable to `epsilon`.
    """
    ci_delta = 1.0 - confidence
    rng = np.random.RandomState(random_state)
    w_retain = np.asarray(trained_retain['weights'])
    w_full = np.asarray(trained_full['weights'])
    sigma = gaussian_mechanism_sigma_general(2.0 * C_0, epsilon, delta)

    # Distribution 1: unlearn (full → clip + noise)
    models_unlearn, _ = sample_unlearned_models(
        w_full, C_0, epsilon, delta, n_samples_per_dist, random_state=rng.randint(0, 10000)
    )
    # Distribution 2: retain (retain → clip + same noise)
    models_retain, _ = sample_unlearned_models(
        w_retain, C_0, epsilon, delta, n_samples_per_dist, random_state=rng.randint(0, 10000)
    )

    dist_unlearn = estimate_distribution(models_unlearn, verbose=True)
    dist_retain = estimate_distribution(models_retain, verbose=True)

    # Test: n_test from each, interleaved at random
    which_unlearn = rng.randint(0, 2, size=2 * n_test)
    n_u = int(which_unlearn.sum())
    n_r = 2 * n_test - n_u
    test_unlearn, _ = sample_unlearned_models(w_full, C_0, epsilon, delta, n_u, random_state=rng.randint(0, 10000))
    test_retain, _ = sample_unlearned_models(w_retain, C_0, epsilon, delta, n_r, random_state=rng.randint(0, 10000))
    test_models_ordered = []
    idx_u, idx_r = 0, 0
    for i in range(2 * n_test):
        if which_unlearn[i] == 1:
            test_models_ordered.append(test_unlearn[idx_u])
            idx_u += 1
        else:
            test_models_ordered.append(test_retain[idx_r])
            idx_r += 1
    test_models_ordered = np.array(test_models_ordered)
    labels = which_unlearn.copy()  # 1 = unlearn, 0 = retain

    # LLR: positive => predict unlearn (dist_unlearn)
    llrs = np.array([log_likelihood_ratio(m, dist_unlearn, dist_retain) for m in test_models_ordered])
    if np.any(np.isnan(llrs)):
        llrs = np.where(np.isfinite(llrs), llrs, 0.0)
    predictions = (llrs > 0).astype(int)

    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    fpr_curve, tpr_curve, _ = roc_curve(labels, llrs)
    roc_auc = auc(fpr_curve, tpr_curve)

    n_pos = int(tp + fn)
    n_neg = int(fp + tn)
    tpr_low, tpr_high = clopper_pearson(int(tp), n_pos, confidence=confidence)
    fpr_low, fpr_high = clopper_pearson(int(fp), n_neg, confidence=confidence)
    epsilon_emp_lower = epsilon_empirical_lower_bound(tpr_low, fpr_high, delta)

    # Direct convex pairwise zCDP auditor: def:zcdp_convex_unlearning compares the
    # two distributions the auditor observes, so the Renyi curve is rho*gamma with
    # no transitivity loss (unlike the general unlearning audit).
    zcdp = convex_pairwise_zcdp_audit(
        tp=int(tp), n_pos=n_pos, fp=int(fp), n_neg=n_neg,
        sensitivity=2.0 * C_0, sigma=sigma,
        ci_delta=ci_delta,
        conv_delta=(delta if conv_delta is None else conv_delta),
        conv_method=conv_method,
    )

    T = len(predictions)
    v_list = [2 if predictions[i] == labels[i] else 0 for i in range(T)]

    def _as_float(raw):
        """None (consistent with eps/rho = 0) or empty -> nan; else scalar float."""
        if raw is None:
            return np.nan
        arr = np.asarray(raw).ravel()
        return float(arr[0]) if arr.size else np.nan

    # Convex-setting avg-v tests. Both drop the transitivity loss the general
    # unlearning audit pays: no factor-of-2 halving on epsilon, and the direct
    # curve eps_gamma = rho*gamma (not the weak-triangle bound) for rho.
    try:
        from avg_v_convex import compute_avg_v_test_epsilon_lb_convex
        avg_v_res = compute_avg_v_test_epsilon_lb_convex(
            m=2, r=2, T=T, v_list=v_list, ci_delta=ci_delta
        )
        epsilon_lb_avg_v = _as_float(avg_v_res["epsilon_lb"])
    except Exception as e:
        avg_v_res = {}
        epsilon_lb_avg_v = np.nan
        import sys
        print(f"  Warning: convex avg-v epsilon_lb failed: {e}", file=sys.stderr)

    try:
        from avg_v_convex import compute_avg_v_test_rho_lb_convex
        avg_v_rho_res = compute_avg_v_test_rho_lb_convex(
            m=2, r=2, T=T, v_list=v_list, ci_delta=ci_delta,
            conv_delta=(delta if conv_delta is None else conv_delta),
            conv_method=conv_method,
        )
        rho_lb_avg_v = _as_float(avg_v_rho_res["rho_lb"])
        eps_from_rho_lb_avg_v = _as_float(avg_v_rho_res["eps_estimate"])
    except Exception as e:
        avg_v_rho_res = {}
        rho_lb_avg_v = np.nan
        eps_from_rho_lb_avg_v = np.nan
        import sys
        print(f"  Warning: convex avg-v rho_lb failed: {e}", file=sys.stderr)

    print("\n" + "="*70)
    print("ATTACK RESULTS (output perturbation unlearn audit)")
    print("="*70)
    print(f"  Accuracy: {accuracy:.4f}, TPR: {tpr:.4f} [{tpr_low:.4f}, {tpr_high:.4f}], FPR: {fpr:.4f} [{fpr_low:.4f}, {fpr_high:.4f}]")
    print(f"  ROC AUC: {roc_auc:.4f}")
    if np.isfinite(epsilon_emp_lower):
        print(f"  ε_emp^lower (δ={delta}): {epsilon_emp_lower:.4f}")
    else:
        print(f"  ε_emp^lower (δ={delta}): — (inapplicable)")
    if np.isfinite(epsilon_lb_avg_v):
        print(f"  ε_lb (convex avg-v, m=2,r=2, no /2): {epsilon_lb_avg_v:.4f}")
    else:
        print(f"  ε_lb (convex avg-v, m=2,r=2, no /2): — (consistent with ε=0)")
    if np.isfinite(rho_lb_avg_v):
        print(f"  ρ_lb (convex avg-v, ε_γ=ργ): {rho_lb_avg_v:.6f}"
              f"  →  ε(ρ_lb) = {eps_from_rho_lb_avg_v:.4f}")
    else:
        print(f"  ρ_lb (convex avg-v, ε_γ=ργ): — (consistent with ρ=0)")
    print("-"*70)
    print("  Direct convex pairwise zCDP auditor")
    print(f"    one-sided CP (α={zcdp['cp_alpha_each']:.4g} each): "
          f"TPR^low={zcdp['tpr_low_1sided']:.4f}, FPR^high={zcdp['fpr_high_1sided']:.4f}, "
          f"TNR^low={zcdp['tnr_low_1sided']:.4f}, FNR^high={zcdp['fnr_high_1sided']:.4f}")
    print(f"    ρ_LB,conv^(p) = {zcdp['rho_lb_conv']:.6f}  "
          f"(γ*={zcdp['gamma_star']:.4f}, branch={zcdp['branch']})")
    print(f"    ε(ρ_LB) [δ_conv={zcdp['conv_delta']:.4g}, {zcdp['conv_method']}] = "
          f"{zcdp['eps_from_rho_lb']:.4f}")
    print(f"    ρ_UB (noise added for ε={epsilon}, σ={sigma:.4f}, Δ={2.0 * C_0:.4f}) = "
          f"{zcdp['rho_ub_noise']:.6f}  →  ε(ρ_UB) = {zcdp['eps_from_rho_ub']:.4f}")
    print("="*70)

    return {
        'rho_lb_avg_v': rho_lb_avg_v,
        'eps_from_rho_lb_avg_v': eps_from_rho_lb_avg_v,
        'avg_v_epsilon_detail': avg_v_res,
        'avg_v_rho_detail': avg_v_rho_res,
        'rho_lb_conv': zcdp['rho_lb_conv'],
        'eps_from_rho_lb': zcdp['eps_from_rho_lb'],
        'rho_ub_noise': zcdp['rho_ub_noise'],
        'eps_from_rho_ub': zcdp['eps_from_rho_ub'],
        'zcdp_pairwise': zcdp,
        'accuracy': accuracy,
        'tpr': tpr,
        'fpr': fpr,
        'roc_auc': roc_auc,
        'tpr_low': tpr_low,
        'tpr_high': tpr_high,
        'fpr_low': fpr_low,
        'fpr_high': fpr_high,
        'epsilon_empirical_lower': epsilon_emp_lower,
        'epsilon_lb_avg_v': epsilon_lb_avg_v,
        'sigma': sigma,
        'epsilon': epsilon,
        'delta': delta,
        'confidence': confidence,
        'fpr_curve': fpr_curve,
        'tpr_curve': tpr_curve,
        'predictions': predictions,
        'labels': labels,
        'llrs': llrs,
        'confusion_matrix': (tn, fp, fn, tp),
    }


def save_attack_metrics_json(results, filepath):
    """Save metrics, epsilon_emp_lower, and the convex pairwise zCDP bounds to JSON."""
    def _scalar(x):
        if isinstance(x, (np.floating, float)) and not np.isfinite(x):
            return None
        return float(x) if hasattr(x, '__float__') else x
    metrics = {
        'epsilon': results['epsilon'],
        'delta': results['delta'],
        'sigma': results['sigma'],
        'accuracy': _scalar(results['accuracy']),
        'tpr': _scalar(results['tpr']),
        'fpr': _scalar(results['fpr']),
        'roc_auc': _scalar(results['roc_auc']),
        'tpr_low': _scalar(results['tpr_low']),
        'tpr_high': _scalar(results['tpr_high']),
        'fpr_low': _scalar(results['fpr_low']),
        'fpr_high': _scalar(results['fpr_high']),
        'epsilon_empirical_lower': _scalar(results['epsilon_empirical_lower']),
        'epsilon_lb_avg_v': _scalar(results.get('epsilon_lb_avg_v', float('nan'))),
        # Convex avg-v tests (no factor-of-2 on eps; eps_gamma = rho*gamma for rho)
        'rho_lb_avg_v': _scalar(results.get('rho_lb_avg_v', float('nan'))),
        'eps_from_rho_lb_avg_v': _scalar(results.get('eps_from_rho_lb_avg_v', float('nan'))),
        'avg_v_epsilon_detail': {k: _scalar(v) for k, v in
                                 (results.get('avg_v_epsilon_detail') or {}).items()},
        'avg_v_rho_detail': {k: _scalar(v) for k, v in
                             (results.get('avg_v_rho_detail') or {}).items()},
        # Direct convex pairwise zCDP auditor
        'rho_lb_conv': _scalar(results.get('rho_lb_conv')),
        'eps_from_rho_lb': _scalar(results.get('eps_from_rho_lb')),
        'rho_ub_noise': _scalar(results.get('rho_ub_noise')),
        'eps_from_rho_ub': _scalar(results.get('eps_from_rho_ub')),
        'zcdp_pairwise': {k: _scalar(v) for k, v in
                          (results.get('zcdp_pairwise') or {}).items()},
    }
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics (JSON) saved to: {filepath}")


def plot_roc_curve(results, save_path='roc_curve.png'):
    """Plot ROC curve."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.plot(results['fpr_curve'], results['tpr_curve'], lw=2, label=f"ROC (AUC = {results['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC = 0.5)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"Output Perturbation Unlearn Audit (ε = {results['epsilon']})")
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"ROC curve saved to: {save_path}")
