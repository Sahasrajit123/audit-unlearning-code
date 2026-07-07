"""
Run unlearn audit for output perturbation: config-driven, single partition.
Only computes and saves epsilon_emp_lower (no epsilon_lb_avg_v).
"""
import argparse
import csv
import json
import numpy as np
import os
import yaml

from data_persistence import save_data, load_data
from data_generation import generate_strategic_data, generate_test_val_sets
from training import train
from membership_inference import (
    run_membership_inference_attack_single_partition,
    save_attack_metrics_json,
    plot_roc_curve,
)


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def _scalar_for_json(x):
    if isinstance(x, (np.floating, float)) and not np.isfinite(x):
        return None
    if isinstance(x, (np.integer, np.int64, np.int32)):
        return int(x)
    if isinstance(x, (np.floating, np.float64, np.float32)):
        return float(x)
    return x


def save_epsilon_bounds_summary(results_summary, output_dir, delta=0.01):
    """Write epsilon vs epsilon bounds (and key metrics) as JSON and CSV."""
    rows = []
    for r in results_summary:
        row = {
            'epsilon': _scalar_for_json(r['epsilon']),
            'sigma': _scalar_for_json(r['sigma']),
            'accuracy': _scalar_for_json(r['accuracy']),
            'tpr': _scalar_for_json(r['tpr']),
            'fpr': _scalar_for_json(r['fpr']),
            'roc_auc': _scalar_for_json(r['roc_auc']),
            'epsilon_empirical_lower': _scalar_for_json(r['epsilon_empirical_lower']),
            'epsilon_lb_avg_v': _scalar_for_json(r.get('epsilon_lb_avg_v')),
            'tpr_low': _scalar_for_json(r.get('tpr_low')),
            'tpr_high': _scalar_for_json(r.get('tpr_high')),
            'fpr_low': _scalar_for_json(r.get('fpr_low')),
            'fpr_high': _scalar_for_json(r.get('fpr_high')),
        }
        rows.append(row)
    json_path = os.path.join(output_dir, 'epsilon_bounds_summary.json')
    with open(json_path, 'w') as f:
        json.dump({'delta': delta, 'rows': rows}, f, indent=2)
    print(f"Epsilon bounds summary (JSON) saved to: {json_path}")
    csv_path = os.path.join(output_dir, 'epsilon_bounds_summary.csv')
    if rows:
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys(), extrasaction='ignore')
            w.writeheader()
            w.writerows(rows)
        print(f"Epsilon bounds summary (CSV) saved to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='Run output perturbation unlearn audit from config')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config YAML')
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path) and not os.path.exists(config_path):
        alt = os.path.join(os.path.dirname(__file__) or '.', config_path)
        config_path = alt if os.path.exists(alt) else config_path
    config = load_config(config_path)
    config_id = config.get('config_id', 'output_perturbation_logistic')
    random_state = config.get('random_state', 42)
    loss = config.get('loss', 'logistic')
    data_cfg = config.get('data', {})
    training_cfg = config.get('training', {})
    sample_cfg = config.get('sample_config', {})
    privacy_cfg = config.get('privacy', {})
    op_cfg = config.get('output_perturbation', {})
    epsilon_values = config.get('epsilon_values', [0.1, 0.5, 1.0, 2.0, 5.0, 10.0])

    n_retain = data_cfg.get('n_retain', 900)
    n_forget = data_cfg.get('n_forget', 100)
    C_0 = op_cfg.get('C_0', 5.0)
    delta = privacy_cfg.get('delta', 0.01)

    OUTPUT_DIR = f'privacy_attack_results_config_{config_id}'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    data_path = os.path.join(OUTPUT_DIR, 'strategic_data.pkl')
    config_copy_path = os.path.join(OUTPUT_DIR, 'config.yaml')

    np.random.seed(random_state)

    print("="*70)
    print("OUTPUT PERTURBATION UNLEARN AUDIT (single partition, ε_emp^lower only)")
    print("="*70)
    print(f"Config: {config_path}")
    print(f"Loss: {loss}, Data: n_retain={n_retain}, n_forget={n_forget}, d={data_cfg.get('d', 2)}")
    print(f"C_0={C_0}, δ={delta}")
    print("="*70)

    print("\nGenerating data...")
    train_data = generate_strategic_data(
        n_retain=n_retain,
        n_forget=n_forget,
        d=data_cfg.get('d', 2),
        theta_norm=data_cfg.get('theta_norm', 1.0),
        forget_norm=data_cfg.get('forget_norm', 8.0),
        forget_direction=data_cfg.get('forget_direction', 'orthogonal'),
        link_function=data_cfg.get('link_function', 'f1'),
        eta=data_cfg.get('eta', 0.8),
        zeta=data_cfg.get('zeta', 0.9),
        feature_distribution=data_cfg.get('feature_distribution', 'gaussian'),
        loss=loss,
        range_half=data_cfg.get('range_half', 0.6),
        random_state=random_state,
    )
    eval_data = generate_test_val_sets(
        d=data_cfg.get('d', 2),
        theta_star=train_data['theta_star'],
        n_val=data_cfg.get('n_val', 200),
        n_test=data_cfg.get('n_test', 400),
        link_function=data_cfg.get('link_function', 'f1'),
        eta=data_cfg.get('eta', 0.8),
        zeta=data_cfg.get('zeta', 0.9),
        feature_distribution=data_cfg.get('feature_distribution', 'gaussian'),
        loss=loss,
        range_half=data_cfg.get('range_half', 0.6),
        random_state=random_state,
    )
    data = {**train_data, **eval_data}
    save_data(data, data_path)
    with open(config_copy_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"Saved data and config to {OUTPUT_DIR}/")

    per_sample_reg = training_cfg.get('per_sample_reg', 0.01)
    max_iter = training_cfg.get('max_iter', 1000)
    n_samples_per_dist = sample_cfg.get('n_samples_per_dist', 500)
    n_test_attack = sample_cfg.get('n_test', 500)
    confidence = sample_cfg.get('confidence', 0.95)

    print("\n" + "="*70)
    print("TRAINING MODELS")
    print("="*70)
    X_retain = data['X_retain']
    y_retain = data['y_retain']
    X_forget = data['X_forget']
    y_forget = data['y_forget']
    X_train_full = np.vstack([X_retain, X_forget])
    y_train_full = np.concatenate([y_retain, y_forget])

    trained_full = train(
        X_train_full, y_train_full,
        data['X_val'], data['y_val'],
        data['X_test'], data['y_test'],
        per_sample_reg=per_sample_reg, max_iter=max_iter,
        random_state=random_state, verbose=True,
        loss=loss,
    )
    trained_retain = train(
        X_retain, y_retain,
        data['X_val'], data['y_val'],
        data['X_test'], data['y_test'],
        per_sample_reg=per_sample_reg, max_iter=max_iter,
        random_state=random_state, verbose=True,
        loss=loss,
    )

    w_full = np.asarray(trained_full['weights'])
    w_retain = np.asarray(trained_retain['weights'])
    w_diff = w_full - w_retain
    weights_path = os.path.join(OUTPUT_DIR, 'trained_weights.json')
    with open(weights_path, 'w') as f:
        json.dump({
            'w_full': w_full.tolist(),
            'w_retain': w_retain.tolist(),
            'w_diff': w_diff.tolist(),
            'w_full_norm': float(np.linalg.norm(w_full)),
            'w_retain_norm': float(np.linalg.norm(w_retain)),
            'w_diff_norm': float(np.linalg.norm(w_diff)),
        }, f, indent=2)
    print(f"Trained weights saved to: {weights_path}")

    print("\n" + "="*70)
    print("RUNNING SINGLE-PARTITION AUDITS")
    print("="*70)
    results_summary = []
    for eps in epsilon_values:
        print(f"\n{'#'*70}\n### EPSILON = {eps} ###\n{'#'*70}")
        eps_folder = os.path.join(OUTPUT_DIR, f'epsilon_{eps}')
        os.makedirs(eps_folder, exist_ok=True)
        attack_results = run_membership_inference_attack_single_partition(
            data=data,
            trained_retain=trained_retain,
            trained_full=trained_full,
            C_0=C_0,
            epsilon=eps,
            delta=delta,
            n_samples_per_dist=n_samples_per_dist,
            n_test=n_test_attack,
            random_state=random_state,
            confidence=confidence,
        )
        save_attack_metrics_json(attack_results, os.path.join(eps_folder, 'metrics.json'))
        plot_roc_curve(attack_results, os.path.join(eps_folder, 'roc_curve.png'))
        results_summary.append({
            'epsilon': eps,
            'sigma': attack_results['sigma'],
            'accuracy': attack_results['accuracy'],
            'tpr': attack_results['tpr'],
            'fpr': attack_results['fpr'],
            'roc_auc': attack_results['roc_auc'],
            'tpr_low': attack_results['tpr_low'],
            'tpr_high': attack_results['tpr_high'],
            'fpr_low': attack_results['fpr_low'],
            'fpr_high': attack_results['fpr_high'],
            'epsilon_empirical_lower': attack_results['epsilon_empirical_lower'],
            'epsilon_lb_avg_v': attack_results.get('epsilon_lb_avg_v', float('nan')),
        })

    print("\n" + "="*90)
    print("SUMMARY")
    print("="*90)
    print(f"\n{'ε':<8} {'σ':<10} {'Acc':<8} {'TPR':<8} {'FPR':<8} {'AUC':<8} {'ε_emp^lo':<12} {'ε_lb_avg_v':<12} {'Privacy?'}")
    print("-"*90)
    for res in results_summary:
        privacy_status = "✓ Safe" if res['roc_auc'] < 0.6 else ("⚠ Leak" if res['roc_auc'] < 0.8 else "❌ Broken")
        eps_lo = res['epsilon_empirical_lower']
        eps_lo_str = f"{eps_lo:.4f}" if isinstance(eps_lo, float) and np.isfinite(eps_lo) else "—"
        avg_v = res.get('epsilon_lb_avg_v', float('nan'))
        avg_v_str = f"{avg_v:.4f}" if isinstance(avg_v, float) and np.isfinite(avg_v) else "—"
        print(f"{res['epsilon']:<8.1f} {res['sigma']:<10.4f} {res['accuracy']:<8.4f} {res['tpr']:<8.4f} {res['fpr']:<8.4f} {res['roc_auc']:<8.4f} {eps_lo_str:<12} {avg_v_str:<12} {privacy_status}")
    print("="*90)
    save_epsilon_bounds_summary(results_summary, OUTPUT_DIR, delta=delta)
    return results_summary


if __name__ == "__main__":
    main()
