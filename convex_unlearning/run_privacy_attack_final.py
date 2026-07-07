"""
Unified privacy attack runner.

n_partition=1 : single forget set — retain-only vs retain∪forget→unlearn.
n_partition=2 : two forget sets — f1 vs f2 halves (logistic/MSE/cubic).
n_partition=K : K-partition cubic attack — C(K,K/2) configurations.

Usage:
    python run_privacy_attack_final.py --config <path-to-config.yaml>
"""
import argparse
import csv
import json
import numpy as np
import os
import yaml

from data_generation import (
    generate_strategic_data_logistic,
    generate_strategic_data_multipartition,
    generate_test_val_sets,
)
from data_persistence import save_data
from training import train
from membership_inference import (
    run_membership_inference_attack_two_partition,
    run_membership_inference_attack_single_partition,
    run_multi_partition_attack,
    save_attack_results,
    save_attack_metrics_json,
    save_distributions,
    save_distributions_json,
    plot_roc_curve,
    plot_llr_distributions,
    verify_lemma3_bound,
    save_lemma_verification,
)


def load_config(path):
    if not os.path.isabs(path) and not os.path.exists(path):
        alt = os.path.join(os.path.dirname(__file__) or '.', path)
        path = alt if os.path.exists(alt) else path
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


def save_epsilon_bounds_summary(results_summary, output_dir, delta, n_partition):
    """Write epsilon bounds summary as JSON and CSV. Schema depends on n_partition."""
    if n_partition <= 2:
        fields = ['epsilon', 'sigma', 'accuracy', 'tpr', 'fpr', 'roc_auc',
                  'epsilon_empirical_lower', 'epsilon_lb_avg_v',
                  'tpr_low', 'tpr_high', 'fpr_low', 'fpr_high']
        meta = {'delta': delta}
    else:
        fields = ['epsilon', 'epsilon_lb']
        meta = {'delta': delta, 'K': n_partition}

    rows = [{f: _scalar_for_json(r.get(f)) for f in fields} for r in results_summary]
    json_path = os.path.join(output_dir, 'epsilon_bounds_summary.json')
    with open(json_path, 'w') as f:
        json.dump({**meta, 'rows': rows}, f, indent=2)
    print(f"Epsilon bounds summary saved to: {json_path}")
    csv_path = os.path.join(output_dir, 'epsilon_bounds_summary.csv')
    if rows:
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            w.writeheader()
            w.writerows(rows)
        print(f"Epsilon bounds summary (CSV) saved to: {csv_path}")


def save_prediction_details(prediction_details, epsilon, output_dir, K):
    eps_dir = os.path.join(output_dir, f'epsilon_{epsilon}')
    os.makedirs(eps_dir, exist_ok=True)
    json_path = os.path.join(eps_dir, 'multipartition_predictions.json')
    with open(json_path, 'w') as f:
        json.dump({'epsilon': epsilon, 'K': K,
                   'n_details': len(prediction_details),
                   'samples': prediction_details}, f, indent=2)
    txt_path = os.path.join(eps_dir, 'multipartition_predictions.txt')
    with open(txt_path, 'w') as f:
        f.write(f"Multi-partition predictions (first {len(prediction_details)} samples)\n")
        f.write(f"epsilon = {epsilon}, K = {K}\n{'='*80}\n\n")
        for s in prediction_details:
            f.write(f"--- Sample {s['sample_idx']} ---\n")
            f.write(f"  True config:      {s['true_tuple']}\n")
            f.write(f"  Predicted config: {s['predicted_tuple']}\n")
            f.write(f"  Correct:          {s['correct']}\n")
            f.write(f"  Sigma:            {s['sigma']:.6e}\n")
            f.write(f"  w_bar: norm={s['w_bar_norm']:.6f}, first3={s['w_bar_first3']}\n")
            f.write(f"  obs:   norm={s['obs_norm']:.6f}, first3={s['obs_first3']}\n")
            f.write(f"  Overlap: {s['overlap']} of K/2={K//2},  v={s['v']}\n")
            f.write(f"  log p(obs|true): {s['logp_true']:.6f}\n")
            f.write(f"  log p(obs|pred): {s['logp_pred']:.6f}\n\n")
    print(f"  Prediction details saved to {eps_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/default.yaml')
    args = parser.parse_args()

    config = load_config(args.config)
    config_path = args.config

    config_id    = config.get('config_id', 'default')
    # accept both n_partition (old single/two-partition configs) and n_partitions (old multi-partition configs)
    n_partition  = config.get('n_partition', config.get('n_partitions', 2))
    random_state = config.get('random_state', 42)
    loss         = config.get('loss', 'logistic')
    cubic_cfg    = config.get('cubic', {})
    data_cfg     = config.get('data', {})
    training_cfg = config.get('training', {})
    sample_cfg   = config.get('sample_config', {})
    privacy_cfg  = config.get('privacy', {})
    epsilon_values = config.get('epsilon_values', [0.1, 0.5, 1.0, 2.0, 5.0, 10.0])

    OUTPUT_DIR = os.path.join('eps_lower_bounds', f'partition_{n_partition}_config_{config_id}')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.random.seed(random_state)

    print("="*70)
    print("PRIVACY ATTACK (UNIFIED)")
    print("="*70)
    print(f"Config:      {config_path}")
    print(f"Config ID:   {config_id}")
    print(f"Loss:        {loss}")
    print(f"n_partition: {n_partition}")
    print(f"Output:      {OUTPUT_DIR}/")
    print("="*70)

    # --- data generation ---
    d         = data_cfg.get('d', 20)
    n_retain  = data_cfg.get('n_retain', 1000)
    data_R    = cubic_cfg.get('data_R', 20.0)
    # n_forget_per_partition: prefer explicit key, fall back to n_forget for single-partition configs
    n_forget_per_partition = data_cfg.get('n_forget_per_partition', data_cfg.get('n_forget', 60))
    # Optional explicit point coordinates (absent → backward-compatible defaults)
    retain_coord = cubic_cfg.get('retain_coord', None)
    forget_coord = cubic_cfg.get('forget_coord', None)

    if loss == 'cubic':
        data = generate_strategic_data_multipartition(
            n_retain=n_retain,
            n_forget_per_partition=n_forget_per_partition,
            d=d,
            n_partitions=n_partition,
            data_R=data_R,
            retain_coord=retain_coord,
            forget_coord=forget_coord,
            random_state=random_state,
        )
        eval_data = generate_test_val_sets(
            d=d, theta_star=None,
            n_val=data_cfg.get('n_val', 200),
            n_test=data_cfg.get('n_test', 400),
            loss='cubic', cubic_params=cubic_cfg,
            random_state=random_state,
        )
    elif loss == "logistic":
        n_forget = data_cfg.get('n_forget', 4500)
        data = generate_strategic_data_logistic(
            n_retain=n_retain, n_forget=n_forget, d=d,
            n_partitions=n_partition,
            theta_norm=data_cfg.get('theta_norm', 1.0),
            forget_norm=data_cfg.get('forget_norm', 10.0),
            link_function=data_cfg.get('link_function', 'f1'),
            eta=data_cfg.get('eta', 0.8),
            zeta=data_cfg.get('zeta', 0.9),
            feature_distribution=data_cfg.get('feature_distribution', 'gaussian'),
            loss=loss,
            forget_point_boundary=data_cfg.get('forget_point_boundary', False),
            forget_boundary_label=data_cfg.get('forget_boundary_label', 1),
            random_state=random_state,
        )
        eval_data = generate_test_val_sets(
            d=d, theta_star=data['theta_star'],
            n_val=data_cfg.get('n_val', 5000),
            n_test=data_cfg.get('n_test', 10000),
            link_function=data_cfg.get('link_function', 'f1'),
            eta=data_cfg.get('eta', 0.8),
            zeta=data_cfg.get('zeta', 0.9),
            feature_distribution=data_cfg.get('feature_distribution', 'gaussian'),
            loss=loss,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unsupported loss type: {loss}")

    data.update(eval_data)
    save_data(data, os.path.join(OUTPUT_DIR, 'strategic_data.pkl'))
    with open(os.path.join(OUTPUT_DIR, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    per_sample_reg = training_cfg.get('per_sample_reg', 1e-5)
    max_iter       = training_cfg.get('max_iter', 1000)
    delta          = privacy_cfg.get('delta', 0.01)
    loss_params    = cubic_cfg if loss == 'cubic' else None

    n_samples_per_dist = sample_cfg.get('n_samples_per_dist', 1000)
    n_test_attack      = sample_cfg.get('n_test', 500)
    ci_delta           = sample_cfg.get('ci_delta', 0.05)
    confidence         = 1 - sample_cfg.get('ci_delta', 0.05)

    # ------------------------------------------------------------------ #
    #  K > 2 : multi-partition combinatorial attack                        #
    # ------------------------------------------------------------------ #
    if n_partition > 2:
        K = n_partition
        n_details = sample_cfg.get('n_prediction_details', 20)
        results_summary = []
        for eps in epsilon_values:
            print(f"\n{'#'*70}\n### EPSILON = {eps} ###\n{'#'*70}")
            eps_folder = os.path.join(OUTPUT_DIR, f'epsilon_{eps}')
            os.makedirs(eps_folder, exist_ok=True)
            attack_results = run_multi_partition_attack(
                data=data,
                epsilon=eps, delta=delta,
                n_samples_per_dist=n_samples_per_dist,
                n_test=n_test_attack,
                K=K, ci_delta=ci_delta,
                cubic_params=cubic_cfg,
                per_sample_reg=per_sample_reg,
                max_iter=max_iter,
                X_val=data['X_val'], y_val=data['y_val'],
                X_test=data['X_test'], y_test=data['y_test'],
                random_state=random_state, verbose=True, n_details=n_details,
            )
            results_summary.append({'epsilon': eps, 'epsilon_lb': attack_results['epsilon_lb']})
            with open(os.path.join(eps_folder, 'metrics.json'), 'w') as f:
                json.dump({'epsilon': _scalar_for_json(eps), 'delta': delta, 'K': K,
                           'epsilon_lb': _scalar_for_json(attack_results['epsilon_lb'])}, f, indent=2)
            if attack_results.get('prediction_details'):
                save_prediction_details(attack_results['prediction_details'], eps, OUTPUT_DIR, K)
        save_epsilon_bounds_summary(results_summary, OUTPUT_DIR, delta, n_partition)
        print(f"\n{'='*70}\nSUMMARY: ε_lb (m=K={K}, r=K)\n{'='*70}")
        print(f"{'ε':<12} {'ε_lb':<12}")
        print("-"*25)
        for r in results_summary:
            lb = r['epsilon_lb']
            lb_str = '—' if isinstance(lb, float) and np.isnan(lb) else f'{lb:.4f}'
            print(f"{r['epsilon']:<12.1f} {lb_str:<12}")
        return results_summary

    # ------------------------------------------------------------------ #
    #  n_partition = 1 : single forget set                                 #
    # ------------------------------------------------------------------ #
    if n_partition == 1:
        X_forget = data['X_forget']
        y_forget = data['y_forget']
        X_train_full = np.vstack([data['X_retain'], X_forget])
        y_train_full = np.concatenate([data['y_retain'], y_forget])
        n_retain_actual = len(data['X_retain'])

        trained_full = train(
            X_train=X_train_full, y_train=y_train_full,
            X_val=data['X_val'], y_val=data['y_val'],
            X_test=data['X_test'], y_test=data['y_test'],
            per_sample_reg=per_sample_reg, loss=loss,
            max_iter=max_iter, random_state=random_state,
            verbose=False, loss_params=loss_params,
        )
        trained_retain = train(
            X_train=data['X_retain'], y_train=data['y_retain'],
            X_val=data['X_val'], y_val=data['y_val'],
            X_test=data['X_test'], y_test=data['y_test'],
            per_sample_reg=trained_full['per_sample_reg'][:n_retain_actual],
            loss=loss, max_iter=max_iter, random_state=random_state,
            verbose=False, loss_params=loss_params,
        )
        lemma_result = verify_lemma3_bound(
            trained_result=trained_full,
            X_forget=X_forget, y_forget=y_forget,
            X_retain=data['X_retain'], y_retain=data['y_retain'],
            M=trained_full['M'], L=trained_full['L'],
            epsilon=1.0, delta=delta,
            w_retrain=trained_retain['weights'], verbose=True,
        )
        save_lemma_verification(lemma_result, None, len(X_train_full), len(X_forget),
                                os.path.join(OUTPUT_DIR, 'lemma_verification.json'),
                                single_partition=True)
        results_summary = []
        for eps in epsilon_values:
            print(f"\n{'#'*70}\n### EPSILON = {eps} ###\n{'#'*70}")
            eps_folder = os.path.join(OUTPUT_DIR, f'epsilon_{eps}')
            os.makedirs(eps_folder, exist_ok=True)
            attack_results = run_membership_inference_attack_single_partition(
                data=data, trained_retain=trained_retain, trained_full=trained_full,
                epsilon=eps, delta=delta,
                n_samples_per_dist=n_samples_per_dist, n_test=n_test_attack,
                random_state=random_state, confidence=confidence, ci_delta=ci_delta,
            )
            _save_two_partition_outputs(attack_results, eps, delta, eps_folder)
            results_summary.append(_two_partition_row(attack_results, eps))
        save_epsilon_bounds_summary(results_summary, OUTPUT_DIR, delta, n_partition)
        _print_two_partition_summary(results_summary, confidence, delta, ci_delta)
        return results_summary

    # ------------------------------------------------------------------ #
    #  n_partition = 2 : two forget sets (f1 vs f2)                        #
    # ------------------------------------------------------------------ #
    X_forget_1, X_forget_2 = data['X_partitions'][0], data['X_partitions'][1]
    y_forget_1, y_forget_2 = np.zeros(len(X_forget_1)), np.zeros(len(X_forget_2))

    X_train_f1 = np.vstack([data['X_retain'], X_forget_1])
    y_train_f1 = np.concatenate([data['y_retain'], y_forget_1])
    X_train_f2 = np.vstack([data['X_retain'], X_forget_2])
    y_train_f2 = np.concatenate([data['y_retain'], y_forget_2])
    n_retain_actual = len(data['X_retain'])

    trained_f1 = train(
        X_train=X_train_f1, y_train=y_train_f1,
        X_val=data['X_val'], y_val=data['y_val'],
        X_test=data['X_test'], y_test=data['y_test'],
        per_sample_reg=per_sample_reg, loss=loss,
        max_iter=max_iter, random_state=random_state,
        verbose=False, loss_params=loss_params,
    )
    trained_f2 = train(
        X_train=X_train_f2, y_train=y_train_f2,
        X_val=data['X_val'], y_val=data['y_val'],
        X_test=data['X_test'], y_test=data['y_test'],
        per_sample_reg=per_sample_reg, loss=loss,
        max_iter=max_iter, random_state=random_state,
        verbose=False, loss_params=loss_params,
    )
    trained_retain = train(
        X_train=data['X_retain'], y_train=data['y_retain'],
        X_val=data['X_val'], y_val=data['y_val'],
        X_test=data['X_test'], y_test=data['y_test'],
        per_sample_reg=trained_f1['per_sample_reg'][:n_retain_actual],
        loss=loss, max_iter=max_iter, random_state=random_state,
        verbose=False, loss_params=loss_params,
    )

    lemma_f1 = verify_lemma3_bound(
        trained_result=trained_f1,
        X_forget=X_forget_1, y_forget=y_forget_1,
        X_retain=data['X_retain'], y_retain=data['y_retain'],
        M=trained_f1['M'], L=trained_f1['L'],
        epsilon=1.0, delta=delta, w_retrain=trained_retain['weights'], verbose=True,
    )
    lemma_f2 = verify_lemma3_bound(
        trained_result=trained_f2,
        X_forget=X_forget_2, y_forget=y_forget_2,
        X_retain=data['X_retain'], y_retain=data['y_retain'],
        M=trained_f2['M'], L=trained_f2['L'],
        epsilon=1.0, delta=delta, w_retrain=trained_retain['weights'], verbose=True,
    )
    save_lemma_verification(
        lemma_f1, lemma_f2,
        len(X_train_f1), len(X_forget_1),
        os.path.join(OUTPUT_DIR, 'lemma_verification.json'),
    )

    results_summary = []
    for eps in epsilon_values:
        print(f"\n{'#'*70}\n### EPSILON = {eps} ###\n{'#'*70}")
        eps_folder = os.path.join(OUTPUT_DIR, f'epsilon_{eps}')
        os.makedirs(eps_folder, exist_ok=True)
        attack_results = run_membership_inference_attack_two_partition(
            data=data, trained_f1=trained_f1, trained_f2=trained_f2,
            epsilon=eps, delta=delta,
            n_samples_per_dist=n_samples_per_dist, n_test=n_test_attack,
            random_state=random_state, confidence=confidence, ci_delta=ci_delta,
        )
        _save_two_partition_outputs(attack_results, eps, delta, eps_folder)
        results_summary.append(_two_partition_row(attack_results, eps))

    save_epsilon_bounds_summary(results_summary, OUTPUT_DIR, delta, n_partition)
    _print_two_partition_summary(results_summary, confidence, delta, ci_delta)
    return results_summary


# --- shared helpers for n_partition = 1 and 2 output ---

def _save_two_partition_outputs(attack_results, eps, delta, eps_folder):
    save_attack_results(attack_results, os.path.join(eps_folder, 'attack_results.pkl'))
    save_attack_metrics_json(attack_results, os.path.join(eps_folder, 'metrics.json'))
    save_distributions(
        dist_f1=attack_results['dist_f1'], dist_f2=attack_results['dist_f2'],
        w_bar_f1=attack_results['result_f1']['w_bar'],
        w_bar_f2=attack_results['result_f2']['w_bar'],
        sigma=attack_results['result_f1']['sigma'],
        epsilon=eps, delta=delta,
        filepath=os.path.join(eps_folder, 'distributions.pkl'),
    )
    save_distributions_json(
        dist_f1=attack_results['dist_f1'], dist_f2=attack_results['dist_f2'],
        w_bar_f1=attack_results['result_f1']['w_bar'],
        w_bar_f2=attack_results['result_f2']['w_bar'],
        sigma=attack_results['result_f1']['sigma'],
        epsilon=eps, delta=delta,
        filepath=os.path.join(eps_folder, 'distributions_summary.json'),
    )
    plot_roc_curve(attack_results, os.path.join(eps_folder, 'roc_curve.png'))
    plot_llr_distributions(attack_results, os.path.join(eps_folder, 'llr_distributions.png'))


def _two_partition_row(attack_results, eps):
    return {
        'epsilon': eps,
        'accuracy': attack_results['accuracy'],
        'tpr': attack_results['tpr'],
        'fpr': attack_results['fpr'],
        'roc_auc': attack_results['roc_auc'],
        'sigma': attack_results['result_f1']['sigma'],
        'tpr_low': attack_results['tpr_low'],
        'tpr_high': attack_results['tpr_high'],
        'fpr_low': attack_results['fpr_low'],
        'fpr_high': attack_results['fpr_high'],
        'epsilon_empirical_lower': attack_results['epsilon_empirical_lower'],
        'epsilon_lb_avg_v': attack_results['epsilon_lb_avg_v'],
    }


def _print_two_partition_summary(results_summary, confidence, delta, ci_delta):
    def _fmt(v):
        return f"{v:.4f}" if not (isinstance(v, float) and np.isnan(v)) else "—"

    print("\n" + "="*70)
    print("SUMMARY: ATTACK SUCCESS ACROSS EPSILON VALUES")
    print("="*70)
    print(f"CP confidence: {confidence:.0%}; ε_emp uses δ={delta}; ε_lb avg-v ci_delta={ci_delta}")
    hdr = f"{'ε':<8} {'σ':<10} {'Acc':<8} {'TPR':<8} {'FPR':<8} {'AUC':<8} {'ε_emp^lo':<10} {'ε_lb_avg_v':<10}"
    print(f"\n{hdr}")
    print("-" * len(hdr))
    for r in results_summary:
        status = "✓ Safe" if r['roc_auc'] < 0.6 else ("⚠ Leak" if r['roc_auc'] < 0.8 else "❌ Broken")
        print(f"{r['epsilon']:<8.1f} {r['sigma']:<10.4f} {r['accuracy']:<8.4f} "
              f"{r['tpr']:<8.4f} {r['fpr']:<8.4f} {r['roc_auc']:<8.4f} "
              f"{_fmt(r['epsilon_empirical_lower']):<10} {_fmt(r['epsilon_lb_avg_v']):<10} {status}")
    print("=" * len(hdr))


if __name__ == "__main__":
    main()
