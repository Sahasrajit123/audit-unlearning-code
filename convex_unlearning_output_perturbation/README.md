# Convex Unlearning & Output Perturbation: Empirical Privacy Audits

Code for auditing **(ε, δ)-certified machine unlearning** for convex losses. We run
membership-inference-style attacks on the *released* (noisy) model parameters and turn
the attack's TPR/FPR into **empirical lower bounds** on the privacy parameter. The
bounds come in two forms, both comparable with the theoretical guarantee:

- **(ε, δ)-DP:** lower bounds on ε.
- **ρ-zCDP:** lower bounds on ρ, compared with the upper bound ρ_UB = Δ²/(2σ²) that the
  injected Gaussian noise actually provides.

The repo covers two unlearning mechanisms, and they share the same training,
Lipschitz, and bound-computation code:

| Mechanism | What is released | Runner |
|---|---|---|
| **Newton-step unlearning** (Algorithm 1) | `w̄ = ŵ + H⁻¹ Σ_forget ∇f` (one Newton step removing the forget set) `+ N(0, σ²I)` | `run_privacy_attack_final.py` |
| **Output perturbation** (Equation 1) | `Π_{C₀}(ŵ) + N(0, σ²I)` (clip to an L2 ball, then add Gaussian noise) | `run_output_perturbation_attack.py` |

---

## Setup

The code needs Python ≥ 3.10 and five packages: numpy, scipy, scikit-learn,
matplotlib and PyYAML. It has been tested on Python 3.11. Choose one of these setups:

```bash
# (a) Existing env on this machine; it already has everything, including pytest
conda activate torch_jax_gpu

# (b) Fresh conda env
conda env create -f environment.yml
conda activate convex-unlearning-audit

# (c) Any Python ≥ 3.10 with pip
pip install -r requirements-dev.txt   # runtime deps + pytest; use requirements.txt for runtime only
```

Optionally, run `pip install -e .` to make the modules importable from any directory,
for example from notebooks. The runners work without it as long as you run them from
the repo root.

Check the setup:

```bash
make test         # 12 tests, a few seconds, including tiny end-to-end runs of both runners
make test-fast    # unit tests only
```

`make help` lists all targets. `make run-newton`, `make run-op` and `make run-all` run
every config and save a `log_*.txt` for each.

VS Code: `.vscode/settings.json` points the interpreter at `torch_jax_gpu` and enables
pytest discovery for `tests/`.

## Quick start

```bash
# Newton-step unlearning audit (logistic / MSE / cubic; 1, 2, or K partitions)
python run_privacy_attack_final.py --config config/convex_unlearning/default.yaml

# Output-perturbation audit (logistic or MSE, single partition)
python run_output_perturbation_attack.py --config config/output_perturbation/config.yaml
```

With no `--config`, each runner uses the default config shown above. A relative config
path that doesn't exist from the current directory is also looked up relative to the
script, so you can run from anywhere.

## Running every experiment

Each command below writes to its own output directory, which is named from the
config's `n_partition` and `config_id`. Every run sweeps all ε in the config's
`epsilon_values`.

### Newton-step unlearning (`run_privacy_attack_final.py`)

Output: `eps_lower_bounds/partition_{n_partition}_config_{config_id}/`

```bash
# ── Logistic regression ────────────────────────────────────────────────────
# d=20, n=45k, f1 link, two-partition (f1 vs f2)
python run_privacy_attack_final.py --config config/convex_unlearning/default.yaml
# d=20, f1 link, single-partition (retain-only vs unlearn)
python run_privacy_attack_final.py --config config/convex_unlearning/config_01_f1.yaml
# d=20, linear link, two-partition
python run_privacy_attack_final.py --config config/convex_unlearning/config_01_linear_link.yaml
# d=1, f1 link, two-partition
python run_privacy_attack_final.py --config config/convex_unlearning/config_02_f1_link.yaml
# d=2, linear link, two-partition
python run_privacy_attack_final.py --config config/convex_unlearning/config_02_linear_link.yaml
# d=1, f1 link, two-partition (best logistic setting)
python run_privacy_attack_final.py --config config/convex_unlearning/config_logistic_f1_link_best.yaml

# ── MSE (Ridge) ────────────────────────────────────────────────────────────
# d=2, tanh link, two-partition
python run_privacy_attack_final.py --config config/convex_unlearning/config_03.yaml

# ── Cubic loss ─────────────────────────────────────────────────────────────
# d=5, two-partition baseline
python run_privacy_attack_final.py --config config/convex_unlearning/cubic.yaml
# d=6, two-partition, worst-case "tighten" recipe
python run_privacy_attack_final.py --config config/convex_unlearning/cubic_tighten.yaml
# d=8, two-partition
python run_privacy_attack_final.py --config config/convex_unlearning/cubic_tighten_dual_partition.yaml
# d=8, single-partition
python run_privacy_attack_final.py --config config/convex_unlearning/cubic_tighten_single_partition.yaml
# d=1, single-partition (best cubic setting)
python run_privacy_attack_final.py --config config/convex_unlearning/cubic_tighten_single_partition_best.yaml
# d=8, single-partition, forget set spread over two coordinates
python run_privacy_attack_final.py --config config/convex_unlearning/cubic_tighten_single_partition_two_coordinates.yaml
# d=8, K=6 partitions → C(6,3) = 20 hypotheses
python run_privacy_attack_final.py --config config/convex_unlearning/cubic_tighten_multipartition.yaml
# d=8, single-partition, forget along e1 only (archived variant)
python run_privacy_attack_final.py --config config/convex_unlearning/archive_cubic_tighten_single_partition_retain_e2.yaml
```

### Output perturbation (`run_output_perturbation_attack.py`)

Output: `privacy_attack_results_config_{config_id}/`

```bash
# Logistic regression, d=2, n=1000 (400 forget), C₀=1
python run_output_perturbation_attack.py --config config/output_perturbation/config.yaml
# MSE (Ridge), large separation between full and retain models
python run_output_perturbation_attack.py --config config/output_perturbation/config_mse.yaml
```

### Run everything in sequence

```bash
for c in config/convex_unlearning/*.yaml; do
  python run_privacy_attack_final.py --config "$c" 2>&1 | tee "log_$(basename "$c" .yaml).txt"
done
for c in config/output_perturbation/*.yaml; do
  python run_output_perturbation_attack.py --config "$c" 2>&1 | tee "log_op_$(basename "$c" .yaml).txt"
done
```

These runs don't depend on each other, so you can also start them in parallel, for
example one per `tmux` pane or with `&`.

### Adding your own experiment

Copy the closest config, give it a new `config_id` so its output doesn't overwrite an
existing run, and pass it with `--config`.

---

## Repository layout

```
.
├── README.md
├── pyproject.toml           # package metadata, dependencies, pytest config
├── requirements.txt         # runtime dependencies
├── requirements-dev.txt     # + pytest
├── environment.yml          # conda env (Python 3.11)
├── Makefile                 # install / test / run-all shortcuts
├── tests/
│   ├── test_bounds.py       # unit tests: σ calibration, CP intervals, ρ_LB closed form, ρ→ε
│   └── test_pipelines.py    # end-to-end runs of both runners on shrunken configs (marked slow)
│
│   ── Entry points ──────────────────────────────────────────────────────────
├── run_privacy_attack_final.py       # Newton-step unlearning audit (1 / 2 / K partitions)
├── run_output_perturbation_attack.py # Output-perturbation audit (single partition)
│
│   ── Shared by both pipelines ──────────────────────────────────────────────
├── training.py              # train(): L2-regularized logistic / Ridge (MSE) / cubic
├── lipschitz_constants.py   # per-sample L (gradient) and M (Hessian) Lipschitz constants
├── data_persistence.py      # save_data / load_data (pickle)
├── cum_runs_eps_lab.py      # avg-v hypothesis test combinatorics → ε_lb (general audit, ε halved)
├── avg_v_convex.py          # convex avg-v tests → ε_lb (no halving) and ρ_lb (ε_γ = ργ)
├── zcdp_pairwise.py         # pairwise zCDP auditor: one-sided CP endpoints → ρ_LB; ρ→ε conversion; ρ_UB
│
│   ── Newton-step unlearning (convex_unlearning) ────────────────────────────
├── unlearning.py            # Algorithm 1: Newton-step removal + noise; gradients/Hessians
├── gaussian_mechanism.py    # σ for (ε, δ) via exact Gaussian mechanism (valid for all ε)
├── cubic_loss.py            # cubic loss f(w,z) = (λ/2)||w||² + (M/6)Σw_i³ − <z,w>
├── data_generation.py       # strategic forget sets (orthogonal θ^{f1}, θ^{f2} = −θ^{f1}), multipartition
├── membership_inference.py  # LLR attack (1 / 2 / K partitions), zCDP bounds, Lemma 3 check, plots
│
│   ── Output perturbation (op_ prefix = output-perturbation-specific) ──────
├── output_perturbation.py      # clip_to_ball, σ calibration, sample_unlearned_models
├── op_data_generation.py       # generate_strategic_data: forget set pulls full model away from retain
├── op_membership_inference.py  # single-partition LLR attack for retain vs. unlearn, zCDP bounds
│
└── config/
    ├── convex_unlearning/      # configs for run_privacy_attack_final.py
    └── output_perturbation/    # configs for run_output_perturbation_attack.py
```

### Naming convention

- **`op_*.py`**: used only by the output-perturbation pipeline
  (`run_output_perturbation_attack.py`).
- **Unprefixed files**: either shared by both pipelines or specific to Newton-step
  unlearning, as grouped in the layout above.

The data generation and attack code for each mechanism are separate modules because the
mechanisms need different things. Output perturbation builds its forget set along −θ*,
so that the full-data and retain-only models end up far apart, and it compares
retain-only against unlearned outputs. Newton-step unlearning builds forget sets
orthogonal to θ* and supports 1, 2, or K partitions.

---

## Pipelines in detail

### 1. Newton-step unlearning: `run_privacy_attack_final.py`

The `n_partition` value in the config sets the attack:

- **`n_partition: 1`**: *retain-only vs. unlearn*. Model A is trained on the retain
  set and noised. Model B is trained on retain ∪ forget, then the forget set is removed
  with a Newton step and the result is noised. The attack guesses which of the two
  produced an observed parameter vector.
- **`n_partition: 2`**: *which forget set?* The forget set is split into halves
  generated from θ^{f1} and θ^{f2} = −θ^{f1}, both orthogonal to θ*. The attack guesses
  which half was unlearned. Works for logistic, MSE, and cubic loss.
- **`n_partition: K`** (even, cubic): the forget set is exactly K/2 of K partitions,
  which gives C(K, K/2) hypotheses. Uses the avg-v test.

Per ε, the pipeline:
1. Computes the pre-noise `w̄` for each hypothesis (`membership_inference.compute_w_bar_before_noise`).
2. Calibrates σ from ε, δ, and the Lipschitz constants L and M (`gaussian_mechanism`, `unlearning`).
3. Draws `n_samples_per_dist` noisy models per hypothesis, fits Gaussians, and classifies
   `n_test` fresh draws by log-likelihood ratio.
4. Reports TPR/FPR with Clopper–Pearson intervals and computes the privacy lower
   bounds described in [Privacy bounds reported](#privacy-bounds-reported).
5. Checks the paper's Lemma 3 bound numerically (`lemma_verification.json`).

**Output:** `eps_lower_bounds/partition_{n_partition}_config_{config_id}/`
```
config.yaml, strategic_data.pkl, lemma_verification.json
epsilon_bounds_summary.{json,csv}          # one row per ε
epsilon_{ε}/metrics.json, attack_results.pkl, distributions.{pkl,_summary.json},
            roc_curve.png, llr_distributions.png
            multipartition_predictions.{json,txt}   # K-partition only
```

### 2. Output perturbation: `run_output_perturbation_attack.py`

This is always single-partition. The two hypotheses are:
- **Retain**: `Π_{C₀}(w_retain) + ξ`
- **Unlearn**: `Π_{C₀}(w_full) + ξ`, with ξ ~ N(0, σ²I).

After clipping, any two outputs are at most Δ = 2·C₀ apart, so σ is calibrated for
sensitivity 2·C₀.

The data is built so that `w_full` and `w_retain` are far apart (the forget set is
drawn along −θ*, scaled by `forget_norm`). This puts the audit near the worst case.

**Output:** `privacy_attack_results_config_{config_id}/`
```
config.yaml, strategic_data.pkl, trained_weights.json
epsilon_bounds_summary.{json,csv}
epsilon_{ε}/metrics.json, roc_curve.png
```

---

## Privacy bounds reported

Every bound starts from the attack's confusion matrix on `n_test` fresh draws per
hypothesis. Which bounds are computed depends on the pipeline and partition mode.

| Quantity | Meaning | Newton-step 1-part. | Newton-step 2-part. | Newton-step K-part. | Output pert. |
|---|---|:-:|:-:|:-:|:-:|
| `epsilon_empirical_lower` | ε_emp^lower from two-sided Clopper–Pearson TPR/FPR bounds | ✓ | ✓ | – | ✓ |
| `epsilon_lb_avg_v` | avg-v test ε_lb, general audit (halved) | ✓ | ✓ | ✓ | – |
| `epsilon_lb_avg_v` (OP) / `epsilon_lb_avg_v_convex` | avg-v test ε_lb, convex (not halved) | ✓ | – | – | ✓ |
| `rho_lb_pairwise` (Newton) / `rho_lb_conv` (OP) | pairwise zCDP lower bound ρ_LB | ✓ direct | ✓ reference | – | ✓ direct |
| `rho_lb_avg_v` | convex avg-v ρ lower bound | ✓ | – | – | ✓ |
| `rho_ub_zcdp` (Newton) / `rho_ub_noise` (OP) | ρ_UB = Δ²/(2σ²) of the noise actually added | ✓ | ✓ | – | ✓ |
| `eps_from_rho_*` | ε implied by a ρ value (zCDP → DP conversion) | ✓ | ✓ | – | ✓ |

**ε_emp^lower.** From two-sided Clopper–Pearson bounds at confidence `confidence`:

    ε_emp^lower = max{ log((1−δ−FP^high)/FN^high), log((1−δ−FN^high)/FP^high) }

In Newton-step two-partition mode, two unlearned outputs are compared through a
common reference. So there the formula is evaluated with δ = 1e−8 and the result is
halved, for the same reason as the general avg-v test.

**Avg-v ε test, general vs. convex.** Both use the combinatorics in
`cum_runs_eps_lab.py`. The general version (`compute_avg_v_test_epsilon_lb`) halves the
bound: it compares two unlearned outputs through a common reference, and
(ε, 0)-unlearning only implies the audit mechanism is (2ε, 0)-LDP. The convex version
(`avg_v_convex.compute_avg_v_test_epsilon_lb_convex`) compares the two distributions
the auditor actually observes, so the bound is not halved.

**Pairwise zCDP lower bound ρ_LB** (`zcdp_pairwise.py`). This uses one-sided
Clopper–Pearson endpoints TPR^low, FPR^high, TNR^low, FNR^high, with the `ci_delta`
budget split across the classes. The Rényi change-of-measure inequality gives, for
every γ > 1, a lower bound on ρ. The bound depends on which Rényi curve applies:

- **Direct** (ε_γ = ργ). Used when the guarantee compares the two observed laws
  directly: output perturbation, and Newton-step single-partition. The sup over γ has a
  closed form,
  `ρ_LB = max{0, (√(−log FPR^high) − √(−log TPR^low))², and the TNR/FNR mirror}`,
  attained at `γ* = 1/(1 − √(log TPR^low / log FPR^high))`.
- **Reference** (ε_γ = 2ργ(1 + √(γ/(γ−1)))). Used for Newton-step two-partition,
  where two unlearned laws are compared through a common reference, which costs a
  weak-triangle (transitivity) loss. The sup over γ is found numerically.

**Convex avg-v ρ lower bound** (`avg_v_convex.compute_avg_v_test_rho_lb_convex`).
This aggregates over all T test draws with the direct curve ε_γ = ργ. It has a closed
form, `ρ_lb = (√Q − √C)² / L` with Q = −log M_bound and C = −log(ci_delta), valid when
Q > C.

**ρ_UB.** The Gaussian mechanism that was actually run satisfies ρ-zCDP with
ρ = Δ²/(2σ²). For output perturbation Δ = 2·C₀. **ρ_LB ≤ ρ_UB is the valid check:** a
ρ_LB above ρ_UB means the implementation leaks more than the noise allows.

**ε(ρ).** `zcdp_pairwise.eps_estimate_from_rho` converts ρ to an (ε, δ_conv) estimate.
It takes the minimum of two conversions: the tight Balle et al. (2020) conversion,
optimized over the Rényi order, and the classic Bun & Steinke (2016) bound
ρ + 2√(ρ log(1/δ_conv)). δ_conv defaults to the audit δ, so ε(ρ) sits on the same axis
as the nominal ε. It is an *implied estimate*, not a certified bound on ε.

---

## Configs

All experiment settings live in YAML files. The common keys are:

```yaml
config_id: default            # used in the output directory name
random_state: 42
loss: logistic                # logistic | mse | cubic (cubic: Newton-step only)
n_partition: 1                # 1, 2, or even K (Newton-step; default 2 if omitted); OP is always 1
data:        {n_retain, n_forget, d, theta_norm, forget_norm, link_function, eta, zeta, n_val, n_test, ...}
training:    {per_sample_reg, max_iter}
sample_config: {n_samples_per_dist, n_test, confidence, ci_delta}
privacy:     {delta}           # δ for σ calibration and ε_emp^lower
             # OP only, optional: conv_delta (δ for ρ→ε conversion, default = delta),
             #                    conv_method (tight | classic, default tight)
epsilon_values: [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
output_perturbation: {C_0}    # OP only: clipping radius / sensitivity
```

**`config/convex_unlearning/`**

| File | Loss | Partitions | Notes |
|---|---|---|---|
| `default.yaml` | logistic | 2 | d=20, n=45k baseline |
| `config_01_f1.yaml` | logistic | 1 | f1 (step) link |
| `config_01_linear_link.yaml` | logistic | 2 | d=20, linear link |
| `config_02_f1_link.yaml` | logistic | 2 | d=1, f1 link |
| `config_02_linear_link.yaml` | logistic | 2 | d=2, linear link |
| `config_logistic_f1_link_best.yaml` | logistic | 2 | d=1, best logistic setting |
| `config_03.yaml` | mse | 2 | Ridge regression, d=2, tanh link |
| `cubic.yaml` | cubic | 2 | e1/e2 cubic baseline |
| `cubic_tighten.yaml` | cubic | 2 | worst-case recipe to tighten the bound |
| `cubic_tighten_single_partition*.yaml` | cubic | 1 | retain-only vs. unlearn (`_best`, `_two_coordinates` variants) |
| `cubic_tighten_dual_partition.yaml` | cubic | 2 | |
| `cubic_tighten_multipartition.yaml` | cubic | 6 | C(6,3) = 20 hypotheses |
| `archive_*.yaml` | cubic | 1 | archived variant |

**`config/output_perturbation/`**

| File | Loss | Notes |
|---|---|---|
| `config.yaml` | logistic | d=2, n=1000 (400 forget), C₀=1, δ=0.001 |
| `config_mse.yaml` | mse | large separation between full and retain models, δ=0.001 |

Most Newton-step configs use δ = 0.01. `cubic_tighten_single_partition_best.yaml` and
both output-perturbation configs use δ = 0.001.

---

## Interpreting results

`epsilon_bounds_summary.csv` has one row per nominal ε. The base columns are `sigma`,
`accuracy`, `tpr`, `fpr`, `roc_auc`, `epsilon_empirical_lower`, `epsilon_lb_avg_v` and
the Clopper–Pearson bounds `tpr_low/high`, `fpr_low/high`. It also has the zCDP columns
from [Privacy bounds reported](#privacy-bounds-reported). The per-ε `metrics.json`
files hold the full detail, such as the one-sided endpoints, γ*, and which branch
attained the bound.

- A **negative or NaN** `epsilon_empirical_lower` means the attack found no
  statistically significant leakage.
- If ε_emp^lower > ε, the certified guarantee has been violated. The runners mark
  these rows as "Leak" or "Broken" in their console summary tables.
- For zCDP, compare `rho_lb_*` with `rho_ub_*`: ρ_LB > ρ_UB is a violation. Don't
  compare `eps_from_rho_lb` with ε as if it were a certified lower bound.
- An empty avg-v or ρ entry, sometimes with a printed warning such as
  `avg-v-test epsilon_lb failed: ... NoneType`, means the bound doesn't apply at that
  ε. This usually happens at small ε, where accuracy is near chance, and it is
  consistent with ε = 0 or ρ = 0.
