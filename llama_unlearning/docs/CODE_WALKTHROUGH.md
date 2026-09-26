# Code walkthrough

A reading guide to every file, in the order that makes the design legible. Companion to
`README.md` (how to run it), `docs/IMPLEMENTATION.md` (why each decision was made), and
and `docs/RESOURCE_ESTIMATE.md` (measured cost and pilot findings).

**9,719 lines of implementation** (5,461 in `audit_tofu/` + 4,258 in `scripts/`, excluding
the 995-line vendored epsilon module) **and 4,285 lines of tests** — 259 test functions,
307 collected after parametrization.

---

## Data flow

```
build_manifest.py ──> manifest.json          (immutable: authors, batches, sign vectors, seeds, hash)
                          │
train_reference.py ───────┤   ONCE, shared by every config with the same D_r
  └─ reference/truth_ratios.json    retain-only model; the forget-quality baseline
                          │
run_single.py  ───────────┤   per run_id, independently launchable
  │                       │
  ├─ tofu_data ──> D(S) = D_r ∪ D_f(S), in the fixed global order
  ├─ train ─────> fine-tuned checkpoint            (ONCE per run)
  ├─ unlearn ───> noop | npo | retain_ft | grad_ascent | grad_diff | simnpo
  ├─ scoring ───> losses.json: (run_id, method, author, qa_id) -> ℓ_z
  ├─ separation.json                               (in/out gap, Cohen's d)
  ├─ methods/<m>/model/                            (unlearned weights, if keep_all)
  └─ utility ───> utility.json                     (evaluation runs only)
                     + forget_quality_slices, if the reference exists
                          │
aggregate_audit.py ───────┤
  ├─ attack.fit_calibration  <- CALIBRATION runs only
  ├─ attack.predict          <- evaluation losses; NO labels in scope
  ├─ attack.overlap          <- labels revealed here, after predictions are final
  ├─ epsilon_bounds          <- vendored Lemma 4.2, then /2 per Lemma 4.1
  └─ rho_mu_bounds           <- same {V}: zCDP rho + GDP mu, NOT halved
                          │
                     audit_summary.json + 4 plots
                          │
backfill_dp_bounds.py ────┤   rho/mu onto audits finished before they existed;
  └─ audit/<m>/dp_bounds.json  recomputed from the stored v_list, nothing re-run
                          │
forget_quality.py ────────┤   CPU only, from stored per-row truth ratios
  └─ forget_quality.csv       plus / minus / all KS per group
                          │
collect_results.py ───────┘
  └─ collected/{runs,losses,truth_ratios,manifest_flat}.csv
```

---

## Run book: how to execute an audit end to end

### One command

```bash
scripts/launch_campaign.sh 2,5,7 2        # <gpu_list> <runs_per_gpu>
DRY_RUN=1 scripts/launch_campaign.sh 2,5,7 2     # plan only, nothing runs
```

That orchestrates all four stages below. Every stage is idempotent, so interrupting it
and re-running resumes rather than restarts. Knobs: `CONFIGS`, `SKIP_REFERENCE`,
`SKIP_AUDITS`, `SKIP_AGGREGATE`, `EXTRA_SET`, `STATUS_INTERVAL`.

### Or the four stages by hand

```bash
# 1. manifest -- once per config, then immutable
.venv/bin/python scripts/build_manifest.py --config configs/base.yaml

# 2. reference model -- ONCE for all configs sharing D_r (~20 min)
.venv/bin/python scripts/train_reference.py --config configs/base.yaml --gpu 5

# 3. the runs -- 30 runs x 6 methods, resumable, own GPU scheduler
scripts/run_audit.sh configs/base.yaml 2,5,7 2
#    FAMILIES=evaluation scripts/run_audit.sh ...   skips the calibration family

# 4. aggregate
.venv/bin/python scripts/collect_results.py --config configs/base.yaml
.venv/bin/python scripts/forget_quality.py  --config configs/base.yaml
for m in noop npo retain_ft grad_ascent grad_diff simnpo; do
  .venv/bin/python scripts/aggregate_audit.py --config configs/base.yaml --method $m
done
```

### Stage 2 goes before stage 3, and here is the only reason

The reference must exist when a run's utility suite executes, or that run's
`utility.json` records `forget_quality: null`. It is **not** a correctness issue —
`scripts/forget_quality.py` recovers identical numbers afterwards, because both paths
call the same `audit_tofu.utility.forget_quality_slices`. Reference-first simply buys
the metric in all 120 per-run files for ~20 minutes of GPU. If you have already run the
audits, just run `forget_quality.py`; nothing is lost.

### Re-auditing over a different r grid, keeping the old audit

```bash
DRY_RUN=1 scripts/reaudit_r_grid.sh              # print the plan
scripts/reaudit_r_grid.sh                        # all four configs x six methods
CONFIGS="configs/base.yaml" METHODS="noop" scripts/reaudit_r_grid.sh
```

`r` is a post-processing knob: training and scoring do not depend on it, and `losses.json`
is the only model output the audit consumes. So widening `attack.r_values` costs a CPU
re-aggregation (~3 s per method at `m = 20`), no GPU. Output goes to
`<output_root>/audit_r_grid/<method>/`, so the original `audit/<method>/` tree is never
read or written; because each new grid is a strict superset of the old one, every
previously reported `r` reappears with the same value.

Two things to know. **`r` must be even** — the auditor guesses `r/2` per side and
Lemma 4.2's `f(v)` is built from `C(r/2,a1)·C(r/2,a2)`, so `epsilon_bounds._validate`
raises and `aggregate_audit.py` does not guard the call: one odd entry aborts the whole
aggregation. `tests/test_config_and_runs.py` now checks every config for this. And a
denser grid can make the calibration-only leave-one-out **freeze a different `r`** than
before (measured: `npo` at `m = 100` moved from `r = 60` to `r = 70`). That is legitimate —
selection still never sees the evaluation runs — but it means the two trees' headline `r`
can differ, which is the other reason to keep both.

### Adding rho and mu to audits that already finished

```bash
DRY_RUN=1 scripts/backfill_dp_bounds.sh          # look first
scripts/backfill_dp_bounds.sh                    # writes <audit>/dp_bounds.json + the CSV
IN_PLACE=1 scripts/backfill_dp_bounds.sh         # also folds the fields into audit_summary.json
scripts/backfill_dp_bounds.sh /other/output/root  # extra roots
```

No run, attack or GPU: the overlap vector `V` in each `audit_summary.json` is the audit's
sufficient statistic, so all three bounds are closed-form in `(m, r, {V}, ζ)`. Epsilon is
recomputed purely as a cross-check and must match the stored value (`epsilon_reproduced`).
The script also lists output trees that hold runs but no `audit_summary.json` — those have
no stored `V`, so they need `aggregate_audit.py`, which now emits ρ and μ natively.

### Adding a metric afterwards, without re-training

`retention: keep_all` + `save_unlearned_checkpoint: true` keep the post-unlearning model
for every `(run, method)`, so a new metric costs a forward pass:

```bash
.venv/bin/python scripts/backfill_eval.py --config configs/base.yaml --dry_run
.venv/bin/python scripts/backfill_eval.py --config configs/base.yaml --what utility \
    --gpu 5 --set utility.compute_rouge=true
.venv/bin/python scripts/backfill_eval.py --config configs/base.yaml --what verify --gpu 5
```

`verify` re-scores the candidate pool from the saved weights and compares against
`losses.json` — agreement to ~1e-4 proves the checkpoint is the model that was scored
(measured: `max|d| = 0.00e+00`). What backfill **cannot** do is help when the *split*
changes: a different candidate partition means different training data, so those runs
have to happen for real.

### Measured costs

| | per unit | 30-run audit |
| --- | --- | --- |
| fine-tune | 19 min | — |
| unlearn + score (per method) | 26 s – 433 s | — |
| utility suite, no ROUGE | 116 s | 2.0 GPU-h |
| utility suite, with ROUGE | 33 min | 33 GPU-h |
| calibration run (6 methods) | 34 min | — |
| evaluation run (6 methods) | 46 min | — |
| **whole audit** | | **~19 GPU-h** |
| peak GPU memory | 15.4 GiB reserved | |
| disk at `keep_all` | ~17 GiB/run | ~500 GiB |

ROUGE is 16× the rest of the utility suite because it is autoregressive generation at
batch size 1 (~130 tokens/row against a 36-token answer), while everything else is one
teacher-forced batched forward pass. It is off by default for that reason; add it later
with `backfill_eval.py`.

### Where the outputs land

For the full metric inventory -- every number, its computation site, and its file --
see **`docs/METRICS.md`**. The tree below is just the shape.

```
<output_root>/
  manifest.json                      the immutable split
  reference/ or reference_retain90/  retain-only model + truth_ratios.json
  runs/<run_id>/
    run_state.json                   sign vector + seeds (ground truth)
    trained/                         fine-tuned checkpoint
    methods/<method>/
      losses.json                    THE audit's primary data
      metrics.json  separation.json  utility.json
      model/                         unlearned weights (keep_all)
  audit/<method>/audit_summary.json  epsilon + per_r sweep + 4 plots
  forget_quality/forget_quality.csv  plus / minus / all KS per group
  collected/*.csv                    tidy tables for analysis
```

---

## Read in this order

### 1. `audit_tofu/epsilon_bounds.py` (321 lines) — start here

The audit's output. Read the module docstring first: it states the two conventions that differ
from the paper's reference code and states Lemma 4.2 in full.

| Symbol | Line | What |
| --- | --- | --- |
| `LDP_REDUCTION_FACTOR` | 83 | `= 2.0`, the Lemma 4.1 halving |
| `vendored_halves_internally` | 86 | **the double-halving guard** — see below |
| `epsilon_lb_mean` | 243 | **the headline function** — mean-based bound |
| `epsilon_lb_median` | 287 | median variant, reported alongside |
| `_finalize` | 153 | applies the `/2` *if the vendored module did not*, handles the infeasible case |

The whole file is a wrapper. It performs no arithmetic on the bound itself beyond the `/2`.

That `/2` is conditional on purpose. The file vendored here reports the **LDP** epsilon, so the
wrapper divides. A newer revision of `cum_runs_eps_lab.py` divides internally and returns both
`epsilon_lb_ldp` and an already-halved `epsilon_lb`; dropping it in without the guard would halve
every reported bound twice and understate the audit by 2×. `vendored_halves_internally` detects
that revision two ways (the `_unlearning_eps_from_ldp` helper in the module, or the
`epsilon_lb_ldp` key in the result), and every report records which layer divided under
`halving_applied_by`. That revision also dropped the `delta` argument, so `delta` is passed only
when the signature accepts it — and a non-zero `delta` then raises instead of being dropped.

### 2. `audit_tofu/cum_runs_eps_lab.py` (995 lines) — **vendored, do not edit**

Byte-identical copy of `shakeshpere_plays_audit/cum_runs_eps_lab.py` from the paper's reference code. Only three functions are used:

| Symbol | Line | Role |
| --- | --- | --- |
| `log_f_values` | 40 | `f(v)`, the combinatorial sum in Lemma 4.2 |
| `log_Z_closed_form` | 84 | `log M' = log C(m, ⌊m/2⌋)` |
| `compute_avg_v_test_epsilon_lb` | 518 | the mean-based test we call |
| `compute_median_v_test_epsilon_lb` | 649 | the median variant |

The other entry points (`g`-test, lower-tail, `threshold_c_*`) are unused here but kept so the
file stays identical to the original. `rho_mu_bounds.py` reuses five more of its helpers
(`log_binom`, `log1mexp`, `log_g_from_logf`, `logM_bound_avg_v_ge_a`, `a_from_v_list`) rather
than reimplementing them, which is what keeps the three audits on one null model.

### 2b. `audit_tofu/rho_mu_bounds.py` (934 lines) — the ρ and μ audits

Same overlap scores, the other two privacy parametrisations. Not vendored: the zCDP and GDP
statements are the extended versions of Lemma 4.2, and are absent from the main paper text, so this file carries its derivation in its docstrings.

| Symbol | Line | What |
| --- | --- | --- |
| `log_pi_values` | 119 | the chance-overlap distribution `π(u) = f(u)/M'` |
| `eps_gamma_zcdp` | 179 | `ε_γ^loc(ρ) = 4ργ` — the Rényi reduction |
| `mu_loc_gdp_group` | 474 | `μ_loc(μ) = 2μ`, exact (f-DP group operation, k = 2) |
| `rho_lb_mean` / `rho_lb_median` | 807 / 837 | zCDP reports |
| `mu_lb_mean` / `mu_lb_median` | 877 / 900 | GDP reports; `μ_LB = τ/2` in closed form |
| `rho_lb_pairwise_from_roc` | 353 | pairwise auditor's ρ from one ROC point (no caller here) |
| `eps_estimate_from_rho` / `_mu` | 653 / 690 | display conversions only — **not** lower bounds on ε |

Two contract differences from `epsilon_bounds.py`: no `/2` (the factor lives inside
`ε_γ^loc`/`μ_loc`, so every report says `halved: False`), and `eps_estimate` is a
readability conversion with no bound semantics.

### 3. `audit_tofu/manifest.py` (625 lines) — the immutable split

| Symbol | Line | What |
| --- | --- | --- |
| `is_balanced` | 71 | membership test for the paper's `S_m` |
| `sample_balanced_sign_vectors` | 92 | uniform draw from `S_m`; rejects repeats and an exclusion set |
| `sample_stratified_sign_vectors` | 138 | **opt-in**: forces per-author marginals to `Γ/2` (calibration only) |
| `_resolve_batching` | 236 | maps `B` to `m`; validates that `B` divides `qa_per_author` |
| `build_manifest` | 295 | the whole split; ends by hashing its own content |
| `validate_manifest` | 527 | re-checks every invariant on load |
| `positive_candidates` / `negative_candidates` | 616 / 622 | `S_j = ±1` batch ids |

Read `build_manifest` for: the four independent seed streams, the two separate RNG streams for
calibration vs evaluation, the `|S_m| = C(m, m/2)` capacity check (line ~412), and
`calibration_coverage`, which records how many observations each QA Gaussian will get.

### 4. `audit_tofu/tofu_data.py` (358 lines) — **highest correctness risk**

| Symbol | Line | What |
| --- | --- | --- |
| `load_tofu_examples` | 67 | TOFU load; author index `= row // 20` (verified against the data) |
| `global_order` | 180 | the ONE fixed permutation, from `data_order_seed` |
| `training_examples_for_run` | 242 | `D(S)`, as the global order *restricted* to the run |
| `forget_examples_for_run` | 226 | `D_f(S)` — positive candidates only, never negatives |
| **`encode_example`** | **262** | **read this closely** — chat template + answer-only masking |
| `AnswerLossCollator` | 333 | pads with `IGNORE_INDEX` so padding is out of the loss too |

`encode_example` is where the audit could silently break. It tokenizes the chat-templated prompt
and the answer **separately** and concatenates, so the boundary is exact by construction and
`labels = [-100]*len(prompt_ids) + answer_ids`. It never tokenizes the joined string and hunt
for the boundary afterwards, which is where off-by-one leaks come from.

### 5. `audit_tofu/scoring.py` (171 lines) — the audit score

`score_examples` (line 32) computes `ℓ_z(f) = -(1/|a|) Σ_t log p_f(a_t | q, a_<t)`.

Points worth checking as you read: the causal shift (`shift_logits`/`shift_labels`), the FP32
cast **before** the log-softmax, and that the per-example mean comes from per-example token
sums rather than a batch-level mean — a batch mean would weight examples by answer length and
be a different statistic.

### 6. `audit_tofu/attack.py` (452 lines) — Instantiation I

| Symbol | Line | What |
| --- | --- | --- |
| `log_normal_pdf` | 81 | stable closed form; raises on non-positive variance |
| `QAGaussian` | 94 | one QA pair's in/out fit; `.llr()` is `λ_z` |
| `fit_calibration` | 167 | **calibration runs only**; variance floor + pooled variance |
| `_batch_scores` | 328 | `Λ_j = Σ_z λ_z` (`sum` default, `mean` ablation) |
| **`predict`** | **359** | top `r/2` / bottom `r/2`; **no label parameter exists** |
| `overlap` | 434 | `V = Σ_j max{0, Ŝ_j S_j}`; the only function touching truth |

Label hygiene is structural, not conventional: `predict` has no parameter for labels or a
manifest, and the module never imports `manifest`. Tests assert both by introspection.

Tie handling (line ~405): batches are ranked once by `(-Λ_j, batch_id)`, and the negative side
is taken from the opposite end of that *same* ordering — so no batch can be picked twice even
when every `Λ` coincides.

### 7. `audit_tofu/train.py` (220 lines) and `unlearn.py` (407 lines)

`train_model` (train.py:71) is the shared loop. It accepts a `loss_fn` so a method can reuse the
data ordering, scheduling, accumulation and logging with a different objective.

| Symbol | Line | What |
| --- | --- | --- |
| `sequence_logprob` | unlearn.py:99 | per-example `log π(a\|q)`, FP32, answer tokens only |
| `_forget_retain_unlearn` | unlearn.py:128 | shared loop for `npo`, `grad_ascent`, `grad_diff` |
| `apply_unlearning` | unlearn.py:360 | dispatch over all five methods |

The five methods are `noop`, `npo`, `retain_ft`, `grad_ascent`, `grad_diff`; the last two are
ports of OpenUnlearning's `GradAscent` and `GradDiff`. All three forget-set methods share one
loop so that data ordering, accumulation, LR schedule, clipping and logging are identical —
a difference in `ε_LB` then reflects the objective, not the plumbing.

One asymmetry is deliberate and faithful to upstream: `npo` uses the **sum** of answer-token
log-probs (the DPO-style ratio needs a sequence log-likelihood), while `grad_ascent` and
`grad_diff` use HF's `outputs.loss`, the **mean** over answer tokens.

The NPO objective is `L = (2/β)·softplus(β·(log π_θ − log π_ref))`, verified identical to
OpenUnlearning's `compute_dpo_loss`. **See `docs/RESOURCE_ESTIMATE.md` §0c**: the `2/β`
prefactor does make the retain term ~20× weaker at `β = 0.1`, but that is upstream's own
published default, so it is a reporting decision rather than a bug to fix.

### 8. `audit_tofu/run_manager.py` (289 lines) — plumbing worth a look

`RunState` (line 80) is written once and read back thereafter, refusing to resume if the
`split_hash` or any seed changed. `prune_checkpoints` (line 201) resolves every path and refuses
anything not strictly inside the run's own directory — there is a test that points a symlink
outside and asserts the target survives.

### 9. `audit_tofu/utility.py` (325 lines) and `plotting.py` (154 lines)

Independent of the bound. `forget_quality` (utility.py:242) returns `None` with an explanation
when no retain-only reference model exists, rather than substituting a proxy.

### 10. `scripts/` — the drivers

| Script | What |
| --- | --- |
| `build_manifest.py` | build the split once; `split.candidate_author_ids` pins the pool |
| `run_single.py` | one run by `run_id`; `--dry_run` validates without training |
| `aggregate_audit.py` | calibration → prediction → overlap → epsilon + rho + mu → plots |
| `train_reference.py` | the retain-only reference model; one per `D_r`, not per config |
| `forget_quality.py` | post-hoc KS (plus/minus/all), CPU only, from stored truth ratios |
| `backfill_eval.py` | add a metric to finished runs from saved checkpoints; `--what verify` |
| `backfill_dp_bounds.py` | rho/mu (and an epsilon cross-check) from stored overlaps; `--scan` at any depth, `--in_place`, `--csv` |
| `backfill_dp_bounds.sh` | the one-command form of the above over `runs/`; `DRY_RUN=1`, `IN_PLACE=1`, `ZETA=` |
| `reaudit_r_grid.sh` | re-aggregate finished experiments over a new `r` grid into `audit_r_grid/`, leaving `audit/` alone |
| `latex_audit_table.py` | LaTeX table of ε/ρ/μ (+ the ε conversions) at one `(m, r)`, one row per method; `--out`, `--json_out` |
| `collect_results.py` | tidy CSVs incl. per-row `truth_ratios.csv` |
| `launch_campaign.sh` | the whole campaign: manifests → reference → audits → aggregation |
| `run_audit.sh` | one config's 30 runs, with GPU scheduling, resume, status ticks |
| `analyze_signal.py` | variance decomposition; is there signal at all |
| `report_resources.py` | measured runtime/memory table |
| `run_smoke.py` | tiny end-to-end, offline |

In `run_single.py`, note the invariant block (~line 88) that asserts sign-vector balance,
disjointness and no negative-candidate leakage **before** any compute is spent.

In `aggregate_audit.py`, `_select_r_on_calibration` (line 73) freezes `r` using calibration runs
only. It optimises *projected epsilon*, not overlap margin — the margin is `1.0` at every `r`
under a strong attack and previously chose `r=4` over `r=20`.

---

## Where the tests are

| File | Test fns | Covers |
| --- | --- | --- |
| `test_manifest.py` | 30 | sign-vector balance, disjointness, training membership, hashing, batching |
| `test_unlearn.py` | 26 | NPO objective vs hand arithmetic; it raises forget loss |
| `test_attack.py` | 24 | label hygiene, Gaussians, variance floor, tie determinism |
| `test_config_and_runs.py` | 34 | configs, resume safety, the deletion guard |
| `test_epsilon_bounds.py` | 20 | Lemma 4.2 re-derived in exact integers; the `/2`; the `r` convention |
| `test_backfill_dp_bounds.py` | 10 | the backfill is a pure function of the stored `v_list`; epsilon reproduces |
| `test_latex_audit_table.py` | 6 | the table is a view: it prints the stored numbers, dashes for "nothing certified" |
| `test_rho_mu_bounds.py` | 35 | `π` pinned to the `f` table; `μ_LB = τ/2` saturates `ζ`; the ρ bisection is tight; **no** halving |
| `test_utility.py` | 16 | ROUGE-L, harmonic mean, forget quality |
| `test_modeling.py` | 14 | attention fallback, optimizer, LoRA labelling |
| `test_loss_masking.py` | 10 | **prompt tokens excluded from the loss** |
| `test_aggregate_integration.py` | 9 | full aggregation via subprocess, 3 leakage regimes |

`184 passed, 1 skipped` (the skip is `test_modeling.py:128`, which needs `peft`).
Run: `.venv/bin/python -m pytest tests/ -q`

---

## If you only read four things

1. **`encode_example`** (tofu_data.py:262) — the masking the whole audit rests on.
2. **`predict`** (attack.py:359) — the attack, and why labels cannot reach it.
3. **`epsilon_lb_mean`** (epsilon_bounds.py:140) — the two convention fixes over upstream.
4. **`_select_r_on_calibration`** (aggregate_audit.py:73) — where `r` is frozen, and why on
   epsilon rather than overlap.

And one on the utility side: **`forget_quality_slices`** (utility.py) — the single
implementation of forget quality, shared by the inline path in `run_single.py` and the
post-hoc `forget_quality.py` so the two cannot disagree. Its `minus` slice is the
negative control that makes `plus` interpretable; read it as a distribution across runs
(mean ≈ 0.5 under the null), never as a single value.
