# Metric reference: what is computed, where, and where it lands

Every number this project produces, with its computation site and its file on disk.

The project reports **two independent families** that answer different questions and must
not be conflated:

| family | question | needs | authoritative file |
| --- | --- | --- | --- |
| **audit** (`ε_LB`) | can an attacker tell trained-then-unlearned from never-trained? | calibration + evaluation runs | `audit/<method>/audit_summary.json` |
| **utility / forget quality** | does the model still behave, and does the forget set look un-learned? | evaluation runs + a reference model | `runs/*/methods/*/utility.json`, `forget_quality/forget_quality.csv` |

A method can score well on one and badly on the other. That gap is the point of the audit.

---

## 1. Audit family

| metric | computed in | stored in |
| --- | --- | --- |
| `loss` (answer-only NLL `ℓ_z`) | `scoring.py:32` `score_examples` — eval mode, `no_grad`, FP32 logits | `methods/<m>/losses.json` → `records[]` |
| `num_answer_tokens` | same | same |
| in/out Gaussians `μ_in, μ_out, var_in, var_out` | `attack.py:167` `fit_calibration` — **calibration runs only** | `audit/<m>/calibration.json` → `gaussians[]` |
| `Λ_j` (per-batch log-likelihood ratio) | `attack.py:359` `predict` — labels NOT in scope | `audit/<m>/audit_summary.json` |
| overlap `V` | `attack.py:434` `overlap` — labels revealed only here | `audit_summary.json` → `headline.v_list` |
| `ε_LB` (mean / median) | `epsilon_bounds.py:158` / `:202` — vendored Lemma 4.2, then `/2` per Lemma 4.1 | `audit_summary.json` → `headline.epsilon_lb_mean` |
| `ε_LDP` (before halving) | `epsilon_bounds.py:158` | `headline.epsilon_ldp_lb_mean` |
| `ρ_LB` (zCDP, mean / median) | `rho_mu_bounds.py:807` / `:837` — Rényi reduction `ε_γ^loc(ρ)`, **no** halving | `audit_summary.json` → `headline.rho_lb_mean`, `per_r.<r>.rho_lb_mean` |
| `μ_LB` (GDP, mean / median) | `rho_mu_bounds.py:877` / `:900` — f-DP group op, `μ_loc(μ) = 2μ`, **no** halving | `headline.mu_lb_mean`, `per_r.<r>.mu_lb_mean` |
| `eps_estimate_from_{rho,mu}` | `rho_mu_bounds.py:653` / `:690` — display conversion, **not** a bound | `headline.eps_estimate_from_rho_mean` / `_mu_mean` |
| `halving_applied_by` | `epsilon_bounds.py:97` `_finalize` — `wrapper` or `vendored_module` | `headline.halving_applied_by` |
| ρ/μ for audits finished before this existed | `backfill_dp_bounds.py`, or `backfill_dp_bounds.sh` for all of `runs/` — recomputed from the stored `v_list`, nothing re-run | `audit/<m>/dp_bounds.json`, `runs/dp_bounds_summary.csv`, `runs/dp_bounds_all.json` (`--json`; + `--in_place`) |
| the same bounds over a widened `r` grid | `reaudit_r_grid.sh` — CPU re-aggregation from `losses.json`, original `audit/` untouched | `audit_r_grid/<m>/audit_summary.json` |
| paper-ready table at one `(m, r)` | `latex_audit_table.py` — renders stored values, computes nothing | wherever `--out` points, e.g. `audit_r_grid/table_m400_r100.tex` |
| frozen `r`, per-`r` sweep | `aggregate_audit.py:74` `_select_r_on_calibration` | `audit_summary.json` → `r_selection`, `per_r`; `per_r_detail.json` |
| `mean_in`, `mean_out`, `gap`, `cohens_d` | `run_single.py` (in/out separation block) | `methods/<m>/separation.json`, and `runs.csv` as `sep_*` |

`separation.json` is the first-look diagnostic: a positive `gap` means the unlearned
examples are still easier for the model, i.e. residual memorization. It is **persisted**,
not only pushed to wandb, so it survives a wandb outage.

`ρ_LB` and `μ_LB` audit the *same* observation (`headline.v_list`) under the other two
privacy parametrisations, from the same chance-overlap null (`log_pi_values` is pinned to
the ε audit's `f` table in `tests/test_rho_mu_bounds.py`). Two differences from `ε_LB`
matter when reading them:

* **No `/2`.** The Lemma 4.1 reference-law factor is already inside the local parameter
  each bound inverts (`μ_loc(μ) = 2μ`; the leading 2 of `ε_γ^loc(ρ)`), so the solved value
  *is* the certified-unlearning parameter. Every report carries `halved: False`.
* **`eps_estimate` is not a bound.** It converts `ρ_LB`/`μ_LB` into an `(ε, conv_delta)`
  pair (default `conv_delta = 1e-3`) only so the numbers sit on the same axis as `ε_LB`.
  The conversion maps a guarantee to a weaker guarantee, so applying it to a *lower* bound
  carries no bound semantics.

`rho_mu_bounds.py:353` additionally implements the pairwise auditor's ρ bound from one ROC
point; nothing in this repo produces ROC points (the TOFU instantiation reports overlaps
over `m = 20` candidates), so it currently has no caller.

---

## 2. Utility family (single-model — no reference needed)

All computed in `evaluate_split` (`utility.py:150`), one row at a time, over the four
TOFU perturbed configs.

| metric | computed in | note |
| --- | --- | --- |
| `probability` | `utility.py:203` (or `:205` when a split has no perturbations) | softmax over `{true} ∪ perturbed`, length-normalized |
| `truth_ratio` | `utility.py:221` | `mean_p P(a_pert)^(1/|a_pert|) / P(a_para)^(1/|a_para|)`. **Same model on both sides — no reference involved.** |
| `rouge_l_recall` | `utility.py:261` via `rouge_l_recall` (`:100`) | needs generation, so opt-in; see §5. Delegates to upstream's `rouge_score` with `use_stemmer=True`, so values are comparable to published TOFU numbers — see §2.1 |
| `per_row[]` | end of the `evaluate_split` loop | `(author_id, qa_id, probability, truth_ratio, rouge_l_recall)` — the grain forget quality is sliced on |
| `model_utility` | `utility.py:281`, returned at `:303` | harmonic mean over retain / real_authors / world_facts |

Stored in `runs/<id>/methods/<m>/utility.json` → `splits.<group>.*`.

### 2.1 Parity with upstream TOFU

Surveyed against `locuslab/tofu` @ main (fetched 2026-09-16): `evaluate_util.py`,
`utils.py`, `aggregate_eval_stat.py`, `config/eval_everything.yaml`.

**Upstream is internally inconsistent about truth ratio.** `utils.get_model_utility`
uses an *arithmetic* mean of the perturbed probabilities
(`np.exp(-loss).mean(-1) / np.exp(-paraphrase_loss)`), while
`utils.get_forget_quality` uses a *geometric* one in the reciprocal direction
(`np.exp(perturbed_loss.mean(-1) - paraphrase_loss)`). `utility.py:233` follows the
arithmetic form, which is also the form stated in the paper.

Matches upstream: per-token length normalization; answer-only masking; the
`paraphrased_answer`-else-`answer` denominator (upstream's `base_answer_key`, which
is `answer` for `real_authors`/`world_facts`); the truth-ratio formula; the
`{true} ∪ perturbed` softmax for `probability`; `ks_2samp` for forget quality;
ROUGE-L **recall** via upstream's own scorer.

Known remaining divergences, none currently documented as intentional:

| # | divergence | consequence |
| --- | --- | --- |
| 1 | `model_utility` drops zero-valued terms (`utility.py:312`); `scipy.stats.hmean` returns `0.0` if any term is 0 | a retain truth ratio ≥ 1 → `max(0, 1-tr) = 0` collapses upstream's Model Utility to zero, but is silently discarded here. Flatters an aggressively-unlearned model. |
| 2 | `model_utility` has a variable term count; upstream's is a fixed 9 (3 tasks × 3 metrics) | at the default `compute_rouge: false` this is a **6-term** harmonic mean, so it is not comparable to any published TOFU Model Utility |
| 3 | `probability` is normalized for all four groups; upstream branches on the filename substring `'eval_log' in k`, true for **both** `eval_log.json` (retain) and `eval_log_forget.json` (forget), so those two get the bare `mean(exp(-loss))` | `splits.retain.probability` and `splits.forget.probability` are on a different scale from TOFU's published figures, and the retain one feeds `model_utility` |
| 4 | forget quality is computed on the arithmetic truth ratio, upstream's on the geometric one | applied to model and reference alike, so the statistic is coherent, but not numerically equal to upstream's. The *reciprocal direction* is harmless: two-sided KS `sup|F−G|` is invariant under strictly monotone transforms. |

Fixed 2026-09-16: `rouge_l_recall` was a hand-rolled LCS over `str.split()` tokens,
which is a different metric from upstream's `rouge_score` (no lowercasing, no
punctuation stripping, no Porter stemming — `"Paris."` vs `"paris"` scored 0). It now
delegates to `rouge_score`, pinned by
`test_rouge_l_recall_matches_upstream_tofu_scorer`. No stored results changed: every
`rouge_l_recall` on disk was `null`, the metric never having been enabled.

### Author identity

`per_row` gets `(author_id, qa_id)` from `tofu_identity_map` (`utility.py:62`), which joins
each row's **question text** against the `full` config. Joining on text rather than row
position is deliberate: `forget10_perturbed` is author-ordered today, so
`author = 180 + i//20` happens to work, but that is an undocumented coincidence of
upstream's file layout. `real_authors` and `world_facts` are real trivia, not TOFU authors,
so they join to nothing and carry `author_id: null` — which is why only their `all` slice
exists in §3.

### The gate: 60 files, not 180

`run_single.py` (`want_utility`) runs the suite only when
`state.family == "evaluation"` **or** `utility.calibration_runs` is true. With the default
`calibration_runs: false`, a 30-run audit yields **10 eval runs × 6 methods = 60**
`utility.json` files. The 20 calibration runs produce `losses.json` only — all `ε` needs.

---

## 3. Forget quality (the KS family — needs the reference)

**One implementation**, `forget_quality_slices` (`utility.py:348`), used by both callers so
they cannot drift. It calls `ks_with_stats` (`:325`), which wraps `forget_quality` (`:455`)
— a single `scipy.stats.ks_2samp`, the only statistical test in the repo.

| slice | rows | meaning |
| --- | --- | --- |
| `plus` | the run's `S_j = +1` pairs | **the headline** — trained, then unlearned |
| `minus` | the run's `S_j = -1` pairs | **negative control** — neither model ever saw them |
| `all` | every row | the conventional TOFU number; **biased high**, it mixes the two above |

Each slice carries `forget_quality` (p), `ks_statistic`, `n_model`, `n_reference`,
`mean_tr_model`, `mean_tr_reference`, `median_tr_*`, `n_authors`.

Sign is resolved per `(author, qa)`, not per author, so it is exact at every batch size
`B`. At `B < qa_per_author` one author's pairs carry mixed signs; an author-level map would
drop them all and silently empty the slices (at `batching: qa`, all 20 of 20).

### How to read `minus`

Under the null its p-value is **`Uniform(0,1)`**, so a single value near 0.1 means nothing.
Judge the **mean across runs and methods**: ≈ 0.5 means the reference is exchangeable with
the runs and `plus` is readable. Well below 0.5 means a systematic nuisance difference —
the reference differs from each run by training seed (`train_seed_base - 1` vs
`1000 + run_index`) and by ~63 optimizer steps (3600 vs 3800 examples at fixed epochs) —
and `plus` cannot then be taken at face value.

### Two places it is written, and which to trust

| location | written by | content |
| --- | --- | --- |
| `utility.json` → `forget_quality_slices.<group>.<slice>` | `run_single.py`, inline | all groups × all slices |
| `utility.json` → `forget_quality_plus`, `ks_statistic_plus` | `run_single.py` | promoted headline (forget/plus) |
| `utility.json` → `forget_quality`, `ks_statistic` | `utility.py:559-560` | **legacy flat** = the `all` slice |
| `forget_quality/forget_quality.csv` | `scripts/forget_quality.py` | **authoritative**; one row per (run, method, group, slice) |

Both paths produce **identical** numbers — verified 12/12 on a live fixture. The inline
fields require the reference to exist when the run executes; `forget_quality.py` recovers
them afterwards from stored `per_row` data, on CPU.

> **Trap.** `runs.csv` currently surfaces only the legacy `forget_quality` /
> `ks_statistic`, i.e. the **diluted `all`** value, under a name that looks authoritative.
> On a real run: `all` p = 0.155 while `plus` p = 0.016. Read KS from
> `forget_quality.csv`, or add `fq_<group>_<slice>` columns to the collector.

---

## 4. The reference model

Built once by `scripts/train_reference.py` into `utility.reference_dir`
(default `runs/reference_retain90`). It is a function of
`D_r` **alone**, so one model serves every config pinning the same retain set — all four
audit configs do, despite having different `split_hash` values because `m` differs.
`forget_quality.py` therefore validates on the **retain author set**, not on `split_hash`.

| file | content |
| --- | --- |
| `trained/` | the `D_r`-only checkpoint |
| `truth_ratios.json` | per-row, identified, per group — the KS baseline |
| `utility.json` | its own suite; `forget_quality` null by construction (KS against itself) |
| `candidate_losses.json` | answer-only NLL on all `m` candidate batches — the never-trained loss floor, directly comparable to each run's `-1` losses |
| `metrics.json`, `run.log` | training telemetry, provenance |

`retain_ft` is **not** a substitute: it loads `trained/`, which already saw the forget data,
so both of its distributions are post-exposure and the KS measures nothing.

---

## 5. What is NOT computed by default

| metric | why | how to get it |
| --- | --- | --- |
| `rouge_l_recall` | 1854 ms/row measured — 16× the rest of the suite, ~31 GPU-h per audit. Autoregressive generation at batch 1 (~130 tokens/row against a 36-token answer) vs one batched teacher-forced pass for everything else. | `utility.compute_rouge=true`, or `backfill_eval.py` from saved weights |
| utility on calibration runs | `ε` needs only `losses.json` | `utility.calibration_runs=true`, or `backfill_eval.py --families calibration` |
| `forget_quality` when no reference exists | returns `None` with `forget_quality_note`, never a faked value | train the reference, then `forget_quality.py` |

Do **not** use `utility.limit` to make ROUGE cheaper: it truncates every split to the first
N rows, and since `forget10_perturbed` is author-ordered, `limit: 100` keeps only authors
180–184 and destroys the `+1`/`-1` slicing. `backfill_eval.py` refuses to merge a
row-count mismatch for this reason.

---

## 6. Adding a metric to finished runs

`retention: keep_all` + `save_unlearned_checkpoint: true` keep the post-unlearning model at
`runs/<id>/methods/<m>/model/`, so a new metric costs a forward pass:

```bash
python scripts/backfill_eval.py --config configs/base.yaml --dry_run
python scripts/backfill_eval.py --config configs/base.yaml --what utility \
    --gpu 5 --set utility.compute_rouge=true
python scripts/backfill_eval.py --config configs/base.yaml --what verify --gpu 5
```

`verify` re-scores the candidate pool from the saved weights and compares against
`losses.json`. Measured agreement: **`max|d| = 0.00e+00` over 400 rows** — proof the saved
tensors are the scored tensors.

For the live path that guarantee is structural rather than checked: `run_single.py` loads
`trained/` once, `apply_unlearning` mutates it **in place** (it returns metrics, not a
model — `unlearn.py:449`), and scoring, saving and the utility suite all read that same
object. Each method reloads `trained/` fresh, so methods cannot contaminate each other.

Backfill cannot help when the **split** changes: a different candidate partition means
different training data, so those runs must actually happen.

---

## 7. Tidy tables

`scripts/collect_results.py` → `<output_root>/collected/`

| file | grain | rows for a 30-run, 6-method audit |
| --- | --- | --- |
| `losses.csv` / `.parquet` | (run, method, author, qa) + ground-truth `sign` | 72,000 |
| `runs.csv` | (run, method) — timings, memory, unlearning metrics, `sep_*`, all utility | 180 |
| `truth_ratios.csv` | (run, method, group, row) + `author_id`, `qa_id`, `sign` | ~61,000 |
| `manifest_flat.csv` | (run, batch) → sign | 600 |
| `summary.json` | counts, coverage, `split_hash` | — |

Everything is keyed by `split_hash` and the collector **refuses to mix splits**
(`collect_results.py:119`), so a regenerated manifest cannot silently contaminate a table.
