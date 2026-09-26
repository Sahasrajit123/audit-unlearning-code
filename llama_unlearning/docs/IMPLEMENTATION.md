# Implementation record

What was built for the TOFU / Llama-3.2-1B unlearning audit, why each decision was made, and
what remains open. Companion to `README.md` (how to run it) and
`docs/RESOURCE_ESTIMATE.md` (measured cost).

---

## 1. Relationship to the paper's reference code

The auditing math reuses the code release accompanying the paper. Its epsilon module is
vendored byte-identically as `audit_tofu/cum_runs_eps_lab.py`.

The closest prior art is `shakeshpere_plays_audit/` — the Shakespeare character-LM audit, which
already implements Instantiation I of the paper's auditor. Its conventions were carried over:
`run_<id>/` directories holding `config.json` / `metrics.json` / `run.log`; the seed derivation
pattern (`base + offset + run_index`); and the field names `overlap_v`, `v_list`,
`epsilon_lb`, `delta`, `ci_delta`.

Two things were deliberately *not* carried over. Upstream uses JSON configs; the spec requires
YAML, so YAML won. And upstream hard-codes a device (`cuda:5`); runs here take `--gpu`.

---

## 2. The epsilon bound: vendored verbatim, with two adapters

`audit_tofu/cum_runs_eps_lab.py` is a **byte-identical copy** of the upstream module
(`cmp`-verified, 995 lines). The spec forbids rederiving or silently modifying a tested
implementation, so all corrections live in `audit_tofu/epsilon_bounds.py`, which changes
nothing about the math and only adapts two conventions.

### 2.1 `r` is total here, per-side upstream

Upstream's caller `evaluate_llr_predictions.py --r 50` means "top 50 **and** bottom 50", then
passes `epsilon_r = 2 * r_value` into the bound. The paper ("Fix an even guess budget `r`") and
this spec ("make exactly `r/2` positive and `r/2` negative guesses") define `r` as the *total*
budget. This project uses the spec's convention throughout and passes `r` straight through,
unhalved and undoubled. `test_r_is_total_not_per_side` pins the equivalence.

### 2.2 Upstream omits the Lemma 4.1 halving

Lemma 4.1: if `(A, U)` is `(ε,0)`-certified unlearning under the threat model, the auditor
mechanism `M` is **`(2ε,0)`-locally differentially private**. §4.2 then states: *"dividing by 2
to undo the reduction of Theorem 4.1 yields the reported `ε_LB`."*

`evaluate_llr_predictions.py` does not perform this division — it reports the LDP epsilon. So
this project reports **both**, and never silently:

```
epsilon_ldp_lb  = solution of the LDP bound     (== upstream's "epsilon_lb")
epsilon_lb      = epsilon_ldp_lb / 2            (the unlearning bound, reported)
```

### 2.3 Independent verification of Lemma 4.2

Because the vendored file is trusted but copied, `tests/test_epsilon_bounds.py` re-derives
`f(v)` from Lemma 4.2 in **exact integer arithmetic** with `math.comb` and compares against
`log_f_values`, for `(m,r)` in `{(20,4),(20,8),(20,12),(20,16),(20,20),(6,2),(6,6),(400,100)}`.
It also checks the partition identity `Σ_v f(v) = |S_m| = C(m, ⌊m/2⌋)` — every balanced sign
vector yields exactly one overlap value against a fixed guess, so the table must partition
`S_m`. Both hold to `1e-9`.

Other properties verified: monotone in observed overlap; growing in `r` for a perfect attack;
tightening with more runs; larger `m` raising the ceiling (Remark 4.3); stricter `ζ` giving a
smaller bound; chance-level and below-chance overlap certifying nothing.

### 2.4 `delta`

The paper's main test fixes `δ = 0`; footnote 3 explains the LDP reduction does not extend
cleanly to `δ > 0` (`X ≈_{ε,δ} Y` and `Y ≈_{ε,δ} Z` give only `X ≈_{2ε,(1+e^ε)δ} Z`). The
default is therefore `epsilon.delta: 0.0`, not upstream's example value of `1e-5`.

---

## 3. Audit split and randomness

`audit_tofu/manifest.py` builds one immutable, SHA-256-hashed JSON manifest. Downstream stages
re-verify the hash; `save_manifest` refuses to clobber without `overwrite=True`.

Randomness is split into four independent streams so no re-run can perturb an earlier decision:

| Seed | Governs |
| --- | --- |
| `split_seed` | Author partition and all sign vectors |
| `data_order_seed` | The single global example ordering |
| `train_seed_base` | Per-run stochastic training seed (`base + run_index`) |
| `unlearn_seed_base` | Per-run unlearning seed (`base + run_index`) |

Calibration and evaluation vectors come from **separate RNG streams**
(`default_rng([split_seed, 0xC0FFEE])` and `[split_seed, 0xE7A1]`), not sequential draws from
one, so changing `Γ` cannot shift the evaluation vectors. Evaluation additionally excludes
every calibration vector, and `validate_manifest` re-checks disjointness on every load.

### 3.1 Global example ordering

The spec requires one fixed ordering restricted per run, and warns against creating a
systematically different order for positive and negative candidates. `global_order` permutes
*all* 4000 examples once from `data_order_seed`; `training_examples_for_run` filters that
sequence to the run's examples. Per-epoch reshuffling is opt-in and **off by default**, so the
realized order matches the order the paper's threat model assumes the adversary observes (`π`).

Why it matters: a per-run order would place positive and negative candidates at systematically
different epoch positions, and the attacker would then be partly reading off training-order
effects rather than unlearning failure — inflating `ε` for reasons unrelated to unlearning.
`test_positive_and_negative_candidates_have_no_systematic_order_bias` asserts the mean
normalized position of positives is ~0.5.

### 3.2 A capacity constraint the tests found

Strict disjointness needs `Γ + L ≤ |S_m| = C(m, m/2)`. Trivial at `m = 20` (184,756), but
binding for small configurations: the first smoke config asked for 6 + 3 disjoint vectors at
`m = 4`, where only 6 exist. `build_manifest` now fails immediately with the actual numbers
instead of exhausting rejection sampling, and `smoke.yaml` uses 4 + 2.

### 3.3 Author identity in TOFU

TOFU's `full` config holds 4000 rows ordered by author, so **author index = row index // 20**.
Verified directly against the data: rows 0–19 are Jaime Vasquez, 20–39 Chukwu Akabueze,
3980–3999 Nikolai Abilov. This is the convention OpenUnlearning's `forget10` / `retain90`
splits rely on. A content fingerprint (SHA-256 over all questions and answers) is stored in the
manifest so upstream dataset drift is detectable; the current value is `6283b48ad77d9f7a...`.

### 3.4 Batching abstraction: the audit batch size `B`

`split.batching` is the paper's audit batch size `B` (§6), and accepts `"author"`, `"qa"`, or
**any integer `B` dividing `qa_per_author`**. The candidate pool is fixed at 20 authors x 20 QA
= 400 pairs (10% of the data, matching TOFU's own `forget10` and the paper's CIFAR-100 10%
forget fraction); `B` only changes how that pool is *partitioned into canaries*, so `m = 400/B`.

| `B` | `m` | perfect-attack ceiling at `L=10` | |
| --- | --- | --- | --- |
| 20 | 20 | **6.59** | `"author"` — the spec default |
| 10 | 40 | 13.35 | |
| 5 | 80 | ~24 | |
| 4 | 100 | 33.92 | |
| 2 | 200 | 68.40 | |
| 1 | 400 | **137.54** | `"qa"` — see `configs/qa_level.yaml` |

Grouping is **author-respecting** at every `B`: an author's 20 pairs split into contiguous
blocks, so a batch never spans two authors.
`B = qa_per_author` canonicalizes to `"author"` and `B = 1` to `"qa"`, so each split has exactly
one manifest representation. Non-divisors are rejected with the list of valid `B`.

Which `B` to use is a real decision, not a formality:

* **Larger `m` raises the attainable epsilon** (Remark 4.3) — identifying the right sign vector
  among `|S_m|` candidates is information-theoretically harder, and `|S_400| ~ 10^119` versus
  `|S_20| = 184,756`.
* Per Remark 4.3, **heuristic methods prefer small `B`** (they leak strongly per batch) and
  **certified methods need large `B`**. Every method implemented here is heuristic, so `m = 20`
  is arguably the wrong end of the trade-off for them — but the spec sets it, so it stays the
  default.

### 3.5 Derived fields are excluded from the hash

`DERIVED_FIELDS` lists manifest fields the `split_hash` deliberately does **not** cover, and
`hashable_content()` is the single source of truth used by `build_manifest`, `validate_manifest`
and the tests. Currently that is `calibration_coverage`, which is computed *from* the sign
vectors and so adds no information to the hash.

This was learned the hard way: adding `calibration_coverage` as a diagnostic silently changed
the hash of every manifest, which would have made previously-completed runs unresumable. The
first `m = 20` manifest on disk (`7569508c...`) predates those fields and no longer matches a
fresh build — its *split content is byte-identical* (verified: authors, batches, sign vectors,
seeds), so the five pilot runs' measurements stand, but they cannot be resumed against a
regenerated manifest. Re-running them costs ~1.8 h if ever needed.

---

## 4. A finding: calibration coverage is uneven, and the ceiling is low

Two properties of the configured shape are worth stating plainly, because neither is a bug and
both limit what the experiment can show.

### 4.1 `ε_LB` cannot exceed ≈ 6.59 here

At `m = 20`, `L = 10`, `ζ = 0.05`, `δ = 0`, a **perfect** attack (`V = r` on every run) gives:

| `r` | 4 | 8 | 12 | 16 | 20 |
| --- | --- | --- | --- | --- | --- |
| `ε_ldp` | 2.365 | 4.999 | 7.589 | 10.167 | 13.179 |
| **`ε_LB`** | **1.182** | **2.500** | **3.795** | **5.084** | **6.589** |

So `ε_LB ≈ 6.59` is the ceiling, pinned in `test_audit_ceiling_at_the_configured_shape`. For
context, the paper reports 50–60+ for heuristic methods and 142.5 for Hessian unlearning — but
those used `m = 400` (Shakespeare) and `m = 4500` (CIFAR-100). Per Remark 4.3, larger `m`
raises the attainable bound because identifying the right sign vector among exponentially many
candidates is information-theoretically harder.

This is a consequence of the spec's `m = 20`, not of the implementation. If a bound comparable
to the paper's is wanted, the lever is `split.batching: qa` (`m = 400`), which raises the
ceiling substantially; the trade-off is weaker per-batch leakage, since each batch is then a
single QA pair. Both are supported; `m = 20` remains the default because the spec sets it.

### 4.2 Some QA Gaussians are fitted from only 4 points

With i.i.d. balanced sign vectors, each author's number of `+1` calibration runs is
`~Binomial(20, ½)`. The realized manifest gives **`n_in ∈ [4, 14]`** and `n_out ∈ [6, 16]`
(mean 10). An author observed in-condition 4 times yields a poor per-QA variance estimate.

The spec anticipated this and prescribed the mitigations, which are implemented and on by
default: a configurable **variance floor** (`1e-4`), **pooled variance** across an author's 20
QA pairs (`pool_variance: author`, pooling *centred residuals* so between-QA mean differences
are not absorbed into the variance), and explicit degeneracy warnings.

Additionally, `split.calibration_balance: stratified` is offered as an **opt-in**: it forces
each author to be `+1` in exactly `Γ/2 = 10` calibration runs, so every QA pair gets 10 in and
10 out observations. This is legitimate because calibration runs only *construct* the mechanism
`M` — the bound's validity requires the **evaluation** vectors to be i.i.d. uniform over `S_m`,
and they always are, stratified or not. It is not the default because the spec says each run
samples `S`; the departure is in the per-batch marginal across runs, not in any individual
vector, which stays in `S_m`. Construction: circulant base randomized by margin-preserving
2×2 swaps.

| `calibration_balance` | `n_in` range |
| --- | --- |
| `iid` (default) | `[4, 14]` |
| `stratified` | `[10, 10]` |

---

## 5. Training and unlearning

### 5.1 Fine-tuning

Every run starts from the same pretrained checkpoint. The released TOFU "full" checkpoint is
deliberately **not** used: each run has a different inclusion vector `S` and so needs its own
fine-tune. Defaults follow the spec — full-parameter BF16, 5 epochs, `max_seq_length` 512,
gradient checkpointing on, micro-batch 2 × accumulation 8 (effective batch 16), AdamW,
no generation during training (`.generate()` is never called on the training path).

**Learning rate.** The spec asks for the established Llama-3.2-1B TOFU value where available,
otherwise an exposed, documented default. OpenUnlearning's TOFU configs use `1e-5` for
full-parameter 1B-scale fine-tuning, so `training.learning_rate: 1.0e-5` is the default, in
YAML and overridable. This is the one hyperparameter most worth revisiting after the pilot: if
`noop` does not separate in/out losses, too low an LR (under-memorization) is the first
suspect.

### 5.2 The three methods, branched from one checkpoint

The initial fine-tune runs **once per sign-vector run**; the saved checkpoint is then reloaded
and branched into each method, as the spec requires.

- **`noop`** — returns the trained model unchanged. The positive leakage control.
- **`npo`** — Negative Preference Optimization (Zhang et al. 2024), the form OpenUnlearning
  uses:
  `L = (2/β)·E_{D_f}[ softplus( β·(log π_θ(a|q) − log π_ref(a|q)) ) ]`
  with `π_ref` the frozen trained model at the start of unlearning. The sigmoid saturates, so
  per-example gradients stay bounded and the model degrades far more gracefully than under
  plain gradient ascent. `retain_weight > 0` adds a retain cross-entropy term — this is
  OpenUnlearning's `NPO_RT`, the variant that preserves usable utility, and the metadata
  records which variant ran. Default `β = 0.1`, `retain_weight = 1.0`, sequence log-likelihood
  reduced by `sum` (DPO/NPO convention). Retain batches are walked cyclically so the retain
  gradient is not dominated by whichever examples sit at the front.
- **`retain_ft`** — fine-tunes only on the fixed 180-author retain set. The paper's "pure
  fine-tuning on the retain set" heuristic.

**The forget set never contains `S_j = -1` candidates.** Those were never trained on, so
exposing them to the unlearner would leak the hidden sign vector into the pipeline and void the
audit. Enforced in `forget_examples_for_run`, asserted at the top of every run in
`run_single.py` before any compute is spent, and tested for all 30 runs.

---

## 6. Loss extraction

`audit_tofu/scoring.py` computes the audit score

```
l_z(f) = -(1/|a|) · Σ_t log p_f(a_t | q, a_<t)
```

Every spec requirement is enforced: the same chat template as training; all prompt tokens
masked; the mean over non-padding **answer** tokens only; FP32 accumulation even under a BF16
forward (BF16 has ~3 decimal digits of mantissa, coarse relative to the in/out gaps the attack
must resolve); `model.eval()` under `torch.no_grad()` so dropout is off; one record per
`(run_id, method, candidate_author, qa_id)`. Generated-answer ROUGE is **not** used as the
audit score — it appears only in the utility suite.

Two details that carry most of the correctness risk:

**Exact prompt masking.** The prompt and answer are tokenized *separately* and concatenated, so
the boundary is exact by construction and `labels = [-100]*len(prompt_ids) + answer_ids`. We
never tokenize the joined string and try to locate the boundary afterwards, which is where
off-by-one leaks come from. `test_question_text_does_not_leak_into_unmasked_labels` uses
lexically disjoint question and answer vocabularies and asserts no question token id appears
among unmasked labels.

**Per-example, not per-batch, means.** The mean is computed from per-example token sums, not a
batch-level mean, because a batch-level mean weights examples by answer length and would be a
different statistic than the one defined above.

`append_eos` must match between training and scoring; it is recorded in the run config and read
back by the scorer rather than re-guessed.

---

## 7. The attack, and label hygiene

`audit_tofu/attack.py` implements Instantiation I. `fit_calibration` consumes calibration runs
only; `predict` consumes an evaluation run's losses plus the frozen calibration; `overlap` is
applied afterwards by the caller.

The spec requires that evaluation labels cannot reach the predictor. That is enforced
**structurally**, not by convention:

- `predict(scores, calibration, r, *, aggregate)` has no parameter for a sign vector, labels,
  or a manifest, so a caller cannot smuggle them in without a `TypeError`.
- `attack.py` never imports `manifest`, so it cannot reach ground truth indirectly. Asserted by
  source inspection in `test_attack_module_does_not_import_manifest`.
- `test_predict_signature_has_no_label_parameter` pins the exact parameter set by
  introspection, so a future refactor that adds a label argument fails the suite.

**Aggregation.** `Λ_j = Σ_{z ∈ D_f,j} λ_z` with `λ_z = log p_in(l_z) − log p_out(l_z)`. `sum` is
the default per spec; `mean` is available as the ablation.

**Tie handling** is deterministic and label-free: batches are ranked by `(-Λ_j, batch_id)`, so
ties break on the lexicographic batch id. The negative side is taken from the opposite end of
the *same* ordering, so a batch can never be selected twice even when many `Λ` values coincide
— which is exactly what happens in the degenerate all-equal case. Verified stable across 20
repetitions with shuffled input dict order.

**Log densities** use a stable closed form rather than `scipy.stats.norm.logpdf`, and raise on
non-positive variance rather than returning `nan`. Verified against scipy to `1e-10`, including
a far-out point at `var = 1e-8` which must give a large finite negative value.

**Selecting `r`.** `aggregate_audit.py --select_r_from calibration` runs leave-one-out over the
*calibration* runs, refitting on the rest, and picks the `r` maximising the **projected epsilon
lower bound**. `r` is then frozen before the evaluation runs are touched; picking `r` by
maximising epsilon on the *evaluation* runs would invalidate the stated confidence level. All
`r` values are reported regardless, so the dependence stays visible.

The objective matters, and the first implementation got it wrong. It selected on the normalized
overlap margin `(V − r/2)/(r/2)`, which equals **1.0 for every `r`** under a near-perfect
attack — so it could not discriminate at all, ties resolved to the first key, and it chose
`r = 4` (`ε_LB = 1.18`) over `r = 20` (`ε_LB = 6.59`) on exactly the case the audit cares most
about. Epsilon still separates the candidates because the attainable ceiling grows with `r`, so
epsilon is the correct objective; it is also what makes the inverted U of Remark 4.3 visible.
Fixed, with `test_r_selection_maximizes_epsilon_not_overlap_margin` asserting both that the
margins are degenerate here and that the epsilon projection is strictly increasing. Ties now
break toward the *smaller* `r` (fewer forced low-confidence guesses), and if nothing is
certified on calibration it falls back to the largest `r`.

The projection evaluates the LOO overlaps at `L = 10` — the horizon the frozen `r` will
actually face — and floors the mean overlap, so the selection statistic is conservative. It is
a selection statistic only; the reported bound always comes from the evaluation runs.

---

## 8. Storage, resume, execution

- One fine-tune per run, branched into methods; no repeated initial training.
- Retention: `keep_all` | `keep_trained` | `delete_after_scoring` (default). JSON results are
  always kept; only weights are dropped.
- **Deletion safety.** `prune_checkpoints` resolves every path (symlinks included) and refuses
  anything not strictly inside the run's own resolved directory.
  `test_retention_never_deletes_outside_the_run_directory` replaces a method's model directory
  with a symlink to an outside directory and asserts the outside directory survives.
- Resume: `run_state.json` is written once and read back thereafter. It refuses to continue if
  the manifest's `split_hash` changed, or if a stored seed or sign vector disagrees with the
  manifest — either would silently corrupt the run families.
- `save_json` writes atomically (`os.replace`), so an interrupted run never leaves a truncated
  result file.
- Resources: `ResourceTracker` records wall-clock and **measured**
  `max_memory_allocated` / `max_memory_reserved` per stage (`train`, `unlearn:<method>`,
  `score:<method>`, `utility:<method>`) — not theoretical estimates. It degrades to zeros
  without torch/CUDA rather than raising, so the CPU-only path stays usable.
- Launch: `scripts/launch_all.sh` (local, GPU-aware, concurrency-throttled) and
  `scripts/slurm_run.sbatch` (one array task per run). Runs are independently launchable by
  `run_id`.
- `output_root` defaults to `runs/<experiment>` under the repository root; point it at a
  large scratch volume for full audits (`keep_all` is ~17 GiB/run).

---

## 9. Utility evaluation

`audit_tofu/utility.py` implements the standard TOFU measurements — length-normalized
probability, truth ratio, optional ROUGE-L recall (generation, so opt-in), over the retain /
forget / real-authors / world-facts splits, plus `model_utility` as the harmonic mean.

`forget_quality` is the KS-test p-value against a reference retain-only model's truth ratios.
When no reference is supplied it returns **`None` with an explanatory note** rather than
substituting a proxy. Producing it requires training a retain-only reference model, which is
not part of the 30-run budget.

Per the spec the suite runs on **evaluation runs only** (`utility.calibration_runs: false`);
calibration runs need candidate losses alone. The audit score and `ε_LB` are kept strictly
separate from forget quality — the former is the hypothesis test, the latter a
utility/behaviour metric, and a method can look good on the latter while admitting a large
`ε_LB`.

---

## 10. Validation status

Spec's required validations, items 1–10:

| # | Validation | Status |
| --- | --- | --- |
| 1 | Prompt tokens excluded from the loss | **Pass** — 7 tests; 3 more need torch |
| 2 | Every sign vector has exactly ten `+1` and ten `-1` | **Pass** — all 30 runs |
| 3 | Positive candidates in training, negatives absent | **Pass** |
| 4 | Forget loader contains only positive candidates | **Pass** — both families |
| 5 | Calibration and evaluation manifests disjoint | **Pass** — plus tamper detection |
| 6 | Attack cannot receive evaluation labels | **Pass** — signature + import introspection |
| 7 | Gaussian likelihoods, variance flooring, tie handling | **Pass** — 24 tests |
| 8 | Epsilon lower bound | **Pass** — 33 tests, incl. exact-integer Lemma 4.2 check |
| 9 | Tiny end-to-end smoke test | **Pass** — `scripts/run_smoke.py`, full pipeline, offline |
| 10 | One-run Llama-3.2-1B pilot | **Pass** — gated access granted 2026-09-06; see `docs/RESOURCE_ESTIMATE.md` §0 |

`184 passed, 1 skipped` with torch installed (torch 2.6.0+cu124, transformers 5.16.1,
8×A100 visible); the single skip is the optional LoRA fallback, which needs `peft`.

Beyond the spec's list, two further checks were run because they close real gaps:

- **Dry run verified for all 30 runs.** Each gives 3800 train examples = 3600 retain + 200
  forget, 400 candidates, 10 positive / 10 negative, all invariants passing.
- **The aggregation path is tested end to end without a GPU.**
  `tests/test_aggregate_integration.py` fabricates candidate losses with a controllable in/out
  gap, writes a realistic output tree, and drives `scripts/aggregate_audit.py` as a subprocess.
  This is how the `r`-selection bug in §7 was caught. It verifies that a strong gap reaches the
  6.589 ceiling, that **zero gap certifies nothing at every `r`**, that the Lemma 4.1 halving
  appears in the output, that all four plots render, and that mismatched `split_hash` and
  empty-input cases fail loudly rather than blending incompatible data.

The pipeline's arithmetic is therefore exercised on synthetic data across three leakage
regimes; what remains untested is only the model-dependent part (does Llama actually leak
enough after real fine-tuning and unlearning), which is precisely what the pilot answers.

### Deviations from the spec, and why

1. **`epsilon_lb` is halved relative to upstream's output.** Required by Lemma 4.1; upstream
   omits it. Both values are reported. (§2.2)
2. **`delta` defaults to `0.0`**, not upstream's `1e-5`. The paper's main test fixes `δ = 0`
   and the LDP reduction does not extend cleanly beyond it. (§2.4)
3. **`r` is the total budget**, not per-side as in the upstream caller. Matches the paper and
   this spec. (§2.1)
4. **No ungated-mirror fallback for the model**, by explicit instruction, so a run cannot
   silently audit different weights.

### Open items

- **NPO's retain/forget balance is a judgement call, not a bug.** See §13 — our
  `beta=0.1, retain_weight=1.0` reproduces OpenUnlearning's `NPO.yaml` defaults exactly, and
  the loss is verified identical to their `compute_dpo_loss`. The measured `retain_loss` rise
  (1.912 → 2.796) is therefore NPO's known utility/forgetting trade-off at `beta=0.1`, not a
  defect. Raising `alpha`/`retain_weight` would preserve utility but depart from the
  established configuration, making results less comparable to OpenUnlearning's published
  numbers. Decide deliberately; do not treat it as a fix.
- **Learning rate is CONFIRMED**, not merely adequate. OpenUnlearning's
  `configs/experiment/unlearn/tofu/default.yaml` specifies `learning_rate: 1e-5` and
  `weight_decay: 0.01` for TOFU with `Llama-3.2-1B-Instruct` — exactly our defaults. It also
  produces a clear leakage signal under `noop` (in/out gap 0.266, 11.8x the run-to-run noise
  floor). This closes the spec's "learning rate from the established TOFU configuration" item.
- **Two deliberate divergences from OpenUnlearning's TOFU config**, both required by the spec:
  they start from the released `open-unlearning/tofu_Llama-3.2-1B-Instruct_full` checkpoint
  (we must fine-tune per run, since each run has its own `S`), and they use
  `num_train_epochs: 10` with `warmup_epochs: 1.0` for unlearning where we use 5 epochs and no
  warmup. Their unlearning-phase epoch count is worth reconsidering if NPO looks under-trained.
- **TOFU forget quality** needs a retain-only reference model, not in the current budget.
- **The 30-run experiment has not been launched**, per the spec's explicit instruction to
  report measured runtime and memory first.

---

## 11. Environment notes

Verified working configuration:

| Component | Version |
| --- | --- |
| Python | 3.12.3 (`/usr/bin/python3`) |
| torch | 2.6.0+cu124 |
| transformers | 5.16.1 |
| accelerate | 1.14.0 |
| datasets | 5.0.1 |
| GPUs | 8 × A100-SXM4-80GB, SM 8.0 |

Environment-specific things that cost time and are worth recording:

**Do not build the venv on a uv-managed interpreter in a network-mounted `$HOME`.** If the
home directory becomes unreadable, `.venv/bin/python` becomes a dangling symlink even though
every package is present locally. Build the venv on the system `python3`.

**transformers v5 renamed `torch_dtype` to `dtype`.** The old name still works but warns on
every model load, and there are four loads per run across thirty runs.
`load_model_and_tokenizer` selects the keyword by major version, so it is correct on both v4
and v5.

**FlashAttention-2 is not installed, and the fallback works.** `resolve_attn_implementation`
returned `sdpa`, which the pilot confirms. SDPA is numerically fine, just slower — so measured
timings are a conservative upper bound relative to a flash-attn build. Installing `flash-attn`
would speed up training but is not required.

---

## 12. Proxy pilot

Because `meta-llama/Llama-3.2-1B-Instruct` is gated, a **proxy pilot** was run to exercise the
full 1B-scale code path and obtain measured resource numbers:
`configs/proxy_pilot.yaml`, using `Qwen/Qwen2.5-1.5B-Instruct` (already cached locally).

**This is explicitly not an audit result.** The model is not the audited model. It was chosen
because at 1.54 B parameters it is *larger* than Llama-3.2-1B's 1.24 B, so its memory and
timing figures bound the real thing from above rather than below. It writes to its own
`output_root`, and the `split_hash` check would refuse to aggregate it with real runs anyway.

*(§12 continues below; see §13 for the OpenUnlearning method survey.)*

Measured results are in `docs/RESOURCE_ESTIMATE.md` §4; the diagnostic findings, which matter
more, are in §4b. Summary: **48.3 min per run** for all three methods, **18.30 GiB** peak
allocated (NPO's frozen reference model is the ceiling), **23.69 GiB** peak reserved (the FP32
logits tensor during scoring), projecting to **24.2 GPU-hours** for the 30-run audit. Every
projection in §1–§3 was met or beaten.

The run also validated the pieces that only execute at scale: the trained checkpoint is saved
once and successfully reloaded and branched into all three methods; `npo` correctly reports
`variant: npo_rt`; all four seeds, the effective config, and per-stage resources are recorded;
and `losses.json` carries exactly one record per `(run_id, method, candidate_author, qa_id)`
with `loss` and `num_answer_tokens`, 400 per method as expected.

**Three findings from the pilot change what should happen next**, and they are set out in full
in `docs/RESOURCE_ESTIMATE.md` §4b:

1. **A one-run pilot cannot confirm the attack has signal.** Under `noop`, raw in/out losses
   separate only weakly (1.410 vs 1.578, Cohen's *d* = 0.32) because between-QA difficulty
   variance is ~5x the in/out gap. Removing that nuisance variance is exactly what the per-QA
   calibration does, so the decisive quantity is the *run-to-run* spread of a fixed QA pair —
   unmeasurable from one run. Needs 3–5 `noop`-only runs (~32 min each) before the full 30.
2. **NPO's retain term is ~20x under-weighted at `beta = 0.1`**, because the objective carries
   a `2/beta` prefactor. `retain_loss` rose from 1.912 to 2.796 during unlearning, and NPO
   degraded candidate losses on both conditions alike (~2.9 versus ~1.5 under `noop`), erasing
   the in/out separation. An `ε_LB` near zero from this configuration would show model damage,
   not sound unlearning. Raise `unlearning.npo.retain_weight` before the real launch.
3. **`retain_ft` is nearly identical to `noop`** (*d* = 0.31 vs 0.32): two epochs at
   `lr = 1e-5` do not undo five epochs of memorization.

---

## 13. OpenUnlearning method survey — what else could be audited

Surveyed against `locuslab/open-unlearning` @ main (fetched 2026-09-09). It ships **11
unlearning methods plus `finetune`**:

```
configs/trainer/   CEU  DPO  GradAscent  GradDiff  NPO  PDU  RMU
                   SatImp  SimNPO  UNDIAL  WGA  finetune
```

### Our NPO is verified identical to theirs

`src/trainer/unlearn/npo.py` calls `compute_dpo_loss(win_inputs=None, lose_inputs=forget)`,
where `compute_batch_nll` returns the **sum** of per-token NLL over answer tokens. Unrolling:

```
lose_loss      = -log pi_theta(a|q)          (sum over answer tokens)
lose_log_ratio = -(lose_loss - lose_ref_loss) = log pi_theta - log pi_ref
loss           = -(2/beta) * logsigmoid(beta * (0 - lose_log_ratio))
               =  (2/beta) * softplus(beta * (log pi_theta - log pi_ref))     [-logsigmoid(-z) = softplus(z)]
```

That is exactly `_npo_unlearn` in `audit_tofu/unlearn.py`, including the sum reduction and the
`2/beta` prefactor. Their `NPO.yaml` defaults (`beta: 0.1, alpha: 1.0, gamma: 1.0`) match ours,
where our `retain_weight` plays the role of their `alpha`.

### Mapping to the methods the paper already audited

| Paper §7.2 | OpenUnlearning equivalent | Status here |
| --- | --- | --- |
| Ascent on the forget set | `GradAscent` | **not implemented** — trivial to add |
| Pure fine-tuning on retain | `finetune` | implemented as `retain_ft` |
| Interleaved descent–ascent (IDA) | `GradDiff` (closest) | **not implemented** — easy |
| Hessian-based unlearning | — | not an LLM method; absent from OpenUnlearning |
| Model clipping, rewind-to-delete | — | likewise absent |

Adding `GradAscent` and `GradDiff` would let three of the four *heuristic* methods the paper
audited on CIFAR-100/Shakespeare be re-audited at LLM scale — a direct test of whether the
certified-vs-heuristic separation reproduces.

### Candidates ranked by implementation cost

**Tier 1 — cheap, high value.** Each reuses the existing two-stream loop in `_npo_unlearn`.

| Method | Objective | Est. new code | Notes |
| --- | --- | --- | --- |
| `GradAscent` | `-forget_loss` | ~15 lines | 5 lines in upstream. Maps to a method the paper audited. No reference model. |
| `GradDiff` | `gamma*(-forget_loss) + alpha*retain_loss` | ~30 lines | Closest to the paper's IDA. Optional KL retain term needs a reference model. |
| `SimNPO` | `-(2/beta)*logsigmoid(beta*(nll/len - delta))` + retain | ~30 lines | **Reference-free** NPO, so ~2.5 GiB cheaper and faster than NPO. Defaults `beta: 4.5, gamma: 0.125, delta: 0`. Scientifically interesting: does dropping the reference model change leakage? |

**Tier 2 — moderate (~50-80 lines).** `WGA` (token-weighted ascent, needs `compute_wga_loss`),
`UNDIAL` (self-distillation against a teacher whose target-token logit is lowered by `beta`;
needs a reference model), `CEU`, `DPO` (needs TOFU's "idk" alternate answers — extra data
plumbing, and `configs/experiment/unlearn/tofu/idk.yaml` exists upstream).

**Tier 3 — substantial.** `RMU` operates on hidden activations at a chosen layer and needs
forward hooks; `PDU` and `SatImp` are less standard. Different plumbing from everything above.

### Why adding methods is nearly free

The fine-tune dominates and is **shared** across methods within a run (one checkpoint, branched).
From the measured pilot: fine-tune 21.3 min of a 33.0 min run. Estimated marginal cost, scaling
the measured `npo` compute (2.4 min = ~1.0 min model load + ~1.4 min compute) by each method's
forward/backward count:

| Configuration | Per run | 30 runs |
| --- | --- | --- |
| current 3 methods | 33.0 min | **16.5 GPU-h** |
| + `GradAscent` | +~1.5 min | +0.8 GPU-h |
| + `GradDiff` | +~1.9 min | +1.0 GPU-h |
| + `SimNPO` | +~1.9 min | +1.0 GPU-h |
| **all 6 methods** | **~38.4 min** | **~19.2 GPU-h** |

Doubling the method count costs ~16% more compute. **Caveat:** each method needs its own
calibration distributions, since `p_in` is method-specific — but they share the fine-tune, so
`Gamma = 20` and `L = 10` still means 30 runs total, not 30 per method.

---

## 14. Making forget quality measurable: the pinned-candidate split

`forget_quality` was `None` in every run of the 30-run audit. Section 9 attributed that to
the missing reference model. That is only half the reason, and the other half is a property
of the **split** that training a reference would not have fixed.

### Why the random split makes the KS test blind

The KS test compares an unlearned model's forget-set truth ratios against a reference that
never trained on those rows. `utility.py` feeds it `results["forget"]`, and
`TOFU_UTILITY_SPLITS` hardwires `forget` to `forget10_perturbed` — **authors 180–199,
always**. Meanwhile `base.yaml` draws its 20 candidates from `split_seed`, giving
`{0, 2, 15, 32, 41, 52, 58, 62, 64, 80, 82, 106, 107, 140, 145, 155, 178, 182, 183, 195}`.

Measured against the built manifest, of the 20 authors in the KS group:

| | count | status in the audit |
| --- | --- | --- |
| also candidates | 3 (182, 183, 195) | trained then unlearned |
| in the retain set | 17 | trained in every run, **never unlearned** |

So 340 of the 400 KS-tested rows belong to authors the audit deliberately keeps. Neither
choice of reference recovers a usable metric:

* **Reference on `D_r`.** `D_r` contains those 17 authors, so the reference memorized 85% of
  the tested rows — and so did every audit model, which never unlearns them. Two
  same-trained models are compared, `p → 1` for every method including `noop`.
* **Reference on `retain90`.** Now genuinely ignorant of 180–199, so the test is valid — but
  the 17 retain authors are in every audit run's training set and absent from the reference,
  producing a large permanent gap. `p → 0` for every method, including a perfect unlearner.

One option is pinned near 1, the other near 0, and **neither moves with unlearning quality**.
Per-run it is worse than 3/20 suggests: sign vectors are balanced, so only the `+1` half of
the candidates is ever unlearned. For `eval_002` exactly **one** author (182) of the KS
group's 20 was trained-then-unlearned — 20 rows of 400.

### The fix, and why it is legitimate

`build_manifest(candidate_author_ids=...)` pins the pool. With `"180-199"`:

```
candidate pool == forget10_perturbed's authors   (the KS group IS the forget set)
D_r            == authors 0..179 == retain90     (the reference is ignorant of it)
retain_perturbed (authors 0..22) \subset D_r      (no retain contamination)
```

Verified on the built manifest (`split_hash 9e2c26ef...`): all 400 `forget10_perturbed` rows
belong to candidate authors and **0** to `D_r`; all 400 `retain_perturbed` rows are in `D_r`
with 0 candidates. `real_authors`/`world_facts` join to no TOFU author, as expected — they
are real trivia — so only their `all` slice is defined.

This does not weaken the bound. Validity requires the candidate pool to be **fixed before
any run** and the **sign vectors** i.i.d. uniform over `S_m` (paper §4). It does not require
the pool itself to be randomly drawn; a pre-registered, published split is if anything more
defensible than a seed-dependent draw. Only the pinned case writes
`split.candidate_selection`, so default manifests hash exactly as before and existing audits
stay valid.

Checked for a selection confound: authors 180–199 are not atypical. Per-author mean answer
length is 26.87 words vs 25.11 for authors 0–179 (~7% longer), KS `p = 0.43`,
Mann-Whitney `p = 0.24` at `n = 20`.

### What pinning does *not* buy

**Comparability with published TOFU numbers.** Standard TOFU fine-tunes on all 200 authors
then unlearns `forget10`; an audit run trains on `D_r` plus the `+1` candidates — 190 authors
under a balanced sign vector. Different base model, different forget-set size. This is not
fixable, because the audit *requires* the `-1` candidates to be absent from training (the
in/out contrast is the mechanism) and standard TOFU has no "out" condition at all. Pinning
buys internal coherence, not external comparability.

### Three slices, and why the control is mandatory

`scripts/forget_quality.py` runs post-hoc and CPU-only, from stored per-row truth ratios:

* `plus` — the run's `S_j = +1` authors (200 rows). The headline.
* `minus` — the run's `S_j = -1` authors (200 rows). Neither model ever saw them.
* `all` — 400 rows. The conventional TOFU number.

Both samples are always the **same rows** on both sides; comparing 200 `+1` rows against all
400 reference rows would partly measure author composition rather than forgetting.

`minus` is necessary rather than decorative, because the reference differs from each run's
model by more than the forget set: a different training seed (`train_seed_base - 1` vs
`1000 + run_index`) and ~63 fewer optimizer steps (3600 vs 3800 examples at fixed epochs).
Both move truth ratios for reasons unrelated to unlearning, and `minus` is the only way to
size that nuisance floor.

Read `minus` as a distribution, not a value: under the null the p-value is `Uniform(0,1)`, so
judge the **mean over runs and methods** (~0.5 = exchangeable). A single `minus` p of 0.1 is
uninformative. The script suppresses its verdict below 5 tests for exactly this reason.

The `all` dilution is measurable, not theoretical. On a controlled fixture where only the
`+1` authors' truth ratios were shifted, `plus` gave `KS = 1.00` while `all` gave
`KS = 0.50` — halved precisely by the untouched `-1` rows, which match the reference by
construction.

### Forward compatibility

`configs/forgetq5_pinned180.yaml` pins **20 calibration + 5 evaluation** sign vectors but
runs only the evaluation family (`FAMILIES=evaluation`). The vectors are all sampled up front
from independent RNG streams and checked disjoint, so `ε` remains available later for the
cost of running the calibration family alone — same `split_hash`, no re-training of the 5
evaluation runs. Building with `num_calibration_runs: 1` would have forced a fresh split, and
therefore a full re-run, to get `ε`.

### Retention note

`keep_all` and `keep_trained` are **behaviourally identical** today. `run_single.py`'s only
`save_pretrained` call is for `trained/`, so the unlearned per-method weights are never
written and `prune_checkpoints`' `method_dir/model` targets never exist. This config uses
`keep_trained` so a later metric costs model-load + unlearn (26 s to 433 s per method)
instead of a 19-minute re-fine-tune.
