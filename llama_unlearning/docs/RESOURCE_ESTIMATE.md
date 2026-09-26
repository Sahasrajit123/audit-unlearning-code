# Resource estimate

**Status: MEASURED on `meta-llama/Llama-3.2-1B-Instruct`.** Gated access was granted
2026-09-06 and spec validation 10 is complete. **Read §0 first** — it is the real thing.
Sections 1–3 are the original analytic projections, kept for comparison; §4 is the earlier
Qwen-1.5B proxy pilot, superseded.

---

## 0. Measured — the real pilot (validation 10)

Five runs on `meta-llama/Llama-3.2-1B-Instruct` (1,235,814,400 params), one A100-80GB each,
GPUs 0/1/2/4/7 under contention from other jobs. `calib_000` ran all three methods (the pilot);
`calib_001`–`calib_004` ran `noop` only, to measure run-to-run variance.

| Stage | n | Mean time | Max time | Peak allocated | Peak reserved |
| --- | --- | --- | --- | --- | --- |
| Fine-tune, 5 epochs | 5 | **21.3 min** | 22.4 min | **11.60 GiB** | 12.60 GiB |
| `unlearn:noop` (load only) | 5 | 59.5 s | 88.9 s | 2.32 GiB | 2.32 GiB |
| `unlearn:npo` | 1 | **2.4 min** | — | **14.95 GiB** | 15.36 GiB |
| `unlearn:retain_ft` | 1 | **8.2 min** | — | 11.59 GiB | 13.09 GiB |
| `score:*` (400 QA pairs) | 7 | **2.6–2.9 s** | 2.9 s | 4.43 GiB | 14.57 GiB |
| **Per run, all 3 methods** | | **33.0 min** | | **14.95 GiB** | **15.36 GiB** |

Trained checkpoint on disk: **2.32 GiB**. Attention backend resolved to **sdpa** (flash-attn
not installed), so these timings are a conservative upper bound.

### Cost of the full 30-run audit

**16.5 GPU-hours** — ~4.1 h wall-clock on 4 GPUs, ~2.1 h on 8. Comfortably below the original
24–72 GPU-hour range, and below the 24.2 h the Qwen-1.5B proxy suggested.

Fits easily in one 40 GB GPU as the spec anticipated: the ceiling is **14.95 GiB**, set by NPO's
frozen reference model. The memory-saving 8-bit mode is not needed at this scale.

### Disk

| Retention | 30 runs |
| --- | --- |
| `keep_all` | 30 x 3 x 2.32 GiB ≈ 209 GiB |
| `keep_trained` | 30 x 2.32 GiB ≈ 70 GiB |
| `delete_after_scoring` (default) | **< 1 GiB** — JSON only |

---

## 0b. Does the attack have signal? YES — measured

This was the open question the proxy pilot could not answer. Five `noop` runs settle it.

### Variance decomposition (400 QA pairs, 1260 residual dof)

| Quantity | Value | Role |
| --- | --- | --- |
| `sigma_between` — spread of per-QA mean loss across QA pairs | **0.5374** | nuisance; calibrated away |
| `sigma_within` — one QA pair's loss across runs | **0.0225** | the real noise floor |
| ratio | **23.9x** | |
| `gap_z` = mean(`mu_out` − `mu_in`) | **0.2660 ± 0.0086** | the signal |

### Effect sizes

| | |
| --- | --- |
| `d_qa` (per QA pair, calibrated) | **11.83** |
| `d_uncalibrated` (single run, raw loss) | **0.495** |
| **calibration gain** | **107x** |

These come from the 5 `noop` runs available on 2026-09-06. They are a **snapshot**: the same
estimator on 9 runs gives `d_qa = 11.85`, `d_uncalibrated = 0.500`, gain 106x. Recompute with
`scripts/analyze_signal.py --config configs/base.yaml --method noop` rather than quoting these
figures; the conclusion (calibration is worth ~100x) is what is stable.

This is the crux, and it vindicates the per-QA calibration design. A naive attack on raw losses
sees only *d* = 0.50 — weak, exactly the *d* = 0.32 the Qwen proxy showed, which is why one run
looked discouraging. But the in/out gap is **11.8x the run-to-run noise**; it merely sits under
a between-QA difficulty spread 23.9x larger. Conditioning on the QA pair removes that nuisance
entirely.

### Empirical leave-one-out attack (the real check)

Fitting calibration on 4 runs and predicting the held-out 5th, labels revealed only afterwards:

| `r` | overlap `V` per run | mean | chance | `ε_LB` |
| --- | --- | --- | --- | --- |
| 4 | 4, 4, 4, 4, 4 | 4.00 / 4 | 2 | 1.033 |
| 8 | 8, 8, 8, 8, 8 | 8.00 / 8 | 4 | 2.350 |
| 12 | 12, 12, 12, 12, 12 | 12.00 / 12 | 6 | 3.644 |
| 16 | 15, 16, 16, 15, 15 | 15.40 / 16 | 8 | **3.691** |
| 20 | 18, 18, 16, 18, 18 | 17.60 / 20 | 10 | 2.746 |

**Perfect classification through `r = 12`**, and the paper's **inverted U in `r` (Remark 4.3)
is clearly reproduced** — `ε_LB` rises to a peak at `r = 16` then falls at `r = 20` as the
auditor is forced onto low-confidence authors.

Two reasons the real audit should do better than this table: it uses `Γ = 20` calibration runs
rather than 4, and `L = 10` evaluation runs rather than 5 (more runs tighten the bound).

Caveat on the projection in `scripts/analyze_signal.py`: it predicted `E[V] = r` at *every* `r`,
i.e. full saturation, by assuming per-QA noise is independent across an author's 20 QA pairs.
The measured degradation at `r = 16, 20` shows those QA pairs are in fact **correlated**, so the
`sqrt(n_qa)` gain is not fully realised. The script states this caveat; trust the empirical
table over the simulation.

**This LOO exercise is a validation, not an audit**: it reuses calibration runs as
pseudo-evaluation, which violates the independence the bound requires. The real audit uses the
disjoint `eval_*` family.

### Verdict: GO

The attack is strong, `noop` leaks clearly as the positive control demands, and the full audit
costs 16.5 GPU-hours. See §0c for the one change to make first.

---

## 0c. NPO's retain/forget balance — a judgement call, not a bug

**Correction to an earlier version of this document**, which said the retain weight had to be
raised "before launching" as though our configuration were wrong. It is not.

Observed during unlearning:

```
epoch 1  npo_loss=7.7620  retain_loss=1.9120
epoch 5  npo_loss=0.6180  retain_loss=2.7958   <-- retain loss got WORSE
```

The mechanism is real: the NPO objective carries a `2/beta` prefactor, so at `beta = 0.1` it
enters at scale ~13.9 at parity (`(2/0.1)·log 2`) against a retain cross-entropy of ~2 at
`retain_weight: 1.0`. The forget term dominates by roughly an order of magnitude.

But this **is** the established configuration. OpenUnlearning's `configs/trainer/NPO.yaml`
specifies `beta: 0.1, alpha: 1.0, gamma: 1.0`, and our loss is verified line-for-line identical
to their `compute_dpo_loss` (see `docs/IMPLEMENTATION.md` §13). So the retain degradation is
NPO's documented utility/forgetting trade-off at `beta = 0.1`, not a defect we introduced.

What that means for the audit:

- Leaving it as-is audits **NPO as it is actually configured and published**. That is the
  defensible default, and the resulting `ε_LB` is a statement about that method.
- Raising `retain_weight` to ≈10–20 would preserve utility, but the result would no longer be
  comparable to OpenUnlearning's numbers, and it becomes "NPO with a hyperparameter we chose".
- Either way, **report the retain loss trajectory alongside `ε_LB`**. A small `ε_LB` from a
  model whose retain loss rose 46% is weak evidence of good unlearning; it may just be a
  damaged model. The two numbers must be read together.

Recommendation: run the audit at the published defaults, and additionally at
`retain_weight: 10` as a second configuration if budget allows. The marginal cost is small
(§0d) because the fine-tune is shared.

---

## 0d. Marginal cost of adding methods or configurations

The fine-tune dominates each run and is **shared** — one checkpoint per run, branched into every
method. From the measured pilot, fine-tune is 21.3 min of a 33.0 min run, so extra methods only
pay their own unlearn + score time.

Decomposing the measured `unlearn` timings: `noop` at 59.5 s is essentially pure model load, so
`npo`'s 2.4 min is ~1.0 min load + ~1.4 min compute. Scaling that compute by each method's
forward/backward count per micro-step:

| Configuration | Per run | 30 runs | vs baseline |
| --- | --- | --- | --- |
| **current: noop + npo + retain_ft** | **33.0 min** | **16.5 GPU-h** | — |
| + `GradAscent` (no ref model, no retain) | +~1.5 min | 17.3 GPU-h | +5% |
| + `GradDiff` (no ref model) | +~1.9 min | 18.3 GPU-h | +11% |
| + `SimNPO` (reference-free) | +~1.9 min | 19.2 GPU-h | +16% |
| + a second `npo` at `retain_weight: 10` | +~2.4 min | 20.4 GPU-h | +24% |

So doubling the method count costs ~16% more compute, and a second NPO configuration is ~1.2
GPU-h. On 4 GPUs the all-six-methods run is ~4.8 h wall-clock versus ~4.1 h for three.

Memory is unaffected: `npo`'s frozen reference model already sets the 14.95 GiB ceiling, and
`GradAscent` / `GradDiff` / `SimNPO` need no reference model, so they run *below* it.

Two things this does **not** buy cheaply:

- **Each method needs its own calibration**, since `p_in` is method-specific. That is already
  accounted for above — the methods share the 30 runs, so it is still `Γ = 20` + `L = 10` total,
  not per method.
- **Changing the base fine-tune** (epochs, LR, `m`, `batching: qa`) invalidates every
  checkpoint and costs a full re-run, ~16.5 GPU-h. Decide those before launching.

---

## 0e. Measured: `grad_ascent` and `grad_diff` (added from OpenUnlearning)

Both ran on `calib_000` by reusing its existing fine-tuned checkpoint — no re-training, which
is the point of the branch design. Measured on one A100-80GB:

| Method | Wall-clock | Peak allocated | Reference model? |
| --- | --- | --- | --- |
| `grad_ascent`, 2 epochs | **50.4 s** | **11.53 GiB** | no |
| `grad_diff`, 5 epochs | **103.1 s** | **12.65 GiB** | no |

Both sit **below** NPO's 14.95 GiB ceiling exactly as predicted, since neither needs a frozen
reference copy. Revised full-audit cost with all five methods: **~35.6 min per run → ~17.8
GPU-hours for 30 runs**, only **+8%** over the three-method 16.5 GPU-h (better than the +16%
estimated in §0d).

### The methods behave as specified

```
grad_ascent  forget_nll  1.28 -> 4.26 -> 10.84      retain_loss 0.00 (forget-only, by design)
grad_diff    forget_nll  1.38 -> ... -> 3.07        retain_loss 1.44 -> 2.15
```

`grad_ascent` runs away, as an unbounded objective must; `grad_diff`'s retain term holds it to a
gentle rise. That contrast is the whole reason both are worth auditing.

### In/out separation under each method (single run, raw losses)

| Method | mean_in | mean_out | gap | Cohen *d* | Reading |
| --- | --- | --- | --- | --- | --- |
| `noop` | 1.2784 | 1.5975 | **+0.319** | +0.60 | positive control: clear leakage |
| `retain_ft` | 1.2977 | 1.6161 | **+0.318** | +0.58 | ≈ identical to `noop`; weak heuristic |
| `npo` | 2.9655 | 2.6389 | **−0.327** | −0.41 | **over-forgets** (see below) |
| `grad_diff` | 3.0866 | 2.6284 | **−0.458** | −0.42 | over-forgets, most strongly |
| `grad_ascent` | 11.7537 | 11.9189 | +0.165 | +0.05 | near-chance, but model destroyed |

**A negative gap is still leakage.** `npo` and `grad_diff` push the forget set's loss *above*
the never-trained baseline — they over-forget, leaving a signature in the opposite direction.
The audit does not care about the sign: the attack fits per-QA in/out Gaussians and scores a
likelihood ratio, so it detects "systematically higher" exactly as well as "systematically
lower". This is precisely why calibration beats thresholding, and it means an over-forgetting
method is **not** safe from the audit.

**`grad_ascent` is the interesting case.** Its *d* = 0.05 is near chance, so it may yield a
small `ε_LB` — but its candidate losses are ~11.8 versus ~1.3 under `noop`. The model is
destroyed. A small `ε_LB` here would certify nothing useful about unlearning; it would just say
the model no longer knows anything. This is the clearest illustration of why the audit's
`ε_LB` must be read next to the utility metrics, and it is why `diverged` and `final_forget_nll`
are now recorded in every method's `metrics.json`.

Caveat: these are single-run *raw* gaps. The real attack calibrates per QA pair across `Γ = 20`
runs, which removed a 23.9x nuisance variance for `noop` (§0b) and should likewise sharpen these.
Treat the directions and relative magnitudes as informative, the |*d*| values as lower bounds.

---

## 0e2. MEASURED: `simnpo` — saturated at upstream's `beta = 4.5`

Added from OpenUnlearning's `SimNPO` (reference-free NPO) and branched off `calib_000`'s
existing checkpoint.

| | measured |
| --- | --- |
| wall-clock, 5 epochs | **111.3 s** |
| peak allocated | **12.65 GiB** (vs NPO's 14.95 — no reference model) |

Cheaper than NPO in both time and memory, exactly as expected.

### But at upstream's defaults its forget term does essentially nothing

```
[simnpo] epoch 1/5  forget_loss=0.0016  forget_nll=1.2984  retain_loss=1.3819
[simnpo] epoch 5/5  forget_loss=0.0014  forget_nll=1.3060  retain_loss=1.3651
```

The forget loss sits at ~0.0015 and the forget NLL is **flat** across five epochs. The
objective is deep in saturation. The arithmetic:

```
L        = -(2/beta) * logsigmoid(beta * (nll - delta))
dL/d(nll) = -2 * sigmoid(-beta * (nll - delta))
```

At Llama's measured TOFU forget NLL of ~1.28 with `beta = 4.5`:

| `beta` | `|dL/d(nll)|` at nll=1.28 | after `gamma = 0.125` |
| --- | --- | --- |
| 0.1 | 0.9361 | 0.1170 |
| 1.0 | 0.4351 | 0.0544 |
| 2.0 | 0.1435 | 0.0179 |
| **4.5 (upstream)** | **0.0063** | **0.00079** |

The retain term enters at `alpha = 1.0` with an O(1) gradient, so the forget signal is
roughly **three orders of magnitude weaker**. SimNPO at published defaults is, on this
setup, almost exactly "fine-tune on the retain set".

The candidate losses confirm it — `simnpo` is nearly indistinguishable from `noop`:

| method | mean_in | mean_out | gap | Cohen *d* |
| --- | --- | --- | --- | --- |
| `noop` (control) | 1.2784 | 1.5975 | +0.3191 | 0.60 |
| **`simnpo`** | **1.3062** | **1.6037** | **+0.2975** | **0.56** |
| `retain_ft` | 1.2977 | 1.6161 | +0.3184 | 0.58 |
| `npo` | 2.9655 | 2.6389 | −0.3266 | −0.41 |
| `grad_diff` | 3.0866 | 2.6284 | −0.4582 | −0.42 |
| `grad_ascent` | 11.7537 | 11.9189 | +0.1652 | 0.05 |

`simnpo`'s separation (0.56) is within noise of `noop`'s (0.60) and matches `retain_ft`'s
(0.58) — three ways of saying the forget term never engaged.

### What to do about it

This is upstream's published configuration, so the same reasoning as §0c applies: running
it as-is audits **SimNPO as configured**, and that is the defensible default. But unlike the
NPO case, the result here is close to vacuous — a large `ε_LB` would say "retain
fine-tuning does not remove leakage", which `retain_ft` already tells us.

`beta` interacts with the model's loss scale, and TOFU's operating point (nll ≈ 1.3) is
well past where `beta = 4.5` has slope. If SimNPO is meant to be informative here, run a
second configuration at `beta` in 0.5–1.0 — cheap, since the fine-tune is shared (§0d).
Either way, report `final_forget_nll` next to `ε_LB`: it is recorded in every method's
`metrics.json` precisely so a saturated no-op cannot be mistaken for successful unlearning.

Pinned by `test_simnpo_saturates_at_upstream_beta_on_a_high_loss_model` and
`test_simnpo_gradient_magnitude_is_documented`.

---

## 0f. MEASURED: which audit batch size `B`? — `B = 1` wins by 18.6x

Five `noop` runs on a `B = 1` (`m = 400`) manifest, alongside the five already run at
`B = 20`. Same model, same seeds, same 3800-example training sets — only the audit-side
partition of the 400-pair candidate pool differs.

### Realized bounds


Realized bounds, leave-one-out over the five runs (`scripts/compare_batching.py`):

| config | `B` | `m` | best `r` | **realized `ε_LB`** | ceiling | % of ceiling |
| --- | --- | --- | --- | --- | --- | --- |
| `base.yaml` | 20 | 20 | 16 | **3.691** | 6.16 | 60% |
| `qa_level.yaml` | 1 | 400 | 400 | **68.792** | 137.12 | 50% |

**18.6x in favour of `B = 1`.** And 68.8 lands squarely in the range the paper reports
for heuristic methods (50–60+), which `m = 20` cannot reach at all — its ceiling is 6.59.

Per-`r` detail at `B = 1` shows why: accuracy stays high deep into the sweep
(`V/r` = 99% at `r = 40`, 98% at `r = 200`, 89% at `r = 400`) while the attainable bound
keeps climbing, so epsilon rises monotonically to the `r = m` boundary rather than peaking
in the interior. At `B = 20` the inverted U peaks early, at `r = 16`.

Two reasons this understates the real audit: the LOO fits on 4 calibration runs rather
than `Γ = 20`, and at `B = 1` with `pool_variance: null` each per-QA variance comes from
only ~2 observations.

### Recommendation

**Audit the heuristic methods at `B = 1` (`configs/qa_level.yaml`).** This is also what
Remark 4.3 prescribes — heuristics leak strongly per batch and prefer small `B`. Keep
`m = 20` for the spec-faithful result and for any future certified method, where large
`B` is required.

Cost: a separate manifest, so a separate 30 runs (~17.8 GPU-h). Running both is ~35.6
GPU-h and yields a spec-faithful number plus one directly comparable to the paper.

---

## Original projections (superseded by §0, kept for comparison)

---

## 1. Memory projection — full-parameter BF16

Llama-3.2-1B-Instruct has 1.236 B parameters.

| Component | Precision | Size |
| --- | --- | --- |
| Weights | BF16 | 2.30 GiB |
| Gradients | BF16 | 2.30 GiB |
| AdamW exp_avg | FP32 | 4.60 GiB |
| AdamW exp_avg_sq | FP32 | 4.60 GiB |
| **Optimizer subtotal** | | **13.80 GiB** |
| Activations, seq 512, micro-batch 2, grad ckpt on | BF16 | ~0.6–1.5 GiB |
| Logits, `2 × 512 × 128256`, FP32 for the loss | FP32 | ~0.5 GiB |
| CUDA context, fragmentation, allocator slack | | ~1–3 GiB |
| **Projected peak** | | **~16–19 GiB** |

The vocabulary is large (128,256), so the FP32 logits tensor is a real cost at scoring time as
well. `scoring.batch_size: 8` implies `8 × 512 × 128256 × 4 B ≈ 2.1 GiB` for logits alone —
reduce it first if scoring OOMs.

### NPO needs a second model

NPO keeps a frozen reference copy `π_ref` (`copy.deepcopy` of the trained model, no gradients):

| | Extra |
| --- | --- |
| Reference weights, BF16 | +2.30 GiB |
| **Projected NPO peak** | **~19–22 GiB** |

### Memory-saving mode

`configs/memory_saving.yaml` swaps the two FP32 moment buffers (9.20 GiB) for 8-bit
(2.30 GiB) and drops micro-batch to 1 while keeping the effective batch at 16:

| | Projected peak |
| --- | --- |
| Default (FP32 AdamW) | ~16–19 GiB |
| 8-bit optimizer | **~9–12 GiB** |

Requires `bitsandbytes`. If it is missing, `build_optimizer` falls back to FP32 AdamW and prints
a **warning** — deliberately loud, because a silent fallback would invalidate exactly the
memory numbers a sizing decision rests on.

### Fit on shared GPUs

Measured on 8 × A100-80GB shared with other jobs. The projected ~16–22 GiB fits comfortably,
but **check free memory before launching** on a shared machine.

---

## 2. Runtime projection

Per run, per epoch: 3800 examples ÷ micro-batch 2 = 1900 forward/backward steps.

TOFU answers are short; at `max_seq_length` 512 most sequences are well under the cap. Assuming
0.10–0.25 s per micro-step for a 1B model in BF16 with gradient checkpointing on an A100
(checkpointing costs roughly a 30% throughput penalty in exchange for the activation savings):

| Stage | Work | Projected |
| --- | --- | --- |
| Fine-tune | 5 epochs × 1900 steps | **25–80 min** |
| `noop` | none | ~0 s |
| `npo` | 5 epochs × 100 forget steps, ×2 fwd (policy + ref) + retain | **5–15 min** |
| `retain_ft` | 2 epochs × 1800 steps | **10–30 min** |
| Scoring | 400 examples, forward only | **< 1 min** |
| Utility suite | 4 splits × ~400 rows × (1 + #perturbed) forwards | **10–40 min** |

**Per-run totals (excluding utility):**

| Runs | Methods | Projected wall-clock |
| --- | --- | --- |
| 1 calibration run | `noop` only (pilot) | **~30–80 min** |
| 1 calibration run | all three | **~45–130 min** |
| 20 calibration | all three | **15–43 GPU-hours** |
| 10 evaluation | all three + utility | **9–29 GPU-hours** |
| **Full 30-run audit** | | **~24–72 GPU-hours** |

On 4 GPUs at 1 run each that is roughly **6–18 hours wall-clock**; on 8 GPUs, **3–9 hours**.
The range is wide because per-step time is the unmeasured quantity — the pilot collapses it.

Cheaper option: calibration runs only need `noop` candidate losses if you are auditing `noop`,
so running one method instead of three cuts calibration cost by roughly half. But each audited
method needs its **own** calibration distributions, since `π_in` is method-specific. Auditing
all three methods properly means all three run on all 30 runs.

---

## 3. Disk projection

A BF16 1B checkpoint is ~2.3 GiB (weights) plus tokenizer files.

| Retention | Peak disk |
| --- | --- |
| `keep_all` | 30 runs × (1 trained + 2 unlearned) × 2.3 GiB ≈ **207 GiB** |
| `keep_trained` | 30 × 2.3 GiB ≈ **69 GiB** |
| `delete_after_scoring` (default) | **< 1 GiB** — JSON results only |

JSON results are tiny: 400 loss records per (run, method) is ~60 KB, so ~5 MB for the whole
audit.

`output_root` defaults to `runs/<experiment>` under the repository root. For `keep_all`,
point it (or symlink `runs/`) at a volume with enough free space.

---

## 4. Measured proxy-pilot results

**Run: `configs/proxy_pilot.yaml`, `Qwen/Qwen2.5-1.5B-Instruct`, one A100-80GB (shared,
already ~90% busy with other jobs), 2026-09-04.** Not the audited model — see
`docs/IMPLEMENTATION.md` §12. At 1.54 B parameters it is larger than Llama-3.2-1B's 1.24 B, so
these figures bound the real thing from above.

| Stage | Wall-clock | Peak allocated | Peak reserved |
| --- | --- | --- | --- |
| Fine-tune, 5 epochs, 1190 opt steps | **31.9 min** | **14.48 GiB** | 16.21 GiB |
| `unlearn:noop` (model load only) | 28.7 s | 2.89 GiB | 3.08 GiB |
| `unlearn:npo`, 5 epochs | **3.5 min** | **18.30 GiB** | 19.63 GiB |
| `unlearn:retain_ft`, 2 epochs | **12.3 min** | 14.47 GiB | 16.64 GiB |
| `score:*`, 400 QA pairs each | **3.0 s** | 5.27 GiB | 23.69 GiB |
| **Per-run total (all 3 methods)** | **48.3 min** | **18.30 GiB** | **23.69 GiB** |

Trained checkpoint on disk: **2.89 GiB**.

### Projections vs. measurements

| Quantity | Projected | Measured | Verdict |
| --- | --- | --- | --- |
| Fine-tune wall-clock | 25–80 min | 31.9 min | in range |
| Fine-tune peak allocated | 16–19 GiB | 14.48 GiB | better than projected |
| NPO peak allocated | 19–22 GiB | 18.30 GiB | better than projected |
| `retain_ft` wall-clock | 10–30 min | 12.3 min | in range |
| Scoring wall-clock | < 1 min | 3.0 s | far better |
| Per-run, all methods | 45–130 min | 48.3 min | low end |
| Attention backend | flash or sdpa | **sdpa** | flash-attn not installed |

### Cost of the full 30-run audit

At the observed 48.3 min per run: **24.2 GPU-hours**, i.e. ~6.0 h wall-clock on 4 GPUs or
~3.0 h on 8. That is the *low end* of the original 24–72 GPU-hour range. Treat it as a floor:
these GPUs are shared, and the measurement was already taken under contention.

NPO is the memory ceiling (18.30 GiB) because of its frozen reference model, and `score:*`
sets the *reserved* ceiling (23.69 GiB) from the FP32 logits tensor at `batch_size: 8`. Reduce
`scoring.batch_size` first if anything OOMs.

---

## 4b. Diagnostic findings from the pilot — read before launching

Three things the pilot surfaced that matter more than the timings.

### The `noop` control separates only weakly, and one run cannot settle it

Per-QA answer losses under `noop`, split by the true sign:

| | mean loss | author-level range |
| --- | --- | --- |
| `S_j = +1` (trained, then noop) | 1.4101 | [1.084, 1.932] |
| `S_j = -1` (never trained) | 1.5783 | [1.157, 1.869] |
| gap | **0.168** | Cohen's *d* = **0.32** (small) |

The gap has the right sign — trained-on candidates score lower — but the author-level ranges
overlap almost entirely.

**This is not yet evidence the audit will fail.** The spread *between* QA pairs (author means
span 1.08–1.93, a range of ~0.85) is roughly five times the in/out gap, and removing exactly
that nuisance variance is what the per-QA calibration is for: the attack scores
`λ_z = log p_in(l_z) − log p_out(l_z)` against distributions fitted *per QA pair*, not against
a global mean. The quantity that determines the attack's power is the **run-to-run** spread of
a fixed QA pair's loss, and that cannot be estimated from a single run.

**Consequence: a one-run pilot cannot confirm the attack has signal.** Confirming it needs at
least 3–5 runs with differing sign vectors, enough to estimate σ_in and σ_out per QA pair. The
cheap version is `--methods noop` only, which skips NPO and `retain_ft` and costs ~32 min per
run. **Do this before committing to the full 30.**

### NPO's retain term is roughly 20x under-weighted at `beta = 0.1`

Observed during unlearning:

```
epoch 1  npo_loss=7.7620  retain_loss=1.9120
epoch 5  npo_loss=0.6180  retain_loss=2.7958   <-- retain loss got WORSE
```

The NPO objective carries a `2/beta` prefactor, so at `beta = 0.1` it enters at scale ~20
(indeed ~13.9 at parity, `(2/0.1)·log 2`), while the retain cross-entropy enters at scale ~2
with `retain_weight: 1.0`. The forget term therefore dominates by more than an order of
magnitude, and retain performance degrades instead of being preserved — which defeats the point
of the `NPO_RT` variant.

The damage is visible in the candidate losses too: NPO raised the mean loss to ~2.9 on **both**
conditions (`+1`: 2.912, `-1`: 2.867) versus ~1.5 under `noop`, i.e. it degraded the model
broadly rather than selectively, and destroyed the in/out separation entirely (*d* = −0.06).

An `ε_LB` of zero from *this* configuration would be uninformative: it would show the model was
damaged, not that unlearning was sound. Before the real launch, raise `unlearning.npo.retain_weight`
(≈10–20 is the scale implied by the prefactor) or reduce the NPO scale, and check that
`retain_loss` stays flat or falls across epochs. Both are already exposed in YAML.

### `retain_ft` is nearly indistinguishable from `noop`

`retain_ft` gave mean_in 1.4095 / mean_out 1.5740 (*d* = 0.31) versus `noop`'s 1.4101 / 1.5783
(*d* = 0.32) — essentially unchanged. Two epochs of retain-set fine-tuning at `lr = 1e-5` does
not undo five epochs of memorization. That is a plausible and interesting finding (it is the
kind of weak heuristic the paper reports large bounds for), not obviously a bug, but it does
mean `retain_ft` and `noop` may yield similar bounds.

---

## 4c. Template for the real Llama-3.2-1B pilot

Populate from `<output_root>/runs/calib_000/metrics.json` after:

```bash
.venv/bin/python scripts/run_single.py --config configs/pilot.yaml --run_id calib_000 --gpu 0
```

The `resources` block reports per-stage duration and measured peak memory.

| Quantity | Projected | **Measured** |
| --- | --- | --- |
| Fine-tune wall-clock | 25–80 min | _TBD_ |
| Fine-tune peak allocated | 16–19 GiB | _TBD_ |
| Fine-tune peak reserved | — | _TBD_ |
| Attention backend resolved | flash_attention_2 or sdpa | _TBD_ |
| `npo` wall-clock | 5–15 min | _TBD_ |
| `npo` peak allocated | 19–22 GiB | _TBD_ |
| `retain_ft` wall-clock | 10–30 min | _TBD_ |
| Scoring wall-clock (400 ex) | < 1 min | _TBD_ |
| Scoring peak allocated | — | _TBD_ |
| Utility suite wall-clock | 10–40 min | _TBD_ |
| Trained checkpoint on disk | 2.3 GiB | _TBD_ |
| Final training loss | — | _TBD_ |

### The check that actually gates the launch

Beyond cost, the pilot must confirm the audit has signal at all. After one run:

```
mean answer loss on S_j = +1 candidates   (trained, then noop-"unlearned")
mean answer loss on S_j = -1 candidates   (never trained)
```

Under `noop` the first should be **clearly lower** than the second. If it is not, do not proceed
to NPO — diagnose in this order:

1. **Loss masking** — is `num_answer_tokens` plausible, and are prompt tokens excluded?
   (`tests/test_loss_masking.py` covers this, but on a stub tokenizer.)
2. **Training exposure** — 5 epochs at `lr = 1e-5` may under-memorize 20 QA pairs per author.
   Raising the learning rate or epoch count is the first lever; the LR is the least-confirmed
   hyperparameter in the config.
3. **Aggregation** — are the per-QA `λ_z` values being summed over the right author?

`scripts/aggregate_audit.py` prints this warning automatically when `noop` yields no bound, and
`plot_calibration_distributions` renders `μ_out − μ_in` per QA pair — the median of that
distribution should be clearly positive.

---

## 5. Interpreting the outcome before spending the compute

Worth internalizing before the launch: at `m = 20, L = 10, ζ = 0.05`, **a perfect attack
certifies `ε_LB ≈ 6.59`** and nothing higher. The full ceiling table is in
`docs/IMPLEMENTATION.md` §4.1.

So the expected results are ordered, not absolute:

- `noop` should approach the ceiling — it is the positive control.
- `retain_ft` and `npo` should fall below it, and the *gap* is the audit's finding.
- Any of them yielding `None` means overlap indistinguishable from chance at this confidence.

If a bound numerically comparable to the paper's 50–60+ is wanted, that requires `m = 400`
(`split.batching: qa`), which is supported but is a different experiment with its own
calibration cost. Deciding this **before** the 30-run launch avoids paying twice.
