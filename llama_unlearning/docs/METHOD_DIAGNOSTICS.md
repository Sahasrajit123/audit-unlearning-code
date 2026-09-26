# Per-method diagnostics (`scripts/compare_methods.py`)

Reports, for every method with results, the raw in/out separation on the candidate pool
**alongside how badly the model was damaged getting there**. Both are needed: a method can
score well on the first for entirely the wrong reason.

```bash
python scripts/compare_methods.py --config configs/base.yaml
python scripts/compare_methods.py --config configs/base.yaml --run_id calib_000
python scripts/compare_methods.py --config configs/base.yaml --out diag.json
```

This is a **diagnostic, not the audit**. It uses single-run raw losses. The audit's `ε_LB`
comes from `scripts/aggregate_audit.py`, which uses per-QA calibrated likelihood ratios over
independent evaluation runs — and calibration is worth ~107x on this data
(`docs/RESOURCE_ESTIMATE.md` §0b), so the `d` values here badly understate the real attack.
Use them to rank methods against each other, never to predict epsilon.

---

## Columns

| Column | Meaning |
| --- | --- |
| `mean_in` | mean answer loss on candidates trained on, then unlearned |
| `mean_out` | mean answer loss on candidates never trained on |
| `gap` | `mean_out − mean_in`. Positive = leakage remains. **Negative = the method over-forgot** |
| `d` | Cohen's *d* for the gap. Raw, single-run — see the caveat above |
| `%unif` | `mean loss / ln(vocab_size)`. **100% = the model has collapsed to noise** |
| `fgt_nll` | final forget-set NLL during unlearning |
| `retain` | final retain loss during unlearning |
| `div` | the `diverged` flag |

### Why `%unif` is the column that matters

A near-zero `d` has two opposite explanations:

1. the method genuinely removed the leakage, or
2. it destroyed the model, so there is no signal left **anywhere**.

Those are indistinguishable in `gap` and `d`, and trivially distinguished by `%unif`. A model
emitting a uniform distribution over its vocabulary scores exactly `ln(V)` — **11.762** for
Llama-3.2's 128,256 tokens. So `%unif ≈ 100` means the model predicts nothing at all.

---

## Measured output (`calib_000`, all six methods)

```
method         n  mean_in  mean_out      gap      d  %unif  fgt_nll  retain  div
--------------------------------------------------------------------------------
noop           5   1.2850    1.5849  +0.2999   0.56    12%        -       -    -
grad_diff      1   3.0866    2.6284  -0.4582  -0.42    24%     3.07    2.15    -
npo            1   2.9655    2.6389  -0.3266  -0.41    24%        -    2.32    -
retain_ft      1   1.2977    1.6161  +0.3184   0.58    12%        -       -    -
simnpo         1   1.3062    1.6037  +0.2975   0.56    12%     1.31    1.37    -
grad_ascent    1  11.7537   11.9189  +0.1652   0.05   101%    10.84    0.00 yes*
```

Three groups fall out, and the script flags each:

**Untouched leakage (12% of uniform, gap ≈ control's).** `retain_ft` (106% of the control's
|gap|) and `simnpo` (99%) left the leakage essentially as it was. For `simnpo` the cause is
known and quantified: its objective is saturated at upstream's `beta = 4.5`
(`docs/RESOURCE_ESTIMATE.md` §0e2).

**Over-forgetting (24% of uniform, NEGATIVE gap).** `npo` (−0.327) and `grad_diff` (−0.458)
pushed trained-on candidates *above* never-seen ones. They roughly doubled the loss — real
damage, but the model still functions. **This is still fully detectable**: the attack fits
per-QA in/out Gaussians and takes a likelihood ratio, so "systematically higher" is as
visible as "systematically lower". Over-forgetting is not safety from this audit. These two
should produce the substantive results.

**Collapse (101% of uniform).** `grad_ascent` reached `d = 0.05`, near chance — but at 101%
of the uniform-output loss, on **both** conditions. Its forget NLL went 1.28 → 10.84 in two
epochs; `L = -forget_nll` is unbounded below, so descent drives the model to a random-token
generator. Even the never-trained candidates sit at 11.92, so the damage is global, not
targeted. A small `ε_LB` here would read as "excellent unlearning" and mean the opposite.
That is faithful to upstream (`GradAscent` is exactly `-outputs.loss`, no retain term, and
they run 10 epochs where we default to 2), and it is the sharpest illustration of why `ε_LB`
must be read beside the utility metrics.

---

## A bug this script found

`grad_ascent` above shows `div: yes*`, not `YES` — and that asterisk is the bug's fingerprint.
`yes*` means collapse was inferred from `%uniform`, while the value the run actually *stored*
for `diverged` is `False`.

The guard was `DIVERGENCE_NLL = 20.0`, an absolute forget-NLL threshold. But loss saturates
at `ln(V) = 11.76`, so **20.0 was unreachable** — a completely collapsed model at 10.84 was
never flagged. The guard could not fire on any vocabulary smaller than `e^20 ≈ 4.9 × 10^8`
tokens.

Now relative:

```python
DIVERGENCE_FRACTION_OF_UNIFORM = 0.6

def divergence_threshold(model):
    return 0.6 * log(model.config.vocab_size)     # 7.06 for Llama-3.2
```

| model | `ln(V)` | threshold |
| --- | --- | --- |
| Llama-3.2 (V=128,256) | 11.76 | **7.06** |
| tiny test model (V=32,000) | 10.37 | 6.22 |
| no readable config | — | 7.00 (fallback) |

Correctly separates the measured cases: `grad_ascent` 10.84 fires; `grad_diff` 3.07 and
`simnpo` 1.31 do not. Runs completed *before* the fix keep their stored `False`, which is
why `compare_methods.py` also derives collapse from `%uniform` — that criterion needs no
stored state and so is immune to the same class of error. The threshold is now recorded per run as `divergence_threshold` in
`metrics.json`, so the flag is interpretable after the fact. Pinned by
`test_divergence_threshold_is_relative_to_vocab_and_reachable` and
`test_divergence_threshold_scales_with_vocab_and_has_a_fallback`.
