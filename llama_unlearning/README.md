# Unlearning audit on TOFU with Llama-3.2-1B-Instruct

An LLM-scale instantiation of the repeated-run unlearning auditor from the accompanying
paper (anonymous submission).

The attacker predicts, from the losses of the *final unlearned model alone*, whether each
candidate author was present during fine-tuning and then unlearned (`S_j = +1`) or never
present (`S_j = -1`). Overlap statistics from held-out evaluation runs are converted into a
lower bound `ε_LB` on the unlearning parameter via the paper's Lemma 4.2, at confidence
`ζ = 0.05`.

This is a hypothesis test, not a generic membership-inference benchmark. It is also distinct
from TOFU forget quality, which is reported separately as a utility metric.

---

## What is where

```
audit_tofu/
  cum_runs_eps_lab.py   VENDORED VERBATIM from the paper's reference code. The tested epsilon math.
  epsilon_bounds.py     Wrapper: applies Lemma 4.1's /2, fixes the r convention.
  manifest.py           Immutable audit split: authors, batches, sign vectors, seeds, hash.
  tofu_data.py          TOFU loading, chat formatting, answer-only masking, global ordering.
  modeling.py           Model/tokenizer, FlashAttention fallback, 8-bit optimizer, LoRA.
  train.py              Fine-tuning loop on D(S) = D_r u D_f(S).
  unlearn.py            noop / npo / retain_ft / grad_ascent / grad_diff.
  scoring.py            Teacher-forced answer loss l_z(f), FP32, eval mode.
  attack.py             In/out Gaussians, Lambda_j, top/bottom r/2 prediction, overlap V.
  utility.py            TOFU probability / truth ratio / ROUGE / forget quality.
  run_manager.py        Run dirs, resume, retention policy, resource accounting.
  plotting.py           Calibration, Lambda, epsilon-vs-r, per-run overlap plots.

configs/                base | pilot | smoke | memory_saving | lora_fallback
                        proxy_pilot | qa_level | forgetq5_pinned180
scripts/                build_manifest | run_single | aggregate_audit | run_smoke
                        analyze_signal | compare_batching
                        compare_methods | collect_results | report_resources
                        train_reference | forget_quality | backfill_eval
                        run_audit.sh | launch_all.sh | slurm_run.sbatch
tests/                  unit + integration tests (CPU-only; no GPU needed)
docs/METRICS.md         Every metric: what is computed, where in the code, where stored
docs/IMPLEMENTATION.md  What was built and why; every design decision
docs/RESOURCE_ESTIMATE.md  Runtime/memory, projected and measured
docs/CODE_WALKTHROUGH.md   Reading guide to every module
docs/METHOD_DIAGNOSTICS.md Per-method leakage vs model damage
```

---

## Setup

```bash
# CPU-only: audit math, manifest, attack, and the non-GPU tests
python3 -m venv .venv
.venv/bin/python -m pip install numpy scipy pyyaml pytest matplotlib datasets

# Additionally required for training, scoring and the pilot
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
.venv/bin/python -m pip install transformers accelerate
.venv/bin/python -m pip install bitsandbytes   # optional: 8-bit optimizer mode
.venv/bin/python -m pip install peft           # optional: LoRA fallback

export HF_HOME=/path/to/hf_cache   # optional; defaults to ~/.cache/huggingface
```

> **Tip.** If `$HOME` is on a network filesystem, avoid building the venv on a uv-managed
> interpreter (`uv venv --python 3.11`): uv places the interpreter under `$HOME`, and
> `.venv/bin/python` breaks if the home directory becomes unreadable. Use the system
> `python3` as the base, as above.

### Model access

The spec pins `meta-llama/Llama-3.2-1B-Instruct`, which is **gated**. There is deliberately no
automatic fallback to an ungated mirror, so a run can never silently audit different weights
than the ones reported. Check access:

```bash
curl -s -o /dev/null -w '%{http_code}\n' \
  -H "Authorization: Bearer $(cat $HF_HOME/token)" \
  https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct/resolve/main/config.json
```

`200`/`307` means you are in. `403` means the token is valid but the repo has not been granted —
request access on the model page. Everything except training and the pilot works regardless.

---

## Running the audit

> **In a hurry?** One command runs the whole campaign — reference model, all four
> audits, aggregation — and is safe to interrupt and re-run:
>
> ```bash
> DRY_RUN=1 scripts/launch_campaign.sh 2,5,7 2   # see the plan first
> scripts/launch_campaign.sh 2,5,7 2             # <gpu_list> <runs_per_gpu>
> ```
>
> `docs/CODE_WALKTHROUGH.md` → **Run book** has the stage-by-stage equivalent, the
> measured costs, and where every output lands. The rest of this section explains
> what each stage does.

### 0. Tests

```bash
.venv/bin/python -m pytest tests/ -q
```

### 1. Build the immutable manifest — once

```bash
.venv/bin/python scripts/build_manifest.py --config configs/base.yaml
```

Fixes the 20 candidate authors, the 180-author retain set, all 20 calibration and 10
evaluation sign vectors, and the seeds; then hashes the result. Every later stage verifies the
hash. Regenerating requires `--overwrite` and invalidates all existing runs.

### 2. Dry run — no training

```bash
.venv/bin/python scripts/run_single.py --config configs/base.yaml --run_id calib_000 --dry_run
```

Resolves paths, builds the datasets, prints the positive/negative candidates, and asserts the
invariants (balance, disjointness, no negative-candidate leakage into training or the forget
loader). Expect `3800 = 3600 retain + 200 forget` train examples and `400` candidates.

### 3. Tiny end-to-end smoke test

```bash
.venv/bin/python scripts/run_smoke.py --config configs/smoke.yaml
```

Full pipeline on a tiny random LM with `m = 4` synthetic batches, offline. A plumbing test —
the epsilon it prints is meaningless.

### 4. One-run pilot — measure before committing

```bash
.venv/bin/python scripts/run_single.py --config configs/pilot.yaml --run_id calib_000 --gpu 0
```

Reports wall-clock and **measured** peak CUDA memory per stage, plus the output schema. Do this
before any multi-run launch. Then summarize:

```bash
.venv/bin/python scripts/report_resources.py --config configs/pilot.yaml --extrapolate 30
```

See `docs/RESOURCE_ESTIMATE.md`.

If `meta-llama` access is still pending, `configs/proxy_pilot.yaml` runs the same path on an
ungated stand-in to exercise the code and measure resources. It is clearly labelled and writes
to its own `output_root` — **it is not an audit result**, and the `split_hash` check prevents it
being aggregated with real runs.

### 5. The full audit — all runs, all methods

```bash
mkdir -p run_logs
scripts/run_audit.sh configs/base.yaml 0,1,2,3 2            # <config> <gpus> <runs_per_gpu>
scripts/run_audit.sh configs/base.yaml 0,1,2,3 2 wandb.mode=offline
scripts/run_audit.sh configs/base.yaml 0,1 1 utility.enabled=false
STATUS_INTERVAL=3600 scripts/run_audit.sh configs/base.yaml 0,1,2,3 2   # hourly ticks
```

Prints the planned GPU assignment **before** starting, then a `LAUNCH` line per run and a
`DONE`/`FAILED` line the moment each finishes, with a running `(n/29 complete)` count. Ends
with a per-run duration table, slowest first.

A periodic status tick reports per-run state, progress and an ETA extrapolated from finished
runs. Since completion and failure are already reported immediately, the tick exists mainly to
show progress and to catch a **hung** run — one that neither finishes nor fails emits no line at
all. Default `STATUS_INTERVAL=1800` (30 min); set `0` to disable.

Three levels of log:

| Level | Where |
| --- | --- |
| terminal | launch table, DONE/FAILED per run as it exits, periodic status ticks |
| stdout/stderr per run | `run_logs/<stamp>/<run_id>.log` |
| that run's structured log | `<output_root>/runs/<run_id>/run.log` |

Plus `run_logs/<stamp>/assignments.tsv` (run → gpu → pid → log) and `completed.tsv`
(run → gpu → exit → seconds).

### 5c. Config provenance — what was actually run

Every run and every launch snapshots its own configuration, written **before** any compute so
it survives a crash:

| File | Contents |
| --- | --- |
| `config.effective.yaml` | the fully-merged config as YAML — readable, diffable, and loadable straight back via `--config` |
| `config.json` | the same content, for programmatic use |
| `config_sources/00_base.yaml`, `01_qa_level.yaml`, … | **verbatim copies** of every contributing YAML, prefixed with merge order |
| `invocation.json` | `argv`, `--set` overrides, `config_hash`, host, `git_commit`/`git_dirty`, library versions, and the `HF_HOME` / `CUDA_VISIBLE_DEVICES` / `WANDB_*` env |

Written to `<output_root>/runs/<run_id>/` per run, and to `run_logs/<stamp>/` per launch —
the launch snapshot also includes `manifest.snapshot.json`.

Why keep the sources as well as the merged result? The merged config records the values that
were *used*, but not where they came from — and the files on disk will have moved on by the
time anyone debugs a stale result. Keeping both lets you diff what ran against what the configs
say today. The `--set` overrides in particular were previously invisible: they were baked into
the merged values with no record that they had been applied.

Or Slurm, one array task per run:

```bash
mkdir -p slurm_logs
sbatch --array=0-19 scripts/slurm_run.sbatch configs/base.yaml calibration
sbatch --array=0-9  scripts/slurm_run.sbatch configs/base.yaml evaluation
```

Runs are independently launchable by `run_id`. Interrupting is safe: `run_state.json` pins each
run's sign vector and seeds, and completed methods are skipped on restart — so re-running the
same command does only the work that is missing.

### 5b. Weights & Biases (optional)

One wandb run per `(run_id, method)`, grouped by `experiment.name + split_hash[:8]`, with
`job_type` = the method. Logs the training and unlearning curves live, plus a summary carrying
`candidates/gap`, `candidates/cohens_d`, timings, peak memory and `diverged`.

**It can never break a run.** Every wandb call swallows its own exceptions and degrades to a
no-op; the authoritative outputs are always the JSON files under `output_root`.

> **Auth.** `run_audit.sh` preflights wandb *before* spending GPU time, so an unreadable
> `~/.netrc` fails fast. Alternatives: `export WANDB_API_KEY=...`, or pass
> `wandb.mode=offline` and `wandb sync` the runs later. Disable entirely with
> `wandb.enabled=false`.

### 6. Aggregate into `ε_LB` and plots

```bash
for m in noop npo retain_ft grad_ascent grad_diff; do
  .venv/bin/python scripts/aggregate_audit.py --config configs/base.yaml --method $m
done
```

Each method needs its **own** calibration distributions (`p_in` is method-specific), but they
all branch from the same fine-tuned checkpoint, so the 30 runs are shared — not 30 per method.

Writes `audit_summary.json`, `per_r_detail.json`, and four PNGs per method. Order of operations
is enforced: calibration is fitted first, predictions are produced without any sign vector in
scope, and labels are revealed only afterwards to compute `V`.

---

## Reading the output

`audit_summary.json` headline fields:

| Field | Meaning |
| --- | --- |
| `epsilon_lb_mean` | **The reported bound.** LDP solution halved per Lemma 4.1. |
| `epsilon_ldp_lb_mean` | Before halving. Equals what the upstream repo prints. |
| `mean_overlap` / `v_list` | Mean and per-run overlap `V`, each in `[0, r]`. |
| `random_guess_baseline` | `r/2`. `V` at this level certifies nothing, and `epsilon_lb` is `None`. |
| `per_r` | The full `r ∈ {4,8,12,16,20}` sweep. Expect an inverted U (Remark 4.3). |
| `frozen_r` | `r` chosen by leave-one-out on **calibration only**, then frozen. |
| `per_qa_llr_accuracy` | Diagnostic. Not an input to the bound. |

`epsilon_lb: None` means the bound is infeasible even at `ε = 0` — the observed overlap is
consistent with random guessing, so no positive lower bound is certified. That is a valid
result, not an error.

### Two numbers to sanity-check first

1. **`noop` must leak.** It is the positive control: the "unlearned" model is just the trained
   model, so in/out losses should separate clearly and `V` should sit near `r`. If `noop` gives
   no bound, stop and diagnose training exposure, loss masking, or aggregation before spending
   compute on NPO. `aggregate_audit.py` prints this warning itself.
2. **The ceiling is `ε_LB ≈ 6.59`.** At `m = 20, L = 10, ζ = 0.05`, a *perfect* attack
   (`V = 20` on all ten runs) certifies `ε_LB = 6.589`. No method can score higher in this
   configuration. See `docs/IMPLEMENTATION.md` for how to raise it.

---

## TOFU forget quality (the KS test)

`forget_quality` is `null` in every run of the main audit, and that is a property of the
split rather than a missing feature. `base.yaml` draws its 20 candidate authors at random,
while the utility suite's `forget` group is hardwired to `forget10_perturbed` (authors
180–199). The two overlap in only 3 authors; the other 17 sit in the audit's **retain** set —
trained in every run, never unlearned. So a `D_r`-trained reference has memorized 340 of the
400 KS-tested rows (p → 1 for every method), and a `retain90`-trained reference is ignorant
of 17 authors the audit deliberately keeps (p → 0 for every method). Neither number moves
with unlearning quality.

`configs/forgetq5_pinned180.yaml` fixes this by **pinning** the candidate pool to authors
180–199, so `D_f` = `forget10_perturbed`'s authors and `D_r` = `retain90` exactly. Verified
on the built manifest: all 400 `forget10_perturbed` rows belong to candidate authors and
**none** to `D_r`, and all 400 `retain_perturbed` rows are in `D_r` with no candidate
contamination. Pinning is compatible with the bound — validity needs the pool fixed before
any run and the *sign vectors* i.i.d. uniform over `S_m` (§4), not the pool itself drawn at
random.

```bash
python scripts/build_manifest.py  --config configs/forgetq5_pinned180.yaml
python scripts/train_reference.py --config configs/forgetq5_pinned180.yaml --gpu 3
FAMILIES=evaluation scripts/run_audit.sh configs/forgetq5_pinned180.yaml 3,4 1
python scripts/forget_quality.py  --config configs/forgetq5_pinned180.yaml
python scripts/collect_results.py --config configs/forgetq5_pinned180.yaml
```

`train_reference.py` fine-tunes the **base** checkpoint on `D_r` alone. `retain_ft` cannot
substitute: it loads `trained/`, which already saw the forget data, so both of its
distributions are post-exposure. One reference serves every run, because `D_r` is fixed by
the manifest.

`forget_quality.py` is CPU-only and reads stored per-row truth ratios, so KS variants are
recomputable without re-scoring. Three slices per group:

| Slice | Rows | Meaning |
| --- | --- | --- |
| `plus` | the run's `S_j = +1` authors | **The headline.** Trained, then unlearned. |
| `minus` | the run's `S_j = -1` authors | Negative control: neither model ever saw them. |
| `all` | every row | The conventional TOFU number. Biased high — it mixes the two above. |

Read `minus` as a **distribution**, not a value: under the null it is `Uniform(0,1)`, so the
mean over runs and methods should sit near 0.5. Well below that means the reference is not
exchangeable with the runs — it differs by training seed and by ~63 optimizer steps (3600 vs
3800 examples at fixed epochs) — and `plus` cannot then be read at face value.

The `all` dilution is not hypothetical: on a controlled fixture where only the `+1` authors
were perturbed, `plus` gave `KS = 1.00` while `all` gave `KS = 0.50`, exactly halved by the
untouched `-1` rows.

> **Not comparable to published TOFU numbers.** Standard TOFU fine-tunes on all 200 authors
> then unlearns `forget10`; an audit run trains on `D_r` plus the `+1` candidates (190
> authors). Pinning buys internal coherence, not external comparability — and the two are
> mutually exclusive, since standard TOFU has no "out" condition at all.

`FAMILIES=evaluation` skips the 20 calibration runs the manifest pins. They are sampled up
front from independent RNG streams, so ε stays available later for the cost of just running
them: same `split_hash`, no re-training of the 5 evaluation runs.

---

## Adding a metric without re-training

The audit configs set `retention: keep_all` and `save_unlearned_checkpoint: true`, so the
post-unlearning model for every `(run, method)` stays on disk at
`runs/<id>/methods/<m>/model`. `scripts/backfill_eval.py` spends a forward pass on those
saved weights instead of re-running a fine-tune:

```bash
# what is missing, and which checkpoints survive? (no GPU)
python scripts/backfill_eval.py --config configs/base.yaml --dry_run

# add ROUGE to every evaluation run from saved weights
python scripts/backfill_eval.py --config configs/base.yaml --what utility \
    --gpu 3 --set utility.compute_rouge=true

# prove a saved checkpoint is the model that produced the stored losses
python scripts/backfill_eval.py --config configs/base.yaml --what verify --gpu 3
```

`utility` **merges** into the existing `utility.json` rather than replacing it, and a new
`None` never overwrites an existing value — so a ROUGE-only backfill cannot erase fields an
earlier run recorded. `verify` re-scores the candidate pool and compares against
`losses.json` without writing anything; agreement to ~1e-4 is proof the saved checkpoint is
the one the audit scored.

**What backfill cannot do.** It never re-trains and never re-unlearns. So it does not help
when the *split* changes: pinning the candidate pool changed what each model was trained on
(`D_r ∪ D_f(S)` over different authors), so the four pinned audits need one real run each —
backfill only pays off for metrics invented *after* that. It also cannot rescue the
`old_*` audits, whose weights were deleted under `retention: delete_after_scoring`;
`--dry_run` reports them as `0 have a saved model` and says so.

---

## Configuration notes

Override any leaf from the CLI; unknown keys are rejected rather than ignored:

```bash
.venv/bin/python scripts/run_single.py --config configs/base.yaml --run_id calib_000 \
    --set training.epochs=3 attack.pool_variance=null storage.retention=keep_all
```

| Config | Purpose |
| --- | --- |
| `base.yaml` | The audit. Full-parameter BF16, 5 epochs, effective batch 16, seq len 512. |
| `pilot.yaml` | One run, `noop` only, keeps the checkpoint. |
| `smoke.yaml` | Tiny random LM, `m = 4`, synthetic data, offline. |
| `memory_saving.yaml` | 8-bit optimizer, micro-batch 1. Same effective batch size. |
| `lora_fallback.yaml` | LoRA. **Audits a different (PEFT) pipeline — `ε` does not transfer.** |
| `qa_level.yaml` | `B = 1`, `m = 400`. Predicts per (question, answer) pair; ceiling 137.5 vs 6.59. |

Checkpoint retention defaults to `delete_after_scoring` (JSON results kept, weights dropped).
`output_root` defaults to `runs/<experiment>` under the repository root; with `keep_all`
this needs ~17 GiB per run, so point it (or symlink `runs/`) at a large scratch volume. Retention never
deletes anything outside a run's own resolved directory — there is a test for that, including
the symlink-escape case.
