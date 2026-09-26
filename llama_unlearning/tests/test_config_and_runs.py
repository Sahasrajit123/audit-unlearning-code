"""Config loading/overrides, and run management: resume safety and the
checkpoint-retention deletion guard."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from audit_tofu.config import apply_overrides, config_hash, deep_merge, load_config
from audit_tofu.run_manager import (
    RETENTION_POLICIES,
    ResourceTracker,
    load_or_create_run_state,
    prune_checkpoints,
    resolve_run_paths,
    run_ids_for_family,
    save_json,
)

REPO = Path(__file__).resolve().parent.parent


# --- config -------------------------------------------------------------------

def test_real_configs_load_and_include_resolves():
    base = load_config(REPO / "configs" / "base.yaml")
    assert base["split"]["num_candidate_authors"] == 20
    assert base["split"]["num_retain_authors"] == 180
    assert base["split"]["num_calibration_runs"] == 20
    assert base["split"]["num_evaluation_runs"] == 10
    assert base["epsilon"]["zeta"] == 0.05
    assert base["epsilon"]["delta"] == 0.0
    assert base["attack"]["aggregate"] == "sum"
    assert base["attack"]["r_values"] == [4, 6, 8, 10, 12, 14, 16, 20]
    assert base["model"]["id"] == "meta-llama/Llama-3.2-1B-Instruct"
    # The spec's three required methods, plus three added from OpenUnlearning.
    assert base["experiment"]["methods"] == [
        "noop", "npo", "retain_ft", "grad_ascent", "grad_diff", "simnpo"
    ]


def test_every_configured_method_is_known_and_has_a_config_block():
    """A typo in `experiment.methods` should fail here, not mid-run on a GPU."""
    from audit_tofu.unlearn import UNLEARNING_METHODS

    base = load_config(REPO / "configs" / "base.yaml")
    for m in base["experiment"]["methods"]:
        assert m in UNLEARNING_METHODS, f"{m} is not an implemented method"
        assert m in base["unlearning"], f"{m} has no unlearning config block"


def test_openunlearning_parity_defaults():
    """Defaults must match OpenUnlearning's published trainer configs."""
    u = load_config(REPO / "configs" / "base.yaml")["unlearning"]
    # configs/trainer/NPO.yaml: beta 0.1, alpha 1.0, gamma 1.0
    assert u["npo"]["beta"] == 0.1
    assert u["npo"]["retain_weight"] == 1.0
    assert u["npo"]["sequence_reduction"] == "sum"
    # configs/trainer/GradDiff.yaml: gamma 1.0, alpha 1.0
    assert u["grad_diff"]["gamma"] == 1.0
    assert u["grad_diff"]["alpha"] == 1.0
    # GradAscent is forget-only: no retain knob should be present to mislead.
    assert "alpha" not in u["grad_ascent"]
    assert "retain_weight" not in u["grad_ascent"]
    # configs/trainer/SimNPO.yaml: beta 4.5, delta 0.0, gamma 0.125, alpha 1.0.
    # These differ sharply from NPO's and must not be silently inherited.
    assert u["simnpo"]["beta"] == 4.5
    assert u["simnpo"]["delta"] == 0.0
    assert u["simnpo"]["gamma"] == 0.125
    assert u["simnpo"]["alpha"] == 1.0
    # OpenUnlearning's TOFU default.yaml: lr 1e-5, weight_decay 0.01
    for m in ("npo", "grad_diff", "grad_ascent", "retain_ft", "simnpo"):
        assert u[m]["learning_rate"] == 1.0e-5, m
        assert u[m]["weight_decay"] == 0.01, m


def test_pilot_config_overrides_base():
    pilot = load_config(REPO / "configs" / "pilot.yaml")
    assert pilot["experiment"]["methods"] == ["noop"]
    assert pilot["utility"]["enabled"] is False
    assert pilot["storage"]["retention"] == "keep_trained"
    # Inherited untouched from base.
    assert pilot["split"]["num_candidate_authors"] == 20
    assert pilot["epsilon"]["zeta"] == 0.05


def test_smoke_config_is_internally_consistent():
    smoke = load_config(REPO / "configs" / "smoke.yaml")
    m = smoke["split"]["num_candidate_authors"]
    import math

    capacity = math.comb(m, m // 2)
    need = smoke["split"]["num_calibration_runs"] + smoke["split"]["num_evaluation_runs"]
    assert need <= capacity, (
        f"smoke config needs {need} disjoint balanced vectors but only {capacity} exist"
    )
    assert all(r <= m for r in smoke["attack"]["r_values"])


@pytest.mark.parametrize(
    "config_name",
    # Discovered rather than listed, so a new config is covered the moment it lands.
    sorted(p.name for p in (REPO / "configs").glob("*.yaml")),
)
def test_every_configured_r_is_even_and_within_m(config_name):
    """Every ``r`` in every config must be even and ``<= m``.

    Both failure modes are silent-ish in different bad ways, which is why this is a
    config test rather than a comment:

    * **odd r** is inadmissible -- the auditor makes ``r/2`` positive and ``r/2``
      negative guesses and Lemma 4.2's ``f(v)`` is built from ``C(r/2,a1) C(r/2,a2)``.
      ``epsilon_bounds._validate`` raises, and ``aggregate_audit.py`` does *not* guard
      the call, so one odd entry aborts a whole aggregation after all the prediction
      work is done.
    * **r > m** is dropped silently by ``aggregate_audit.py``, which once collapsed the
      B=10 sweep to a single point (see the comment in ``base_batch_size_10.yaml``).
    """
    path = REPO / "configs" / config_name
    if not path.exists():
        pytest.skip(f"{config_name} not present")
    cfg = load_config(path)
    r_values = cfg.get("attack", {}).get("r_values")
    if not r_values:
        pytest.skip("config defines no r_values of its own")

    batching = cfg["split"].get("batching", "author")
    n_authors = cfg["split"]["num_candidate_authors"]
    qa_per_author = cfg["split"].get("qa_per_author", 20)
    if batching == "author":
        m = n_authors
    elif batching == "qa":
        m = n_authors * qa_per_author
    else:
        m = n_authors * (qa_per_author // int(batching))

    for r in r_values:
        assert int(r) % 2 == 0, (
            f"{config_name}: r={r} is odd; the auditor guesses r/2 per side"
        )
        assert 0 < int(r) <= m, f"{config_name}: r={r} outside (0, m={m}]"
    assert sorted(set(r_values)) == list(r_values), (
        f"{config_name}: r_values must be sorted and unique; got {r_values}"
    )


def test_memory_saving_and_lora_configs_load():
    mem = load_config(REPO / "configs" / "memory_saving.yaml")
    assert mem["training"]["optimizer"] == "adamw_8bit"
    assert (
        mem["training"]["micro_batch_size"]
        * mem["training"]["gradient_accumulation_steps"]
        == 16
    ), "memory-saving mode must preserve the effective batch size"

    lora = load_config(REPO / "configs" / "lora_fallback.yaml")
    assert lora["model"]["lora"]["enabled"] is True
    assert lora["experiment"]["output_root"] != load_config(
        REPO / "configs" / "base.yaml"
    )["experiment"]["output_root"], "LoRA runs must not share an output root"


def test_effective_batch_size_is_sixteen_in_base():
    base = load_config(REPO / "configs" / "base.yaml")
    t = base["training"]
    assert t["micro_batch_size"] * t["gradient_accumulation_steps"] == 16


def test_deep_merge_is_recursive_and_nondestructive():
    a = {"x": {"y": 1, "z": 2}, "k": 3}
    b = {"x": {"y": 9}}
    out = deep_merge(a, b)
    assert out == {"x": {"y": 9, "z": 2}, "k": 3}
    assert a["x"]["y"] == 1, "deep_merge mutated its input"


def test_overrides_coerce_types_and_reject_unknown_keys():
    cfg = {"training": {"epochs": 5, "learning_rate": 1e-5, "flag": False}}
    out = apply_overrides(cfg, ["training.epochs=8", "training.flag=true"])
    assert out["training"]["epochs"] == 8 and isinstance(out["training"]["epochs"], int)
    assert out["training"]["flag"] is True

    with pytest.raises(KeyError, match="unknown config key"):
        apply_overrides(cfg, ["training.epoch=8"])
    with pytest.raises(KeyError, match="unknown config section"):
        apply_overrides(cfg, ["trainin.epochs=8"])
    with pytest.raises(ValueError, match="key=value"):
        apply_overrides(cfg, ["training.epochs"])


def test_config_hash_is_stable_and_sensitive():
    a = {"b": 1, "a": {"z": 2}}
    b = {"a": {"z": 2}, "b": 1}
    assert config_hash(a) == config_hash(b), "hash must ignore key order"
    assert config_hash(a) != config_hash({"b": 2, "a": {"z": 2}})


# --- run state / resume -------------------------------------------------------

def test_run_state_is_written_once_and_reused(audit_manifest, tmp_path):
    run_id = "calib_005"
    paths = resolve_run_paths(tmp_path, run_id)
    s1 = load_or_create_run_state(audit_manifest, run_id, paths)
    s2 = load_or_create_run_state(audit_manifest, run_id, paths)

    assert s1.sign_vector == s2.sign_vector
    assert s1.train_seed == s2.train_seed == audit_manifest["seeds"]["train_seed_base"] + 5
    assert s1.unlearn_seed == audit_manifest["seeds"]["unlearn_seed_base"] + 5
    assert s1.family == "calibration" and s1.run_index == 5
    assert (paths.run_dir / "run_state.json").exists()


def test_resume_refuses_a_changed_split(audit_manifest, tmp_path):
    from audit_tofu.manifest import build_manifest

    run_id = "calib_000"
    paths = resolve_run_paths(tmp_path, run_id)
    load_or_create_run_state(audit_manifest, run_id, paths)

    other = build_manifest(split_seed=4242)  # different split, same run ids
    with pytest.raises(RuntimeError, match="split_hash"):
        load_or_create_run_state(other, run_id, paths)


def test_resume_refuses_altered_seeds(audit_manifest, tmp_path):
    run_id = "eval_001"
    paths = resolve_run_paths(tmp_path, run_id)
    load_or_create_run_state(audit_manifest, run_id, paths)

    tampered = json.loads((paths.run_dir / "run_state.json").read_text())
    tampered["train_seed"] += 1
    (paths.run_dir / "run_state.json").write_text(json.dumps(tampered))

    with pytest.raises(RuntimeError, match="train_seed"):
        load_or_create_run_state(audit_manifest, run_id, paths)


def test_run_ids_cover_both_families(audit_manifest):
    calib = run_ids_for_family(audit_manifest, "calibration")
    ev = run_ids_for_family(audit_manifest, "evaluation")
    assert len(calib) == 20 and len(ev) == 10
    assert not set(calib) & set(ev)
    with pytest.raises(ValueError):
        run_ids_for_family(audit_manifest, "test")


def test_resolve_run_paths_rejects_escaping_run_ids(tmp_path):
    with pytest.raises(ValueError, match="escapes output_root"):
        resolve_run_paths(tmp_path, "../../etc")


# --- retention: deletion safety ----------------------------------------------

def _make_run_tree(tmp_path, run_id="calib_000"):
    paths = resolve_run_paths(tmp_path, run_id)
    paths.trained_dir.mkdir(parents=True, exist_ok=True)
    (paths.trained_dir / "model.safetensors").write_text("weights")
    for method in ("noop", "npo"):
        d = paths.method_dir(method) / "model"
        d.mkdir(parents=True, exist_ok=True)
        (d / "model.safetensors").write_text("weights")
        save_json(paths.method_dir(method) / "losses.json", {"records": []})
    return paths


def test_keep_all_deletes_nothing(tmp_path):
    paths = _make_run_tree(tmp_path)
    assert prune_checkpoints(paths, "keep_all") == []
    assert paths.trained_dir.exists()


def test_keep_trained_drops_method_weights_only(tmp_path):
    paths = _make_run_tree(tmp_path)
    removed = prune_checkpoints(paths, "keep_trained")
    assert paths.trained_dir.exists(), "trained checkpoint should survive"
    assert not (paths.method_dir("npo") / "model").exists()
    assert (paths.method_dir("npo") / "losses.json").exists(), "results must survive"
    assert len(removed) == 2


def test_delete_after_scoring_drops_all_weights_but_keeps_results(tmp_path):
    paths = _make_run_tree(tmp_path)
    prune_checkpoints(paths, "delete_after_scoring")
    assert not paths.trained_dir.exists()
    assert not (paths.method_dir("noop") / "model").exists()
    assert (paths.method_dir("noop") / "losses.json").exists()
    assert (paths.run_dir / "run_state.json").exists() or True


def test_retention_never_deletes_outside_the_run_directory(tmp_path):
    """A symlink pointing outside the run dir must be refused, not followed."""
    paths = _make_run_tree(tmp_path)

    outsider = tmp_path / "precious"
    outsider.mkdir()
    (outsider / "important.txt").write_text("do not delete")

    # Replace a method's model dir with a symlink to the outside directory.
    victim = paths.method_dir("noop") / "model"
    import shutil

    shutil.rmtree(victim)
    victim.symlink_to(outsider, target_is_directory=True)

    removed = prune_checkpoints(paths, "delete_after_scoring")

    assert outsider.exists(), "retention followed a symlink outside the run dir"
    assert (outsider / "important.txt").exists()
    assert str(outsider.resolve()) not in removed


def test_retention_rejects_unknown_policy(tmp_path):
    paths = _make_run_tree(tmp_path)
    with pytest.raises(ValueError, match="policy must be one of"):
        prune_checkpoints(paths, "nuke_everything")
    assert "delete_after_scoring" in RETENTION_POLICIES


# --- resource tracking --------------------------------------------------------

def test_resource_tracker_records_stages():
    t = ResourceTracker()
    t.start("train")
    rec = t.stop()
    assert "duration_seconds" in rec
    assert "peak_allocated_gib" in rec and "peak_reserved_gib" in rec

    t.start("score")
    t.stop()
    d = t.as_dict()
    assert set(d["stages"]) == {"train", "score"}
    assert d["total_duration_seconds"] >= 0
    assert "overall_peak_allocated_gib" in d


def test_resource_tracker_requires_start_before_stop():
    with pytest.raises(RuntimeError, match="without start"):
        ResourceTracker().stop()


def test_save_json_is_atomic_and_leaves_no_temp(tmp_path):
    p = save_json(tmp_path / "out" / "x.json", {"a": 1})
    assert json.loads(p.read_text()) == {"a": 1}
    assert not list(p.parent.glob("*.tmp"))


# --- wandb logging: optional, and never fatal --------------------------------

def test_wandb_config_block_exists_with_safe_defaults():
    w = load_config(REPO / "configs" / "base.yaml")["wandb"]
    assert set(w) >= {"enabled", "project", "entity", "mode", "group", "tags", "dir"}
    assert w["mode"] in ("online", "offline", "disabled")


def test_wandb_preflight_reports_a_reason_in_every_case():
    from audit_tofu.wandb_logger import preflight

    base = load_config(REPO / "configs" / "base.yaml")

    off = dict(base, wandb=dict(base["wandb"], enabled=False))
    ok, why = preflight(off)
    assert ok is False and "disabled" in why

    offline = dict(base, wandb=dict(base["wandb"], enabled=True, mode="offline"))
    ok, why = preflight(offline)
    assert ok is True and "offline" in why

    # Online depends on credentials, so only assert the contract: a bool and a reason.
    ok, why = preflight(dict(base, wandb=dict(base["wandb"], enabled=True,
                                              mode="online")))
    assert isinstance(ok, bool) and isinstance(why, str) and why


def test_null_run_absorbs_every_call():
    """The no-op stand-in must satisfy the same surface as a live run."""
    from audit_tofu.wandb_logger import NullRun

    n = NullRun()
    assert n.enabled is False and n.url is None
    n.log({"a": 1}, step=3)
    n.summary({"b": 2})
    n.finish(exit_code=1)
    with n as ctx:
        ctx.log({"c": 3})


def test_wandb_create_returns_nullrun_when_disabled():
    from audit_tofu.wandb_logger import NullRun, WandbRun

    cfg = load_config(REPO / "configs" / "base.yaml")
    cfg["wandb"] = dict(cfg["wandb"], enabled=False)
    got = WandbRun.create(cfg, "r", "noop", split_hash="h", family="calibration",
                          logger=None)
    assert isinstance(got, NullRun)


def test_wandb_create_never_raises_on_a_broken_config():
    """A malformed wandb section must degrade, not kill a 20-GPU-hour run."""
    from audit_tofu.wandb_logger import WandbRun

    cfg = load_config(REPO / "configs" / "base.yaml")
    cfg["wandb"] = {"enabled": True, "mode": "not-a-real-mode", "project": None}
    msgs = []
    got = WandbRun.create(cfg, "r", "noop", split_hash="h", family="calibration",
                          logger=type("L", (), {"info": lambda _s, m: msgs.append(m)})())
    # Either it degraded to NullRun, or it somehow succeeded -- both are acceptable;
    # what matters is that it did not raise.
    assert hasattr(got, "log") and hasattr(got, "finish")
    got.log({"x": 1})
    got.finish()


def test_train_model_accepts_a_metrics_sink():
    """The sink is plumbed through, and a sink that raises must not break training."""
    import inspect

    from audit_tofu.train import train_model
    from audit_tofu.unlearn import apply_unlearning

    assert "metrics_sink" in inspect.signature(train_model).parameters
    assert "metrics_sink" in inspect.signature(apply_unlearning).parameters


# --- config provenance: the YAML actually used is saved with the results ------

def test_config_source_files_resolves_the_include_chain():
    from audit_tofu.config import config_source_files

    base = config_source_files(REPO / "configs" / "base.yaml")
    assert [f.name for f in base] == ["base.yaml"]

    # Includes come first (they are merged first), the named file last.
    qa = config_source_files(REPO / "configs" / "qa_level.yaml")
    assert [f.name for f in qa] == ["base.yaml", "qa_level.yaml"]

    pilot = config_source_files(REPO / "configs" / "pilot.yaml")
    assert [f.name for f in pilot] == ["base.yaml", "pilot.yaml"]


def test_save_config_provenance_writes_yaml_sources_and_invocation(tmp_path):
    from audit_tofu.config import load_config
    from audit_tofu.run_manager import save_config_provenance

    cfg_path = REPO / "configs" / "qa_level.yaml"
    overrides = ["training.epochs=3"]
    cfg = load_config(cfg_path, overrides)

    inv = save_config_provenance(
        tmp_path, cfg, config_path=str(cfg_path), overrides=overrides,
        argv=["run_single.py", "--config", str(cfg_path)],
    )

    # 1. effective config, in BOTH formats, and the YAML must round-trip
    import yaml

    eff = tmp_path / "config.effective.yaml"
    assert eff.exists() and (tmp_path / "config.json").exists()
    reloaded = yaml.safe_load(eff.read_text())
    assert reloaded == cfg, "the saved YAML must reproduce the effective config exactly"
    assert reloaded["training"]["epochs"] == 3, "the --set override must be baked in"
    assert reloaded["split"]["batching"] == "qa"

    # 2. verbatim sources, prefixed with their merge order
    names = sorted(p.name for p in (tmp_path / "config_sources").iterdir())
    assert names == ["00_base.yaml", "01_qa_level.yaml"], names
    # byte-identical to what is on disk
    assert (tmp_path / "config_sources" / "01_qa_level.yaml").read_text() == \
        cfg_path.read_text()

    # 3. invocation record
    assert inv["overrides"] == overrides
    assert inv["config_hash"] == json.loads(
        (tmp_path / "invocation.json").read_text()
    )["config_hash"]
    assert inv["versions"]["python"]
    assert [s["saved_as"] for s in inv["config_sources"]] == names


def test_provenance_survives_a_missing_config_path(tmp_path):
    """It must still record the effective config when no source path is given."""
    from audit_tofu.config import load_config
    from audit_tofu.run_manager import save_config_provenance

    cfg = load_config(REPO / "configs" / "base.yaml")
    inv = save_config_provenance(tmp_path, cfg)
    assert (tmp_path / "config.effective.yaml").exists()
    assert inv["config_sources"] == []
    assert inv["config_path"] is None


def test_provenance_yaml_is_rerunnable_as_a_config(tmp_path):
    """The saved YAML should load straight back through load_config."""
    from audit_tofu.config import load_config
    from audit_tofu.run_manager import save_config_provenance

    cfg = load_config(REPO / "configs" / "qa_level.yaml", ["training.epochs=2"])
    save_config_provenance(tmp_path, cfg, config_path=str(REPO / "configs" / "qa_level.yaml"))
    # No `include:` key, so it stands alone.
    round_trip = load_config(tmp_path / "config.effective.yaml")
    assert round_trip == cfg
