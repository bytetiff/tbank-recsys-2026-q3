import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
from catboost import CatBoostClassifier

from tlab_recsys.cli import synthetic_control
from tlab_recsys.config import Config, write_json
from tlab_recsys.download import file_hash
from tlab_recsys.checkpoints import read_checkpoint, write_checkpoint
from tlab_recsys.modeling import model_frame, pool
from tlab_recsys.pipeline import evaluate_run, train_run
import tlab_recsys.pipeline as pipeline


@pytest.fixture
def partial_run(tmp_path, monkeypatch):
    events, manifest, base = synthetic_control()
    config = Config(dict(base.values, model_seeds=[17, 29, 43],
                         search={"max_fits": 0, "max_groups": 0, "min_improvement": 0.0001}))
    calls = []
    state = {"interrupt_seed": 29}

    class Model:
        def __init__(self, seed, columns):
            self.seed, self.columns = seed, columns

        def predict_proba(self, frame, **kwargs):
            scores = np.full(len(frame), 0.5)
            return np.column_stack((1 - scores, scores))

        def save_model(self, path):
            Path(path).write_bytes(json.dumps([self.seed, self.columns]).encode())

    def tune(*args):
        seed = args[-1]
        calls.append(("tune", seed))
        if seed == state["interrupt_seed"]:
            raise RuntimeError("intentional interruption")
        return {"parameters": {"iterations": 2, "depth": 2, "learning_rate": 0.1, "l2_leaf_reg": 3}, "fits": 1, "trace": []}

    def fit(train, labels, columns, parameters, config, seed):
        calls.append(("fit", seed))
        return Model(seed, columns)

    actual_slice = pipeline.make_slice

    def pretest_only(*args, **kwargs):
        assert args[2] != "test", "Test labels read before freeze"
        return actual_slice(*args, **kwargs)

    monkeypatch.setattr(pipeline, "tune_manual", tune)
    monkeypatch.setattr(pipeline, "fit_fixed", fit)
    monkeypatch.setattr(pipeline, "pool", lambda frame, columns: frame[columns])
    monkeypatch.setattr(pipeline, "make_slice", pretest_only)
    with pytest.raises(RuntimeError, match="intentional"):
        train_run(events, manifest, config, tmp_path, "yambda")
    assert (tmp_path / "seed-17/checkpoint.json").exists()
    assert not (tmp_path / "seed-29/checkpoint.json").exists()
    assert not (tmp_path / "frozen.json").exists()
    state["interrupt_seed"] = None
    calls.clear()
    return tmp_path, events, manifest, config, calls, state


def test_resume_skips_completed_seed_reuses_tables_and_keeps_test_closed(partial_run, monkeypatch):
    root, events, manifest, config, calls, _ = partial_run
    originals = {str(path.relative_to(root)): path.read_bytes() for path in (root / "seed-17").iterdir()}
    checkpoint = (root / "pretest-checkpoint.json").read_bytes()
    monkeypatch.setattr(pipeline, "compute_features", lambda *args: pytest.fail("Completed pretest features recomputed"))
    monkeypatch.setattr(pipeline, "make_slice", lambda *args: pytest.fail("Completed pretest slice recomputed or test accessed"))
    with pytest.raises(ValueError, match="requires frozen"):
        evaluate_run(events, manifest, config, root)
    frozen = train_run(events, manifest, config, root, "yambda")
    assert [decision["model_seed"] for decision in frozen["decisions"]] == [17, 29, 43]
    assert not any(seed == 17 for _, seed in calls)
    assert {seed for action, seed in calls if action == "tune"} == {29, 43}
    assert (root / "pretest-checkpoint.json").read_bytes() == checkpoint
    for relative, content in originals.items():
        assert (root / relative).read_bytes() == content
    write_json(root / "diagnostics/new.json", {"mutable": True})
    write_json(root / "failure.json", {"mutable": True})
    (root / "temporary.json.tmp").write_text("partial")
    assert pipeline.load_frozen(root, config, manifest) == frozen
    assert not any("diagnostics" in relative or relative.endswith(".tmp") for relative in frozen["artifacts"])


@pytest.mark.parametrize("artifact", ["seed-17/manual.cbm", "seed-17/decision.json", "seed-17/tuning.json", "seed-17/selection.json", "seed-17/checkpoint.json", "pretest-checkpoint.json", "run-identity.json", "config.json", "slice", "missing-model"])
def test_resume_rejects_corruption_before_fit_or_overwrite(partial_run, monkeypatch, artifact):
    root, events, manifest, config, calls, _ = partial_run
    relative = next((root / "slices").glob("*.parquet")) if artifact == "slice" else root / ("seed-17/manual.cbm" if artifact == "missing-model" else artifact)
    if artifact == "missing-model":
        relative.unlink()
    else:
        relative.write_bytes(relative.read_bytes() + b"corrupted")
    original = {path: file_hash(path) for path in root.rglob("*") if path.is_file()}
    monkeypatch.setattr(pipeline, "_evaluate_chunks", lambda *args: pytest.fail("Corrupt run reached test"))
    with pytest.raises(ValueError):
        train_run(events, manifest, config, root, "yambda")
    assert calls == []
    assert all(file_hash(path) == digest for path, digest in original.items())
    assert not (root / "frozen.json").exists()


@pytest.mark.parametrize("changed", ["config", "manifest", "protocol", "runtime"])
def test_resume_binds_complete_run_identity_before_fit(partial_run, monkeypatch, changed):
    root, events, manifest, config, calls, _ = partial_run
    dataset = "yambda"
    if changed == "config":
        config = Config(dict(config.values, cohort_seed=config["cohort_seed"] + 1))
    elif changed == "manifest":
        manifest = dict(manifest, changed=True)
    elif changed == "protocol":
        dataset = "kion"
    else:
        current = pipeline.runtime_fingerprint()
        monkeypatch.setattr(pipeline, "runtime_fingerprint", lambda: dict(current, python_version="changed"))
    before = (root / "config.json").read_bytes()
    with pytest.raises(ValueError, match="identity"):
        train_run(events, manifest, config, root, dataset)
    assert calls == []
    assert (root / "config.json").read_bytes() == before


def test_seed_with_decision_but_without_checkpoint_restarts_entire_seed(partial_run):
    root, events, manifest, config, calls, _ = partial_run
    (root / "seed-17/checkpoint.json").unlink()
    (root / "seed-17/checkpoint.json.tmp").write_text("incomplete atomic write")
    frozen = train_run(events, manifest, config, root, "yambda")
    assert ("tune", 17) in calls and ("fit", 17) in calls
    assert len(frozen["decisions"]) == 3
    assert not any(relative.endswith(".tmp") for relative in frozen["artifacts"])


def test_all_seed_checkpoints_checked_before_first_unfinished_fit(partial_run):
    root, events, manifest, config, calls, _ = partial_run
    train_run(events, manifest, config, root, "yambda")
    (root / "frozen.json").unlink()
    (root / "seed-17/checkpoint.json").unlink()
    (root / "seed-43/manual.cbm").write_bytes(b"corrupt later seed")
    calls.clear()
    with pytest.raises(ValueError, match="artifact"):
        train_run(events, manifest, config, root, "yambda")
    assert calls == []


def test_completed_last_seed_can_freeze_without_any_fit(partial_run, monkeypatch):
    root, events, manifest, config, calls, _ = partial_run
    train_run(events, manifest, config, root, "yambda")
    (root / "frozen.json").unlink()
    calls.clear()
    monkeypatch.setattr(pipeline.pl, "read_parquet", lambda *args, **kwargs: pytest.fail("All-completed resume read feature Parquet"))
    monkeypatch.setattr(pipeline.pd, "concat", lambda *args, **kwargs: pytest.fail("All-completed resume allocated training frames"))
    frozen = train_run(events, manifest, config, root, "yambda")
    assert calls == [] and len(frozen["decisions"]) == 3


def test_runtime_change_during_seed_prevents_completion_checkpoint(partial_run, monkeypatch):
    root, events, manifest, config, _, _ = partial_run
    original_fit = pipeline.fit_fixed
    original_fingerprint = pipeline.runtime_fingerprint()

    def changed_after_fit(*args):
        model = original_fit(*args)
        monkeypatch.setattr(pipeline, "runtime_fingerprint", lambda: dict(original_fingerprint, python_version="changed-during-fit"))
        return model

    monkeypatch.setattr(pipeline, "fit_fixed", changed_after_fit)
    with pytest.raises(ValueError, match="changed during training"):
        train_run(events, manifest, config, root, "yambda")
    assert not (root / "seed-29/checkpoint.json").exists()
    assert not (root / "frozen.json").exists()


@pytest.mark.parametrize("valid_digest", [False, True])
def test_pretest_checkpoint_change_during_fit_blocks_seed_completion(partial_run, monkeypatch, valid_digest):
    root, events, manifest, config, _, _ = partial_run
    originals = {path: path.read_bytes() for path in (root / "seed-17").iterdir()}
    original_fit = pipeline.fit_fixed

    def changed_after_fit(*args):
        model = original_fit(*args)
        path = root / "pretest-checkpoint.json"
        if valid_digest:
            record = read_checkpoint(path, "pretest")
            record.pop("checkpoint_hash")
            record["features_resources"] = {"changed": True}
            write_checkpoint(path, "pretest", record)
        else:
            path.write_bytes(path.read_bytes() + b"corrupt")
        return model

    monkeypatch.setattr(pipeline, "fit_fixed", changed_after_fit)
    with pytest.raises(ValueError, match="Checkpoint integrity|Pretest checkpoint changed"):
        train_run(events, manifest, config, root, "yambda")
    assert not (root / "seed-29/checkpoint.json").exists()
    assert not (root / "seed-43/checkpoint.json").exists()
    assert not (root / "frozen.json").exists()
    assert all(path.read_bytes() == content for path, content in originals.items())


def test_read_pretest_preserves_exact_features_labels_and_order(partial_run):
    root, events, manifest, config, _, _ = partial_run
    specs = pipeline.grammar(list(dict.fromkeys(config["generated_windows_days"] + config["manual_windows_days"])))
    record = read_checkpoint(root / "pretest-checkpoint.json", "pretest")
    slices, frames = pipeline._read_pretest(root, record, specs)
    for entry, current, frame in zip(record["slices"], slices, frames, strict=True):
        expected = pipeline.make_slice(events, manifest["users"], entry["split"], entry["cutoff"], "yambda", config)
        expected_frame = model_frame(pipeline.compute_features(events, expected.rows, expected.cutoff, specs, config))
        assert current.rows.equals(expected.rows)
        np.testing.assert_array_equal(current.rows["label"].to_numpy(), expected.rows["label"].to_numpy())
        pd.testing.assert_frame_equal(frame, expected_frame, check_exact=True)
        assert current.targets == expected.targets
        assert current.catalog == expected.catalog
        assert current.catalog_targets == expected.catalog_targets
        assert current.users == expected.users


def test_real_catboost_resume_matches_uninterrupted_decisions_and_predictions(tmp_path, monkeypatch):
    events, manifest, base = synthetic_control()
    values = dict(base.values, model_seeds=[17, 29], search={"max_fits": 1, "max_groups": 1, "min_improvement": 0.0001})
    values["catboost"] = dict(values["catboost"], depth=[2], learning_rate=[0.1], l2_leaf_reg=[3], max_iterations=3, early_stopping_rounds=1)
    config = Config(values)
    uninterrupted_root, resumed_root = tmp_path / "uninterrupted", tmp_path / "resumed"
    uninterrupted = train_run(events, manifest, config, uninterrupted_root, "yambda")
    original_tune = pipeline.tune_manual

    def interrupt_second_seed(*args):
        if args[-1] == 29:
            raise RuntimeError("real-model interruption")
        return original_tune(*args)

    monkeypatch.setattr(pipeline, "tune_manual", interrupt_second_seed)
    with pytest.raises(RuntimeError, match="real-model interruption"):
        train_run(events, manifest, config, resumed_root, "yambda")
    originals = {path: path.read_bytes() for path in (resumed_root / "seed-17").iterdir()}
    monkeypatch.setattr(pipeline, "tune_manual", original_tune)
    resumed = train_run(events, manifest, config, resumed_root, "yambda")
    assert all(path.read_bytes() == content for path, content in originals.items())
    for expected, actual in zip(uninterrupted["decisions"], resumed["decisions"], strict=True):
        assert actual["parameters"] == expected["parameters"]
        assert actual["selected_groups"] == expected["selected_groups"]
        for variant, record in expected["models"].items():
            actual_record = actual["models"][variant]
            assert actual_record["columns"] == record["columns"]
            frame = model_frame(pl.read_parquet(uninterrupted_root / "slices" / f"selection-{uninterrupted['pretest_slices'][-1]['cutoff']}.parquet").select(record["columns"]))
            expected_model, actual_model = CatBoostClassifier(), CatBoostClassifier()
            expected_model.load_model(str(uninterrupted_root / record["path"]))
            actual_model.load_model(str(resumed_root / actual_record["path"]))
            candidates = pool(frame, record["columns"])
            np.testing.assert_array_equal(expected_model.predict_proba(candidates, thread_count=2), actual_model.predict_proba(candidates, thread_count=2))


@pytest.mark.parametrize("boundary", ["model-save", "checkpoint-save"])
def test_write_failure_restarts_incomplete_seed_and_preserves_completed_seed(partial_run, monkeypatch, boundary):
    root, events, manifest, config, calls, _ = partial_run
    originals = {path: path.read_bytes() for path in (root / "seed-17").iterdir()}
    if boundary == "model-save":
        original_fit = pipeline.fit_fixed

        def model_with_failed_save(*args):
            model = original_fit(*args)

            def fail(path):
                Path(path).write_bytes(b"partial model save")
                raise OSError("injected model save failure")

            model.save_model = fail
            return model

        monkeypatch.setattr(pipeline, "fit_fixed", model_with_failed_save)
    else:
        original_replace = Path.replace

        def fail_checkpoint_replace(path, target):
            if path.name == "checkpoint.json.tmp" and path.parent.name == "seed-29":
                raise OSError("injected checkpoint save failure")
            return original_replace(path, target)

        monkeypatch.setattr(Path, "replace", fail_checkpoint_replace)
    with pytest.raises(OSError, match="injected"):
        train_run(events, manifest, config, root, "yambda")
    assert not (root / "seed-29/checkpoint.json").exists()
    assert not (root / "frozen.json").exists()
    assert all(path.read_bytes() == content for path, content in originals.items())
    if boundary == "model-save":
        monkeypatch.setattr(pipeline, "fit_fixed", original_fit)
    else:
        monkeypatch.setattr(Path, "replace", original_replace)
    calls.clear()
    frozen = train_run(events, manifest, config, root, "yambda")
    assert ("tune", 29) in calls and ("fit", 29) in calls
    assert not any(seed == 17 for _, seed in calls)
    assert len(frozen["decisions"]) == 3
    assert all(path.read_bytes() == content for path, content in originals.items())


def test_failed_parquet_replace_recovers_before_any_seed_is_completed(tmp_path, monkeypatch):
    events, manifest, base = synthetic_control()
    values = dict(base.values, search={"max_fits": 0, "max_groups": 0, "min_improvement": 0.0001})
    values["catboost"] = dict(values["catboost"], depth=[2], learning_rate=[0.1], l2_leaf_reg=[3], max_iterations=2, early_stopping_rounds=1)
    config = Config(values)
    original_replace = Path.replace

    def fail_parquet_replace(path, target):
        if path.name.endswith(".parquet.tmp"):
            raise OSError("injected parquet replace failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_parquet_replace)
    with pytest.raises(OSError, match="injected parquet"):
        train_run(events, manifest, config, tmp_path, "yambda")
    assert list((tmp_path / "slices").glob("*.parquet.tmp"))
    assert not (tmp_path / "pretest-checkpoint.json").exists()
    assert not list(tmp_path.glob("seed-*/checkpoint.json"))
    identity = (tmp_path / "run-identity.json").read_bytes()
    monkeypatch.setattr(Path, "replace", original_replace)
    frozen = train_run(events, manifest, config, tmp_path, "yambda")
    assert len(frozen["decisions"]) == 1
    assert (tmp_path / "run-identity.json").read_bytes() == identity
    assert not list((tmp_path / "slices").glob("*.parquet.tmp"))


def test_ambiguous_legacy_partial_directory_is_not_overwritten(tmp_path):
    events, manifest, config = synthetic_control()
    write_json(tmp_path / "seed-17/decision.json", {"old": True})
    with pytest.raises(ValueError, match="Ambiguous partial"):
        train_run(events, manifest, config, tmp_path, "yambda")
    assert json.loads((tmp_path / "seed-17/decision.json").read_text()) == {"old": True}


def test_resume_binds_categories_before_rebuilding_incomplete_pretest(tmp_path, monkeypatch):
    events, manifest, base = synthetic_control()
    config = Config(dict(base.values, metadata_mode="categories"))
    monkeypatch.setattr(pipeline, "compute_features", lambda *args: (_ for _ in ()).throw(RuntimeError("interrupt features")))
    with pytest.raises(RuntimeError, match="interrupt features"):
        train_run(events, manifest, config, tmp_path, "yambda", {1: ["a"]})
    with pytest.raises(ValueError, match="identity"):
        train_run(events, manifest, config, tmp_path, "yambda", {1: ["b"]})


def test_legacy_frozen_format_does_not_require_resume_checkpoints(partial_run):
    root, events, manifest, config, _, _ = partial_run
    frozen = train_run(events, manifest, config, root, "yambda")
    for path in [root / "run-identity.json", root / "pretest-checkpoint.json", *root.glob("seed-*/checkpoint.json")]:
        frozen["artifacts"].pop(str(path.relative_to(root)))
        path.unlink()
    from tlab_recsys.config import canonical_hash
    frozen.pop("decision_hash")
    frozen["decision_hash"] = canonical_hash(frozen)
    write_json(root / "frozen.json", frozen)
    assert train_run(events, manifest, config, root, "yambda") == frozen
