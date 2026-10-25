import json
from pathlib import Path

import polars as pl
import pytest
import numpy as np
from catboost import CatBoostClassifier, Pool

from tlab_recsys.cli import synthetic_control
from tlab_recsys.config import Config, DAY, canonical_hash, write_json
from tlab_recsys.download import SOURCES, file_hash, verify_manifest, validate_source_file
from tlab_recsys.data import prepare_dataset
import tlab_recsys.data as data_module
from tlab_recsys.modeling import model_frame
from tlab_recsys.evaluation import evaluate
from tlab_recsys.protocol import make_slice
from tlab_recsys.pipeline import evaluate_run, load_frozen, train_run
import tlab_recsys.pipeline as pipeline


def test_download_corruption_and_schema_rejected(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    manifest = {"dataset": "yambda", "revision": SOURCES["yambda"]["revision"], "files": {"flat/50m/multi_event.parquet": {"bytes": 384465869, "sha256": "x"}}}
    with pytest.raises(ValueError, match="integrity"):
        verify_manifest(raw, manifest)
    broken = tmp_path / "broken.parquet"
    pl.DataFrame({"wrong": [1]}).write_parquet(broken)
    with pytest.raises(ValueError, match="schema"):
        validate_source_file(broken, "multi_event.parquet")


def test_preparation_is_reproducible_preserves_source_rows_and_rejects_cache_corruption(tmp_path, monkeypatch):
    events, _, config = synthetic_control()
    source = events.select(pl.col("user_id").alias("uid"), "item_id", "timestamp", "event_type", pl.when(pl.col("is_target")).then(51).otherwise(None).alias("played_ratio_pct"))
    path = tmp_path / "raw" / "flat/50m/multi_event.parquet"
    path.parent.mkdir(parents=True)
    source.write_parquet(path)
    patched = dict(SOURCES["yambda"], files={"flat/50m/multi_event.parquet": path.stat().st_size})
    monkeypatch.setitem(SOURCES, "yambda", patched)
    source_manifest = {"dataset": "yambda", "revision": patched["revision"], "files": {"flat/50m/multi_event.parquet": {"bytes": path.stat().st_size, "sha256": file_hash(path)}}}
    write_json(tmp_path / "source-manifest.json", source_manifest)
    first, manifest = prepare_dataset("yambda", tmp_path, config)
    second, repeated = prepare_dataset("yambda", tmp_path, config)
    assert first.equals(second)
    assert manifest == repeated
    assert manifest["rows"] == source.height
    assert set(first["row_id"].to_list()) == set(range(source.height))
    other_config = Config(dict(config.values, model_seeds=[29]))
    _, other_manifest = prepare_dataset("yambda", tmp_path, other_config)
    assert manifest["users"] == other_manifest["users"]
    assert manifest["row_ids_sha256"] == other_manifest["row_ids_sha256"]
    cache = next((tmp_path / "prepared").glob(f"v3-{config.digest}-*/events.parquet"))
    cache.write_bytes(cache.read_bytes() + b"corrupted")
    with pytest.raises(ValueError, match="cache integrity"):
        prepare_dataset("yambda", tmp_path, config)


def local_yambda_source(root, monkeypatch):
    events, _, config = synthetic_control()
    source = events.select(pl.col("user_id").alias("uid"), "item_id", "timestamp", "event_type", pl.when(pl.col("is_target")).then(51).otherwise(None).alias("played_ratio_pct"))
    path = root / "raw/flat/50m/multi_event.parquet"
    path.parent.mkdir(parents=True)
    source.write_parquet(path)
    monkeypatch.setitem(SOURCES, "yambda", dict(SOURCES["yambda"], files={"flat/50m/multi_event.parquet": path.stat().st_size}))
    manifest = {"dataset": "yambda", "revision": SOURCES["yambda"]["revision"], "files": {"flat/50m/multi_event.parquet": {"bytes": path.stat().st_size, "sha256": file_hash(path)}}}
    write_json(root / "source-manifest.json", manifest)
    return config, manifest


def test_preparation_rejects_stale_adapter_receipt_and_changes_cache_identity(tmp_path, monkeypatch):
    config, _ = local_yambda_source(tmp_path, monkeypatch)
    original_events, original_manifest = prepare_dataset("yambda", tmp_path, config)
    receipt = next((tmp_path / "prepared").glob("*/manifest.json"))
    stale = dict(original_manifest, preparation_hash="old-adapter")
    stale["manifest_hash"] = canonical_hash({key: value for key, value in stale.items() if key != "manifest_hash"})
    write_json(receipt, stale)
    with pytest.raises(ValueError, match="cache integrity"):
        prepare_dataset("yambda", tmp_path, config)
    write_json(receipt, original_manifest)
    original_adapter = data_module.adapt_yambda
    monkeypatch.setattr(data_module, "preparation_fingerprint", lambda: "f" * 64)
    monkeypatch.setattr(data_module, "adapt_yambda", lambda frame, threshold: original_adapter(frame, threshold).with_columns((~pl.col("is_target")).alias("is_target")))
    changed_events, changed_manifest = prepare_dataset("yambda", tmp_path, config)
    assert changed_manifest["preparation_hash"] != original_manifest["preparation_hash"]
    assert not changed_events["is_target"].equals(original_events["is_target"])
    assert len(list((tmp_path / "prepared").glob("*/manifest.json"))) == 2


def test_none_mode_source_identity_ignores_optional_metadata_and_its_corruption(tmp_path, monkeypatch):
    config, source = local_yambda_source(tmp_path, monkeypatch)
    first, manifest = prepare_dataset("yambda", tmp_path, config)
    optional = tmp_path / "raw/artist_item_mapping.parquet"
    pl.DataFrame({"item_id": [1, 2], "artist_id": [3, 4]}).write_parquet(optional)
    monkeypatch.setitem(SOURCES["yambda"], "optional_files", {"artist_item_mapping.parquet": optional.stat().st_size})
    source["files"]["artist_item_mapping.parquet"] = {"bytes": optional.stat().st_size, "sha256": file_hash(optional)}
    write_json(tmp_path / "source-manifest.json", source)
    second, repeated = prepare_dataset("yambda", tmp_path, config)
    assert first.equals(second) and manifest == repeated
    optional.write_bytes(optional.read_bytes() + b"changed unused metadata")
    _, after_corruption = prepare_dataset("yambda", tmp_path, config)
    assert after_corruption == manifest
    with pytest.raises(ValueError, match="integrity"):
        prepare_dataset("yambda", tmp_path, Config(dict(config.values, metadata_mode="categories")))


@pytest.fixture(scope="module")
def six_user_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("six-user-run")
    _, _, base = synthetic_control()
    values = dict(base.values, cohort_size=6, full_catalog_users=6, popular_candidates=8, recent_candidates=0)
    values["catboost"] = dict(values["catboost"], depth=[2], learning_rate=[0.2], l2_leaf_reg=[3], max_iterations=12, early_stopping_rounds=3)
    values["search"] = {"max_fits": 1, "max_groups": 1, "min_improvement": 0.0001}
    config = Config(values)
    records = []
    for day in range(70):
        for user in range(1, 7):
            records.extend([(len(records), user, (day // 7 + user) % 12, day * DAY, "listen", True),
                            (len(records) + 1, user, 20 + user, day * DAY, "like", False)])
    records.extend([(len(records), 1, 99, 0, "listen", True), (len(records) + 1, 6, 99, 64 * DAY, "listen", True)])
    events = pl.DataFrame(records, schema={"row_id": pl.UInt32, "user_id": pl.Int64, "item_id": pl.Int64, "timestamp": pl.Int64, "event_type": pl.String, "is_target": pl.Boolean}, orient="row")
    manifest = {"dataset": "synthetic", "users": list(range(1, 7)), "full_catalog_users": list(range(1, 7)), "end_exclusive": 70 * DAY}
    write_json(root / "failure.json", {"message": "old recoverable diagnostic"})
    frozen = train_run(events, manifest, config, root, "yambda")
    result = evaluate_run(events, manifest, config, root)
    return root, events, manifest, config, frozen, result


def test_mutable_failure_diagnostic_does_not_invalidate_recovered_run(six_user_run):
    root, _, manifest, config, frozen, _ = six_user_run
    assert "failure.json" not in frozen["artifacts"]
    write_json(root / "failure.json", {"message": "later failure overwritten by CLI"})
    assert load_frozen(root, config, manifest)["decision_hash"] == frozen["decision_hash"]


def test_saved_refit_matches_independent_training_on_all_five_slices(six_user_run):
    root, _, _, config, frozen, _ = six_user_run
    slices = [pl.read_parquet(root / "slices" / f"{entry['split']}-{entry['cutoff']}.parquet") for entry in frozen["pretest_slices"]]
    all_rows = pl.concat(slices)
    train_rows = pl.concat(slices[:3])
    decision = frozen["decisions"][0]
    record = decision["models"]["manual"]
    columns = record["columns"]
    frame = model_frame(all_rows.select(columns))
    evaluation_pool = Pool(frame, cat_features=["user_id", "item_id"])
    params = dict(decision["parameters"], loss_function="Logloss", task_type="CPU", thread_count=config["catboost"]["thread_count"], random_seed=decision["model_seed"], verbose=False, allow_writing_files=False)
    independent = CatBoostClassifier(**params)
    independent.fit(Pool(frame, label=all_rows["label"].to_numpy(), cat_features=["user_id", "item_id"]))
    saved = CatBoostClassifier()
    saved.load_model(str(root / record["path"]))
    actual = saved.predict_proba(evaluation_pool, thread_count=2)
    np.testing.assert_allclose(actual, independent.predict_proba(evaluation_pool, thread_count=2), rtol=0, atol=1e-12)
    train_only = CatBoostClassifier(**params)
    train_only.fit(Pool(model_frame(train_rows.select(columns)), label=train_rows["label"].to_numpy(), cat_features=["user_id", "item_id"]))
    assert not np.allclose(actual, train_only.predict_proba(evaluation_pool, thread_count=2), rtol=0, atol=1e-8)


def test_full_catalog_crosses_batch_boundary_and_preserves_exact_allowed_pairs(six_user_run):
    root, events, manifest, _, frozen, result = six_user_run
    q = frozen["boundaries"][-1]["cutoff"]
    catalog = set(events.filter(pl.col("timestamp") < q)["item_id"].to_list())
    expected = {(user, item) for user in manifest["users"] for item in catalog}
    full = pl.read_parquet(root / "test-full-catalog-rows.parquet")
    limited = pl.read_parquet(root / "test-limited-candidates-rows.parquet")
    assert set(full.select("user_id", "item_id").iter_rows()) == expected
    assert full.height == len(expected)
    assert set(full["user_id"].to_list()) == set(range(1, 7))
    assert (6, 99) in expected and (6, 99) not in set(limited.select("user_id", "item_id").iter_rows())
    full_metrics = result["catalogs"]["full_catalog"]["seeds"][0]["manual"]["metrics"]
    limited_metrics = result["catalogs"]["limited_candidates"]["seeds"][0]["manual"]["metrics"]
    assert full_metrics["evaluated_users"] == 6
    assert full_metrics["candidate_coverage"] > limited_metrics["candidate_coverage"]
    assert full_metrics["recall_20"] > limited_metrics["recall_20"]


def test_kion_full_catalog_excludes_seen_pairs_across_six_user_batches(tmp_path, six_user_run):
    _, events, manifest, config, frozen, _ = six_user_run
    q = frozen["boundaries"][-1]["cutoff"]
    past = events.filter(pl.col("timestamp") < q)
    catalog = set(past["item_id"].to_list())
    seen = {user: set(past.filter(pl.col("user_id") == user)["item_id"].to_list()) for user in manifest["users"]}
    expected = {(user, item) for user in manifest["users"] for item in catalog - seen[user]}
    pipeline._evaluate_chunks(events, manifest["users"], q, "kion", config, [], None, {"models": {}}, tmp_path, True)
    rows = pl.read_parquet(tmp_path / "test-full-catalog-rows.parquet")
    assert set(rows.select("user_id", "item_id").iter_rows()) == expected
    assert rows.height == len(expected)
    assert 6 in rows["user_id"] and (6, 99) in expected


@pytest.mark.parametrize("mode,categories", [("none", {1: ("a",)}), ("categories", None)])
def test_category_argument_must_match_mode_before_any_work(tmp_path, mode, categories):
    events, manifest, base = synthetic_control()
    config = Config(dict(base.values, metadata_mode=mode))
    with pytest.raises(ValueError, match="metadata_mode"):
        train_run(events, manifest, config, tmp_path / "invalid", "yambda", categories)
    with pytest.raises(ValueError, match="metadata_mode"):
        evaluate_run(events, manifest, config, tmp_path / "invalid", categories)
    assert not (tmp_path / "invalid").exists()


def test_actual_category_contents_are_bound_and_checked_before_scoring(tmp_path, six_user_run, monkeypatch):
    _, events, manifest, base, _, _ = six_user_run
    values = dict(base.values, metadata_mode="categories", ablation_manual_selected=False)
    values["catboost"] = dict(values["catboost"], max_iterations=2, early_stopping_rounds=1)
    values["search"] = {"max_fits": 0, "max_groups": 0, "min_improvement": 0.0001}
    config = Config(values)
    categories = {item: [f"category-{item % 2}"] for item in events["item_id"].unique().to_list()}
    frozen = train_run(events, manifest, config, tmp_path, "yambda", categories)
    assert frozen["categories_hash"] is not None
    assert load_frozen(tmp_path, config, manifest, dict(reversed(list(categories.items()))))["decision_hash"] == frozen["decision_hash"]
    assert evaluate_run(events, manifest, config, tmp_path, categories)["decision_hash"] == frozen["decision_hash"]
    changed = dict(categories)
    changed[99] = ("changed",)
    monkeypatch.setattr(pipeline, "_evaluate_chunks", lambda *args: pytest.fail("Changed categories reached scoring"))
    with pytest.raises(ValueError, match="Category contents"):
        evaluate_run(events, manifest, config, tmp_path, changed)
    with pytest.raises(ValueError, match="Category contents"):
        train_run(events, manifest, config, tmp_path, "yambda", changed)


def test_selection_uses_selection_frame_and_scores_with_opposing_dev_preferences(tmp_path, monkeypatch):
    _, _, base = synthetic_control()
    values = dict(base.values, cohort_size=1, full_catalog_users=1, horizon_days=1,
                  popular_candidates=2, recent_candidates=0, manual_windows_days=[None], generated_windows_days=[1, 3])
    values["search"] = {"max_fits": 2, "max_groups": 1, "min_improvement": 0.0001}
    config = Config(values)
    records = [(0, 1, 10, 0, "listen", True), (1, 1, 20, 0, "listen", True)]
    records += [(len(records) + i, 1, item, day * DAY, "listen", True) for i, (day, item) in enumerate([(1, 10), (2, 20), (3, 10), (4, 10), (5, 20)])]
    records.append((len(records), 1, 10, 7 * DAY - 5, "listen", True))
    events = pl.DataFrame(records, schema={"row_id": pl.UInt32, "user_id": pl.Int64, "item_id": pl.Int64, "timestamp": pl.Int64, "event_type": pl.String, "is_target": pl.Boolean}, orient="row")
    manifest = {"dataset": "synthetic", "users": [1], "full_catalog_users": [1], "end_exclusive": 7 * DAY}
    dev = make_slice(events, [1], "dev", 4 * DAY, "yambda", config)
    selection = make_slice(events, [1], "selection", 5 * DAY, "yambda", config)
    first = np.array([0.9, 0.1])
    second = first[::-1]
    assert evaluate(dev, first)["metrics"]["ndcg_10"] > evaluate(dev, second)["metrics"]["ndcg_10"]
    assert evaluate(selection, first)["metrics"]["ndcg_10"] < evaluate(selection, second)["metrics"]["ndcg_10"]

    class ControlledModel:
        def __init__(self, columns):
            self.columns = columns

        def predict_proba(self, frame, **kwargs):
            assert (frame["window_all__all__available_period"] == 5 * DAY).all(), "Scorer read dev features"
            probabilities = second if any(name.startswith("window_003d") for name in self.columns) else first if any(name.startswith("window_001d") for name in self.columns) else np.array([0.5, 0.5])
            return np.column_stack([1 - probabilities, probabilities])

        def save_model(self, path):
            Path(path).write_bytes(b"controlled selection integration fixture")

    monkeypatch.setattr(pipeline, "tune_manual", lambda *args: {"parameters": {"depth": 2, "learning_rate": 0.1, "l2_leaf_reg": 3, "iterations": 1}, "fits": 1, "trace": []})
    monkeypatch.setattr(pipeline, "fit_fixed", lambda train, labels, columns, *args: ControlledModel(columns))
    monkeypatch.setattr(pipeline, "pool", lambda frame, *args: frame)
    frozen = train_run(events, manifest, config, tmp_path, "yambda")
    saved = json.loads((tmp_path / "seed-17/selection.json").read_text(encoding="utf-8"))
    baseline = 1 / np.log2(3)
    assert [entry["score"] for entry in saved["trace"]] == pytest.approx([baseline, baseline, 1])
    assert saved["groups"] == ["window_003d"]
    assert frozen["decisions"][0]["selected_groups"] == ["window_003d"]


def test_p6_orchestration_never_reads_test_until_frozen_and_relabels_do_not_change_selection(tmp_path, monkeypatch):
    events, manifest, config = synthetic_control()
    values = dict(config.values)
    values["catboost"] = dict(values["catboost"], max_iterations=2, early_stopping_rounds=1)
    values["search"] = {"max_fits": 1, "max_groups": 1, "min_improvement": 0.0001}
    config = Config(values)
    actual_make_slice = pipeline.make_slice
    reads = []
    def guarded(events_, users, split, *args, **kwargs):
        reads.append(split)
        assert split != "test", "Test was accessed before freeze"
        return actual_make_slice(events_, users, split, *args, **kwargs)
    monkeypatch.setattr(pipeline, "make_slice", guarded)
    original = train_run(events, manifest, config, tmp_path / "one", "yambda")
    test_q = original["boundaries"][-1]["cutoff"]
    changed = events.with_columns(pl.when(pl.col("timestamp") >= test_q).then(~pl.col("is_target")).otherwise(pl.col("is_target")).alias("is_target"))
    repeated = train_run(changed, manifest, config, tmp_path / "two", "yambda")
    assert reads == ["train", "train", "train", "dev", "selection"] * 2
    assert original["pretest_slices"] == repeated["pretest_slices"]
    for first, second in zip(original["decisions"], repeated["decisions"]):
        assert first["parameters"] == second["parameters"]
        assert first["selected_groups"] == second["selected_groups"]
        assert first["fit_counts"] == second["fit_counts"]
        assert first["refit"]["original_row_cutoffs_preserved"]
    monkeypatch.setattr(pipeline, "make_slice", actual_make_slice)
    result = evaluate_run(events, manifest, config, tmp_path / "one")
    assert set(result["catalogs"]) == {"limited_candidates", "full_catalog"}
    assert load_frozen(tmp_path / "one", config, manifest)["decision_hash"] == original["decision_hash"]
    actual_fingerprint = pipeline.runtime_fingerprint
    for field in ("code_sha256", "uv_lock_sha256", "dependencies"):
        changed_runtime = dict(actual_fingerprint())
        changed_runtime[field] = "changed"
        monkeypatch.setattr(pipeline, "runtime_fingerprint", lambda: changed_runtime)
        with pytest.raises(ValueError, match="incompatible"):
            load_frozen(tmp_path / "one", config, manifest)
        monkeypatch.setattr(pipeline, "runtime_fingerprint", actual_fingerprint)
    with pytest.raises(ValueError, match="requires frozen"):
        evaluate_run(events, manifest, config, tmp_path / "absent")
    record = next(iter(original["decisions"][0]["models"].values()))
    model_path = tmp_path / "one" / record["path"]
    model_path.write_bytes(model_path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        load_frozen(tmp_path / "one", config, manifest)
