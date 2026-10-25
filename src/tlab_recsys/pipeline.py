from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import platform
import sys
import time

from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
import polars as pl

from .config import Config, DAY, EVALUATION_BATCH_USERS, canonical_hash, require_integer, write_json
from .checkpoints import artifact_hashes, read_checkpoint, verify_artifacts, write_checkpoint
from .download import file_hash
from .evaluation import ResourceMeasure, artifact_bytes, bootstrap_contrast, evaluate
from .features import compute_features, feature_groups, grammar
from .modeling import CORE, fit_fixed, model_frame, pool, tune_manual
from .protocol import Slice, cutoffs, make_slice, table_hash
from .selection import greedy_select
from .reproducibility import runtime_fingerprint


def _columns(groups: list[str] | tuple[str, ...], mapping: dict[str, list[str]]) -> list[str]:
    return list(dict.fromkeys(CORE + [name for group in groups for name in mapping[group]]))


def category_hash(config: Config, categories: dict | None) -> str | None:
    if (config["metadata_mode"] == "categories") != (categories is not None):
        raise ValueError("Category argument must agree with metadata_mode")
    if categories is None:
        return None
    if not isinstance(categories, dict):
        raise ValueError("Categories must be a mapping of integer item IDs to category sequences")
    for item, values in categories.items():
        require_integer(item, "category item ID")
        if not isinstance(values, (list, tuple)) or not values or any(not isinstance(value, str) or not value for value in values):
            raise ValueError("Each item must have a nonempty sequence of category strings")
    return canonical_hash([[int(item), sorted(set(values))] for item, values in sorted(categories.items())])


def _require_runtime(fingerprint: dict) -> None:
    if runtime_fingerprint() != fingerprint:
        raise ValueError("Code, lockfile or dependency versions changed during training; use a new run directory")


def _verify_pretest_checkpoint(run_root: Path, accepted: dict, digest: str) -> None:
    path = run_root / "pretest-checkpoint.json"
    if read_checkpoint(path, "pretest") != accepted or file_hash(path) != digest:
        raise ValueError("Pretest checkpoint changed during training")
    verify_artifacts(run_root, accepted["artifacts"])


def _slice_metadata(current: Slice) -> dict:
    return {"split": current.split, "cutoff": current.cutoff, "rows": current.rows.height,
            "candidate_label_hash": table_hash(current.rows), "users": current.users,
            "targets": [[user, sorted(items)] for user, items in current.targets.items()],
            "catalog": sorted(current.catalog),
            "catalog_targets": [[user, sorted(items)] for user, items in current.catalog_targets.items()] if current.catalog_targets is not None else None}


def _read_pretest(run_root: Path, record: dict, specs: list) -> tuple[list, list]:
    slices, frames = [], []
    row_columns = ["query_id", "user_id", "item_id", "cutoff", "popularity", "label"]
    for entry in record["slices"]:
        table = pl.read_parquet(run_root / "slices" / f"{entry['split']}-{entry['cutoff']}.parquet")
        rows = table.select(row_columns)
        if rows.height != entry["rows"] or table_hash(rows) != entry["candidate_label_hash"]:
            raise ValueError("Pretest candidate/label integrity mismatch")
        slices.append(Slice(entry["split"], entry["cutoff"], rows, {user: set(items) for user, items in entry["targets"]},
                            set(entry["catalog"]), entry["users"],
                            {user: set(items) for user, items in entry["catalog_targets"]} if entry["catalog_targets"] is not None else None))
        frames.append(model_frame(table.select(CORE + [spec.name for spec in specs])))
    return slices, frames


def _seed_checkpoint(run_root: Path, seed: int, identity_hash: str, pretest_hash: str) -> tuple[dict, dict] | None:
    seed_root = run_root / f"seed-{seed}"
    path = seed_root / "checkpoint.json"
    if not path.exists():
        return None
    checkpoint = read_checkpoint(path, "seed")
    if checkpoint.get("identity_hash") != identity_hash or checkpoint.get("pretest_checkpoint_hash") != pretest_hash or checkpoint.get("model_seed") != seed:
        raise ValueError(f"Seed checkpoint identity mismatch: {seed}")
    verify_artifacts(run_root, checkpoint["artifacts"])
    decision = json.loads((seed_root / "decision.json").read_text(encoding="utf-8"))
    expected = {str((seed_root / name).relative_to(run_root)) for name in ("tuning.json", "selection.json", "decision.json")}
    for model in decision["models"].values():
        if Path(model["path"]).parent != Path(f"seed-{seed}") or checkpoint["artifacts"].get(model["path"]) != model["sha256"]:
            raise ValueError(f"Seed model identity mismatch: {seed}")
        expected.add(model["path"])
    if decision["model_seed"] != seed or set(checkpoint["artifacts"]) != expected:
        raise ValueError(f"Seed checkpoint artifact set mismatch: {seed}")
    if {str(path.relative_to(run_root)) for path in seed_root.glob("*.cbm")} != {model["path"] for model in decision["models"].values()}:
        raise ValueError(f"Unexpected seed model artifact: {seed}")
    return decision, checkpoint


def train_run(events: pl.DataFrame, data_manifest: dict, config: Config, run_root: Path, dataset: str, categories: dict | None = None) -> dict:
    config.validate()
    categories_digest = category_hash(config, categories)
    if categories is not None:
        categories = {int(item): tuple(sorted(set(values))) for item, values in categories.items()}
    run_root.mkdir(parents=True, exist_ok=True)
    if (run_root / "frozen.json").exists():
        frozen = load_frozen(run_root, config, data_manifest, categories)
        if frozen.get("protocol_dataset", frozen["dataset"]) != dataset:
            raise ValueError("Frozen protocol dataset mismatch")
        return frozen
    fingerprint = runtime_fingerprint()
    identity = {"config_hash": config.digest, "data_manifest_hash": canonical_hash(data_manifest),
                "protocol_dataset": dataset, "categories_hash": categories_digest, "runtime_fingerprint": fingerprint}
    identity_hash = canonical_hash(identity)
    identity_path = run_root / "run-identity.json"
    if identity_path.exists():
        if read_checkpoint(identity_path, "run-identity").get("identity") != identity:
            raise ValueError("Run identity mismatch: config/data/protocol/categories/runtime changed")
    else:
        # Diagnostics may precede the first attempt; old computation without an identity is ambiguous.
        existing = [path for path in run_root.iterdir() if path.name not in {"failure.json", "ingestion-resources.json", "diagnostics"} and not path.name.endswith(".tmp")]
        if existing:
            raise ValueError("Ambiguous partial run without run-identity.json; use a new run directory")
        write_checkpoint(identity_path, "run-identity", {"identity": identity})
    users = data_manifest["users"]
    boundaries = cutoffs(data_manifest["end_exclusive"], config["horizon_days"] * DAY)
    specs = grammar(list(dict.fromkeys(config["generated_windows_days"] + config["manual_windows_days"])), categories is not None)
    manual = [spec.name for spec in grammar(config["manual_windows_days"], categories is not None)]
    generated = [spec.name for spec in grammar(config["generated_windows_days"], categories is not None)]
    mapping = feature_groups([spec for spec in specs if spec.name in generated])
    base_paths = ["run-identity.json", "runtime-fingerprint.json", "config.json", "data-manifest.json", "boundaries.json", "grammar.json"]
    pretest_paths = base_paths + [str(Path("slices") / f"{split}-{q}.parquet") for split, q in boundaries if split != "test"]
    pretest_path = run_root / "pretest-checkpoint.json"
    completed = {}
    for seed_root in run_root.glob("seed-*"):
        if seed_root.name not in {f"seed-{seed}" for seed in config["model_seeds"]}:
            raise ValueError(f"Unexpected seed directory: {seed_root.name}")
    if pretest_path.exists():
        pretest = read_checkpoint(pretest_path, "pretest")
        if pretest.get("identity_hash") != identity_hash or set(pretest["artifacts"]) != set(pretest_paths):
            raise ValueError("Pretest checkpoint identity/artifact set mismatch")
        verify_artifacts(run_root, pretest["artifacts"])
        if [(entry["split"], entry["cutoff"]) for entry in pretest["slices"]] != [(split, q) for split, q in boundaries if split != "test"]:
            raise ValueError("Pretest checkpoint boundaries mismatch")
        # Check every completed seed before any fit or overwrite, including later seeds.
        for seed in config["model_seeds"]:
            saved = _seed_checkpoint(run_root, seed, identity_hash, pretest["checkpoint_hash"])
            if saved is not None:
                completed[seed] = saved
        if len(completed) != len(config["model_seeds"]):
            slices, frames = _read_pretest(run_root, pretest, specs)
    else:
        if any(run_root.glob("seed-*/*")):
            raise ValueError("Seed artifacts exist without a completed pretest checkpoint")
        write_json(run_root / "runtime-fingerprint.json", fingerprint)
        write_json(run_root / "config.json", config.values)
        write_json(run_root / "data-manifest.json", data_manifest)
        write_json(run_root / "boundaries.json", [{"split": split, "cutoff": q, "end": q + config["horizon_days"] * DAY} for split, q in boundaries])
        write_json(run_root / "grammar.json", [asdict(spec) | {"name": spec.name, "group": spec.group} for spec in specs])
        frames, slices = [], []
        with ResourceMeasure() as feature_measure:
            for split, q in boundaries:
                if split == "test":
                    continue
                current = make_slice(events, users, split, q, dataset, config)
                features = compute_features(events, current.rows, q, specs, config, categories)
                artifact = run_root / "slices" / f"{split}-{q}.parquet"
                artifact.parent.mkdir(parents=True, exist_ok=True)
                temporary = artifact.with_suffix(".parquet.tmp")
                current.rows.hstack(features.drop("user_id", "item_id")).write_parquet(temporary)
                temporary.replace(artifact)
                slices.append(current)
                frames.append(model_frame(features))
        _require_runtime(fingerprint)
        artifacts = artifact_hashes(run_root, pretest_paths)
        verify_artifacts(run_root, artifacts)
        _require_runtime(fingerprint)
        pretest = write_checkpoint(pretest_path, "pretest", {"identity_hash": identity_hash, "artifacts": artifacts,
                                                             "slices": [_slice_metadata(current) for current in slices], "features_resources": feature_measure.result})
    slice_hashes = [{key: entry[key] for key in ("split", "cutoff", "rows", "candidate_label_hash")} for entry in pretest["slices"]]
    pretest_digest = file_hash(pretest_path)
    if len(completed) != len(config["model_seeds"]):
        train = pd.concat(frames[:3], ignore_index=True)
        labels = np.concatenate([slice_.rows["label"].to_numpy() for slice_ in slices[:3]])
        refit = pd.concat(frames, ignore_index=True)
        refit_labels = np.concatenate([slice_.rows["label"].to_numpy() for slice_ in slices])
        if slices[4].rows.is_empty() or not any(slices[4].targets.values()):
            raise ValueError("Selection NDCG@10 is undefined: no candidates or no target users")
    decisions = []
    for seed in config["model_seeds"]:
        if seed in completed:
            print(f"{dataset}: model_seed={seed}: verified checkpoint; skip completed seed", flush=True)
            decisions.append(completed[seed][0])
            continue
        _require_runtime(fingerprint)
        print(f"{dataset}: model_seed={seed}: dev tuning", flush=True)
        seed_root = run_root / f"seed-{seed}"
        seed_root.mkdir(exist_ok=True)
        with ResourceMeasure() as tuning_measure:
            tuning = tune_manual(train, labels, frames[3], slices[3], CORE + manual, config, seed)
        write_json(seed_root / "tuning.json", tuning | {"resources": tuning_measure.result})
        parameters = tuning["parameters"]
        print(f"{dataset}: model_seed={seed}: feature selection", flush=True)
        fit_costs = []

        def scorer(selected_groups: tuple[str, ...]) -> float | None:
            columns = _columns(selected_groups, mapping)
            started = time.perf_counter()
            model = fit_fixed(train, labels, columns, parameters, config, seed)
            scores = model.predict_proba(pool(frames[4], columns), thread_count=config["catboost"]["thread_count"])[:, 1]
            metric = evaluate(slices[4], scores)["metrics"]["ndcg_10"]
            fit_costs.append({"groups": list(selected_groups), "wall_seconds": time.perf_counter() - started, "features": len(columns)})
            print(f"selection fit {len(fit_costs)}: groups={list(selected_groups)}, NDCG@10={metric}", flush=True)
            return metric

        with ResourceMeasure() as search_measure:
            selection = greedy_select(list(mapping), scorer, **config["search"])
        write_json(seed_root / "selection.json", selection | {"fit_costs": fit_costs, "resources": search_measure.result})
        variants = {"manual": CORE + manual, "generated": CORE + generated, "selected": _columns(selection["groups"], mapping)}
        if config["ablation_manual_selected"]:
            variants["manual_selected"] = list(dict.fromkeys(CORE + manual + _columns(selection["groups"], mapping)))
        print(f"{dataset}: model_seed={seed}: final refits", flush=True)
        models = {}
        with ResourceMeasure() as refit_measure:
            for variant, columns in variants.items():
                model = fit_fixed(refit, refit_labels, columns, parameters, config, seed)
                path = seed_root / f"{variant}.cbm"
                model.save_model(str(path))
                models[variant] = {"path": str(path.relative_to(run_root)), "sha256": file_hash(path), "columns": columns, "feature_count": len(columns)}
        decision = {"model_seed": seed, "parameters": parameters, "selected_groups": selection["groups"], "models": models,
                    "fit_counts": {"dev": tuning["fits"], "search_initial_core": 1, "search_additional": selection["additional_fits"], "final_refits": len(variants), "total": tuning["fits"] + 1 + selection["additional_fits"] + len(variants)},
                    "refit": {"splits": ["train", "dev", "selection"], "early_stopping": False, "original_row_cutoffs_preserved": True, "resources": refit_measure.result}}
        write_json(seed_root / "decision.json", decision)
        _require_runtime(fingerprint)
        seed_paths = [str((seed_root / name).relative_to(run_root)) for name in ("tuning.json", "selection.json", "decision.json")]
        artifacts = artifact_hashes(run_root, seed_paths + [model["path"] for model in models.values()])
        verify_artifacts(run_root, artifacts)
        _verify_pretest_checkpoint(run_root, pretest, pretest_digest)
        _require_runtime(fingerprint)
        write_checkpoint(seed_root / "checkpoint.json", "seed", {"model_seed": seed, "identity_hash": identity_hash,
                                                                   "pretest_checkpoint_hash": pretest["checkpoint_hash"], "artifacts": artifacts})
        decisions.append(decision)
    _require_runtime(fingerprint)
    _verify_pretest_checkpoint(run_root, pretest, pretest_digest)
    frozen_artifacts = dict(pretest["artifacts"])
    frozen_artifacts["pretest-checkpoint.json"] = pretest_digest
    for seed in config["model_seeds"]:
        _, checkpoint = _seed_checkpoint(run_root, seed, identity_hash, pretest["checkpoint_hash"])
        frozen_artifacts.update(checkpoint["artifacts"])
        frozen_artifacts.update(artifact_hashes(run_root, [str(Path(f"seed-{seed}") / "checkpoint.json")]))
    frozen = {"schema_version": 1, "dataset": data_manifest.get("dataset", dataset), "protocol_dataset": dataset, "config_hash": config.digest, "data_manifest_hash": canonical_hash(data_manifest), "cohort_users": users, "full_catalog_users": data_manifest["full_catalog_users"],
              "boundaries": [{"split": split, "cutoff": q} for split, q in boundaries], "pretest_slices": slice_hashes, "decisions": decisions, "features_resources": pretest["features_resources"],
              "metadata_assumption": "Static categories assumed immutable and historically available" if categories is not None else "No static metadata",
              "categories_hash": categories_digest,
              "runtime": {"python": sys.version, "platform": platform.platform()}, "runtime_fingerprint": fingerprint,
              "artifacts": frozen_artifacts}
    frozen["decision_hash"] = canonical_hash(frozen)
    write_json(run_root / "frozen.json", frozen)
    return frozen


def load_frozen(run_root: Path, config: Config, data_manifest: dict, categories: dict | None = None) -> dict:
    config.validate()
    categories_digest = category_hash(config, categories)
    path = run_root / "frozen.json"
    if not path.exists():
        raise ValueError("Evaluation requires frozen.json; train and freeze first")
    frozen = json.loads(path.read_text(encoding="utf-8"))
    declared = frozen.pop("decision_hash")
    if canonical_hash(frozen) != declared or frozen["config_hash"] != config.digest or frozen["data_manifest_hash"] != canonical_hash(data_manifest):
        raise ValueError("Frozen decision/config/data integrity mismatch")
    if frozen.get("categories_hash") != categories_digest:
        raise ValueError("Category contents differ from the frozen decision")
    if frozen.get("runtime_fingerprint") != runtime_fingerprint():
        raise ValueError("Frozen decision is incompatible with current code, lockfile or dependency versions")
    for relative, digest in frozen["artifacts"].items():
        if file_hash(run_root / relative) != digest:
            raise ValueError(f"Frozen artifact changed: {relative}")
    frozen["decision_hash"] = declared
    return frozen


def _evaluate_chunks(events, users, q, dataset, config, specs, categories, decision, run_root, full):
    per_variant = {"popularity": [], **{variant: [] for variant in decision["models"]}}
    all_rows, targets, catalog, catalog_targets = [], {}, set(), {}
    score_parts = {variant: [] for variant in per_variant}
    models = {}
    for variant, record in decision["models"].items():
        model = CatBoostClassifier()
        model.load_model(str(run_root / record["path"]))
        models[variant] = model
    batch_size = EVALUATION_BATCH_USERS if full else len(users)
    for offset in range(0, len(users), batch_size):
        batch = users[offset:offset + batch_size]
        current = make_slice(events, batch, "test", q, dataset, config, full)
        features = model_frame(compute_features(events, current.rows, q, specs, config, categories))
        all_rows.append(current.rows)
        targets.update(current.targets)
        catalog.update(current.catalog)
        catalog_targets.update(current.catalog_targets)
        popularity = current.rows["popularity"].to_numpy()
        score_parts["popularity"].append(popularity)
        for variant, model in models.items():
            record = decision["models"][variant]
            scores = model.predict_proba(pool(features, record["columns"]), thread_count=config["catboost"]["thread_count"])[:, 1] if len(features) else np.array([])
            score_parts[variant].append(scores)
    rows = pl.concat(all_rows)
    rows_path = run_root / ("test-full-catalog-rows.parquet" if full else "test-limited-candidates-rows.parquet")
    if rows_path.exists():
        if table_hash(pl.read_parquet(rows_path)) != table_hash(rows):
            raise ValueError("Saved test candidate/label rows changed for frozen decision")
    else:
        rows.write_parquet(rows_path)
    merged = Slice("test", q, rows, targets, catalog, users, catalog_targets)
    results = {variant: evaluate(merged, np.concatenate(parts), probabilities=variant != "popularity") for variant, parts in score_parts.items()}
    return results, table_hash(rows)


def evaluate_run(events: pl.DataFrame, data_manifest: dict, config: Config, run_root: Path, categories: dict | None = None) -> dict:
    frozen = load_frozen(run_root, config, data_manifest, categories)
    if categories is not None:
        categories = {int(item): tuple(sorted(set(values))) for item, values in categories.items()}
    q = next(row["cutoff"] for row in frozen["boundaries"] if row["split"] == "test")
    specs = grammar(list(dict.fromkeys(config["generated_windows_days"] + config["manual_windows_days"])), categories is not None)
    catalog_results = {}
    with ResourceMeasure() as measure:
        for scope, users, full in (("limited_candidates", frozen["cohort_users"], False), ("full_catalog", frozen["full_catalog_users"], True)):
            seeds = []
            shared_hash = None
            for decision in frozen["decisions"]:
                print(f"{frozen['dataset']}: {scope}: seed={decision['model_seed']}: test evaluation", flush=True)
                result, row_hash = _evaluate_chunks(events, users, q, frozen.get("protocol_dataset", frozen["dataset"]), config, specs, categories, decision, run_root, full)
                if shared_hash is not None and row_hash != shared_hash:
                    raise ValueError("Candidate/label rows changed between model seeds")
                shared_hash = row_hash
                seeds.append({"model_seed": decision["model_seed"], **result})
            catalog_results[scope] = {"users": users, "candidate_label_hash": shared_hash, "seeds": seeds, "contrast": bootstrap_contrast(seeds, **config["bootstrap"])}
    result = {"dataset": frozen["dataset"], "decision_hash": frozen["decision_hash"], "catalogs": catalog_results, "resources": measure.result, "artifact_bytes_before_results": artifact_bytes(run_root)}
    write_json(run_root / "results.json", result)
    return result
