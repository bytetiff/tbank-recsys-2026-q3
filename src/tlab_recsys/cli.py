from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback
from uuid import uuid4

import polars as pl

from .config import Config, DAY, ROOT, canonical_hash, write_json
from .data import load_categories, prepare_dataset
from .download import download_dataset
from .evaluation import ResourceMeasure
from .pipeline import evaluate_run, train_run
from .reporting import create_report


def synthetic_control() -> tuple[pl.DataFrame, dict, Config]:
    config_values = dict(Config.load().values)
    config_values.update(cohort_size=4, popular_candidates=8, recent_candidates=2, full_catalog_users=4, model_seeds=[17], generated_windows_days=[1, 7, None], manual_windows_days=[7, None])
    config_values["catboost"] = dict(config_values["catboost"], max_iterations=8, early_stopping_rounds=3, thread_count=2)
    config_values["search"] = {"max_fits": 3, "max_groups": 2, "min_improvement": 0.0001}
    records = []
    for day in range(70):
        for user in range(1, 5):
            item = (day // 7 + user) % 12
            records.append((len(records), user, item, day * DAY, "listen", True))
            records.append((len(records), user, 20 + user, day * DAY, "like", False))
    events = pl.DataFrame(records, schema={"row_id": pl.UInt32, "user_id": pl.Int64, "item_id": pl.Int64, "timestamp": pl.Int64, "event_type": pl.String, "is_target": pl.Boolean}, orient="row")
    manifest = {"dataset": "synthetic", "users": [1, 2, 3, 4], "full_catalog_users": [1, 2, 3, 4], "end_exclusive": 70 * DAY, "events_hash": canonical_hash(events.to_dicts())}
    return events, manifest, Config(config_values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Сравнение временных признаков KION/Yambda")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("download", "prepare", "train", "evaluate", "run"):
        child = subparsers.add_parser(command)
        child.add_argument("--dataset", choices=["kion", "yambda"], required=True)
        child.add_argument("--profile", choices=["pilot", "expanded"], default="pilot")
        child.add_argument("--config", type=Path)
        child.add_argument("--run-dir", type=Path)
    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    report_parser.add_argument("--output", type=Path, default=ROOT / "reports" / "comparison.md")
    control_parser = subparsers.add_parser("control")
    control_parser.add_argument("--run-dir", type=Path, default=ROOT / "runs" / "synthetic-control")
    args = parser.parse_args()
    if args.command == "report":
        create_report(args.runs_dir, args.output)
        print(args.output)
        return
    if args.command == "control":
        events, manifest, config = synthetic_control()
        train_run(events, manifest, config, args.run_dir, "yambda")
        result = evaluate_run(events, manifest, config, args.run_dir)
        print(json.dumps({key: value["contrast"] for key, value in result["catalogs"].items()}, ensure_ascii=False))
        return
    config = Config.load(args.profile, args.config)
    data_root = ROOT / "data" / f"data_{args.dataset}"
    ingestion = {}
    if args.command in ("download", "prepare", "run"):
        print(f"{args.dataset}: verify/download source", flush=True)
        with ResourceMeasure() as download_measure:
            download_dataset(args.dataset, data_root, config["metadata_mode"] == "categories")
        ingestion["download_or_verify"] = download_measure.result
    if args.command == "download":
        return
    print(f"{args.dataset}: prepare complete user histories", flush=True)
    with ResourceMeasure() as prepare_measure:
        events, manifest = prepare_dataset(args.dataset, data_root, config)
    ingestion["prepare_or_verify_cache"] = prepare_measure.result
    if args.command == "prepare":
        print(json.dumps(manifest, ensure_ascii=False))
        return
    categories = load_categories(args.dataset, data_root) if config["metadata_mode"] == "categories" else None
    run_root = args.run_dir or ROOT / "runs" / f"{args.dataset}-{args.profile}-{config.digest[:12]}"
    if not (run_root / "frozen.json").exists():
        write_json(run_root / "diagnostics" / f"ingestion-{uuid4().hex}.json", ingestion)
    try:
        if args.command in ("train", "run"):
            train_run(events, manifest, config, run_root, args.dataset, categories)
        if args.command in ("evaluate", "run"):
            evaluate_run(events, manifest, config, run_root, categories)
    except Exception as error:
        write_json(run_root / "failure.json", {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
        raise
    print(run_root, flush=True)


if __name__ == "__main__":
    main()
