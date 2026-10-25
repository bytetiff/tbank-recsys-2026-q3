from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile
from importlib.metadata import version

import numpy as np
import polars as pl

from .config import Config, DAY, canonical_hash, write_json
from .download import file_hash, source_for_mode, validate_id_types, verify_manifest

EVENT_COLUMNS = ["row_id", "user_id", "item_id", "timestamp", "event_type", "is_target"]


def preparation_fingerprint() -> str:
    package = Path(__file__).resolve().parent
    return canonical_hash({"code": {name: file_hash(package / name) for name in ("data.py", "download.py", "config.py")},
                           "dependencies": {name: version(name) for name in ("polars", "numpy", "pyarrow")}})


def adapt_kion(frame: pl.DataFrame) -> pl.DataFrame:
    required = {"user_id", "item_id", "last_watch_dt"}
    if not required.issubset(frame.columns):
        raise ValueError("Missing KION fields")
    validate_id_types(frame, ("user_id", "item_id"))
    if "row_id" not in frame.columns:
        frame = frame.with_row_index("row_id")
    return frame.with_columns(
        pl.col("user_id").cast(pl.Int64), pl.col("item_id").cast(pl.Int64),
        pl.col("last_watch_dt").cast(pl.String).str.to_date("%Y-%m-%d", strict=True).cast(pl.Datetime("us")).dt.epoch("s").alias("timestamp"),
        pl.lit("watch").alias("event_type"), pl.lit(True).alias("is_target"),
    ).select(EVENT_COLUMNS)


def adapt_yambda(frame: pl.DataFrame, threshold: int = 50) -> pl.DataFrame:
    if not {"uid", "item_id", "timestamp", "event_type", "played_ratio_pct"}.issubset(frame.columns):
        raise ValueError("Missing Yambda fields")
    if "row_id" not in frame.columns:
        frame = frame.with_row_index("row_id")
    return frame.with_columns(
        pl.col("uid").cast(pl.Int64).alias("user_id"), pl.col("item_id").cast(pl.Int64),
        pl.col("timestamp").cast(pl.Int64), pl.col("event_type").cast(pl.String),
        ((pl.col("event_type").cast(pl.String) == "listen") & (pl.col("played_ratio_pct") > threshold)).fill_null(False).alias("is_target"),
    ).select(EVENT_COLUMNS)


def validate_events(events: pl.DataFrame) -> None:
    if events.select(pl.any_horizontal(pl.all().is_null()).any()).item():
        raise ValueError("Null in normalized event schema")
    if events["row_id"].n_unique() != events.height:
        raise ValueError("row_id must be unique; duplicate contents must retain distinct row IDs")


def cohort_order(users: list[int], seed: int) -> list[int]:
    return sorted(set(users), key=lambda user: (hashlib.sha256(f"{seed}:{user}".encode()).digest(), user))


def complete_cohort(events: pl.DataFrame, users: list[int]) -> pl.DataFrame:
    return events.filter(pl.col("user_id").is_in(users)).sort("row_id")


def prepare_dataset(dataset: str, root: Path, config: Config) -> tuple[pl.DataFrame, dict]:
    config.validate()
    source = json.loads((root / "source-manifest.json").read_text(encoding="utf-8"))
    if source["dataset"] != dataset:
        raise ValueError("Dataset mismatch")
    categories = config["metadata_mode"] == "categories"
    verify_manifest(root / "raw", source, categories)
    source = source_for_mode(source, categories)
    preparation = preparation_fingerprint()
    prepared = root / "prepared" / f"v3-{config.digest}-{canonical_hash(source)[:12]}-{preparation[:12]}"
    manifest_path = prepared / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        path = prepared / "events.parquet"
        manifest_digest = canonical_hash({key: value for key, value in manifest.items() if key != "manifest_hash"})
        if manifest.get("manifest_hash") != manifest_digest or manifest.get("preparation_hash") != preparation or manifest["source_hash"] != canonical_hash(source) or manifest["config_hash"] != config.digest or file_hash(path) != manifest["events_sha256"]:
            raise ValueError("Prepared cache integrity mismatch")
        events = pl.read_parquet(path)
        validate_events(events)
        return events, manifest
    if dataset == "kion":
        csv_path = root / "raw" / "interactions.csv"
        with zipfile.ZipFile(root / "raw" / "data_original.zip") as archive:
            # Re-extract the verified archive so an unmanifested CSV cannot poison a run.
            with archive.open("data_original/interactions.csv") as src, csv_path.open("wb") as dst:
                while chunk := src.read(4 * 1024 * 1024):
                    dst.write(chunk)
            for member in ("data_original/users.csv", "data_original/items.csv"):
                destination = root / "raw" / member
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as src, destination.open("wb") as dst:
                    while chunk := src.read(4 * 1024 * 1024):
                        dst.write(chunk)
        lazy = pl.scan_csv(csv_path, row_index_name="row_id").select("row_id", "user_id", "item_id", "last_watch_dt")
        summary = lazy.select(pl.len().alias("rows"), pl.col("last_watch_dt").min().alias("min"), pl.col("last_watch_dt").max().alias("max")).collect()
        end = int(np.datetime64(summary["max"][0], "D").astype("datetime64[s]").astype(np.int64)) + DAY
        first_cutoff = end - 6 * config["horizon_days"] * DAY
        first_day = str(np.datetime64(first_cutoff, "s").astype("datetime64[D]"))
        eligible = lazy.filter(pl.col("last_watch_dt") < first_day).select("user_id").unique().collect()["user_id"].to_list()
    else:
        lazy = pl.scan_parquet(root / "raw" / "flat/50m/multi_event.parquet", row_index_name="row_id")
        summary = lazy.select(pl.len().alias("rows"), pl.col("timestamp").min().alias("min"), pl.col("timestamp").max().alias("max")).collect()
        end = int(summary["max"][0]) + 5
        first_cutoff = end - 6 * config["horizon_days"] * DAY
        eligible = lazy.filter(pl.col("timestamp") < first_cutoff).select("uid").unique().collect()["uid"].to_list()
    ordered = cohort_order(eligible, config["cohort_seed"])
    if len(ordered) < config["cohort_size"]:
        raise ValueError(f"Only {len(ordered)} eligible users; requested {config['cohort_size']}")
    users = ordered[:config["cohort_size"]]
    user_column = "user_id" if dataset == "kion" else "uid"
    selected = lazy.filter(pl.col(user_column).is_in(users)).collect()
    events = adapt_kion(selected) if dataset == "kion" else adapt_yambda(selected, config["target_played_ratio_gt"])
    validate_events(events)
    events = events.sort("row_id")
    counts = events.group_by("user_id").len().sort("user_id")
    source_counts = lazy.filter(pl.col(user_column).is_in(users)).group_by(user_column).len().collect().sort(user_column)
    if counts["len"].to_list() != source_counts["len"].to_list() or counts["user_id"].to_list() != source_counts[user_column].to_list():
        raise ValueError("Complete-history audit failed")
    row_ids = events["row_id"].to_numpy()
    source_ids = selected["row_id"].sort().to_numpy()
    if not np.array_equal(row_ids, source_ids):
        raise ValueError("Source row multiplicity audit failed")
    prepared.mkdir(parents=True, exist_ok=True)
    events.write_parquet(prepared / "events.parquet")
    manifest = {
        "dataset": dataset, "source_hash": canonical_hash(source), "config_hash": config.digest, "preparation_hash": preparation,
        "source_rows": int(summary["rows"][0]), "source_min": str(summary["min"][0]), "source_max": str(summary["max"][0]),
        "end_exclusive": end, "first_train_cutoff": first_cutoff, "cohort_seed": config["cohort_seed"], "users": users,
        "full_catalog_users": users[:config["full_catalog_users"]], "rows": events.height,
        "per_user_rows": dict(zip(map(str, counts["user_id"].to_list()), counts["len"].to_list())),
        "row_ids_sha256": hashlib.sha256(row_ids.tobytes()).hexdigest(), "events_sha256": file_hash(prepared / "events.parquet"),
        "completeness": "All published source rows of selected users, with original multiplicities",
    }
    manifest["manifest_hash"] = canonical_hash(manifest)
    write_json(manifest_path, manifest)
    return events, manifest


def load_categories(dataset: str, root: Path) -> dict[int, tuple[str, ...]]:
    source = json.loads((root / "source-manifest.json").read_text(encoding="utf-8"))
    verify_manifest(root / "raw", source, categories=True)
    if dataset == "kion":
        with zipfile.ZipFile(root / "raw" / "data_original.zip") as archive:
            with archive.open("data_original/items.csv") as handle:
                frame = pl.read_csv(handle, columns=["item_id", "genres"])
        return {int(item): tuple(sorted(set(part.strip() for part in (genres or "unknown").split(",") if part.strip()))) or ("unknown",) for item, genres in frame.iter_rows()}
    frame = pl.read_parquet(root / "raw" / "artist_item_mapping.parquet")
    grouped = frame.group_by("item_id").agg(pl.col("artist_id").unique().sort())
    return {int(item): tuple(map(str, artists)) or ("unknown",) for item, artists in grouped.iter_rows()}
