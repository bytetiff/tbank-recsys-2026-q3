from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import time
import urllib.request
import zipfile

import pyarrow.parquet as pq
import pyarrow as pa
import polars as pl

from .config import write_json

SOURCES = {
    "kion": {
        "revision": "f69775be31fa5779907cf0a92ddedb70037fb5ae",
        "base": "https://raw.githubusercontent.com/irsafilo/KION_DATASET/{revision}/",
        "files": {"data_original.zip": 78795295},
    },
    "yambda": {
        "revision": "dd6f3a19eef5866e346c3270e098baa641a44948",
        "base": "https://huggingface.co/datasets/yandex/yambda/resolve/{revision}/",
        "files": {"flat/50m/multi_event.parquet": 384465869},
        "optional_files": {"artist_item_mapping.parquet": 38274621},
    },
}


def source_for_mode(manifest: dict, categories: bool = False) -> dict:
    source = SOURCES[manifest["dataset"]]
    required = dict(source["files"])
    if categories:
        required.update(source.get("optional_files", {}))
    missing = set(required) - manifest["files"].keys()
    if missing:
        raise ValueError(f"Missing source in manifest: {sorted(missing)}")
    return {"dataset": manifest["dataset"], "revision": manifest["revision"], "files": {name: manifest["files"][name] for name in sorted(required)}}


def validate_id_types(frame: pl.DataFrame, columns: tuple[str, ...]) -> None:
    if any(not frame.schema[name].is_integer() for name in columns):
        raise ValueError("Source IDs must have integer types before normalization")


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_source_file(path: Path, source_path: str) -> None:
    if source_path.endswith(".zip"):
        with zipfile.ZipFile(path) as archive:
            member = "data_original/interactions.csv" if source_path == "data_original.zip" else "interactions.csv"
            required_members = {member, "data_original/users.csv", "data_original/items.csv"} if source_path == "data_original.zip" else {member}
            if archive.testzip() is not None or not required_members.issubset(archive.namelist()):
                raise ValueError("Invalid interactions ZIP")
            with archive.open(member) as handle:
                header = handle.readline().decode("utf-8-sig").strip().split(",")
            required = {"user_id", "item_id", "last_watch_dt", "total_dur", "watched_pct"}
            if required.issubset(header):
                with archive.open(member) as handle:
                    frame = pl.read_csv(handle, columns=["user_id", "item_id", "last_watch_dt"])
                validate_id_types(frame, ("user_id", "item_id"))
                dates = frame["last_watch_dt"].str.to_date("%Y-%m-%d", strict=True)
                if dates.null_count() or frame["user_id"].null_count() or frame["item_id"].null_count():
                    raise ValueError("Incomplete source interaction row")
    elif source_path.endswith(".parquet"):
        schema = pq.ParquetFile(path).schema_arrow
        header = schema.names
        required = {"uid", "item_id", "timestamp", "event_type", "played_ratio_pct"} if "multi_event" in source_path else {"artist_id", "item_id"}
        if required.issubset(header) and "multi_event" in source_path:
            if any(not pa.types.is_integer(schema.field(name).type) for name in ("uid", "item_id", "timestamp")):
                raise ValueError("Yambda IDs and seconds must have integer source types")
            invalid = pl.scan_parquet(path).select(pl.any_horizontal(
                pl.col("uid").is_null(), pl.col("item_id").is_null(), pl.col("timestamp").is_null(), pl.col("event_type").is_null(),
                pl.col("timestamp") < 0, pl.col("timestamp") % 5 != 0,
                ~pl.col("event_type").cast(pl.String).is_in(["listen", "like", "dislike", "unlike", "undislike"]),
            ).sum()).collect().item()
            if invalid:
                raise ValueError(f"Yambda has {invalid} invalid source event rows")
    else:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            header = next(csv.reader(handle))
        required = {"item_id", "genres"} if "items" in source_path else {"user_id"}
    if not required.issubset(header):
        raise ValueError(f"Invalid source schema for {source_path}: {header}")


def verify_manifest(raw: Path, manifest: dict, categories: bool = False) -> None:
    source = SOURCES[manifest["dataset"]]
    if manifest["revision"] != source["revision"]:
        raise ValueError("Source revision mismatch")
    relevant = source_for_mode(manifest, categories)
    expected = dict(source["files"])
    if categories:
        expected.update(source.get("optional_files", {}))
    for source_path, expected_bytes in expected.items():
        if relevant["files"][source_path]["bytes"] != expected_bytes:
            raise ValueError(f"Source size disagrees with pinned revision: {source_path}")
    for source_path, record in relevant["files"].items():
        path = raw / source_path
        if not path.is_file() or path.stat().st_size != record["bytes"] or file_hash(path) != record["sha256"]:
            raise ValueError(f"Source integrity check failed: {path}")
        validate_source_file(path, source_path)


def download_dataset(dataset: str, root: Path, categories: bool = False) -> dict:
    source = SOURCES[dataset]
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "source-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"dataset": dataset, "revision": source["revision"], "files": {}}
    if manifest["dataset"] != dataset or manifest["revision"] != source["revision"]:
        raise ValueError("Existing source manifest has a different dataset/revision")
    if dataset == "kion" and "interactions.csv.zip" in manifest["files"]:
        rejected = manifest.setdefault("rejected_files", {})
        rejected["interactions.csv.zip"] = manifest["files"].pop("interactions.csv.zip") | {"reason": "Published member is truncated at 52428800 bytes; final date is '20'"}
        write_json(manifest_path, manifest)
    files = dict(source["files"])
    if categories:
        files.update(source.get("optional_files", {}))
    for source_path, size in files.items():
        destination = raw / source_path
        if destination.exists():
            record = manifest["files"].get(source_path)
            if not record or destination.stat().st_size != size or file_hash(destination) != record["sha256"]:
                raise ValueError(f"Existing file is unverified or changed: {destination}")
            validate_source_file(destination, source_path)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".part")
        url = source["base"].format(revision=source["revision"]) + source_path
        last_error = None
        for attempt in range(3):
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "tlab-recsys/0.1"})
                with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
                    while chunk := response.read(4 * 1024 * 1024):
                        output.write(chunk)
                if temporary.stat().st_size != size:
                    raise ValueError(f"Unexpected source size for {source_path}")
                validate_source_file(temporary, source_path)
                last_error = None
                break
            except Exception as error:
                last_error = error
                if attempt < 2:
                    time.sleep(1 + attempt)
        if last_error is not None:
            raise RuntimeError(f"Download failed; incomplete file retained as {temporary}") from last_error
        record = {"url": url, "bytes": size, "sha256": file_hash(temporary)}
        temporary.replace(destination)
        manifest["files"][source_path] = record
        write_json(manifest_path, manifest)
    verify_manifest(raw, manifest, categories)
    return manifest
