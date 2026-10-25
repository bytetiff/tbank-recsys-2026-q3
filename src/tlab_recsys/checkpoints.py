from __future__ import annotations

import json
from pathlib import Path

from .config import canonical_hash, write_json
from .download import file_hash


def artifact_path(root: Path, relative: str) -> Path:
    path = root / relative
    if Path(relative).is_absolute() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Checkpoint artifact escapes run directory: {relative}")
    return path


def artifact_hashes(root: Path, relatives: list[str]) -> dict[str, str]:
    result = {}
    for relative in sorted(set(relatives)):
        path = artifact_path(root, relative)
        if not path.is_file():
            raise ValueError(f"Checkpoint artifact missing: {relative}")
        result[relative] = file_hash(path)
    return result


def verify_artifacts(root: Path, artifacts: dict[str, str]) -> None:
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("Checkpoint must bind its artifacts")
    for relative, digest in artifacts.items():
        path = artifact_path(root, relative)
        if not path.is_file() or file_hash(path) != digest:
            raise ValueError(f"Checkpoint artifact changed or missing: {relative}")


def write_checkpoint(path: Path, kind: str, payload: dict) -> dict:
    record = {**payload, "schema_version": 1, "kind": kind}
    record["checkpoint_hash"] = canonical_hash(record)
    write_json(path, record)
    return record


def read_checkpoint(path: Path, kind: str) -> dict:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        payload = {key: value for key, value in record.items() if key != "checkpoint_hash"}
        if record.get("schema_version") != 1 or record.get("kind") != kind or record.get("checkpoint_hash") != canonical_hash(payload):
            raise ValueError("Invalid schema, kind or digest")
    except (OSError, ValueError, AttributeError, TypeError) as error:
        raise ValueError(f"Checkpoint integrity mismatch: {path}") from error
    return record
