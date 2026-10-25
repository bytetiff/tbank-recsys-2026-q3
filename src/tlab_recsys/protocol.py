from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
import polars as pl

from .config import Config, DAY

SPLITS = ("train", "train", "train", "dev", "selection", "test")


def cutoffs(end: int, horizon: int) -> list[tuple[str, int]]:
    if not isinstance(horizon, int) or horizon <= 0:
        raise ValueError("Horizon must be a positive integer")
    return [(split, end - (6 - index) * horizon) for index, split in enumerate(SPLITS)]


def history(events: pl.DataFrame, q: int, window: int | None = None) -> pl.DataFrame:
    if window is not None and (not isinstance(window, int) or window <= 0):
        raise ValueError("Window must be a positive integer")
    predicate = pl.col("timestamp") < q
    if window is not None:
        predicate &= pl.col("timestamp") >= q - window
    return events.filter(predicate)


def target_sets(events: pl.DataFrame, users: list[int], q: int, horizon: int) -> dict[int, set[int]]:
    if horizon <= 0:
        raise ValueError("Horizon must be positive")
    future = events.filter((pl.col("timestamp") >= q) & (pl.col("timestamp") < q + horizon) & pl.col("is_target"))
    result = {user: set() for user in users}
    for user, item in future.select("user_id", "item_id").iter_rows():
        if user in result:
            result[user].add(item)
    return result


def candidate_rows(events: pl.DataFrame, users: list[int], q: int, dataset: str, config: Config, full: bool = False) -> tuple[pl.DataFrame, set[int]]:
    past = history(events, q)
    catalog = set(past["item_id"].to_list())
    counts = dict(past.filter(pl.col("is_target")).group_by("item_id").len().iter_rows())
    popular = sorted(catalog, key=lambda item: (-counts.get(item, 0), item))
    user_history = {user: frame for (user,), frame in past.partition_by("user_id", as_dict=True).items()}
    rows = []
    for user in users:
        previous = user_history.get(user)
        seen = set(previous["item_id"].to_list()) if previous is not None else set()
        eligible = [item for item in popular if dataset != "kion" or item not in seen]
        if full:
            chosen = set(eligible)
        else:
            chosen = set(eligible[:config["popular_candidates"]])
            if dataset == "yambda" and previous is not None and config["recent_candidates"]:
                recent = previous.sort(["timestamp", "item_id"], descending=[True, False]).unique("item_id", keep="first", maintain_order=True)
                chosen.update(recent["item_id"].head(config["recent_candidates"]).to_list())
        for item in sorted(chosen):
            rows.append((f"{q}:{user}", user, item, q, float(counts.get(item, 0))))
    return pl.DataFrame(rows, schema={"query_id": pl.String, "user_id": pl.Int64, "item_id": pl.Int64, "cutoff": pl.Int64, "popularity": pl.Float64}, orient="row"), catalog


@dataclass
class Slice:
    split: str
    cutoff: int
    rows: pl.DataFrame
    targets: dict[int, set[int]]
    catalog: set[int]
    users: list[int]
    catalog_targets: dict[int, set[int]] | None = None


def make_slice(events: pl.DataFrame, users: list[int], split: str, q: int, dataset: str, config: Config, full: bool = False) -> Slice:
    rows, catalog = candidate_rows(events, users, q, dataset, config, full)
    targets = target_sets(events, users, q, config["horizon_days"] * DAY)
    labels = [int(item in targets[user]) for user, item in rows.select("user_id", "item_id").iter_rows()]
    seen = {}
    if dataset == "kion":
        for user, item in history(events, q).select("user_id", "item_id").iter_rows():
            seen.setdefault(user, set()).add(item)
    catalog_targets = {user: (targets[user] & catalog) - seen.get(user, set()) for user in users}
    return Slice(split, q, rows.with_columns(pl.Series("label", labels, dtype=pl.Int8)), targets, catalog, users, catalog_targets)


def table_hash(frame: pl.DataFrame) -> str:
    # Include schema and ordered values; no Python hash randomization is involved.
    return hashlib.sha256(str(frame.schema).encode() + frame.hash_rows(seed=1729).to_numpy().tobytes()).hexdigest()


def require_train(labels: np.ndarray) -> None:
    if len(labels) == 0 or len(np.unique(labels)) != 2:
        raise ValueError("Training slice must be nonempty and contain both classes")
