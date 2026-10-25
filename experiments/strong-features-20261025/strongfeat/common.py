"""Shared data, split, candidate and evaluation code for the strong-features experiment.

Every function that builds model inputs for a cutoff q only reads events with timestamp < q.
Targets are read only from [q, q + horizon).
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import psutil
from sklearn.metrics import log_loss, roc_auc_score

DAY = 86400
HORIZON = 7 * DAY
SPLITS = ("train", "train", "train", "dev", "selection", "test")
EXPERIMENT = Path(__file__).resolve().parents[1]


def load_config(path: Path | None = None) -> dict:
    return json.loads((path or EXPERIMENT / "configs" / "experiment.json").read_text(encoding="utf-8"))


def cutoffs(end_exclusive: int) -> list[tuple[str, int]]:
    return [(split, end_exclusive - (6 - index) * HORIZON) for index, split in enumerate(SPLITS)]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1 << 22):
            digest.update(chunk)
    return digest.hexdigest()


def build_events(prepared_events: Path, raw_events: Path, artist_map: Path, album_map: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Attach played_ratio_pct, is_organic and track metadata to the already prepared cohort.

    The prepared file of the original pipeline keeps every source row of the 1000 sampled users and
    stores the source row index as row_id. We read the same rows from the raw file and assert that
    user, item, timestamp and event type coincide, so the cohort and its full histories are unchanged.
    """
    prepared = pl.read_parquet(prepared_events)
    raw = pl.scan_parquet(raw_events, row_index_name="row_id").filter(pl.col("row_id").is_in(prepared["row_id"].implode())).collect()
    joined = prepared.join(raw, on="row_id", how="inner", suffix="_raw")
    if joined.height != prepared.height:
        raise ValueError("Raw rows do not cover the prepared cohort")
    for left, right in (("user_id", "uid"), ("item_id", "item_id_raw"), ("timestamp", "timestamp_raw")):
        if not (joined[left] == joined[right].cast(pl.Int64)).all():
            raise ValueError(f"Mismatch between prepared and raw column {left}")
    if not (joined["event_type"] == joined["event_type_raw"].cast(pl.String)).all():
        raise ValueError("Mismatch between prepared and raw event_type")
    artists = pl.read_parquet(artist_map).group_by("item_id").agg(pl.col("artist_id").min()).with_columns(pl.col("item_id").cast(pl.Int64), pl.col("artist_id").cast(pl.Int64))
    albums = pl.read_parquet(album_map).group_by("item_id").agg(pl.col("album_id").min()).with_columns(pl.col("item_id").cast(pl.Int64), pl.col("album_id").cast(pl.Int64))
    items = joined.select(pl.col("item_id").unique().sort()).join(artists, on="item_id", how="left").join(albums, on="item_id", how="left")
    # Tracks without metadata get their own pseudo artist/album so that grouping never merges them.
    items = items.with_columns(pl.col("artist_id").fill_null(-pl.col("item_id")), pl.col("album_id").fill_null(-pl.col("item_id")))
    events = joined.select(
        "row_id", "user_id", "item_id", "timestamp", "event_type", "is_target",
        pl.col("is_organic").cast(pl.Int8), pl.col("played_ratio_pct").cast(pl.Float32), pl.col("track_length_seconds").cast(pl.Float32),
    ).join(items, on="item_id", how="left").sort("row_id")
    return events, items


def history(events: pl.DataFrame, q: int) -> pl.DataFrame:
    return events.filter(pl.col("timestamp") < q)


def target_frame(events: pl.DataFrame, q: int) -> pl.DataFrame:
    future = events.filter((pl.col("timestamp") >= q) & (pl.col("timestamp") < q + HORIZON) & pl.col("is_target"))
    return future.select("user_id", "item_id").unique().sort("user_id", "item_id")


def candidates(events: pl.DataFrame, users: list[int], q: int, config: dict) -> pl.DataFrame:
    """One candidate set shared by every ranking model.

    Sources (all computed from history before q):
      history: up to N tracks the user fully listened most often in the last 90 days (ties: more recent first)
      recent:  up to N most recently listened distinct tracks
      artist:  up to N not yet heard tracks of the user's top-10 artists of the last 90 days, by cohort popularity
      popular: top N tracks by qualifying listens of the cohort in the last 7 days
    """
    c = config["candidates"]
    past = history(events, q)
    listens = past.filter(pl.col("event_type") == "listen")
    recent90 = listens.filter(pl.col("timestamp") >= q - 90 * DAY)
    hist = (recent90.group_by("user_id", "item_id").agg((pl.col("played_ratio_pct") > 50).sum().alias("c"), pl.col("timestamp").max().alias("t"))
            .sort(["user_id", "c", "t", "item_id"], descending=[False, True, True, False])
            .with_columns(pl.int_range(pl.len()).over("user_id").alias("r")).filter(pl.col("r") < c["history"]).select("user_id", "item_id"))
    recent = (listens.group_by("user_id", "item_id").agg(pl.col("timestamp").max().alias("t"))
              .sort(["user_id", "t", "item_id"], descending=[False, True, False])
              .with_columns(pl.int_range(pl.len()).over("user_id").alias("r")).filter(pl.col("r") < c["recent"]).select("user_id", "item_id"))
    seen = past.select("user_id", "item_id").unique()
    top_artists = (recent90.filter(pl.col("played_ratio_pct") > 50).group_by("user_id", "artist_id").len()
                   .sort(["user_id", "len", "artist_id"], descending=[False, True, False])
                   .with_columns(pl.int_range(pl.len()).over("user_id").alias("r")).filter(pl.col("r") < 10).select("user_id", "artist_id"))
    item_pop = past.filter(pl.col("is_target")).group_by("item_id", "artist_id").len()
    artist = (top_artists.join(item_pop, on="artist_id").join(seen, on=["user_id", "item_id"], how="anti")
              .sort(["user_id", "len", "item_id"], descending=[False, True, False])
              .with_columns(pl.int_range(pl.len()).over("user_id").alias("r")).filter(pl.col("r") < c["artist"]).select("user_id", "item_id"))
    popular_items = (past.filter(pl.col("is_target") & (pl.col("timestamp") >= q - 7 * DAY)).group_by("item_id").len()
                     .sort(["len", "item_id"], descending=[True, False]).head(c["popular"]).select("item_id"))
    popular = pl.DataFrame({"user_id": users}, schema={"user_id": pl.Int64}).join(popular_items, how="cross")
    sources = [hist.with_columns(pl.lit(1, pl.Int8).alias("src_history")), recent.with_columns(pl.lit(1, pl.Int8).alias("src_recent")),
               artist.with_columns(pl.lit(1, pl.Int8).alias("src_artist")), popular.with_columns(pl.lit(1, pl.Int8).alias("src_popular"))]
    rows = pl.concat([s.select("user_id", "item_id") for s in sources]).unique().filter(pl.col("user_id").is_in(users))
    for source in sources:
        rows = rows.join(source, on=["user_id", "item_id"], how="left")
    rows = rows.with_columns(pl.col("^src_.*$").fill_null(0)).sort("user_id", "item_id")
    return rows.with_columns(pl.lit(q, pl.Int64).alias("cutoff"))


@dataclass
class Slice:
    split: str
    cutoff: int
    rows: pl.DataFrame  # user_id, item_id, src_*, cutoff, label
    targets: dict[int, int]  # number of distinct targets per user, all targets, not only candidates
    users: list[int]


def make_slice(events: pl.DataFrame, users: list[int], split: str, q: int, config: dict) -> Slice:
    rows = candidates(events, users, q, config)
    targets = target_frame(events, q).filter(pl.col("user_id").is_in(users))
    rows = rows.join(targets.with_columns(pl.lit(1, pl.Int8).alias("label")), on=["user_id", "item_id"], how="left").with_columns(pl.col("label").fill_null(0))
    counts = dict(targets.group_by("user_id").len().iter_rows())
    return Slice(split, q, rows, {user: int(counts.get(user, 0)) for user in users}, users)


def evaluate(slice_: Slice, scores: np.ndarray, probabilities: bool = True, k: int = 10) -> dict:
    """Same definitions as the original pipeline: NDCG@10 and Recall@K use all distinct targets of the horizon,
    including ones the candidate generator missed; ties are broken by ascending item_id; users without targets are excluded."""
    if len(scores) != slice_.rows.height or not np.isfinite(scores).all():
        raise ValueError("Scores must be finite and match the candidates")
    frame = slice_.rows.select("user_id", "item_id", "label").with_columns(pl.Series("score", scores.astype(np.float64)))
    frame = frame.sort(["user_id", "score", "item_id"], descending=[False, True, False]).with_columns(pl.int_range(pl.len()).over("user_id").alias("rank"))
    discount = 1 / np.log2(np.arange(2, 102))
    per_user = []
    grouped = {user: (ranks, labels) for user, ranks, labels in
               frame.filter(pl.col("rank") < 100).group_by("user_id", maintain_order=True).agg("rank", "label").iter_rows()}
    candidate_hits = dict(frame.group_by("user_id").agg(pl.col("label").sum()).iter_rows())
    for user in slice_.users:
        n = slice_.targets[user]
        if n == 0:
            continue
        labels = np.array(grouped.get(user, ([], []))[1], dtype=np.float64)
        top10 = labels[:k]
        idcg = discount[: min(n, k)].sum()
        per_user.append({"user_id": user, "ndcg_10": float((top10 * discount[: len(top10)]).sum() / idcg), "recall_10": float(top10.sum() / n),
                         "recall_100": float(labels[:100].sum() / n), "candidate_recall": float(candidate_hits.get(user, 0) / n), "target_count": n})
    metrics = {key: float(np.mean([row[key] for row in per_user])) for key in ("ndcg_10", "recall_10", "recall_100", "candidate_recall")}
    labels = slice_.rows["label"].to_numpy()
    metrics["logloss"] = float(log_loss(labels, np.clip(scores, 1e-15, 1 - 1e-15), labels=[0, 1])) if probabilities else None
    metrics["roc_auc"] = float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None
    metrics["evaluated_users"] = len(per_user)
    metrics["rows"] = int(len(scores))
    return {"metrics": metrics, "per_user": per_user}


def paired_bootstrap(per_user_a: dict[int, float], per_user_b: dict[int, float], samples: int, seed: int) -> dict:
    """Mean of (a - b) over users with a 95% percentile interval from user-level resampling."""
    if per_user_a.keys() != per_user_b.keys():
        raise ValueError("Paired contrast requires identical users")
    users = sorted(per_user_a)
    diffs = np.array([per_user_a[u] - per_user_b[u] for u in users])
    rng = np.random.default_rng(seed)
    replicates = rng.choice(diffs, size=(samples, len(diffs)), replace=True).mean(axis=1)
    low, high = np.quantile(replicates, [0.025, 0.975])
    return {"mean": float(diffs.mean()), "ci95": [float(low), float(high)], "users": len(users), "samples": samples, "seed": seed}


class ResourceMeasure:
    """Wall time and sampled peak RSS of this process and its children (0.1 s sampling)."""

    def __enter__(self):
        self.started = time.perf_counter()
        self.peak = 0
        self._stop = threading.Event()
        self._process = psutil.Process(os.getpid())
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def _sample(self):
        while not self._stop.is_set():
            total = 0
            for process in [self._process, *self._process.children(recursive=True)]:
                try:
                    total += process.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            self.peak = max(self.peak, total)
            self._stop.wait(0.1)

    def __exit__(self, *args):
        self._stop.set()
        self._thread.join()
        self.result = {"wall_seconds": round(time.perf_counter() - self.started, 3), "peak_rss_gib": round(self.peak / 2**30, 3)}


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)

