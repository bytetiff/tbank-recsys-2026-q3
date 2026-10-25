"""Hand-designed baseline: a human choice of grammar specs plus a few extras the grammar cannot express."""
from __future__ import annotations

import polars as pl

from .common import DAY
from .features import Spec

S = Spec
MANUAL_SPECS = [
    # user activity and taste profile
    S("user", "all", None, "count"), S("user", "listen", 7, "count"), S("user", "listen", 30, "count"), S("user", "listen", None, "count"),
    S("user", "full", 7, "count"), S("user", "full", 30, "count"), S("user", "skip", 30, "count"), S("user", "like", None, "count"),
    S("user", "listen", 30, "mean_ratio"), S("user", "full", 30, "nuniq"), S("user", "listen", None, "recency"),
    # item popularity and quality inside the cohort
    S("item", "full", 7, "count"), S("item", "full", 30, "count"), S("item", "full", None, "count"), S("item", "listen", 7, "count"),
    S("item", "listen", 30, "ratio_full"), S("item", "listen", 30, "ratio_skip"), S("item", "like", None, "count"),
    S("item", "dislike", None, "count"), S("item", "listen", None, "recency"),
    S("artist", "full", 7, "count"), S("artist", "full", 30, "count"), S("artist", "listen", 30, "ratio_skip"), S("album", "full", 30, "count"),
    # direct user x track relation
    S("user_item", "full", 7, "count"), S("user_item", "full", 30, "count"), S("user_item", "full", None, "count"),
    S("user_item", "listen", None, "count"), S("user_item", "listen", None, "recency"), S("user_item", "full", None, "recency"),
    S("user_item", "listen", None, "mean_ratio"), S("user_item", "like", None, "count"), S("user_item", "dislike", None, "count"),
    S("user_item", "skip", None, "count"), S("user_item", "listen", None, "ratio_full"), S("user_item", "full", 30, "share"),
    # indirect relations through artist and album
    S("user_artist", "full", 7, "count"), S("user_artist", "full", 30, "count"), S("user_artist", "full", None, "count"),
    S("user_artist", "listen", None, "recency"), S("user_artist", "full", 30, "share"), S("user_artist", "like", None, "count"),
    S("user_artist", "skip", 30, "count"), S("user_artist", "dislike", None, "count"),
    S("user_album", "full", 30, "count"), S("user_album", "full", None, "count"), S("user_album", "listen", None, "recency"),
    S("user_album", "full", 30, "share"),
]

EXTRA_NAMES = ["x_item_trend", "x_user_artist_trend", "x_pair_liked", "x_pair_disliked", "x_item_age", "x_pair_recency_rank", "x_track_length"]


def extras(past: pl.DataFrame, rows: pl.DataFrame, q: int) -> pl.DataFrame:
    """Hand-made features outside the grammar: trends, like state, rank in the user's history, item age."""
    if past.height and past["timestamp"].max() >= q:
        raise ValueError("Feature history contains events at or after the cutoff")
    base = rows.select("user_id", "item_id", "artist_id").with_row_index("_pos")
    full = past.filter((pl.col("event_type") == "listen") & (pl.col("played_ratio_pct") > 50))
    item = full.group_by("item_id").agg((pl.col("timestamp") >= q - 7 * DAY).sum().alias("i7"), (pl.col("timestamp") >= q - 30 * DAY).sum().alias("i30"))
    ua = full.group_by("user_id", "artist_id").agg((pl.col("timestamp") >= q - 7 * DAY).sum().alias("a7"), (pl.col("timestamp") >= q - 90 * DAY).sum().alias("a90"))
    # like state: the last of like / unlike (and dislike / undislike) decides
    state = (past.filter(pl.col("event_type").is_in(["like", "unlike", "dislike", "undislike"]))
             .sort("timestamp", "row_id").group_by("user_id", "item_id")
             .agg(pl.col("event_type").filter(pl.col("event_type").is_in(["like", "unlike"])).last().alias("ls"),
                  pl.col("event_type").filter(pl.col("event_type").is_in(["dislike", "undislike"])).last().alias("ds")))
    age = past.group_by("item_id").agg((q - pl.col("timestamp").min()).alias("age"), pl.col("track_length_seconds").median().alias("len"))
    last = (past.filter(pl.col("event_type") == "listen").group_by("user_id", "item_id").agg(pl.col("timestamp").max().alias("t"))
            .with_columns(pl.col("t").rank("min", descending=True).over("user_id").alias("rank")))
    j = (base.join(item, on="item_id", how="left").join(ua, on=["user_id", "artist_id"], how="left").join(state, on=["user_id", "item_id"], how="left")
         .join(age, on="item_id", how="left").join(last.select("user_id", "item_id", "rank"), on=["user_id", "item_id"], how="left").sort("_pos"))
    return j.select(
        (pl.col("i7").fill_null(0) / (pl.col("i30").fill_null(0) / 4 + 1)).cast(pl.Float32).alias("x_item_trend"),
        (pl.col("a7").fill_null(0) / (pl.col("a90").fill_null(0) / 13 + 1)).cast(pl.Float32).alias("x_user_artist_trend"),
        (pl.col("ls") == "like").fill_null(False).cast(pl.Float32).alias("x_pair_liked"),
        (pl.col("ds") == "dislike").fill_null(False).cast(pl.Float32).alias("x_pair_disliked"),
        pl.col("age").cast(pl.Float32).alias("x_item_age"),
        pl.col("rank").cast(pl.Float32).alias("x_pair_recency_rank"),
        pl.col("len").cast(pl.Float32).alias("x_track_length"),
    )
