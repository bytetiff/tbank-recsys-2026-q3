import zipfile

import polars as pl
import pytest

from tlab_recsys.download import validate_source_file
from tlab_recsys.download import SOURCES, file_hash
from tlab_recsys.data import adapt_kion, load_categories, prepare_dataset
from tlab_recsys.config import Config, write_json


def test_valid_zip_crc_does_not_make_truncated_source_valid(tmp_path):
    path = tmp_path / "interactions.csv.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("interactions.csv", "user_id,item_id,last_watch_dt,total_dur,watched_pct\n1,2,2021-08-22,100,50\n3,4,20\n")
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
    with pytest.raises(pl.exceptions.InvalidOperationError):
        validate_source_file(path, "interactions.csv.zip")


def test_valid_source_retains_same_day_events(tmp_path):
    path = tmp_path / "interactions.csv.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("interactions.csv", "user_id,item_id,last_watch_dt,total_dur,watched_pct\n1,2,2021-08-22,100,50\n1,3,2021-08-22,200,80\n")
    validate_source_file(path, "interactions.csv.zip")


@pytest.mark.parametrize("column", ["user_id", "item_id"])
def test_kion_fractional_ids_are_rejected_in_zip_and_adapter(tmp_path, column):
    frame = pl.DataFrame({"user_id": [1, 2], "item_id": [10, 20], "last_watch_dt": ["2021-08-22"] * 2,
                          "total_dur": [100, 100], "watched_pct": [50, 50]}).with_columns((pl.col(column) + 0.5).alias(column))
    path = tmp_path / "fractional.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("interactions.csv", frame.write_csv())
    with pytest.raises(ValueError, match="integer types"):
        validate_source_file(path, "interactions.csv.zip")
    with pytest.raises(ValueError, match="integer types"):
        adapt_kion(frame)


def test_kion_metadata_uses_verified_archive_even_after_preparation_cache_hit(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    path = raw / "data_original.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("data_original/interactions.csv", "user_id,item_id,last_watch_dt,total_dur,watched_pct\n1,10,2021-03-13,100,50\n1,20,2021-08-22,200,80\n")
        archive.writestr("data_original/users.csv", "user_id,age\n1,25\n")
        archive.writestr("data_original/items.csv", "item_id,genres\n10,original\n20,other\n")
    source = dict(SOURCES["kion"], files={"data_original.zip": path.stat().st_size})
    monkeypatch.setitem(SOURCES, "kion", source)
    manifest = {"dataset": "kion", "revision": source["revision"], "files": {"data_original.zip": {"bytes": path.stat().st_size, "sha256": file_hash(path)}}}
    write_json(tmp_path / "source-manifest.json", manifest)
    config = Config(dict(Config.load().values, cohort_size=1, metadata_mode="categories"))
    prepare_dataset("kion", tmp_path, config)
    (raw / "data_original/items.csv").write_text("item_id,genres\n10,poisoned\n", encoding="utf-8")
    prepare_dataset("kion", tmp_path, config)
    assert load_categories("kion", tmp_path)[10] == ("original",)
