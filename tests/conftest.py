import pytest
import polars as pl

from tlab_recsys.config import Config, DAY


@pytest.fixture
def config():
    return Config.load()


@pytest.fixture
def events():
    rows = [
        (0, 1, 10, 0, "listen", True),
        (1, 1, 10, DAY, "listen", True),
        (2, 1, 20, 2 * DAY, "like", False),
        (3, 2, 20, 2 * DAY, "listen", True),
        (4, 2, 30, 3 * DAY, "listen", True),
        (5, 1, 30, 3 * DAY, "listen", True),
        (6, 1, 99, 4 * DAY, "listen", True),
        (7, 1, 10, 0, "listen", True),
    ]
    return pl.DataFrame(rows, schema={"row_id": pl.UInt32, "user_id": pl.Int64, "item_id": pl.Int64, "timestamp": pl.Int64, "event_type": pl.String, "is_target": pl.Boolean}, orient="row")
