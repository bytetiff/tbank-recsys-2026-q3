from __future__ import annotations

from itertools import product
import time

from catboost import CatBoostClassifier, Pool
import numpy as np
import pandas as pd

from .config import Config
from .evaluation import evaluate
from .protocol import Slice, require_train

CORE = ["user_id", "item_id"]


def model_frame(features) -> pd.DataFrame:
    frame = features.to_pandas()
    for name in CORE:
        frame[name] = frame[name].astype(str)
    return frame


def pool(frame: pd.DataFrame, columns: list[str], labels: np.ndarray | None = None) -> Pool:
    return Pool(frame[columns], label=labels, cat_features=CORE)


def fit_fixed(train: pd.DataFrame, labels: np.ndarray, columns: list[str], parameters: dict, config: Config, seed: int) -> CatBoostClassifier:
    require_train(labels)
    model = CatBoostClassifier(**parameters, loss_function="Logloss", task_type="CPU", thread_count=config["catboost"]["thread_count"], random_seed=seed, verbose=False, allow_writing_files=False)
    model.fit(pool(train, columns, labels))
    return model


def tune_manual(train: pd.DataFrame, labels: np.ndarray, dev: pd.DataFrame, dev_slice: Slice, columns: list[str], config: Config, seed: int) -> dict:
    require_train(labels)
    if dev_slice.rows.is_empty() or not any(dev_slice.targets.values()):
        raise ValueError("Dev must have candidates and at least one target user")
    settings = config["catboost"]
    trace = []
    for index, (depth, rate, l2) in enumerate(product(settings["depth"], settings["learning_rate"], settings["l2_leaf_reg"])):
        started = time.perf_counter()
        model = CatBoostClassifier(depth=depth, learning_rate=rate, l2_leaf_reg=l2, iterations=settings["max_iterations"], loss_function="Logloss", eval_metric="Logloss", task_type="CPU", thread_count=settings["thread_count"], random_seed=seed, verbose=False, allow_writing_files=False)
        model.fit(pool(train, columns, labels), eval_set=pool(dev, columns, dev_slice.rows["label"].to_numpy()), early_stopping_rounds=settings["early_stopping_rounds"], use_best_model=True)
        scores = model.predict_proba(pool(dev, columns), thread_count=settings["thread_count"])[:, 1]
        metrics = evaluate(dev_slice, scores)["metrics"]
        if metrics["ndcg_10"] is None or not np.isfinite(metrics["ndcg_10"]):
            raise ValueError("Dev ranking metric is undefined")
        trace.append({"index": index, "depth": depth, "learning_rate": rate, "l2_leaf_reg": l2, "iterations": int(model.get_best_iteration()) + 1, "dev_ndcg_10": metrics["ndcg_10"], "dev_logloss": metrics["logloss"], "wall_seconds": time.perf_counter() - started})
        print(f"dev fit {index + 1}: NDCG@10={metrics['ndcg_10']:.6f}, iterations={trace[-1]['iterations']}", flush=True)
    chosen = min(trace, key=lambda row: (-row["dev_ndcg_10"], row["dev_logloss"], row["index"]))
    return {"parameters": {key: chosen[key] for key in ("depth", "learning_rate", "l2_leaf_reg", "iterations")}, "fits": len(trace), "trace": trace}
