from __future__ import annotations

import math
import os
from pathlib import Path
import threading
import time

import numpy as np
import psutil
from sklearn.metrics import log_loss, roc_auc_score

from .protocol import Slice
from .config import validate_bootstrap


def evaluate(slice_: Slice, scores: np.ndarray, probabilities: bool = True) -> dict:
    if len(scores) != slice_.rows.height or not np.isfinite(scores).all():
        raise ValueError("Predictions must be finite and match candidates")
    user_ids = slice_.rows["user_id"].to_numpy()
    item_ids = slice_.rows["item_id"].to_numpy()
    labels = slice_.rows["label"].to_numpy()
    # The shared index supports shuffled rows without Python pairs for every user.
    user_order = np.argsort(user_ids, kind="stable")
    per_user = []
    for user in slice_.users:
        targets = slice_.targets[user]
        if not targets:
            continue
        start = np.searchsorted(user_ids, user, side="left", sorter=user_order)
        stop = np.searchsorted(user_ids, user, side="right", sorter=user_order)
        indices = user_order[start:stop]
        rank_order = np.lexsort((item_ids[indices], -scores[indices].astype(float)))
        top_labels = [int(value) for value in labels[indices[rank_order[:20]]]]
        n_targets = len(targets)
        idcg = sum(1 / math.log2(rank + 2) for rank in range(min(n_targets, 10)))
        dcg = sum(label / math.log2(rank + 2) for rank, label in enumerate(top_labels[:10]))
        available = set(item_ids[indices])
        catalog_n = len(slice_.catalog_targets[user]) if slice_.catalog_targets is not None else len(targets & slice_.catalog)
        record = {"user_id": user, "ndcg_10": dcg / idcg, "recall_10": sum(top_labels[:10]) / n_targets, "recall_20": sum(top_labels[:20]) / n_targets,
                  "candidate_coverage": len(targets & available) / n_targets, "catalog_coverage": catalog_n / n_targets, "target_count": n_targets}
        candidate_n = len(targets & available)
        record["recall_10_conditional_catalog"] = sum(top_labels[:10]) / catalog_n if catalog_n else None
        conditional_idcg = sum(1 / math.log2(rank + 2) for rank in range(min(catalog_n, 10)))
        record["ndcg_10_conditional_catalog"] = dcg / conditional_idcg if conditional_idcg else None
        record["candidate_target_count"] = candidate_n
        per_user.append(record)
        del available, top_labels
    metrics = {}
    for key in ("ndcg_10", "recall_10", "recall_20", "candidate_coverage", "catalog_coverage", "recall_10_conditional_catalog", "ndcg_10_conditional_catalog"):
        values = [row[key] for row in per_user if row[key] is not None]
        metrics[key] = float(np.mean(values)) if values else None
    clipped = np.clip(scores, 1e-15, 1 - 1e-15)
    metrics["logloss"] = float(log_loss(labels, clipped, labels=[0, 1])) if len(labels) and probabilities else None
    metrics["logloss_reason"] = None if metrics["logloss"] is not None else "uncalibrated popularity scores or empty candidates"
    metrics["roc_auc"] = float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None
    metrics["roc_auc_reason"] = None if metrics["roc_auc"] is not None else "empty or one-class labels"
    metrics["evaluated_users"] = len(per_user)
    metrics["excluded_no_targets"] = len(slice_.users) - len(per_user)
    metrics["rows"] = len(scores)
    return {"metrics": metrics, "per_user": per_user}


def bootstrap_contrast(seed_results: list[dict], samples: int = 2000, seed: int = 2718) -> dict:
    validate_bootstrap(samples, seed)
    differences = {}
    for result in seed_results:
        manual = {row["user_id"]: row["ndcg_10"] for row in result["manual"]["per_user"]}
        selected = {row["user_id"]: row["ndcg_10"] for row in result["selected"]["per_user"]}
        if manual.keys() != selected.keys():
            raise ValueError("Paired contrast requires identical eligible users")
        for user in sorted(manual):
            differences.setdefault(user, []).append(selected[user] - manual[user])
    if any(len(values) != len(seed_results) for values in differences.values()):
        raise ValueError("All seeds must evaluate the same users")
    means = np.array([np.mean(values) for _, values in sorted(differences.items())])
    contrast = float(np.mean(means)) if len(means) else None
    result = {"selected_minus_manual_ndcg_10": contrast, "users": len(means), "samples": samples, "bootstrap_seed": seed, "interval_95": None,
              "interpretation": "Exploratory interval conditional on the fixed cohort; no equivalence claim"}
    if len(means) >= 2:
        rng = np.random.default_rng(seed)
        replicates = np.array([rng.choice(means, size=len(means), replace=True).mean() for _ in range(samples)])
        result["interval_95"] = list(map(float, np.quantile(replicates, [0.025, 0.975])))
    return result


class ResourceMeasure:
    def __enter__(self):
        self.started = time.perf_counter()
        self.peak = 0
        self.stop = threading.Event()
        self.process = psutil.Process(os.getpid())
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def _sample(self):
        while not self.stop.is_set():
            total = 0
            for process in [self.process, *self.process.children(recursive=True)]:
                try:
                    total += process.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            self.peak = max(self.peak, total)
            self.stop.wait(0.1)

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        self.result = {"wall_seconds": time.perf_counter() - self.started, "peak_process_tree_rss_bytes": self.peak, "sampling_interval_seconds": 0.1}


def artifact_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
