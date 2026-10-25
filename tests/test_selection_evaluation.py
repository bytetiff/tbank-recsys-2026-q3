import math

import numpy as np
import polars as pl
import pytest
from sklearn.metrics import log_loss, roc_auc_score

from tlab_recsys.evaluation import bootstrap_contrast, evaluate
from tlab_recsys.protocol import Slice, require_train
from tlab_recsys.selection import greedy_select


def test_p5_budget_and_p6_tie_order():
    called = []
    def scorer(groups):
        called.append(groups)
        return 0.1 + 0.1 * len(groups)
    result = greedy_select(["b", "a", "c"], scorer, 4, 3, 0.0001)
    assert result["additional_fits"] == 4
    assert result["groups"] == ["a", "b"]
    assert called == [(), ("a",), ("b",), ("c",), ("a", "b")]


def test_search_retains_actual_core_and_rejects_invalid_initial():
    assert greedy_select(["a"], lambda groups: 0.2 if not groups else 0.1, 24, 4, 0.0001)["groups"] == []
    assert greedy_select(["a"], lambda groups: 0.2 if not groups else float("nan"), 24, 4, 0.0001)["score"] == 0.2
    assert greedy_select(["a"], lambda groups: 0.2, 0, 4, 0.0001)["additional_fits"] == 0
    for value in (None, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="Initial"):
            greedy_select(["a"], lambda groups: value, 24, 4, 0.0001)


@pytest.mark.parametrize("labels", [[], [0], [1, 1]])
def test_train_rejects_unusable(labels):
    with pytest.raises(ValueError):
        require_train(np.array(labels))


def test_ranking_uses_all_targets_and_stable_item_ties():
    rows = pl.DataFrame({"user_id": [1, 1, 2], "item_id": [20, 10, 10], "label": [0, 1, 0]})
    slice_ = Slice("test", 0, rows, {1: {10, 99}, 2: set()}, {10, 20}, [1, 2])
    result = evaluate(slice_, np.array([0.5, 0.5, 0.5]))
    assert result["metrics"]["recall_10"] == 0.5
    assert result["metrics"]["ndcg_10"] == pytest.approx(1 / (1 + 1 / np.log2(3)))
    assert result["metrics"]["candidate_coverage"] == 0.5
    assert result["metrics"]["excluded_no_targets"] == 1
    assert result["metrics"]["ndcg_10_conditional_catalog"] == 1


def test_one_class_auc_and_no_target_metrics_are_undefined():
    slice_ = Slice("test", 0, pl.DataFrame({"user_id": [1], "item_id": [1], "label": [0]}), {1: set()}, {1}, [1])
    result = evaluate(slice_, np.array([0.3]))
    assert result["metrics"]["roc_auc"] is None
    assert result["metrics"]["ndcg_10"] is None
    assert result["metrics"]["logloss"] > 0


def test_bootstrap_pairs_users_and_averages_seeds_first():
    def seed(delta):
        return {"manual": {"per_user": [{"user_id": 1, "ndcg_10": 0.1}, {"user_id": 2, "ndcg_10": 0.4}]}, "selected": {"per_user": [{"user_id": 1, "ndcg_10": 0.1 + delta}, {"user_id": 2, "ndcg_10": 0.4 + delta}]}}
    result = bootstrap_contrast([seed(0.2), seed(0.4)])
    assert result["selected_minus_manual_ndcg_10"] == pytest.approx(0.3)
    assert result["interval_95"] == pytest.approx([0.3, 0.3])
    assert result == bootstrap_contrast([seed(0.2), seed(0.4)])


# Keep the pre-optimization implementation as an independent behavioral oracle.
def _legacy_evaluate(slice_: Slice, scores: np.ndarray, probabilities: bool = True) -> dict:
    if len(scores) != slice_.rows.height or not np.isfinite(scores).all():
        raise ValueError("Predictions must be finite and match candidates")
    groups = {}
    for (user, item, label), score in zip(slice_.rows.select("user_id", "item_id", "label").iter_rows(), scores):
        groups.setdefault(user, []).append((item, int(label), float(score)))
    per_user = []
    for user in slice_.users:
        targets = slice_.targets[user]
        if not targets:
            continue
        ranked = sorted(groups.get(user, []), key=lambda value: (-value[2], value[0]))
        n_targets = len(targets)
        idcg = sum(1 / math.log2(rank + 2) for rank in range(min(n_targets, 10)))
        dcg = sum(label / math.log2(rank + 2) for rank, (_, label, _) in enumerate(ranked[:10]))
        available = {item for item, _, _ in ranked}
        catalog_n = len(slice_.catalog_targets[user]) if slice_.catalog_targets is not None else len(targets & slice_.catalog)
        record = {"user_id": user, "ndcg_10": dcg / idcg, "recall_10": sum(label for _, label, _ in ranked[:10]) / n_targets, "recall_20": sum(label for _, label, _ in ranked[:20]) / n_targets,
                  "candidate_coverage": len(targets & available) / n_targets, "catalog_coverage": catalog_n / n_targets, "target_count": n_targets}
        candidate_n = len(targets & available)
        record["recall_10_conditional_catalog"] = sum(label for _, label, _ in ranked[:10]) / catalog_n if catalog_n else None
        conditional_idcg = sum(1 / math.log2(rank + 2) for rank in range(min(catalog_n, 10)))
        record["ndcg_10_conditional_catalog"] = dcg / conditional_idcg if conditional_idcg else None
        record["candidate_target_count"] = candidate_n
        per_user.append(record)
    metrics = {}
    for key in ("ndcg_10", "recall_10", "recall_20", "candidate_coverage", "catalog_coverage", "recall_10_conditional_catalog", "ndcg_10_conditional_catalog"):
        values = [row[key] for row in per_user if row[key] is not None]
        metrics[key] = float(np.mean(values)) if values else None
    labels = slice_.rows["label"].to_numpy()
    clipped = np.clip(scores, 1e-15, 1 - 1e-15)
    metrics["logloss"] = float(log_loss(labels, clipped, labels=[0, 1])) if len(labels) and probabilities else None
    metrics["logloss_reason"] = None if metrics["logloss"] is not None else "uncalibrated popularity scores or empty candidates"
    metrics["roc_auc"] = float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None
    metrics["roc_auc_reason"] = None if metrics["roc_auc"] is not None else "empty or one-class labels"
    metrics["evaluated_users"] = len(per_user)
    metrics["excluded_no_targets"] = len(slice_.users) - len(per_user)
    metrics["rows"] = len(scores)
    return {"metrics": metrics, "per_user": per_user}


@pytest.mark.parametrize("case", ["shuffled_ties", "explicit_catalog", "empty_candidates", "empty_targets", "one_class", "duplicate_rows"])
@pytest.mark.parametrize("probabilities", [True, False])
def test_evaluate_matches_legacy_reference(case, probabilities):
    rng = np.random.default_rng(731)
    users = [6, 3, 2, 1, 4, 5]
    targets = {1: {1, 3, 5, 39}, 2: set(range(1, 26)), 3: set(), 4: {11, 40}, 5: {3}, 6: {1}}
    records = [(user, item, int(item in targets[user])) for user in users[1:] for item in range(1, 36)]
    if case == "empty_candidates":
        records = []
    elif case == "empty_targets":
        targets = {user: set() for user in users}
        records = [(user, item, 0) for user, item, _ in records]
    elif case == "one_class":
        records = [(user, item, 1) for user, item, _ in records]
    elif case == "duplicate_rows":
        records += [(1, 1, 1), (1, 1, 0), (4, 11, 1)]
    permutation = rng.permutation(len(records))
    records = [records[index] for index in permutation]
    rows = pl.DataFrame(records, schema={"user_id": pl.Int64, "item_id": pl.Int64, "label": pl.Int8}, orient="row")
    scores = rng.choice([0.0, 0.3, 0.7, 1.0], size=len(records)).astype(np.float32)
    catalog = set(range(1, 38))
    catalog_targets = {user: targets[user] & catalog for user in users} if case == "explicit_catalog" else None
    slice_ = Slice("selection", 0, rows, targets, catalog, users, catalog_targets)
    assert evaluate(slice_, scores, probabilities) == _legacy_evaluate(slice_, scores, probabilities)
