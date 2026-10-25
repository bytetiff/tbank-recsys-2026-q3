"""Stages of the strong-features experiment.

    python -m strongfeat.run slices    # cohort events, 6 time slices, shared candidates and labels
    python -m strongfeat.run features  # manual and old-manual matrices for every slice
    python -m strongfeat.run screen    # full grammar on a user subsample, importance ranking (train + dev only)
    python -m strongfeat.run select    # choose K of the automatic set on selection
    python -m strongfeat.run final     # fit every variant with 3 seeds, open test once, bootstrap
    python -m strongfeat.run tune      # per-variant depth / learning rate grid, chosen on dev
    python -m strongfeat.run final --tuned

Selection never reads test; test metrics are written only by `final`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from catboost import CatBoostClassifier, Pool

from .common import (EXPERIMENT, ResourceMeasure, Slice, build_events, cutoffs, evaluate, file_sha256, history, load_config,
                     make_slice, paired_bootstrap, write_json)
from .features import Spec, compute, grammar, parse
from .manual import EXTRA_NAMES, MANUAL_SPECS, extras

SRC = ["src_history", "src_recent", "src_artist", "src_popular"]
PRE_TEST = ("train0", "train1", "train2", "dev", "selection")


class Run:
    def __init__(self, root: Path, out: Path):
        self.root, self.out, self.config = root, out, load_config()
        self.out.mkdir(parents=True, exist_ok=True)
        names = []
        counter = 0
        for split, q in cutoffs(self.config["end_exclusive"]):
            names.append((f"train{counter}" if split == "train" else split, q))
            counter += split == "train"
        self.slices = names

    # ---------- data ----------
    def events(self) -> pl.DataFrame:
        path = self.out / "events.parquet"
        if not path.exists():
            c = self.config
            events, items = build_events(self.root / c["prepared_events"], self.root / c["raw_events"], self.root / c["artist_map"], self.root / c["album_map"])
            events.write_parquet(path)
            items.write_parquet(self.out / "items.parquet")
            write_json(self.out / "data-manifest.json", {"prepared_events_sha256": file_sha256(self.root / c["prepared_events"]),
                                                         "artist_map_sha256": file_sha256(self.root / c["artist_map"]),
                                                         "album_map_sha256": file_sha256(self.root / c["album_map"]),
                                                         "events": events.height, "users": events["user_id"].n_unique(), "items": events["item_id"].n_unique()})
        return pl.read_parquet(path)

    def users(self) -> list[int]:
        return sorted(self.events()["user_id"].unique().to_list())

    def load_slice(self, name: str, users: list[int] | None = None) -> Slice:
        rows = pl.read_parquet(self.out / "slices" / f"{name}.parquet")
        meta = json.loads((self.out / "slices" / f"{name}.json").read_text())
        targets = {int(k): v for k, v in meta["targets"].items()}
        if users is not None:
            keep = set(users)
            rows = rows.filter(pl.col("user_id").is_in(users))
            targets = {u: n for u, n in targets.items() if u in keep}
        return Slice(name, meta["cutoff"], rows, targets, sorted(targets))

    def stage_slices(self):
        events, users = self.events(), self.users()
        items = pl.read_parquet(self.out / "items.parquet")
        summary = {}
        for name, q in self.slices:
            with ResourceMeasure() as m:
                s = make_slice(events, users, name, q, self.config)
            rows = s.rows.join(items, on="item_id", how="left")
            (self.out / "slices").mkdir(exist_ok=True)
            rows.write_parquet(self.out / "slices" / f"{name}.parquet")
            write_json(self.out / "slices" / f"{name}.json", {"cutoff": q, "targets": {str(u): n for u, n in s.targets.items()}})
            pos = int(rows["label"].sum()); n_targets = sum(s.targets.values())
            summary[name] = {"cutoff": q, "rows": rows.height, "rows_per_user": rows.height / len(users), "positives": pos,
                             "targets": n_targets, "candidate_recall_micro": pos / n_targets, "users_with_targets": sum(v > 0 for v in s.targets.values()),
                             "source_hits": {c: int(rows.filter(pl.col(c) == 1)["label"].sum()) for c in SRC}, **m.result}
            print(name, summary[name], flush=True)
        write_json(self.out / "slices-summary.json", summary)

    # ---------- feature matrices ----------
    def matrix(self, kind: str, name: str, specs: list[Spec] | None = None, users: list[int] | None = None, tag: str | None = None) -> pl.DataFrame:
        path = self.out / "features" / f"{tag or kind}-{name}.parquet"
        if path.exists():
            return pl.read_parquet(path)
        s = self.load_slice(name, users)
        past = history(self.events(), s.cutoff)
        if kind == "manual":
            frame = pl.concat([compute(past, s.rows, s.cutoff, MANUAL_SPECS), extras(past, s.rows, s.cutoff), s.rows.select(SRC)], how="horizontal_extend")
        elif kind == "grammar":
            frame = pl.concat([compute(past, s.rows, s.cutoff, specs), s.rows.select(SRC)], how="horizontal_extend")
        elif kind == "old":
            sys.path.insert(0, str(self.root / "src"))
            from tlab_recsys import features as old  # read-only import of the original feature code
            old_specs = old.grammar([7, 30, None])
            frame = old.compute_features(past.select("user_id", "item_id", "timestamp"), s.rows.select("user_id", "item_id"), s.cutoff, old_specs,
                                         {"missing_recency_seconds": 0})
        else:
            raise ValueError(kind)
        path.parent.mkdir(exist_ok=True)
        frame.write_parquet(path)
        return frame

    def stage_features(self):
        timing = {}
        for kind in ("manual", "old"):
            for name, _ in self.slices:
                with ResourceMeasure() as m:
                    frame = self.matrix(kind, name)
                timing[f"{kind}/{name}"] = {"columns": frame.width, **m.result}
                print(kind, name, timing[f"{kind}/{name}"], flush=True)
        write_json(self.out / "features-timing.json", timing)

    # ---------- models ----------
    def fit(self, train: list[pl.DataFrame], train_labels: np.ndarray, dev: pl.DataFrame, dev_labels: np.ndarray, seed: int,
            iterations: int | None = None, cat: list[str] | None = None, params: dict | None = None) -> CatBoostClassifier:
        c = self.config["catboost"] | (params or {})
        model = CatBoostClassifier(iterations=iterations or c["iterations"], learning_rate=c["learning_rate"], depth=c["depth"], l2_leaf_reg=c["l2_leaf_reg"],
                                   loss_function="Logloss", task_type="CPU", thread_count=c["thread_count"], random_seed=seed, verbose=False,
                                   allow_writing_files=False)
        X = pl.concat(train)
        model.fit(self.pool(X, train_labels, cat), eval_set=self.pool(dev, dev_labels, cat), early_stopping_rounds=c["early_stopping_rounds"], use_best_model=True)
        return model

    @staticmethod
    def pool(frame: pl.DataFrame, labels=None, cat: list[str] | None = None) -> Pool:
        if cat:
            pdf = frame.to_pandas()
            for col in cat:
                pdf[col] = pdf[col].astype(str)
            return Pool(pdf, label=labels, cat_features=cat)
        return Pool(frame.to_numpy().astype(np.float32), label=labels, feature_names=frame.columns)

    def labels(self, name: str, users=None) -> np.ndarray:
        return self.load_slice(name, users).rows["label"].to_numpy()

    def variant_frames(self, variant: str, name: str) -> tuple[pl.DataFrame, list[str] | None]:
        if variant == "old_manual":
            return self.matrix("old", name), ["user_id", "item_id"]
        if variant == "manual":
            return self.matrix("manual", name), None
        selected = json.loads((self.out / "selection.json").read_text())
        auto_names = selected["auto_features"]
        auto = self.matrix("grammar", name, [parse(n) for n in auto_names], tag="auto")
        auto = pl.concat([auto.select(auto_names), auto.select(SRC)], how="horizontal_extend")
        if variant == "auto":
            return auto, None
        if variant == "auto_manual":
            manual = self.matrix("manual", name)
            extra_auto = [n for n in auto_names if n not in manual.columns]
            return pl.concat([manual, auto.select(extra_auto)], how="horizontal_extend"), None
        raise ValueError(variant)

    def fit_variant(self, variant: str, seed: int, frames=None, cat=None, params: dict | None = None):
        if frames is None:
            got = {name: self.variant_frames(variant, name) for name in ("train0", "train1", "train2", "dev")}
            frames = {name: f for name, (f, _) in got.items()}
            cat = got["dev"][1]
        train_labels = np.concatenate([self.labels(n) for n in ("train0", "train1", "train2")])
        return self.fit([frames[n] for n in ("train0", "train1", "train2")], train_labels, frames["dev"], self.labels("dev"), seed, cat=cat, params=params)

    # ---------- automatic search ----------
    def stage_screen(self):
        cfg = self.config["search"]
        users = self.users()
        rng = np.random.default_rng(cfg["screen_seed"])
        screen_users = sorted(int(u) for u in rng.choice(users, size=cfg["screen_users"], replace=False))
        specs = grammar()
        with ResourceMeasure() as m_feat:
            frames = {n: self.matrix("grammar", n, specs, screen_users, tag="screen") for n in ("train0", "train1", "train2", "dev")}
        train_labels = np.concatenate([self.labels(n, screen_users) for n in ("train0", "train1", "train2")])
        with ResourceMeasure() as m_fit:
            model = self.fit([frames[n] for n in ("train0", "train1", "train2")], train_labels, frames["dev"], self.labels("dev", screen_users),
                             self.config["selection_seed"], iterations=cfg["screen_iterations"])
        model.save_model(str(self.out / "screen-model.cbm"))
        importance = dict(zip(frames["dev"].columns, map(float, model.get_feature_importance(type="PredictionValuesChange"))))
        ranked = sorted([n for n in importance if n not in SRC], key=lambda n: (-importance[n], n))
        # remove near-duplicates: keep the more important of any pair with |corr| > threshold on a row sample
        sample = pl.concat([frames[n] for n in ("train0", "train1", "train2")]).sample(n=min(60000, len(train_labels)), seed=1)
        kept, kept_values = [], []
        for name in ranked:
            if importance[name] <= 0:
                break
            v = sample[name].fill_null(-1).to_numpy().astype(np.float64)
            if v.std() == 0:
                continue
            v = (v - v.mean()) / v.std()
            if any(abs(float(v @ k) / len(v)) > cfg["dedup_corr"] for k in kept_values):
                continue
            kept.append(name); kept_values.append(v)
            if len(kept) >= max(cfg["k_grid"]):
                break
        write_json(self.out / "screen.json", {"screen_users": screen_users, "grammar_size": len(specs), "importance": importance,
                                              "ranked_after_dedup": kept, "nonzero_importance": sum(v > 0 for v in importance.values()),
                                              "best_iteration": int(model.get_best_iteration()), "feature_resources": m_feat.result, "fit_resources": m_fit.result})
        print("screen done", len(specs), len(kept), m_feat.result, m_fit.result, flush=True)

    def stage_select(self):
        screen = json.loads((self.out / "screen.json").read_text())
        pool_names = screen["ranked_after_dedup"]
        cfg = self.config["search"]
        seed = self.config["selection_seed"]
        trace = []
        sel = self.load_slice("selection")
        with ResourceMeasure() as m_feat:
            full = {n: self.matrix("grammar", n, [parse(x) for x in pool_names], tag="pool") for n, _ in self.slices if n != "test"}
        for k in cfg["k_grid"]:
            names = pool_names[:k] + SRC
            with ResourceMeasure() as m:
                model = self.fit([full[n].select(names) for n in ("train0", "train1", "train2")],
                                 np.concatenate([self.labels(n) for n in ("train0", "train1", "train2")]), full["dev"].select(names), self.labels("dev"), seed)
                scores = model.predict_proba(self.pool(full["selection"].select(names)))[:, 1]
            metrics = evaluate(sel, scores)["metrics"]
            trace.append({"k": k, "selection": metrics, "best_iteration": int(model.get_best_iteration()), **m.result})
            print("k", k, metrics["ndcg_10"], m.result, flush=True)
        best = min(trace, key=lambda r: (-r["selection"]["ndcg_10"], r["k"]))
        reference = {}
        for variant in ("manual", "old_manual"):
            frame_sel, cat = self.variant_frames(variant, "selection")
            model = self.fit_variant(variant, seed)
            reference[variant] = evaluate(sel, model.predict_proba(self.pool(frame_sel, cat=cat))[:, 1])["metrics"] | {"best_iteration": int(model.get_best_iteration())}
            print(variant, reference[variant]["ndcg_10"], flush=True)
        write_json(self.out / "selection.json", {"k_trace": trace, "chosen_k": best["k"], "auto_features": pool_names[: best["k"]],
                                                 "reference_on_selection": reference, "pool_feature_resources": m_feat.result})
        # materialise the chosen automatic set for every slice including test (features only, no labels used)
        for n, _ in self.slices:
            self.matrix("grammar", n, [parse(x) for x in pool_names[: best["k"]]], tag="auto")

    # ---------- hyperparameters ----------
    def stage_tune(self, variants: list[str]):
        """Grid over depth and learning rate for every boosted variant; fit on train0-2, early stopping and choice on dev NDCG@10."""
        grid = self.config["tuning"]
        dev = self.load_slice("dev")
        path = self.out / "tuning.json"
        result = json.loads(path.read_text()) if path.exists() else {}
        for variant in variants:
            got = {name: self.variant_frames(variant, name) for name in ("train0", "train1", "train2", "dev")}
            frames, cat = {n: f for n, (f, _) in got.items()}, got["dev"][1]
            trace = []
            for depth in grid["depth"]:
                for rate in grid["learning_rate"]:
                    params = {"depth": depth, "learning_rate": rate}
                    with ResourceMeasure() as m:
                        model = self.fit_variant(variant, self.config["selection_seed"], frames, cat, params)
                    ndcg = evaluate(dev, model.predict_proba(self.pool(frames["dev"], cat=cat))[:, 1])["metrics"]["ndcg_10"]
                    trace.append({**params, "dev_ndcg_10": ndcg, "best_iteration": int(model.get_best_iteration()), **m.result})
                    print(variant, params, ndcg, m.result, flush=True)
            best = min(trace, key=lambda r: (-r["dev_ndcg_10"], r["depth"], -r["learning_rate"]))
            result[variant] = {"trace": trace, "chosen": {"depth": best["depth"], "learning_rate": best["learning_rate"]}}
            write_json(path, result)

    # ---------- final ----------
    def stage_final(self, variants: list[str], tuned: bool = False):
        test = self.load_slice("test")
        results_path = self.out / ("final-tuned.json" if tuned else "final.json")
        tuning = json.loads((self.out / "tuning.json").read_text()) if tuned else {}
        results = json.loads(results_path.read_text()) if results_path.exists() else {"variants": {}}
        # non-learned baselines on test
        events = self.events()
        past = history(events, test.cutoff)
        base = compute(past, test.rows, test.cutoff, [Spec("item", "full", 7, "count"), Spec("user_item", "full", 90, "count"), Spec("user_item", "listen", None, "recency")])
        pop = base["item__full__7d__count"].fill_null(0).to_numpy().astype(float)
        rec = base["user_item__full__90d__count"].fill_null(0).to_numpy() - 1e-9 * base["user_item__listen__all__recency"].fill_null(1e12).to_numpy()
        results["baselines"] = {"popular": evaluate(test, pop, probabilities=False), "history_count": evaluate(test, rec, probabilities=False)}
        for variant in variants:
            runs = []
            frame_test, cat = self.variant_frames(variant, "test")
            for seed in self.config["seeds"]:
                with ResourceMeasure() as m:
                    model = self.fit_variant(variant, seed, params=tuning[variant]["chosen"] if tuned else None)
                with ResourceMeasure() as mp:
                    scores = model.predict_proba(self.pool(frame_test, cat=cat))[:, 1]
                ev = evaluate(test, scores)
                runs.append({"seed": seed, "metrics": ev["metrics"], "per_user": {r["user_id"]: r["ndcg_10"] for r in ev["per_user"]},
                             "best_iteration": int(model.get_best_iteration()), "fit": m.result, "predict": mp.result, "columns": frame_test.width,
                             "importance": dict(zip(frame_test.columns, map(float, model.get_feature_importance(type="PredictionValuesChange"))))})
                print(variant, seed, ev["metrics"]["ndcg_10"], m.result, flush=True)
            results["variants"][variant] = runs
            write_json(results_path, results)
        # paired bootstrap: per user, average across seeds, then contrast with manual
        def per_user_mean(runs):
            users = runs[0]["per_user"].keys()
            return {int(u): float(np.mean([r["per_user"][u] for r in runs])) for u in users}
        b = self.config["bootstrap"]
        reference = per_user_mean(results["variants"]["manual"])
        contrasts = {}
        for variant, runs in results["variants"].items():
            if variant != "manual":
                contrasts[f"{variant}-manual"] = paired_bootstrap(per_user_mean(runs), reference, b["samples"], b["seed"])
        for name, ev in results["baselines"].items():
            contrasts[f"{name}-manual"] = paired_bootstrap({r["user_id"]: r["ndcg_10"] for r in ev["per_user"]}, reference, b["samples"], b["seed"])
        results["contrasts_ndcg_10"] = contrasts
        write_json(results_path, results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["slices", "features", "screen", "select", "tune", "final"])
    parser.add_argument("--root", default=str(EXPERIMENT.parents[1]), help="project root that holds data/ and src/")
    parser.add_argument("--out", default=str(EXPERIMENT / "artifacts" / "main"))
    parser.add_argument("--tuned", action="store_true", help="final with the per-variant parameters from the tune stage")
    parser.add_argument("--variants", default="old_manual,manual,auto,auto_manual")
    args = parser.parse_args()
    run = Run(Path(args.root), Path(args.out))
    started = time.time()
    with ResourceMeasure() as m:
        if args.stage == "final":
            run.stage_final(args.variants.split(","), args.tuned)
        elif args.stage == "tune":
            run.stage_tune(args.variants.split(","))
        else:
            getattr(run, f"stage_{args.stage}")()
    log = run.out / "stage-resources.json"
    data = json.loads(log.read_text()) if log.exists() else {}
    data[args.stage] = m.result | {"finished_unix": int(time.time())}
    write_json(log, data)
    print(args.stage, "done", m.result, flush=True)


if __name__ == "__main__":
    main()
