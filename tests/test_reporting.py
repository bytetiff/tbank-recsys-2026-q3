import pytest

from tlab_recsys.config import canonical_hash, write_json
from tlab_recsys.reporting import create_report
from tlab_recsys.download import file_hash
from pathlib import Path
import json


def seed_report_run(root, dataset, cohort_size=100, seeds=(17,)):
    manifest = {"dataset": dataset}
    write_json(root / "data-manifest.json", manifest)
    decisions = []
    for seed in seeds:
        write_json(root / f"seed-{seed}/tuning.json", {"resources": {}})
        write_json(root / f"seed-{seed}/selection.json", {"resources": {}})
        decisions.append({"model_seed": seed, "parameters": {}, "selected_groups": [], "fit_counts": {}, "models": {}, "refit": {"resources": {}}})
    frozen = {"runtime_fingerprint": {"code": "bound"}, "features_resources": {}, "decisions": decisions, "metadata_assumption": "none", "cohort_users": list(range(cohort_size)),
              "data_manifest_hash": canonical_hash(manifest), "artifacts": {str(path.relative_to(root)): file_hash(path) for path in root.rglob("*.json")}}
    frozen["decision_hash"] = canonical_hash(frozen)
    write_json(root / "frozen.json", frozen)
    write_json(root / "results.json", {"dataset": dataset, "decision_hash": frozen["decision_hash"], "catalogs": {}, "resources": {}, "artifact_bytes_before_results": 0})


def test_report_rejects_results_from_another_decision(tmp_path):
    seed_report_run(tmp_path / "run", "kion")
    write_json(tmp_path / "run" / "results.json", {"decision_hash": "another"})
    with pytest.raises(ValueError, match="mismatch"):
        create_report(tmp_path, tmp_path / "report.md")


def test_report_separates_control_and_omits_diagnostic_and_legacy(tmp_path):
    seed_report_run(tmp_path / "control", "synthetic")
    seed_report_run(tmp_path / "pilot", "kion")
    seed_report_run(tmp_path / "rejected", "yambda")
    write_json(tmp_path / "rejected" / "run-status.json", {"status": "diagnostic"})
    write_json(tmp_path / "legacy" / "frozen.json", {})
    write_json(tmp_path / "legacy" / "results.json", {})
    text = create_report(tmp_path, tmp_path / "report.md")
    assert "## Синтетический контроль" in text
    assert "## KION, окончательный пилот" in text
    assert "## YAMBDA" not in text
    assert "Исключённые диагностические запуски" in text
    assert "legacy" in text


def test_incomplete_diagnostic_is_skipped_before_opening_results_or_frozen(tmp_path):
    run = tmp_path / "diagnostic"
    write_json(run / "run-status.json", {"status": "diagnostic"})
    (run / "results.json").write_text("incomplete invalid JSON", encoding="utf-8")
    text = create_report(tmp_path, tmp_path / "report.md")
    assert "Исключённые диагностические запуски" in text


@pytest.mark.parametrize("relative", ["data-manifest.json", "seed-17/tuning.json", "seed-17/selection.json"])
def test_report_rejects_corrupt_frozen_bound_artifacts(tmp_path, relative):
    run = tmp_path / "run"
    seed_report_run(run, "kion")
    write_json(run / relative, {"corrupted": True})
    with pytest.raises(ValueError, match="artifact mismatch"):
        create_report(tmp_path, tmp_path / "report.md")


def test_report_checks_data_manifest_semantic_hash_even_if_file_hash_matches(tmp_path):
    run = tmp_path / "run"
    seed_report_run(run, "kion")
    frozen = json.loads((run / "frozen.json").read_text(encoding="utf-8"))
    frozen["data_manifest_hash"] = "wrong"
    frozen["decision_hash"] = canonical_hash({key: value for key, value in frozen.items() if key != "decision_hash"})
    write_json(run / "frozen.json", frozen)
    result = json.loads((run / "results.json").read_text(encoding="utf-8"))
    result["decision_hash"] = frozen["decision_hash"]
    write_json(run / "results.json", result)
    with pytest.raises(ValueError, match="data manifest hash"):
        create_report(tmp_path, tmp_path / "report.md")


def test_expanded_report_label_uses_stored_cohort_and_seed_counts(tmp_path):
    seed_report_run(tmp_path / "expanded", "yambda", cohort_size=1000, seeds=(17, 29, 43))
    text = create_report(tmp_path, tmp_path / "report.md")
    assert "YAMBDA, расширенный профиль" in text
    assert "Пользователей: 1000" in text and "Seeds модели: 3" in text
    assert "окончательный пилот" not in text
