from __future__ import annotations

import json
from pathlib import Path

from .config import canonical_hash
from .download import file_hash


def _read_bound_json(root: Path, relative: str, frozen: dict) -> dict:
    path = root / relative
    expected = frozen.get("artifacts", {}).get(relative)
    if not expected or not path.is_file() or file_hash(path) != expected:
        raise ValueError(f"Report frozen-bound artifact mismatch: {path}")
    return json.loads(path.read_text(encoding="utf-8"))

def create_report(runs_root: Path, output: Path) -> str:
    lines = ["# Сравнение временных признаков", "", "Исследование разведочное. Программные проверки не являются формальным доказательством, а результаты не устанавливают эквивалентность или промышленную эффективность.", ""]
    found = False
    diagnostic_runs = []
    for path in sorted(runs_root.rglob("results.json")):
        status_path = path.parent / "run-status.json"
        status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
        if status.get("status") == "diagnostic":
            diagnostic_runs.append(str(path.parent))
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        frozen = json.loads((path.parent / "frozen.json").read_text(encoding="utf-8"))
        if not frozen.get("runtime_fingerprint"):
            diagnostic_runs.append(str(path.parent))
            continue
        declared = frozen["decision_hash"]
        if declared != canonical_hash({key: value for key, value in frozen.items() if key != "decision_hash"}) or result["decision_hash"] != declared:
            raise ValueError(f"Report result/frozen decision mismatch: {path}")
        data_manifest = _read_bound_json(path.parent, "data-manifest.json", frozen)
        if canonical_hash(data_manifest) != frozen.get("data_manifest_hash"):
            raise ValueError(f"Report data manifest hash mismatch: {path.parent}")
        found = True
        user_count = len(frozen["cohort_users"])
        seed_count = len(frozen["decisions"])
        label = "окончательный пилот" if (user_count, seed_count) == (100, 1) else "расширенный профиль" if (user_count, seed_count) == (1000, 3) else "эксперимент"
        section = "Синтетический контроль" if data_manifest["dataset"] == "synthetic" else f"{data_manifest['dataset'].upper()}, {label}"
        lines += [f"## {section}", "", f"Каталог: {path.parent.name}. Пользователей: {user_count}. Seeds модели: {seed_count}. Фиксация решения: [{frozen['decision_hash'][:16]}]({(path.parent / 'frozen.json').resolve().as_posix()}).", ""]
        if data_manifest["dataset"] == "synthetic":
            lines += ["Контроль использует искусственные данные и сокращённую конфигурацию. Его метрики проверяют работу контура и не относятся к качеству на KION или Yambda.", ""]
        for scope, data in result["catalogs"].items():
            lines += [f"### {scope}", "", "| Seed | Вариант | NDCG@10 | Recall@10 | Recall@20 | LogLoss | ROC-AUC | Полнота кандидатов | Полнота каталога | Пользователей с целью | Без целей |", "|---|---|---|---|---|---|---|---|---|---|---|"]
            def show(value):
                return "не определено" if value is None else f"{value:.6f}"
            for seed in data["seeds"]:
                for name, variant in seed.items():
                    if name == "model_seed":
                        continue
                    m = variant["metrics"]
                    values = [m[key] for key in ("ndcg_10", "recall_10", "recall_20", "logloss", "roc_auc", "candidate_coverage", "catalog_coverage")]
                    lines.append(f"| {seed['model_seed']} | {name} | " + " | ".join(map(show, values)) + f" | {m['evaluated_users']} | {m['excluded_no_targets']} |")
            contrast = data["contrast"]
            lines += ["", f"Парная разность selected − manual NDCG@10: {show(contrast['selected_minus_manual_ndcg_10'])}; разведочный 95% интервал: {contrast['interval_95']}; пользователей: {contrast['users']}.", ""]
            for seed in data["seeds"]:
                manual_users = {row["user_id"]: row["ndcg_10"] for row in seed["manual"]["per_user"]}
                selected_users = {row["user_id"]: row["ndcg_10"] for row in seed["selected"]["per_user"]}
                differences = [selected_users[user] - manual_users[user] for user in manual_users]
                higher = sum(value > 0 for value in differences)
                equal = sum(value == 0 for value in differences)
                lower = sum(value < 0 for value in differences)
                lines += [f"Seed {seed['model_seed']}: selected выше manual по NDCG@10 у {higher} пользователей, равенство у {equal}, ниже у {lower}. NDCG@10 популярности равен {show(seed['popularity']['metrics']['ndcg_10'])}.", ""]
        limited = result["catalogs"].get("limited_candidates")
        full = result["catalogs"].get("full_catalog")
        if limited and full and limited["users"] == full["users"] and limited["candidate_label_hash"] == full["candidate_label_hash"]:
            lines += ["В этом запуске строки ограниченных кандидатов и полного допустимого каталога совпали. Пропущенные цели отсутствуют в историческом каталоге выбранной когорты, а не потеряны из-за лимита кандидатов. Влияние размера когорты и особенностей публикации исходных событий отдельно не измерялось.", ""]
        lines += ["### Затраты и фиксация", "", f"Построение признаков: {frozen['features_resources']}. Итоговая оценка: {result['resources']}. Объём артефактов перед сохранением результатов: {result['artifact_bytes_before_results']} байт.", ""]
        for decision in frozen["decisions"]:
            tuning = _read_bound_json(path.parent, str(Path(f"seed-{decision['model_seed']}") / "tuning.json"), frozen)
            selection = _read_bound_json(path.parent, str(Path(f"seed-{decision['model_seed']}") / "selection.json"), frozen)
            lines += [f"Seed {decision['model_seed']}: параметры {decision['parameters']}; группы {decision['selected_groups']}; обучения {decision['fit_counts']}. Число признаков: { {name: record['feature_count'] for name, record in decision['models'].items()} }. Настройка: {tuning['resources']}; поиск: {selection['resources']}; переобучение: {decision['refit']['resources']}.", ""]
        lines += [f"Метаданные: {frozen['metadata_assumption']}. Когорта, границы, кандидаты и метки общие для сравниваемых вариантов. Финальные модели переобучены на train, dev и selection с исходными моментами строк и фиксированным числом итераций.", ""]
    if not found:
        lines += ["Завершённых оценок пока нет. Качество и ресурсы неизвестны.", ""]
    for status_path in runs_root.rglob("run-status.json"):
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") == "diagnostic":
            diagnostic_runs.append(str(status_path.parent))
    if diagnostic_runs:
        lines += ["## Исключённые диагностические запуски", "", "Следующие каталоги исключены из сравнения качества и затрат: " + ", ".join(sorted(set(diagnostic_runs))) + ".", ""]
    lines += ["## Ограничения", "", "KION содержит опубликованные последние взаимодействия пар, а не полный поток просмотров. Yambda сохраняет секунды с округлением до пяти секунд; цель — listen с played_ratio_pct > 50. Нулевые метки означают отсутствие наблюдаемого целевого события, а не отказ после показа. Популярность не имеет LogLoss, поскольку её оценки не калиброваны как вероятности.", "", "Первая грамматика использует только event_type=all. Поиск меняет группы временных окон, а события разных типов агрегируются вместе. Поиск по типам событий и более богатым связям пока не исследован. Выводы относятся к этой ограниченной грамматике и этому ручному набору.", "", "Параметры CatBoost настроены на manual и затем общие для всех вариантов; это может благоприятствовать manual. Ограниченная конечная грамматика и жадный поиск не гарантируют глобальный оптимум. Интервалы условны относительно фиксированной когорты и протокола. Результаты ограниченных кандидатов и полного каталога разделены. Признаки метаданных требуют допущения об исторической доступности статических категорий.", "", "Тесты проверяют конкретные границы, сохранение кратностей, независимость от будущих строк, эталонные агрегаты, конечность поиска и закрытие test до фиксации. Формальных доказательств не заявлено. Переносимость качества и масштабируемость требуют отдельных экспериментов."]
    text = "\n".join(lines) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    return text
