"""Write scope-separated reports from verified saved scores, without inference."""

import csv
import hashlib
import json
import math
from pathlib import Path


MASSIVE_LANGS = {"in": ("en", "ko", "ja", "es"), "out": ("fr", "de", "it")}
MT_DIRECTIONS = {
    "in": ("en-de", "en-cs", "en-ja", "de-en", "cs-en", "ja-en"),
    "out": ("en-zh", "en-ru", "en-uk", "zh-en", "ru-en", "uk-en"),
}
LANGUAGE_NAMES = dict(en="영어", ko="한국어", ja="일본어", es="스페인어", fr="프랑스어",
                      de="독일어", it="이탈리아어", cs="체코어", zh="중국어", ru="러시아어", uk="우크라이나어")
METHODS = {"transfer_only": "T", "contrastive_only": "C", "contrastive_then_transfer": "C→T", "alternative": "Alt"}
LOSSES = {"infonce": "InfoNCE", "centered_infonce": "Center", "gap_distance_infonce": "Distance",
          "gap_direction_infonce": "Direction", "gap_consistency": "Gap"}
STATUSES = {"completed": "완료", "running": "진행 중", "queued": "대기", "pending": "대기",
            "deferred": "후순위 보류",
            "not_requested": "대상 아님", "scoring_pending": "COMET 대기", "verification_pending": "검증 대기",
            "awaiting_result": "결과 대기"}
MASSIVE_METRICS = {"slot_f1": "Slot F1", "exact_match": "Exact Match", "slot_precision": "Slot Precision", "slot_recall": "Slot Recall"}
MT_METRICS = {"sacrebleu": "BLEU", "comet22_x100": "COMET-22 ×100"}
CONFIG_FIELDS = ("batch_size", "downstream_micro_batch_size", "accumulative_steps", "learning_rate",
                 "lr_scheduler_type", "warmup_ratio", "alignment_data", "alignment_num_samples_per_lang",
                 "alignment_sampling_seed", "alignment_hidden_state_position", "alignment_temperature",
                 "alignment_gap_scale", "alignment_max_length", "peft_lora_r", "peft_lora_alpha",
                 "peft_lora_dropout", "peft_target_modules", "quantization_load_in_4bit",
                 "quantization_type", "quantization_use_double_quant", "training_anchor_langs",
                 "training_lang", "out_inference_lang", "wmt23_corpus_profile", "wmt23_downstream_sampling",
                 "wmt23_manifest_sha256", "eval_language_scope")


def table(headers, rows):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(map(cell, headers)) + " |",
                      "| " + " | ".join(["---"] * len(headers)) + " |"] +
                     ["| " + " | ".join(map(cell, row)) + " |" for row in rows])


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                             for k, v in row.items()})


def number(value):
    return "—" if value is None else f"{value:.2f}"


def average(values):
    return math.fsum(values) / len(values) if values and all(v is not None for v in values) else None


def method(row):
    name = METHODS.get(row["method"], row["method"])
    return name if row["method"] == "transfer_only" else name + "/" + LOSSES.get(row["loss"], row["loss"])


def identity(row):
    return {k: row[k] for k in ("id", "display_id", "model", "method", "loss", "seed")}


def condition_rows(records):
    rows = []
    for record in records:
        config = record["config"]
        run_metadata = Path(record["run"]) / "run_metadata.json" if record["run"] else None
        metadata = json.loads(run_metadata.read_text()) if run_metadata and run_metadata.is_file() else {}
        updates = metadata.get("planned_objective_updates", {})
        derived = metadata.get("derived", {})
        rows.append({**identity(record), "applied_training_alignment_loss": "not_used" if record["method"] == "transfer_only" else record["loss"],
                     "alignment_layer": config.get("alignment_hidden_state_layer"),
                     "alignment_batching": config.get("alignment_batching", "mixed"),
                     "compute_dtype": config.get("quantization_compute_dtype"),
                     "target_steps": config["num_steps"],
                     "planned_alignment_updates": updates.get("alignment"), "planned_downstream_updates": updates.get("downstream"),
                     "effective_global_batch_size": derived.get("effective_global_batch_size"),
                     "alignment_pair_sizes": derived.get("alignment_pair_sizes"),
                     "downstream_language_sizes": derived.get("downstream_language_sizes"),
                     "trainable_parameters": derived.get("trainable_parameters"),
                     **{key: config.get(key) for key in CONFIG_FIELDS},
                     "evaluation_batch_size": record["evaluation_batch_size"],
                     "max_new_tokens": record["max_new_tokens"],
                     "evaluation_scopes": record["evaluation_language_scopes"],
                     "evaluation_status": record["status"], "eos_policy": record.get("eos_policy"),
                     "run": record["run"], "metadata_path": record.get("metadata_path"),
                     "run_metadata_path": str(run_metadata) if metadata else None,
                     "evaluation_dir": record.get("evaluation_dir"),
                     "config_source": record.get("config_source", record.get("metadata_path"))})
    return rows


def write_conditions(folder, records, updated, task):
    rows = condition_rows(records)
    write_csv(folder / "experiments.csv", rows)
    write_text(folder / "experiment_configs.json", json.dumps(
        {"updated_at": updated, "experiments": [{"id": r["id"], "display_id": r["display_id"],
         "config_source": r.get("config_source", r.get("metadata_path")), "config": r["config"]} for r in records]},
        ensure_ascii=False, indent=2, allow_nan=False))
    lines = [f"# {task} 실험 조건", "", f"집계 시각: {updated}", "",
             "[In 결과](in.md) · [Out 결과](out.md) · [전체 조건 CSV](experiments.csv) · [전체 설정 JSON](experiment_configs.json)", "",
             "T=transfer_only, C=contrastive_only, C→T=contrastive_then_transfer, Alt=alternative. "
             "Center=centered_infonce, Distance=gap_distance_infonce, Direction=gap_direction_infonce, Gap=gap_consistency.", "",
             "T의 저장된 loss는 실제 정렬 학습에 사용되지 않는다. C는 downstream SFT가 없다. "
             "ID는 이 보고서 내 행 식별자이며 원본 run 경로가 실행을 식별한다. 층·dtype·seed·batching이 다른 실행을 합치지 않았다.", "",
             "## 학습 설정", "", table(
                 ["ID", "모델", "방법", "층", "seed", "batching", "dtype", "steps", "train batch", "micro batch", "LR"],
                 [[r["display_id"], r["model"].split("/")[-1], method(r), r["alignment_layer"], r["seed"],
                   r["alignment_batching"], r["compute_dtype"], r["target_steps"], r["batch_size"],
                   r["downstream_micro_batch_size"] if r["downstream_micro_batch_size"] is not None else "미기록", r["learning_rate"]] for r in rows]), "",
             "층 -1은 마지막 층이다. micro batch 0은 별도 분할을 비활성화한다. 미기록 값은 다른 실행에서 추정하지 않았다.", "",
             "## 목적함수별 학습 횟수", "",
             "완료 실행의 run_metadata에 기록된 예정 update 수다. 총 step과 정렬/downstream update를 구분한다. "
             "원본 학습 데이터의 언어별 구성 수, 유효 global batch, LoRA·양자화·scheduler 설정은 조건 CSV에 함께 보존한다.", "",
             table(["ID", "방법", "총 steps", "정렬 updates", "downstream updates"],
                   [[r["display_id"], method(r), r["target_steps"],
                     r["planned_alignment_updates"] if r["planned_alignment_updates"] is not None else "미기록",
                     r["planned_downstream_updates"] if r["planned_downstream_updates"] is not None else "미기록"] for r in rows]), "",
             "## 인퍼런스 설정", "", table(["ID", "범위", "상태", "eval batch", "max new tokens", "EOS 정책"],
                 [[r["display_id"], ",".join(r["evaluation_scopes"]), STATUSES.get(r["evaluation_status"], r["evaluation_status"]),
                   r["evaluation_batch_size"], r["max_new_tokens"], r["eos_policy"] or "기존 MASSIVE tokenizer EOS"] for r in rows]), "",
             "`eval_language_scope`는 학습 중 validation 설정이다. 실제 test 평가 범위는 `evaluation_scopes` 열에 별도로 기록한다.", "",
             "## 원본 실행", ""]
    for row in rows:
        lines.append(f"- {row['display_id']} (`{row['id']}`): [checkpoint]({row['run']})" +
                     (f" · [평가 metadata]({row['metadata_path']})" if row["metadata_path"] else ""))
    write_text(folder / "experiments.md", "\n".join(lines))


def write_massive(folder, records, updated):
    folder = Path(folder)
    records = [{**r, "display_id": r["id"], "status": "completed", "evaluation_language_scopes": ["in", "out"]} for r in records]
    write_conditions(folder, records, updated, "MASSIVE")
    for scope, languages in MASSIVE_LANGS.items():
        flat, macros = [], []
        for record in records:
            for language in languages:
                score = record["scores"][language]
                flat.append({**identity(record), "scope": scope, "language": language,
                             "language_name": LANGUAGE_NAMES[language], "status": "completed", "num_examples": score["num_examples"],
                             **{key + "_x100": score[key] * 100 for key in MASSIVE_METRICS},
                             "metrics_path": score["metrics_path"], "metrics_sha256": score["metrics_sha256"]})
            macros.append({**identity(record), "scope": scope, "aggregation": "equal_language_macro",
                           "num_languages": len(languages), "num_examples": sum(record["scores"][l]["num_examples"] for l in languages),
                           **{key + "_x100": average([record["scores"][l][key] * 100 for l in languages]) for key in MASSIVE_METRICS}})
        write_csv(folder / f"{scope}.csv", flat)
        write_csv(folder / f"{scope}_macro.csv", macros)
        lines = [f"# MASSIVE {scope.upper()} — 언어별 test 성능", "", f"집계 시각: {updated}", "",
                 "[실험 조건](experiments.md) · [언어별 CSV](" + scope + ".csv) · [macro CSV](" + scope + "_macro.csv)", "",
                 f"언어: {', '.join(f'{l} ({LANGUAGE_NAMES[l]})' for l in languages)}. 언어별 2,974개. 완료 실행 {len(records)}개.", "",
                 "모든 수치는 ×100이며 높을수록 좋다. 한 표는 한 지표만 표시한다. Macro는 이 범위 내 언어 점수의 단순평균이다. "
                 "언어별 F1은 해당 언어의 슬롯 TP/FP/FN을 합산한 micro F1이다. 언어 macro F1은 전체 슬롯을 합친 micro F1과 다르다.", "",
                 "0.00은 실제 점수이며 누락값이 아니다. EM은 정규화한 슬롯 이름·값 목록이 완전히 같은 문장 비율이며 intent accuracy가 아니다.", ""]
        for model in dict.fromkeys(r["model"] for r in records):
            selected = [r for r in records if r["model"] == model]
            lines.extend([f"## {model}", ""])
            for key, label in MASSIVE_METRICS.items():
                lines.extend([f"### {label} ×100", "", table(["ID", "방법", "층", "seed", "dtype", "batching", *languages, "언어 macro"],
                    [[r["display_id"], method(r), r["alignment_layer"], r["seed"], r["compute_dtype"], r["alignment_batching"],
                      *[number(r["scores"][l][key] * 100) for l in languages],
                      number(average([r["scores"][l][key] * 100 for l in languages]))] for r in selected]), ""])
        write_text(folder / f"{scope}.md", "\n".join(lines))


def mt_score_rows(folder, scope, bleu, comet=None):
    """Normalize already verified metric reports; do not read prediction contents."""
    folder = Path(folder)
    bleu_path, comet_path = folder / "wmt23_metrics.json", folder / "wmt23_comet22_metrics.json"
    bleu_hash = hashlib.sha256(bleu_path.read_bytes()).hexdigest()
    comet_hash = hashlib.sha256(comet_path.read_bytes()).hexdigest() if comet else None
    scores = {}
    for direction, b in bleu["by_language"].items():
        c = comet["by_language"][direction] if comet else {}
        meta = comet.get("scorer_metadata", {}) if comet else {}
        scores[direction] = {
            "scope": scope, "direction": direction, "num_examples": b["num_examples"],
            "sacrebleu": b["sacrebleu"], "comet22": c.get("comet22"), "comet22_x100": c.get("comet22_x100"),
            "generation_limit_hits": b["generation_limit_hits"], "paragraph_mismatches": b.get("paragraph_mismatches"),
            "metric_signature": b.get("metric_signature"), "comet_model": meta.get("model_id"),
            "comet_revision": meta.get("model_revision"), "comet_checkpoint_sha256": meta.get("checkpoint_sha256"),
            "bleu_metrics_path": str(bleu_path), "comet_metrics_path": str(comet_path) if comet else None,
            "bleu_metrics_sha256": bleu_hash, "comet_metrics_sha256": comet_hash,
        }
    return scores


def scope_status(record, scope, directions):
    if scope not in record["evaluation_language_scopes"]:
        return "not_requested"
    scores = [record["scores"].get(d, {}) for d in directions]
    if all(s.get(k) is not None for s in scores for k in MT_METRICS):
        return "completed"
    if all(s.get("sacrebleu") is not None for s in scores):
        return "scoring_pending"
    if record["status"] == "completed":
        return "verification_pending"
    return "awaiting_result" if record["status"] == "running" else record["status"]


def write_mt(folder, records, counts, updated):
    folder = Path(folder)
    records = [{**r, "display_id": f"W{i:02d}"} for i, r in enumerate(sorted(records, key=lambda r: r["id"]), 1)]
    write_conditions(folder, records, updated, "WMT23")
    status_rows, diagnostics = [], []
    for scope, standard_directions in MT_DIRECTIONS.items():
        directions = [d for d in standard_directions if d in counts.get(scope, {})]
        if not directions:
            continue
        groups = {"en_to_x": [d for d in directions if d.startswith("en-")],
                  "x_to_en": [d for d in directions if d.endswith("-en")], "all_directions": directions}
        flat, macros = [], []
        for record in records:
            status = scope_status(record, scope, directions)
            applicable = status != "not_requested"
            status_rows.append({**identity(record), "scope": scope, "scope_status": status,
                                "job_status": record["status"], "gpu": record.get("gpu"),
                                "required_directions": len(directions) if applicable else 0,
                                "verified_directions": sum(all(record["scores"].get(d, {}).get(k) is not None for k in MT_METRICS) for d in directions) if applicable else 0,
                                "required_examples": sum(counts[scope].values()) if applicable else 0})
            for direction in directions:
                s = record["scores"].get(direction, {}) if applicable else {}
                src, tgt = direction.split("-")
                row = {**identity(record), "scope": scope, "direction": direction, "source_language": src,
                       "target_language": tgt, "translation_group": "en_to_x" if src == "en" else "x_to_en",
                       "scope_status": status, "job_status": record["status"], "dataset_num_examples": counts[scope][direction],
                       "evaluated_num_examples": s.get("num_examples"),
                       **{k: s.get(k) for k in ("sacrebleu", "comet22", "comet22_x100", "metric_signature",
                          "comet_model", "comet_revision", "comet_checkpoint_sha256", "bleu_metrics_path",
                          "comet_metrics_path", "bleu_metrics_sha256", "comet_metrics_sha256")}}
                flat.append(row)
                if s:
                    diagnostics.append({**identity(record), "scope": scope, "direction": direction,
                        "num_examples": s["num_examples"], "generation_limit_hits": s["generation_limit_hits"],
                        "generation_limit_rate_pct": 100 * s["generation_limit_hits"] / s["num_examples"],
                        "paragraph_mismatches": s["paragraph_mismatches"], "max_new_tokens": record["max_new_tokens"],
                        "bleu_metrics_path": s["bleu_metrics_path"]})
            for group, selected in groups.items():
                macros.append({**identity(record), "scope": scope, "translation_group": group, "scope_status": status,
                    "aggregation": "equal_direction_macro", "num_directions": len(selected),
                    **{k: average([record["scores"].get(d, {}).get(k) for d in selected]) if applicable else None for k in MT_METRICS}})
        write_csv(folder / f"{scope}.csv", flat)
        write_csv(folder / f"{scope}_macro.csv", macros)
        lines = [f"# WMT23 {scope.upper()} — 번역 방향별 test 성능", "", f"집계 시각: {updated}", "",
                 f"[실험 조건](experiments.md) · [방향별 CSV]({scope}.csv) · [방향군별 macro CSV]({scope}_macro.csv) · [평가 상태](status.md) · [생성 진단](generation_diagnostics.md)", "",
                 "방법 표기: **Alt/Distance = alternative + gap_distance_infonce (Distance Gap)**, "
                 "Alt/Center = alternative + centered_infonce, Alt/InfoNCE = alternative + infonce. "
                 "C = contrastive_only, C→T = contrastive_then_transfer, T = transfer_only.", "",
                 f"이 범위의 BLEU·COMET 모두 완료: {sum(scope_status(r, scope, directions) == 'completed' for r in records)} / "
                 f"{sum(scope_status(r, scope, directions) != 'not_requested' for r in records)}개 평가 대상 조건.", "",
                 "BLEU와 COMET은 각각 별도 표다. 영어→다른 언어와 다른 언어→영어를 구분한다. "
                 "Macro는 지정한 방향의 단순평균이며 문장 수 가중평균이나 합친 corpus BLEU가 아니다. "
                 "COMET-22는 원점수 ×100이며 CSV에 원점수도 보존한다.", "",
                 "`—`는 아직 검증된 점수가 없음, `대상 아님`은 이 평가에서 제외한 범위다. 0.00과 구분한다. "
                 "기존 In+Out 평가의 완료 결과는 보존하며, In만 지정한 후속 평가는 Out 완료를 기다리지 않는다.", "",
                 table(["방향", "출발 언어", "도착 언어", "test 예제 수"],
                       [[d, LANGUAGE_NAMES[d.split('-')[0]], LANGUAGE_NAMES[d.split('-')[1]], counts[scope][d]] for d in directions]), ""]
        for key, label in MT_METRICS.items():
            lines.extend([f"## {label}", ""])
            for group, title in (("en_to_x", "영어 → 다른 언어"), ("x_to_en", "다른 언어 → 영어")):
                selected = groups[group]
                rows = []
                for record in records:
                    status = scope_status(record, scope, directions)
                    values = [record["scores"].get(d, {}).get(key) for d in selected]
                    rows.append([record["display_id"], record["model"].split("/")[-1], method(record), STATUSES.get(status, status),
                                 *(["대상 아님"] * (len(selected) + 1) if status == "not_requested" else
                                   [*[number(v) for v in values], number(average(values))])])
                lines.extend([f"### {title}", "", table(["ID", "모델", "방법", "범위 상태", *selected, "방향군 macro"], rows), ""])
            lines.extend(["### 전체 방향 macro", "", table(["ID", "모델", "방법", "범위 상태", label],
                [[r["display_id"], r["model"].split("/")[-1], method(r), STATUSES.get(r["scope_status"], r["scope_status"]),
                  "대상 아님" if r["scope_status"] == "not_requested" else number(r[key])]
                 for r in macros if r["translation_group"] == "all_directions"]), ""])
        write_text(folder / f"{scope}.md", "\n".join(lines))
    write_csv(folder / "status.csv", status_rows)
    write_text(folder / "status.md", "\n".join(["# WMT23 평가 상태", "", f"집계 시각: {updated}", "",
        "[In](in.md) · [Out](out.md) · [CSV](status.csv)", "",
        "방향 완료는 검증된 BLEU와 COMET이 모두 있을 때만 인정한다. 개별 배치 진행률은 이 결과 보고서에 포함하지 않는다. "
        "결과 대기는 실행이 진행 중이나 해당 범위의 검증된 결과가 없음을 뜻하며, 그 범위의 생성이 시작됐다는 뜻은 아니다. "
        "실행 전체의 상태와 In/Out별 결과 상태를 따로 기록한다. 대상 아님은 필수 방향·예제 수가 0이고 성능 값은 비워 둔다.", "",
        table(["ID", "모델", "방법", "범위", "범위 결과", "실행 상태", "완료/필수 방향", "필수 예제", "GPU"],
              [[r["display_id"], r["model"].split("/")[-1], method(r), r["scope"], STATUSES.get(r["scope_status"], r["scope_status"]),
                STATUSES.get(r["job_status"], r["job_status"]),
                f"{r['verified_directions']}/{r['required_directions']}", r["required_examples"], r["gpu"] if r["gpu"] is not None else "—"] for r in status_rows])]))
    diagnostic_fields = ["id", "display_id", "model", "method", "loss", "seed", "scope", "direction", "num_examples",
                         "generation_limit_hits", "generation_limit_rate_pct", "paragraph_mismatches", "max_new_tokens", "bleu_metrics_path"]
    write_csv(folder / "generation_diagnostics.csv", diagnostics, diagnostic_fields)
    lines = ["# WMT23 생성 진단", "", f"집계 시각: {updated}", "", "[전체 방향별 CSV](generation_diagnostics.csv)", "",
             "BLEU/COMET 성능과 구분한 생성 길이·문단 형식 진단이다. 길이 상한 도달은 오류 판정 자체가 아니며 문단 수 불일치도 오역 여부를 판정하지 않는다.", ""]
    for scope in MT_DIRECTIONS:
        lines.extend([f"## {scope.upper()}", "", table(["ID", "모델", "방법", "예제 수", "길이 상한 도달", "도달률 %", "문단 수 불일치"],
            [[r["display_id"], r["model"].split("/")[-1], method(r), sum(d["num_examples"] for d in selected),
              sum(d["generation_limit_hits"] for d in selected), number(100 * sum(d["generation_limit_hits"] for d in selected) / sum(d["num_examples"] for d in selected)),
              sum(d["paragraph_mismatches"] or 0 for d in selected)]
             for r in records if (selected := [d for d in diagnostics if d["id"] == r["id"] and d["scope"] == scope])]), ""])
    write_text(folder / "generation_diagnostics.md", "\n".join(lines))
