"""Summarize verified final WMT23 runs; keep validation loss and test scores distinct."""

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.select_checkpoint import collect_losses
from scripts.wmt23_pipeline import (
    comet_complete, completed_run, evaluation_counts, find_completed, generation_complete, read_json,
)
from scripts.downstream_report_tables import mt_score_rows, table, write_csv, write_mt, write_text


def mean(values):
    return math.fsum(values) / len(values) if values else None


def live_step(log, total):
    if not log or not Path(log).is_file():
        return None
    with Path(log).open("rb") as handle:
        handle.seek(max(0, Path(log).stat().st_size - 65536))
        text = handle.read().decode("utf-8", errors="replace")
    steps = re.findall(rf"(\d+)/{total}\s*\[", text)
    return int(steps[-1]) if steps else None


def collect(manifest, states):
    records, details = [], []
    for job in manifest["jobs"]:
        counts = evaluation_counts(manifest, job)
        status = states.get(job["id"], {})
        pinned = status.get("run")
        run = (Path(pinned) if pinned and completed_run(pinned, job)
               else find_completed(manifest["results_root"], job))
        complete = run is not None
        record = {
            "job_id": job["id"], "model": job["model"], "method": job["mode"], "loss": job["loss"],
            "priority": job["priority"], "training_status": "completed" if complete else status.get("training_status", "pending"),
            "evaluation_scopes": ",".join(counts), "test_num_examples": sum(sum(v.values()) for v in counts.values()),
            "evaluation_status": status.get("evaluation_status", "queued" if job["priority"] else "not_queued"),
            "scheduled_gpu": status.get("evaluation_gpu", status.get("gpu", job["gpu"])),
            "step": job["steps"] if complete else live_step(status.get("log"), job["steps"]),
            "target_steps": job["steps"], "run": str(run or pinned or ""),
            "mt_validation_macro": None, "mt_validation_de": None,
            "mt_validation_cs": None, "mt_validation_ja": None,
            "alignment_in_loss_macro": None, "alignment_out_loss_macro": None,
            "bleu_in": None, "bleu_out": None, "comet22_in_x100": None, "comet22_out_x100": None,
        }
        detail = {"job_id": job["id"], "validation": {}, "test": {}}
        if complete:
            state = read_json(run / "trainer_state.json")
            losses = collect_losses(state["log_history"]).get(job["steps"], {})
            detail["validation"] = losses
            mt = losses.get("wmt23_in", {})
            # Never substitute a previous/best checkpoint's validation for the final one.
            if set(mt) == set(job["config"]["training_lang"]):
                record["mt_validation_macro"] = mean(list(mt.values()))
                for language, value in mt.items():
                    record[f"mt_validation_{language}"] = value
            for scope in ("in", "out"):
                record[f"alignment_{scope}_loss_macro"] = mean(list(losses.get(f"align_{scope}", {}).values()))
            output = run / "evaluations" / job.get("evaluation_id", manifest["evaluation_id"])
            if generation_complete(output, run, counts, manifest["max_new_tokens"],
                                   manifest.get("wmt23_batch_size", 1), manifest.get("wmt_eos_policy")):
                for scope in counts:
                    folder = output / "test" / scope
                    bleu = read_json(folder / "wmt23_metrics.json")
                    record[f"bleu_{scope}"] = bleu["macro_average"]["sacrebleu"]
                    detail["test"][scope] = {"bleu": bleu}
                    if comet_complete(folder, counts[scope]):
                        comet = read_json(folder / "wmt23_comet22_metrics.json")
                        record[f"comet22_{scope}_x100"] = comet["macro_average"]["comet22_x100"]
                        detail["test"][scope]["comet22"] = comet
        records.append(record)
        details.append(detail)
    return records, details


def render_validation(records, updated):
    def score(value):
        return "—" if value is None else f"{value:.4f}"

    lines = [
        "# WMT23 학습 진단 — validation loss", "", f"집계 시각: {updated}", "",
        "이 파일은 학습 진단용이다. 요청한 BLEU·COMET 번역 평가 결과는 summary.md에 기록한다.", "",
        "검증된 최종 adapter의 마지막 학습 step에서 기록한 validation loss만 비교한다.",
        "MT validation macro는 DE/CS/JA 언어쌍별 CE의 단순평균이며 낮을수록 좋다.",
        "각 언어쌍의 validation에는 양방향 번역이 포함된다. 서로 다른 모델의 CE는 tokenizer가 달라 직접 비교하지 않는다.",
        "`—`는 미완료/미평가이며 0점이 아니다. Validation CE와 test BLEU/COMET은 다른 지표다.", "",
        "| 모델 | 학습 방법 | 정렬 loss | 상태 | step | MT val macro | DE | CS | JA |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in records:
        lines.append(f"| {row['model'].split('/')[-1]} | {row['method']} | {row['loss']} | {row['training_status']} | "
                     f"{row['step'] if row['step'] is not None else '—'}/{row['target_steps']} | "
                     + " | ".join(score(row[key]) for key in ("mt_validation_macro", "mt_validation_de", "mt_validation_cs", "mt_validation_ja")) + " |")
    lines.extend(["", "## 해석 범위", "",
                  "- Contrastive-only는 MT SFT가 없으므로 SFT를 수행한 조건과 구분한다.",
                  "- InfoNCE와 centered InfoNCE는 계산하는 정렬 목적함수가 달라 alignment loss의 절댓값을 직접 순위로 비교하지 않는다.",
                  "- 현재 비교는 seed 42의 단일 실행이다. 번역 성능과 미학습 언어로의 전이는 test BLEU/COMET 평가 후 판단한다.",
                  "- [Validation CSV](validation.csv)에 CE만 기록하며, [정렬 validation](alignment_validation.md)은 별도로 분리한다.",
                  "- 언어별 validation 및 원본 test metric은 [details.json](details.json)에 보존한다. [Test 결과 목차](summary.md).", ""])
    return "\n".join(lines)


def render(records, updated):
    rows = [row for row in records if row["priority"]]
    completed = sum(all(row[key] is not None
                        for scope in row["evaluation_scopes"].split(",")
                        for key in (f"bleu_{scope}", f"comet22_{scope}_x100")) for row in rows)
    lines = [
        "# WMT23 BLEU·COMET-22 평가 결과", "", f"집계 시각: {updated}", "",
        f"평가 큐: {len(rows)}개 모델 설정. BLEU와 COMET 모두 완료: {completed}/{len(rows)}.",
        "최종 checkpoint의 지정된 범위를 평가한다: In 9,320개, Out 11,747개. In+Out이면 21,067개다.",
        "In/Out 파일에서 BLEU와 COMET, 영어→타 언어와 타 언어→영어, 방향별 점수와 macro를 각각 구분한다.",
        "`—`는 아직 없는 결과, `대상 아님`은 해당 실험에서 제외한 범위다. 둘 다 0점이 아니며 validation loss로 대체하지 않는다.", "",
        table(["범위", "언어", "결과", "CSV", "Macro CSV"], [
            ["In", "en↔de·cs·ja", "[In 보고서](test/in.md)", "[방향별](test/in.csv)", "[방향군별](test/in_macro.csv)"],
            ["Out", "en↔zh·ru·uk", "[Out 보고서](test/out.md)", "[방향별](test/out.csv)", "[방향군별](test/out_macro.csv)"],
        ]), "",
        "- [실험 조건](test/experiments.md) · [설정 CSV](test/experiments.csv)",
        "- [In/Out별 상태](test/status.md) · [생성 진단](test/generation_diagnostics.md)",
        "- [MT validation CE](validation.md) · [정렬 validation In/Out loss](alignment_validation.md)",
        "- [MASSIVE와 MT 통합 목차](../downstream_per_language_20261006/summary.md)",
        "- [전체 범위 macro CSV](summary.csv) · [원본 지표 JSON](details.json)", "",
        "## 현재 평가 큐", "",
        table(["모델", "학습 방법", "저장된 정렬 loss", "평가 상태", "지정 범위"],
              [[r["model"].split("/")[-1], r["method"], r["loss"], r["evaluation_status"], r["evaluation_scopes"]] for r in rows]),
    ]
    lines.extend(["", "큐의 결과 집계 시 분리된 보고서와 CSV도 함께 자동 갱신한다. T의 저장된 loss는 정렬 학습에 사용되지 않는다.", ""])
    return "\n".join(lines)


def write_separated_reports(output, manifest, records, details, updated):
    jobs = {job["id"]: job for job in manifest["jobs"]}
    detail_by_id = {detail["job_id"]: detail for detail in details}
    experiments = []
    for row in records:
        if not row["priority"]:
            continue
        job = jobs[row["job_id"]]
        run = Path(row["run"]) if row["run"] else None
        evaluation = run / "evaluations" / job.get("evaluation_id", manifest["evaluation_id"]) if run else None
        metadata = evaluation / "test/evaluation_metadata.json" if evaluation else None
        config_path = run / "experiment_config.json" if run else None
        if metadata and metadata.is_file():
            config, source = read_json(metadata)["experiment_config"], str(metadata)
        elif config_path and config_path.is_file():
            config, source = read_json(config_path), str(config_path)
        else:
            config, source = job["config"], "queue manifest job.config"
        record = {"id": row["job_id"], "model": row["model"], "method": row["method"], "loss": row["loss"],
                  "seed": config["training_seed"], "config": config, "config_source": source,
                  "status": row["evaluation_status"], "gpu": row["scheduled_gpu"], "run": row["run"],
                  "metadata_path": str(metadata) if metadata and metadata.is_file() else None,
                  "evaluation_dir": str(evaluation) if evaluation else None,
                  "evaluation_language_scopes": row["evaluation_scopes"].split(","),
                  "evaluation_batch_size": manifest.get("wmt23_batch_size", 1),
                  "max_new_tokens": manifest["max_new_tokens"], "eos_policy": manifest.get("wmt_eos_policy"), "scores": {}}
        for scope, metrics in detail_by_id[row["job_id"]]["test"].items():
            record["scores"].update(mt_score_rows(evaluation / "test" / scope, scope, metrics["bleu"], metrics.get("comet22")))
        experiments.append(record)
    write_mt(output / "test", experiments, manifest["test_counts"], updated)

    alignment_rows = []
    for row in records:
        for scope in ("in", "out"):
            values = detail_by_id[row["job_id"]]["validation"].get(f"align_{scope}", {})
            for pair, value in values.items():
                alignment_rows.append({**{k: row[k] for k in ("job_id", "model", "method", "loss", "training_status", "step", "run")},
                                       "scope": scope, "language_pair": pair, "alignment_validation_loss": value})
    write_csv(output / "alignment_validation.csv", alignment_rows,
              ["job_id", "model", "method", "loss", "training_status", "step", "run", "scope", "language_pair", "alignment_validation_loss"])
    lines = ["# 정렬 validation loss — In/Out", "", f"집계 시각: {updated}", "",
             "[번역 test 결과](summary.md) · [MT validation CE](validation.md) · [언어쌍별 CSV](alignment_validation.csv)", "",
             "최종 checkpoint의 마지막 step에서 저장된 OPUS 정렬 목적함수 값이다. MT BLEU/COMET과 별개의 학습 진단이다. "
             "서로 다른 정렬 loss의 절댓값을 직접 순위 비교하지 않는다. T에서도 평가용 loss는 계산될 수 있으나 정렬 학습은 수행하지 않았다. "
             "—는 해당 마지막 step의 기록이 없음이며 0이 아니다. 언어쌍 표기는 원본 로그 키 그대로 보존한다.", ""]
    for scope in ("in", "out"):
        pairs = sorted({r["language_pair"] for r in alignment_rows if r["scope"] == scope})
        rows = []
        for row in records:
            values = detail_by_id[row["job_id"]]["validation"].get(f"align_{scope}", {})
            scores = [values.get(p) for p in pairs]
            rows.append([row["model"].split("/")[-1], row["method"], row["loss"], row["training_status"],
                         *["—" if v is None else f"{v:.4f}" for v in scores],
                         "—" if not pairs or any(v is None for v in scores) else f"{mean(scores):.4f}"])
        lines.extend([f"## {scope.upper()}", "", table(["모델", "방법", "정렬 loss", "학습 상태", *pairs, "언어쌍 macro"], rows), ""])
    write_text(output / "alignment_validation.md", "\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--state_dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = read_json(args.manifest)
    states = {}
    for directory in [*args.state_dirs, *map(Path, manifest.get("additional_state_dirs", []))]:
        states.update(read_json(directory / "state.json").get("jobs", {}))
    records, details = collect(manifest, states)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    updated = datetime.now().astimezone().isoformat()
    write_separated_reports(args.output_dir, manifest, records, details, updated)
    (args.output_dir / "summary.md").write_text(render(records, updated), encoding="utf-8")
    (args.output_dir / "validation.md").write_text(render_validation(records, updated), encoding="utf-8")
    test_fields = ["job_id", "model", "method", "loss", "evaluation_status", "evaluation_scopes", "test_num_examples", "scheduled_gpu", "run",
                   "bleu_in", "bleu_out", "comet22_in_x100", "comet22_out_x100"]
    for name, fields, rows in (
        ("summary.csv", test_fields, [row for row in records if row["priority"]]),
        ("validation.csv", ["job_id", "model", "method", "loss", "priority", "training_status", "step", "target_steps", "run",
                            "mt_validation_macro", "mt_validation_de", "mt_validation_cs", "mt_validation_ja"], records),
    ):
        with (args.output_dir / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    (args.output_dir / "details.json").write_text(json.dumps({
        "updated_at": updated, "manifest": str(args.manifest.resolve()),
        "records": records, "details": details,
    }, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved {len(records)} experiments to {args.output_dir}")


if __name__ == "__main__":
    main()
