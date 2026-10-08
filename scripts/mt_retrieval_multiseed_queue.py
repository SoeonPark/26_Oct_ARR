"""MT retrieval -> Distance Gap Alt + immediate evaluation -> MASSIVE seeds.

Use saved final adapters and separate retrieval/translation output directories.
Loss selection follows a verified comparison and the manifest's selection policy.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.downstream_report_tables import table, write_csv, write_text
from scripts.wmt23_pipeline import (
    Pipeline, comet_complete, completed_run, config_matches, count_jsonl,
    evaluation_counts, evaluation_dependency_complete, file_sha256, find_completed,
    generation_complete, read_json,
)

LOSSES = ("infonce", "centered_infonce", "gap_distance_infonce")
RETRIEVAL_KEYS = ("recall_at_1", "recall_at_5", "mrr")
MASSIVE_EVAL_OPTIONS = (
    "alignment_batch_size", "massive_batch_size", "retrieval_chunk_size",
    "max_new_tokens", "eval_sample_log_limit", "save_alignment_embeddings",
    "save_alignment_sample_metrics",
)


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def retrieval_complete(output, run, job, settings):
    metadata_path = Path(output) / "test/evaluation_metadata.json"
    if not metadata_path.is_file():
        return False
    meta = read_json(metadata_path)
    if not (meta.get("status") == "completed" and meta.get("tasks") == ["alignment"]
            and meta.get("split") == "test" and meta.get("language_scopes") == ["in"]
            and Path(meta.get("checkpoint_path", "")).resolve() == Path(run).resolve()
            and meta.get("alignment_batch_size") == settings["batch_size"]
            and meta.get("retrieval_chunk_size") == settings["chunk_size"]
            and config_matches(job["config"], meta.get("experiment_config", {}))):
        return False
    path = Path(output) / "test/in/alignment_metrics.json"
    if not path.is_file():
        return False
    metrics = read_json(path)
    if (metrics != meta.get("results", {}).get("in", {}).get("alignment")
            or metrics.get("similarity") != "cosine"
            or metrics.get("candidate_pool") != "full language-pair evaluation split"
            or metrics.get("tie_break") != "candidate_index_ascending"
            or set(metrics.get("pairs", {})) != set(settings["pair_counts"])):
        return False
    for pair, count in settings["pair_counts"].items():
        row = metrics["pairs"][pair]
        data = metrics.get("dataset_metadata", {}).get(pair, {})
        if (row.get("num_parallel_pairs") != count or data.get("split") != "test"
                or data.get("file_sha256") != settings["pair_file_sha256"][pair]):
            return False
        src, tgt = pair.split("-")
        for key, source, target in (("source_to_target", src, tgt), ("target_to_source", tgt, src)):
            direction = row.get(key, {})
            if (direction.get("num_queries") != count or direction.get("query_language") != source
                    or direction.get("candidate_language") != target):
                return False
            if any(not isinstance(direction.get(k), (int, float)) or not math.isfinite(direction[k])
                   or not 0 <= direction[k] <= 1 for k in RETRIEVAL_KEYS):
                return False
        for key in RETRIEVAL_KEYS:
            if not math.isclose(row["bidirectional_average"][key],
                                (row["source_to_target"][key] + row["target_to_source"][key]) / 2, abs_tol=1e-10):
                return False
    for key in RETRIEVAL_KEYS:
        macro = sum(row["bidirectional_average"][key] for row in metrics["pairs"].values()) / len(metrics["pairs"])
        if not math.isclose(metrics["language_pair_macro"][key], macro, abs_tol=1e-10):
            return False
    return True


def massive_evaluation_complete(output, run, job, settings):
    """Require final artifacts for the requested retrieval and downstream scopes."""
    try:
        root = Path(output) / "test"
        meta = read_json(root / "evaluation_metadata.json")
        alignment_scopes = settings.get("alignment_language_scopes", settings["language_scopes"])
        if not (meta.get("status") == "completed" and meta.get("split") == "test"
                and meta.get("tasks") == ["alignment", "massive"]
                and meta.get("language_scopes") == settings["language_scopes"]
                and set(alignment_scopes) <= set(meta.get("alignment_language_scopes", meta.get("language_scopes", [])))
                and Path(meta.get("checkpoint_path", "")).resolve() == Path(run).resolve()
                and meta.get("experiment_config", {}).get("checkpoint_global_step") == job["steps"]
                and config_matches(job["config"], meta.get("experiment_config", {}))
                and all(meta.get(key) == settings[key] for key in MASSIVE_EVAL_OPTIONS)):
            return False
        for scope in settings["language_scopes"]:
            folder = root / scope
            massive = read_json(folder / "massive_metrics.json")
            results = meta.get("results", {}).get(scope, {})
            if massive != results.get("massive"):
                return False
            if scope in alignment_scopes:
                retrieval = read_json(folder / "alignment_metrics.json")
                if (retrieval != results.get("alignment") or retrieval.get("similarity") != "cosine"
                        or retrieval.get("candidate_pool") != "full language-pair evaluation split"
                        or retrieval.get("tie_break") != "candidate_index_ascending"
                        or {pair: row["num_parallel_pairs"] for pair, row in retrieval["pairs"].items()}
                        != settings["pair_counts"][scope]):
                    return False
                for pair, count in settings["pair_counts"][scope].items():
                    row = retrieval["pairs"][pair]
                    for direction in ("source_to_target", "target_to_source"):
                        if row[direction]["num_queries"] != count:
                            return False
                        if any(not math.isfinite(row[direction][key]) or not 0 <= row[direction][key] <= 1
                               for key in RETRIEVAL_KEYS):
                            return False
                    for key in RETRIEVAL_KEYS:
                        if not math.isclose(row["bidirectional_average"][key],
                                            (row["source_to_target"][key] + row["target_to_source"][key]) / 2,
                                            abs_tol=1e-10):
                            return False
                for key in RETRIEVAL_KEYS:
                    macro = sum(row["bidirectional_average"][key] for row in retrieval["pairs"].values()) / len(retrieval["pairs"])
                    if not math.isclose(retrieval["language_pair_macro"][key], macro, abs_tol=1e-10):
                        return False
            counts = settings["language_counts"][scope]
            if ({lang: row["num_examples"] for lang, row in massive["per_language"].items()} != counts
                    or massive.get("num_examples") != sum(counts.values())
                    or count_jsonl(folder / "massive_predictions.jsonl") != sum(counts.values())):
                return False
            for row in massive["per_language"].values():
                if any(not math.isfinite(row[key]) or not 0 <= row[key] <= 1
                       for key in ("slot_precision", "slot_recall", "slot_f1", "exact_match")):
                    return False
            required = []
            if settings["eval_sample_log_limit"]:
                required += ["eval_samples.json", "eval_samples_embeddings.pkl"]
            if scope in alignment_scopes and settings["save_alignment_embeddings"]:
                required += ["alignment_embeddings.pt"]
            if scope in alignment_scopes and settings["save_alignment_sample_metrics"]:
                required += ["alignment_samples.jsonl", "alignment_batches.jsonl"]
            if any(not (folder / name).is_file() or (folder / name).stat().st_size == 0 for name in required):
                return False
        return True
    except (OSError, ValueError, TypeError, KeyError, AttributeError, ZeroDivisionError):
        return False


def reviewed_loss(decision, comparison_sha256):
    if decision.get("status") != "reviewed":
        return None
    if decision.get("comparison_sha256") != comparison_sha256:
        raise ValueError("Selection must refer to the completed comparison JSON hash.")
    if decision.get("selected_loss") not in LOSSES or not decision.get("reason", "").strip():
        raise ValueError("A reviewed selection needs an allowed loss and a reason.")
    assessment = decision.get("gap_assessment")
    if assessment not in ("best", "similar", "worse"):
        raise ValueError("gap_assessment must be best, similar, or worse.")
    # Explicit user preference: even a tie selects Distance Gap.
    return "gap_distance_infonce" if assessment in ("best", "similar") else decision["selected_loss"]


def automatic_loss(rows, tolerance, model=None):
    if not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("The similarity tolerance must be finite and nonnegative.")
    by_loss = {}
    for loss in LOSSES:
        candidates = [r for r in rows if r["method"] == "alternative" and r["loss"] == loss]
        expected_models = {"meta-llama/Llama-3.2-1B-Instruct", "Qwen/Qwen3.5-2B"}
        if len(candidates) != 2 or {r.get("model") for r in candidates} != expected_models:
            raise ValueError("Automatic selection requires both distinct model scores for every candidate loss.")
        if any(r.get("comet22_x100") is None or not math.isfinite(r["comet22_x100"]) for r in candidates):
            raise ValueError("Automatic selection requires finite COMET scores for both models.")
        values = [r["comet22_x100"] for r in candidates if model is None or r["model"] == model]
        if not values:
            raise ValueError(f"Unknown selection model: {model}")
        by_loss[loss] = math.fsum(values) / len(values)
    best = max(by_loss, key=by_loss.get)
    selected = "gap_distance_infonce" if by_loss[best] - by_loss["gap_distance_infonce"] <= tolerance else best
    return selected, by_loss


class ExperimentQueue(Pipeline):
    def parallel(self, action):
        def guarded(gpu):
            try:
                return action(gpu)
            except Exception:
                self.failed.set()
                raise
        with ThreadPoolExecutor(max_workers=2) as workers:
            futures = [workers.submit(guarded, gpu) for gpu in (0, 1)]
            for future in as_completed(futures):
                future.result()

    def release_gpus(self):
        for handle in self.gpu_locks:
            handle.close()
        self.gpu_locks.clear()
        self.reserved_gpus.clear()

    def evaluation_jobs(self, gpu):
        yield from (j for j in self.manifest["jobs"] if j["gpu"] == gpu)

    def verified_translation_folder(self, job, run, evaluation_manifest):
        output = Path(run) / "evaluations" / evaluation_manifest["evaluation_id"]
        counts = evaluation_counts(evaluation_manifest, job)
        if not generation_complete(output, run, counts, evaluation_manifest["max_new_tokens"],
                                   evaluation_manifest["wmt23_batch_size"], evaluation_manifest["wmt_eos_policy"]):
            raise RuntimeError(f"Unverified MT at comparison: {job['id']}")
        folder = output / "test/in"
        if not comet_complete(folder, counts["in"]):
            raise RuntimeError(f"Unverified COMET at comparison: {job['id']}")
        return folder

    def validate_existing_checkpoints(self):
        """Retrieval needs adapters; loss selection only needs Alt translation baselines."""
        for job in self.manifest["retrieval_jobs"]:
            run = self.state["jobs"].get(job["id"], {}).get("run") or job.get("run")
            if not run or not completed_run(run, job):
                raise RuntimeError(f"Unverified retrieval checkpoint: {job['id']}")
            if job["mode"] == "alternative":
                self.verified_translation_folder(job, run, self.manifest["previous_evaluation"])

    def retrieve_job(self, gpu, job):
        if self.failed.is_set():
            raise RuntimeError("Another worker failed; no new retrieval will start.")
        run = self.state["jobs"].get(job["id"], {}).get("run") or job.get("run")
        if not run or not completed_run(run, job):
            raise RuntimeError(f"Retrieval needs a verified final adapter: {job['id']}")
        output = Path(run) / "evaluations" / self.manifest["retrieval"]["evaluation_id"]
        settings = self.manifest["retrieval"]
        log = self.state_dir / f"{job['id']}.retrieval.log"
        self.update(job=job, run=str(run), retrieval_status="running", retrieval_dir=str(output), retrieval_log=str(log), retrieval_gpu=gpu)
        if not retrieval_complete(output, run, job, settings):
            command = [self.manifest["python"], "-u", "evaluate.py", "--checkpoint_path", str(run),
                       "--split", "test", "--language_scope", "in", "--tasks", "alignment",
                       "--alignment_batch_size", str(settings["batch_size"]), "--retrieval_chunk_size", str(settings["chunk_size"]),
                       "--output_dir", str(output), "--eval_sample_log_limit", "64",
                       "--no-save_alignment_embeddings", "--save_alignment_sample_metrics"]
            with self.lock(ROOT / "logs" / f"train-gpu-{gpu}.lock") as handle:
                self.execute(command, self.environment(gpu), log, handle)
        if not retrieval_complete(output, run, job, settings):
            raise RuntimeError(f"Incomplete retrieval artifacts: {job['id']}")
        self.update(job=job, retrieval_status="completed")
        self.write_retrieval_report()

    def existing_retrieval(self, gpu):
        self.reserve_gpu(gpu)
        self.update(gpu=gpu, phase="mt_retrieval", status="running")
        for job in self.manifest["retrieval_jobs"]:
            if job["gpu"] == gpu:
                if self.failed.is_set():
                    raise RuntimeError("Another worker failed.")
                self.retrieve_job(gpu, job)
        self.update(gpu=gpu, status="retrieval_complete")

    def gap_train_and_evaluate(self, gpu):
        self.reserve_gpu(gpu)
        job = next(j for j in self.manifest["jobs"] if j["gpu"] == gpu)
        self.update(gpu=gpu, phase="gap_alt_training", status="running")
        self.train_job(gpu, job)
        # Each GPU evaluates its own model immediately, without a training barrier.
        self.retrieve_job(gpu, job)
        self.evaluate(gpu)

    def write_retrieval_report(self):
        with self.summary_lock:
            rows, macros = [], []
            mt_followups = [job for job in self.manifest.get("massive_followup_jobs", [])
                            if job["config"].get("downstream_task") == "wmt23"]
            for job in [*self.manifest["retrieval_jobs"], *self.manifest["jobs"], *mt_followups]:
                saved = self.state["jobs"].get(job["id"], {})
                if saved.get("retrieval_status") != "completed":
                    continue
                path = Path(saved["retrieval_dir"]) / "test/in/alignment_metrics.json"
                metrics = read_json(path)
                base = {"job_id": job["id"], "model": job["model"], "method": job["mode"], "loss": job["loss"],
                        "scope": "in", "run": saved["run"], "metrics_path": str(path), "metrics_sha256": file_sha256(path)}
                macros.append({**base, **{k + "_x100": metrics["language_pair_macro"][k] * 100 for k in RETRIEVAL_KEYS}})
                for pair, result in metrics["pairs"].items():
                    for direction in ("source_to_target", "target_to_source"):
                        score = result[direction]
                        rows.append({**base, "pair": pair, "source": score["query_language"], "target": score["candidate_language"],
                                     "num_queries": score["num_queries"], "num_candidates": result["num_parallel_pairs"],
                                     **{k + "_x100": score[k] * 100 for k in RETRIEVAL_KEYS}})
            if rows:
                folder = Path(self.manifest["report_dir"])
                write_csv(folder / "retrieval_in.csv", rows)
                write_csv(folder / "retrieval_in_macro.csv", macros)
                write_text(folder / "retrieval_in.md", "\n".join([
                    "# MT checkpoint retrieval — In", "", f"집계: {now()}", "",
                    "OPUS-100 test의 de-en/cs-en/en-ja, 언어쌍별 2,000개 후보 전체에서 cosine retrieval. "
                    "마지막 층(-1)의 last token 표현. MASSIVE와 같은 평가 함수, batch 16, chunk 256. "
                    "각 언어쌍 양방향 평균을 다시 3쌍 평균한다. 아래 지표는 ×100이다. 번역 test의 BLEU/COMET과 다른 평가다.", "",
                    "[방향별 CSV](retrieval_in.csv) · [macro CSV](retrieval_in_macro.csv)", "",
                    table(["모델", "방법", "loss", "R@1", "R@5", "MRR"],
                          [[r["model"].split("/")[-1], r["method"], r["loss"], *[f"{r[k + '_x100']:.3f}" for k in RETRIEVAL_KEYS]] for r in macros]),
                ]))

    def compare(self):
        rows = []
        for job in [*self.manifest["retrieval_jobs"], *self.manifest["jobs"]]:
            saved = self.state["jobs"][job["id"]]
            run = saved["run"]
            if not retrieval_complete(saved["retrieval_dir"], run, job, self.manifest["retrieval"]):
                raise RuntimeError(f"Unverified retrieval at comparison: {job['id']}")
            # C/T/C→T retrieval stays in the retrieval report. Deferred C translation
            # must never block the Alt-only comparison or MASSIVE loss selection.
            if job["mode"] != "alternative":
                continue
            old = job in self.manifest["retrieval_jobs"]
            evaluation_manifest = self.manifest["previous_evaluation"] if old else self.manifest
            folder = self.verified_translation_folder(job, run, evaluation_manifest)
            paths = {"bleu": folder / "wmt23_metrics.json", "comet22": folder / "wmt23_comet22_metrics.json",
                     "retrieval": Path(saved["retrieval_dir"]) / "test/in/alignment_metrics.json"}
            metrics = {k: read_json(p) for k, p in paths.items()}
            rows.append({"job_id": job["id"], "model": job["model"], "method": job["mode"], "loss": job["loss"],
                         "seed": job["config"]["training_seed"], "scope": "in",
                         "bleu": metrics["bleu"]["macro_average"]["sacrebleu"],
                         "comet22_x100": metrics["comet22"]["macro_average"]["comet22_x100"],
                         **{k + "_x100": metrics["retrieval"]["language_pair_macro"][k] * 100 for k in RETRIEVAL_KEYS},
                         "sources": {k: {"path": str(p), "sha256": file_sha256(p)} for k, p in paths.items()}})
        # Content-derived review hash remains stable across controller restarts.
        comparison = {"schema_version": 1, "scope": "in", "records": rows,
                      "candidate_losses": list(LOSSES), "similarity_preference": "gap_distance_infonce"}
        folder = Path(self.manifest["report_dir"])
        fixed = self.manifest["selection_policy"]["mode"] == "fixed"
        atomic_json(folder / "comparison.json", comparison)
        write_csv(folder / "comparison.csv", rows)
        write_text(folder / "comparison.md", "\n".join([
            "# MT 평가 후 MASSIVE loss 선택", "", f"집계: {now()}", "",
            "두 모델의 Alt InfoNCE / Centered / Distance Gap을 비교한다. "
            "C/T/C→T retrieval은 별도 상세 보고서에 보존하고, 보류한 C 번역 평가는 loss 선택의 선행 조건에서 제외한다. "
            "BLEU·COMET은 In 6방향 macro, retrieval은 OPUS test 3쌍의 양방향 macro다. "
            + ("MASSIVE loss는 사용자 지정으로 이미 고정했으며, 이 비교로 다시 선택하지 않는다. " if fixed else
               "Distance Gap이 지정한 기준에서 비슷하거나 더 좋으면 Distance Gap을 선택한다. ")
            + f"선택 정책: {json.dumps(self.manifest['selection_policy'], ensure_ascii=False)}", "",
            table(["모델", "방법", "loss", "BLEU", "COMET ×100", "R@1 ×100", "R@5 ×100", "MRR ×100"],
                  [[r["model"].split("/")[-1], r["method"], r["loss"],
                    *[f"{r[k]:.3f}" for k in ("bleu", "comet22_x100", "recall_at_1_x100", "recall_at_5_x100", "mrr_x100")]] for r in rows]), "",
            "[원본 출처 JSON](comparison.json) · [CSV](comparison.csv) · [retrieval 상세](retrieval_in.md)", "",
            ("Llama MASSIVE Alt seed 43은 GPU 0에서 먼저 시작하고, seed 44는 GPU 0·1 중 먼저 준비된 곳에 배정한다. " if fixed else
             "선택 후 Llama MASSIVE Alt seed 43·44를 각각 GPU 0·1에서 학습한다. ")
            + "총 100,000 step, 마지막 층 -1, BF16/NF4, train batch 16, seed 외 같은 설정. "
              "데이터 샘플링 seed는 42로 유지한다. 기존 seed 42 결과와 연결해 비교한다.",
        ]))
        return rows, file_sha256(folder / "comparison.json")

    def wait_for_selection(self, rows, comparison_hash):
        decision_path = self.state_dir / "selection.json"
        if self.manifest["selection_policy"]["mode"] == "automatic_comet":
            policy = self.manifest["selection_policy"]
            selected, means = automatic_loss(rows, policy["tolerance_x100"], policy.get("selection_model"))
            atomic_json(decision_path, {"status": "automatic", "selected_loss": selected,
                        "comparison_sha256": comparison_hash, "selection_comet22_x100": means,
                        "selection_model": policy.get("selection_model", "two_model_mean"),
                        "tolerance_x100": policy["tolerance_x100"],
                        "reason": "Prefer Distance Gap within the configured absolute COMET point margin of the best candidate; otherwise select the highest COMET candidate.",
                        "decided_at": now()})
            write_text(Path(self.manifest["report_dir"]) / "selection.md", "\n".join([
                "# MASSIVE 멀티시드 loss 선택", "", f"선택 시각: {now()}", "",
                f"선택: **{selected}**. 대상은 Llama MASSIVE alternative seed 43·44.", "",
                f"비교 기준: {policy.get('selection_model') or '두 모델 평균'}, In COMET-22 ×100. "
                f"최고점과의 차이가 {policy['tolerance_x100']}점 이내이면 Distance Gap을 우선한다.", "",
                table(["loss", "선택에 사용한 COMET-22 ×100"], [[loss, f"{value:.4f}"] for loss, value in means.items()]), "",
                "[두 모델 BLEU·COMET·retrieval 비교](comparison.md)", "",
                f"[결정 기록]({decision_path}) · 비교 JSON SHA256: `{comparison_hash}`",
            ]))
            return selected
        self.update(status="waiting_for_result_review", comparison_sha256=comparison_hash,
                    selection_file=str(decision_path), review_report=str(Path(self.manifest["report_dir"]) / "comparison.md"))
        while True:
            if decision_path.is_file():
                selected = reviewed_loss(read_json(decision_path), comparison_hash)
                if selected:
                    return selected
            time.sleep(30)

    def massive_evaluation_settings(self):
        settings = self.manifest.get("massive_post_training_evaluation")
        return settings if isinstance(settings, dict) and settings.get("enabled") else None

    def evaluate_massive_job(self, gpu, job):
        settings = self.massive_evaluation_settings()
        if settings is None:
            return
        try:
            if self.failed.is_set():
                raise RuntimeError("Another worker failed; no new MASSIVE evaluation will start.")
            run = self.state["jobs"].get(job["id"], {}).get("run")
            if not run or not completed_run(run, job):
                raise RuntimeError(f"MASSIVE evaluation needs a verified final adapter: {job['id']}")
            output = Path(run) / "evaluations" / settings["evaluation_id"]
            log = self.state_dir / f"{job['id']}.evaluation.log"
            self.update(gpu=gpu, phase="massive_final_evaluation", status="running", job_id=job["id"])
            self.update(job=job, evaluation_status="running", retrieval_status="running",
                        evaluation_dir=str(output), retrieval_dir=str(output), evaluation_log=str(log),
                        evaluation_gpu=gpu, evaluation_language_scopes=settings["language_scopes"],
                        retrieval_language_scopes=settings.get("alignment_language_scopes", settings["language_scopes"]),
                        evaluation_tasks=["alignment", "massive"], evaluation_error=None)
            reused = massive_evaluation_complete(output, run, job, settings)
            if not reused:
                scope = "both" if settings["language_scopes"] == ["in", "out"] else settings["language_scopes"][0]
                command = [self.manifest["python"], "-u", "evaluate.py", "--checkpoint_path", str(run),
                           "--split", "test", "--language_scope", scope, "--tasks", "alignment", "massive",
                           "--output_dir", str(output)]
                if "alignment_language_scopes" in settings:
                    alignment_scopes = settings["alignment_language_scopes"]
                    command.extend(["--alignment_language_scope", "both" if alignment_scopes == ["in", "out"] else alignment_scopes[0]])
                for key in MASSIVE_EVAL_OPTIONS:
                    value = settings[key]
                    if isinstance(value, bool):
                        command.append(f"--{'' if value else 'no-'}{key}")
                    else:
                        command.extend([f"--{key}", str(value)])
                with self.lock(ROOT / "logs" / f"train-gpu-{gpu}.lock") as handle:
                    self.execute(command, self.environment(gpu), log, handle)
            if not massive_evaluation_complete(output, run, job, settings):
                raise RuntimeError(f"Incomplete MASSIVE retrieval/slot-filling artifacts: {job['id']}")
            self.update(job=job, evaluation_status="completed", retrieval_status="completed",
                        evaluation_completed_at=now(), evaluation_reused=reused)
        except Exception as error:
            self.update(job=job, evaluation_status="failed", retrieval_status="failed", evaluation_error=str(error))
            raise

    def massive_train(self, gpu, selected):
        job = next(j for j in self.manifest["massive_candidates"][selected] if j["gpu"] == gpu)
        self.reserve_gpu(gpu)
        self.update(job=job, training_status="queued", selected_loss=selected)
        self.update(gpu=gpu, phase="massive_multiseed_training", status="running")
        self.train_job(gpu, job)
        self.evaluate_massive_job(gpu, job)
        self.update(gpu=gpu, status="completed")

    def fixed_massive_jobs(self):
        selected = self.manifest["selection_policy"]["selected_loss"]
        if selected not in LOSSES:
            raise ValueError("Unknown fixed MASSIVE loss.")
        jobs = sorted(self.manifest["massive_candidates"][selected], key=lambda j: j["config"]["training_seed"])
        if ([j["config"]["training_seed"] for j in jobs] != [43, 44]
                or len({j["id"] for j in jobs}) != 2
                or any(j["model"] != "meta-llama/Llama-3.2-1B-Instruct"
                       or j["mode"] != "alternative" or j["loss"] != selected
                       or j["config"]["downstream_task"] != "massive" for j in jobs)):
            raise ValueError("Fixed scheduling requires Llama MASSIVE Alt seeds 43 and 44.")
        return jobs

    def claim_fixed_massive(self, gpu):
        """Seed 43 belongs to GPU 0; claim seed 44 once on the first ready GPU."""
        with self.mutex:
            jobs = self.fixed_massive_jobs()
            first = self.state["jobs"].get(jobs[0]["id"], {})
            pending = False
            for job in jobs:
                saved = self.state["jobs"].get(job["id"], {})
                status = saved.get("training_status", "queued")
                if status == "failed":
                    raise RuntimeError(f"MASSIVE training previously failed: {job['id']}")
                if status in ("claimed", "running", "completed"):
                    continue
                pending = True
                seed = job["config"]["training_seed"]
                if seed == 43 and gpu != 0:
                    continue
                if seed == 44 and first.get("training_status") not in ("running", "completed"):
                    continue
                assigned = {**job, "gpu": gpu}
                self.update(job=assigned, training_status="claimed", assigned_gpu=gpu, seed=seed,
                            claimed_at=now(), selected_loss=job["loss"])
                return assigned, True
            return None, pending

    def claim_massive_followup(self, gpu):
        """Claim once across workers, preserving manifest priority and dependencies."""
        with self.mutex:
            pending = False
            for job in self.manifest.get("massive_followup_jobs", []):
                saved = self.state["jobs"].get(job["id"], {})
                training = saved.get("training_status", "queued")
                evaluation = saved.get("evaluation_status", "queued")
                if training == "failed" or evaluation == "failed":
                    raise RuntimeError(f"MASSIVE follow-up previously failed: {job['id']}")
                if training == "completed" and evaluation == "completed":
                    continue
                eligible = job.get("eligible_gpus", [job.get("gpu")])
                # Once claimed, retain this GPU through final evaluation and restarts.
                if training in ("claimed", "running", "completed") and saved.get("assigned_gpu") is not None:
                    eligible = [saved["assigned_gpu"]]
                if gpu not in eligible:
                    continue
                pending = True
                if training in ("claimed", "running") or evaluation in ("claimed", "running"):
                    continue
                if any(self.state["jobs"].get(job_id, {}).get("evaluation_status") != "completed"
                       for job_id in job.get("after_jobs", [])):
                    continue
                assigned = {**job, "gpu": gpu}
                self.update(job=assigned, assigned_gpu=gpu, claimed_at=now(),
                            training_status="completed" if training == "completed" else "claimed")
                return assigned, True
            return None, pending

    def run_massive_followup(self, gpu, job, *, recovered=False):
        if self.failed.is_set():
            raise RuntimeError("Another worker failed; no MASSIVE follow-up will start.")
        saved = self.state["jobs"].get(job["id"], {})
        if saved.get("training_status") == "failed":
            raise RuntimeError(f"MASSIVE follow-up previously failed: {job['id']}")
        self.update(gpu=gpu, phase="massive_followup_training", status="running", job_id=job["id"])
        self.update(job=job, assigned_gpu=gpu)
        try:
            # reserve_gpu already waited for the previous child on restart.
            if recovered and saved.get("training_status") == "running":
                run = find_completed(job.get("results_root", self.manifest["results_root"]), job)
                if run is None:
                    raise RuntimeError(f"Previous MASSIVE follow-up ended without a verified adapter: {job['id']}")
                self.update(job=job, training_status="completed", run=str(run), reused=True)
            else:
                self.train_job(gpu, job)
        except Exception as error:
            self.update(job=job, training_status="failed", error=str(error))
            raise
        if job["config"].get("downstream_task", "massive") == "wmt23":
            self.retrieve_job(gpu, job)
            self.evaluate(gpu, jobs=[job])
        else:
            self.evaluate_massive_job(gpu, job)

    def run_massive_followups(self, gpu, recovered_jobs=()):
        for job in recovered_jobs:
            self.run_massive_followup(gpu, job, recovered=True)
        while not self.failed.is_set():
            job, pending = self.claim_massive_followup(gpu)
            if job is not None:
                self.run_massive_followup(gpu, job)
            elif pending:
                self.update(gpu=gpu, phase="massive_followup_waiting", status="waiting_for_dependencies")
                time.sleep(5)
            else:
                return
        raise RuntimeError("Another worker failed; no further MASSIVE follow-up will start.")

    def fixed_gpu_worker(self, gpu, recovered_jobs):
        # reserve_gpu waits for the previous live child without terminating it.
        self.reserve_gpu(gpu)
        for job in recovered_jobs:
            saved = self.state["jobs"][job["id"]]
            run = find_completed(job.get("results_root", self.manifest["results_root"]), job)
            if run:
                self.update(job=job, training_status="completed", run=str(run), reused=True)
                self.evaluate_massive_job(gpu, job)
            elif saved["training_status"] == "claimed":
                self.update(job=job, training_status="queued")
            else:
                raise RuntimeError(f"Previous MASSIVE child ended without a verified adapter: {job['id']}")
        # Keep each model's MT retrieval, translation and COMET together.
        # Completed Llama artifacts are verified/reused before seed 43 starts.
        self.gap_train_and_evaluate(gpu)
        with self.mutex:
            if (not self.state.get("fixed_policy_comparison_sha256")
                    and all(self.state["jobs"].get(j["id"], {}).get("evaluation_status") == "completed"
                            for j in self.manifest["jobs"])):
                _, digest = self.compare()
                self.update(fixed_policy_comparison_sha256=digest)
        while not self.failed.is_set():
            job, pending = self.claim_fixed_massive(gpu)
            if job is None:
                if not pending:
                    self.run_massive_followups(gpu, getattr(self, "recovered_followups", {}).get(gpu, []))
                    self.update(gpu=gpu, status="assigned_work_completed")
                    return
                time.sleep(5)
                continue
            self.update(gpu=gpu, phase="massive_multiseed_training", status="running", job_id=job["id"])
            with self.mutex:
                self.update(massive_multiseed={**self.state.get("massive_multiseed", {}), "status": "running"})
            try:
                self.train_job(gpu, job)
            except Exception as error:
                self.update(job=job, training_status="failed", error=str(error))
                raise
            # Finish this seed's evaluation before this GPU can claim another seed.
            self.evaluate_massive_job(gpu, job)
        raise RuntimeError("Another worker failed; no further MASSIVE job will start.")

    def run_fixed_massive(self):
        jobs = self.fixed_massive_jobs()
        self.validate_existing_checkpoints()
        for job in self.manifest["retrieval_jobs"]:
            saved = self.state["jobs"].get(job["id"], {})
            if not saved.get("retrieval_dir") or not retrieval_complete(
                    saved["retrieval_dir"], saved["run"], job, self.manifest["retrieval"]):
                raise RuntimeError("Immediate MASSIVE scheduling requires completed baseline retrieval.")
        selected = self.manifest["selection_policy"]["selected_loss"]
        evaluation = self.massive_evaluation_settings()
        followups = self.manifest.get("massive_followup_jobs", [])
        all_jobs = [*jobs, *followups]
        if len({job["id"] for job in all_jobs}) != len(all_jobs):
            raise ValueError("MASSIVE follow-ups need unique job IDs.")
        preceding = {job["id"] for job in jobs}
        for job in followups:
            eligible = job.get("eligible_gpus", [job.get("gpu")])
            if (not eligible or any(gpu not in (0, 1) for gpu in eligible)
                    or job["config"]["downstream_task"] not in ("massive", "wmt23")
                    or not set(job.get("after_jobs", [])) <= preceding):
                raise ValueError("Follow-ups need eligible GPUs, MASSIVE/MT config and dependencies on preceding jobs.")
            preceding.add(job["id"])
        if followups and evaluation is None:
            raise ValueError("MASSIVE follow-ups require immediate final evaluation.")
        recovered = {gpu: [] for gpu in (0, 1)}
        self.recovered_followups = {gpu: [] for gpu in (0, 1)}
        for job in jobs:
            saved = self.state["jobs"].get(job["id"], {})
            if (saved.get("training_status") in ("claimed", "running")
                    or (evaluation and saved.get("training_status") == "completed")):
                gpu = saved.get("assigned_gpu", saved.get("gpu", job["gpu"]))
                recovered[gpu].append({**job, "gpu": gpu})
            elif not saved:
                self.update(job=job, model=job["model"], mode=job["mode"], loss=selected,
                            seed=job["config"]["training_seed"], training_status="queued")
            if evaluation:
                self.update(job=job, evaluation_status=saved.get("evaluation_status", "queued"),
                            retrieval_status=saved.get("retrieval_status", "queued"),
                            evaluation_language_scopes=evaluation["language_scopes"],
                            retrieval_language_scopes=evaluation.get("alignment_language_scopes", evaluation["language_scopes"]),
                            evaluation_tasks=["alignment", "massive"])
        for job in followups:
            saved = self.state["jobs"].get(job["id"], {})
            assigned_gpu = saved.get("assigned_gpu", job.get("gpu"))
            mt = job["config"]["downstream_task"] == "wmt23"
            evaluation_scopes = list(evaluation_counts(self.manifest, job)) if mt else evaluation["language_scopes"]
            retrieval_scopes = ["in"] if mt else evaluation.get("alignment_language_scopes", evaluation["language_scopes"])
            if saved.get("training_status") in ("claimed", "running", "completed"):
                if assigned_gpu not in job.get("eligible_gpus", [job.get("gpu")]):
                    raise ValueError(f"Invalid recovery GPU for {job['id']}")
                self.recovered_followups[assigned_gpu].append({**job, "gpu": assigned_gpu})
            self.update(job=job, model=job["model"], mode=job["mode"], loss=job["loss"],
                        seed=job["config"]["training_seed"], assigned_gpu=assigned_gpu,
                        eligible_gpus=job.get("eligible_gpus", [job.get("gpu")]), after_jobs=job.get("after_jobs", []),
                        training_status=saved.get("training_status", "queued"),
                        evaluation_status=saved.get("evaluation_status", "queued"),
                        retrieval_status=saved.get("retrieval_status", "queued"),
                        evaluation_language_scopes=evaluation_scopes,
                        retrieval_language_scopes=retrieval_scopes,
                        evaluation_tasks=["alignment", "wmt23" if mt else "massive"])
        decision = {"status": "user_selected", "selected_loss": selected,
                    "reason": self.manifest["selection_policy"]["user_instruction"],
                    "authorized_at": self.manifest["selection_policy"]["authorized_at"],
                    "wait_for_mt_comparison": False,
                    "scheduling": "seed43_gpu0_then_seed44_first_ready_gpu_after_its_mt_work"}
        atomic_json(self.state_dir / "selection.json", decision)
        write_text(Path(self.manifest["report_dir"]) / "selection.md", "\n".join([
            "# MASSIVE 멀티시드 실행 지정", "", f"선택: **{selected}** (사용자 지정).", "",
            "두 모델의 MT 점수 비교를 기다리지 않는다. Llama MASSIVE alternative seed 43을 GPU 0에서 먼저 시작한다.",
            "Seed 44는 이후 GPU 0·1 중 먼저 준비된 곳에서 실행한다. GPU 1의 진행 중 Qwen MT 학습·평가는 유지한다.", "",
            *([f"각 seed 학습 직후 같은 GPU에서 최종 test 평가: retrieval {evaluation.get('alignment_language_scopes', evaluation['language_scopes'])}, "
               f"slot filling {evaluation['language_scopes']}.", ""]
              if evaluation else []),
            *[f"추가 실행: GPU 후보 {job.get('eligible_gpus', [job.get('gpu')])} / {job['model']} / {job['mode']} / "
              f"layer {job['config']['alignment_hidden_state_layer']} / seed {job['config']['training_seed']}. "
              f"선행 평가 완료 조건: {job.get('after_jobs', [])}. "
              "각 학습 직후 최종 평가까지 완료한 뒤 다음 작업을 시작한다." for job in followups],
            f"지시: {decision['reason']}", f"[결정 기록]({self.state_dir / 'selection.json'})",
        ]))
        massive = {"model": jobs[0]["model"], "mode": "alternative", "seeds": [43, 44],
                   "selected_loss": selected, "status": "scheduled", "job_ids": [j["id"] for j in jobs],
                   "scheduling": decision["scheduling"], "wait_for_mt_comparison": False,
                   "post_training_evaluation": evaluation}
        self.update(status="mt_and_massive_training", selection_policy=self.manifest["selection_policy"],
                    selected_loss=selected, massive_multiseed=massive,
                    massive_followup_job_ids=[job["id"] for job in followups])
        self.parallel(lambda gpu: self.fixed_gpu_worker(gpu, recovered[gpu]))
        if any(self.state["jobs"].get(job["id"], {}).get("training_status") != "completed"
               or (evaluation and self.state["jobs"][job["id"]].get("evaluation_status") != "completed")
               for job in all_jobs):
            raise RuntimeError("MASSIVE queue ended before every seed finished training and its requested evaluation.")
        self.update(status="completed", completed_at=now(), massive_multiseed={**massive, "status": "completed"})

    def run(self):
        self.update(pid=os.getpid(), status="verifying_prerequisites", phase_order=self.manifest["phase_order"])
        try:
            if self.manifest.get("selection_policy", {}).get("mode") == "fixed":
                return self.run_fixed_massive()
            if "wait_for_evaluation" in self.manifest:
                self.update(status="waiting_for_mt_evaluation")
                while not evaluation_dependency_complete(self.manifest["wait_for_evaluation"]):
                    time.sleep(30)
            self.validate_existing_checkpoints()
            self.update(status="mt_retrieval")
            self.parallel(self.existing_retrieval)
            self.update(status="gap_alt_train_and_evaluate")
            self.parallel(self.gap_train_and_evaluate)
            self.release_gpus()
            rows, comparison_hash = self.compare()
            selected = self.wait_for_selection(rows, comparison_hash)
            massive_status = {"model": "meta-llama/Llama-3.2-1B-Instruct", "mode": "alternative", "seeds": [43, 44],
                              "selected_loss": selected, "status": "running",
                              "job_ids": [j["id"] for j in self.manifest["massive_candidates"][selected]]}
            self.update(status="massive_multiseed_training", selected_loss=selected, massive_multiseed=massive_status)
            self.parallel(lambda gpu: self.massive_train(gpu, selected))
            self.update(status="completed", completed_at=now(), massive_multiseed={**massive_status, "status": "completed"})
        except Exception as error:
            self.update(status="failed", error=str(error))
            raise
        finally:
            self.release_gpus()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--state_dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--select-loss", choices=LOSSES)
    parser.add_argument("--gap-assessment", choices=("best", "similar", "worse"))
    parser.add_argument("--reason")
    args = parser.parse_args()
    manifest = read_json(args.manifest)
    if args.select_loss:
        state = read_json(args.state_dir / "state.json")
        if state.get("status") != "waiting_for_result_review":
            parser.error("Selection is accepted only after all comparison results are verified.")
        decision = {"status": "reviewed", "selected_loss": args.select_loss, "gap_assessment": args.gap_assessment,
                    "reason": args.reason or "", "comparison_sha256": state["comparison_sha256"], "decided_at": now()}
        selected = reviewed_loss(decision, state["comparison_sha256"])
        decision["effective_loss"] = selected
        atomic_json(args.state_dir / "selection.json", decision)
        print(f"Queued MASSIVE seed 43/44 with {selected}")
        return
    if args.dry_run:
        print(json.dumps({"phase_order": manifest["phase_order"], "retrieval_jobs": [j["id"] for j in manifest["retrieval_jobs"]],
                          "gap_jobs": [j["id"] for j in manifest["jobs"]], "selection_policy": manifest["selection_policy"],
                          "massive_followup_jobs": [j["id"] for j in manifest.get("massive_followup_jobs", [])],
                          "massive_candidates": {k: [j["id"] for j in v] for k, v in manifest["massive_candidates"].items()}}, indent=2))
        return
    if file_sha256(manifest["data_manifest"]) != manifest["manifest_sha256"]:
        raise ValueError("Frozen WMT data manifest changed.")
    args.state_dir.mkdir(parents=True, exist_ok=True)
    with (args.state_dir / "controller.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ExperimentQueue(manifest, args.state_dir).run()


if __name__ == "__main__":
    main()
