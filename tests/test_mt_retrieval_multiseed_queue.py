import json
from copy import deepcopy
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from scripts.mt_retrieval_multiseed_queue import (
    ExperimentQueue, LOSSES, automatic_loss, massive_evaluation_complete, retrieval_complete, reviewed_loss,
)
from scripts.wmt23_pipeline import Pipeline
from config import parse_args as parse_training_args


class QueueTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def retrieval_fixture(self):
        run, output = self.root / "run", self.root / "retrieval"
        job = {"config": {"training_seed": 42, "alignment_hidden_state_layer": -1}}
        settings = {"batch_size": 16, "chunk_size": 256, "pair_counts": {"de-en": 2}, "pair_file_sha256": {"de-en": "frozen"}}
        scores = {"recall_at_1": .5, "recall_at_5": 1.0, "mrr": .75}
        metrics = {"similarity": "cosine", "candidate_pool": "full language-pair evaluation split",
                   "tie_break": "candidate_index_ascending", "dataset_metadata": {"de-en": {"split": "test", "file_sha256": "frozen"}},
                   "pairs": {"de-en": {"num_parallel_pairs": 2,
                       "source_to_target": {"query_language": "de", "candidate_language": "en", "num_queries": 2, **scores},
                       "target_to_source": {"query_language": "en", "candidate_language": "de", "num_queries": 2, **scores},
                       "bidirectional_average": scores.copy()}}, "language_pair_macro": scores.copy()}
        meta = {"status": "completed", "tasks": ["alignment"], "split": "test", "language_scopes": ["in"],
                "checkpoint_path": str(run), "alignment_batch_size": 16, "retrieval_chunk_size": 256,
                "experiment_config": job["config"], "results": {"in": {"alignment": metrics}}}
        self.write(output / "test/evaluation_metadata.json", meta)
        self.write(output / "test/in/alignment_metrics.json", metrics)
        return run, output, job, settings, meta, metrics

    def test_retrieval_requires_matching_dataset_checkpoint_and_scope(self):
        run, output, job, settings, meta, metrics = self.retrieval_fixture()
        self.assertTrue(retrieval_complete(output, run, job, settings))
        for field, value in (("language_scopes", ["in", "out"]), ("tasks", ["wmt23"]), ("status", "running"),
                             ("alignment_batch_size", 32), ("checkpoint_path", str(self.root / "other"))):
            self.write(output / "test/evaluation_metadata.json", {**meta, field: value})
            self.assertFalse(retrieval_complete(output, run, job, settings), field)
        self.write(output / "test/evaluation_metadata.json", meta)
        self.assertFalse(retrieval_complete(output, run, job, {**settings, "pair_counts": {"de-en": 3}}))
        self.assertFalse(retrieval_complete(output, run, job, {**settings, "pair_file_sha256": {"de-en": "changed"}}))

    def test_retrieval_rejects_inconsistent_macro_and_partial_artifacts(self):
        run, output, job, settings, meta, metrics = self.retrieval_fixture()
        metrics["language_pair_macro"]["mrr"] = .1
        self.write(output / "test/evaluation_metadata.json", meta)
        self.write(output / "test/in/alignment_metrics.json", metrics)
        self.assertFalse(retrieval_complete(output, run, job, settings))
        (output / "test/in/alignment_metrics.json").unlink()
        self.assertFalse(retrieval_complete(output, run, job, settings))

    def test_review_gate_requires_current_results_and_prefers_gap_for_ties(self):
        self.assertIsNone(reviewed_loss({"status": "pending_review"}, "current"))
        decision = {"status": "reviewed", "selected_loss": "centered_infonce", "reason": "Comparable scores",
                    "gap_assessment": "similar", "comparison_sha256": "current"}
        self.assertEqual(reviewed_loss(decision, "current"), "gap_distance_infonce")
        self.assertEqual(reviewed_loss({**decision, "gap_assessment": "worse"}, "current"), "centered_infonce")
        for changes in ({"comparison_sha256": "stale"}, {"reason": ""}, {"selected_loss": "bad"}, {"gap_assessment": None}):
            with self.assertRaises(ValueError):
                reviewed_loss({**decision, **changes}, "current")

    def test_optional_automatic_policy_needs_both_models_and_gap_tie_margin(self):
        rows = [{"model": model, "method": "alternative", "loss": loss, "comet22_x100": score}
                for loss, score in zip(LOSSES, [73, 74, 73.5])
                for model in ("meta-llama/Llama-3.2-1B-Instruct", "Qwen/Qwen3.5-2B")]
        self.assertEqual(automatic_loss(rows, .5)[0], "gap_distance_infonce")
        self.assertEqual(automatic_loss(rows, .49)[0], "centered_infonce")
        with self.assertRaises(ValueError):
            automatic_loss(rows[:-1], .5)

    def test_llama_selection_keeps_qwen_from_masking_llama_regression(self):
        llama, qwen = "meta-llama/Llama-3.2-1B-Instruct", "Qwen/Qwen3.5-2B"
        rows = [{"model": model, "method": "alternative", "loss": loss, "comet22_x100": score}
                for model, scores in ((llama, [70, 72, 70]), (qwen, [74, 76, 80]))
                for loss, score in zip(LOSSES, scores)]
        self.assertEqual(automatic_loss(rows, .5)[0], "gap_distance_infonce")
        self.assertEqual(automatic_loss(rows, .5, llama)[0], "centered_infonce")
        rows[2]["comet22_x100"] = 71.5
        self.assertEqual(automatic_loss(rows, .5, llama)[0], "gap_distance_infonce")
        rows[-1]["model"] = llama
        with self.assertRaises(ValueError):
            automatic_loss(rows, .5, llama)

    def test_selection_rejects_invalid_tolerance(self):
        for tolerance in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                automatic_loss([], tolerance)

    def test_automatic_selection_records_comparison_before_massive_training(self):
        queue = ExperimentQueue({"selection_policy": {"mode": "automatic_comet", "tolerance_x100": .5},
                                 "report_dir": str(self.root / "report")}, self.root / "queue")
        rows = [{"model": model, "method": "alternative", "loss": loss, "comet22_x100": score}
                for model in ("meta-llama/Llama-3.2-1B-Instruct", "Qwen/Qwen3.5-2B")
                for loss, score in zip(LOSSES, [70, 72, 71.5])]
        self.assertEqual(queue.wait_for_selection(rows, "verified_comparison"), "gap_distance_infonce")
        decision = json.loads((self.root / "queue/selection.json").read_text())
        self.assertEqual(decision["comparison_sha256"], "verified_comparison")
        self.assertEqual(decision["selection_model"], "two_model_mean")
        self.assertEqual(decision["status"], "automatic")
        self.assertTrue((self.root / "report/selection.md").is_file())

    def test_gap_worker_evaluates_immediately_after_its_own_training(self):
        job = {"id": "gap", "gpu": 0}
        queue = ExperimentQueue({"jobs": [job]}, self.root)
        calls = []
        with patch.object(queue, "reserve_gpu"), \
                patch.object(queue, "train_job", side_effect=lambda *args: calls.append("train")), \
                patch.object(queue, "retrieve_job", side_effect=lambda *args: calls.append("retrieve")), \
                patch.object(queue, "evaluate", side_effect=lambda *args: calls.append("translate_and_score")):
            queue.gap_train_and_evaluate(0)
        self.assertEqual(calls, ["train", "retrieve", "translate_and_score"])

    def test_retrieval_uses_separate_directory_and_only_in_alignment(self):
        run = self.root / "run"
        job = {"id": "old", "run": str(run)}
        queue = ExperimentQueue({"python": "python", "retrieval": {"evaluation_id": "retrieval_only", "batch_size": 16, "chunk_size": 256}}, self.root / "queue")
        with patch("scripts.mt_retrieval_multiseed_queue.completed_run", return_value=True), \
                patch("scripts.mt_retrieval_multiseed_queue.retrieval_complete", side_effect=[False, True]), \
                patch.object(queue, "lock", return_value=nullcontext(None)), \
                patch.object(queue, "write_retrieval_report"), patch.object(queue, "execute") as execute:
            queue.retrieve_job(0, job)
        command = execute.call_args.args[0]
        self.assertEqual(command[command.index("--tasks") + 1], "alignment")
        self.assertEqual(command[command.index("--language_scope") + 1], "in")
        self.assertEqual(Path(command[command.index("--output_dir") + 1]), run / "evaluations/retrieval_only")
        self.assertEqual(queue.state["jobs"]["old"]["retrieval_status"], "completed")

    def test_dependency_failure_does_not_launch_gpu_work(self):
        queue = ExperimentQueue({"phase_order": [], "wait_for_evaluation": {}}, self.root)
        with patch("scripts.mt_retrieval_multiseed_queue.evaluation_dependency_complete", side_effect=RuntimeError("failed dependency")), \
                patch.object(queue, "parallel") as parallel:
            with self.assertRaisesRegex(RuntimeError, "dependency"):
                queue.run()
        parallel.assert_not_called()
        self.assertEqual(queue.state["status"], "failed")

    def test_massive_training_finds_results_in_its_own_root(self):
        queue = Pipeline({"results_root": "mt_root"}, self.root)
        job = {"id": "massive", "results_root": "massive_root"}
        with patch("scripts.wmt23_pipeline.find_completed", return_value=Path("massive_root/run")) as find:
            queue.train_job(0, job)
        find.assert_called_once_with("massive_root", job)

    def test_retrieval_prerequisite_does_not_require_contrastive_only_translation(self):
        jobs = [{"id": mode, "mode": mode, "run": str(self.root / mode)}
                for mode in ("alternative", "contrastive_only", "transfer_only", "contrastive_then_transfer")]
        queue = ExperimentQueue({"retrieval_jobs": jobs, "previous_evaluation": {}}, self.root / "queue")
        with patch("scripts.mt_retrieval_multiseed_queue.completed_run", return_value=True) as completed, \
                patch.object(queue, "verified_translation_folder") as translation:
            queue.validate_existing_checkpoints()
        self.assertEqual(completed.call_count, 4)
        translation.assert_called_once_with(jobs[0], jobs[0]["run"], {})
        with patch("scripts.mt_retrieval_multiseed_queue.completed_run", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "checkpoint"):
                queue.validate_existing_checkpoints()

    def test_retrieval_first_runs_without_waiting_for_deferred_evaluation(self):
        queue = ExperimentQueue({"phase_order": [], "massive_candidates": {"gap_distance_infonce": []}}, self.root)
        stages = []
        with patch("scripts.mt_retrieval_multiseed_queue.evaluation_dependency_complete") as dependency, \
                patch.object(queue, "validate_existing_checkpoints"), \
                patch.object(queue, "parallel", side_effect=lambda fn: stages.append(fn.__name__)), \
                patch.object(queue, "compare", return_value=([], "hash")), \
                patch.object(queue, "wait_for_selection", return_value="gap_distance_infonce"):
            queue.run()
        dependency.assert_not_called()
        self.assertEqual(stages[:2], ["existing_retrieval", "gap_train_and_evaluate"])
        self.assertEqual(queue.state["status"], "completed")

    def test_comparison_ignores_deferred_translation_but_keeps_retrieval(self):
        jobs = [{"id": mode, "mode": mode} for mode in ("alternative", "contrastive_only")]
        queue = ExperimentQueue({"retrieval_jobs": jobs, "jobs": [], "retrieval": {}, "previous_evaluation": {}}, self.root)
        queue.state["jobs"] = {job["id"]: {"run": "run", "retrieval_dir": "retrieval"} for job in jobs}
        # Start with C so a regression would require its absent translation first.
        queue.manifest["retrieval_jobs"].reverse()
        with patch("scripts.mt_retrieval_multiseed_queue.retrieval_complete", return_value=True) as retrieval, \
                patch.object(queue, "verified_translation_folder", side_effect=RuntimeError("Alt reached")) as translation:
            with self.assertRaisesRegex(RuntimeError, "Alt reached"):
                queue.compare()
        self.assertEqual(retrieval.call_count, 2)
        self.assertEqual(translation.call_args.args[0]["mode"], "alternative")

    def fixed_queue(self):
        loss = "gap_distance_infonce"
        jobs = [{"id": f"seed{seed}", "gpu": gpu, "model": "meta-llama/Llama-3.2-1B-Instruct",
                 "mode": "alternative", "loss": loss, "results_root": "massive_root",
                 "config": {"training_seed": seed, "downstream_task": "massive"}}
                for gpu, seed in enumerate([43, 44])]
        queue = ExperimentQueue({"selection_policy": {"mode": "fixed", "selected_loss": loss},
                                 "massive_candidates": {loss: jobs}, "phase_order": [],
                                 "results_root": "mt_root", "jobs": [{"id": "mt0"}, {"id": "mt1"}]}, self.root / "fixed")
        return queue, jobs

    def test_fixed_seed43_starts_only_on_gpu0_and_before_seed44(self):
        queue, jobs = self.fixed_queue()
        self.assertEqual(queue.claim_fixed_massive(1), (None, True))
        first, _ = queue.claim_fixed_massive(0)
        self.assertEqual(first["id"], "seed43")
        self.assertEqual(first["gpu"], 0)
        self.assertEqual(queue.claim_fixed_massive(1), (None, True))
        queue.update(job=jobs[0], training_status="running")
        second, _ = queue.claim_fixed_massive(1)
        self.assertEqual((second["id"], second["gpu"]), ("seed44", 1))

    def test_fixed_seed44_uses_gpu0_when_seed43_finishes_first(self):
        queue, jobs = self.fixed_queue()
        queue.update(job=jobs[0], training_status="completed", gpu=0)
        job, _ = queue.claim_fixed_massive(0)
        self.assertEqual((job["id"], job["gpu"]), ("seed44", 0))
        self.assertEqual(queue.state["jobs"]["seed44"]["assigned_gpu"], 0)
        self.assertEqual(queue.claim_fixed_massive(1), (None, False))

    def test_fixed_seed44_cannot_be_claimed_twice(self):
        queue, jobs = self.fixed_queue()
        queue.update(job=jobs[0], training_status="running", gpu=0)
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(queue.claim_fixed_massive, [0, 1]))
        self.assertEqual(sum(job is not None for job, _ in results), 1)

    def test_fixed_worker_finishes_its_mt_evaluation_before_seed44(self):
        queue, jobs = self.fixed_queue()
        queue.update(job=jobs[0], training_status="running", gpu=0)
        calls = []
        def mt(gpu):
            calls.append("mt_evaluation_complete")
        def train(gpu, job):
            calls.append((gpu, job["id"]))
            queue.update(job=job, training_status="completed")
        with patch.object(queue, "reserve_gpu"), patch.object(queue, "gap_train_and_evaluate", side_effect=mt), \
                patch.object(queue, "train_job", side_effect=train), patch.object(queue, "compare") as compare:
            queue.fixed_gpu_worker(1, [])
        self.assertEqual(calls, ["mt_evaluation_complete", (1, "seed44")])
        compare.assert_not_called()

    def test_fixed_policy_bypasses_automatic_loss_selection(self):
        queue, _ = self.fixed_queue()
        with patch.object(queue, "run_fixed_massive") as fixed, \
                patch.object(queue, "wait_for_selection") as selection, patch.object(queue, "parallel") as parallel:
            queue.run()
        fixed.assert_called_once()
        selection.assert_not_called()
        parallel.assert_not_called()

    def test_fixed_restart_does_not_silently_retrain_failed_massive_child(self):
        queue, jobs = self.fixed_queue()
        queue.update(job=jobs[0], training_status="running", gpu=0)
        with patch.object(queue, "reserve_gpu"), \
                patch("scripts.mt_retrieval_multiseed_queue.find_completed", return_value=None), \
                patch.object(queue, "gap_train_and_evaluate") as mt:
            with self.assertRaisesRegex(RuntimeError, "without a verified adapter"):
                queue.fixed_gpu_worker(0, [jobs[0]])
        mt.assert_not_called()

    def massive_evaluation_fixture(self):
        run, output, job, _, _, retrieval = self.retrieval_fixture()
        job.update(id="seed43", steps=100000)
        settings = {"enabled": True, "evaluation_id": "massive_final", "language_scopes": ["in", "out"],
                    "alignment_batch_size": 16, "massive_batch_size": 16, "retrieval_chunk_size": 256,
                    "max_new_tokens": 128, "eval_sample_log_limit": 64, "save_alignment_embeddings": False,
                    "save_alignment_sample_metrics": True,
                    "pair_counts": {scope: {"de-en": 2} for scope in ("in", "out")},
                    "language_counts": {scope: {lang: 2} for scope, lang in (("in", "en"), ("out", "fr"))}}
        meta = {**settings, "status": "completed", "split": "test", "tasks": ["alignment", "massive"],
                "checkpoint_path": str(run), "experiment_config": {**job["config"], "checkpoint_global_step": 100000},
                "results": {}}
        for scope, lang in (("in", "en"), ("out", "fr")):
            massive = {"num_examples": 2, "per_language": {lang: {"num_examples": 2, "slot_precision": .5,
                       "slot_recall": .5, "slot_f1": .5, "exact_match": .5}}}
            meta["results"][scope] = {"alignment": retrieval, "massive": massive}
            for task, metrics in meta["results"][scope].items():
                self.write(output / "test" / scope / f"{task}_metrics.json", metrics)
            (output / "test" / scope / "massive_predictions.jsonl").write_text('{}\n{}\n')
            for name in ("eval_samples.json", "eval_samples_embeddings.pkl", "alignment_samples.jsonl", "alignment_batches.jsonl"):
                (output / "test" / scope / name).write_text("fixture")
        self.write(output / "test/evaluation_metadata.json", meta)
        return run, output, job, settings, meta

    def test_massive_completion_requires_both_tasks_scopes_final_step_and_settings(self):
        run, output, job, settings, meta = self.massive_evaluation_fixture()
        self.assertTrue(massive_evaluation_complete(output, run, job, settings))
        for changes in ({"tasks": ["alignment"]}, {"language_scopes": ["in"]}, {"status": "running"},
                        {"checkpoint_path": str(self.root / "other")}, {"max_new_tokens": 16384},
                        {"experiment_config": {**meta["experiment_config"], "checkpoint_global_step": 99000}}):
            self.write(output / "test/evaluation_metadata.json", {**meta, **changes})
            self.assertFalse(massive_evaluation_complete(output, run, job, settings), changes)

    def test_massive_completion_rejects_partial_predictions_metrics_and_samples(self):
        run, output, job, settings, meta = self.massive_evaluation_fixture()
        for name in ("out/massive_predictions.jsonl", "out/massive_metrics.json", "out/alignment_metrics.json",
                     "out/eval_samples_embeddings.pkl", "in/alignment_samples.jsonl"):
            path = output / "test" / name
            previous = path.read_bytes()
            path.unlink()
            self.assertFalse(massive_evaluation_complete(output, run, job, settings), name)
            path.write_bytes(previous)
        (output / "test/out/massive_predictions.jsonl").write_text('{}\n')
        self.assertFalse(massive_evaluation_complete(output, run, job, settings))
        changed = deepcopy(settings)
        changed["pair_counts"]["in"]["de-en"] = 3
        self.assertFalse(massive_evaluation_complete(output, run, job, changed))

    def test_massive_evaluator_uses_its_own_settings_and_reuses_verified_outputs(self):
        run, _, job, settings, _ = self.massive_evaluation_fixture()
        settings["alignment_language_scopes"] = ["in"]
        queue = ExperimentQueue({"python": "python", "max_new_tokens": 16384,
                                 "massive_post_training_evaluation": settings}, self.root / "queue")
        queue.update(job=job, run=str(run), training_status="completed")
        module = "scripts.mt_retrieval_multiseed_queue"
        with patch(f"{module}.completed_run", return_value=True), \
                patch(f"{module}.massive_evaluation_complete", side_effect=[False, True, True, True]), \
                patch.object(queue, "lock", return_value=nullcontext(None)), \
                patch.object(queue, "execute") as execute:
            queue.evaluate_massive_job(1, job)
            command, env, _, _ = execute.call_args.args
            self.assertEqual(command[command.index("--max_new_tokens") + 1], "128")
            self.assertEqual(command[command.index("--language_scope") + 1], "both")
            self.assertEqual(command[command.index("--alignment_language_scope") + 1], "in")
            self.assertEqual(command[command.index("--tasks") + 1:command.index("--tasks") + 3], ["alignment", "massive"])
            self.assertEqual(command[command.index("--output_dir") + 1], str(run / "evaluations/massive_final"))
            self.assertIn("--no-save_alignment_embeddings", command)
            self.assertIn("--save_alignment_sample_metrics", command)
            self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "1")
            queue.evaluate_massive_job(1, job)
            execute.assert_called_once()
        self.assertTrue(queue.state["jobs"]["seed43"]["evaluation_reused"])
        self.assertEqual(queue.state["jobs"]["seed43"]["evaluation_status"], "completed")

    def test_in_only_retrieval_completion_keeps_out_slot_metrics_and_reuses_legacy_results(self):
        run, output, job, settings, meta = self.massive_evaluation_fixture()
        settings["alignment_language_scopes"] = ["in"]
        self.assertTrue(massive_evaluation_complete(output, run, job, settings))
        meta["alignment_language_scopes"] = ["in"]
        del meta["results"]["out"]["alignment"]
        self.write(output / "test/evaluation_metadata.json", meta)
        for name in ("alignment_metrics.json", "alignment_samples.jsonl", "alignment_batches.jsonl"):
            (output / "test/out" / name).unlink()
        self.assertTrue(massive_evaluation_complete(output, run, job, settings))
        self.assertFalse(massive_evaluation_complete(output, run, job, {**settings, "alignment_language_scopes": ["in", "out"]}))
        (output / "test/out/massive_metrics.json").unlink()
        self.assertFalse(massive_evaluation_complete(output, run, job, settings))

    def test_fixed_worker_evaluates_each_seed_before_claiming_the_next(self):
        queue, _ = self.fixed_queue()
        calls = []
        def train(gpu, job):
            calls.append(("train", job["id"]))
            queue.update(job=job, training_status="completed")
        with patch.object(queue, "reserve_gpu"), patch.object(queue, "gap_train_and_evaluate"), \
                patch.object(queue, "train_job", side_effect=train), \
                patch.object(queue, "evaluate_massive_job", side_effect=lambda gpu, job: calls.append(("evaluate", job["id"]))):
            queue.fixed_gpu_worker(0, [])
        self.assertEqual(calls, [("train", "seed43"), ("evaluate", "seed43"), ("train", "seed44"), ("evaluate", "seed44")])

    def test_fixed_worker_does_not_advance_after_failed_final_evaluation(self):
        queue, _ = self.fixed_queue()
        def train(gpu, job):
            queue.update(job=job, training_status="completed")
        with patch.object(queue, "reserve_gpu"), patch.object(queue, "gap_train_and_evaluate"), \
                patch.object(queue, "train_job", side_effect=train) as training, \
                patch.object(queue, "evaluate_massive_job", side_effect=RuntimeError("evaluation failed")):
            with self.assertRaisesRegex(RuntimeError, "evaluation failed"):
                queue.fixed_gpu_worker(0, [])
        training.assert_called_once()
        self.assertEqual(queue.state["jobs"]["seed43"]["training_status"], "completed")
        self.assertNotIn("seed44", queue.state["jobs"])

    def test_fixed_restart_evaluates_recovered_seed_before_other_work(self):
        queue, jobs = self.fixed_queue()
        queue.update(job=jobs[0], training_status="running", assigned_gpu=0)
        queue.update(job=jobs[1], training_status="completed", assigned_gpu=1)
        calls = []
        with patch.object(queue, "reserve_gpu"), \
                patch("scripts.mt_retrieval_multiseed_queue.find_completed", return_value=self.root / "run"), \
                patch.object(queue, "gap_train_and_evaluate", side_effect=lambda gpu: calls.append("mt")), \
                patch.object(queue, "evaluate_massive_job", side_effect=lambda gpu, job: calls.append(job["id"])), \
                patch.object(queue, "train_job") as train:
            queue.fixed_gpu_worker(0, [jobs[0]])
        self.assertEqual(calls, ["seed43", "mt"])
        train.assert_not_called()

    def test_fixed_restart_recovers_completed_training_with_pending_evaluation_on_assigned_gpu(self):
        queue, jobs = self.fixed_queue()
        queue.manifest.update(massive_post_training_evaluation={"enabled": True, "language_scopes": ["in", "out"]},
                              retrieval_jobs=[], report_dir=str(self.root / "report"))
        queue.manifest["selection_policy"].update(user_instruction="each seed then evaluate", authorized_at="today")
        queue.update(job=jobs[0], training_status="completed", assigned_gpu=0, evaluation_status="completed")
        queue.update(job=jobs[1], training_status="completed", assigned_gpu=0, evaluation_status="running")
        recovered = {}
        def worker(gpu, recovered_jobs):
            recovered[gpu] = [job["id"] for job in recovered_jobs]
            for job in recovered_jobs:
                self.assertEqual(job["gpu"], gpu)
                queue.update(job=job, evaluation_status="completed")
        with patch.object(queue, "validate_existing_checkpoints"), patch.object(queue, "fixed_gpu_worker", side_effect=worker):
            queue.run_fixed_massive()
        self.assertEqual(recovered, {0: ["seed43", "seed44"], 1: []})
        self.assertEqual(queue.state["status"], "completed")

    def followup_queue(self):
        queue, jobs = self.fixed_queue()
        followups = [{**job, "id": f"sft{job['config']['training_seed']}", "gpu": 0, "mode": "transfer_only",
                      "config": {**job["config"], "training_type": "transfer_only", "alignment_hidden_state_layer": -1}}
                     for job in jobs]
        queue.manifest["massive_followup_jobs"] = followups
        queue.manifest["massive_post_training_evaluation"] = {"enabled": True, "language_scopes": ["in", "out"]}
        queue.update(job=jobs[0], training_status="completed", evaluation_status="completed", assigned_gpu=0)
        queue.update(job=jobs[1], training_status="running", evaluation_status="queued", assigned_gpu=1)
        return queue, followups

    def test_gpu0_runs_pinned_followups_and_evaluates_each_while_gpu1_is_still_training(self):
        queue, _ = self.followup_queue()
        calls = []
        def train(gpu, job):
            calls.append((gpu, "train", job["id"]))
            queue.update(job=job, training_status="completed")
        def evaluate(gpu, job):
            calls.append((gpu, "evaluate", job["id"]))
            queue.update(job=job, evaluation_status="completed")
        with patch.object(queue, "reserve_gpu"), patch.object(queue, "gap_train_and_evaluate"), \
                patch.object(queue, "train_job", side_effect=train), \
                patch.object(queue, "evaluate_massive_job", side_effect=evaluate):
            queue.fixed_gpu_worker(0, [])
            queue.run_massive_followups(1)
        self.assertEqual(calls, [(0, "train", "sft43"), (0, "evaluate", "sft43"),
                                 (0, "train", "sft44"), (0, "evaluate", "sft44")])
        self.assertEqual(queue.state["jobs"]["seed44"]["training_status"], "running")

    def test_followup_restart_recovers_training_and_evaluates_before_next_seed(self):
        queue, jobs = self.followup_queue()
        queue.update(job=jobs[0], training_status="running")
        calls = []
        def train(gpu, job):
            calls.append(("train", job["id"]))
            queue.update(job=job, training_status="completed")
        def evaluate(gpu, job):
            calls.append(("evaluate", job["id"]))
            queue.update(job=job, evaluation_status="completed")
        with patch("scripts.mt_retrieval_multiseed_queue.find_completed", return_value=self.root / "run") as find, \
                patch.object(queue, "train_job", side_effect=train) as training, \
                patch.object(queue, "evaluate_massive_job", side_effect=evaluate):
            queue.run_massive_followups(0, [jobs[0]])
        self.assertEqual(calls, [("evaluate", "sft43"), ("train", "sft44"), ("evaluate", "sft44")])
        find.assert_called_once_with("massive_root", jobs[0])
        training.assert_called_once_with(0, jobs[1])

    def test_followup_restart_never_silently_retrains_failed_child(self):
        queue, jobs = self.followup_queue()
        queue.update(job=jobs[0], training_status="running")
        with patch("scripts.mt_retrieval_multiseed_queue.find_completed", return_value=None), \
                patch.object(queue, "train_job") as train, patch.object(queue, "evaluate_massive_job") as evaluate:
            with self.assertRaisesRegex(RuntimeError, "without a verified adapter"):
                queue.run_massive_followups(0, [jobs[0]])
        train.assert_not_called()
        evaluate.assert_not_called()
        self.assertEqual(queue.state["jobs"]["sft43"]["training_status"], "failed")

    def test_followup_evaluation_failure_preserves_training_success_and_blocks_next_seed(self):
        queue, _ = self.followup_queue()
        def train(gpu, job):
            queue.update(job=job, training_status="completed")
        with patch.object(queue, "train_job", side_effect=train) as training, \
                patch.object(queue, "evaluate_massive_job", side_effect=RuntimeError("evaluation failed")):
            with self.assertRaisesRegex(RuntimeError, "evaluation failed"):
                queue.run_massive_followups(0)
        training.assert_called_once()
        self.assertEqual(queue.state["jobs"]["sft43"]["training_status"], "completed")
        self.assertNotIn("sft44", queue.state["jobs"])

    def test_queue_completion_requires_followup_evaluations(self):
        queue, followups = self.followup_queue()
        queue.manifest.update(retrieval_jobs=[], report_dir=str(self.root / "report"))
        queue.manifest["selection_policy"].update(user_instruction="SFT seeds on GPU0", authorized_at="today")
        for job in queue.fixed_massive_jobs():
            queue.update(job=job, training_status="completed", evaluation_status="completed")
        for job in followups:
            queue.update(job=job, training_status="completed", evaluation_status="queued")
        with patch.object(queue, "validate_existing_checkpoints"), patch.object(queue, "parallel"):
            with self.assertRaisesRegex(RuntimeError, "every seed"):
                queue.run_fixed_massive()
        self.assertNotEqual(queue.state["status"], "completed")

    def test_sft_launcher_selects_only_requested_llama_seed_and_fifty_thousand_sft_steps(self):
        root = Path(__file__).resolve().parents[1]
        for seed in (43, 44):
            env = {**os.environ, "DRY_RUN": "1", "CUDA_VISIBLE_DEVICES": "0", "PYTHON_BIN": sys.executable,
                   "MODEL_NAME": "meta-llama/Llama-3.2-1B-Instruct", "TRAINING_SEED": str(seed),
                   "ALIGNMENT_LOSS": "gap_distance_infonce", "TRAIN_EVAL_LANGUAGE_SCOPE": "both",
                   "ALIGNMENT_BATCHING": "same_pair"}
            result = subprocess.run(["bash", "scripts/massive_transfer_only.sh", "transfer_only"], cwd=root,
                                    env=env, capture_output=True, text=True, check=True)
            lines = [line for line in result.stdout.splitlines() if line.startswith("CUDA_VISIBLE_DEVICES=")]
            self.assertEqual(len(lines), 1)
            command = shlex.split(lines[0])
            with patch.object(sys, "argv", ["main.py", *command[command.index("main.py") + 1:]]):
                config = parse_training_args()
            self.assertEqual(config.model_name, env["MODEL_NAME"])
            self.assertEqual(config.training_seed, seed)
            self.assertEqual(config.training_type, "transfer_only")
            self.assertEqual(config.num_steps, 50000)
            self.assertEqual(config.alignment_hidden_state_layer, -1)
            self.assertEqual(config.quantization_compute_dtype, "bfloat16")
            self.assertEqual(config.eval_language_scope, "both")

    def shared_layer_queue(self):
        queue, pinned = self.followup_queue()
        shared = [{**pinned[0], "id": name, "gpu": None, "eligible_gpus": [0, 1], "mode": mode, "loss": loss,
                   "config": {**pinned[0]["config"], "training_type": mode, "alignment_loss": loss,
                              "alignment_hidden_state_layer": 8}}
                  for name, mode, loss in [("layer8_gap_c", "contrastive_only", "gap_distance_infonce"),
                                           ("layer8_gap_alt", "alternative", "gap_distance_infonce"),
                                           ("layer8_info_alt", "alternative", "infonce")]]
        shared[-1]["after_jobs"] = [job["id"] for job in shared[:2]]
        queue.manifest["massive_followup_jobs"] = [*pinned, *shared]
        return queue, pinned, shared

    def test_shared_jobs_use_first_ready_gpu_without_stealing_pinned_sft(self):
        queue, pinned, shared = self.shared_layer_queue()
        queue.update(job=pinned[0], training_status="running", assigned_gpu=0)
        first, _ = queue.claim_massive_followup(1)
        self.assertEqual((first["id"], first["gpu"]), (shared[0]["id"], 1))
        queue.update(job=pinned[0], training_status="completed", evaluation_status="completed")
        next_sft, _ = queue.claim_massive_followup(0)
        self.assertEqual((next_sft["id"], next_sft["gpu"]), (pinned[1]["id"], 0))

    def test_concurrent_shared_claims_are_distinct_and_keep_infonce_blocked(self):
        queue, pinned, shared = self.shared_layer_queue()
        for job in pinned:
            queue.update(job=job, training_status="completed", evaluation_status="completed")
        with ThreadPoolExecutor(max_workers=2) as pool:
            claims = list(pool.map(queue.claim_massive_followup, [0, 1]))
        self.assertEqual({job["id"] for job, _ in claims}, {job["id"] for job in shared[:2]})
        self.assertEqual({job["gpu"] for job, _ in claims}, {0, 1})
        for gpu in (0, 1):
            self.assertEqual(queue.claim_massive_followup(gpu), (None, True))

    def test_infonce_waits_for_both_gap_final_evaluations_not_only_training(self):
        queue, pinned, shared = self.shared_layer_queue()
        for job in [*pinned, shared[0]]:
            queue.update(job=job, training_status="completed", evaluation_status="completed")
        queue.update(job=shared[1], training_status="completed", evaluation_status="running", assigned_gpu=1)
        self.assertEqual(queue.claim_massive_followup(0), (None, True))
        queue.update(job=shared[1], evaluation_status="completed")
        job, _ = queue.claim_massive_followup(0)
        self.assertEqual(job["id"], shared[2]["id"])

    def test_restart_preserves_shared_jobs_assigned_gpu_for_training_or_evaluation(self):
        queue, pinned, shared = self.shared_layer_queue()
        queue.manifest.update(retrieval_jobs=[], report_dir=str(self.root / "report"))
        queue.manifest["selection_policy"].update(user_instruction="first available GPU", authorized_at="today")
        for job in [*queue.fixed_massive_jobs(), *pinned]:
            queue.update(job=job, training_status="completed", evaluation_status="completed", assigned_gpu=job["gpu"])
        queue.update(job=shared[0], training_status="completed", evaluation_status="running", assigned_gpu=1)
        queue.update(job=shared[1], training_status="running", evaluation_status="queued", assigned_gpu=0)
        with patch.object(queue, "validate_existing_checkpoints"), patch.object(queue, "parallel"):
            with self.assertRaisesRegex(RuntimeError, "every seed"):
                queue.run_fixed_massive()
        self.assertIn(shared[0]["id"], [job["id"] for job in queue.recovered_followups[1]])
        self.assertIn(shared[1]["id"], [job["id"] for job in queue.recovered_followups[0]])
        self.assertEqual(queue.state["jobs"][shared[0]["id"]]["assigned_gpu"], 1)
        self.assertEqual(queue.state["jobs"][shared[1]["id"]]["assigned_gpu"], 0)

    def test_layer_launcher_respects_frozen_depth_and_objective_budget(self):
        root = Path(__file__).resolve().parents[1]
        for mode, loss, layer, steps in [("contrastive_only", "gap_distance_infonce", 8, 50000),
                                          ("alternative", "gap_distance_infonce", 8, 100000),
                                          ("alternative", "infonce", 8, 100000),
                                          ("alternative", "gap_distance_infonce", -1, 100000),
                                          ("contrastive_then_transfer", "gap_distance_infonce", 8, 100000)]:
            manifest = {"python": sys.executable, "results_root": str(self.root), "manifest_sha256": "fixture"}
            queue = ExperimentQueue(manifest, self.root / "launcher")
            job = {"id": "fixture", "model": "meta-llama/Llama-3.2-1B-Instruct", "mode": mode, "loss": loss,
                   "script": "scripts/massive_transfer_only.sh" if mode == "contrastive_then_transfer" else "scripts/massive_contrastive_only.sh", "steps": steps,
                   "config": {"model_name": "meta-llama/Llama-3.2-1B-Instruct", "training_type": mode,
                              "alignment_loss": loss, "alignment_hidden_state_layer": layer, "training_seed": 43,
                              "num_steps": steps, "output_root": str(self.root)}}
            with patch("scripts.wmt23_pipeline.find_completed", side_effect=[None, self.root / "run"]), \
                    patch.object(queue, "lock", return_value=nullcontext(None)), patch.object(queue, "execute") as execute:
                queue.train_job(0, job)
            command, env, _, _ = execute.call_args.args
            self.assertEqual(command[command.index("--alignment_hidden_state_layer") + 1], str(layer))
            self.assertEqual(command[command.index("--training_type") + 1], mode)
            self.assertEqual(command[command.index("--num_steps") + 1], str(steps))
            self.assertEqual(env["ALIGNMENT_HIDDEN_STATE_LAYER"], str(layer))

    def test_mt_followup_finishes_retrieval_and_translation_before_returning(self):
        queue, jobs = self.followup_queue()
        job = {**jobs[0], "id": "mt_gap_then", "config": {**jobs[0]["config"], "downstream_task": "wmt23"}}
        calls = []
        with patch.object(queue, "train_job", side_effect=lambda *a: calls.append("train")), \
                patch.object(queue, "retrieve_job", side_effect=lambda *a: calls.append("retrieval")), \
                patch.object(queue, "evaluate", side_effect=lambda *a, **k: calls.append(("translation", k["jobs"]))) as evaluate, \
                patch.object(queue, "evaluate_massive_job") as massive:
            queue.run_massive_followup(1, job)
        self.assertEqual(calls, ["train", "retrieval", ("translation", [job])])
        evaluate.assert_called_once_with(1, jobs=[job])
        massive.assert_not_called()

    def test_mt_retrieval_failure_does_not_start_translation(self):
        queue, jobs = self.followup_queue()
        job = {**jobs[0], "config": {**jobs[0]["config"], "downstream_task": "wmt23"}}
        with patch.object(queue, "train_job"), \
                patch.object(queue, "retrieve_job", side_effect=RuntimeError("incomplete retrieval")), \
                patch.object(queue, "evaluate") as evaluate:
            with self.assertRaisesRegex(RuntimeError, "incomplete retrieval"):
                queue.run_massive_followup(0, job)
        evaluate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
