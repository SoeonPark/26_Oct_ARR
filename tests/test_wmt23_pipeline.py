"""Global phase barriers and artifact validation for the live MT pipeline."""

import json
from pathlib import Path
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import sys
from unittest.mock import patch

from scripts.wmt23_pipeline import (
    CHECKPOINT_SHA256, MODEL_ID, Pipeline, comet_complete, completed_run,
    config_matches, evaluation_counts, evaluation_dependency_complete, file_sha256, generation_complete, process_identity, run_phases,
)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def write(self, name, payload):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
        return path

    def test_all_gpus_must_complete_each_stage_before_the_next(self):
        priority, evaluation, later = set(), set(), set()
        mutex = threading.Lock()
        together = threading.Barrier(2, timeout=5)

        def train(gpu):
            together.wait()
            with mutex:
                self.assertFalse(evaluation)
                priority.add(gpu)

        def evaluate(gpu):
            with mutex:
                self.assertEqual(priority, {0, 1})
                self.assertFalse(later)
            together.wait()
            with mutex:
                evaluation.add(gpu)

        def deferred(gpu):
            with mutex:
                self.assertEqual(evaluation, {0, 1})
                later.add(gpu)

        run_phases([0, 1], train, evaluate, deferred)
        self.assertEqual(later, {0, 1})

    def test_priority_failure_blocks_evaluation_and_later_training(self):
        called = []

        def fail(gpu):
            if gpu == 1:
                raise RuntimeError("failed priority training")

        with self.assertRaisesRegex(RuntimeError, "priority"):
            run_phases([0, 1], fail, called.append, called.append)
        self.assertEqual(called, [])

    def test_later_lane_runs_without_waiting_for_training_or_evaluation(self):
        called = []

        def forbidden(gpu):
            self.fail("An independent later lane must not enter priority/evaluation stages.")

        run_phases([0], forbidden, forbidden, called.append, stages=["later_training"])
        self.assertEqual(called, [0])

    def test_evaluation_lane_does_not_launch_later_training(self):
        called = []
        run_phases([1], lambda gpu: called.append(("priority", gpu)),
                   lambda gpu: called.append(("evaluation", gpu)),
                   lambda gpu: self.fail("Later training belongs to GPU 0."),
                   stages=["priority_training", "mt_evaluation"])
        self.assertEqual(called, [("priority", 1), ("evaluation", 1)])

    def test_evaluation_only_queue_never_launches_training(self):
        called = []

        def forbidden(gpu):
            self.fail("The BLEU/COMET queue must not launch any training.")

        run_phases([1], forbidden, called.append, forbidden, stages=["mt_evaluation"])
        self.assertEqual(called, [1])

    def test_new_training_finishes_on_both_gpus_before_preserved_later_training(self):
        completed = set()
        barrier = threading.Barrier(2, timeout=5)

        def priority(gpu):
            barrier.wait()
            completed.add(gpu)

        def later(gpu):
            self.assertEqual(completed, {0, 1})

        run_phases([0, 1], priority, lambda gpu: self.fail("No inference was queued."),
                   later, stages=["priority_training", "later_training"])

    def test_scope_overrides_preserve_active_evaluations_and_change_new_jobs(self):
        manifest = {"evaluation_language_scopes": ["in"],
                    "test_counts": {"in": {"en-de": 2}, "out": {"en-zh": 3}}}
        self.assertEqual(evaluation_counts(manifest, {}), {"in": {"en-de": 2}})
        self.assertEqual(evaluation_counts(manifest, {"evaluation_language_scopes": ["in", "out"]}),
                         manifest["test_counts"])
        for scopes in ([], ["in", "in"], ["unknown"]):
            with self.assertRaises(ValueError):
                evaluation_counts(manifest, {"evaluation_language_scopes": scopes})

    def test_in_only_evaluation_launches_generation_and_comet_for_in_only(self):
        job = {"id": "ready", "priority": True}
        manifest = {"jobs": [job], "python": sys.executable, "comet_python": sys.executable,
                    "evaluation_id": "in_only", "results_root": str(self.root),
                    "test_counts": {"in": {"en-de": 2}, "out": {"en-zh": 3}},
                    "evaluation_language_scopes": ["in"], "wmt23_batch_size": 16,
                    "max_new_tokens": 16384}
        pipeline = Pipeline(manifest, self.root / "queue")
        pipeline.state["jobs"]["ready"] = {"run": str(self.root / "checkpoint")}
        with patch.object(pipeline, "reserve_gpu"), \
                patch.object(pipeline, "evaluation_jobs", return_value=iter([job])), \
                patch.object(pipeline, "lock", return_value=nullcontext(None)), \
                patch.object(pipeline, "execute") as execute, \
                patch("scripts.wmt23_pipeline.completed_run", return_value=True), \
                patch("scripts.wmt23_pipeline.generation_complete", side_effect=[False, True]), \
                patch("scripts.wmt23_pipeline.comet_complete", side_effect=[False, True]):
            pipeline.evaluate(0)
        self.assertEqual(execute.call_count, 2)
        self.assertEqual(execute.call_args_list[0].args[1]["EVAL_LANGUAGE_SCOPE"], "in")
        self.assertEqual(execute.call_args_list[0].args[1]["EVAL_WMT23_BATCH_SIZE"], "16")
        command = execute.call_args_list[1].args[0]
        self.assertTrue(command[command.index("--prediction_dir") + 1].endswith("/test/in"))
        self.assertEqual(pipeline.state["jobs"]["ready"]["evaluation_status"], "completed")

    def test_explicit_mt_followup_evaluates_only_its_checkpoint_and_uses_its_output_id(self):
        job = {"id": "followup", "evaluation_id": "gap_then_final", "evaluation_language_scopes": ["in"]}
        manifest = {"jobs": [{"id": "unrelated"}], "evaluation_id": "old_eval", "results_root": str(self.root),
                    "python": sys.executable, "comet_python": sys.executable,
                    "test_counts": {"in": {"en-de": 2}}, "max_new_tokens": 16384, "wmt23_batch_size": 16}
        pipeline = Pipeline(manifest, self.root / "explicit")
        run = self.root / "checkpoint"
        pipeline.state["jobs"][job["id"]] = {"run": str(run)}
        with patch.object(pipeline, "reserve_gpu"), \
                patch.object(pipeline, "evaluation_jobs", side_effect=AssertionError("Must not run unrelated jobs")), \
                patch.object(pipeline, "lock", return_value=nullcontext(None)), \
                patch.object(pipeline, "execute") as execute, \
                patch("scripts.wmt23_pipeline.completed_run", return_value=True), \
                patch("scripts.wmt23_pipeline.generation_complete", side_effect=[False, True]), \
                patch("scripts.wmt23_pipeline.comet_complete", side_effect=[False, True]):
            pipeline.evaluate(1, jobs=[job])
        self.assertEqual(execute.call_count, 2)
        self.assertEqual(execute.call_args_list[0].args[1]["EVAL_OUTPUT_DIR"], str(run / "evaluations/gap_then_final"))
        self.assertEqual(pipeline.state["jobs"][job["id"]]["evaluation_status"], "completed")
        self.assertNotIn("unrelated", pipeline.state["jobs"])

    def test_gpu_workers_claim_distinct_ready_models_and_skip_running_training(self):
        jobs = [{"id": name, "priority": True} for name in ("training", "ready-a", "ready-b")]
        pipeline = Pipeline({"jobs": jobs, "results_root": str(self.root)}, self.root)
        pipeline.state["jobs"] = {job["id"]: {"run": str(self.root / job["id"])} for job in jobs}
        with patch("scripts.wmt23_pipeline.completed_run", side_effect=lambda path, job: job["id"] != "training"):
            with ThreadPoolExecutor(max_workers=2) as workers:
                claimed = list(workers.map(pipeline.claim_evaluation, (0, 1)))
            self.assertEqual({job["id"] for job, _ in claimed}, {"ready-a", "ready-b"})
            self.assertEqual(pipeline.claim_evaluation(0), (None, True))
        self.assertNotIn("evaluation_status", pipeline.state["jobs"]["training"])

    def test_evaluation_failure_blocks_later_training(self):
        called = []

        def fail(gpu):
            if gpu == 0:
                raise RuntimeError("COMET failed")

        with self.assertRaisesRegex(RuntimeError, "COMET"):
            run_phases([0, 1], lambda gpu: None, fail, called.append)
        self.assertEqual(called, [])

    def test_only_verified_final_adapters_count_as_finished_training(self):
        job = {"steps": 100, "config": {"training_seed": 42, "num_steps": 100}}
        self.write("experiment_config.json", {**job["config"], "checkpoint_global_step": 100})
        self.write("trainer_state.json", {"global_step": 100})
        self.write("run_metadata.json", {"status": "completed"})
        self.write("adapter_config.json", {})
        (self.root / "adapter_model.safetensors").write_bytes(b"fixture")
        self.assertTrue(completed_run(self.root, job))
        self.write("trainer_state.json", {"global_step": 99})
        self.assertFalse(completed_run(self.root, job))
        self.write("trainer_state.json", {"global_step": 100})
        self.write("run_metadata.json", {"status": "failed"})
        self.assertFalse(completed_run(self.root, job))

    def test_scientific_config_must_match_but_data_symlinks_can_differ(self):
        real = self.root / "data"
        real.mkdir()
        alias = self.root / "alias"
        alias.symlink_to(real, target_is_directory=True)
        wanted = {"training_seed": 42, "wmt23_data_dir": str(alias), "wandb_run_name": "old"}
        actual = {**wanted, "wmt23_data_dir": str(real), "wandb_run_name": "new"}
        self.assertTrue(config_matches(wanted, actual))
        self.assertFalse(config_matches(wanted, {**actual, "training_seed": 43}))

    def test_partial_generation_is_not_success(self):
        counts = {"in": {"en-de": 2}, "out": {"en-zh": 3}}
        run = self.root / "checkpoint"
        metadata = {"status": "completed", "tasks": ["wmt23"], "checkpoint_path": str(run),
                    "language_scopes": ["in", "out"], "wmt23_metric": "sacrebleu",
                    "wmt23_batch_size": 1, "wmt23_max_new_tokens": 16384}
        self.write("test/evaluation_metadata.json", metadata)
        for scope, values in counts.items():
            self.write(f"test/{scope}/wmt23_metrics.json", {
                "status": "scored", "metric": "sacrebleu",
                "by_language": {key: {"num_examples": n} for key, n in values.items()}})
            for direction, count in values.items():
                (self.root / f"test/{scope}/wmt23_predictions.{direction}.jsonl").write_text('{}\n' * count)
        self.assertTrue(generation_complete(self.root, run, counts, 16384))
        self.assertFalse(generation_complete(self.root, run, counts, 512))
        # A restarted batch-16/EOS-corrected queue must reject the old artifacts.
        policy = "generation_config_plus_tokenizer_v1"
        self.assertFalse(generation_complete(self.root, run, counts, 16384, 16, policy))
        metadata["wmt23_batch_size"] = 16
        self.write("test/evaluation_metadata.json", metadata)
        self.assertFalse(generation_complete(self.root, run, counts, 16384, 16, policy))
        metadata["wmt_eos_policy"] = policy
        self.write("test/evaluation_metadata.json", metadata)
        self.assertTrue(generation_complete(self.root, run, counts, 16384, 16, policy))
        self.assertFalse(generation_complete(self.root, run, counts, 16384, 1, policy))
        (self.root / "test/out/wmt23_predictions.en-zh.jsonl").unlink()
        self.assertFalse(generation_complete(self.root, run, counts, 16384, 16, policy))
        metadata["language_scopes"] = ["in"]
        self.write("test/evaluation_metadata.json", metadata)
        self.assertTrue(generation_complete(self.root, run, {"in": counts["in"]}, 16384, 16, policy))
        self.assertFalse(generation_complete(self.root, run, counts, 16384, 16, policy))

    def test_stale_comet_predictions_do_not_release_the_barrier(self):
        prediction = self.root / "wmt23_predictions.en-de.jsonl"
        prediction.write_text('{"sample_id":"1"}\n')
        self.write("wmt23_comet22_metrics.json", {
            "status": "scored", "num_examples": 1,
            "scorer_metadata": {"model_id": MODEL_ID, "checkpoint_sha256": CHECKPOINT_SHA256},
            "by_language": {"en-de": {"num_examples": 1}},
            "prediction_files": [{"path": str(prediction), "sha256": file_sha256(prediction)}]})
        (self.root / "wmt23_comet22_scores.jsonl").write_text('{}\n')
        self.assertTrue(comet_complete(self.root, {"en-de": 1}))
        prediction.write_text('{"sample_id":"changed"}\n')
        self.assertFalse(comet_complete(self.root, {"en-de": 1}))

    def test_missing_adopted_process_is_not_alive(self):
        self.assertIsNone(process_identity(999999999))

    def evaluation_dependency(self, status):
        manifest = {
            "jobs": [{"id": "priority", "priority": True}], "evaluation_id": "batch16_eos",
            "test_counts": {"in": {"en-de": 2}, "out": {"en-zh": 3}},
            "max_new_tokens": 16384, "wmt23_batch_size": 16,
            "wmt_eos_policy": "generation_config_plus_tokenizer_v1",
        }
        path = self.write("upstream/manifest.json", manifest)
        self.write("upstream/state.json", {"status": status, "jobs": {
            "priority": {"evaluation_status": "completed", "run": str(self.root / "run")}}})
        return {"state_dir": str(path.parent), "manifest_sha256": file_sha256(path)}

    def test_later_queue_waits_for_all_mt_even_if_a_job_is_complete(self):
        dependency = self.evaluation_dependency("running")
        self.assertFalse(evaluation_dependency_complete(dependency))

    def test_later_queue_rejects_failed_cancelled_or_changed_mt_queue(self):
        for status in ("failed", "cancelled_by_user", "superseded_by_user"):
            with self.subTest(status=status):
                with self.assertRaisesRegex(RuntimeError, "cannot start"):
                    evaluation_dependency_complete(self.evaluation_dependency(status))
        dependency = self.evaluation_dependency("completed")
        dependency["manifest_sha256"] = "wrong"
        with self.assertRaisesRegex(RuntimeError, "manifest changed"):
            evaluation_dependency_complete(dependency)

    def test_later_queue_requires_both_bleu_and_comet_under_the_new_protocol(self):
        dependency = self.evaluation_dependency("completed")
        with patch("scripts.wmt23_pipeline.completed_run", return_value=True), \
                patch("scripts.wmt23_pipeline.generation_complete", return_value=True) as generation, \
                patch("scripts.wmt23_pipeline.comet_complete", return_value=True) as comet:
            self.assertTrue(evaluation_dependency_complete(dependency))
            self.assertEqual(generation.call_args.args[-2:], (16, "generation_config_plus_tokenizer_v1"))
            self.assertEqual(comet.call_count, 2)
            generation.return_value = False
            with self.assertRaisesRegex(RuntimeError, "generation/BLEU"):
                evaluation_dependency_complete(dependency)
            generation.return_value = True
            comet.return_value = False
            with self.assertRaisesRegex(RuntimeError, "COMET-22"):
                evaluation_dependency_complete(dependency)

    def test_failed_dependency_never_starts_training_or_reserves_a_gpu(self):
        dependency = self.evaluation_dependency("failed")
        pipeline = Pipeline({"stages": ["later_training"], "jobs": [],
                             "wait_for_evaluation": dependency}, self.root / "later")
        with patch("scripts.wmt23_pipeline.run_phases") as phases, \
                patch.object(pipeline, "reserve_gpu") as reserve:
            with self.assertRaises(RuntimeError):
                pipeline.run()
            phases.assert_not_called()
            reserve.assert_not_called()

    def test_dependency_checks_only_the_scopes_required_for_each_job(self):
        dependency = self.evaluation_dependency("completed")
        path = Path(dependency["state_dir"]) / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["evaluation_language_scopes"] = ["in"]
        path.write_text(json.dumps(manifest))
        dependency["manifest_sha256"] = file_sha256(path)
        with patch("scripts.wmt23_pipeline.completed_run", return_value=True), \
                patch("scripts.wmt23_pipeline.generation_complete", return_value=True) as generation, \
                patch("scripts.wmt23_pipeline.comet_complete", return_value=True) as comet:
            self.assertTrue(evaluation_dependency_complete(dependency))
            self.assertEqual(generation.call_args.args[2], {"in": {"en-de": 2}})
            self.assertEqual(comet.call_count, 1)


if __name__ == "__main__":
    unittest.main()
