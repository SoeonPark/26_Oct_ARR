import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.run_massive_gap_variants import check_saved_plan
from scripts.run_wmt23_gap_variants import WMT23GapQueue, build_manifest, main, parse_args
from scripts.wmt23_pipeline import file_sha256


class WMT23GapLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data_dir = self.root / "data"
        self.data_dir.mkdir()
        pairs = ("de-en", "cs-en", "en-ja")
        data = dict(
            corpus_profile="alma_ja_opus", data_seed=42,
            counts={"test": {direction: 10 for lang in ("de", "cs", "ja")
                             for direction in (f"en-{lang}", f"{lang}-en")}},
            opus_files={pair: {"test": f"{pair}.parquet"} for pair in pairs},
            files={f"{pair}.parquet": {"sha256": "0" * 64} for pair in pairs},
        )
        self.data_path = self.data_dir / "manifest.json"
        self.data_path.write_text(json.dumps(data))
        self.data_hash = file_sha256(self.data_path)

    def arguments(self, model="llama", *extra):
        return ["--model", model, "--data-dir", str(self.data_dir),
                "--manifest-sha256", self.data_hash,
                "--comet-python", str(self.root / ".venv-comet22/bin/python"), *extra]

    def manifest(self, model="llama", *extra):
        return build_manifest(parse_args(self.arguments(model, *extra)))

    def queue(self, model="llama", *extra):
        return WMT23GapQueue(self.manifest(model, *extra), self.root / "state")

    def test_default_loss_order_and_frozen_evaluation(self):
        llama = self.manifest()
        qwen = self.manifest("qwen")
        self.assertEqual([job["loss"] for job in llama["jobs"]],
                         ["gap_distance_detach", "gap_distance_rms"])
        self.assertEqual([job["loss"] for job in qwen["jobs"]], ["gap_distance_rms"])
        self.assertEqual(llama["adopted"], {})
        self.assertEqual(llama["python"], sys.executable)
        self.assertEqual(llama["data_dir"], str(self.data_dir))
        self.assertEqual(llama["manifest_sha256"], self.data_hash)
        self.assertEqual(llama["evaluation_id"], "wmt23_gap_variants_final")
        self.assertEqual(llama["evaluation_language_scopes"], ["in"])
        self.assertEqual(llama["wmt23_batch_size"], 16)
        self.assertEqual(llama["max_new_tokens"], 16384)
        self.assertEqual(llama["wmt_eos_policy"], "generation_config_plus_tokenizer_v1")
        self.assertEqual(sum(llama["test_counts"]["in"].values()), 60)
        retrieval = llama["retrieval"]
        self.assertEqual(retrieval["evaluation_id"], "wmt23_gap_variants_retrieval")
        self.assertEqual(retrieval["pair_counts"], {pair: 2000 for pair in ("de-en", "cs-en", "en-ja")})
        self.assertEqual(set(retrieval["pair_file_sha256"].values()), {"0" * 64})
        self.assertEqual(llama, self.manifest())

    def test_interactive_overrides_cannot_change_experiment(self):
        with patch.dict("os.environ", {
            "ALIGNMENT_HIDDEN_STATE_LAYER": "8", "ALIGNMENT_BATCHING": "mixed",
            "ALIGNMENT_LOSS": "infonce", "TRAINING_SEED": "44",
            "TRAIN_EVAL_LANGUAGE_SCOPE": "both", "WANDB_MODE": "online",
            "MODEL_NAME": "wrong", "TRAIN_SAMPLE_LOG_INTERVAL": "0",
            "DOWNSTREAM_MICRO_BATCH_SIZE": "2", "OUTPUT_ROOT": "/wrong",
            "WMT23_DATA_DIR": "/wrong", "WMT23_CORPUS_PROFILE": "full_parallel",
        }):
            manifest = self.manifest()
        for job in manifest["jobs"]:
            config = job["config"]
            self.assertEqual((config["downstream_task"], config["training_type"],
                              config["training_seed"], config["alignment_hidden_state_layer"],
                              config["num_steps"], config["batch_size"]),
                             ("wmt23", "alternative", 42, -1, 100000, 16))
            self.assertEqual(config["eval_language_scope"], "in")
            self.assertEqual(config["wandb_mode"], "disabled")
            self.assertEqual(config["train_sample_log_interval"], 1000)
            self.assertEqual(config["downstream_micro_batch_size"], 0)
            self.assertEqual(config["alignment_batching"], "same_pair")
            self.assertEqual(config["wmt23_data_dir"], str(self.data_dir))
            self.assertEqual(config["wmt23_corpus_profile"], "alma_ja_opus")
            self.assertEqual(config["training_lang"], ["de", "cs", "ja"])
            self.assertEqual(config["alignment_hidden_state_position"], "last_token")

    def test_explicit_variant_gpu_and_qwen_microbatch(self):
        manifest = self.manifest("qwen", "--loss", "both", "--gpu", "3")
        self.assertEqual(len(manifest["jobs"]), 2)
        self.assertTrue(all(job["gpu"] == 3 for job in manifest["jobs"]))
        self.assertTrue(all(job["config"]["downstream_micro_batch_size"] == 8 for job in manifest["jobs"]))
        self.assertEqual(len(self.manifest("llama", "--loss", "rms")["jobs"]), 1)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(self.arguments("llama", "--gpu", "-1"))

    def test_comet_interpreter_symlink_is_not_resolved_outside_venv(self):
        executable = self.root / ".venv-comet22/bin/python"
        executable.parent.mkdir(parents=True)
        executable.symlink_to(sys.executable)
        self.assertEqual(self.manifest()["comet_python"], str(executable))

    def test_missing_or_different_prepared_data_stops_before_training(self):
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.manifest("llama", "--manifest-sha256", "f" * 64)
        self.data_path.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Copy the entire"):
            self.manifest()

    def test_dry_run_needs_no_gpu_imports_and_writes_no_state(self):
        state = self.root / "not-created"
        code = (
            "import sys; from scripts.run_wmt23_gap_variants import main; "
            "main(sys.argv[1:]); "
            "assert 'torch' not in sys.modules; assert 'datasets' not in sys.modules; "
            "assert 'comet' not in sys.modules"
        )
        result = subprocess.run([sys.executable, "-c", code,
                                 *self.arguments("llama", "--dry-run", "--state-dir", str(state))],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("gap_distance_detach", result.stdout)
        self.assertIn("gap_distance_rms", result.stdout)
        self.assertIn("COMET", result.stdout)
        self.assertFalse(state.exists())

    def test_each_final_retrieval_and_mt_evaluation_precedes_next_training(self):
        queue = self.queue()
        events = []
        with patch.object(queue, "reserve_gpu"), patch.object(queue, "release_gpus"), \
             patch.object(queue, "train_job", side_effect=lambda gpu, job: events.append((job["loss"], "train"))), \
             patch.object(queue, "retrieve_job", side_effect=lambda gpu, job: events.append((job["loss"], "retrieval"))), \
             patch.object(queue, "evaluate", side_effect=lambda gpu, jobs: events.append((jobs[0]["loss"], "mt"))), \
             contextlib.redirect_stdout(io.StringIO()):
            queue.run()
        self.assertEqual(events, [(loss, stage) for loss in ("gap_distance_detach", "gap_distance_rms")
                                  for stage in ("train", "retrieval", "mt")])
        self.assertEqual(queue.state["status"], "completed")

    def test_retrieval_failure_stops_mt_and_next_training(self):
        queue = self.queue()
        with patch.object(queue, "reserve_gpu"), patch.object(queue, "release_gpus"), \
             patch.object(queue, "train_job") as train, \
             patch.object(queue, "retrieve_job", side_effect=RuntimeError("failed retrieval")), \
             patch.object(queue, "evaluate") as evaluate, \
             contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "failed retrieval"):
            queue.run()
        self.assertEqual(train.call_count, 1)
        evaluate.assert_not_called()
        self.assertEqual(queue.state["status"], "failed")

    def test_translation_or_comet_failure_stops_next_training(self):
        queue = self.queue()
        with patch.object(queue, "reserve_gpu"), patch.object(queue, "release_gpus"), \
             patch.object(queue, "train_job") as train, patch.object(queue, "retrieve_job"), \
             patch.object(queue, "evaluate", side_effect=RuntimeError("failed COMET")), \
             contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "failed COMET"):
            queue.run()
        self.assertEqual(train.call_count, 1)
        self.assertEqual(queue.state["status"], "failed")

    def test_environment_freezes_mt_settings_and_clears_evaluation_overrides(self):
        queue = self.queue()
        with patch.dict("os.environ", {"WMT23_DATA_DIR": "/wrong", "COMET22_PYTHON": "/missing",
                                       "ALIGNMENT_LOSS": "infonce", "WANDB_MODE": "online",
                                       "EVAL_TASKS": "massive", "EVAL_COMET22": "true"}):
            env = queue.environment(2)
        self.assertEqual(env["WMT23_DATA_DIR"], str(self.data_dir))
        self.assertEqual(env["WMT23_CORPUS_PROFILE"], "alma_ja_opus")
        self.assertEqual(env["WMT23_DOWNSTREAM_SAMPLING"], "balanced_mixed")
        for key in ("COMET22_PYTHON", "ALIGNMENT_LOSS", "EVAL_TASKS", "EVAL_COMET22"):
            self.assertNotIn(key, env)
        self.assertEqual(env["WANDB_MODE"], "disabled")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "2")
        self.assertEqual(env["PYTHON_BIN"], sys.executable)

    def test_generation_finishes_before_comet_on_the_same_physical_gpu(self):
        queue = self.queue("llama", "--loss", "detach", "--gpu", "3")
        job = queue.manifest["jobs"][0]
        run = self.root / "final-adapter"
        queue.state["jobs"][job["id"]] = {"run": str(run)}
        with patch.object(queue, "reserve_gpu"), \
             patch.object(queue, "lock", return_value=contextlib.nullcontext(None)), \
             patch.object(queue, "execute") as execute, \
             patch("scripts.wmt23_pipeline.completed_run", return_value=True), \
             patch("scripts.wmt23_pipeline.generation_complete", side_effect=[False, True]), \
             patch("scripts.wmt23_pipeline.comet_complete", side_effect=[False, True]), \
             contextlib.redirect_stdout(io.StringIO()):
            queue.evaluate(3, jobs=[job])
        calls = execute.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].args[0], ["bash", "scripts/wmt23_eval.sh", str(run)])
        self.assertEqual(calls[0].args[1]["EVAL_TASKS"], "wmt23")
        self.assertEqual(calls[0].args[1]["EVAL_COMET22"], "false")
        comet_command = calls[1].args[0]
        self.assertEqual(comet_command[0], queue.manifest["comet_python"])
        self.assertEqual(comet_command[1], "scripts/score_comet22.py")
        self.assertEqual(comet_command[comet_command.index("--gpus") + 1], "1")
        self.assertTrue(all(call.args[1]["CUDA_VISIBLE_DEVICES"] == "3" for call in calls))
        self.assertEqual(queue.state["jobs"][job["id"]]["evaluation_status"], "completed")

    def test_completed_generation_is_reused_when_only_comet_is_missing(self):
        queue = self.queue("llama", "--loss", "detach")
        job = queue.manifest["jobs"][0]
        queue.state["jobs"][job["id"]] = {"run": str(self.root / "final-adapter")}
        with patch.object(queue, "reserve_gpu"), \
             patch.object(queue, "lock", return_value=contextlib.nullcontext(None)), \
             patch.object(queue, "execute") as execute, \
             patch("scripts.wmt23_pipeline.completed_run", return_value=True), \
             patch("scripts.wmt23_pipeline.generation_complete", return_value=True), \
             patch("scripts.wmt23_pipeline.comet_complete", side_effect=[False, True]), \
             contextlib.redirect_stdout(io.StringIO()):
            queue.evaluate(0, jobs=[job])
        execute.assert_called_once()
        self.assertEqual(execute.call_args.args[0][1], "scripts/score_comet22.py")

    def test_saved_plan_rejects_changed_or_unidentified_state(self):
        manifest = self.manifest()
        state_dir = self.root / "state"
        state_dir.mkdir()
        state_path = state_dir / "state.json"
        state_path.write_text(json.dumps({"schema_version": 1, "jobs": {}, "gpus": {}}))
        with self.assertRaisesRegex(RuntimeError, "no frozen manifest"):
            check_saved_plan(state_dir, manifest)
        (state_dir / "manifest.json").write_text(json.dumps(manifest))
        check_saved_plan(state_dir, manifest)
        with self.assertRaisesRegex(RuntimeError, "settings differ"):
            check_saved_plan(state_dir, self.manifest("qwen"))
        state_path.write_text(json.dumps({"schema_version": 1, "jobs": {"foreign": {}}, "gpus": {}}))
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            check_saved_plan(state_dir, manifest)

    def test_check_only_does_not_start_queue_or_create_state(self):
        state = self.root / "not-created"
        with patch("scripts.run_wmt23_gap_variants.preflight") as check, \
             patch.object(WMT23GapQueue, "run") as run, contextlib.redirect_stdout(io.StringIO()):
            main(self.arguments("qwen", "--check-only", "--state-dir", str(state)))
        check.assert_called_once()
        run.assert_not_called()
        self.assertFalse(state.exists())

    def test_metadata_only_incomplete_run_cannot_silently_restart(self):
        queue = self.queue()
        job = queue.manifest["jobs"][0]
        job["results_root"] = str(self.root / "results")
        run = Path(job["results_root"]) / job["model"].replace("/", "__") / "interrupted"
        run.mkdir(parents=True)
        (run / "run_metadata.json").write_text(json.dumps({
            "status": "training", "experiment_config": job["config"],
        }))
        with patch("scripts.run_remote_experiments.RemoteQueue.train_job") as train, \
             self.assertRaisesRegex(RuntimeError, "incomplete matching run"):
            queue.train_job(0, job)
        train.assert_not_called()

    def test_dead_previous_launch_without_metadata_cannot_silently_restart(self):
        queue = self.queue()
        job = queue.manifest["jobs"][0]
        job["results_root"] = str(self.root / "results")
        queue.state["jobs"][job["id"]] = {"training_status": "running"}
        with patch("scripts.run_remote_experiments.RemoteQueue.train_job") as train, \
             self.assertRaisesRegex(RuntimeError, "no verified final adapter"):
            queue.train_job(0, job)
        train.assert_not_called()


if __name__ == "__main__":
    unittest.main()
