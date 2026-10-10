import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.run_massive_gap_variants import (
    MassiveGapQueue, build_manifest, check_saved_plan, main, parse_args, preflight,
)


class MassiveGapLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def manifest(self, model="llama", *extra):
        return build_manifest(parse_args(["--model", model, *extra]))

    def test_default_order_and_no_mt_prerequisites(self):
        llama = self.manifest()
        qwen = self.manifest("qwen")
        self.assertEqual([job["loss"] for job in llama["jobs"]],
                         ["gap_distance_detach", "gap_distance_rms"])
        self.assertEqual([job["loss"] for job in qwen["jobs"]], ["gap_distance_rms"])
        self.assertNotIn("data_dir", llama)
        self.assertNotIn("comet_python", llama)
        self.assertEqual(llama["adopted"], {})
        self.assertEqual(llama["python"], sys.executable)
        self.assertTrue(all(job["lane"] == "llama" for job in llama["jobs"]))
        self.assertEqual(llama, self.manifest())

    def test_interactive_overrides_cannot_change_experiment(self):
        with patch.dict("os.environ", {"ALIGNMENT_HIDDEN_STATE_LAYER": "8", "TRAINING_SEED": "44",
                                       "TRAIN_EVAL_LANGUAGE_SCOPE": "both", "WANDB_MODE": "online",
                                       "MODEL_NAME": "wrong", "TRAIN_SAMPLE_LOG_INTERVAL": "0",
                                       "DOWNSTREAM_MICRO_BATCH_SIZE": "2", "OUTPUT_ROOT": "/wrong"}):
            manifest = self.manifest()
        for job in manifest["jobs"]:
            config = job["config"]
            self.assertEqual((config["training_seed"], config["alignment_hidden_state_layer"],
                              config["training_type"], config["num_steps"], config["batch_size"]),
                             (42, -1, "alternative", 100000, 16))
            self.assertEqual(config["eval_language_scope"], "in")
            self.assertEqual(config["wandb_mode"], "disabled")
            self.assertEqual(config["train_sample_log_interval"], 1000)
            self.assertEqual(config["downstream_micro_batch_size"], 0)
            self.assertEqual(config["alignment_batching"], "same_pair")
        settings = manifest["massive_post_training_evaluation"]
        self.assertEqual(settings["language_scopes"], ["in", "out"])
        self.assertEqual(settings["alignment_language_scopes"], ["in"])
        self.assertEqual(settings["pair_counts"]["in"]["en-ko"], 2000)

    def test_explicit_loss_and_gpu(self):
        manifest = self.manifest("qwen", "--loss", "both", "--gpu", "3")
        self.assertEqual(len(manifest["jobs"]), 2)
        self.assertTrue(all(job["gpu"] == 3 for job in manifest["jobs"]))
        self.assertEqual(len(self.manifest("llama", "--loss", "rms")["jobs"]), 1)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--model", "llama", "--gpu", "-1"])

    def test_dry_run_needs_no_gpu_imports_and_writes_no_state(self):
        state = self.root / "not-created"
        code = (
            "import sys; from scripts.run_massive_gap_variants import main; "
            "main(['--model','llama','--dry-run','--state-dir',sys.argv[1]]); "
            "assert 'torch' not in sys.modules; assert 'datasets' not in sys.modules; "
            "assert 'comet' not in sys.modules"
        )
        result = subprocess.run([sys.executable, "-c", code, str(state)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("gap_distance_detach", result.stdout)
        self.assertIn("gap_distance_rms", result.stdout)
        self.assertFalse(state.exists())

    def test_preflight_only_checks_selected_model_without_comet(self):
        manifest = self.manifest("qwen")
        with patch("scripts.run_massive_gap_variants.version", return_value="unused"), \
             patch("scripts.run_massive_gap_variants.TRAINING_PACKAGES", set()), \
             patch("scripts.run_massive_gap_variants.subprocess.run") as run:
            preflight(manifest, 2)
        command = run.call_args.args[0]
        self.assertEqual(command[-1], "Qwen/Qwen3.5-2B")
        self.assertNotIn("comet", " ".join(command))
        self.assertEqual(run.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "2")

    def test_each_final_evaluation_precedes_next_training(self):
        queue = MassiveGapQueue(self.manifest(), self.root / "state")
        events = []
        with patch.object(queue, "reserve_gpu"), patch.object(queue, "release_gpus"), \
             patch.object(queue, "train_job", side_effect=lambda gpu, job: events.append((job["loss"], "train"))), \
             patch.object(queue, "evaluate_massive_job", side_effect=lambda gpu, job: events.append((job["loss"], "eval"))), \
             contextlib.redirect_stdout(io.StringIO()):
            queue.run()
        self.assertEqual(events, [("gap_distance_detach", "train"), ("gap_distance_detach", "eval"),
                                  ("gap_distance_rms", "train"), ("gap_distance_rms", "eval")])

    def test_evaluation_failure_stops_next_training(self):
        queue = MassiveGapQueue(self.manifest(), self.root / "state")
        with patch.object(queue, "reserve_gpu"), patch.object(queue, "release_gpus"), \
             patch.object(queue, "train_job") as train, \
             patch.object(queue, "evaluate_massive_job", side_effect=RuntimeError("failed eval")), \
             contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "failed eval"):
            queue.run()
        self.assertEqual(train.call_count, 1)
        self.assertEqual(queue.state["status"], "failed")

    def test_environment_removes_mt_and_interactive_overrides(self):
        queue = MassiveGapQueue(self.manifest(), self.root / "state")
        with patch.dict("os.environ", {"WMT23_DATA_DIR": "/not-needed", "COMET22_PYTHON": "/missing",
                                       "ALIGNMENT_LOSS": "infonce", "WANDB_MODE": "online"}):
            env = queue.environment(2)
        self.assertNotIn("WMT23_DATA_DIR", env)
        self.assertNotIn("COMET22_PYTHON", env)
        self.assertNotIn("ALIGNMENT_LOSS", env)
        self.assertEqual(env["WANDB_MODE"], "disabled")
        self.assertEqual(env["PYTHON_BIN"], sys.executable)

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
        with patch("scripts.run_massive_gap_variants.preflight") as check, \
             patch.object(MassiveGapQueue, "run") as run, contextlib.redirect_stdout(io.StringIO()):
            main(["--model", "qwen", "--check-only", "--state-dir", str(state)])
        check.assert_called_once()
        run.assert_not_called()
        self.assertFalse(state.exists())

    def test_metadata_only_incomplete_run_cannot_silently_restart(self):
        queue = MassiveGapQueue(self.manifest(), self.root / "state")
        job = queue.manifest["jobs"][0]
        job["results_root"] = str(self.root / "results")
        run = Path(job["results_root"]) / job["model"].replace("/", "__") / "interrupted"
        run.mkdir(parents=True)
        (run / "run_metadata.json").write_text(json.dumps({
            "status": "training", "experiment_config": job["config"],
        }))
        with patch("scripts.run_massive_gap_variants.RemoteQueue.train_job") as train, \
             self.assertRaisesRegex(RuntimeError, "incomplete matching run"):
            queue.train_job(0, job)
        train.assert_not_called()

    def test_dead_previous_launch_without_metadata_cannot_silently_restart(self):
        queue = MassiveGapQueue(self.manifest(), self.root / "state")
        job = queue.manifest["jobs"][0]
        job["results_root"] = str(self.root / "results")
        queue.state["jobs"][job["id"]] = {"training_status": "running"}
        with patch("scripts.run_massive_gap_variants.RemoteQueue.train_job") as train, \
             self.assertRaisesRegex(RuntimeError, "no verified final adapter"):
            queue.train_job(0, job)
        train.assert_not_called()


if __name__ == "__main__":
    unittest.main()
