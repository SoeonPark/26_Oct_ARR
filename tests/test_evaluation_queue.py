"""Queue control-flow tests: no GPUs, real training, or network requests."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import evaluation_queue as queue


class EvaluationQueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.results = self.root / "results"
        self.results.mkdir()
        self.args = SimpleNamespace(
            results_root=self.results, queue_dir=self.root / "queue-test",
            gpu=0, no_notify=True, quiet_seconds=0, poll_seconds=0.01,
        )
        self.args.queue_dir.mkdir()

    def make_run(self, name, method="alternative", **changes):
        run = self.results / "model" / name
        run.mkdir(parents=True)
        expected = queue.STEP_BUDGETS[method]
        queue.write_json(run / "run_metadata.json", {"status": changes.get("status", "completed")})
        queue.write_json(run / "experiment_config.json", {
            "training_type": method, "num_steps": changes.get("planned", expected),
            "checkpoint_global_step": changes.get("saved_step", expected),
        })
        queue.write_json(run / "trainer_state.json", {"global_step": changes.get("step", expected)})
        queue.write_json(run / "train_results.json", {})
        queue.write_json(run / "adapter_config.json", {})
        (run / "adapter_model.safetensors").write_bytes(b"test-only-weight-marker")
        return run

    def test_completion_requires_the_full_budget_and_final_artifacts(self):
        for method, expected in queue.STEP_BUDGETS.items():
            run = self.make_run(method, method)
            entry, reason = queue.inspect_run(run, self.results)
            self.assertIsNone(reason)
            self.assertEqual(entry["global_step"], expected)
        for index, changes in enumerate((
            {"status": "training"}, {"status": "failed"}, {"step": 99999},
            {"planned": 2000}, {"saved_step": 99500},
        )):
            with self.subTest(changes=changes):
                run = self.make_run(f"bad-{index}", **changes)
                self.assertIsNone(queue.inspect_run(run, self.results)[0])
        run = self.make_run("empty-adapter")
        (run / "adapter_model.safetensors").write_bytes(b"")
        self.assertIsNone(queue.inspect_run(run, self.results)[0])
        run = self.make_run("malformed-json")
        (run / "run_metadata.json").write_text("{")
        self.assertIsNone(queue.inspect_run(run, self.results)[0])

    def test_every_completed_run_is_assigned_exactly_once(self):
        for name in ("run-e", "run-a", "run-d", "run-b", "run-c"):
            self.make_run(name)
        self.make_run("unfinished", status="training")
        report = queue.scan_runs(self.results)
        assigned = [[r["run"] for r in report["runs"] if r["gpu"] == gpu] for gpu in (0, 1)]
        self.assertEqual([len(paths) for paths in assigned], [3, 2])
        self.assertFalse(set(assigned[0]) & set(assigned[1]))
        self.assertEqual(len(set(assigned[0] + assigned[1])), 5)
        self.assertEqual(len(report["excluded"]), 1)

    def test_manifest_rescans_after_preview_then_stays_shared(self):
        self.make_run("run-a")
        queue.save_plan(self.args, queue.scan_runs(self.results), "preview_manifest.json")
        self.make_run("run-b")  # A training job completes while workers wait.
        first = queue.load_manifest(self.args)
        self.assertEqual(len(first["runs"]), 2)
        self.assertEqual(queue.load_manifest(self.args), first)
        self.assertEqual((self.args.queue_dir / "runs_gpu0.txt").read_text(), "model/run-a\n")
        self.assertEqual((self.args.queue_dir / "runs_gpu1.txt").read_text(), "model/run-b\n")

    def test_shell_queues_are_detected_between_python_children(self):
        proc = self.root / "proc"
        proc.mkdir()
        for pid, cwd, argv in (
            (100, queue.PROJECT_ROOT, ["bash", "scripts/transfer_only.sh"]),
            (101, queue.PROJECT_ROOT, ["python", "-u", "main.py"]),
            (102, self.root, ["bash", "scripts/transfer_only.sh"]),
            (103, queue.PROJECT_ROOT, ["python", "evaluate.py"]),
        ):
            path = proc / str(pid)
            path.mkdir()
            fields = ["S", "1"] + ["0"] * 17 + [str(pid * 10)]
            (path / "stat").write_text(f"{pid} (process) " + " ".join(fields))
            (path / "cwd").symlink_to(cwd, target_is_directory=True)
            (path / "cmdline").write_bytes("\0".join(argv).encode() + b"\0")
        self.assertEqual([p["pid"] for p in queue.training_processes(proc_root=proc)], [100, 101])

    def test_idle_wait_requires_no_training_and_no_gpu_processes(self):
        shell = [{"pid": 100, "script": "transfer_only.sh"}]
        with patch.object(queue, "training_processes", side_effect=[shell, [], [], []]), \
             patch.object(queue, "training_lock_busy", return_value=False), \
             patch.object(queue, "gpu_processes", side_effect=[[], [123], RuntimeError("NVML unavailable"), []]), \
             patch.object(queue.time, "sleep") as sleep, redirect_stdout(io.StringIO()):
            queue.wait_until_idle(self.args, {})
        self.assertEqual(sleep.call_count, 3)

    def test_priorities_are_all_pinned_to_gpu1_before_other_runs(self):
        priorities = [f"model/priority-{index}" for index in range(4)]
        for name in [p.split("/")[1] for p in priorities] + [f"other-{i}" for i in range(6)]:
            self.make_run(name)
        report = queue.scan_runs(self.results, priority_runs=priorities, priority_gpu=1)
        self.assertEqual([r["run"] for r in report["runs"][:4]], priorities)
        self.assertTrue(all(r["gpu"] == 1 for r in report["runs"][:4]))
        self.assertEqual([sum(r["gpu"] == gpu for r in report["runs"]) for gpu in (0, 1)], [5, 5])

    def test_refresh_keeps_assignments_and_adds_late_completions_once(self):
        self.args.refresh_manifest = True
        self.make_run("b")
        self.make_run("c")
        first = queue.load_manifest(self.args)
        assignments = {r["run"]: r["gpu"] for r in first["runs"]}
        self.make_run("a-late")  # Sorting earlier must not change prior owners.
        second = queue.load_manifest(self.args)
        self.assertEqual(len(second["runs"]), 3)
        for entry in second["runs"][:2]:
            self.assertEqual(entry["gpu"], assignments[entry["run"]])
        self.assertEqual(len(queue.load_manifest(self.args)["runs"]), 3)

    def test_gpu_wait_ignores_other_gpu_but_waits_for_own_training(self):
        self.args.wait_scope = "gpu"
        self.args.gpu = 1
        other = {"pid": 100, "gpu_ids": [0]}
        own = {"pid": 101, "gpu_ids": [1]}
        with patch.object(queue, "training_processes", side_effect=[[other, own], [other], [other]]), \
             patch.object(queue, "training_lock_busy", side_effect=[True, True, False]), \
             patch.object(queue, "gpu_processes", return_value=[]), \
             patch.object(queue.time, "sleep") as sleep, redirect_stdout(io.StringIO()):
            queue.wait_until_idle(self.args, {})
        self.assertEqual(sleep.call_count, 2)

    def test_gpu_detection_uses_environment_or_inherited_training_lock(self):
        process = self.root / "proc" / "123"
        (process / "fd").mkdir(parents=True)
        (process / "environ").write_bytes(b"CUDA_VISIBLE_DEVICES=1\0")
        self.assertEqual(queue.process_gpu_ids(process, self.root), [1])
        (process / "environ").write_bytes(b"UNRELATED=value\0")
        (process / "fd" / "9").symlink_to(self.root / "logs" / "train-gpu-0.lock")
        self.assertEqual(queue.process_gpu_ids(process, self.root), [0])

    def test_completion_date_filter_uses_kst_and_not_run_name(self):
        for name, completed in (("run-20260918", "2026-09-18T16:00:00+00:00"),
                                ("old", "2026-09-18T23:59:59+09:00")):
            run = self.make_run(name)
            queue.write_json(run / "run_metadata.json", {"status": "completed", "completed_at": completed})
        report = queue.scan_runs(self.results, completed_since="2026-09-19")
        self.assertEqual([r["run"] for r in report["runs"]], ["model/run-20260918"])

    def test_worker_evaluates_a_run_that_completes_during_the_queue(self):
        first = self.make_run("first")
        late = self.make_run("late", status="training")
        self.args.gpu = 1
        self.args.priority_runs = ["model/first", "model/late"]
        self.args.priority_gpu = 1
        self.args.refresh_manifest = True
        evaluated = []

        def evaluate(args, entry, settings, state, result_path):
            evaluated.append(entry["run"])
            queue.write_json(late / "run_metadata.json", {"status": "completed"})
            return True

        with patch.object(queue, "QUEUE_ROOT", self.root / "locks"), \
             patch.object(queue, "wait_until_idle"), \
             patch.object(queue, "training_pending", return_value=False), \
             patch.object(queue, "evaluate_one", side_effect=evaluate), redirect_stdout(io.StringIO()):
            self.assertEqual(queue.worker(self.args), 0)
        self.assertEqual(evaluated, ["model/first", "model/late"])

    def test_detached_worker_restores_selection_and_gpu_wait_policy(self):
        queue.write_json(self.args.queue_dir / "queue_config.json", {
            "selection": {"completed_since": "2026-09-19", "priority_runs": ["model/first"],
                          "priority_gpu": 1, "skip_evaluated": True},
            "scheduling": {"wait_scope": "gpu", "refresh_manifest": True},
            "settings": queue.EVAL_DEFAULTS,
        })
        argv = ["queue", "worker", "--gpu", "1", "--queue-dir", str(self.args.queue_dir),
                "--results-root", str(self.results)]
        with patch.object(queue.sys, "argv", argv):
            args = queue.parse_args()
        self.assertEqual(args.priority_runs, ["model/first"])
        self.assertEqual(args.priority_gpu, 1)
        self.assertEqual(args.wait_scope, "gpu")
        self.assertTrue(args.refresh_manifest and args.skip_evaluated)
        self.assertEqual(args.settings, queue.EVAL_DEFAULTS)

    def test_skip_evaluated_requires_complete_matching_metrics(self):
        run = self.make_run("already-evaluated")
        entry, _ = queue.inspect_run(run, self.results)
        p = run / "evaluations" / "previous" / "test"
        metadata = {
            "status": "completed", "split": "test", "checkpoint_path": str(run),
            "experiment_config": {"checkpoint_global_step": entry["global_step"]},
            "language_scopes": ["in", "out"], "tasks": ["alignment", "massive"],
            "alignment_batch_size": 16, "massive_batch_size": 16,
            "retrieval_chunk_size": 256, "max_new_tokens": 128,
            "eval_sample_log_limit": 64, "save_alignment_embeddings": False,
            "save_alignment_sample_metrics": True, "results": {},
        }
        for scope in ("in", "out"):
            metadata["results"][scope] = {}
            for task in ("alignment", "massive"):
                metric = {"value": 0.5}
                metadata["results"][scope][task] = metric
                queue.write_json(p / scope / f"{task}_metrics.json", metric)
        queue.write_json(p / "evaluation_metadata.json", metadata)
        self.assertEqual(queue.scan_runs(self.results, skip_evaluated=True)["runs"], [])
        (p / "out" / "massive_metrics.json").unlink()
        self.assertEqual(len(queue.scan_runs(self.results, skip_evaluated=True)["runs"]), 1)

    def test_gpu_visibility_is_one_device_and_outputs_are_separate(self):
        entry, _ = queue.inspect_run(self.make_run("run"), self.results)
        for gpu in (0, 1):
            self.args.gpu = gpu
            env = queue.run_environment(self.args, entry, queue.EVAL_DEFAULTS)
            self.assertEqual(env["CUDA_VISIBLE_DEVICES"], str(gpu))
            self.assertEqual(Path(env["EVAL_OUTPUT_DIR"]).name, self.args.queue_dir.name)
            self.assertNotIn("WORLD_SIZE", env)

    def test_gpu_lock_rejects_a_second_worker(self):
        path = self.root / "gpu0.lock"
        with queue.lock_file(path, blocking=False):
            with self.assertRaises(BlockingIOError):
                with queue.lock_file(path, blocking=False):
                    pass

    def test_real_shell_dispatch_logs_and_continues_after_failed_run(self):
        # Execute the real eval.sh with a tiny stand-in evaluator. Only the
        # Python program is replaced; argument quoting and env handling run.
        project = self.root / "project with spaces"
        (project / "scripts").mkdir(parents=True)
        shutil.copy2(queue.PROJECT_ROOT / "scripts/eval.sh", project / "scripts/eval.sh")
        (project / "models.py").write_text("# test placeholder\n")
        (project / "data_utils.py").write_text("# test placeholder\n")
        (project / "evaluate.py").write_text('''import json, os, sys
from pathlib import Path
args = sys.argv
run = Path(args[args.index("--checkpoint_path") + 1])
print("GPU=" + os.environ["CUDA_VISIBLE_DEVICES"], flush=True)
print("RUN=" + run.name, flush=True)
if run.name == "a-fail": sys.exit(7)
output = Path(args[args.index("--output_dir") + 1]) / "test"
output.mkdir(parents=True, exist_ok=True)
(output / "evaluation_metadata.json").write_text(json.dumps({"status": "completed"}))
''')
        for name in ("a-fail", "b-other-gpu", "c-success with spaces"):
            self.make_run(name)
        with patch.object(queue, "PROJECT_ROOT", project), \
             patch.object(queue, "QUEUE_ROOT", self.root / "locks"), \
             patch.object(queue, "wait_until_idle") as wait, redirect_stdout(io.StringIO()):
            returncode = queue.worker(self.args)
        self.assertEqual(returncode, 1)
        self.assertEqual(wait.call_count, 3)
        state = queue.read_json(self.args.queue_dir / "gpu0_state.json")
        self.assertEqual(state["status"], "finished")
        self.assertEqual(state["failed"], ["model/a-fail"])
        self.assertEqual(state["completed"], ["model/c-success with spaces"])
        outcomes = list((self.args.queue_dir / "outcomes/model").glob("*.json"))
        self.assertEqual(len(outcomes), 2)
        for outcome in outcomes:
            saved = queue.read_json(outcome)
            log = Path(saved["log"]).read_text()
            self.assertIn("GPU=0", log)
            self.assertIn("RUN=", log)


if __name__ == "__main__":
    unittest.main()
