import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.run_remote_experiments import RemoteQueue, build_manifest
from scripts.wmt23_pipeline import file_sha256


class RemoteExperimentsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        pairs = ("de-en", "cs-en", "en-ja")
        data = dict(corpus_profile="alma_ja_opus", data_seed=42,
                    counts={"test": {direction: 10 for lang in ("de", "cs", "ja")
                                     for direction in (f"en-{lang}", f"{lang}-en")}},
                    opus_files={pair: {"test": f"{pair}.parquet"} for pair in pairs},
                    files={f"{pair}.parquet": {"sha256": "0" * 64} for pair in pairs})
        path = self.root / "manifest.json"
        path.write_text(json.dumps(data))
        self.args = argparse.Namespace(data_dir=self.root, manifest_sha256=file_sha256(path),
                                       comet_python=self.root / "comet/bin/python", gpu=0, wandb_mode="disabled")

    def test_portable_plan_keeps_order_settings_and_scopes(self):
        with patch.dict("os.environ", {"ALIGNMENT_HIDDEN_STATE_LAYER": "8", "TRAINING_SEED": "44",
                                       "DOWNSTREAM_MICRO_BATCH_SIZE": "2", "WANDB_MODE": "online"}):
            manifest = build_manifest(self.args)
        jobs = manifest["jobs"]
        self.assertEqual([j["id"] for j in jobs], ["massive_qwen_infonce_then_sft",
                                                  "mt_llama_gap_then_sft", "mt_qwen_gap_then_sft"])
        self.assertEqual([j["config"]["downstream_micro_batch_size"] for j in jobs], [0, 0, 8])
        for job in jobs:
            config = job["config"]
            self.assertEqual((config["alignment_hidden_state_layer"], config["training_seed"],
                              config["quantization_compute_dtype"], config["num_steps"]),
                             (-1, 42, "bfloat16", 100000))
            self.assertEqual(config["wandb_mode"], "disabled")
        self.assertEqual(manifest["massive_post_training_evaluation"]["alignment_language_scopes"], ["in"])
        self.assertEqual(manifest["massive_post_training_evaluation"]["language_scopes"], ["in", "out"])
        self.assertEqual(manifest["evaluation_language_scopes"], ["in"])
        self.assertEqual(manifest, build_manifest(self.args))

    def test_rejects_different_data_before_launch(self):
        self.args.manifest_sha256 = "f" * 64
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            build_manifest(self.args)

    def queue(self):
        return RemoteQueue(build_manifest(self.args), self.root / "state")

    def test_each_final_evaluation_precedes_the_next_training(self):
        queue = self.queue()
        events = []
        with patch.object(queue, "reserve_gpu"), patch.object(queue, "release_gpus"), \
             patch.object(queue, "train_job", side_effect=lambda gpu, job: events.append((job["id"], "train"))), \
             patch.object(queue, "evaluate_massive_job", side_effect=lambda gpu, job: events.append((job["id"], "massive"))), \
             patch.object(queue, "retrieve_job", side_effect=lambda gpu, job: events.append((job["id"], "retrieval"))), \
             patch.object(queue, "evaluate", side_effect=lambda gpu, jobs: events.append((jobs[0]["id"], "mt"))):
            queue.run()
        self.assertEqual(events, [
            ("massive_qwen_infonce_then_sft", "train"), ("massive_qwen_infonce_then_sft", "massive"),
            ("mt_llama_gap_then_sft", "train"), ("mt_llama_gap_then_sft", "retrieval"), ("mt_llama_gap_then_sft", "mt"),
            ("mt_qwen_gap_then_sft", "train"), ("mt_qwen_gap_then_sft", "retrieval"), ("mt_qwen_gap_then_sft", "mt"),
        ])

    def test_evaluation_failure_stops_later_training(self):
        queue = self.queue()
        with patch.object(queue, "reserve_gpu"), patch.object(queue, "release_gpus"), \
             patch.object(queue, "train_job") as train, \
             patch.object(queue, "evaluate_massive_job", side_effect=RuntimeError("evaluation failed")):
            with self.assertRaisesRegex(RuntimeError, "evaluation failed"):
                queue.run()
        self.assertEqual(train.call_count, 1)
        self.assertEqual(queue.state["status"], "failed")

    def test_incomplete_training_is_not_silently_restarted(self):
        queue = self.queue()
        job = queue.manifest["jobs"][0]
        job["results_root"] = str(self.root / "results")
        run = Path(job["results_root"]) / job["model"].replace("/", "__") / "interrupted"
        run.mkdir(parents=True)
        (run / "experiment_config.json").write_text(json.dumps(job["config"]))
        with patch("scripts.wmt23_pipeline.Pipeline.train_job") as train:
            with self.assertRaisesRegex(RuntimeError, "incomplete matching run"):
                queue.train_job(0, job)
        train.assert_not_called()


if __name__ == "__main__":
    unittest.main()
