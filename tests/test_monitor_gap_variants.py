"""Meaningful CPU diagnostics checks; no model, process, network, or GPU use."""
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import monitor_gap_variants as monitor


class GapMonitorMetricsTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.x = rng.normal(size=(8, 5))
        self.y = self.x + rng.normal(scale=.35, size=(8, 5))
        self.d = np.linalg.norm(self.x[:, None]-self.y[None], axis=-1)

    def test_rms_loss_and_ranking_are_scale_invariant_above_floor(self):
        original = monitor.batch_metrics(self.x, self.y, "gap_distance_rms")
        for scale in (.001, .5, 2., 100.):
            scaled = monitor.batch_metrics(scale*self.x, scale*self.y, "gap_distance_rms")
            self.assertAlmostEqual(scaled["selected_loss"], original["selected_loss"], places=12)
            self.assertEqual(scaled["native_batch_top1_pct"], original["native_batch_top1_pct"])
            self.assertAlmostEqual(scaled["normalized_shell_margin"], original["normalized_shell_margin"], places=12)
            self.assertEqual(scaled["selected_backward_radial"], 0.)
            self.assertLess(abs(scaled["selected_reforward_radial"]), 1e-8)
            self.assertFalse(scaled["rms_floor_active"])

    def test_original_radial_matches_reforward_and_detach_only_changes_backward(self):
        d = np.abs(np.array([0., 2.])[:, None]-np.array([4., 1.])[None, :])
        original = monitor.radial_diagnostics(d, "gap_distance_infonce", temperature=1.)
        detached = monitor.radial_diagnostics(d, "gap_distance_detach", temperature=1.)
        self.assertEqual(original[0], detached[0])
        self.assertAlmostEqual(original[1], original[2], places=8)
        self.assertEqual(original[2], detached[2])
        self.assertGreater(abs(original[1]-detached[1]), .1)

    def test_native_ranking_uses_shell_distance_not_nearest_euclidean(self):
        d = np.array([[2., .1], [4., 2.]])
        z, _, _ = monitor.logits_and_loss(d, "gap_distance_infonce")
        self.assertEqual(d[0].argmin(), 1)
        self.assertEqual(z[0].argmax(), 0)
        self.assertEqual(monitor.tie_aware_top1(z)[0], 1.)

    def test_full_collapse_gives_chance_credit_not_arbitrary_diagonal_ties(self):
        metrics = monitor.batch_metrics(np.zeros((4, 3)), np.zeros((4, 3)), "gap_distance_rms")
        self.assertAlmostEqual(metrics["selected_loss"], np.log(4))
        self.assertEqual(metrics["native_batch_top1_pct"], 25.)
        self.assertEqual(metrics["tied_query_fraction"], 1.)
        self.assertEqual(metrics["maximum_tie_count"], 4)
        self.assertTrue(metrics["rms_floor_active"])
        self.assertEqual(metrics["normalized_shell_margin"], 0.)
        self.assertTrue(np.isfinite(list(metrics.values())).all())

    def test_rms_floor_breaks_exact_scale_invariance(self):
        x, y = self.x*1e-8, self.y*1e-8
        metrics = monitor.batch_metrics(x, y, "gap_distance_rms")
        self.assertTrue(metrics["rms_floor_active"])
        self.assertNotEqual(metrics["selected_backward_radial"], 0.)
        self.assertAlmostEqual(metrics["selected_backward_radial"], metrics["selected_reforward_radial"], places=8)

    def test_nonfinite_embeddings_are_counted_without_fake_scores(self):
        self.x[0, 0] = np.nan
        self.y[2, 1] = np.inf
        self.assertEqual(monitor.batch_metrics(self.x, self.y, "gap_distance_rms"), {"nonfinite_elements": 2})
        for value in (np.inf, np.nan, -1., 0.):
            with self.assertRaises(ValueError):
                monitor.batch_metrics(self.x, self.y, "gap_distance_rms", scale=value)


class GapMonitorArtifactTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.state = self.root / "state"
        self.state.mkdir()
        self.out = self.root / "out"
        self.run = self.root / "results" / "Qwen__fixture" / "training-run"
        self.config = dict(alignment_loss="gap_distance_rms", alignment_temperature=.05,
                           alignment_gap_scale=1., model_name="Qwen/fixture",
                           alignment_hidden_state_layer=-1, training_seed=42)
        self.job = dict(id="new_rms", lane="qwen", model="Qwen/fixture", config=self.config)
        self.manifest = dict(jobs=[self.job], results_root=str(self.root/"results"))
        (self.state/"manifest.json").write_text(json.dumps(self.manifest))
        (self.state/"state.json").write_text(json.dumps({"status": "running", "jobs": {
            "new_rms": {"training_status": "running", "log": str(self.root/"training-run.log")}}}))

    def create_dump(self):
        folder = self.run / "eval_samples"
        folder.mkdir(parents=True)
        samples, embeddings = {}, {}
        for pair in monitor.PAIRS:
            records = []
            for i in range(64):
                keys = {f"{side}_embedding_key": f"{pair}/{side}/{i}" for side in ("source", "target")}
                for offset, side in enumerate(("source", "target")):
                    embeddings[keys[f"{side}_embedding_key"]] = np.array([i, offset+.1, i % 7], dtype=np.float32)
                records.append(dict(sample_id=f"{pair}/{i}", batch_id=f"{pair}/batch-{i//16}",
                                    actual_batch_size=16, batch_position=i % 16,
                                    alignment_loss_type="gap_distance_rms", embedding_keys=keys,
                                    source_token_hash=f"s{i}", target_token_hash=f"t{i}"))
            samples["alignment/"+pair] = records
        path = folder/"step-2500.json"
        path.write_text(json.dumps(samples))
        with path.with_name("step-2500_embeddings.pkl").open("wb") as handle:
            pickle.dump(embeddings, handle)
        return path, samples, embeddings

    def test_new_run_is_found_from_log_before_state_run_is_set(self):
        state = monitor.read_json(self.state/"state.json")
        runs, pending = monitor.discover_runs(self.manifest, state, [])
        self.assertEqual(runs[0]["run"], str(self.run))
        self.assertEqual(pending, [])
        self.manifest["jobs"].append({"id": "adopted", "model": "irrelevant"})
        self.assertEqual(len(monitor.discover_runs(self.manifest, state, [])[0]), 1)

    def test_batches_recover_original_order_and_skip_partial_candidates(self):
        _, samples, embeddings = self.create_dump()
        records = samples["alignment/en-ko"]
        reversed_batch = list(reversed(records[:16]))
        batches, skipped = monitor.complete_batches(reversed_batch, embeddings)
        self.assertEqual(len(batches), 1)
        self.assertEqual(skipped, [])
        np.testing.assert_array_equal(batches[0][3][:, 0], np.arange(16))
        self.assertEqual(monitor.complete_batches(records[:16], embeddings)[0][0][1], batches[0][1])
        batches, skipped = monitor.complete_batches(records[:15]+records[16:32], embeddings)
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0][0], "en-ko/batch-1")
        self.assertEqual(len(skipped), 1)
        changed = [dict(r) for r in records[:16]]
        changed[0]["source_token_hash"] = "different-token-input"
        self.assertNotEqual(monitor.complete_batches(changed, embeddings)[0][0][1],
                            monitor.complete_batches(records[:16], embeddings)[0][0][1])

    def test_mismatched_recorded_batch_ids_are_rejected(self):
        _, samples, embeddings = self.create_dump()
        records = samples["alignment/en-ko"][:16]
        records[0]["batch_sample_ids"] = ["wrong"]*16
        with self.assertRaisesRegex(ValueError, "Candidate identity mismatch"):
            monitor.complete_batches(records, embeddings)

    def test_incremental_cache_and_late_validation_metrics(self):
        path, _, _ = self.create_dump()
        with patch.object(monitor, "plot"):
            first, changed = monitor.scan_once(self.state, self.out)
            self.assertTrue(changed)
            self.assertEqual(first["completed_step_count"], 1)
            self.assertEqual(first["latest"][0]["batch_count"], 12)
            self.assertEqual(first["latest"][0]["sampled_pair_count"], 192)
            self.assertTrue(first["completed_steps_share_sample_and_token_ids_by_model"]["Qwen/fixture"])
            before = (self.out/"latest.csv").stat().st_mtime_ns
            with patch.object(monitor, "batch_metrics", side_effect=AssertionError("Should reuse cached values")):
                _, changed = monitor.scan_once(self.state, self.out)
            self.assertFalse(changed)
            self.assertEqual((self.out/"latest.csv").stat().st_mtime_ns, before)
            heartbeat = monitor.read_json(self.out/"status.json")
            self.assertIn("last_polled_at", heartbeat)
            self.assertEqual(heartbeat["updated_at"], first["updated_at"])
            path.with_name("step-2500_metrics.json").write_text(json.dumps({
                "eval_massive_in_"+language: {"selected_loss_mean": i+1.}
                for i, language in enumerate(("en", "ko", "ja", "es"))}))
            latest, changed = monitor.scan_once(self.state, self.out)
            self.assertTrue(changed)
            self.assertEqual(latest["latest"][0]["downstream_validation_loss"], 2.5)

    def test_transient_incomplete_pickle_is_retried_and_not_cached(self):
        path, _, _ = self.create_dump()
        embedding_path = path.with_name("step-2500_embeddings.pkl")
        payload = embedding_path.read_bytes()
        embedding_path.write_bytes(payload[:10])
        with patch.object(monitor, "plot"):
            status, changed = monitor.scan_once(self.state, self.out)
            self.assertFalse(changed)
            self.assertEqual(status["completed_step_count"], 0)
            self.assertEqual(len(status["retry"]), 1)
            self.assertEqual(monitor.read_json(self.out/"cache.json")["steps"], {})
            embedding_path.write_bytes(payload)
            status, changed = monitor.scan_once(self.state, self.out)
            self.assertTrue(changed)
            self.assertEqual(status["completed_step_count"], 1)
            self.assertEqual(status["retry"], [])

    def test_missing_figures_are_regenerated_without_recomputing_cached_metrics(self):
        self.create_dump()
        with patch.object(monitor, "plot"):
            monitor.scan_once(self.state, self.out)
        with patch.object(monitor, "batch_metrics", side_effect=AssertionError("Metrics are already cached")), \
                patch.object(monitor, "plot") as plot:
            _, changed = monitor.scan_once(self.state, self.out)
        self.assertFalse(changed)
        plot.assert_called_once()

    def test_watch_refreshes_then_exits_when_queue_has_finished(self):
        with patch("sys.argv", ["monitor", "--watch"]), \
                patch.object(monitor, "scan_once", return_value=({"queue_status": "completed"}, False)) as scan, \
                patch.object(monitor.time, "sleep", side_effect=AssertionError("Finished queue must not poll")):
            monitor.main()
            scan.assert_called_once()


if __name__ == "__main__":
    unittest.main()
