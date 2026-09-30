"""Aggregation and observation provenance, without model downloads."""

import json
from pathlib import Path
import tempfile
import unittest

import torch

from alignment_logging import AlignmentStatistics, alignment_records, append_jsonl, batch_record


def observation(distances, pairs=None, ids=None, loss_type="gap_consistency"):
    distances = torch.tensor(distances, dtype=torch.float32, requires_grad=True)
    n = len(distances)
    batch = {
        "lang_pair": pairs or ["en-ko"] * n,
        "sample_id": ids or [f"row-{i}" for i in range(n)],
        "source_text": ["source"] * n, "target_text": ["target"] * n,
        "source_input_ids": torch.tensor([[2, 3, 0]] * n),
        "target_input_ids": torch.tensor([[4, 5, 0]] * n),
        "source_attention_mask": torch.tensor([[1, 1, 0]] * n),
        "target_attention_mask": torch.tensor([[1, 1, 0]] * n),
    }
    losses = (distances - distances.mean()).square()
    outputs = {
        "gap_distance": distances, "gap_distance_mean": distances.mean(),
        "per_sample_loss": losses, "loss": losses.mean(),
        "source_norm": torch.ones(n), "target_norm": torch.ones(n) * 2,
        "positive_cosine": torch.ones(n) * 0.5,
        "alignment_loss_type": loss_type,
    }
    return batch, outputs


class AlignmentLoggingTests(unittest.TestCase):
    def test_corpus_variance_is_partition_independent_and_not_batch_variance(self):
        stats = AlignmentStatistics()
        for distances in ([1, 1], [5, 5]):
            stats.update(*observation(distances))
        summary = stats.summary()["en-ko"]
        self.assertEqual(summary["num_examples"], 4)
        self.assertEqual(summary["batch_gap_loss_mean"], 0)
        self.assertEqual(summary["corpus_gap_distance_variance"], 4)
        self.assertEqual(summary["gap_distance_p50"], 3)
        repartitioned = AlignmentStatistics()
        for distances in ([1, 5], [1, 5]):
            repartitioned.update(*observation(distances))
        other = repartitioned.summary()["en-ko"]
        self.assertEqual(other["batch_gap_loss_mean"], 4)
        self.assertEqual(summary["corpus_gap_distance_variance"], other["corpus_gap_distance_variance"])

    def test_actual_sample_count_weights_and_training_keeps_no_distances(self):
        stats = AlignmentStatistics(retain_distances=False)
        stats.update(*observation([1, 1]))
        stats.update(*observation([5, 5, 5]))
        summary = stats.summary(corpus=False)["en-ko"]
        self.assertEqual(summary["gap_distance_mean"], 3.4)
        self.assertEqual(stats.groups["en-ko"]["distances"], [])
        self.assertNotIn("corpus_gap_distance_variance", summary)

    def test_mixed_infonce_diagnostics_keep_pair_means_separate(self):
        batch, outputs = observation([1, 9, 3, 9], ["en-ko", "en-ja", "en-ko", "en-ja"], loss_type="infonce")
        stats = AlignmentStatistics()
        stats.update(batch, outputs)
        summary = stats.summary()
        self.assertEqual(summary["en-ko"]["batch_gap_loss_mean"], 1)
        self.assertEqual(summary["en-ja"]["batch_gap_loss_mean"], 0)
        rows = alignment_records(batch, outputs, [0, 1], {"batch_id": "run/batch-1"})
        self.assertEqual([r["batch_gap_distance_mean"] for r in rows], [2, 9])

    def test_records_link_to_full_batch_and_hash_effective_tokens(self):
        batch, outputs = observation([1, 3, 5])
        context = {"batch_id": "session/step-4/micro-2", "global_step": 4}
        rng_before = torch.random.get_rng_state().clone()
        rows = alignment_records(batch, outputs, [1], context)
        whole = batch_record(batch, outputs, context)
        self.assertEqual(whole["sample_ids"], ["row-0", "row-1", "row-2"])
        self.assertEqual(rows[0]["record_id"], context["batch_id"] + "/sample-1")
        self.assertEqual(rows[0]["source_num_tokens"], 2)
        batch["source_input_ids"][1, 2] = 999
        changed_padding = alignment_records(batch, outputs, [1], context)
        self.assertEqual(rows[0]["source_token_hash"], changed_padding[0]["source_token_hash"])
        self.assertTrue(torch.equal(rng_before, torch.random.get_rng_state()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "observations.jsonl"
            append_jsonl(path, rows)
            self.assertEqual(json.loads(path.read_text()), rows[0])

    def test_gap_records_preserve_forward_precision_reference(self):
        # FP32 mean rounds to 1e8: actual loss 32, whereas FP64 corpus var is 16.
        batch, outputs = observation([1e8, 1e8 + 8])
        self.assertEqual(outputs["loss"].item(), 32)
        stats = AlignmentStatistics()
        stats.update(batch, outputs)
        summary = stats.summary()["en-ko"]
        self.assertEqual(summary["batch_gap_loss_mean"], 32)
        self.assertEqual(summary["corpus_gap_distance_variance"], 16)
        rows = alignment_records(batch, outputs, [0, 1], {"batch_id": "precision"})
        self.assertEqual([r["gap_squared_residual"] for r in rows], [0, 64])
        self.assertEqual(rows[0]["batch_gap_distance_mean"], outputs["gap_distance_mean"].item())


if __name__ == "__main__":
    unittest.main()
