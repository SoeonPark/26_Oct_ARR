"""CPU regression checks for retrieval ranks and deterministic exact ties."""

import unittest

import torch
from torch.nn import functional as F

from evaluate import evaluate_alignment_retrieval, evaluate_retrieval_direction


class RetrievalTests(unittest.TestCase):
    def assert_metrics(self, metrics, r1, r5, mrr):
        self.assertAlmostEqual(metrics["recall_at_1"], r1, places=6)
        self.assertAlmostEqual(metrics["recall_at_5"], r5, places=6)
        self.assertAlmostEqual(metrics["mrr"], mrr, places=6)

    def test_identical_embeddings_do_not_receive_perfect_scores(self):
        for size in (4, 7):
            embeddings = torch.ones(size, 3)
            for chunk_size in (1, 2, 3, size + 1):
                with self.subTest(size=size, chunk_size=chunk_size):
                    metrics = evaluate_retrieval_direction(
                        embeddings, embeddings, chunk_size, "cpu",
                    )
                    self.assertEqual(metrics["num_queries"], size)
                    self.assert_metrics(
                        metrics, 1 / size, min(5, size) / size,
                        sum(1 / rank for rank in range(1, size + 1)) / size,
                    )

    def test_duplicate_pairs_have_ranks_one_two_one_two(self):
        embeddings = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])
        for chunk_size in (1, 2, 3, 4):
            metrics = evaluate_retrieval_direction(
                embeddings, embeddings, chunk_size, "cpu",
            )
            self.assert_metrics(metrics, 0.5, 1.0, 0.75)

    def test_unique_top_matches_still_receive_perfect_scores(self):
        embeddings = torch.eye(7)
        metrics = evaluate_retrieval_direction(embeddings, embeddings, 3, "cpu")
        self.assert_metrics(metrics, 1.0, 1.0, 1.0)

    def test_ranks_match_explicit_sort_with_and_without_ties(self):
        generator = torch.Generator().manual_seed(17)
        queries = torch.randn(9, 5, generator=generator)
        candidates = torch.randn(9, 5, generator=generator)
        for duplicate in (False, True):
            if duplicate:
                candidates[1] = candidates[0]
                candidates[4] = candidates[0]
                candidates[8] = candidates[7]
            for chunk_size in (1, 2, 4, 9):
                with self.subTest(duplicate=duplicate, chunk_size=chunk_size):
                    # Use the same arithmetic chunks: float32 matmul kernels
                    # can round differently at different matrix sizes.
                    scores = torch.cat([
                        F.normalize(queries[start:start + chunk_size], dim=-1)
                        @ F.normalize(candidates, dim=-1).T
                        for start in range(0, len(queries), chunk_size)
                    ])
                    # Independent ranking reference: explicitly sort every
                    # candidate by score/index, then locate the gold.
                    ranks = []
                    for index, row in enumerate(scores.tolist()):
                        ordered = sorted(range(len(row)), key=lambda j: (-row[j], j))
                        ranks.append(ordered.index(index) + 1)
                    metrics = evaluate_retrieval_direction(
                        queries, candidates, chunk_size, "cpu",
                    )
                    self.assert_metrics(
                        metrics,
                        sum(rank <= 1 for rank in ranks) / len(ranks),
                        sum(rank <= 5 for rank in ranks) / len(ranks),
                        sum(1 / rank for rank in ranks) / len(ranks),
                    )

    def test_both_directions_and_aggregates_include_the_policy(self):
        groups = {
            "en-ko": {"source": torch.ones(4, 3), "target": torch.ones(4, 3)},
            "en-ja": {"source": torch.eye(2), "target": torch.eye(2)},
        }
        metrics = evaluate_alignment_retrieval(groups, chunk_size=3, device="cpu")
        self.assertEqual(metrics["tie_break"], "candidate_index_ascending")
        tied_mrr = 25 / 48
        for direction in ("source_to_target", "target_to_source", "bidirectional_average"):
            self.assert_metrics(metrics["pairs"]["en-ko"][direction], 0.25, 1., tied_mrr)
            self.assert_metrics(metrics["pairs"]["en-ja"][direction], 1., 1., 1.)
        self.assert_metrics(metrics["language_pair_macro"], 0.625, 1., (tied_mrr + 1) / 2)
        self.assert_metrics(metrics["query_micro"], 0.5, 1., (8 * tied_mrr + 4) / 12)


if __name__ == "__main__":
    unittest.main()
