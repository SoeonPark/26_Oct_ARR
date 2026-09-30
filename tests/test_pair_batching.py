"""CPU-only checks; no model downloads, datasets, or experiment runs needed."""

from collections import Counter
from contextlib import redirect_stdout
import io
from itertools import islice
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from accelerate import skip_first_batches
from transformers import TrainingArguments

from config import parse_args
from custom_trainer import AlternativeRoutingTrainer
from data_utils import AlignmentDataset, CombinedDataset
from samplers import AlignmentEvalBatchSampler, PairBatchSampler, shuffled_batches


class ToyTokenizer:
    pad_token_id = 0

    def __call__(self, text, **kwargs):
        return {
            "input_ids": torch.tensor([[int(text) + 1]]),
            "attention_mask": torch.ones(1, 1, dtype=torch.long),
        }


class ToyAlignment(AlignmentDataset):
    def __init__(self, pairs=("en-ko", "en-ja", "en-es")):
        self.config = SimpleNamespace(alignment_data="toy-alignment")
        self.tokenizer = ToyTokenizer()
        self.dataset_metadata = {pair: {"split": "train"} for pair in pairs}
        self.all_data = {
            pair: [
                {"translation": {lang: str(i) for lang in pair.split("-")},
                 "_alignment_row_index": i}
                for i in range(13)
            ]
            for pair in pairs
        }


class ToyDownstream(torch.utils.data.Dataset):
    def __len__(self):
        return 17

    def __getitem__(self, idx):
        return {"input_ids": torch.tensor([idx + 1])}

    def collate_fn(self, batch):
        return {"input_ids": torch.stack([row["input_ids"] for row in batch])}


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
        self.seen = []

    def forward(self, forward_type, return_per_sample=False, **inputs):
        self.seen.append((forward_type, set(inputs)))
        batch = inputs[forward_type]
        key = "source_input_ids" if forward_type == "alignment" else "input_ids"
        return {"loss": self.linear(batch[key].float() / 100).square().mean()}


def planner(mode, steps):
    trainer = AlternativeRoutingTrainer.__new__(AlternativeRoutingTrainer)
    trainer.training_type = mode
    trainer.total_steps = steps
    trainer.schedule = ("alignment", "downstream")
    trainer.alignment_batching = "same_pair"
    # Planning must be independent of the Trainer's live step/prefetch timing.
    trainer.state = SimpleNamespace(global_step=10000)
    return trainer


def make_sampler(mode, steps, accumulation=1, pairs=("en-ko", "en-ja", "en-es"), seed=42):
    return PairBatchSampler(
        pair_ranges=ToyAlignment(pairs).pair_ranges,
        downstream_size=17,
        batch_size=4,
        num_steps=steps,
        accumulation_steps=accumulation,
        seed=seed,
        objective_at=planner(mode, steps).objective_at,
    )


class PairBatchingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def make_trainer(self, tmpdir, mode, accumulation=1, batching="same_pair", workers=0):
        dataset = CombinedDataset(ToyAlignment(), ToyDownstream(), 6 * 4 * accumulation)
        args = TrainingArguments(
            output_dir=tmpdir, use_cpu=True, max_steps=6,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=accumulation,
            save_strategy="no", logging_strategy="no", eval_strategy="no",
            report_to=[], remove_unused_columns=False, disable_tqdm=True,
            dataloader_pin_memory=False, dataloader_num_workers=workers,
            seed=42,
        )
        return AlternativeRoutingTrainer(
            model=ToyModel(), args=args, train_dataset=dataset,
            data_collator=dataset.collate_fn, training_type=mode,
            total_steps=6, alignment_batching=batching, eval_sample_log_limit=0,
        )

    def test_config_defaults_and_option(self):
        with patch("sys.argv", ["main.py"]):
            self.assertEqual(parse_args().alignment_batching, "mixed")
        with patch("sys.argv", ["main.py", "--alignment_batching", "same_pair"]):
            self.assertEqual(parse_args().alignment_batching, "same_pair")

    def test_eval_batches_preserve_all_examples_and_pair_boundaries(self):
        for batch_size in (2, 3, 16, 32):
            for size in range(2, 66):
                with self.subTest(batch_size=batch_size, size=size):
                    sampler = AlignmentEvalBatchSampler(
                        {"en-ko": (0, size), "en-ja": (size, 2 * size)}, batch_size,
                    )
                    batches = list(sampler)
                    self.assertEqual(len(sampler), len(batches))
                    self.assertEqual(batches, list(sampler))
                    self.assertEqual(
                        sorted(i for batch in batches for i in batch),
                        list(range(2 * size)),
                    )
                    for batch in batches:
                        self.assertGreaterEqual(len(batch), 2)
                        self.assertLessEqual(len(batch), batch_size + 1)
                        self.assertEqual(len({i // size for i in batch}), 1)

    def test_eval_singleton_tail_merges_and_short_pairs_fail(self):
        sampler = AlignmentEvalBatchSampler({"en-ko": (0, 9)}, batch_size=4)
        self.assertEqual(list(sampler), [[0, 1, 2, 3], [4, 5, 6, 7, 8]])
        for ranges, batch_size in (({}, 4), ({"en-ko": (0, 1)}, 4),
                                   ({"en-ko": (0, 4)}, 1)):
            with self.subTest(ranges=ranges, batch_size=batch_size):
                with self.assertRaises(ValueError):
                    AlignmentEvalBatchSampler(ranges, batch_size)

    def test_index_ranges_and_dataset_routing(self):
        alignment = ToyAlignment()
        self.assertEqual(alignment.pair_ranges["en-ja"], (13, 26))
        dataset = CombinedDataset(alignment, ToyDownstream(), 100)
        self.assertEqual(set(dataset[20]), {"alignment", "downstream"})
        self.assertEqual(set(dataset[(20, None)]), {"alignment"})
        self.assertEqual(dataset[(20, None)]["alignment"]["lang_pair"], "en-ja")
        self.assertEqual(set(dataset[(None, 5)]), {"downstream"})
        for invalid in [(None, None), (1, 2)]:
            with self.assertRaises(ValueError):
                dataset[invalid]
        validation = CombinedDataset(alignment_dataset=alignment)
        self.assertEqual(set(validation[20]), {"alignment"})

    def test_all_modes_and_accumulation(self):
        for mode, steps in [("transfer_only", 7), ("contrastive_only", 7),
                            ("alternative", 15), ("contrastive_then_transfer", 15)]:
            for accumulation in (1, 3):
                for pairs in (("en-ko", "en-ja"), ("en-ko", "en-ja", "en-es")):
                    with self.subTest(mode=mode, accumulation=accumulation, pairs=pairs):
                        sampler = make_sampler(mode, steps, accumulation, pairs)
                        batches = list(sampler)
                        self.assertEqual(len(batches), steps * accumulation)
                        self.assertEqual(len(batches), len(sampler))
                        self.assertEqual(batches, list(sampler))
                        counts = Counter()
                        for step in range(steps):
                            expected = sampler.objective_at(step)
                            pair_ids = set()
                            for batch in batches[step * accumulation:(step + 1) * accumulation]:
                                self.assertEqual(len(batch), 4)
                                self.assertEqual(len(set(batch)), 4)
                                if expected == "alignment":
                                    self.assertTrue(all(a is not None and d is None for a, d in batch))
                                    pair_ids.update(a // 13 for a, _ in batch)
                                else:
                                    self.assertTrue(all(a is None and d is not None for a, d in batch))
                            if expected == "alignment":
                                self.assertEqual(len(pair_ids), 1)
                                counts[next(iter(pair_ids))] += 1
                        if counts:
                            values = [counts[i] for i in range(len(pairs))]
                            self.assertLessEqual(max(values) - min(values), 1)

    def test_objective_streams_match_across_modes(self):
        alignment = list(make_sampler("contrastive_only", 12, 3))
        downstream = list(make_sampler("transfer_only", 12, 3))
        for mode in ("alternative", "contrastive_then_transfer"):
            batches = list(make_sampler(mode, 24, 3))
            self.assertEqual([b for b in batches if b[0][0] is not None], alignment)
            self.assertEqual([b for b in batches if b[0][1] is not None], downstream)

    def test_shuffle_and_short_pools(self):
        sampler = make_sampler("contrastive_only", 10)
        rng_state = torch.random.get_rng_state().clone()
        batches = list(sampler)
        self.assertTrue(torch.equal(rng_state, torch.random.get_rng_state()))
        self.assertNotEqual(batches, list(make_sampler("contrastive_only", 10, seed=43)))
        full_batches = list(islice(shuffled_batches(10, 20, 4, 42), 2))
        self.assertEqual(len({i for b in full_batches for i in b}), 8)
        self.assertTrue(all(10 <= i < 20 for b in full_batches for i in b))
        with self.assertRaisesRegex(ValueError, "smaller than batch size"):
            next(shuffled_batches(0, 3, 4, 42))

    def test_actual_trainer_schedule_and_validation(self):
        for mode in ("transfer_only", "contrastive_only", "alternative", "contrastive_then_transfer"):
            for accumulation in (1, 3):
                with self.subTest(mode=mode, accumulation=accumulation), tempfile.TemporaryDirectory() as td:
                    trainer = self.make_trainer(td, mode, accumulation)
                    with redirect_stdout(io.StringIO()):
                        trainer.train()
                    expected = [trainer.objective_at(s) for s in range(6) for _ in range(accumulation)]
                    self.assertEqual([x[0] for x in trainer.model.seen], expected)
                    self.assertTrue(all(keys == {objective} for objective, keys in trainer.model.seen))
                    self.assertEqual(trainer.state.global_step, 6)
                    with redirect_stdout(io.StringIO()):
                        metrics = trainer.evaluate({
                            "align_in_en-ko": CombinedDataset(alignment_dataset=ToyAlignment(("en-ko",))),
                            "massive_in_en": CombinedDataset(downstream_dataset=ToyDownstream()),
                        })
                    self.assertIn("eval_align_in_en-ko_loss", metrics)
                    self.assertIn("eval_massive_in_en_loss", metrics)

    def test_mixed_path_keeps_both_objectives(self):
        with tempfile.TemporaryDirectory() as td:
            trainer = self.make_trainer(td, "alternative", 3, batching="mixed")
            with redirect_stdout(io.StringIO()):
                trainer.train()
            self.assertTrue(all(keys == {"alignment", "downstream"} for _, keys in trainer.model.seen))
            self.assertEqual([x[0] for x in trainer.model.seen],
                             [trainer.objective_at(s) for s in range(6) for _ in range(3)])

    def test_prepared_loader_skip_with_worker_prefetch(self):
        with tempfile.TemporaryDirectory() as td:
            trainer = self.make_trainer(td, "alternative", 3, workers=2)
            loader = trainer.get_train_dataloader()

            def signature(batch):
                objective = next(iter(batch))
                key = "source_input_ids" if objective == "alignment" else "input_ids"
                return objective, batch[objective][key].tolist(), batch[objective].get("lang_pair")

            full = [signature(batch) for batch in loader]
            # Five completed updates, with one update remaining.
            remaining = [signature(batch) for batch in skip_first_batches(loader, 15)]
            self.assertEqual(remaining, full[15:])
            self.assertEqual(len(full), 18)

    def test_schedule_and_pair_mismatch_fail_before_forward(self):
        trainer = planner("alternative", 6)
        trainer.state.global_step = 0
        with self.assertRaisesRegex(RuntimeError, "expected alignment"):
            trainer.training_step(ToyModel(), {"downstream": {}})
        with self.assertRaisesRegex(RuntimeError, "one language pair"):
            trainer.training_step(ToyModel(), {"alignment": {"lang_pair": ["en-ko", "en-ja"]}})

    def test_unsupported_sharding_and_inconsistent_budget_fail(self):
        trainer = planner("alternative", 6)
        trainer.accelerator = SimpleNamespace(num_processes=2)
        trainer.args = SimpleNamespace(n_gpu=1, max_steps=6)
        with self.assertRaisesRegex(ValueError, "one process"):
            trainer.get_train_dataloader()
        trainer.accelerator.num_processes = 1
        trainer.args.max_steps = 7
        with self.assertRaisesRegex(ValueError, "total_steps == max_steps"):
            trainer.get_train_dataloader()


if __name__ == "__main__":
    unittest.main()
