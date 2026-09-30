"""Actual CPU Trainer integration for gap loss, diagnostics, and eval tails."""

from collections import defaultdict
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from transformers import PretrainedConfig, TrainingArguments

from custom_trainer import AlternativeRoutingTrainer
from config import ALIGNMENT_LOSSES
from data_utils import AlignmentDataset, CombinedDataset
from models import CustomModel


class TinyTokenizer:
    pad_token_id = 0

    def __call__(self, text, **kwargs):
        return {
            "input_ids": torch.tensor([[int(text) + 1]]),
            "attention_mask": torch.ones(1, 1, dtype=torch.long),
        }


class TinyAlignment(AlignmentDataset):
    def __init__(self, sizes=None):
        sizes = sizes or {"en-ko": 13, "en-ja": 13}
        self.config = SimpleNamespace(alignment_data="toy-gap")
        self.tokenizer = TinyTokenizer()
        self.dataset_metadata = {pair: {"split": "train"} for pair in sizes}
        self.all_data = {}
        for pair, size in sizes.items():
            source, target = pair.split("-")
            self.all_data[pair] = [
                {"translation": {source: str(i), target: str(i + 20 + i % 3)},
                 "_alignment_row_index": i}
                for i in range(size)
            ]


class UnusedDownstream(torch.utils.data.Dataset):
    def __len__(self):
        return 17

    def __getitem__(self, index):
        raise AssertionError("Alignment-only training must not access downstream examples.")


class TinyBaseModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = PretrainedConfig()
        positions = torch.arange(64, dtype=torch.float32)
        weights = torch.stack((positions / 8, positions.square() / 64,
                               torch.sin(positions), torch.cos(positions)), dim=1)
        self.embedding = torch.nn.Embedding.from_pretrained(weights, freeze=False)
        self.dropout = torch.nn.Dropout(.2)

    def forward(self, input_ids, **kwargs):
        hidden = self.dropout(self.embedding(input_ids))
        return SimpleNamespace(hidden_states=(hidden, 2 * hidden))


class GapTrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def make_trainer(self, output_dir, train_interval=0, eval_limit=0,
                     loss_type="gap_consistency"):
        config = SimpleNamespace(
            alignment_loss=loss_type,
            alignment_hidden_state_layer=0,
            alignment_hidden_state_position="last_token",
            alignment_temperature=.2,
            train_sample_log_interval=train_interval,
            train_sample_log_limit=2,
        )
        model = CustomModel(config, TinyBaseModel())
        dataset = CombinedDataset(TinyAlignment(), UnusedDownstream(), 3 * 4 * 2)
        args = TrainingArguments(
            output_dir=str(output_dir), run_name="gap-trainer-test", use_cpu=True,
            max_steps=3, per_device_train_batch_size=4, per_device_eval_batch_size=4,
            gradient_accumulation_steps=2, learning_rate=.03, optim="sgd",
            lr_scheduler_type="constant", max_grad_norm=0,
            save_strategy="no", eval_strategy="no", logging_strategy="steps",
            logging_steps=1, report_to=[], remove_unused_columns=False,
            dataloader_pin_memory=False, disable_tqdm=True, seed=42,
        )
        return AlternativeRoutingTrainer(
            model=model, args=args, train_dataset=dataset,
            data_collator=dataset.collate_fn, training_type="contrastive_only",
            total_steps=3, alignment_batching="same_pair", eval_sample_log_limit=eval_limit,
        )

    @staticmethod
    def read_jsonl(path):
        return [json.loads(line) for line in Path(path).read_text().splitlines()]

    def test_training_accumulation_gradients_and_observation_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.make_trainer(directory, train_interval=1)
            initial = trainer.model.basemodel.embedding.weight.detach().clone()
            gradients = []
            trainer.model.basemodel.embedding.weight.register_hook(
                lambda gradient: gradients.append(gradient.detach().clone())
            )
            with redirect_stdout(io.StringIO()):
                trainer.train()
            self.assertEqual(trainer.state.global_step, 3)
            self.assertEqual(len(gradients), 6)
            self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
            self.assertTrue(any(torch.count_nonzero(gradient) for gradient in gradients))
            self.assertFalse(torch.equal(initial, trainer.model.basemodel.embedding.weight))

            records = self.read_jsonl(Path(directory) / "train_samples/alignment-observations.jsonl")
            batches = self.read_jsonl(Path(directory) / "train_samples/alignment-batches.jsonl")
            self.assertEqual(len(records), 12)
            self.assertEqual(len(batches), 6)
            by_id = {batch["batch_id"]: batch for batch in batches}
            self.assertEqual(len(by_id), 6)
            self.assertEqual(len({record["record_id"] for record in records}), 12)
            pairs_by_update = defaultdict(set)
            for batch in batches:
                self.assertEqual(batch["actual_batch_size"], 4)
                self.assertEqual(len(batch["sample_ids"]), 4)
                self.assertEqual(len(set(batch["language_pairs"])), 1)
                pairs_by_update[batch["alignment_update_index"]].update(batch["language_pairs"])
                self.assertEqual(batch["global_step_before_update"], batch["alignment_update_index"] - 1)
                self.assertIn(batch["microbatch_index"], (1, 2))
            self.assertEqual(set(pairs_by_update), {1, 2, 3})
            self.assertTrue(all(len(pairs) == 1 for pairs in pairs_by_update.values()))
            for record in records:
                batch = by_id[record["batch_id"]]
                self.assertEqual(record["sample_id"], batch["sample_ids"][record["batch_position"]])
                self.assertEqual(record["record_id"], f"{record['batch_id']}/sample-{record['batch_position']}")
                self.assertEqual(record["alignment_loss_type"], "gap_consistency")
                self.assertTrue(record["dropout_active"])
                self.assertEqual(record["source_num_tokens"], 1)
                self.assertEqual(len(record["source_token_hash"]), 64)
                self.assertAlmostEqual(record["gap_squared_residual"], record["per_sample_loss"], places=4)
            summaries = [row for row in trainer.state.log_history if "alignment_loss" in row]
            self.assertEqual(len(summaries), 3)
            self.assertTrue(all(any(key.endswith("/gap_distance_mean") for key in row) for row in summaries))

    def test_logging_does_not_change_dropout_training_or_final_weights(self):
        weights = []
        with tempfile.TemporaryDirectory() as directory:
            for interval in (0, 1):
                output_dir = Path(directory) / str(interval)
                trainer = self.make_trainer(output_dir, train_interval=interval)
                with redirect_stdout(io.StringIO()):
                    trainer.train()
                weights.append(trainer.model.basemodel.embedding.weight.detach().clone())
                self.assertEqual((output_dir / "train_samples").exists(), bool(interval))
            self.assertTrue(torch.equal(weights[0], weights[1]))

    def test_gap_evaluation_tails_weighted_loss_and_unlimited_statistics(self):
        summaries = []
        with tempfile.TemporaryDirectory() as directory:
            for limit in (0, 2):
                output_dir = Path(directory) / str(limit)
                trainer = self.make_trainer(output_dir, eval_limit=limit)
                dataset = CombinedDataset(alignment_dataset=TinyAlignment({"en-ko": 5, "en-ja": 7}))
                trainer.model.eval()
                weighted_losses = []
                distances = defaultdict(list)
                batch_sizes = []
                with torch.no_grad():
                    for inputs in trainer.get_eval_dataloader(dataset):
                        batch = inputs["alignment"]
                        outputs = trainer.model(forward_type="alignment", return_per_sample=True, **inputs)
                        size = len(batch["sample_id"])
                        batch_sizes.append(size)
                        weighted_losses.append(outputs["loss"].item() * size)
                        distances[batch["lang_pair"][0]].extend(outputs["gap_distance"].tolist())
                self.assertEqual(sorted(batch_sizes), [3, 4, 5])
                with redirect_stdout(io.StringIO()):
                    metrics = trainer.evaluate({"align_in_all": dataset})
                self.assertAlmostEqual(metrics["eval_align_in_all_loss"], sum(weighted_losses) / 12, places=6)
                payload = json.loads((output_dir / "eval_samples/step-0_metrics.json").read_text())["eval_align_in_all"]
                self.assertEqual(payload["observed_num_examples"], 12)
                summary = payload["distance_statistics"]
                summaries.append(summary)
                for pair, values in distances.items():
                    self.assertEqual(summary[pair]["num_examples"], len(values))
                    reference = torch.tensor(values, dtype=torch.float64).var(unbiased=False).item()
                    self.assertAlmostEqual(summary[pair]["corpus_gap_distance_variance"], reference, places=10)
                self.assertNotAlmostEqual(
                    summary["en-ja"]["batch_gap_loss_mean"],
                    summary["en-ja"]["corpus_gap_distance_variance"], places=5,
                )
                sample_path = output_dir / "eval_samples/step-0.json"
                self.assertEqual(sample_path.exists(), bool(limit))
                if limit:
                    records = json.loads(sample_path.read_text())
                    self.assertEqual({key: len(rows) for key, rows in records.items()},
                                     {"alignment/en-ko": 2, "alignment/en-ja": 2})
                    old_ids = {key: [row["sample_id"] for row in rows] for key, rows in records.items()}
                    old_records = {row["record_id"] for rows in records.values() for row in rows}
                    with redirect_stdout(io.StringIO()):
                        trainer.evaluate({"align_in_all": dataset})
                    repeated = json.loads(sample_path.read_text())
                    self.assertEqual(old_ids, {key: [row["sample_id"] for row in rows] for key, rows in repeated.items()})
                    self.assertTrue(old_records.isdisjoint(
                        {row["record_id"] for rows in repeated.values() for row in rows}
                    ))
            self.assertEqual(summaries[0], summaries[1])

    def test_contrastive_variants_train_and_evaluate_pair_batches(self):
        for method in ALIGNMENT_LOSSES[2:]:
            with self.subTest(method=method), tempfile.TemporaryDirectory() as directory:
                trainer = self.make_trainer(directory, loss_type=method)
                trainer.args.learning_rate = 1e-4
                before = trainer.model.basemodel.embedding.weight.detach().clone()
                with redirect_stdout(io.StringIO()):
                    result = trainer.train()
                self.assertTrue(torch.isfinite(torch.tensor(result.training_loss)))
                self.assertFalse(torch.equal(before, trainer.model.basemodel.embedding.weight))
                dataset = CombinedDataset(alignment_dataset=TinyAlignment({"en-ko": 5, "en-ja": 7}))
                trainer.model.eval()
                total = 0.0
                sizes = []
                with torch.no_grad():
                    for inputs in trainer.get_eval_dataloader(dataset):
                        batch = inputs["alignment"]
                        self.assertEqual(len(set(batch["lang_pair"])), 1)
                        sizes.append(len(batch["lang_pair"]))
                        total += trainer.model(**inputs)["loss"].item() * sizes[-1]
                self.assertEqual(sorted(sizes), [3, 4, 5])
                with redirect_stdout(io.StringIO()):
                    metrics = trainer.evaluate({"align_in_all": dataset})
                self.assertAlmostEqual(metrics["eval_align_in_all_loss"], total/12, places=5)

    def test_custom_evaluation_prefix_flushes_at_outermost_call(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.make_trainer(directory, eval_limit=2)
            dataset = CombinedDataset(alignment_dataset=TinyAlignment({"en-ko": 5}))
            with redirect_stdout(io.StringIO()):
                trainer.evaluate({"align_in_en-ko": dataset}, metric_key_prefix="probe")
            output_dir = Path(directory) / "eval_samples"
            metrics = json.loads((output_dir / "step-0_metrics.json").read_text())
            self.assertIn("probe_align_in_en-ko", metrics)
            self.assertEqual(trainer.eval_sample_buffer, {})
            self.assertEqual(trainer._eval_batch_buffer, [])
            first = json.loads((output_dir / "step-0.json").read_text())["alignment/en-ko"]
            with redirect_stdout(io.StringIO()):
                trainer.evaluate({"align_in_en-ko": dataset}, metric_key_prefix="probe")
            second = json.loads((output_dir / "step-0.json").read_text())["alignment/en-ko"]
            self.assertEqual([row["sample_id"] for row in first], [row["sample_id"] for row in second])
            self.assertNotEqual(first[0]["record_id"], second[0]["record_id"])

    def test_resume_rejects_objective_mismatch_before_loading_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint"
            checkpoint.mkdir()
            for saved_config, current_loss in (({}, "gap_consistency"),
                                               ({"alignment_loss": "gap_consistency"}, "infonce")):
                (checkpoint / "experiment_config.json").write_text(json.dumps(saved_config))
                trainer = self.make_trainer(Path(directory) / current_loss, loss_type=current_loss)
                with patch("transformers.Trainer._load_from_checkpoint") as parent_load:
                    with self.assertRaisesRegex(ValueError, "Cannot resume"):
                        trainer._load_from_checkpoint(str(checkpoint))
                    parent_load.assert_not_called()
            for saved_config, current_loss in (({}, "infonce"),
                                               ({"alignment_loss": "gap_consistency"}, "gap_consistency")):
                (checkpoint / "experiment_config.json").write_text(json.dumps(saved_config))
                trainer = self.make_trainer(Path(directory) / current_loss, loss_type=current_loss)
                with patch("transformers.Trainer._load_from_checkpoint") as parent_load:
                    trainer._load_from_checkpoint(str(checkpoint))
                    parent_load.assert_called_once()


if __name__ == "__main__":
    unittest.main()
