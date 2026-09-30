"""Stable sample identities using in-memory datasets; no network access."""

from contextlib import redirect_stdout
import io
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from datasets import Dataset
import torch

from data_utils import AlignmentDataset, MassiveDataset, make_sample_id


class CharacterTokenizer:
    pad_token_id = 0
    eos_token = ""
    chat_template = None

    def __call__(self, text, return_tensors=None, **kwargs):
        ids = [ord(char) + 1 for char in text]
        result = {"input_ids": ids, "attention_mask": [1] * len(ids)}
        if return_tensors == "pt":
            return {key: torch.tensor([value]) for key, value in result.items()}
        return result


class AlignmentIdentityTests(unittest.TestCase):
    def make_alignment(self, seed, split="in_validation", sample_count=12):
        # Duplicate text must still retain different original row identities.
        raw = Dataset.from_dict({
            "translation": [{"en": "same", "ko": "same"} for _ in range(12)],
        })
        config = SimpleNamespace(
            alignment_data="test/parallel", alignment_sampling_seed=seed,
            alignment_num_samples_per_lang=sample_count,
        )
        with patch("data_utils.load_dataset", return_value=raw), redirect_stdout(io.StringIO()):
            dataset = AlignmentDataset(
                config, CharacterTokenizer(), split=split, lang_pairs=["en-ko"],
            )
        return dataset, raw

    def test_alignment_identity_survives_shuffle_scope_and_selection(self):
        first, raw = self.make_alignment(42)
        second, _ = self.make_alignment(7, split="out_validation")
        subset, _ = self.make_alignment(42, sample_count=4)
        first_ids = [first[i]["sample_id"] for i in range(len(first))]
        second_ids = [second[i]["sample_id"] for i in range(len(second))]
        self.assertNotEqual(first_ids, second_ids)
        self.assertEqual(set(first_ids), set(second_ids))
        self.assertEqual(len(set(first_ids)), len(first))
        self.assertEqual([subset[i]["sample_id"] for i in range(len(subset))], first_ids[:4])
        self.assertEqual(first.dataset_metadata["en-ko"]["fingerprint"], raw._fingerprint)
        self.assertEqual(first.dataset_metadata["en-ko"]["num_rows"], 12)
        for i in range(len(first)):
            item = first[i]
            self.assertEqual(json.loads(item["sample_id"]), [
                "test/parallel", "en-ko", "validation", item["original_row_index"],
            ])
            self.assertNotIn("_alignment_row_index", item["item"])

    def test_alignment_collation_preserves_identity_and_token_lengths(self):
        dataset, _ = self.make_alignment(42)
        items = [dataset[0], dataset[1]]
        batch = dataset.collate_fn(items)
        self.assertEqual(batch["sample_id"], [item["sample_id"] for item in items])
        self.assertEqual(batch["original_row_index"], [item["original_row_index"] for item in items])
        self.assertEqual(batch["source_attention_mask"].sum(dim=1).tolist(), [4, 4])

    def test_massive_uses_native_id_and_locale_and_collates(self):
        raw = Dataset.from_dict({"id": [77], "utt": ["hello"], "annot_utt": ["hello"]})
        config = SimpleNamespace(
            downstream_task_data="test/massive", training_anchor_langs="en",
            training_lang=["ko"], out_inference_lang=[],
        )
        with patch("data_utils.load_dataset", return_value=raw), redirect_stdout(io.StringIO()):
            dataset = MassiveDataset(config, CharacterTokenizer(), split="in_validation")
        items = [dataset[0], dataset[1]]
        self.assertEqual(json.loads(items[0]["sample_id"]), [
            "test/massive", "en-US", "validation", 77,
        ])
        self.assertNotEqual(items[0]["sample_id"], items[1]["sample_id"])
        self.assertEqual(dataset.collate_fn(items)["sample_id"], [item["sample_id"] for item in items])
        self.assertEqual(items[0]["sample_id"], make_sample_id(
            "test/massive", "en-US", "validation", 77,
        ))
        self.assertEqual(dataset.dataset_metadata["en"], {
            "dataset": "test/massive", "config": "en-US", "split": "validation",
            "fingerprint": raw._fingerprint, "num_rows": 1,
        })


if __name__ == "__main__":
    unittest.main()
