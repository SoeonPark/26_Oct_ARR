"""MT direction, partition, causal masking and generation."""

import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from datasets import Dataset
import torch
from torch.utils.data import DataLoader
from transformers import LlamaConfig, LlamaForCausalLM

from data_utils import CombinedDataset, WMT25Dataset
from evaluate import (build_wmt_evaluation_samples, collate_massive_evaluation_samples,
                      evaluate_wmt25_predictions, generate_wmt_predictions)
from main import build_eval_datasets
from models import CustomModel
from samplers import PairBatchSampler
from scripts.prepare_wmt25 import (iter_parallel_rows, load_official_evaluation, normalize_pair,
                                   split_training_validation, write_jsonl)


class ByteTokenizer:
    pad_token_id, eos_token_id = 0, 1
    pad_token, eos_token = "<pad>", "<eos>"
    padding_side, chat_template, model_max_length = "right", None, 4096

    def __call__(self, text, **kwargs):
        def encode(value):
            end = value.endswith(self.eos_token)
            if end:
                value = value[:-len(self.eos_token)]
            return [byte + 3 for byte in value.encode("utf-8")] + ([1] if end else [])
        if isinstance(text, str):
            ids = encode(text)
            return {"input_ids": ids, "attention_mask": [1] * len(ids)}
        rows = [encode(value) for value in text]
        width = max(map(len, rows))
        ids, masks = [], []
        for row in rows:
            padding = [0] * (width - len(row))
            ids.append(padding + row if self.padding_side == "left" else row + padding)
            masks.append([0] * len(padding) + [1] * len(row) if self.padding_side == "left" else [1] * len(row) + [0] * len(padding))
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(masks)}

    def batch_decode(self, rows, **kwargs):
        return [bytes(token - 3 for token in row.tolist() if token >= 3).decode("utf-8") for row in rows]


class RecordingBase(torch.nn.Module):
    def __init__(self, tokenizer, terminate=True):
        super().__init__()
        self.embedding = torch.nn.Embedding(260, 4)
        self.config = SimpleNamespace(max_position_embeddings=4096)
        self.generation_config = SimpleNamespace(eos_token_id=tokenizer.eos_token_id)
        self.tokenizer, self.terminate, self.prompts = tokenizer, terminate, []

    def get_input_embeddings(self):
        return self.embedding

    def generate(self, input_ids, attention_mask, max_new_tokens, **kwargs):
        assert self.tokenizer.padding_side == "left"
        assert (attention_mask[:, -1] == 1).all()
        self.prompts.extend(self.tokenizer.batch_decode(input_ids))
        answer = [ord("x") + 3, 1] if self.terminate else [ord("x") + 3] * max_new_tokens
        return torch.cat([input_ids, torch.tensor([answer] * len(input_ids))], dim=1)


class ChatByteTokenizer(ByteTokenizer):
    chat_template = "fixture"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs):
        assert not tokenize and kwargs["enable_thinking"] is False
        text = "".join(f"<{message['role']}>{message['content']}" for message in messages)
        return text + ("<assistant>" if add_generation_prompt else self.eos_token)


class WMT25Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.tokenizer = ByteTokenizer()
        # self.config = SimpleNamespace(wmt25_data_dir=str(self.root), training_seed=42)
        self.config = SimpleNamespace(
            wmt25_data_dir=str(self.root),
            training_seed=42,
            training_anchor_langs="en",
            wmt25_downstream_sampling="proportional",
        )
        targets = {
            "ko": "한국어 정답", "ja": "日本語の答え", "cs": "Česká odpověď",
            "et": "Eestikeelne vastus", "ru": "Русский ответ", "ar": "ده الرد",
        }
        counts = {"train": {}, "validation": {}, "test": {}}
        for lang, train_count in zip(WMT25Dataset.training_langs, (3, 5, 7)):
            for split, count in (("train", train_count), ("validation", 1), ("test", 1)):
                rows = [normalize_pair(f"English {split} text {i}", targets[lang], lang, "fixture", f"{split}:{i}", split) for i in range(count)]
                write_jsonl(self.root / f"{split}.{lang}.jsonl", rows)
                counts[split][lang] = count
        for lang in WMT25Dataset.out_inference_langs:
            for split in ("validation", "test"):
                row = normalize_pair(f"English {split} document\n\nSecond paragraph", targets[lang], lang, "official", f"en-{lang}:{split}", split)
                row.update(doc_id=f"en-{lang}:{split}", dataset_id="wmttest2025")
                write_jsonl(self.root / f"{split}.{lang}.jsonl", [row])
                counts[split][lang] = 1
        manifest = {"source_lang": "en", "training_langs": list(WMT25Dataset.training_langs),
                    "out_inference_langs": list(WMT25Dataset.out_inference_langs),
                    "seed": 42, "counts": counts, "corpus_profile": "ted"}
        (self.root / "manifest.json").write_text(json.dumps(manifest))

    def test_required_profile_rejects_legacy_ted_and_incomplete_full_recipe(self):
        self.config.wmt25_corpus_profile = "full_recipe"
        with self.assertRaisesRegex(ValueError, "corpus_profile mismatch"):
            WMT25Dataset.validate_prepared_manifest(self.config)
        path = self.root / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["corpus_profile"] = "full_recipe"
        manifest["train_resources"] = {
            lang: {"corpus_ids": ["ted"], "recipe_train_corpus_ids": ["ted", "news"]}
            for lang in WMT25Dataset.training_langs
        }
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "incomplete"):
            WMT25Dataset.validate_prepared_manifest(self.config)
        for resource in manifest["train_resources"].values():
            resource["corpus_ids"] = ["news", "ted"]
        path.write_text(json.dumps(manifest))
        self.assertEqual(WMT25Dataset.validate_prepared_manifest(self.config), manifest)

    def test_shuffled_mixed_batches_keep_each_rows_language_prompt_and_target(self):
        for tokenizer in (ByteTokenizer(), ChatByteTokenizer()):
            for languages in (None, ["cs", "ko", "ja"]):
                with self.subTest(tokenizer=type(tokenizer).__name__, languages=languages):
                    dataset = WMT25Dataset(self.config, tokenizer, languages=languages)
                    self.assertEqual(dataset.downstream_sampling, "proportional")
                    combined = CombinedDataset(downstream_dataset=dataset)
                    sampler = PairBatchSampler(
                        pair_ranges={"en-ko": (0, 4)}, downstream_size=len(dataset),
                        batch_size=4, num_steps=6, accumulation_steps=2, seed=42,
                        objective_at=lambda step: "downstream", downstream_ranges=None,
                    )
                    loader = DataLoader(combined, batch_sampler=sampler, collate_fn=combined.collate_fn)
                    mixed_batches, seen = 0, set()
                    for routed in loader:
                        batch = routed["downstream"]
                        mixed_batches += len(set(batch["lang"])) > 1
                        for i, lang in enumerate(batch["lang"]):
                            seen.add(lang)
                            text = tokenizer.batch_decode([batch["input_ids"][i]])[0]
                            expected_prompt, _ = dataset._apply_chat_template(batch["utt"][i], lang=lang)
                            self.assertTrue(text.startswith(expected_prompt))
                            self.assertIn(f"English to {dataset.LANGUAGE_NAMES[lang]}", text)
                            self.assertEqual(batch["item"][i]["tgt_lang"], lang)
                            self.assertEqual(batch["utt"][i], batch["item"][i]["source_normalized"])
                            labels = batch["labels"][i]
                            gold = tokenizer.batch_decode([labels[labels != -100]])[0]
                            self.assertEqual(gold.strip(), batch["target"][i])
                            self.assertEqual(batch["target"][i], batch["item"][i]["target_normalized"])
                    self.assertGreater(mixed_batches, 0)
                    self.assertEqual(seen, {"ko", "ja", "cs"})

    def test_anchor_language_must_be_explicitly_english(self):
        for anchor in ("ko", None):
            config = SimpleNamespace(**vars(self.config))
            if anchor is None:
                del config.training_anchor_langs
            else:
                config.training_anchor_langs = anchor

            with self.subTest(anchor=anchor):
                with self.assertRaisesRegex(ValueError, "training_anchor_langs"):
                    WMT25Dataset(config, self.tokenizer)

    def test_training_counts_masking_and_collation(self):
        ds = WMT25Dataset(self.config, self.tokenizer)
        self.assertEqual(len(ds), 15)
        self.assertEqual(ds.language_ranges, {"ko": (0, 3), "ja": (3, 8), "cs": (8, 15)})
        self.assertEqual(set(ds.all_data), {"ko", "ja", "cs"})
        for index in (0, 3, 8):
            sample = ds[index]
            prompt, _ = ds._apply_chat_template(sample["utt"], lang=sample["lang"])
            language = ds.LANGUAGE_NAMES[sample["lang"]]
            self.assertEqual(sample["utt"], sample["item"]["source_normalized"])
            self.assertEqual(prompt, (
                f"System: Translate the following sentences from English to {language}.\n"
                f"User: {sample['utt']}\nAssistant:"
            ))
            prompt_length = len(self.tokenizer(prompt)["input_ids"])
            self.assertTrue((sample["labels"][:prompt_length] == -100).all())
            decoded = self.tokenizer.batch_decode([sample["labels"][sample["labels"] != -100]])[0]
            self.assertEqual(decoded.strip(), sample["target"])
            self.assertEqual(sample["item"]["src_lang"], "en")
        combined = CombinedDataset(downstream_dataset=ds)
        batch = combined.collate_fn([combined[0], combined[3]])["downstream"]
        self.assertEqual(batch["labels"].shape, batch["input_ids"].shape)
        self.assertTrue((batch["labels"][batch["attention_mask"] == 0] == -100).all())
        self.assertEqual(ds[-1]["lang"], "cs")

    def test_heldout_partition_and_referenced_generation(self):
        ds = WMT25Dataset(self.config, self.tokenizer, split="out_test")
        self.assertEqual(set(ds.all_data), {"et", "ru", "ar"})
        self.assertEqual(str(ds.all_data["et"].split), "test")
        sample = ds.get_generation_sample(0)
        self.assertTrue(sample["target"])
        self.assertNotIn("labels", sample)
        self.assertEqual(WMT25Dataset(self.config, self.tokenizer, split="out_test", languages=["ru"])[0]["lang"], "ru")
        for split, languages in (("train", ["et"]), ("train", ["ko"]), ("out_test", ["ko"]), ("in_validation", ["ar"]), ("out_validation", ["ko"]), ("in_test", ["ru"])):
            with self.subTest(split=split, languages=languages), self.assertRaises(ValueError):
                WMT25Dataset(self.config, self.tokenizer, split, languages)

    def test_all_evaluation_splits_supervised_masking(self):
        for split, languages in (
            ("in_validation", {"ko", "ja", "cs"}), ("in_test", {"ko", "ja", "cs"}),
            ("out_validation", {"et", "ru", "ar"}), ("out_test", {"et", "ru", "ar"}),
        ):
            ds = WMT25Dataset(self.config, self.tokenizer, split=split)
            self.assertEqual(set(ds.all_data), languages)
            for sample in ds:
                prompt, _ = ds._apply_chat_template(sample["utt"], lang=sample["lang"])
                prompt_length = len(self.tokenizer(prompt)["input_ids"])
                self.assertTrue((sample["labels"][:prompt_length] == -100).all())
                decoded = self.tokenizer.batch_decode([sample["labels"][sample["labels"] != -100]])[0]
                self.assertEqual(decoded.strip(), sample["target"])
            batch = ds.collate_fn([ds[i] for i in range(len(ds))])
            self.assertTrue((batch["labels"][batch["attention_mask"] == 0] == -100).all())

    def test_existing_downstream_model_accepts_mt_batch_and_backpropagates(self):
        dataset = WMT25Dataset(self.config, self.tokenizer)
        combined = CombinedDataset(downstream_dataset=dataset)
        batch = combined.collate_fn([combined[0], combined[3]])
        base = LlamaForCausalLM(LlamaConfig(vocab_size=260, hidden_size=16,
                                          intermediate_size=32, num_hidden_layers=1,
                                          num_attention_heads=2, num_key_value_heads=2))
        config = SimpleNamespace(alignment_hidden_state_layer=-1,
                                 alignment_hidden_state_position="last_token",
                                 alignment_temperature=0.05)
        model = CustomModel(config, base)
        output = model(**batch, forward_type="downstream")
        self.assertTrue(torch.isfinite(output["loss"]))
        output["loss"].backward()
        self.assertTrue(torch.isfinite(base.get_input_embeddings().weight.grad).all())

    def test_reversed_row_and_seed_mismatch_are_rejected(self):
        path = self.root / "train.ko.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["src_lang"] = "ko"
        write_jsonl(path, rows)
        with self.assertRaisesRegex(ValueError, "direction"):
            WMT25Dataset(self.config, self.tokenizer)
        self.config.training_seed = 17
        with self.assertRaisesRegex(ValueError, "seed"):
            WMT25Dataset(self.config, self.tokenizer)

    def test_full_pool_split_reproducible_and_no_cross_language_validation_leakage(self):
        datasets = {
            lang: Dataset.from_list([
                normalize_pair(f"source {i}", f"target {lang} {i}", lang, "fixture", i)
                for i in range(count)
            ])
            for lang, count in (("ko", 20), ("ja", 25), ("cs", 30))
        }
        train, validation = split_training_validation(datasets, 42, 2)
        again, validation_again = split_training_validation(datasets, 42, 2)
        self.assertEqual(validation, validation_again)
        heldout = {row["source_normalized"] for rows in validation.values() for row in rows}
        for lang, dataset in datasets.items():
            self.assertEqual(train[lang]["example_id"], again[lang]["example_id"])
            self.assertEqual(set(train[lang]["example_id"]), {
                row["example_id"] for row in dataset if row["source_normalized"] not in heldout
            })
            self.assertEqual(len(validation[lang]), 2)
            self.assertTrue(all(row["split"] == "validation" for row in validation[lang]))
        with self.assertRaisesRegex(ValueError, "more than 2"):
            split_training_validation({"ko": datasets["ko"].select(range(2))}, 42, 2)
        duplicates = Dataset.from_list([
            normalize_pair("one shared source", str(i), "ko", "fixture", i) for i in range(4)
        ])
        with self.assertRaisesRegex(ValueError, "no training rows"):
            split_training_validation({"ko": duplicates}, 42, 2)

    def test_unicode_and_czech_file_order_and_mismatched_lengths(self):
        for lang, text in [("ko", "안녕하세요"), ("ja", "こんにちは"), ("hi", "नमस्ते"), ("th", "สวัสดี")]:
            row = normalize_pair("Hello\n\nWorld\n", text + "\n", lang, "fixture", 0)
            self.assertEqual(row["target"], text)
            self.assertEqual(row["source"], "Hello\n\nWorld")
            self.assertEqual(json.loads(json.dumps(row, ensure_ascii=False)), row)
        corpus = "OPUS-neulab_tedtalks-v1-ces-eng"
        (self.root / f"{corpus}.eng").write_text("English first\nEnglish second\n")
        (self.root / f"{corpus}.ces").write_text("Čeština první\nČeština druhá\n", encoding="utf-8")
        rows = list(iter_parallel_rows(self.root, [corpus], "cs"))
        self.assertEqual(rows[0]["source"], "English first")
        (self.root / f"{corpus}.ces").write_text("Čeština\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            list(iter_parallel_rows(self.root, [corpus], "cs"))
        self.assertIsNone(normalize_pair("Hello", "__NULL__", "ja", "fixture", 0))

    def test_official_filter_excludes_testsuites_reverse_and_system_outputs(self):
        rows = [{"src_lang": "en", "tgt_lang": locale, "collection_id": "general", "dataset_id": "wmttest2025",
                 "doc_id": f"en-{locale}:doc", "src_text": "English", "refs": {"refA": {"ref": "Reference"}}, "tgt_text": {"system": "not gold"}}
                for locale in ("ko_KR", "ja_JP", "cs_CZ", "et_EE", "ru_RU", "ar_EG")]
        rows += [{**rows[1], "collection_id": "testsuites"}, {**rows[0], "src_lang": "cs"}]
        path = self.root / "official.jsonl"
        write_jsonl(path, rows)
        groups = load_official_evaluation(path)
        self.assertEqual({lang: len(group) for lang, group in groups.items()}, {lang: 1 for lang in ("ko", "ja", "cs", "et", "ru", "ar")})
        self.assertTrue(all(group[0]["target"] == "Reference" for group in groups.values()))
        rows[0] = {**rows[0], "refs": {}}
        write_jsonl(path, rows)
        with self.assertRaisesRegex(ValueError, "Missing WMT25 reference"):
            load_official_evaluation(path)

    def test_validation_routing_keeps_opus_independent(self):
        self.config.downstream_task = "wmt25"
        self.config.training_anchor_langs = "en"
        self.config.training_lang, self.config.out_inference_lang = ["es"], ["fr"]
        self.config.eval_language_scope = "both"
        with patch("main.AlignmentDataset", return_value=[0, 1]) as opus, patch("main.MassiveDataset") as massive:
            groups = build_eval_datasets(self.config, self.tokenizer)
        self.assertEqual(set(groups), {"align_in_en-es", "align_out_en-fr", "wmt25_in_ko", "wmt25_in_ja", "wmt25_in_cs", "wmt25_out_et", "wmt25_out_ru", "wmt25_out_ar"})
        self.assertEqual(opus.call_count, 2)
        massive.assert_not_called()

    def test_generation_does_not_use_gold_or_slots_and_restores_padding(self):
        ds = WMT25Dataset(self.config, self.tokenizer, split="in_validation", languages=["ko", "ja"])
        samples = build_wmt_evaluation_samples(ds)
        loader = DataLoader(samples, batch_size=2, collate_fn=collate_massive_evaluation_samples)
        base = RecordingBase(self.tokenizer)
        predictions = generate_wmt_predictions(SimpleNamespace(basemodel=base), self.tokenizer, ds, loader, 8)
        self.assertEqual(self.tokenizer.padding_side, "right")
        self.assertTrue(all(sample["target"] not in prompt for sample, prompt in zip(samples, base.prompts)))
        for sample, prompt in zip(samples, base.prompts):
            language = ds.LANGUAGE_NAMES[sample["lang"]]
            self.assertEqual(prompt, (
                f"System: Translate the following sentences from English to {language}.\n"
                f"User: {sample['utt']}\nAssistant:"
            ))
        self.assertEqual(predictions[0]["prediction"], "x")
        self.assertNotIn("predicted_slots", predictions[0])
        self.assertFalse(predictions[0]["generation_limit_reached"])
        base.config.max_position_embeddings = 8
        with self.assertRaisesRegex(ValueError, "No input was truncated"):
            generate_wmt_predictions(SimpleNamespace(basemodel=base), self.tokenizer, ds, loader, 8)
        self.assertEqual(self.tokenizer.padding_side, "right")

    def test_generation_honors_chat_eos_even_when_model_eos_differs(self):
        ds = WMT25Dataset(self.config, self.tokenizer, split="out_test")
        loader = DataLoader(build_wmt_evaluation_samples(ds), batch_size=3,
                            collate_fn=collate_massive_evaluation_samples)
        for model_eos, expected in ((2, [1, 2]), ([2, 1], [1, 2]), (None, [1])):
            with self.subTest(model_eos=model_eos):
                base = RecordingBase(self.tokenizer)
                base.generation_config.eos_token_id = model_eos
                with patch.object(base, "generate", wraps=base.generate) as generate:
                    predictions = generate_wmt_predictions(
                        SimpleNamespace(basemodel=base), self.tokenizer, ds, loader, 2)
                self.assertEqual(generate.call_args.kwargs["eos_token_id"], expected)
                self.assertTrue(all(row["prediction"] == "x" for row in predictions))
                self.assertFalse(any(row["generation_limit_reached"] for row in predictions))
                self.assertEqual(self.tokenizer.padding_side, "right")

    def test_generation_caps_and_missing_reference_metrics(self):
        ds = WMT25Dataset(self.config, self.tokenizer, split="out_test")
        loader = DataLoader(build_wmt_evaluation_samples(ds), batch_size=3, collate_fn=collate_massive_evaluation_samples)
        predictions = generate_wmt_predictions(SimpleNamespace(basemodel=RecordingBase(self.tokenizer, False)), self.tokenizer, ds, loader, 2)
        self.assertTrue(all(row["generation_limit_reached"] for row in predictions))
        result = evaluate_wmt25_predictions(predictions)
        self.assertEqual(result["status"], "generation_only")
        self.assertIsNone(result["macro_average"]["chrf"])
        with self.assertRaisesRegex(ValueError, "requires a reference"):
            evaluate_wmt25_predictions([{**row, "target": None} for row in predictions], "chrf")

    @unittest.skipUnless(importlib.util.find_spec("sacrebleu"), "optional sacrebleu not installed")
    def test_chrf_language_macro(self):
        rows = [{"lang": lang, "target": "a real translation", "prediction": "a real translation",
                 "generation_limit_reached": False, "source_paragraphs": 1, "prediction_paragraphs": 1}
                for lang in ("ko", "ja", "cs")]
        result = evaluate_wmt25_predictions(rows, "chrf")
        self.assertEqual(result["status"], "scored")
        self.assertAlmostEqual(result["macro_average"]["chrf"], 100)


if __name__ == "__main__":
    unittest.main()
