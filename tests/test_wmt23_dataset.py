"""Frozen WMT23 data, split isolation and mixed-language SFT; no network/GPU."""
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from data_utils import AlignmentDataset, CombinedDataset, WMT23Dataset, file_sha256
from config import WMT23_ACCESSIBLE_EXCLUSIONS
from samplers import PairBatchSampler
from scripts.prepare_wmt23 import item, write_arrow, merge_deduplicate, rows, selected_recipe_ids, prepare


class Tokenizer:
    pad_token_id, eos_token_id, eos_token, chat_template = 0, 1, "~", None

    def __call__(self, text, return_tensors=None, **kwargs):
        ids = [ord(char) + 2 for char in text]
        result = {"input_ids": ids, "attention_mask": [1] * len(ids)}
        return {key: torch.tensor([value]) for key, value in result.items()} if return_tensors else result


class WMT23Tests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.tokenizer = Tokenizer()
        self.config = SimpleNamespace(
            downstream_task="wmt23", wmt23_data_dir=str(self.root),
            wmt23_corpus_profile="full_parallel", wmt23_downstream_sampling="proportional",
            training_anchor_langs="en", training_lang=["de", "he", "ja"],
            out_inference_lang=["zh", "ru", "uk"], alignment_data="Helsinki-NLP/opus-100",
            alignment_sampling_seed=42, alignment_num_samples_per_lang=4,
        )
        self.manifest = dict(
            schema_version=1, dataset="wmt23", corpus_profile="full_parallel", source_lang="en",
            training_langs=list(WMT23Dataset.training_langs),
            out_inference_langs=list(WMT23Dataset.out_inference_langs),
            policy={"train_sample_cap": None, "domain_filter": None},
            files={}, splits={}, counts={}, opus_files={}, train_resources={},
            eval_revision="fixture-eval", opus_revision="fixture-opus", data_seed=42,
        )
        for lang in WMT23Dataset.training_langs + WMT23Dataset.out_inference_langs:
            seen = lang in WMT23Dataset.training_langs
            if seen:
                self.manifest["train_resources"][lang] = {
                    "original_recipe_ids": [f"fixture-{lang}"],
                    "effective_recipe_ids": [f"fixture-{lang}"],
                    "usable_counts": {f"fixture-{lang}": 50},
                }
            for split in (("train", "validation", "test") if seen else ("test",)):
                count = {"de": 32, "he": 48, "ja": 64}[lang] if split == "train" else 2
                records = [item(f"English {lang} {split} {i}", f"answer-{lang}-{i}",
                                "fixture", f"{split}:{lang}:{i}") for i in range(count)]
                names, total = write_arrow(self.root, f"{split}.{lang}", records,
                                          self.manifest["files"], shard_rows=20)
                self.manifest["splits"].setdefault(split, {})[lang] = names
                self.manifest["counts"].setdefault(split, {})[lang] = total
            pair = "-".join(sorted(("en", lang)))
            self.manifest["opus_files"][pair] = {}
            for split in (("train", "validation", "test") if seen else ("validation", "test")):
                records = [{"translation": {"en": "heldout" if i == 0 else f"text-{i}", lang: f"target-{i}"}}
                           for i in range(8)]
                path = self.root / f"opus.{pair}.{split}.parquet"
                pq.write_table(pa.Table.from_pylist(records), path)
                self.manifest["opus_files"][pair][split] = path.name
                self.register(path)
        path = self.root / "excluded_alignment_sources.json"
        path.write_text(json.dumps(["heldout"]))
        self.register(path)
        self.save_manifest()

    def register(self, path):
        self.manifest["files"][path.name] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}

    def save_manifest(self):
        path = self.root / "manifest.json"
        path.write_text(json.dumps(self.manifest))
        self.config.wmt23_manifest_sha256 = file_sha256(path)

    def test_complete_pool_and_mixed_language_prompts(self):
        data = WMT23Dataset(self.config, self.tokenizer)
        self.assertEqual(len(data), 144)
        self.assertEqual(data.downstream_sampling, "proportional")
        combined = CombinedDataset(downstream_dataset=data)
        sampler = PairBatchSampler({"de-en": (0, 16)}, len(data), 16, 9, 1, 42, lambda _: "downstream")
        mixed = 0
        for batch in sampler:
            samples = [combined[index]["downstream"] for index in batch]
            mixed += len({sample["lang"] for sample in samples}) > 1
            for sample in samples:
                lang = sample["lang"]
                prompt, _ = data._apply_chat_template(sample["utt"], lang=lang)
                self.assertIn(data.LANGUAGE_NAMES[lang], prompt)
                self.assertNotIn(sample["target"], prompt)
                target = "".join(chr(value - 2) for value in sample["labels"].tolist() if value != -100)
                self.assertIn(sample["target"], target)
        self.assertGreater(mixed, 0)

    def test_frozen_opus_filters_before_sampling_and_preserves_row_ids(self):
        first = AlignmentDataset(self.config, self.tokenizer)
        second = AlignmentDataset(self.config, self.tokenizer)
        self.assertEqual(set(first.all_data), {"de-en", "en-he", "en-ja"})
        self.assertEqual(len(first), 12)
        self.assertEqual([first[i]["sample_id"] for i in range(12)], [second[i]["sample_id"] for i in range(12)])
        for dataset in first.all_data.values():
            self.assertNotIn(0, dataset["_alignment_row_index"])
        out = AlignmentDataset(self.config, self.tokenizer, split="out_test")
        self.assertEqual(set(out.all_data), {"en-zh", "en-ru", "en-uk"})

    def test_unseen_test_and_no_unseen_validation(self):
        data = WMT23Dataset(self.config, self.tokenizer, split="out_test")
        self.assertEqual(set(data.all_data), {"zh", "ru", "uk"})
        self.assertIn("Chinese", data._apply_chat_template("Hello", lang="zh")[0])
        with self.assertRaisesRegex(ValueError, "out_test"):
            WMT23Dataset(self.config, self.tokenizer, split="out_validation")
        with self.assertRaisesRegex(ValueError, "drawn from"):
            WMT23Dataset(self.config, self.tokenizer, languages=["zh"])

    def test_manifest_and_data_tampering_are_rejected(self):
        self.config.wmt23_manifest_sha256 = "0" * 64
        with self.assertRaisesRegex(ValueError, "manifest SHA256"):
            WMT23Dataset.validate_prepared_manifest(self.config)
        self.save_manifest()
        path = self.root / self.manifest["splits"]["train"]["de"][0]
        path.write_bytes(path.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            WMT23Dataset.validate_prepared_manifest(self.config)

    def test_incomplete_recipe_and_wrong_partition_are_rejected(self):
        self.manifest["train_resources"]["he"]["effective_recipe_ids"] = []
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            WMT23Dataset.validate_prepared_manifest(self.config)
        self.config.training_lang = ["de", "cs", "ja"]
        with self.assertRaisesRegex(ValueError, "de/he/ja"):
            WMT23Dataset.validate_prepared_manifest(self.config)

    def test_accessible_profile_allows_only_named_exclusions(self):
        self.config.wmt23_corpus_profile = self.manifest["corpus_profile"] = "accessible_parallel"
        self.manifest["policy"].update(
            excluded_sources=dict(WMT23_ACCESSIBLE_EXCLUSIONS), cross_corpus_deduplication=True,
            deduplication_key="exact_stripped_source_target_within_language",
        )
        for lang, resource in self.manifest["train_resources"].items():
            resource["excluded_recipe_ids"] = {}
            if lang == "he":
                resource["original_recipe_ids"] += list(WMT23_ACCESSIBLE_EXCLUSIONS)
                resource["excluded_recipe_ids"] = dict(WMT23_ACCESSIBLE_EXCLUSIONS)
        self.save_manifest()
        WMT23Dataset.validate_prepared_manifest(self.config)
        self.manifest["train_resources"]["he"]["original_recipe_ids"].append("another-required-corpus")
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            WMT23Dataset.validate_prepared_manifest(self.config)

    def test_unseen_opus_training_file_is_rejected(self):
        self.manifest["opus_files"]["en-zh"]["train"] = self.manifest["opus_files"]["en-zh"]["test"]
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "Incorrect frozen OPUS splits"):
            WMT23Dataset.validate_prepared_manifest(self.config)

    def test_merge_dedupe_is_exact_and_independent_of_input_corpus_order(self):
        resources = []
        for name, pairs in [
            ("a", [(" same ", " target "), ("same", "target"), ("same", "different"), ("tab\there", "quote\"here")]),
            ("b", [("same", "target"), ("Same", "target"), ("日本語", "עברית")]),
        ]:
            source, target = self.root / f"{name}.en", self.root / f"{name}.tgt"
            source.write_text("\n".join(pair[0] for pair in pairs) + "\n")
            target.write_text("\n".join(pair[1] for pair in pairs) + "\n")
            resources.append({"id": name, "kind": "parallel", "paths": [str(source), str(target)]})
        first, counts = merge_deduplicate(resources, self.root / "merged1.tsv")
        second, _ = merge_deduplicate(list(reversed(resources)), self.root / "merged2.tsv")
        first_rows, second_rows = list(rows(first)), list(rows(second))
        self.assertEqual(first_rows, second_rows)
        self.assertEqual(len(first_rows), 5)
        self.assertEqual(counts, {"a": 4, "b": 3})
        kept = next(row for row in first_rows if (row["source_normalized"], row["target_normalized"]) == ("same", "target"))
        self.assertEqual(kept["example_id"], "a:0")
        self.assertEqual(Path(first["paths"][0]).read_bytes(), Path(second["paths"][0]).read_bytes())

    def test_recipe_selection_keeps_other_hebrew_sources(self):
        ids = [*WMT23_ACCESSIBLE_EXCLUSIONS, "OPUS-elrc_2922-v1-eng-heb", "OPUS-ccmatrix-v1-eng-heb"]
        recipes = {"wmt23-enhe": {"train": ids}}
        self.assertEqual(selected_recipe_ids(recipes, "he", "accessible_parallel"), sorted(ids[2:]))
        self.assertEqual(selected_recipe_ids(recipes, "he", "full_parallel"), sorted(ids))

    def test_external_merge_deduplication_keeps_first_identity(self):
        resources = []
        for name in ("a", "b"):
            source, target = self.root / f"large.{name}.en", self.root / f"large.{name}.tgt"
            source.write_text("".join(f"source {i}\n" for i in reversed(range(2000))))
            target.write_text("".join(f"target {i}\n" for i in reversed(range(2000))))
            resources.append({"id": name, "kind": "parallel", "paths": [str(source), str(target)]})
        merged, _ = merge_deduplicate(resources, self.root / "large.tsv", buffer_size="1K")
        result = list(rows(merged))
        self.assertEqual(len(result), 2000)
        self.assertTrue(all(row["corpus"] == "a" for row in result))

    def test_complete_preparation_deduplication_holdouts_and_no_unseen_train(self):
        import gzip
        import yaml
        from mtdata.data import Dataset as MTData
        from scripts.prepare_wmt23 import TRAIN, OUT, RECIPES, ISO3
        output, raw = self.root / "prepared", self.root / "raw"
        recipes = []
        for lang in TRAIN:
            ids = [f"Test-{part}-1-eng-{ISO3[lang]}" for part in ("a", "b")]
            if lang == "he":
                ids += list(WMT23_ACCESSIBLE_EXCLUSIONS)
            recipes.append({"id": RECIPES[lang], "train": ids})

        def fake_fetch(url, path, expected=None):
            path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
            if path.name == "wmt23.recipe.yml":
                path.write_text(yaml.safe_dump(recipes))
            else:
                lang = path.name.split("en-", 1)[1].split(".", 1)[0]
                count = 557 if lang == "de" else 2074
                if ".meta." in path.name:
                    content = [json.dumps({"docid": str(i), "domain": "news"}) for i in range(count)]
                elif ".src." in path.name:
                    content = [f"official {i}" for i in range(count)]
                else:
                    content = [f"reference {lang} {i}" for i in range(count)]
                path.write_text("\n".join(content) + "\n")
            return path

        requested_opus = []
        def fake_opus(repo, filename, *, local_dir, **kwargs):
            pair, split_file = filename.split("/")
            split = split_file.split("-")[0]
            requested_opus.append((pair, split))
            lang = next(lang for lang in pair.split("-") if lang != "en")
            path = Path(local_dir) / filename; path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pylist([
                {"translation": {"en": f"OPUS {split} {i}", lang: f"target {i}"}} for i in range(8)
            ]), path)
            return str(path)

        def fake_prepare(*, out_dir, dataset_ids, **kwargs):
            part = out_dir / "train-parts"; part.mkdir(parents=True, exist_ok=True)
            for did in dataset_ids["train"]:
                codes = str(did).split("-")[-2:]
                target_code = next(code for code in codes if code != "eng")
                for code in codes:
                    texts = ([f"{target_code} source {i}" for i in range(3)] + ["official 0", f"{target_code} source 0"]
                             if code == "eng" else [f"target {i}" for i in range(3)] + ["heldout", "target 0"])
                    with gzip.open(part / f"{did}.{code}.gz", "wt", encoding="utf-8") as stream:
                        stream.write("\n".join(texts) + "\n")

        args = SimpleNamespace(output_dir=output, mtdata_dir=raw, czeng_dir=None,
                               corpus_profile="accessible_parallel", seed=42, validation_per_language=1)
        with patch("scripts.prepare_wmt23.fetch", side_effect=fake_fetch), \
             patch("scripts.prepare_wmt23.check_recipe_sources"), \
             patch("scripts.prepare_wmt23.hf_hub_download", side_effect=fake_opus), \
             patch.object(MTData, "resolve_entries", side_effect=lambda ids: [SimpleNamespace(did=did, url="https://fixture") for did in ids]), \
             patch.object(MTData, "prepare", side_effect=fake_prepare):
            prepare(args)
        self.config.wmt23_data_dir = str(output)
        self.config.wmt23_corpus_profile = "accessible_parallel"
        self.config.wmt23_manifest_sha256 = file_sha256(output / "manifest.json")
        manifest = WMT23Dataset.validate_prepared_manifest(self.config)
        self.assertEqual(manifest["counts"]["train"], {lang: 2 for lang in TRAIN})
        self.assertTrue(all(manifest["train_resources"][lang]["duplicates_removed"] == 6 for lang in TRAIN))
        for lang in OUT:
            self.assertNotIn(("-".join(sorted(("en", lang))), "train"), requested_opus)
        training = WMT23Dataset(self.config, self.tokenizer)
        validation = WMT23Dataset(self.config, self.tokenizer, split="in_validation")
        train_sources = {training.get_generation_sample(i)["utt"] for i in range(len(training))}
        valid_sources = {validation.get_generation_sample(i)["utt"] for i in range(len(validation))}
        self.assertTrue(train_sources.isdisjoint(valid_sources | {"official 0"}))


class ALMAJapaneseTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.tokenizer = Tokenizer()
        self.config = SimpleNamespace(
            downstream_task="wmt23", wmt23_data_dir=str(self.root),
            wmt23_corpus_profile="alma_ja_opus", wmt23_downstream_sampling="proportional",
            training_anchor_langs="en", training_lang=["de", "cs", "ja"],
            out_inference_lang=["zh", "ru", "uk"], alignment_data="Helsinki-NLP/opus-100",
            alignment_sampling_seed=42, alignment_num_samples_per_lang=4,
        )
        self.manifest = dict(
            schema_version=2, dataset="wmt23", corpus_profile="alma_ja_opus", source_lang="en",
            training_langs=["de", "cs", "ja"], out_inference_langs=["zh", "ru", "uk"],
            policy={"train_sample_cap": None, "directions": "bidirectional",
                    "alignment": "OPUS-100 seen-language train only",
                    "test": "haoranxu/WMT23-Test; cs-en reverses en-cs"},
            files={}, splits={}, counts={}, opus_files={},
            train_resources={lang: {} for lang in self.config.training_lang},
            eval_revision="fixture-eval", opus_revision="fixture-opus", data_seed=42,
        )
        for lang in self.config.training_lang + self.config.out_inference_lang:
            seen = lang in self.config.training_lang
            for split in (("train", "validation", "test") if seen else ("test",)):
                for direction in (f"en-{lang}", f"{lang}-en"):
                    src, tgt = direction.split("-")
                    records = [item(f"{src} {split} input {i}", f"{tgt} {split} answer {i}",
                                    "fixture", f"{split}:{direction}:{i}") for i in range(8)]
                    names, count = write_arrow(self.root, f"{split}.{direction}", records, self.manifest["files"])
                    self.manifest["splits"].setdefault(split, {})[direction] = names
                    self.manifest["counts"].setdefault(split, {})[direction] = count
            pair = "-".join(sorted(("en", lang)))
            self.manifest["opus_files"][pair] = {}
            for split in (("train", "validation", "test") if seen else ("validation", "test")):
                path = self.root / f"opus.{pair}.{split}.parquet"
                pq.write_table(pa.Table.from_pylist([
                    {"translation": {"en": "heldout" if i == 0 else f"source {i}", lang: f"target {i}"}}
                    for i in range(8)
                ]), path)
                self.manifest["opus_files"][pair][split] = path.name
                self.register(path)
        path = self.root / "excluded_alignment_sources.json"
        path.write_text(json.dumps(["heldout"]))
        self.register(path)
        self.save_manifest()

    def register(self, path):
        self.manifest["files"][path.name] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}

    def save_manifest(self):
        path = self.root / "manifest.json"
        path.write_text(json.dumps(self.manifest))
        self.config.wmt23_manifest_sha256 = file_sha256(path)

    def test_bidirectional_prompts_and_supervised_labels_in_mixed_batches(self):
        data = WMT23Dataset(self.config, self.tokenizer)
        self.assertEqual(len(data), 48)
        combined = CombinedDataset(downstream_dataset=data)
        plan = lambda: list(PairBatchSampler({"cs-en": (0, 8)}, len(data), 8, 6, 1, 42, lambda _: "downstream"))
        self.assertEqual(plan(), plan())
        seen = set()
        for batch in plan():
            for index in batch:
                row = combined[index]["downstream"]
                direction = row["item"]["direction"]
                src, tgt = direction.split("-")
                prompt, _ = data._apply_chat_template(row["utt"], lang=row["lang"])
                self.assertIn(f"from {data.LANGUAGE_NAMES[src]} to {data.LANGUAGE_NAMES[tgt]}", prompt)
                self.assertNotIn(row["target"], prompt)
                labels = "".join(chr(v - 2) for v in row["labels"].tolist() if v != -100)
                self.assertIn(row["target"], labels)
                seen.add(direction)
        self.assertEqual(seen, {f"{src}-{tgt}" for lang in ["de", "cs", "ja"] for src, tgt in [("en", lang), (lang, "en")]})

    def test_opus_czech_works_and_unseen_train_is_rejected(self):
        alignment = AlignmentDataset(self.config, self.tokenizer)
        self.assertEqual(set(alignment.all_data), {"cs-en", "de-en", "en-ja"})
        self.assertEqual(len(alignment), 12)
        self.assertTrue(all(alignment[i]["source_text"] != "heldout" and alignment[i]["target_text"] != "heldout" for i in range(len(alignment))))
        self.manifest["opus_files"]["en-zh"]["train"] = self.manifest["opus_files"]["en-zh"]["test"]
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "Incorrect frozen OPUS splits"):
            AlignmentDataset(self.config, self.tokenizer)

    def test_balanced_languages_group_both_directions_and_preserve_full_pool(self):
        self.config.wmt23_downstream_sampling = "balanced_mixed"
        data = WMT23Dataset(self.config, self.tokenizer)
        self.assertEqual(len(data), 48)
        self.assertEqual(data.balanced_language_ranges, {"de": (0, 16), "cs": (16, 32), "ja": (32, 48)})
        sampler = PairBatchSampler({"cs-en": (0, 8)}, len(data), 8, 18, 1, 42,
                                  lambda _: "downstream", data.balanced_language_ranges, "balanced_mixed")
        counts = {"de": 0, "cs": 0, "ja": 0}
        seen = set()
        for batch in sampler:
            for _, index in batch:
                row = data.get_generation_sample(index)
                source, target = row["item"]["direction"].split("-")
                counts[target if source == "en" else source] += 1
                seen.add(index)
        self.assertEqual(counts, {"de": 48, "cs": 48, "ja": 48})
        self.assertEqual(seen, set(range(48)))

    def test_test_directions_and_missing_reverse_data(self):
        data = WMT23Dataset(self.config, self.tokenizer, split="out_test")
        self.assertEqual(set(data.all_data), {"en-zh", "zh-en", "en-ru", "ru-en", "en-uk", "uk-en"})
        del self.manifest["splits"]["test"]["cs-en"]
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "bidirectional test"):
            WMT23Dataset.validate_prepared_manifest(self.config)

    def test_bleu_uses_target_language_and_keeps_directions_separate(self):
        from evaluate import evaluate_wmt_predictions, wmt_scorer
        self.assertEqual(wmt_scorer("sacrebleu", "en-zh").tokenizer_signature, "zh")
        self.assertEqual(wmt_scorer("sacrebleu", "zh-en").tokenizer_signature, "13a")
        self.assertEqual(wmt_scorer("sacrebleu", "en-ja").tokenizer_signature, "ja-mecab-0.996-IPA")
        self.assertEqual(wmt_scorer("sacrebleu", "ja-en").tokenizer_signature, "13a")
        predictions = [dict(lang=direction, direction=direction, source="input", target="one two three four",
                            prediction="one two three four", sample_id=direction, generation_limit_reached=False,
                            source_paragraphs=1, prediction_paragraphs=1) for direction in ("en-de", "de-en")]
        metrics = evaluate_wmt_predictions(predictions, "sacrebleu")
        self.assertEqual(set(metrics["by_language"]), {"en-de", "de-en"})
        self.assertEqual(metrics["by_language"]["de-en"]["direction"], "de-en")
        self.assertAlmostEqual(metrics["macro_average"]["sacrebleu"], 100.)


if __name__ == "__main__":
    unittest.main()
