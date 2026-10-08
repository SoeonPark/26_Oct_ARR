"""Document-level WMT25 splitting and prepared-file integration."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from datasets import Dataset
import yaml

from data_utils import WMT25Dataset
from scripts.prepare_wmt25 import (
    EVAL_LOCALES, ISO3, TED_CORPORA, normalize_pair, prepare,
    select_training_resources, split_official_evaluation, split_training_validation, write_jsonl,
)


class PrepareWMT25Tests(unittest.TestCase):
    def test_recipe_profiles_and_read_only_plan(self):
        recipes = [
            {"id": f"wmt25-eng-{ISO3[lang]}",
             "train": [TED_CORPORA[lang], f"Fixture-news-1-eng-{ISO3[lang]}"],
             "mono_train": ["must-not-be-used"]}
            for lang in WMT25Dataset.training_langs
        ]
        full = select_training_resources(recipes, "full_recipe")
        ted = select_training_resources(recipes, "ted")
        for lang in full:
            self.assertEqual(len(full[lang]["corpus_ids"]), 2)
            self.assertEqual(full[lang]["corpus_ids"], full[lang]["recipe_train_corpus_ids"])
            self.assertEqual(ted[lang]["corpus_ids"], [TED_CORPORA[lang]])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recipe = root / "recipes.yml"
            recipe.write_text(yaml.safe_dump(recipes))
            args = SimpleNamespace(plan=True, recipe_file=recipe, mtdata_dir=root / "raw",
                                   output_dir=root / "prepared", corpus_profile="full_recipe")
            plan = prepare(args)
            self.assertEqual(plan["corpus_counts"], {"ko": 2, "ja": 2, "cs": 2})
            self.assertFalse(args.output_dir.exists())
            self.assertFalse(args.mtdata_dir.exists())
        with self.assertRaisesRegex(ValueError, "Duplicate recipe"):
            select_training_resources(recipes + [recipes[0]], "full_recipe")

    def make_groups(self):
        return {
            lang: [
                {**normalize_pair(f"Document {i}", f"Translation {lang} {i}", lang,
                                  "wmttest2025", f"{lang}:{i}", "test"),
                 "doc_id": f"{lang}:{i}"}
                for i in range(5 if lang == "cs" else 4)
            ]
            for lang in EVAL_LOCALES.values()
        }

    def test_shared_documents_are_removed_from_every_test_language(self):
        groups = self.make_groups()
        validation, test = split_official_evaluation(groups, 42, 2)
        sources = {row["source_normalized"] for row in validation["et"]}
        self.assertEqual(set(validation), {"et", "ru", "ar"})
        for lang, rows in validation.items():
            self.assertEqual(len(rows), 2)
            self.assertEqual({row["source_normalized"] for row in rows}, sources)
            self.assertTrue(all(row["split"] == "validation" for row in rows))
        for lang, rows in test.items():
            self.assertEqual(len(rows), 3 if lang == "cs" else 2)
            self.assertTrue(sources.isdisjoint(row["source_normalized"] for row in rows))
            self.assertTrue(all(row["split"] == "test" for row in rows))
        self.assertTrue(all(row["split"] == "test" for rows in groups.values() for row in rows))

    def test_split_membership_is_reproducible_and_input_order_independent(self):
        groups = self.make_groups()
        first = split_official_evaluation(groups, 42, 2)
        self.assertEqual(first, split_official_evaluation(groups, 42, 2))
        reordered = split_official_evaluation(
            {lang: list(reversed(rows)) for lang, rows in groups.items()}, 42, 2,
        )
        for left, right in zip(first, reordered):
            self.assertEqual(
                {lang: {row["example_id"] for row in rows} for lang, rows in left.items()},
                {lang: {row["example_id"] for row in rows} for lang, rows in right.items()},
            )

    def test_split_requires_both_validation_and_test(self):
        for count in (0, 4, 5):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "out_validation_per_language"):
                split_official_evaluation(self.make_groups(), 42, count)

    def test_official_sources_never_enter_training_or_in_validation(self):
        dataset = Dataset.from_list([
            normalize_pair(source, f"target {i}", "ko", "fixture", i)
            for i, source in enumerate(["Official document", "A", "B", "C", "D"])
        ])
        train, validation = split_training_validation(
            {"ko": dataset}, 42, 1, excluded_sources={"Official document"},
        )
        self.assertEqual(len(train["ko"]), 3)
        self.assertEqual(len(validation["ko"]), 1)
        self.assertNotIn("Official document", train["ko"]["source_normalized"])
        self.assertNotIn("Official document", [row["source_normalized"] for row in validation["ko"]])

    def test_prepare_writes_all_five_splits_with_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            recipes = []
            for lang in WMT25Dataset.training_langs:
                recipe_id = f"wmt25-eng-{ISO3[lang]}"
                parts = raw / recipe_id / "train-parts"
                parts.mkdir(parents=True)
                corpora = [TED_CORPORA[lang], f"Fixture-news-1-eng-{ISO3[lang]}"]
                for corpus in corpora:
                    sources = ["Document 0", *[f"Training source {corpus} {i}" for i in range(6)]]
                    (parts / f"{corpus}.eng").write_text("\n".join(sources) + "\n", encoding="utf-8")
                    (parts / f"{corpus}.{ISO3[lang]}").write_text("\n".join(f"Target {i}" for i in range(7)) + "\n", encoding="utf-8")
                recipes.append({"id": recipe_id, "train": corpora, "mono_train": ["unused"]})
            recipe_file = root / "recipes.yml"
            recipe_file.write_text(yaml.safe_dump(recipes))
            official_rows = []
            for locale, lang in EVAL_LOCALES.items():
                for i in range(5 if lang == "cs" else 4):
                    official_rows.append({
                        "collection_id": "general", "dataset_id": "wmttest2025",
                        "src_lang": "en", "tgt_lang": locale, "doc_id": f"{lang}:{i}",
                        "domain": "news", "src_text": f"Document {i}",
                        "refs": {"refA": {"ref": f"Reference {lang} {i}"}},
                    })
            eval_file = root / "official.jsonl"
            write_jsonl(eval_file, official_rows)
            args = SimpleNamespace(
                output_dir=root / "prepared", mtdata_dir=raw, recipe_file=recipe_file,
                eval_file=eval_file, download=False, corpus_profile="full_recipe", seed=42,
                validation_per_language=1, out_validation_per_language=2,
            )
            manifest = prepare(args)
            self.assertEqual(manifest["counts"]["train"], {"ko": 11, "ja": 11, "cs": 11})
            self.assertEqual(manifest["counts"]["validation"], {"ko": 1, "ja": 1, "cs": 1, "et": 2, "ru": 2, "ar": 2})
            self.assertEqual(manifest["counts"]["test"], {"ko": 2, "ja": 2, "cs": 3, "et": 2, "ru": 2, "ar": 2})
            self.assertEqual(manifest["reference_counts"], manifest["counts"]["test"])
            self.assertEqual(len(manifest["evaluation_split"]["validation_source_sha256"]), 2)
            for lang in WMT25Dataset.training_langs:
                resource = manifest["train_resources"][lang]
                self.assertEqual(resource["corpus_ids"], resource["recipe_train_corpus_ids"])
                self.assertEqual(len(resource["input_files"]), 4)
            for split, counts in manifest["counts"].items():
                for lang, count in counts.items():
                    path = args.output_dir / f"{split}.{lang}.jsonl"
                    rows = [json.loads(line) for line in path.read_text().splitlines()]
                    self.assertEqual(len(rows), count)
                    self.assertTrue(all(row["split"] == split and row["direction"] == f"en-{lang}" and row["target_normalized"] for row in rows))
                    if split == "train":
                        self.assertEqual({row["corpus"] for row in rows}, set(manifest["train_resources"][lang]["corpus_ids"]))
                        self.assertNotIn("Document 0", {row["source_normalized"] for row in rows})
            with self.assertRaisesRegex(ValueError, "output_dir must be empty"):
                prepare(args)


if __name__ == "__main__":
    unittest.main()
