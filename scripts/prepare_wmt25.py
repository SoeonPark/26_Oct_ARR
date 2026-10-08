"""Prepare WMT25 downstream MT data; downloading is separate from training.

Use --download to run mtdata==0.4.3, or point --mtdata_dir at its existing
per-recipe train-parts directories. full_recipe selects every parallel train
resource for EN->KO/JA/CS; mono_train is not used. --plan lists the resources
without downloading corpora or creating output files. The explicit ted
profile is only a legacy subset. No OPUS-100 alignment split is loaded.
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import unicodedata
from urllib.request import urlopen, urlretrieve

from datasets import Dataset
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from data_utils import WMT25Dataset


RECIPE_URL = "https://www.statmt.org/wmt25/mtdata/mtdata.recipes.wmt25-constrained.yml"
EVAL_REVISION = "56c0a513f64ba63500e222b25bf87ac2201cb1eb"
EVAL_URL = (
    "https://raw.githubusercontent.com/wmt-conference/wmt25-general-mt/"
    f"{EVAL_REVISION}/data/wmt25-genmt.jsonl"
)
ISO3 = {"ko": "kor", "ja": "jpn", "cs": "ces"}
EVAL_LOCALES = {
    "ko_KR": "ko",
    "ja_JP": "ja",
    "cs_CZ": "cs",
    "et_EE": "et",
    "ru_RU": "ru",
    "ar_EG": "ar",
}
TED_CORPORA = {
    "ko": "OPUS-neulab_tedtalks-v1-eng-kor",
    "ja": "OPUS-neulab_tedtalks-v1-eng-jpn",
    "cs": "OPUS-neulab_tedtalks-v1-ces-eng",
}


def select_training_resources(recipes, profile):
    """Resolve the entire requested recipe; never substitute missing corpora."""
    if profile not in ("full_recipe", "ted"):
        raise ValueError(f"Unknown corpus_profile: {profile!r}.")
    by_id = {recipe["id"]: recipe for recipe in recipes}
    if len(by_id) != len(recipes):
        raise ValueError("Duplicate recipe IDs.")
    resources = {}
    for lang in WMT25Dataset.training_langs:
        recipe_id = f"wmt25-eng-{ISO3[lang]}"
        train_ids = by_id[recipe_id]["train"]
        if not train_ids or len(set(train_ids)) != len(train_ids):
            raise ValueError(f"Empty or duplicate train resources in {recipe_id}.")
        selected = sorted(train_ids if profile == "full_recipe" else [TED_CORPORA[lang]])
        if not set(selected) <= set(train_ids):
            raise ValueError(f"Requested corpus is not in official recipe {recipe_id}.")
        resources[lang] = {
            "recipe_id": recipe_id, "corpus_ids": selected,
            "recipe_train_corpus_ids": sorted(train_ids),
        }
    return resources


def preparation_plan(args):
    """Read only the small recipe file, leaving raw/prepared directories untouched."""
    path = args.recipe_file or args.mtdata_dir / "mtdata.recipes.wmt25-constrained.yml"
    if args.recipe_file is not None or path.is_file():
        recipe_bytes = path.read_bytes()
    else:
        with urlopen(RECIPE_URL, timeout=60) as response:
            recipe_bytes = response.read()
    resources = select_training_resources(yaml.safe_load(recipe_bytes), args.corpus_profile)
    plan = {
        "corpus_profile": args.corpus_profile,
        "recipe_sha256": hashlib.sha256(recipe_bytes).hexdigest(),
        "training_resources": resources,
        "corpus_counts": {lang: len(resource["corpus_ids"]) for lang, resource in resources.items()},
        "uses_mono_train": False,
        "evaluation_dataset": "wmttest2025",
        "output_dir": str(args.output_dir.resolve()),
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    return plan


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_pair(source, target, lang, corpus, example_id, split="train"):
    # Preserve extracted text separately; NFC/strip is only for prompts/hashes.
    source_raw = source.rstrip("\r\n")
    target_raw = target.rstrip("\r\n") if target is not None else None
    source_normalized = unicodedata.normalize("NFC", source_raw).strip()
    target_normalized = unicodedata.normalize("NFC", target_raw).strip() if target_raw is not None else None
    if not source_normalized or (split != "test" and not target_normalized):
        return None
    if split != "test" and any(
        marker in text for marker in ("__NULL__", "_ _ NULL _ _")
        for text in (source_normalized, target_normalized)
    ):
        return None
    key = json.dumps(["en", lang, source_normalized, target_normalized], ensure_ascii=False)
    return {
        "source": source_raw, "target": target_raw,
        "source_normalized": source_normalized, "target_normalized": target_normalized,
        "src_lang": "en", "tgt_lang": lang, "direction": f"en-{lang}",
        "dataset": "wmt25", "split": split, "corpus": corpus,
        "example_id": str(example_id),
        "pair_hash": hashlib.sha256(key.encode("utf-8")).hexdigest() if target_normalized else None,
    }


def parallel_paths(parts_dir, corpus, lang):
    # Some corpus IDs are ces-eng; region variants (eng_GB, kor_KR) also occur.
    codes = corpus.split("-")[-2:]
    if sorted(code.split("_")[0] for code in codes) != sorted(["eng", ISO3[lang]]):
        raise ValueError(f"Unexpected language pair for {lang}: {corpus}")
    paths = {}
    for code in codes:
        plain = Path(parts_dir) / f"{corpus}.{code}"
        compressed = Path(str(plain) + ".gz")
        candidates = [p for p in (plain, compressed) if p.is_file()]
        if len(candidates) != 1:
            raise ValueError(f"Expected one extracted file for {plain} (plain or .gz). Run mtdata first.")
        paths[code.split("_")[0]] = candidates[0]
    return paths["eng"], paths[ISO3[lang]]


def iter_parallel_rows(parts_dir, corpus_ids, lang, content_digest=None):
    for corpus in sorted(corpus_ids):
        source_path, target_path = parallel_paths(parts_dir, corpus, lang)
        source_open = gzip.open if source_path.suffix == ".gz" else open
        target_open = gzip.open if target_path.suffix == ".gz" else open
        with source_open(source_path, "rt", encoding="utf-8") as sources, target_open(
            target_path, "rt", encoding="utf-8"
        ) as targets:
            for index, (source, target) in enumerate(zip(sources, targets, strict=True)):
                row = normalize_pair(source, target, lang, corpus, f"{corpus}:{index}")
                if row is not None:
                    yield row


def split_training_validation(datasets, seed, count, excluded_sources=()):
    """Exclude official evaluation sources, then reserve training-resource validation."""
    excluded_sources = set(excluded_sources)
    shuffled, validation = {}, {}
    for lang, dataset in datasets.items():
        if excluded_sources:
            dataset = dataset.filter(
                lambda source: source not in excluded_sources,
                input_columns=["source_normalized"],
            )
        if len(dataset) <= count:
            raise ValueError(f"WMT25 {lang} needs more than {count} usable rows for train/validation.")
        shuffled[lang] = dataset.shuffle(seed=seed)
        validation[lang] = [
            {**row, "split": "validation"}
            for row in shuffled[lang].select(range(count))
        ]
    # No validation source may occur in ANY target language's training pool.
    heldout_sources = {
        row["source_normalized"] for rows in validation.values() for row in rows
    }
    train = {}
    for lang, dataset in shuffled.items():
        train[lang] = dataset.filter(
            lambda source: source not in heldout_sources,
            input_columns=["source_normalized"],
        )
        if not len(train[lang]):
            raise ValueError(f"WMT25 {lang} has no training rows after validation source exclusion.")
    return train, validation


def load_official_evaluation(path):
    groups = {
        lang: []
        for lang in (*WMT25Dataset.training_langs, *WMT25Dataset.out_inference_langs)
    }
    seen = set()
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if (row["collection_id"] != "general" or row["src_lang"] != "en"
                    or row["tgt_lang"] not in EVAL_LOCALES):
                continue
            lang = EVAL_LOCALES[row["tgt_lang"]]
            identity = (row["dataset_id"], row["tgt_lang"], row["doc_id"])
            if identity in seen:
                raise ValueError(f"Duplicate WMT25 document: {identity}")
            seen.add(identity)
            target = row.get("refs", {}).get("refA", {}).get("ref")

            if not target or not target.strip():
                raise ValueError(
                    f"Missing WMT25 reference: {row['tgt_lang']}, {row['doc_id']}"
                )
            item = normalize_pair(row["src_text"], target, lang, "wmttest2025", row["doc_id"], "test")
            if item is None:
                raise ValueError(f"Empty WMT25 evaluation source: {identity}")
            item.update({
                key: row.get(key) for key in (
                    "doc_id", "domain", "dataset_id", "collection_id",
                    "prompt_instruction", "video", "screenshot",
                )
            })
            item["reference_id"] = "refA" if target else None
            item["tgt_locale"] = row["tgt_lang"]
            groups[lang].append(item)
    if any(not rows for rows in groups.values()):
        raise ValueError(f"Official WMT25 data must contain general EN→{list(groups)} documents.")
    return groups


def split_official_evaluation(groups, seed, count):
    """Hold out the same English documents across unseen languages and all tests."""
    shared_sources = set.intersection(*(
        {row["source_normalized"] for row in groups[lang]}
        for lang in WMT25Dataset.out_inference_langs
    ))
    if not 0 < count < len(shared_sources):
        raise ValueError(
            f"out_validation_per_language must be between 1 and {len(shared_sources) - 1}."
        )
    validation_sources = set(random.Random(seed).sample(sorted(shared_sources), count))
    validation = {
        lang: [
            {**row, "split": "validation"}
            for row in groups[lang]
            if row["source_normalized"] in validation_sources
        ]
        for lang in WMT25Dataset.out_inference_langs
    }
    test = {
        lang: [row for row in rows if row["source_normalized"] not in validation_sources]
        for lang, rows in groups.items()
    }
    if any(len(rows) != count for rows in validation.values()):
        raise ValueError("Official validation must contain exactly one row per selected source and language.")
    if any(not rows for rows in test.values()):
        raise ValueError("Every WMT25 language must retain a nonempty test set.")
    return validation, test


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--mtdata_dir", type=Path, required=True)
    parser.add_argument("--recipe_file", type=Path)
    parser.add_argument("--eval_file", type=Path)
    parser.add_argument("--download", action="store_true", help="Download parallel resources using mtdata==0.4.3.")
    parser.add_argument("--plan", action="store_true", help="Print the selected recipe resources; do not download corpora or write files.")
    parser.add_argument("--corpus_profile", choices=["full_recipe", "ted"], default="full_recipe")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation_per_language", type=int, default=500)
    parser.add_argument(
        "--out_validation_per_language", type=int, default=20,
        help="Shared WMT25 English documents reserved for each unseen language's validation.",
    )
    return parser.parse_args()


def prepare(args):
    if getattr(args, "plan", False):
        return preparation_plan(args)
    if args.validation_per_language <= 0:
        raise ValueError("validation_per_language must be positive.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("output_dir must be empty; use a separate directory for each corpus profile/seed.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.mtdata_dir.mkdir(parents=True, exist_ok=True)
    recipe_file = args.recipe_file or args.mtdata_dir / "mtdata.recipes.wmt25-constrained.yml"
    eval_file = args.eval_file or args.mtdata_dir / f"wmt25-genmt.{EVAL_REVISION}.jsonl"
    for path, url in ((recipe_file, RECIPE_URL), (eval_file, EVAL_URL)):
        if not path.exists():
            urlretrieve(url, path)
    resources = select_training_resources(
        yaml.safe_load(recipe_file.read_text(encoding="utf-8")), args.corpus_profile,
    )
    official = load_official_evaluation(eval_file)
    out_validation, test_sets = split_official_evaluation(
        official, args.seed, args.out_validation_per_language,
    )
    official_sources = {
        row["source_normalized"] for rows in official.values() for row in rows
    }
    manifest = {
        "schema_version": 3, "dataset": "wmt25", "source_lang": "en",
        "training_langs": list(WMT25Dataset.training_langs),
        "out_inference_langs": list(WMT25Dataset.out_inference_langs),
        "training_pool": "all_usable_except_validation_and_official_evaluation_sources", "seed": args.seed,
        "validation_per_language": args.validation_per_language,
        "out_validation_per_language": args.out_validation_per_language,
        "evaluation_split": {
            "source_dataset": "wmttest2025",
            "policy": "shared_source_document_holdout",
            "seed": args.seed,
            "official_counts": {lang: len(rows) for lang, rows in official.items()},
            "validation_doc_ids": {
                lang: [row["doc_id"] for row in rows]
                for lang, rows in out_validation.items()
            },
            "validation_source_sha256": sorted({
                hashlib.sha256(row["source_normalized"].encode("utf-8")).hexdigest()
                for rows in out_validation.values() for row in rows
            }),
        },
        "filtering_version": 2, "corpus_profile": args.corpus_profile,
        "recipe_url": RECIPE_URL, "recipe_sha256": sha256_file(recipe_file),
        "eval_url": EVAL_URL if args.eval_file is None else None,
        "eval_file": str(eval_file.resolve()), "eval_sha256": sha256_file(eval_file),
        "train_resources": {}, "counts": {"train": {}, "validation": {}, "test": {}},
    }
    datasets = {}
    for lang in WMT25Dataset.training_langs:
        resource = resources[lang]
        recipe_id, corpus_ids = resource["recipe_id"], resource["corpus_ids"]
        root = args.mtdata_dir / recipe_id
        if args.download:
            subprocess.run([
                sys.executable, "-m", "mtdata", "get", "-l", f"eng-{ISO3[lang]}",
                "-tr", *corpus_ids, "-o", str(root), "--no-merge", "--compress",
            ], check=True)
        inputs = []
        for corpus in corpus_ids:
            for path in parallel_paths(root / "train-parts", corpus, lang):
                inputs.append({"path": str(path.resolve()), "sha256": sha256_file(path)})
        resource = {**resource, "input_files": inputs}
        # Include content hashes in HF cache identity, not just filesystem paths.
        dataset = Dataset.from_generator(
            iter_parallel_rows,
            gen_kwargs={
                "parts_dir": str(root / "train-parts"), "corpus_ids": corpus_ids, "lang": lang,
                "content_digest": hashlib.sha256(json.dumps(resource, sort_keys=True).encode()).hexdigest(),
            },
        )
        datasets[lang] = dataset
        manifest["train_resources"][lang] = {**resource, "usable_examples": len(dataset)}
    train_sets, validation_sets = split_training_validation(
        datasets, args.seed, args.validation_per_language, excluded_sources=official_sources,
    )
    validation_sets.update(out_validation)
    for split, split_sets in (
        ("train", train_sets), ("validation", validation_sets), ("test", test_sets),
    ):
        for lang, rows in split_sets.items():
            write_jsonl(args.output_dir / f"{split}.{lang}.jsonl", rows)
            manifest["counts"][split][lang] = len(rows)
    manifest["reference_counts"] = {
        lang: sum(bool(row["target_normalized"]) for row in rows)
        for lang, rows in test_sets.items()
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"counts": manifest["counts"], "reference_counts": manifest["reference_counts"]}, indent=2))
    return manifest


if __name__ == "__main__":
    prepare(parse_args())
