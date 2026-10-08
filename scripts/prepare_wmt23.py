"""Freeze MT SFT, WMT23 tests and OPUS alignment; default: ALMA plus Japanese."""
import argparse
import gzip
import hashlib
import json
import random
import shutil
import os
import subprocess
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from pathlib import Path
from urllib.request import Request, urlopen, urlretrieve

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from huggingface_hub import hf_hub_download

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import WMT23_ACCESSIBLE_EXCLUSIONS, WMT23_PARTITIONS

ALMA_REV = "4cee1a0e3e1daac4b50a759e8d8659ee2e6595bc"
HF_WMT23_REV = "19c6f6f18f37bc2bd58306a2b20c93bcbaa4e75a"
FLORES_REV = "71abf77d8b7beb5cfef59898d6b24d92ab7654fc"
ALMA_FILES = {
    "cs": {"train": "cs-en/train-00000-of-00001-3a60b130a713425b.parquet",
           "validation": "cs-en/validation-00000-of-00001-d1f9a3fc339fbc84.parquet"},
    "de": {"train": "de-en/train-00000-of-00001-39460826cd7ac756.parquet",
           "validation": "de-en/validation-00000-of-00001-34198d3f975c1787.parquet"},
}

TRAIN = ("de", "he", "ja")
OUT = ("zh", "ru", "uk")
ISO3 = {"de": "deu", "he": "heb", "ja": "jpn"}
RECIPES = {
    "de": "wmt23-ende",
    "he": "wmt23-enhe",
    "ja": "wmt23-enjp",
}
RECIPE_URL = (
    "https://www.statmt.org/wmt23/mtdata/"
    "mtdata.recipes.wmt23-constrained.yml"
)
RECIPE_SHA = "477ea6da657ec77a1cd27b72ad1bc38a849d510334ed6eff6d59ce66c99da3b5"
EVAL_REV = "460aa1ca168816b6438e38d3c4ee6026099a4db4"
OPUS_REV = "805090dc28bf78897da9641cdf08b61287580df9"
CZENG = (
    "czeng20-train.gz",
    "czeng20-csmono.gz",
    "czeng20-enmono.gz",
)
SCHEMA = pa.schema([
    (key, pa.string())
    for key in (
        "source_normalized", "target_normalized",
        "example_id", "corpus",
        "doc_id", "domain", "reference_id",
    )
])


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def norm(text):
    return text.strip()


def fetch(url, path, expected=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_name(path.name + ".part")
        urlretrieve(url, temporary)
        temporary.replace(path)
    if expected and sha(path) != expected:
        raise ValueError(f"Checksum mismatch: {path}")
    return path


def item(source, target, corpus, example_id, **extra):
    source, target = norm(source), norm(target)
    if not source or not target:
        return None
    if any(
        marker in text
        for marker in ("__NULL__", "_ _ NULL _ _")
        for text in (source, target)
    ):
        return None
    return dict(
        source_normalized=source,
        target_normalized=target,
        corpus=corpus,
        example_id=str(example_id),
        doc_id=None,
        domain=None,
        reference_id=None,
    ) | extra


def selected_recipe_ids(recipes, lang, profile):
    if profile not in {"full_parallel", "accessible_parallel"}:
        raise ValueError(f"Unknown WMT23 corpus profile: {profile}")
    excluded = WMT23_ACCESSIBLE_EXCLUSIONS if profile == "accessible_parallel" else {}
    return sorted(
        corpus.replace("Statmt-news_commentary-16-", "Statmt-news_commentary-18.1-")
        for corpus in recipes[RECIPES[lang]]["train"] if corpus not in excluded
    )


def check_recipe_sources(recipes, profile):
    """Fail before large downloads if any required recipe URL is unavailable."""
    from mtdata.data import Dataset as MTData
    from mtdata.entry import DatasetId

    ids = sorted({
        corpus for lang in TRAIN for corpus in selected_recipe_ids(recipes, lang, profile)
    })
    entries = sorted(MTData.resolve_entries([DatasetId.parse(value) for value in ids]),
                     key=lambda entry: str(entry.did))

    def check(entry):
        url = entry.url
        if not isinstance(url, str):
            raise TypeError(f"Expected one source URL for {entry.did}, got {url!r}")
        if url.startswith("http://"):
            url = "https://" + url[7:]
        entry.url = url
        try:
            with urlopen(Request(url, headers={"Range": "bytes=0-63"}), timeout=30) as response:
                prefix = response.read(64).lstrip().lower()
                if response.status not in (200, 206):
                    raise ValueError(f"HTTP {response.status}")
                if prefix.startswith((b"<!doctype html", b"<html")):
                    raise ValueError("Returned an HTML page instead of corpus data")
        except Exception as error:
            return f"{entry.did}: {url}: {error}"
        return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        failures = [error for error in pool.map(check, entries) if error]
    if failures:
        raise RuntimeError(f"Required {profile} sources are unavailable; no additional corpus was omitted:\n" + "\n".join(failures))
    print(f"Verified source responses for all {len(entries)} WMT23 resources.", flush=True)


def rows(resource):
    files = [Path(path) for path in resource["paths"]]

    def op(path):
        if path.suffix == ".gz":
            return gzip.open(path, "rt", encoding="utf-8")
        return path.open(encoding="utf-8")

    if resource["kind"] == "merged":
        with op(files[0]) as stream:
            for line in stream:
                source, target, metadata = line.rstrip("\n").split("\t")
                corpus, example_id = json.loads(metadata)
                yield item(json.loads(source), json.loads(target), corpus, example_id)
    elif resource["kind"] == "czeng":
        with op(files[0]) as stream:
            for index, line in enumerate(stream):
                if not line.strip():
                    continue
                fields = line.rstrip("\r\n").split("\t")
                if len(fields) != 6:
                    raise ValueError(
                        f"Invalid CzEng row: {files[0]}:{index}"
                    )
                # Official columns: ID, scores x3, Czech, English.
                row = item(
                    fields[5], fields[4], resource["id"],
                    f'{resource["id"]}:{index}:{fields[0]}',
                )
                if row:
                    yield row
    else:
        with op(files[0]) as sources, op(files[1]) as targets:
            for index, (source, target) in enumerate(
                zip(sources, targets, strict=True)
            ):
                row = item(
                    source, target, resource["id"],
                    f'{resource["id"]}:{index}',
                )
                if row:
                    yield row


def merge_deduplicate(resources, path, buffer_size="1G"):
    """Exact, disk-backed pair deduplication; preserve the first corpus/row ID.

    JSON quoting escapes tabs/newlines without changing sentence contents.
    A GNU-compatible sort compares only the first two fields, in reproducible byte order.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    usable_counts = {}
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for resource in sorted(resources, key=lambda resource: resource["id"]):
            count = 0
            for row in rows(resource):
                fields = (row["source_normalized"], row["target_normalized"],
                          [row["corpus"], row["example_id"]])
                stream.write("\t".join(json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                                       for value in fields) + "\n")
                count += 1
            if not count:
                raise ValueError(f"No usable rows: {resource['id']}")
            usable_counts[resource["id"]] = count
            print(f"Merge {resource['id']}: {count} usable rows", flush=True)
    # Never hold all sentence pairs in RAM. Temporary files live on the raw-data disk.
    with tempfile.TemporaryDirectory(prefix="wmt23-sort-", dir=path.parent) as temporary:
        subprocess.run([
            "sort", "--stable", "--unique", "--field-separator=\t", "--key=1,1", "--key=2,2",
            f"--buffer-size={buffer_size}", "--parallel=2", f"--temporary-directory={temporary}",
            "--output", str(path), str(path),
        ], check=True, env={**os.environ, "LC_ALL": "C"})
    return {"id": path.stem, "kind": "merged", "paths": [str(path)]}, usable_counts


def write_arrow(
    root, stem, records, file_manifest,
    shard_rows=1_000_000,
):
    names, total = [], 0
    writer, sink, batch = None, None, []

    def flush():
        if batch:
            writer.write_table(
                pa.Table.from_pylist(batch, schema=SCHEMA)
            )
            batch.clear()

    def finish():
        flush()
        writer.close()
        sink.close()
        path = root / names[-1]
        file_manifest[names[-1]] = {
            "sha256": sha(path),
            "bytes": path.stat().st_size,
        }

    try:
        for row in records:
            if total % shard_rows == 0:
                if writer is not None:
                    finish()
                    writer = sink = None
                name = f"{stem}.{len(names):05d}.arrow"
                names.append(name)
                sink = pa.OSFile(str(root / name), "wb")
                writer = pa.ipc.new_stream(sink, SCHEMA)
            batch.append(row)
            total += 1
            if len(batch) == 10000:
                flush()
        if writer is not None:
            finish()
            writer = sink = None
    finally:
        if writer is not None:
            writer.close()
        if sink is not None:
            sink.close()

    if total == 0:
        raise ValueError(f"Empty split: {stem}")
    return names, total


def prepare_alma_ja(args):
    """An explicit extension of Mid-Align: ALMA de/cs, Japanese human data, OPUS.

    Keep ALMA rows intact. Japanese follows ALMA's collection window: WMT17-20
    (Japanese exists only in WMT20), plus FLORES dev/devtest; WMT21 is validation.
    Each bilingual pair supplies both SFT directions. WMT23 is evaluation only.
    """
    script_hash = sha(__file__)
    root = args.output_dir.resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError("Use an empty output_dir; manifest.json is written last.")
    root.mkdir(parents=True, exist_ok=True)
    training, unseen = WMT23_PARTITIONS["alma_ja_opus"]
    manifest = {
        "schema_version": 2, "dataset": "wmt23", "corpus_profile": "alma_ja_opus",
        "source_lang": "en", "training_langs": list(training), "out_inference_langs": list(unseen),
        "data_seed": args.seed, "eval_revision": HF_WMT23_REV, "opus_revision": OPUS_REV,
        "alma_revision": ALMA_REV, "flores_revision": FLORES_REV,
        "files": {}, "splits": {}, "counts": {}, "train_resources": {}, "opus_files": {},
        "input_sources": {}, "preparation_script_sha256": script_hash,
        "runtime_versions": {name: version(name) for name in ("datasets", "pyarrow", "sacrebleu")},
        "policy": {
            "directions": "bidirectional", "train_sample_cap": None, "domain_filter": None,
            "text_normalization": "none; preserve upstream text", "cross_corpus_deduplication": False,
            "alignment": "OPUS-100 seen-language train only",
            "test": "haoranxu/WMT23-Test; cs-en reverses en-cs",
            "validation": "ALMA validation for de/cs; WMT21 en-ja for Japanese, both directions",
            "japanese_extension": "WMT20 en-ja + ja-en, FLORES-200 dev + devtest",
            "differences_from_mid_align": ["de/cs/ja supervised; zh/ru/uk transfer", "Japanese SFT extension", "separate OPUS-100 alignment"],
        },
    }

    def register(path):
        relative = str(path.relative_to(root))
        manifest["files"][relative] = {"sha256": sha(path), "bytes": path.stat().st_size}
        return relative

    def freeze_hf(repo, revision, filename, prefix):
        cached = hf_hub_download(repo, filename, repo_type="dataset", revision=revision)
        path = root / "inputs" / prefix / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cached, path)
        relative = register(path)
        manifest["input_sources"][relative] = {"repo": repo, "revision": revision, "filename": filename}
        return path

    def record(source, target, corpus, index):
        if not isinstance(source, str) or not source.strip() or not isinstance(target, str) or not target.strip():
            raise ValueError(f"Empty/non-string upstream pair: {corpus}/{index}")
        return dict(source_normalized=source, target_normalized=target, corpus=corpus,
                    example_id=f"{corpus}:{index}", doc_id=None, domain=None, reference_id=None)

    def save(split, direction, records):
        names, count = write_arrow(root, f"{split}.{direction}", records, manifest["files"])
        manifest["splits"].setdefault(split, {})[direction] = names
        manifest["counts"].setdefault(split, {})[direction] = count

    # Store the exact HF benchmark used by the paper's public inference code.
    blocked = set()
    heldout_pairs = set()
    for lang in training + unseen:
        for direction in (f"en-{lang}", f"{lang}-en"):
            config_name = "en-cs" if direction == "cs-en" else direction
            filename = ("en-cs/test-00000-of-00001-88c4a5c266ea101b.parquet" if config_name == "en-cs"
                        else f"{config_name}/test-00000-of-00001.parquet")
            path = root / "inputs" / "wmt23" / filename
            if not path.exists():
                path = freeze_hf("haoranxu/WMT23-Test", HF_WMT23_REV, filename, "wmt23")
            src, tgt = direction.split("-")
            records = []
            for index, row in enumerate(pq.read_table(path).to_pylist()):
                pair = row[config_name]
                result = record(pair[src], pair[tgt], f"WMT23-Test/{direction}", index)
                result["reference_id"] = "HF-packaged-reference"
                records.append(result)
                blocked.add(pair["en"].strip())
                heldout_pairs.add((lang, pair["en"].strip(), pair[lang].strip()))
            save("test", direction, records)

    parallel = {split: {lang: [] for lang in training} for split in ("train", "validation")}
    for lang in ("de", "cs"):
        manifest["train_resources"][lang] = {"dataset": "haoranxu/ALMA-Human-Parallel", "revision": ALMA_REV}
        for split, filename in ALMA_FILES[lang].items():
            path = freeze_hf("haoranxu/ALMA-Human-Parallel", ALMA_REV, filename, "alma")
            for index, row in enumerate(pq.read_table(path).to_pylist()):
                pair = row["translation"]
                parallel[split][lang].append((pair["en"], pair[lang], f"ALMA/{lang}/{split}", index))

    from sacrebleu import DATASETS
    # Save the extracted source/reference files as immutable local inputs too.
    wmt_sources = []
    for year, direction, split in (("wmt20", "en-ja", "train"), ("wmt20", "ja-en", "train"),
                                    ("wmt21", "en-ja", "validation")):
        dataset = DATASETS[year]
        references = dataset.get_reference_files(direction)
        if len(references) != 1:
            raise ValueError(f"Expected one Japanese human reference: {year}/{direction}")
        texts = []
        for filename in (dataset.get_source_file(direction), references[0]):
            path = root / "inputs" / year / Path(filename).name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(filename, path)
            relative = register(path)
            manifest["input_sources"][relative] = {"urls": dataset.data, "archive_md5": dataset.md5,
                                                   "sacrebleu": version("sacrebleu")}
            texts.append(path.read_text(encoding="utf-8").splitlines())
        for index, (source, target) in enumerate(zip(*texts, strict=True)):
            en, ja = (source, target) if direction == "en-ja" else (target, source)
            parallel[split]["ja"].append((en, ja, f"{year}/{direction}", index))
        wmt_sources.append({"dataset": year, "direction": direction, "split": split, "pairs": len(texts[0])})
    for split in ("dev", "devtest"):
        tables = []
        for lang in ("eng_Latn", "jpn_Jpan"):
            path = freeze_hf("facebook/flores", FLORES_REV,
                             f"data/language/{lang}/{split}-00000-of-00001.parquet", "flores")
            tables.append(pq.read_table(path).to_pylist())
        for en, ja in zip(*tables, strict=True):
            if en["id"] != ja["id"] or en["URL"] != ja["URL"]:
                raise ValueError(f"FLORES parallel row identity mismatch: {split}")
            parallel["train"]["ja"].append((en["sentence"], ja["sentence"], f"FLORES-200/{split}", en["id"]))
    manifest["train_resources"]["ja"] = {"dataset": "Japanese extension of ALMA collection recipe",
                                          "wmt_sources": wmt_sources, "flores_revision": FLORES_REV}

    # Audit exact bilingual overlap without silently changing ALMA's original pool.
    for lang in training:
        for en, target, _, _ in parallel["validation"][lang]:
            blocked.add(en.strip())
            heldout_pairs.add((lang, en.strip(), target.strip()))
    overlap = []
    for lang in training:
        for en, target, corpus, index in parallel["train"][lang]:
            if (lang, en.strip(), target.strip()) in heldout_pairs:
                overlap.append({"lang": lang, "corpus": corpus, "row": index})
        for split in ("train", "validation"):
            pairs = parallel[split][lang]
            save(split, f"en-{lang}", (record(en, target, corpus, index) for en, target, corpus, index in pairs))
            save(split, f"{lang}-en", (record(target, en, corpus, index) for en, target, corpus, index in pairs))
        manifest["train_resources"][lang]["num_parallel_pairs"] = len(parallel["train"][lang])
    manifest["exact_sft_heldout_overlap"] = overlap
    if overlap:
        raise ValueError(f"Exact SFT/heldout parallel pairs overlap ({len(overlap)}); inspect upstream data before training.")

    # The user's alignment protocol remains OPUS, with no unseen-language train.
    for lang in training + unseen:
        pair = "-".join(sorted(("en", lang)))
        manifest["opus_files"][pair] = {}
        for split in (("train", "validation", "test") if lang in training else ("validation", "test")):
            path = freeze_hf("Helsinki-NLP/opus-100", OPUS_REV,
                             f"{pair}/{split}-00000-of-00001.parquet", "opus")
            manifest["opus_files"][pair][split] = str(path.relative_to(root))
            if split != "train":
                blocked.update(row["translation"]["en"].strip() for row in pq.read_table(path).to_pylist())
    path = root / "excluded_alignment_sources.json"
    path.write_text(json.dumps(sorted(blocked), ensure_ascii=False), encoding="utf-8")
    register(path)
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    print("MANIFEST_SHA256=" + sha(path), flush=True)
    print(json.dumps(manifest["counts"], indent=2), flush=True)


def prepare(args):
    if args.corpus_profile == "alma_ja_opus":
        return prepare_alma_ja(args)
    preparation_script_sha256 = sha(__file__)
    if version("mtdata") != "0.4.3":
        raise ValueError("Use mtdata==0.4.3")

    if "cs" in TRAIN:
        if args.czeng_dir is None:
            raise ValueError("--czeng_dir is required when training Czech.")
        for name in CZENG:
            if not (args.czeng_dir / name).is_file():
                raise FileNotFoundError(
                    f"Download the registered CzEng file first: {name}"
                )

    root = args.output_dir.resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError(
            "Use an empty output_dir; manifest.json is written last."
        )
    root.mkdir(parents=True, exist_ok=True)
    raw = args.mtdata_dir.resolve()
    raw.mkdir(parents=True, exist_ok=True)

    recipe_path = fetch(
        RECIPE_URL, raw / "wmt23.recipe.yml", RECIPE_SHA
    )
    recipes = {
        recipe["id"]: recipe
        for recipe in yaml.safe_load(recipe_path.read_bytes())
    }
    check_recipe_sources(recipes, args.corpus_profile)
    manifest = {
        "schema_version": 1,
        "dataset": "wmt23",
        "corpus_profile": args.corpus_profile,
        "source_lang": "en",
        "training_langs": list(TRAIN),
        "out_inference_langs": list(OUT),
        "data_seed": args.seed,
        "recipe_sha256": RECIPE_SHA,
        "eval_revision": EVAL_REV,
        "opus_revision": OPUS_REV,
        "files": {},
        "splits": {},
        "counts": {},
        "train_resources": {},
        "opus_files": {},
        "input_files": {},
        "official_files": {},
        "runtime_versions": {
            package: version(package)
            for package in ("torch", "datasets", "pyarrow", "mtdata")
        },
        "policy": {
            "pool": "all_nonempty_parallel_rows_except_exact_heldout_sources",
            "domain_filter": None,
            "train_sample_cap": None,
            "cross_corpus_deduplication": args.corpus_profile == "accessible_parallel",
            "deduplication_key": "exact_stripped_source_target_within_language" if args.corpus_profile == "accessible_parallel" else None,
            "deduplication_order": "LC_ALL=C JSON source,target; keep first row in corpus-ID order" if args.corpus_profile == "accessible_parallel" else None,
            "excluded_sources": dict(WMT23_ACCESSIBLE_EXCLUSIONS) if args.corpus_profile == "accessible_parallel" else {},
            "validation_per_language": args.validation_per_language,
            "monolingual_training": False,
            "news_commentary": "replace recipe v16 with official-page v18.1",
            "validation": "seeded_reservoir_from_training_resources",
            "test": "all_general_mt_rows_all_domains_no_test_holdout",
        },
    }
    if args.corpus_profile == "accessible_parallel":
        manifest["runtime_versions"]["sort"] = subprocess.run(
            ["sort", "--version"], check=True, capture_output=True, text=True
        ).stdout.splitlines()[0]

    def register(path):
        relative = str(path.relative_to(root))
        manifest["files"][relative] = {
            "sha256": sha(path),
            "bytes": path.stat().st_size,
        }
        return relative

    def save_split(split, lang, records):
        names, count = write_arrow(
            root, f"{split}.{lang}", records, manifest["files"]
        )
        manifest["splits"].setdefault(split, {})[lang] = names
        manifest["counts"].setdefault(split, {})[lang] = count

    # Keep every official General MT evaluation row.
    blocked = set()
    for lang in TRAIN + OUT:
        reference = "refB" if lang == "he" else "refA"
        base = (
            "https://raw.githubusercontent.com/"
            f"wmt-conference/wmt23-news-systems/{EVAL_REV}/txt"
        )
        paths = [
            ("sources", f"generaltest2023.en-{lang}.src.en"),
            (
                "references",
                f"generaltest2023.en-{lang}.ref.{reference}.{lang}",
            ),
            ("metainfo", f"generaltest2023.en-{lang}.meta.jsonl"),
        ]
        content = []
        for folder, name in paths:
            url = f"{base}/{folder}/{name}"
            path = fetch(url, raw / "official" / folder / name)
            manifest["official_files"][url] = sha(path)
            content.append(
                path.read_text(encoding="utf-8").splitlines()
            )

        test = []
        for index, (source, target, metadata) in enumerate(
            zip(*content, strict=True)
        ):
            info = json.loads(metadata)
            row = item(
                source, target, "generaltest2023",
                f"en-{lang}:{index}",
                doc_id=info["docid"],
                domain=info["domain"],
                reference_id=reference,
            )
            if row is None:
                raise ValueError(
                    f"Empty official source/reference: en-{lang}:{index}"
                )
            test.append(row)
            blocked.add(row["source_normalized"])

        expected = 557 if lang == "de" else 2074
        if len(test) != expected:
            raise ValueError(
                f"Unexpected official count: {lang}: {len(test)}"
            )
        save_split("test", lang, test)

    # Freeze OPUS train for seen languages and evaluation for all six.
    for lang in TRAIN + OUT:
        pair = "-".join(sorted(("en", lang)))
        manifest["opus_files"][pair] = {}
        splits = (
            ("train", "validation", "test")
            if lang in TRAIN
            else ("validation", "test")
        )
        for split in splits:
            name = f"{pair}/{split}-00000-of-00001.parquet"
            path = Path(hf_hub_download(
                "Helsinki-NLP/opus-100",
                name,
                repo_type="dataset",
                revision=OPUS_REV,
                local_dir=root / "opus",
            ))
            manifest["opus_files"][pair][split] = register(path)
            if split != "train":
                table = pq.read_table(path, columns=["translation"])
                for row in table.to_pylist():
                    blocked.add(norm(row["translation"]["en"]))

    from mtdata.data import Dataset as MTData
    from mtdata.entry import DatasetId, Langs

    resources = {}
    for lang in TRAIN:
        original_ids = recipes[RECIPES[lang]]["train"]
        corpus_ids = selected_recipe_ids(recipes, lang, args.corpus_profile)
        part_root = raw / RECIPES[lang]
        policy = {
            "version": "0.4.3",
            "ids": corpus_ids,
            "drop_noise": False,
        }
        marker = part_root / "extraction_policy.json"
        if part_root.exists() and any(part_root.iterdir()):
            if (
                not marker.is_file()
                or json.loads(marker.read_text()) != policy
            ):
                raise ValueError(
                    "Unknown extraction policy; use a new raw directory: "
                    f"{part_root}"
                )
        part_root.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            json.dumps(policy, sort_keys=True), encoding="utf-8"
        )

        entries = MTData.resolve_entries([
            DatasetId.parse(corpus) for corpus in corpus_ids
        ])
        for entry in entries:
            if (
                isinstance(entry.url, str)
                and entry.url.startswith("http://")
            ):
                entry.url = "https://" + entry.url[7:]

        MTData.prepare(
            langs=Langs(f"eng-{ISO3[lang]}"),
            out_dir=part_root,
            dataset_ids={
                "train": [
                    DatasetId.parse(corpus)
                    for corpus in corpus_ids
                ]
            },
            drop_noise=(False, False),
            compress=True,
            merge_train=False,
            fail_on_error=True,
            n_jobs=1,
        )

        resources[lang] = []
        for corpus in corpus_ids:
            paths = {}
            for code in corpus.split("-")[-2:]:
                stem = part_root / "train-parts" / f"{corpus}.{code}"
                candidates = [
                    path
                    for path in (stem, Path(str(stem) + ".gz"))
                    if path.is_file()
                ]
                if len(candidates) != 1:
                    raise ValueError(
                        f"Missing/ambiguous extracted corpus: {stem}"
                    )
                paths[code.split("_")[0]] = str(candidates[0])

            resources[lang].append({
                "id": corpus,
                "kind": "parallel",
                "paths": [paths["eng"], paths[ISO3[lang]]],
            })

        if lang == "cs":
            resources[lang] += [
                {
                    "id": name[:-3],
                    "kind": "czeng",
                    "paths": [
                        str((args.czeng_dir / name).resolve())
                    ],
                }
                for name in CZENG
            ]

        manifest["train_resources"][lang] = {
            "recipe_id": RECIPES[lang],
            "original_recipe_ids": original_ids,
            "effective_recipe_ids": corpus_ids,
            "excluded_recipe_ids": {
                corpus: WMT23_ACCESSIBLE_EXCLUSIONS[corpus]
                for corpus in original_ids
                if args.corpus_profile == "accessible_parallel" and corpus in WMT23_ACCESSIBLE_EXCLUSIONS
            },
            "sources": resources[lang],
            "urls": {
                str(entry.did): entry.url
                for entry in entries
            },
        }
        for resource in resources[lang]:
            for path in resource["paths"]:
                manifest["input_files"][path] = sha(path)

    # First pass: hold only a small validation reservoir in memory.
    validation = {}
    for lang in TRAIN:
        if args.corpus_profile == "accessible_parallel":
            merged, usable_counts = merge_deduplicate(
                resources[lang], raw / "merged-accessible_parallel" / f"{lang}.tsv"
            )
            resources[lang] = [merged]
            manifest["train_resources"][lang]["usable_counts"] = usable_counts
            manifest["train_resources"][lang]["merged_sha256"] = sha(merged["paths"][0])
        rng = random.Random(f"{args.seed}:{lang}")
        reservoir, eligible, usable_counts = [], 0, {}
        unique_counts = Counter()

        for resource in resources[lang]:
            usable = 0
            for row in rows(resource):
                usable += 1
                unique_counts[row["corpus"]] += 1
                if row["source_normalized"] in blocked:
                    continue
                eligible += 1

                if len(reservoir) < args.validation_per_language:
                    reservoir.append(row)
                else:
                    index = rng.randrange(eligible)
                    if index < args.validation_per_language:
                        reservoir[index] = row

            if usable == 0:
                raise ValueError(
                    f"No usable rows: {resource['id']}"
                )
            usable_counts[resource["id"]] = usable
            print(lang, resource["id"], usable, flush=True)

        if eligible <= args.validation_per_language:
            raise ValueError(
                f"Not enough training rows: {lang}"
            )
        validation[lang] = reservoir
        if args.corpus_profile == "full_parallel":
            manifest["train_resources"][lang]["usable_counts"] = usable_counts
        else:
            details = manifest["train_resources"][lang]
            details["unique_counts"] = dict(unique_counts)
            details["duplicates_removed"] = sum(details["usable_counts"].values()) - sum(unique_counts.values())
            print(f"Deduplicated {lang}: unique={sum(unique_counts.values())}, removed={details['duplicates_removed']}", flush=True)

    blocked.update(
        row["source_normalized"]
        for records in validation.values()
        for row in records
    )
    path = root / "excluded_alignment_sources.json"
    path.write_text(
        json.dumps(sorted(blocked), ensure_ascii=False),
        encoding="utf-8",
    )
    register(path)

    # Second pass: persist every eligible row in canonical order.
    for lang in TRAIN:
        kept = Counter()

        def train_rows():
            for resource in resources[lang]:
                for row in rows(resource):
                    if row["source_normalized"] not in blocked:
                        kept[row["corpus"]] += 1
                        yield row

        save_split("train", lang, train_rows())
        save_split("validation", lang, validation[lang])
        manifest["train_resources"][lang]["retained_counts"] = (
            dict(kept)
        )
        if args.corpus_profile == "accessible_parallel":
            Path(resources[lang][0]["paths"][0]).unlink()

    manifest["preparation_script_sha256"] = preparation_script_sha256
    path = root / "manifest.json"
    path.write_text(
        json.dumps(
            manifest, ensure_ascii=False, sort_keys=True, indent=2
        ),
        encoding="utf-8",
    )
    print("MANIFEST_SHA256=" + sha(path))
    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--mtdata_dir", type=Path)
    parser.add_argument("--corpus_profile", choices=list(WMT23_PARTITIONS), default="alma_ja_opus")
    parser.add_argument("--czeng_dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--validation_per_language", type=int, default=500
    )
    args = parser.parse_args()
    if args.validation_per_language <= 0:
        parser.error("--validation_per_language must be positive")
    if args.corpus_profile != "alma_ja_opus" and args.mtdata_dir is None:
        parser.error("--mtdata_dir is required for legacy recipe profiles")
    prepare(args)
