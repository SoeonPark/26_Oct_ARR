import json
import hashlib
from functools import lru_cache
from pathlib import Path
import torch
import bisect
import re
from transformers import AutoTokenizer
from datasets import Dataset, concatenate_datasets, load_dataset
from config import WMT23_ACCESSIBLE_EXCLUSIONS, WMT23_PARTITIONS
from utils import MASSIVE_LANG_MAP, MASSIVE_SYSTEM_PROMPT


def make_sample_id(dataset_name, dataset_config, split, row_id):
    """Stable identity independent of sampling order, run, or batch position."""
    return json.dumps(
        [dataset_name, dataset_config, split, row_id],
        ensure_ascii=False, separators=(',', ':'),
    )


@lru_cache(maxsize=4096)
def _file_sha256(path, size, mtime_ns, ctime_ns):
    """Hash once per unchanged file in this process, including large Arrow shards."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_sha256(path):
    path = Path(path).resolve()
    stat = path.stat()
    return _file_sha256(str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


class AlignmentDataset(torch.utils.data.Dataset):
    def __init__(self, config, tokenizer, split='train', lang_pairs=None):
        self.config = config
        self.tokenizer = tokenizer
        self.split = split
        # When lang_pairs is given, load exactly those OPUS configs and ignore
        # the language sets derived from the experiment config. Validation
        # builds one dataset per language pair this way.
        self.lang_pairs = lang_pairs
        self.data = self.load_data(config.alignment_data)
                
    def load_data(self, data_path):
        self.all_data = dict()
        self.dataset_metadata = {}

        if self.lang_pairs is not None:
            lang_subset = list(self.lang_pairs)
        elif "train" in self.split or "in_validation" in self.split or "in_test" in self.split:
            lang_subset = [f"{self.config.training_anchor_langs}-{lang}" for lang in self.config.training_lang]
            inv_lang_subset = [f"{lang}-{self.config.training_anchor_langs}" for lang in self.config.training_lang]
            lang_subset.extend(inv_lang_subset)
        elif "out_validation" in self.split or "out_test" in self.split:
            lang_subset = [f"{self.config.training_anchor_langs}-{lang}" for lang in self.config.out_inference_lang]
            inv_lang_subset = [f"{lang}-{self.config.training_anchor_langs}" for lang in self.config.out_inference_lang]
            lang_subset.extend(inv_lang_subset)
        
        SPLIT_MAPPING = {
            'train': 'train',
            'in_validation': 'validation',
            'out_validation': 'validation',
            'in_test': 'test',
            'out_test': 'test'
        }

        frozen_wmt23 = getattr(self.config, "downstream_task", "massive") == "wmt23"
        blocked = set()
        if frozen_wmt23:
            manifest = WMT23Dataset.validate_prepared_manifest(self.config)
            root = Path(self.config.wmt23_data_dir)
            lang_subset = list(dict.fromkeys(
                "-".join(sorted(pair.split("-"))) for pair in lang_subset
            ))
            if self.split == "train":
                blocked = set(json.loads((root / "excluded_alignment_sources.json").read_text()))
        
        for lang_pair in lang_subset:
            try:
                if frozen_wmt23:
                    filename = manifest["opus_files"][lang_pair][SPLIT_MAPPING[self.split]]
                    dataset = load_dataset("parquet", data_files=str(root / filename), split="train")
                else:
                    dataset = load_dataset(data_path, lang_pair, split=SPLIT_MAPPING[self.split])
                metadata = {
                    'dataset': data_path,
                    'config': lang_pair,
                    'split': SPLIT_MAPPING[self.split],
                    'fingerprint': getattr(dataset, '_fingerprint', None),
                    'num_rows': len(dataset),
                }
                # Preserve original row identity before shuffle and selection.
                dataset = dataset.add_column(
                    '_alignment_row_index', list(range(len(dataset)))
                )
                if frozen_wmt23:
                    metadata.update(revision=manifest["opus_revision"], file_sha256=manifest["files"][filename]["sha256"])
                    if blocked:
                        dataset = dataset.filter(lambda row: row["translation"]["en"].strip() not in blocked)
                    metadata["eligible_rows"] = len(dataset)
                    if self.split == "train" and len(dataset) < self.config.alignment_num_samples_per_lang:
                        raise ValueError(f"Insufficient WMT23 alignment rows for {lang_pair}: {len(dataset)}")
                # Shuffle and sample with seed in config
                dataset = dataset.shuffle(seed=self.config.alignment_sampling_seed).select(range(min(self.config.alignment_num_samples_per_lang, len(dataset))))
                self.all_data[lang_pair] = dataset
                self.dataset_metadata[lang_pair] = metadata
                print(f"Length of {lang_pair} dataset: {len(dataset)}")
                print(f"Sample data for {lang_pair}: {dataset[0]}")
                # Sample data for en-ko: {'translation': {'en': "They're shaped like a bus.", 'ko': '할머니처럼 만들었지만.. ? 엉망이지만..'}}
                # Sample data for en-ja: {'translation': {'en': 'Yeah, Vincent Hanna.', 'ja': '- ラウール - ラウールに ヴィンセント・ハンナだ'}}
                # Sample data for en-es: {'translation': {'en': "It was the asbestos in here, that's what did it!", 'es': 'Fueron los asbestos aquí. ¡Eso es lo que ocurrió!'}}
            except Exception as e:
                if frozen_wmt23:
                    raise RuntimeError(f"Failed loading frozen WMT23 alignment {lang_pair}/{self.split}") from e
                print(f"Failed loading alignment pair {lang_pair}: {e}")

        # WMT experiments require all three alignment language pairs.
        if (
            self.split == "train"
            and getattr(self.config, "downstream_task", "massive") in {"wmt25", "wmt23"}
        ):
            required_pairs = (
                {"-".join(sorted(("en", lang))) for lang in self.config.training_lang} if frozen_wmt23
                else {"en-ko", "en-ja", "cs-en"}
            )
            loaded_pairs = set(self.all_data)

            missing = sorted(required_pairs - loaded_pairs)
            unexpected = sorted(loaded_pairs - required_pairs)
            empty = sorted(
                pair
                for pair in required_pairs & loaded_pairs
                if len(self.all_data[pair]) == 0
            )

            if missing or unexpected or empty:
                raise RuntimeError(
                    f"{self.config.downstream_task} alignment requires {sorted(required_pairs)}. "
                    f"missing={missing}, unexpected={unexpected}, empty={empty}"
                )

        # Training tolerates missing reverse configs (OPUS-100 only ships one
        # direction per pair). Validation must not: a silently empty dataset
        # would drop its eval_*_loss key without any error.
        if self.lang_pairs is not None:
            missing = [
                lang_pair for lang_pair in lang_subset
                if lang_pair not in self.all_data
            ]
            if missing:
                raise RuntimeError(
                    f"Requested alignment pairs failed to load: {missing} "
                    f"(split={self.split}, data={data_path})"
                )

    def __len__(self):
        return sum(len(dataset) for dataset in self.all_data.values())

    @property
    def pair_ranges(self):
        """Global index ranges in the same order used by __getitem__."""
        ranges = {}
        offset = 0
        for pair, dataset in self.all_data.items():
            ranges[pair] = (offset, offset + len(dataset))
            offset += len(dataset)
        return ranges

    def __getitem__(self, idx):
        for lang_pair, dataset in self.all_data.items():
            if idx < len(dataset):
                item = dict(dataset[idx])
                original_row_index = item.pop('_alignment_row_index')
                sample_id = make_sample_id(
                    self.config.alignment_data, lang_pair,
                    self.dataset_metadata[lang_pair]['split'], original_row_index,
                )
                source_lang, target_lang = lang_pair.split('-')
                source_text = item['translation'][source_lang]
                target_text = item['translation'][target_lang]
                
                # truncation=True without max_length falls back to
                # tokenizer.model_max_length (131072 for Llama-3.2), i.e. no
                # effective truncation. Keep that default so existing runs stay
                # reproducible; set --alignment_max_length to cap it.
                max_length = getattr(self.config, 'alignment_max_length', None)

                source_tokens = self.tokenizer(source_text, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
                target_tokens = self.tokenizer(target_text, return_tensors='pt', padding=True, truncation=True, max_length=max_length)

                return {
                    'source_input_ids': source_tokens['input_ids'].squeeze(0),
                    'source_attention_mask': source_tokens['attention_mask'].squeeze(0),
                    'target_input_ids': target_tokens['input_ids'].squeeze(0),
                    'target_attention_mask': target_tokens['attention_mask'].squeeze(0),
                    'lang_pair': lang_pair,
                    'sample_id': sample_id,
                    'original_row_index': original_row_index,
                    # Kept as plain text so per-sample validation logs can show
                    # the exact inputs without decoding token ids back.
                    'source_text': source_text,
                    'target_text': target_text,
                    'item': item
                }
            else:
                idx -= len(dataset)
        
        raise IndexError("Index out of range for the dataset.")
    
    def collate_fn(self, batch):
        source_input_ids = torch.nn.utils.rnn.pad_sequence(
            [item['source_input_ids'] for item in batch], batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        source_attention_mask = torch.nn.utils.rnn.pad_sequence(
            [item['source_attention_mask'] for item in batch], batch_first=True, padding_value=0
        )
        target_input_ids = torch.nn.utils.rnn.pad_sequence(
            [item['target_input_ids'] for item in batch], batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        target_attention_mask = torch.nn.utils.rnn.pad_sequence(
            [item['target_attention_mask'] for item in batch], batch_first=True, padding_value=0
        )
        
        return {
            'source_input_ids': source_input_ids,
            'source_attention_mask': source_attention_mask,
            'target_input_ids': target_input_ids,
            'target_attention_mask': target_attention_mask,
            'lang_pair': [item['lang_pair'] for item in batch],
            'sample_id': [item['sample_id'] for item in batch],
            'original_row_index': [item['original_row_index'] for item in batch],
            'source_text': [item['source_text'] for item in batch],
            'target_text': [item['target_text'] for item in batch],
            'item': [item['item'] for item in batch]
        }

class MassiveDataset(torch.utils.data.Dataset):
    """
    MASSIVE generative slot-filling dataset.

    Paper format:
        System: MASSIVE_SYSTEM_PROMPT
        User:   utterance
        Assistant:
            slot_type: surface_span; slot_type: surface_span

    Example:
        utt:
            wake me up at nine am on friday

        annot_utt:
            wake me up at [time : nine am] on [date : friday]

        target:
            time: nine am; date: friday
    """

    SPLIT_MAPPING = {
        "train": "train",
        "in_validation": "validation",
        "out_validation": "validation",
        "in_test": "test",
        "out_test": "test",
    }

    def __init__(self, config, tokenizer, split="train", languages=None):
        self.config = config
        self.tokenizer = tokenizer
        self.split = split
        # When languages is given, load exactly those locales and ignore the
        # language sets derived from the experiment config. Validation builds
        # one dataset per language this way.
        self.languages = languages

        if split not in self.SPLIT_MAPPING:
            raise ValueError(
                f"Unsupported split: {split}. "
                f"Expected one of {list(self.SPLIT_MAPPING.keys())}."
            )

        # Supervised/downstream training languages
        self.training_langs = (
            [config.training_anchor_langs]
            + list(config.training_lang)
        )

        # Unseen transfer languages
        self.out_inference_langs = list(config.out_inference_lang)

        self.lang_map = MASSIVE_LANG_MAP
        self.system_prompt = MASSIVE_SYSTEM_PROMPT

        # Maximum sequence length.
        # Llama/Qwen decoder-only models need a padding token.
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError(
                    "Tokenizer has neither pad_token_id nor eos_token_id."
                )

            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.all_data = {}
        self.dataset_metadata = {}
        self.cumulative_sizes = []

        self.load_data(config.downstream_task_data)
        self.check_prompt_prefix()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _get_languages(self):
        """
        in_*  : languages observed during downstream-task training
        out_* : languages not observed during downstream-task training
        """

        if self.languages is not None:
            return list(self.languages)

        if self.split in {
            "train",
            "in_validation",
            "in_test",
        }:
            return self.training_langs

        if self.split in {
            "out_validation",
            "out_test",
        }:
            return self.out_inference_langs

        raise ValueError(f"Invalid split: {self.split}")

    def load_data(self, data_path):
        """
        data_path should normally be:
            AmazonScience/massive
        """

        hf_split = self.SPLIT_MAPPING[self.split]
        languages = self._get_languages()

        total_size = 0

        for lang in languages:
            if lang not in self.lang_map:
                raise ValueError(
                    f"Language '{lang}' is not defined in MASSIVE_LANG_MAP."
                )

            locale = self.lang_map[lang]

            try:
                dataset = load_dataset(
                    data_path,
                    locale,
                    split=hf_split,
                    trust_remote_code=True,
                )

            except Exception as e:
                raise RuntimeError(
                    f"Failed loading MASSIVE "
                    f"lang={lang}, locale={locale}, split={hf_split}"
                ) from e

            self.all_data[lang] = dataset
            self.dataset_metadata[lang] = {
                "dataset": data_path,
                "config": locale,
                "split": hf_split,
                "fingerprint": getattr(dataset, "_fingerprint", None),
                "num_rows": len(dataset),
            }

            total_size += len(dataset)
            self.cumulative_sizes.append(total_size)

            print(
                f"[MASSIVE] lang={lang}, "
                f"locale={locale}, "
                f"split={hf_split}, "
                f"size={len(dataset)}"
            )

            if len(dataset) > 0:
                sample = dataset[0]

                print(
                    f"  utt       : {sample['utt']}"
                )
                print(
                    f"  annot_utt : {sample['annot_utt']}"
                )
                print(
                    f"  target    : "
                    f"{self.extract_slots(sample['annot_utt'])}"
                )

        return self.all_data

    def check_prompt_prefix(self):
        """Warn once if this tokenizer's template breaks the masking assumption.

        Label masking keeps loss on the assistant answer only, which requires
        the prompt rendering to tokenize as a prefix of the full rendering. This
        catches a template that does not, instead of silently training on the
        wrong span.
        """
        prompt_text, full_text = self._apply_chat_template(
            utterance="wake me up at nine am on friday",
            target="time: nine am; date: friday",
        )

        prompt_ids = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
        )["input_ids"]
        full_ids = self.tokenizer(
            full_text,
            add_special_tokens=False,
        )["input_ids"]

        if full_ids[: len(prompt_ids)] != prompt_ids:
            print(
                "[MASSIVE][WARNING] The chat template does not tokenize the "
                "prompt as a prefix of the full sequence. Label masking falls "
                "back to the common prefix, so some template scaffolding will "
                "be included in the loss. Check the assistant header for this "
                f"model. prompt_tokens={len(prompt_ids)}, "
                f"full_tokens={len(full_ids)}"
            )

    # ------------------------------------------------------------------
    # MASSIVE slot conversion
    # ------------------------------------------------------------------

    @staticmethod
    def extract_slots(annot_utt):
        """
        Convert MASSIVE annotated utterance to the generative target
        used in the paper/repository.

        Example
        -------
        Input:
            wake me up at [time : nine am] on [date : friday]

        Output:
            time: nine am; date: friday

        No slot:
            olly quiet
        ->
            None
        """

        if not annot_utt:
            return "None"

        # Same basic operation as the original mid-align repository:
        # extract everything inside [...]
        matches = re.findall(
            r"\[([^\[\]]+?)\]",
            annot_utt,
        )

        if not matches:
            return "None"

        normalized_slots = []

        for match in matches:
            # MASSIVE representation:
            #
            #     time : nine am
            #
            # Paper output:
            #
            #     time: nine am
            #
            # Only normalize the separator.
            match = re.sub(
                r"\s+:\s+",
                ": ",
                match.strip(),
                count=1,
            )

            normalized_slots.append(match)

        return "; ".join(normalized_slots)

    # ------------------------------------------------------------------
    # Chat formatting
    # ------------------------------------------------------------------

    def _apply_chat_template(
        self,
        utterance,
        target=None,
        *,
        system_prompt=None,
    ):
        """
        Produce:
            prompt_text
            full_text

        prompt_text:
            system + user + assistant generation header

        full_text:
            system + user + assistant target
        """

        system_prompt = self.system_prompt if system_prompt is None else system_prompt
        prompt_messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": utterance,
            },
        ]

        # Preferred path for Llama-Instruct / Qwen-Instruct.
        if (
            hasattr(self.tokenizer, "apply_chat_template")
            and self.tokenizer.chat_template is not None
        ):
            # Qwen3-style templates open a <think> block in the generation
            # prompt. That leaves the answer preceded by "</think>", which the
            # slot parser reads as part of the first slot name and scores as a
            # false positive, and it breaks the prompt-prefix assumption used
            # for label masking below. Templates that do not reference this
            # flag ignore it, so it is safe for Llama and Qwen2.5.
            template_kwargs = {"enable_thinking": False}

            prompt_text = self.tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
                **template_kwargs,
            )

            if target is None:
                return prompt_text, None

            full_messages = prompt_messages + [
                {
                    "role": "assistant",
                    "content": target,
                }
            ]

            full_text = self.tokenizer.apply_chat_template(
                full_messages,
                tokenize=False,
                add_generation_prompt=False,
                **template_kwargs,
            )

            return prompt_text, full_text

        # Fallback for tokenizers without a chat template.
        prompt_text = (
            f"System: {system_prompt}\n"
            f"User: {utterance}\n"
            f"Assistant:"
        )

        if target is None:
            return prompt_text, None

        full_text = (
            f"{prompt_text} {target}"
            f"{self.tokenizer.eos_token or ''}"
        )

        return prompt_text, full_text

    # ------------------------------------------------------------------
    # Dataset indexing
    # ------------------------------------------------------------------

    def __len__(self):
        if not self.cumulative_sizes:
            return 0

        return self.cumulative_sizes[-1]

    def _resolve_index(self, idx):
        """
        Convert global index -> (language, local index)
        without storing one tuple per MASSIVE sample.
        """

        if idx < 0:
            idx += len(self)

        if idx < 0 or idx >= len(self):
            raise IndexError(
                f"Index {idx} out of range for dataset "
                f"of size {len(self)}."
            )

        dataset_idx = bisect.bisect_right(
            self.cumulative_sizes,
            idx,
        )

        previous_size = (
            0
            if dataset_idx == 0
            else self.cumulative_sizes[dataset_idx - 1]
        )

        local_idx = idx - previous_size
        lang = list(self.all_data.keys())[dataset_idx]

        return lang, local_idx

    # ------------------------------------------------------------------
    # Training sample
    # ------------------------------------------------------------------

    def __getitem__(self, idx):
        lang, local_idx = self._resolve_index(idx)

        item = self.all_data[lang][local_idx]

        utterance = item["utt"]
        annot_utt = item["annot_utt"]

        target = self.extract_slots(annot_utt)

        prompt_text, full_text = self._apply_chat_template(
            utterance=utterance,
            target=target,
        )

        # Chat template already inserts its own model-specific
        # special tokens, so do not add them a second time.
        full_tokens = self.tokenizer(
            full_text,
            add_special_tokens=False,
        )

        # Do NOT truncate the prompt separately before measuring its
        # length. Otherwise we can obtain incorrect masking around the
        # max_length boundary.
        prompt_tokens = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
        )

        input_ids = torch.tensor(
            full_tokens["input_ids"],
            dtype=torch.long,
        )

        attention_mask = torch.tensor(
            full_tokens["attention_mask"],
            dtype=torch.long,
        )

        labels = input_ids.clone()

        # --------------------------------------------------------------
        # IMPORTANT:
        # Ignore system prompt + user utterance + assistant header.
        # Compute LM loss only on the assistant slot-filling output.
        # --------------------------------------------------------------
        # Masking assumes the prompt tokens are an exact prefix of the full
        # tokens. A chat template can merge whitespace differently between the
        # two renderings, so measure the real common prefix. Trusting the raw
        # prompt length would mask answer tokens when they diverge.
        prompt_length = 0
        for prompt_token, full_token in zip(
            prompt_tokens["input_ids"],
            full_tokens["input_ids"],
        ):
            if prompt_token != full_token:
                break
            prompt_length += 1

        labels[:prompt_length] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,

            # Metadata useful for evaluation/debugging.
            "lang": lang,
            "sample_id": make_sample_id(
                self.config.downstream_task_data, self.lang_map[lang],
                self.SPLIT_MAPPING[self.split], item["id"],
            ),
            "utt": utterance,
            "target": target,

            "item": item,
        }

    # ------------------------------------------------------------------
    # Batch collation
    # ------------------------------------------------------------------

    def collate_fn(self, batch):
        pad_token_id = self.tokenizer.pad_token_id

        input_ids = torch.nn.utils.rnn.pad_sequence(
            [x["input_ids"] for x in batch],
            batch_first=True,
            padding_value=pad_token_id,
        )

        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [x["attention_mask"] for x in batch],
            batch_first=True,
            padding_value=0,
        )

        labels = torch.nn.utils.rnn.pad_sequence(
            [x["labels"] for x in batch],
            batch_first=True,
            padding_value=-100,
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,

            "lang": [x["lang"] for x in batch],
            "sample_id": [x["sample_id"] for x in batch],
            "utt": [x["utt"] for x in batch],
            "target": [x["target"] for x in batch],
            "item": [x["item"] for x in batch],
        }
        
class WMT25Dataset(MassiveDataset):
    """English-to-target causal LM data prepared by scripts/prepare_wmt25.py.

    Reuse MASSIVE's chat rendering, indexing and supervised padding only.
    All splits provide supervised labels.
    Generation uses get_generation_sample() without feeding gold targets.
    """

    source_lang = "en"
    training_langs = ("ko", "ja", "cs")
    out_inference_langs = ("et", "ru", "ar")

    LANGUAGE_NAMES = {
        "ko": "Korean",
        "ja": "Japanese",
        "cs": "Czech",
        "et": "Estonian",
        "ru": "Russian",
        "ar": "Egyptian Arabic",
    }

    SPLIT_MAPPING = {
        "train": "train",
        "in_validation": "validation",
        "out_validation": "validation",
        "in_test": "test",
        "out_test": "test",
    }

    def __init__(self, config, tokenizer, split="train", languages=None):
        self.config, self.tokenizer, self.split = config, tokenizer, split
        self.languages = languages
        # Configurations saved before this option used language-balanced updates.
        self.downstream_sampling = getattr(config, "wmt25_downstream_sampling", "language_balanced")
        if self.downstream_sampling not in ("proportional", "language_balanced"):
            raise ValueError(f"Unknown WMT25 downstream sampling: {self.downstream_sampling!r}.")

        anchor_lang = getattr(config, "training_anchor_langs", None)
        if anchor_lang != self.source_lang:
            raise ValueError(
                f"WMT25 training_anchor_langs must be {self.source_lang!r}, "
                f"got {anchor_lang!r}."
            )

        if split not in self.SPLIT_MAPPING:
            raise ValueError(
                f"Unsupported WMT25 split {split!r}; "
                f"expected one of {list(self.SPLIT_MAPPING)}."
            )
        assert set(self.training_langs).isdisjoint(self.out_inference_langs)
        allowed = (
            self.out_inference_langs if split.startswith("out_") else self.training_langs
        )
        selected = self._get_languages()
        if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(allowed):
            raise ValueError(f"WMT25 {split} languages must be drawn from {allowed}.")
        if split == "train" and set(selected) != set(self.training_langs):
            raise ValueError("WMT25 training requires all of ko, ja and cs.")
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id.")
            tokenizer.pad_token = tokenizer.eos_token
        self.all_data, self.dataset_metadata, self.cumulative_sizes = {}, {}, []
        if not getattr(config, "wmt25_data_dir", None):
            raise ValueError("WMT25 requires --wmt25_data_dir; run scripts/prepare_wmt25.py first.")
        self.load_data(config.wmt25_data_dir)

    @classmethod
    def validate_prepared_manifest(cls, config):
        """Check data identity before allocating model weights or loading Arrow files."""
        root = Path(config.wmt25_data_dir).expanduser()
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if (manifest["source_lang"] != cls.source_lang
                or set(manifest["training_langs"]) != set(cls.training_langs)
                or set(manifest["out_inference_langs"]) != set(cls.out_inference_langs)):
            raise ValueError("WMT25 manifest does not match the required language partition.")
        if manifest["seed"] != getattr(config, "training_seed", 42):
            raise ValueError("WMT25 prepared sampling seed must match --training_seed.")
        # Legacy checkpoints have no requested profile; retain their original data.
        expected_profile = getattr(config, "wmt25_corpus_profile", None)
        if expected_profile is not None and manifest.get("corpus_profile") != expected_profile:
            raise ValueError(
                f"WMT25 corpus_profile mismatch: requested {expected_profile!r}, "
                f"prepared {manifest.get('corpus_profile')!r} in {root}. "
                "Prepare a separate directory with scripts/prepare_wmt25.py "
                f"--corpus_profile {expected_profile}; do not relabel an existing manifest."
            )
        if manifest.get("corpus_profile") == "full_recipe":
            for lang in cls.training_langs:
                resource = manifest.get("train_resources", {}).get(lang, {})
                selected = set(resource.get("corpus_ids", []))
                recipe_ids = resource.get("recipe_train_corpus_ids")
                if not selected or (recipe_ids is not None and selected != set(recipe_ids)):
                    raise ValueError(f"WMT25 full_recipe is incomplete for {lang}.")
        return manifest

    def load_data(self, data_path):
        root = Path(data_path).expanduser()
        manifest = self.validate_prepared_manifest(self.config)
        self.manifest = manifest
        hf_split = self.SPLIT_MAPPING[self.split]
        for lang in self._get_languages():
            path = root / f"{hf_split}.{lang}.jsonl"
            dataset = load_dataset("json", data_files={hf_split: str(path)}, split=hf_split)
            expected = manifest["counts"][hf_split][lang]
            if not len(dataset) or len(dataset) != expected:
                raise ValueError(f"WMT25 {hf_split}/{lang}: expected {expected} rows, got {len(dataset)}.")
            for item in dataset:
                if (item["src_lang"] != "en" or item["tgt_lang"] != lang
                        or item["direction"] != f"en-{lang}"
                        or item["dataset"] != "wmt25" or item["split"] != hf_split):
                    raise ValueError(f"Incorrect WMT25 direction/split in {path}.")
                if not item["source_normalized"].strip():
                    raise ValueError(f"Empty WMT25 source in {path}.")
                if not (item["target_normalized"] or "").strip():
                    raise ValueError(f"Missing supervised WMT25 target in {path}.")
            self.all_data[lang] = dataset
            self.cumulative_sizes.append(len(dataset) + (self.cumulative_sizes[-1] if self.cumulative_sizes else 0))
            self.dataset_metadata[lang] = {
                "dataset": "wmt25", "config": f"en-{lang}", "split": hf_split,
                "num_rows": len(dataset), "fingerprint": dataset._fingerprint,
                "manifest": manifest,
            }

    @property
    def language_ranges(self):
        start, ranges = 0, {}
        for lang, dataset in self.all_data.items():
            ranges[lang] = (start, start + len(dataset))
            start += len(dataset)
        return ranges

    def _apply_chat_template(self, utterance, target=None, *, lang):
        return super()._apply_chat_template(
            utterance,
            target,
            system_prompt=(
                "Translate the following sentences from English to "
                f"{self.LANGUAGE_NAMES[lang]}."
            ),
        )

    def get_generation_sample(self, idx):
        lang, local_idx = self._resolve_index(idx)
        item = self.all_data[lang][local_idx]
        return {
            "lang": lang,
            "sample_id": make_sample_id("wmt25", item["direction"], item["split"], item["example_id"]),
            "utt": item["source_normalized"],
            "target": item["target_normalized"], "item": item,
        }

    def __getitem__(self, idx):
        sample = self.get_generation_sample(idx)
        prompt, full = self._apply_chat_template(
            sample["utt"], sample["target"], lang=sample["lang"],
        )
        full_tokens = self.tokenizer(full, add_special_tokens=False)
        prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)
        input_ids = torch.tensor(full_tokens["input_ids"], dtype=torch.long)
        labels = input_ids.clone()
        prefix_length = 0
        for prompt_id, full_id in zip(prompt_tokens["input_ids"], full_tokens["input_ids"]):
            if prompt_id != full_id:
                break
            prefix_length += 1
        labels[:prefix_length] = -100
        return {
            **sample, "input_ids": input_ids, "labels": labels,
            "attention_mask": torch.tensor(full_tokens["attention_mask"], dtype=torch.long),
        }

    def collate_fn(self, batch):
        return super().collate_fn(batch)

class WMT23Dataset(WMT25Dataset):
    """Translation causal LM data prepared by scripts/prepare_wmt23.py.

    Reuse MASSIVE's chat rendering, indexing and supervised padding only.
    All splits provide supervised labels.
    Generation uses get_generation_sample() without feeding gold targets.
    """

    source_lang = "en"
    # Legacy recipe defaults; __init__ selects the configured profile's partition.
    training_langs = ("de", "he", "ja")
    out_inference_langs = ("zh", "ru", "uk")

    LANGUAGE_NAMES = {
        "de": "German",
        "en": "English",
        "cs": "Czech",
        "zh": "Chinese",
        "ja": "Japanese",
        "he": "Hebrew",
        "ru": "Russian",
        "uk": "Ukrainian",
    }

    def __init__(self, config, tokenizer, split="train", languages=None):
        self.config, self.tokenizer, self.split = config, tokenizer, split
        self.languages = languages
        self.training_langs, self.out_inference_langs = WMT23_PARTITIONS[config.wmt23_corpus_profile]
        self.downstream_sampling = config.wmt23_downstream_sampling
        if self.downstream_sampling not in {"proportional", "balanced_mixed"}:
            raise ValueError("Unknown WMT23 downstream sampling policy.")
        if split not in self.SPLIT_MAPPING:
            raise ValueError(f"Unsupported WMT23 split: {split}")
        if split == "out_validation":
            raise ValueError("WMT23 unseen languages have official test data only; use out_test.")
        self.manifest = self.validate_prepared_manifest(config)
        selected = self._get_languages()
        allowed = self.out_inference_langs if split.startswith("out_") else self.training_langs
        if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(allowed):
            raise ValueError(f"WMT23 {split} languages must be drawn from {allowed}.")
        if split == "train" and set(selected) != set(self.training_langs):
            raise ValueError(f"WMT23 training requires the full {'/'.join(self.training_langs)} pool.")
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id.")
            tokenizer.pad_token = tokenizer.eos_token
        self.all_data, self.dataset_metadata, self.cumulative_sizes = {}, {}, []
        self.load_data(config.wmt23_data_dir)

    @classmethod
    def validate_prepared_manifest(cls, config):
        root = Path(config.wmt23_data_dir).expanduser().resolve()
        path = root / "manifest.json"
        expected_hash = config.wmt23_manifest_sha256
        if not expected_hash or file_sha256(path) != expected_hash:
            raise ValueError("WMT23 manifest SHA256 mismatch; use the hash printed by prepare_wmt23.py.")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        training_langs, out_langs = WMT23_PARTITIONS[config.wmt23_corpus_profile]
        if config.wmt23_corpus_profile == "alma_ja_opus":
            return cls._validate_alma_manifest(config, root, manifest)
        if (manifest["schema_version"] != 1 or manifest["dataset"] != "wmt23"
                or manifest["corpus_profile"] != config.wmt23_corpus_profile
                or manifest["source_lang"] != cls.source_lang
                or tuple(manifest["training_langs"]) != cls.training_langs
                or tuple(manifest["out_inference_langs"]) != cls.out_inference_langs):
            raise ValueError("WMT23 manifest does not match the requested profile/language partition.")
        if (config.training_anchor_langs != cls.source_lang
                or sorted(config.training_lang) != sorted(cls.training_langs)
                or sorted(config.out_inference_lang) != sorted(cls.out_inference_langs)):
            raise ValueError("WMT23 requires en -> de/he/ja training and zh/ru/uk evaluation.")
        policy = manifest["policy"]
        profile = manifest["corpus_profile"]
        if profile not in {"full_parallel", "accessible_parallel"}:
            raise ValueError(f"Unknown WMT23 corpus profile: {profile}")
        exclusions = WMT23_ACCESSIBLE_EXCLUSIONS if profile == "accessible_parallel" else {}
        if profile == "accessible_parallel":
            if (policy["excluded_sources"] != exclusions
                    or policy["cross_corpus_deduplication"] is not True
                    or policy["deduplication_key"] != "exact_stripped_source_target_within_language"):
                raise ValueError("WMT23 accessible_parallel must record the exact exclusions and pair deduplication.")
        if policy["train_sample_cap"] is not None or policy["domain_filter"] is not None:
            raise ValueError("WMT23 requires the full, uncapped parallel pool across domains.")
        required = {"excluded_alignment_sources.json"}
        for lang in cls.training_langs:
            resource = manifest["train_resources"][lang]
            effective = resource["effective_recipe_ids"]
            original = [value.replace("Statmt-news_commentary-16-", "Statmt-news_commentary-18.1-")
                        for value in resource["original_recipe_ids"] if value not in exclusions]
            if not effective or sorted(effective) != sorted(original):
                raise ValueError(f"Incomplete WMT23 recipe for {lang}.")
            if profile == "accessible_parallel":
                expected_exclusions = {value: exclusions[value] for value in resource["original_recipe_ids"] if value in exclusions}
                if resource["excluded_recipe_ids"] != expected_exclusions:
                    raise ValueError(f"Incorrect WMT23 excluded resources for {lang}.")
            if set(resource["usable_counts"]) != set(effective) or any(n <= 0 for n in resource["usable_counts"].values()):
                raise ValueError(f"Missing/empty WMT23 resources for {lang}.")
        for split, languages in (("train", cls.training_langs), ("validation", cls.training_langs),
                                 ("test", cls.training_langs + cls.out_inference_langs)):
            if set(manifest["splits"][split]) != set(languages):
                raise ValueError(f"Incorrect WMT23 {split} languages.")
            for lang in languages:
                names = manifest["splits"][split][lang]
                if not names or len(set(names)) != len(names) or manifest["counts"][split][lang] <= 0:
                    raise ValueError(f"Empty/duplicate WMT23 shards: {split}/{lang}")
                required.update(names)
        for lang in cls.training_langs + cls.out_inference_langs:
            pair = "-".join(sorted(("en", lang)))
            splits = ("train", "validation", "test") if lang in cls.training_langs else ("validation", "test")
            if set(manifest["opus_files"][pair]) != set(splits):
                raise ValueError(f"Incorrect frozen OPUS splits for {pair}.")
            required.update(manifest["opus_files"][pair].values())
        if not required <= set(manifest["files"]):
            raise ValueError("WMT23 manifest is missing file checksums.")
        for name in sorted(required):
            file = (root / name).resolve()
            if not file.is_relative_to(root):
                raise ValueError(f"WMT23 file is outside its prepared directory: {name}")
            recorded = manifest["files"][name]
            if file.stat().st_size != recorded["bytes"] or file_sha256(file) != recorded["sha256"]:
                raise ValueError(f"WMT23 file checksum mismatch: {name}")
        return manifest

    @classmethod
    def _validate_alma_manifest(cls, config, root, manifest):
        training, unseen = WMT23_PARTITIONS["alma_ja_opus"]
        if (manifest["schema_version"] != 2 or manifest["dataset"] != "wmt23"
                or manifest["corpus_profile"] != "alma_ja_opus"
                or manifest["source_lang"] != "en"
                or tuple(manifest["training_langs"]) != training
                or tuple(manifest["out_inference_langs"]) != unseen
                or config.training_anchor_langs != "en"
                or set(config.training_lang) != set(training)
                or set(config.out_inference_lang) != set(unseen)):
            raise ValueError("alma_ja_opus requires de/cs/ja training and zh/ru/uk evaluation.")
        policy = manifest["policy"]
        if (policy["directions"] != "bidirectional" or policy["train_sample_cap"] is not None
                or policy["alignment"] != "OPUS-100 seen-language train only"
                or policy["test"] != "haoranxu/WMT23-Test; cs-en reverses en-cs"):
            raise ValueError("Incorrect ALMA+Japanese/OPUS data policy.")
        if set(manifest["train_resources"]) != set(training):
            raise ValueError("Unexpected SFT training languages.")
        required = {"excluded_alignment_sources.json"}
        for split, langs in (("train", training), ("validation", training), ("test", training + unseen)):
            directions = {direction for lang in langs for direction in (f"en-{lang}", f"{lang}-en")}
            if set(manifest["splits"][split]) != directions or set(manifest["counts"][split]) != directions:
                raise ValueError(f"Incorrect ALMA bidirectional {split} pool.")
            for direction in directions:
                names = manifest["splits"][split][direction]
                if not names or len(set(names)) != len(names) or manifest["counts"][split][direction] <= 0:
                    raise ValueError(f"Empty/duplicate ALMA shards: {split}/{direction}")
                required.update(names)
        expected_pairs = {"-".join(sorted(("en", lang))) for lang in training + unseen}
        if set(manifest["opus_files"]) != expected_pairs:
            raise ValueError("Incorrect frozen OPUS language pairs.")
        for lang in training + unseen:
            pair = "-".join(sorted(("en", lang)))
            expected_splits = {"train", "validation", "test"} if lang in training else {"validation", "test"}
            if set(manifest["opus_files"][pair]) != expected_splits:
                raise ValueError(f"Incorrect frozen OPUS splits for {pair}.")
            required.update(manifest["opus_files"][pair].values())
        if not required <= set(manifest["files"]):
            raise ValueError("ALMA manifest is missing file checksums.")
        for name, recorded in manifest["files"].items():
            path = (root / name).resolve()
            if not path.is_relative_to(root):
                raise ValueError(f"ALMA file is outside its prepared directory: {name}")
            if path.stat().st_size != recorded["bytes"] or file_sha256(path) != recorded["sha256"]:
                raise ValueError(f"WMT23 file checksum mismatch: {name}")
        return manifest

    def load_data(self, data_path):
        root = Path(data_path).expanduser()
        split = self.SPLIT_MAPPING[self.split]
        keys = self._get_languages()
        if self.config.wmt23_corpus_profile == "alma_ja_opus":
            keys = [direction for lang in keys for direction in (f"en-{lang}", f"{lang}-en")]
        for lang in keys:
            names = self.manifest["splits"][split][lang]
            dataset = concatenate_datasets([Dataset.from_file(str(root / name)) for name in names])
            expected = self.manifest["counts"][split][lang]
            if len(dataset) != expected:
                raise ValueError(f"WMT23 {split}/{lang}: expected {expected}, got {len(dataset)}")
            self.all_data[lang] = dataset
            self.cumulative_sizes.append(len(dataset) + (self.cumulative_sizes[-1] if self.cumulative_sizes else 0))
            self.dataset_metadata[lang] = {
                "dataset": "wmt23", "config": lang if "-" in lang else f"en-{lang}", "split": split,
                "num_rows": len(dataset), "fingerprint": dataset._fingerprint,
                "manifest_sha256": self.config.wmt23_manifest_sha256,
                "eval_revision": self.manifest["eval_revision"],
                "data_seed": self.manifest["data_seed"],
                "files": names,
            }

    def get_generation_sample(self, idx):
        lang, local_idx = self._resolve_index(idx)
        item = dict(self.all_data[lang][local_idx])
        split = self.SPLIT_MAPPING[self.split]
        direction = lang if "-" in lang else f"en-{lang}"
        src_lang, tgt_lang = direction.split("-")
        item.update(dataset="wmt23", direction=direction, split=split,
                    src_lang=src_lang, tgt_lang=tgt_lang)
        if not item["source_normalized"].strip() or not item["target_normalized"].strip():
            raise ValueError(f"Empty WMT23 example: {item['example_id']}")
        return {
            "lang": lang,
            "sample_id": make_sample_id("wmt23", direction, split, item["example_id"]),
            "utt": item["source_normalized"], "target": item["target_normalized"], "item": item,
        }

    @property
    def balanced_language_ranges(self):
        # Bidirectional rows for each non-English language are stored next to
        # each other; sample both directions from that language's complete pool.
        ranges = self.language_ranges
        grouped = {}
        for lang in self._get_languages():
            if self.config.wmt23_corpus_profile == "alma_ja_opus":
                forward, reverse = ranges[f"en-{lang}"], ranges[f"{lang}-en"]
                if forward[1] != reverse[0]:
                    raise ValueError(f"Noncontiguous bidirectional pool: {lang}")
                grouped[lang] = (forward[0], reverse[1])
            else:
                grouped[lang] = ranges[lang]
        return grouped

    def _apply_chat_template(self, utterance, target=None, *, lang):
        source, target_lang = lang.split("-") if "-" in lang else ("en", lang)
        return MassiveDataset._apply_chat_template(
            self, utterance, target,
            system_prompt=(f"Translate the following sentences from {self.LANGUAGE_NAMES[source]} "
                           f"to {self.LANGUAGE_NAMES[target_lang]}."),
        )

class CombinedDataset(torch.utils.data.Dataset):
    """
    Routes one or both objectives through a single Trainer dataset.

    Training passes both sub-datasets plus an explicit num_total_data so that
    len(self) == batch x world x accum x num_steps and `max_steps` lands exactly
    at the end of one epoch.

    Integer indices preserve the original mixed-objective sampling. The
    same_pair batch sampler supplies (alignment_index, downstream_index)
    tuples with exactly one non-None entry to load only the selected objective.

    Validation passes exactly one sub-dataset and leaves num_total_data as None,
    which yields the real dataset length and a single objective per batch.
    `collate_fn` dispatches on the keys present in the items, so one collator
    instance serves every eval dataset. This is safe because both sub-collators
    only read `self.tokenizer.pad_token_id` and never any per-dataset state.
    """

    def __init__(
        self,
        alignment_dataset=None,
        downstream_dataset=None,
        num_total_data=None,
    ):
        if alignment_dataset is None and downstream_dataset is None:
            raise ValueError(
                "CombinedDataset requires alignment_dataset, "
                "downstream_dataset, or both."
            )

        self.alignment_dataset = alignment_dataset
        self.downstream_dataset = downstream_dataset

        if num_total_data is None:
            num_total_data = min(
                len(dataset)
                for dataset in (alignment_dataset, downstream_dataset)
                if dataset is not None
            )
        self.num_total_data = num_total_data

    def __len__(self):
        return self.num_total_data

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            alignment_idx, downstream_idx = idx
            if (alignment_idx is None) == (downstream_idx is None):
                raise ValueError("Exactly one objective index is required.")
            if alignment_idx is not None:
                return {
                    'alignment': self.alignment_dataset[alignment_idx]
                }
            return {
                'downstream': self.downstream_dataset[downstream_idx]
            }

        item = {}

        if self.alignment_dataset is not None:
            item['alignment'] = self.alignment_dataset[
                idx % len(self.alignment_dataset)
            ]

        if self.downstream_dataset is not None:
            item['downstream'] = self.downstream_dataset[
                idx % len(self.downstream_dataset)
            ]

        return item

    def collate_fn(self, batch):
        collated = {}

        if 'alignment' in batch[0]:
            collated['alignment'] = self.alignment_dataset.collate_fn(
                [item['alignment'] for item in batch]
            )

        if 'downstream' in batch[0]:
            collated['downstream'] = self.downstream_dataset.collate_fn(
                [item['downstream'] for item in batch]
            )

        if not collated:
            raise ValueError(
                f"Unexpected batch item keys: {sorted(batch[0].keys())}. "
                "Expected 'alignment' and/or 'downstream'."
            )

        return collated
