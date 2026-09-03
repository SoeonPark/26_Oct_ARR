import json
import torch
import bisect
import re
from transformers import AutoTokenizer
from datasets import Dataset, load_dataset
from utils import MASSIVE_LANG_MAP, MASSIVE_SYSTEM_PROMPT

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
        
        for lang_pair in lang_subset:
            try:
                dataset = load_dataset(data_path, lang_pair, split=SPLIT_MAPPING[self.split])
                # Shuffle and sample with seed in config
                dataset = dataset.shuffle(seed=self.config.alignment_sampling_seed).select(range(min(self.config.alignment_num_samples_per_lang, len(dataset))))
                self.all_data[lang_pair] = dataset
                print(f"Length of {lang_pair} dataset: {len(dataset)}")
                print(f"Sample data for {lang_pair}: {dataset[0]}")
                # Sample data for en-ko: {'translation': {'en': "They're shaped like a bus.", 'ko': '할머니처럼 만들었지만.. ? 엉망이지만..'}}
                # Sample data for en-ja: {'translation': {'en': 'Yeah, Vincent Hanna.', 'ja': '- ラウール - ラウールに ヴィンセント・ハンナだ'}}
                # Sample data for en-es: {'translation': {'en': "It was the asbestos in here, that's what did it!", 'es': 'Fueron los asbestos aquí. ¡Eso es lo que ocurrió!'}}
            except Exception as e:
                print(f"Failed loading alignment pair {lang_pair}: {e}")

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

    def __getitem__(self, idx):
        for lang_pair, dataset in self.all_data.items():
            if idx < len(dataset):
                item = dataset[idx]
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
                    # Kept as plain text so per-sample validation logs can show
                    # the exact inputs without decoding token ids back.
                    'source_text': source_text,
                    'target_text': target_text
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
            'source_text': [item['source_text'] for item in batch],
            'target_text': [item['target_text'] for item in batch]
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
                )

            except Exception as e:
                raise RuntimeError(
                    f"Failed loading MASSIVE "
                    f"lang={lang}, locale={locale}, split={hf_split}"
                ) from e

            self.all_data[lang] = dataset

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

        prompt_messages = [
            {
                "role": "system",
                "content": self.system_prompt,
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
            f"System: {self.system_prompt}\n"
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
            "utt": utterance,
            "target": target,
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
            "utt": [x["utt"] for x in batch],
            "target": [x["target"] for x in batch],
        }
        
class CombinedDataset(torch.utils.data.Dataset):
    """
    Routes one or both objectives through a single Trainer dataset.

    Training passes both sub-datasets plus an explicit num_total_data so that
    len(self) == batch x world x accum x num_steps and `max_steps` lands exactly
    at the end of one epoch.

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
