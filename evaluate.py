import argparse
from datetime import datetime
import json
from pathlib import Path
import pickle
from uuid import uuid4

import torch
from peft import PeftModel
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from alignment_logging import (
    AlignmentStatistics,
    alignment_records,
    append_jsonl,
    batch_record,
)
from config import ALIGNMENT_REFERENCES, PAIR_BATCH_LOSSES, resolve_alignment_loss
from data_utils import AlignmentDataset, MassiveDataset, make_sample_id
from models import CustomModel
from samplers import AlignmentEvalBatchSampler


def parse_eval_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a saved 2026Oct_ARR PEFT checkpoint."
    )

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Checkpoint or final run directory containing the PEFT adapter.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["validation", "test"],
        help="Hugging Face split to evaluate.",
    )
    parser.add_argument(
        "--language_scope",
        type=str,
        default="in",
        choices=["in", "out", "both"],
        help="Evaluate task-trained languages, unseen languages, or both.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=["alignment", "massive"],
        choices=["alignment", "massive"],
        help="Evaluation tasks to run.",
    )
    parser.add_argument(
        "--alignment_batch_size",
        type=int,
        default=32,
        help="Batch size used to extract alignment representations.",
    )
    parser.add_argument(
        "--massive_batch_size",
        type=int,
        default=32,
        help="Batch size used for MASSIVE generation.",
    )
    parser.add_argument(
        "--retrieval_chunk_size",
        type=int,
        default=256,
        help="Number of retrieval queries processed at once.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=64,
        help="Maximum number of tokens generated for a MASSIVE answer.",
    )
    parser.add_argument(
        "--save_alignment_embeddings",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save extracted representations in addition to retrieval metrics.",
    )
    parser.add_argument(
        "--save_alignment_sample_metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stream scalar diagnostics for every alignment sample to JSONL.",
    )
    parser.add_argument(
        "--eval_sample_log_limit",
        type=int,
        default=64,
        help=(
            "Save inputs and embeddings for the first N samples per task and "
            "language (or language pair). Set 0 to disable sample recording."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Evaluation output root. By default, results are saved under "
            "CHECKPOINT/evaluations."
        ),
    )

    return parser.parse_args()


def validate_eval_args(args):
    if args.eval_sample_log_limit < 0:
        raise ValueError("eval_sample_log_limit must be nonnegative.")

    positive_integer_arguments = {
        "alignment_batch_size": args.alignment_batch_size,
        "massive_batch_size": args.massive_batch_size,
        "retrieval_chunk_size": args.retrieval_chunk_size,
        "max_new_tokens": args.max_new_tokens,
    }

    for argument_name in positive_integer_arguments:
        argument_value = positive_integer_arguments[argument_name]

        if argument_value <= 0:
            raise ValueError(
                f"{argument_name} must be positive, got {argument_value}."
            )


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as output_file:
        json.dump(
            payload,
            output_file,
            ensure_ascii=False,
            indent=2,
            default=str,
        )


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as output_file:
        for record in records:
            json.dump(
                record,
                output_file,
                ensure_ascii=False,
                default=str,
            )
            output_file.write("\n")


class EvalSampleRecorder:
    """Bounded sample JSON and key-to-vector pickle, as in Trainer logs."""

    def __init__(self, limit):
        self.limit = limit
        self.records = {}
        self.embeddings = {}

    def select_indices(self, objective, languages):
        # Count pending samples too: one batch can exceed a group's allowance.
        counts = {}
        indices = []
        for index, language in enumerate(languages):
            group = f"{objective}/{language}"
            count = counts.get(group, len(self.records.get(group, [])))
            if count < self.limit:
                indices.append(index)
                counts[group] = count + 1
        return indices

    def add(self, objective, language, record, embeddings):
        group = f"{objective}/{language}"
        records = self.records.setdefault(group, [])
        if len(records) >= self.limit:
            return None

        sample_index = len(records)
        embedding_keys = {}
        for name, embedding in embeddings.items():
            key = f"{group}/{name}_{sample_index}"
            embedding_keys[f"{name.removesuffix('s')}_key"] = key
            # Own each small array; do not retain a view of a whole batch.
            self.embeddings[key] = (
                embedding.detach().float().cpu().numpy().copy()
            )
        saved_record = {
            **record,
            "sample_id": record.get("sample_id", f"{group}/{sample_index}"),
            "embedding_keys": embedding_keys,
        }
        records.append(saved_record)
        return saved_record

    def save(self, output_dir):
        if not self.records:
            return
        samples_path = output_dir / "eval_samples.json"
        embeddings_path = output_dir / "eval_samples_embeddings.pkl"
        write_json(samples_path, self.records)
        with embeddings_path.open("wb") as output_file:
            pickle.dump(self.embeddings, output_file)
        self.records.clear()
        self.embeddings.clear()
        print(f"Saved evaluation samples to {samples_path}")
        print(f"Saved sample embeddings to {embeddings_path}")


def load_experiment_config(checkpoint_path):
    config_path = checkpoint_path / "experiment_config.json"

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Experiment config not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as config_file:
        experiment_config = json.load(config_file)

    # config.py also returns argparse.Namespace. Keeping the same object type
    # lets the rest of this project use config.attribute consistently.
    config = argparse.Namespace(**experiment_config)
    config.alignment_loss = resolve_alignment_loss(config)
    return config


def validate_alignment_layer(model, experiment_config):
    number_of_layers = getattr(model.config, "num_hidden_layers", None)

    if number_of_layers is None:
        return

    selected_layer = experiment_config.alignment_hidden_state_layer
    number_of_hidden_state_entries = number_of_layers + 1

    if not (
        -number_of_hidden_state_entries
        <= selected_layer
        < number_of_hidden_state_entries
    ):
        raise ValueError(
            "alignment_hidden_state_layer is outside the available hidden "
            f"state range. Selected {selected_layer}, but this model has "
            f"{number_of_hidden_state_entries} entries with valid indices "
            f"from {-number_of_hidden_state_entries} to "
            f"{number_of_hidden_state_entries - 1}."
        )


def build_model(checkpoint_path, experiment_config):
    tokenizer_config_path = checkpoint_path / "tokenizer_config.json"

    if tokenizer_config_path.is_file():
        tokenizer_source = str(checkpoint_path)
    else:
        tokenizer_source = experiment_config.model_name

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # CustomModel.get_alignment_embeddings assumes right padding when it
    # computes the last non-padding position from attention_mask.sum().
    tokenizer.padding_side = "right"

    quantization_config = None

    if experiment_config.quantization_load_in_4bit:
        compute_dtype = getattr(
            torch,
            experiment_config.quantization_compute_dtype,
        )
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=(
                experiment_config.quantization_use_double_quant
            ),
            bnb_4bit_quant_type=experiment_config.quantization_type,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    base_model = AutoModelForCausalLM.from_pretrained(
        experiment_config.model_name,
        quantization_config=quantization_config,
        device_map="auto",
    )
    peft_model = PeftModel.from_pretrained(
        base_model,
        str(checkpoint_path),
        is_trainable=False,
    )
    model = CustomModel(
        config=experiment_config,
        basemodel=peft_model,
    )
    validate_alignment_layer(model, experiment_config)
    model.eval()

    return model, tokenizer


def get_language_scopes(language_scope):
    if language_scope == "in":
        return ["in"]

    if language_scope == "out":
        return ["out"]

    if language_scope == "both":
        return ["in", "out"]

    raise ValueError(f"Unknown language scope: {language_scope}")


def get_model_input_device(model):
    embedding_layer = model.basemodel.get_input_embeddings()
    return embedding_layer.weight.device


def move_tensors_to_device(batch, device):
    model_batch = {}

    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            model_batch[key] = value.to(device)
        else:
            model_batch[key] = value

    return model_batch


def validate_loaded_alignment_pairs(dataset, experiment_config, scope):
    if scope == "in":
        evaluated_languages = experiment_config.training_lang
    elif scope == "out":
        evaluated_languages = experiment_config.out_inference_lang
    else:
        raise ValueError(f"Unknown language scope: {scope}")

    anchor_language = experiment_config.training_anchor_langs
    loaded_pairs = list(dataset.all_data.keys())
    missing_languages = []
    duplicated_languages = []

    for language in evaluated_languages:
        forward_pair = f"{anchor_language}-{language}"
        reverse_pair = f"{language}-{anchor_language}"

        number_of_loaded_directions = int(forward_pair in loaded_pairs)
        number_of_loaded_directions += int(reverse_pair in loaded_pairs)

        if number_of_loaded_directions == 0:
            missing_languages.append(language)
        elif number_of_loaded_directions == 2:
            duplicated_languages.append(language)

    if missing_languages:
        raise RuntimeError(
            "No alignment dataset was loaded for these languages: "
            f"{missing_languages}. Loaded pairs: {loaded_pairs}"
        )

    if duplicated_languages:
        raise RuntimeError(
            "Both direction-specific dataset configs were loaded for these "
            f"languages: {duplicated_languages}. Load each parallel corpus "
            "once and evaluate both retrieval directions from that corpus."
        )

    if len(dataset) == 0:
        raise RuntimeError(
            f"The {scope} alignment evaluation dataset is empty."
        )


@torch.inference_mode()
def collect_alignment_embeddings(
    model,
    dataloader,
    sample_recorder=None,
    diagnostics=None,
    sample_metrics_path=None,
    context=None,
):
    groups = {}
    input_device = get_model_input_device(model)
    number_of_batches = len(dataloader)
    context = dict(context or {})
    context.setdefault("session_id", uuid4().hex)
    context.setdefault("phase", "evaluation")
    if sample_metrics_path is not None:
        sample_metrics_path = Path(sample_metrics_path)
        sample_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        sample_metrics_path.write_text("", encoding="utf-8")
        batch_metrics_path = sample_metrics_path.with_name("alignment_batches.jsonl")
        batch_metrics_path.write_text("", encoding="utf-8")

    for batch_index, batch in enumerate(dataloader, start=1):
        sample_indices = (
            sample_recorder.select_indices("alignment", batch["lang_pair"])
            if sample_recorder is not None else []
        )
        model_batch = move_tensors_to_device(batch, input_device)
        outputs = model(
            forward_type="alignment",
            alignment=model_batch,
            return_per_sample=True,
        )
        batch_context = {
            **context,
            "batch_index": batch_index,
            "batch_id": f"{context['session_id']}/{context.get('scope', 'unspecified')}/{batch_index}",
        }
        if diagnostics is not None:
            diagnostics.update(batch, outputs)
        if sample_metrics_path is not None:
            append_jsonl(
                sample_metrics_path,
                alignment_records(
                    batch, outputs, range(len(batch["lang_pair"])), batch_context,
                ),
            )
            append_jsonl(batch_metrics_path, [batch_record(batch, outputs, batch_context)])

        source_embeddings = outputs["source_embeddings"].float().cpu()
        target_embeddings = outputs["target_embeddings"].float().cpu()

        for item_index, language_pair in enumerate(batch["lang_pair"]):
            if language_pair not in groups:
                groups[language_pair] = {
                    "source": [],
                    "target": [],
                }

            groups[language_pair]["source"].append(
                source_embeddings[item_index]
            )
            groups[language_pair]["target"].append(
                target_embeddings[item_index]
            )

        records = alignment_records(batch, outputs, sample_indices, batch_context)
        for item_index, record in zip(sample_indices, records):
            sample_recorder.add(
                "alignment",
                batch["lang_pair"][item_index],
                {
                    **record,
                    "origin_data": batch["item"][item_index],
                    "source_text": batch["source_text"][item_index],
                    "target_text": batch["target_text"][item_index],
                    "loss": record["per_sample_loss"],
                    "batch_sample_ids": list(batch["sample_id"]),
                    "batch_language_pairs": list(batch["lang_pair"]),
                },
                {
                    name: outputs[name][item_index]
                    for name in (
                        "source_embeddings", "target_embeddings",
                        "source_last_layer_embeddings", "target_last_layer_embeddings",
                    )
                },
            )

        if batch_index == 1 or batch_index % 50 == 0:
            print(
                f"[Alignment] extracted batch "
                f"{batch_index}/{number_of_batches}"
            )

    for language_pair in groups:
        groups[language_pair]["source"] = torch.stack(
            groups[language_pair]["source"]
        )
        groups[language_pair]["target"] = torch.stack(
            groups[language_pair]["target"]
        )

    return groups


def evaluate_retrieval_direction(
    query_embeddings,
    candidate_embeddings,
    chunk_size,
    device,
):
    if len(query_embeddings) != len(candidate_embeddings):
        raise ValueError(
            "Query and candidate counts must match for index-based retrieval. "
            f"Got {len(query_embeddings)} queries and "
            f"{len(candidate_embeddings)} candidates."
        )

    number_of_queries = len(query_embeddings)

    if number_of_queries == 0:
        raise ValueError("Cannot evaluate retrieval with zero examples.")

    candidate_embeddings = F.normalize(
        candidate_embeddings.float().to(device),
        p=2,
        dim=-1,
    )

    recall_at_1_count = 0
    recall_at_5_count = 0
    reciprocal_rank_sum = 0.0

    candidate_indices = torch.arange(
        len(candidate_embeddings),
        device=device,
    )

    for start_index in range(0, number_of_queries, chunk_size):
        end_index = min(
            start_index + chunk_size,
            number_of_queries,
        )
        query_chunk = query_embeddings[start_index:end_index].to(device)
        query_chunk = F.normalize(
            query_chunk.float(),
            p=2,
            dim=-1,
        )
        similarity = query_chunk @ candidate_embeddings.T

        correct_candidate_indices = torch.arange(
            start_index,
            end_index,
            device=device,
        )
        local_query_indices = torch.arange(
            end_index - start_index,
            device=device,
        )
        correct_scores = similarity[
            local_query_indices,
            correct_candidate_indices,
        ]

        # Sort by descending score, then ascending candidate index.
        gold_scores = correct_scores.unsqueeze(1)

        # Candidates with a higher score always precede the gold candidate.
        higher_count = (
            similarity > gold_scores
        ).sum(dim=1)

        # Among exact ties, candidates with smaller indices come first.
        tied_before_count = (
            (similarity == gold_scores)
            & (
                candidate_indices.unsqueeze(0)
                < correct_candidate_indices.unsqueeze(1)
            )
        ).sum(dim=1)

        ranks = 1 + higher_count + tied_before_count

        recall_at_1_count += int((ranks <= 1).sum().item())
        recall_at_5_count += int((ranks <= 5).sum().item())
        reciprocal_rank_sum += float(
            (1.0 / ranks.float()).sum().item()
        )

    return {
        "num_queries": number_of_queries,
        "recall_at_1": recall_at_1_count / number_of_queries,
        "recall_at_5": recall_at_5_count / number_of_queries,
        "mrr": reciprocal_rank_sum / number_of_queries,
    }


def average_retrieval_metrics(first_metrics, second_metrics):
    return {
        "recall_at_1": (
            first_metrics["recall_at_1"]
            + second_metrics["recall_at_1"]
        ) / 2,
        "recall_at_5": (
            first_metrics["recall_at_5"]
            + second_metrics["recall_at_5"]
        ) / 2,
        "mrr": (
            first_metrics["mrr"]
            + second_metrics["mrr"]
        ) / 2,
    }


def evaluate_alignment_retrieval(embedding_groups, chunk_size, device):
    pair_results = {}

    macro_recall_at_1 = 0.0
    macro_recall_at_5 = 0.0
    macro_mrr = 0.0

    weighted_recall_at_1 = 0.0
    weighted_recall_at_5 = 0.0
    weighted_mrr = 0.0
    total_direction_queries = 0

    for language_pair in embedding_groups:
        source_language, target_language = language_pair.split("-")
        source_embeddings = embedding_groups[language_pair]["source"]
        target_embeddings = embedding_groups[language_pair]["target"]

        source_to_target = evaluate_retrieval_direction(
            query_embeddings=source_embeddings,
            candidate_embeddings=target_embeddings,
            chunk_size=chunk_size,
            device=device,
        )
        target_to_source = evaluate_retrieval_direction(
            query_embeddings=target_embeddings,
            candidate_embeddings=source_embeddings,
            chunk_size=chunk_size,
            device=device,
        )
        bidirectional_average = average_retrieval_metrics(
            source_to_target,
            target_to_source,
        )

        pair_results[language_pair] = {
            "num_parallel_pairs": len(source_embeddings),
            "source_to_target": {
                "query_language": source_language,
                "candidate_language": target_language,
                **source_to_target,
            },
            "target_to_source": {
                "query_language": target_language,
                "candidate_language": source_language,
                **target_to_source,
            },
            "bidirectional_average": bidirectional_average,
        }

        macro_recall_at_1 += bidirectional_average["recall_at_1"]
        macro_recall_at_5 += bidirectional_average["recall_at_5"]
        macro_mrr += bidirectional_average["mrr"]

        for direction_metrics in (source_to_target, target_to_source):
            number_of_queries = direction_metrics["num_queries"]
            total_direction_queries += number_of_queries
            weighted_recall_at_1 += (
                direction_metrics["recall_at_1"] * number_of_queries
            )
            weighted_recall_at_5 += (
                direction_metrics["recall_at_5"] * number_of_queries
            )
            weighted_mrr += direction_metrics["mrr"] * number_of_queries

    number_of_pairs = len(pair_results)

    if number_of_pairs == 0:
        raise ValueError("No alignment embedding groups were collected.")

    return {
        "candidate_pool": "full language-pair evaluation split",
        "similarity": "cosine",
        "tie_break": "candidate_index_ascending",
        "pairs": pair_results,
        "language_pair_macro": {
            "recall_at_1": macro_recall_at_1 / number_of_pairs,
            "recall_at_5": macro_recall_at_5 / number_of_pairs,
            "mrr": macro_mrr / number_of_pairs,
        },
        "query_micro": {
            "num_direction_queries": total_direction_queries,
            "recall_at_1": (
                weighted_recall_at_1 / total_direction_queries
            ),
            "recall_at_5": (
                weighted_recall_at_5 / total_direction_queries
            ),
            "mrr": weighted_mrr / total_direction_queries,
        },
    }


def build_massive_evaluation_samples(dataset):
    samples = []

    for language in dataset.all_data:
        language_dataset = dataset.all_data[language]

        for item in language_dataset:
            samples.append(
                {
                    "sample_id": make_sample_id(
                        dataset.config.downstream_task_data,
                        dataset.lang_map[language],
                        dataset.SPLIT_MAPPING[dataset.split],
                        item["id"],
                    ),
                    "lang": language,
                    "utt": item["utt"],
                    "target": dataset.extract_slots(item["annot_utt"]),
                    "item": item,
                }
            )

    return samples


def collate_massive_evaluation_samples(batch):
    return {
        "sample_id": [item["sample_id"] for item in batch],
        "lang": [item["lang"] for item in batch],
        "utt": [item["utt"] for item in batch],
        "target": [item["target"] for item in batch],
        "item": [item["item"] for item in batch],
    }


def normalize_slot_component(text):
    return " ".join(text.strip().casefold().split())


def parse_slot_items(text):
    normalized_text = normalize_slot_component(text)

    if not normalized_text or normalized_text == "none":
        return []

    slot_items = []

    for raw_item in text.split(";"):
        raw_item = raw_item.strip()

        if not raw_item:
            continue

        if ":" not in raw_item:
            # Keep malformed generations as false-positive items rather than
            # silently discarding them and inflating precision.
            malformed_item = normalize_slot_component(raw_item)
            slot_items.append(f"__invalid__: {malformed_item}")
            continue

        slot_name, slot_value = raw_item.split(":", maxsplit=1)
        slot_name = normalize_slot_component(slot_name)
        slot_value = normalize_slot_component(slot_value)

        if not slot_name or not slot_value:
            malformed_item = normalize_slot_component(raw_item)
            slot_items.append(f"__invalid__: {malformed_item}")
            continue

        slot_items.append(f"{slot_name}: {slot_value}")

    return slot_items


def count_slot_matches(predicted_slots, target_slots):
    unmatched_target_slots = list(target_slots)
    true_positives = 0
    false_positives = 0

    for predicted_slot in predicted_slots:
        if predicted_slot in unmatched_target_slots:
            true_positives += 1
            unmatched_target_slots.remove(predicted_slot)
        else:
            false_positives += 1

    false_negatives = len(unmatched_target_slots)

    return true_positives, false_positives, false_negatives


def calculate_precision_recall_f1(
    true_positives,
    false_positives,
    false_negatives,
):
    precision_denominator = true_positives + false_positives
    recall_denominator = true_positives + false_negatives

    if precision_denominator == 0:
        precision = 0.0
    else:
        precision = true_positives / precision_denominator

    if recall_denominator == 0:
        recall = 0.0
    else:
        recall = true_positives / recall_denominator

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


@torch.inference_mode()
def collect_text_embeddings(model, tokens):
    """Pool right-padded inputs without retaining generation hidden states."""
    tokens = move_tensors_to_device(tokens, get_model_input_device(model))
    outputs = model.basemodel(
        **tokens,
        output_hidden_states=True,
        return_dict=True,
        use_cache=False,
    )
    layer = model.experiment_config.alignment_hidden_state_layer
    return {
        name: model.get_alignment_embeddings(
            outputs.hidden_states[position], tokens["attention_mask"],
        ).float().cpu().clone()
        for name, position in (("selected", layer), ("last", -1))
    }


def collect_massive_sample_embeddings(
    model, tokenizer, utterances, prompt_tokens, sample_indices,
):
    # Use the same tokenization/pooling as alignment for utterance-only plots.
    original_padding_side = tokenizer.padding_side
    try:
        tokenizer.padding_side = "right"
        utterance_tokens = tokenizer(
            [utterances[index] for index in sample_indices],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=getattr(model.experiment_config, "alignment_max_length", None),
        )
    finally:
        tokenizer.padding_side = original_padding_side
    utterance_embeddings = collect_text_embeddings(model, utterance_tokens)

    # Generation is left padded. Repack its exact non-pad tokens on the right
    # for the forward pass: pooling and model positions then match training.
    prompt_rows = [
        prompt_tokens["input_ids"][index][
            prompt_tokens["attention_mask"][index].bool()
        ]
        for index in sample_indices
    ]
    right_padded_prompts = {
        "input_ids": torch.nn.utils.rnn.pad_sequence(
            prompt_rows, batch_first=True, padding_value=tokenizer.pad_token_id,
        ),
        "attention_mask": torch.nn.utils.rnn.pad_sequence(
            [torch.ones_like(row) for row in prompt_rows],
            batch_first=True, padding_value=0,
        ),
    }
    prompt_embeddings = collect_text_embeddings(model, right_padded_prompts)
    return {
        "utt_embeddings": utterance_embeddings["selected"],
        "utt_last_layer_embeddings": utterance_embeddings["last"],
        "prompt_embeddings": prompt_embeddings["selected"],
        "prompt_last_layer_embeddings": prompt_embeddings["last"],
    }


@torch.inference_mode()
def generate_massive_predictions(
    model,
    tokenizer,
    dataset,
    dataloader,
    max_new_tokens,
    sample_recorder=None,
    context=None,
):
    predictions = []
    input_device = get_model_input_device(model)
    number_of_batches = len(dataloader)
    original_padding_side = tokenizer.padding_side
    context = dict(context or {})
    context.setdefault("session_id", uuid4().hex)
    context.setdefault("phase", "evaluation")

    try:
        # Decoder-only batched generation must be left padded so generation
        # starts after the last prompt token for every sample in the batch.
        tokenizer.padding_side = "left"

        for batch_index, batch in enumerate(dataloader, start=1):
            batch_id = (
                f"{context['session_id']}/{context.get('scope', 'unspecified')}"
                f"/downstream/{batch_index}"
            )
            batch_context = {
                **context,
                "batch_id": batch_id,
                "batch_index": batch_index,
                "actual_batch_size": len(batch["lang"]),
            }
            prompt_texts = []

            for utterance in batch["utt"]:
                prompt_text, _ = dataset._apply_chat_template(
                    utterance=utterance,
                    target=None,
                )
                prompt_texts.append(prompt_text)

            prompt_tokens = tokenizer(
                prompt_texts,
                add_special_tokens=False,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            prompt_tokens = move_tensors_to_device(
                prompt_tokens,
                input_device,
            )

            generation_arguments = {
                "input_ids": prompt_tokens["input_ids"],
                "attention_mask": prompt_tokens["attention_mask"],
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "pad_token_id": tokenizer.pad_token_id,
            }

            if tokenizer.eos_token_id is not None:
                generation_arguments["eos_token_id"] = (
                    tokenizer.eos_token_id
                )

            generated_token_ids = model.basemodel.generate(
                **generation_arguments
            )
            prompt_width = prompt_tokens["input_ids"].shape[1]
            answer_token_ids = generated_token_ids[:, prompt_width:]
            generated_answers = tokenizer.batch_decode(
                answer_token_ids,
                skip_special_tokens=True,
            )

            sample_indices = (
                sample_recorder.select_indices("downstream", batch["lang"])
                if sample_recorder is not None else []
            )
            sample_embeddings = (
                collect_massive_sample_embeddings(
                    model, tokenizer, batch["utt"], prompt_tokens, sample_indices,
                )
                if sample_indices else {}
            )
            sample_positions = {
                index: position for position, index in enumerate(sample_indices)
            }

            for item_index, generated_answer in enumerate(
                generated_answers
            ):
                target = batch["target"][item_index]
                predicted_slots = parse_slot_items(generated_answer)
                target_slots = parse_slot_items(target)

                predictions.append(
                    {
                        **batch_context,
                        "record_id": f"{batch_id}/sample-{item_index}",
                        "batch_position": item_index,
                        "sample_id": batch["sample_id"][item_index],
                        "lang": batch["lang"][item_index],
                        "utt": batch["utt"][item_index],
                        "target": target,
                        "prediction": generated_answer.strip(),
                        "target_slots": target_slots,
                        "predicted_slots": predicted_slots,
                        "exact_match": sorted(predicted_slots) == sorted(target_slots),
                    }
                )

                if item_index in sample_positions:
                    position = sample_positions[item_index]
                    saved_record = sample_recorder.add(
                        "downstream",
                        batch["lang"][item_index],
                        {
                            **predictions[-1],
                            "origin_data": batch["item"][item_index],
                            "prompt_text": prompt_texts[item_index],
                            "embedding_inputs": {
                                "utt": "utterance_only",
                                "prompt": "generation_prompt_without_answer",
                            },
                        },
                        {
                            name: embeddings[position]
                            for name, embeddings in sample_embeddings.items()
                        },
                    )
                    predictions[-1]["sample_id"] = saved_record["sample_id"]
                    predictions[-1]["embedding_keys"] = saved_record["embedding_keys"]

            if batch_index == 1 or batch_index % 50 == 0:
                print(
                    f"[MASSIVE] generated batch "
                    f"{batch_index}/{number_of_batches}"
                )
    finally:
        tokenizer.padding_side = original_padding_side

    return predictions


def evaluate_massive_predictions(predictions):
    if not predictions:
        raise ValueError("No MASSIVE predictions were generated.")

    overall_true_positives = 0
    overall_false_positives = 0
    overall_false_negatives = 0
    overall_exact_matches = 0
    language_statistics = {}

    for prediction in predictions:
        language = prediction["lang"]
        predicted_slots = prediction["predicted_slots"]
        target_slots = prediction["target_slots"]

        true_positives, false_positives, false_negatives = (
            count_slot_matches(predicted_slots, target_slots)
        )
        exact_match = sorted(predicted_slots) == sorted(target_slots)

        prediction["exact_match"] = exact_match

        overall_true_positives += true_positives
        overall_false_positives += false_positives
        overall_false_negatives += false_negatives
        overall_exact_matches += int(exact_match)

        if language not in language_statistics:
            language_statistics[language] = {
                "num_examples": 0,
                "true_positives": 0,
                "false_positives": 0,
                "false_negatives": 0,
                "exact_matches": 0,
            }

        language_statistics[language]["num_examples"] += 1
        language_statistics[language]["true_positives"] += true_positives
        language_statistics[language]["false_positives"] += false_positives
        language_statistics[language]["false_negatives"] += false_negatives
        language_statistics[language]["exact_matches"] += int(exact_match)

    per_language = {}
    language_macro_f1 = 0.0
    language_macro_exact_match = 0.0

    for language in language_statistics:
        statistics = language_statistics[language]
        precision_recall_f1 = calculate_precision_recall_f1(
            statistics["true_positives"],
            statistics["false_positives"],
            statistics["false_negatives"],
        )
        exact_match = (
            statistics["exact_matches"]
            / statistics["num_examples"]
        )

        per_language[language] = {
            "num_examples": statistics["num_examples"],
            "slot_precision": precision_recall_f1["precision"],
            "slot_recall": precision_recall_f1["recall"],
            "slot_f1": precision_recall_f1["f1"],
            "exact_match": exact_match,
        }
        language_macro_f1 += precision_recall_f1["f1"]
        language_macro_exact_match += exact_match

    overall_precision_recall_f1 = calculate_precision_recall_f1(
        overall_true_positives,
        overall_false_positives,
        overall_false_negatives,
    )
    number_of_languages = len(per_language)

    return {
        "slot_matching": (
            "case-folded exact match of slot-name/value pairs; "
            "slot order is ignored and duplicate slots are counted"
        ),
        "num_examples": len(predictions),
        "overall_slot_micro": {
            "precision": overall_precision_recall_f1["precision"],
            "recall": overall_precision_recall_f1["recall"],
            "f1": overall_precision_recall_f1["f1"],
        },
        "overall_exact_match": (
            overall_exact_matches / len(predictions)
        ),
        "language_macro_slot_f1": (
            language_macro_f1 / number_of_languages
        ),
        "language_macro_exact_match": (
            language_macro_exact_match / number_of_languages
        ),
        "per_language": per_language,
    }


def get_evaluation_root(args, checkpoint_path):
    if args.output_dir is not None:
        return Path(args.output_dir).expanduser().resolve()

    return checkpoint_path / "evaluations"


def evaluate_alignment_scope(
    args,
    experiment_config,
    model,
    tokenizer,
    scope,
    scope_output_dir,
    sample_recorder=None,
):
    split_name = f"{scope}_{args.split}"
    dataset = AlignmentDataset(
        experiment_config,
        tokenizer=tokenizer,
        split=split_name,
    )
    validate_loaded_alignment_pairs(
        dataset,
        experiment_config,
        scope,
    )
    if resolve_alignment_loss(experiment_config) in PAIR_BATCH_LOSSES:
        dataloader = DataLoader(
            dataset,
            batch_sampler=AlignmentEvalBatchSampler(
                dataset.pair_ranges, args.alignment_batch_size,
            ),
            collate_fn=dataset.collate_fn,
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=args.alignment_batch_size,
            shuffle=False,
            collate_fn=dataset.collate_fn,
        )
    diagnostics = AlignmentStatistics()
    context = {
        "run_id": getattr(experiment_config, "wandb_run_name", None),
        "session_id": args.evaluation_session_id,
        "phase": "evaluation",
        "split": args.split,
        "scope": scope,
        "global_step": getattr(experiment_config, "checkpoint_global_step", None),
        "alignment_hidden_state_layer": experiment_config.alignment_hidden_state_layer,
        "alignment_hidden_state_position": experiment_config.alignment_hidden_state_position,
    }
    embedding_groups = collect_alignment_embeddings(
        model,
        dataloader,
        sample_recorder,
        diagnostics=diagnostics,
        sample_metrics_path=(
            scope_output_dir / "alignment_samples.jsonl"
            if args.save_alignment_sample_metrics else None
        ),
        context=context,
    )
    metrics = evaluate_alignment_retrieval(
        embedding_groups,
        chunk_size=args.retrieval_chunk_size,
        device=get_model_input_device(model),
    )
    metrics["alignment_loss_type"] = resolve_alignment_loss(experiment_config)
    metrics["schema_version"] = 1
    metrics["gap_reference"] = ALIGNMENT_REFERENCES[metrics["alignment_loss_type"]]
    metrics["distance_diagnostics"] = diagnostics.summary(corpus=True)
    metrics["dataset_metadata"] = dataset.dataset_metadata

    metrics_path = scope_output_dir / "alignment_metrics.json"
    write_json(metrics_path, metrics)
    print(f"Saved alignment metrics to {metrics_path}")

    if args.save_alignment_embeddings:
        embeddings_path = scope_output_dir / "alignment_embeddings.pt"
        torch.save(embedding_groups, embeddings_path)
        print(f"Saved alignment embeddings to {embeddings_path}")

    return metrics


def evaluate_massive_scope(
    args,
    experiment_config,
    model,
    tokenizer,
    scope,
    scope_output_dir,
    sample_recorder=None,
):
    split_name = f"{scope}_{args.split}"
    dataset = MassiveDataset(
        experiment_config,
        tokenizer=tokenizer,
        split=split_name,
    )
    samples = build_massive_evaluation_samples(dataset)
    dataloader = DataLoader(
        samples,
        batch_size=args.massive_batch_size,
        shuffle=False,
        collate_fn=collate_massive_evaluation_samples,
    )
    predictions = generate_massive_predictions(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        dataloader=dataloader,
        max_new_tokens=args.max_new_tokens,
        sample_recorder=sample_recorder,
        context={
            "run_id": getattr(experiment_config, "wandb_run_name", None),
            "session_id": args.evaluation_session_id,
            "phase": "evaluation",
            "split": args.split,
            "scope": scope,
            "global_step": getattr(experiment_config, "checkpoint_global_step", None),
            "alignment_hidden_state_layer": experiment_config.alignment_hidden_state_layer,
            "alignment_hidden_state_position": experiment_config.alignment_hidden_state_position,
        },
    )
    metrics = evaluate_massive_predictions(predictions)
    metrics["schema_version"] = 1
    metrics["dataset_metadata"] = dataset.dataset_metadata

    metrics_path = scope_output_dir / "massive_metrics.json"
    predictions_path = scope_output_dir / "massive_predictions.jsonl"
    write_json(metrics_path, metrics)
    write_jsonl(predictions_path, predictions)
    print(f"Saved MASSIVE metrics to {metrics_path}")
    print(f"Saved MASSIVE predictions to {predictions_path}")

    return metrics


def main():
    args = parse_eval_args()
    validate_eval_args(args)
    args.evaluation_session_id = uuid4().hex
    checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()

    if not checkpoint_path.is_dir():
        raise NotADirectoryError(
            f"Checkpoint directory not found: {checkpoint_path}"
        )

    experiment_config = load_experiment_config(checkpoint_path)
    if (
        experiment_config.alignment_loss in PAIR_BATCH_LOSSES
        and "alignment" in args.tasks
        and args.alignment_batch_size < 2
    ):
        raise ValueError("Gap evaluation requires alignment_batch_size >= 2.")
    model, tokenizer = build_model(
        checkpoint_path,
        experiment_config,
    )

    scopes = get_language_scopes(args.language_scope)
    evaluation_root = get_evaluation_root(args, checkpoint_path)
    split_output_dir = evaluation_root / args.split
    split_output_dir.mkdir(parents=True, exist_ok=True)

    evaluation_metadata = {
        "schema_version": 1,
        "session_id": args.evaluation_session_id,
        "gap_reference": ALIGNMENT_REFERENCES[experiment_config.alignment_loss],
        "gap_embedding_space": "unnormalized",
        "status": "running",
        "started_at": datetime.now().isoformat(),
        "checkpoint_path": str(checkpoint_path),
        "split": args.split,
        "language_scopes": scopes,
        "tasks": args.tasks,
        "alignment_batch_size": args.alignment_batch_size,
        "massive_batch_size": args.massive_batch_size,
        "retrieval_chunk_size": args.retrieval_chunk_size,
        "max_new_tokens": args.max_new_tokens,
        "save_alignment_embeddings": args.save_alignment_embeddings,
        "save_alignment_sample_metrics": args.save_alignment_sample_metrics,
        "eval_sample_log_limit": args.eval_sample_log_limit,
        "sample_embedding_inputs": {
            "alignment": ["source_text", "target_text"],
            "downstream": ["utterance_only", "generation_prompt_without_answer"],
        },
        "experiment_config": vars(experiment_config).copy(),
        "results": {},
    }
    metadata_path = split_output_dir / "evaluation_metadata.json"
    write_json(metadata_path, evaluation_metadata)

    try:
        for scope in scopes:
            print("\n" + "*" * 60)
            print(f"Checkpoint: {checkpoint_path}")
            print(f"Split: {args.split}")
            print(f"Language scope: {scope}")
            print(f"Tasks: {', '.join(args.tasks)}")
            print("*" * 60 + "\n")

            scope_output_dir = split_output_dir / scope
            scope_output_dir.mkdir(parents=True, exist_ok=True)
            scope_results = {}
            sample_recorder = EvalSampleRecorder(args.eval_sample_log_limit)

            if "alignment" in args.tasks:
                scope_results["alignment"] = evaluate_alignment_scope(
                    args=args,
                    experiment_config=experiment_config,
                    model=model,
                    tokenizer=tokenizer,
                    scope=scope,
                    scope_output_dir=scope_output_dir,
                    sample_recorder=sample_recorder,
                )

            if "massive" in args.tasks:
                scope_results["massive"] = evaluate_massive_scope(
                    args=args,
                    experiment_config=experiment_config,
                    model=model,
                    tokenizer=tokenizer,
                    scope=scope,
                    scope_output_dir=scope_output_dir,
                    sample_recorder=sample_recorder,
                )

            sample_recorder.save(scope_output_dir)
            evaluation_metadata["results"][scope] = scope_results
            write_json(metadata_path, evaluation_metadata)

        evaluation_metadata["status"] = "completed"
        evaluation_metadata["completed_at"] = datetime.now().isoformat()
        write_json(metadata_path, evaluation_metadata)
    except Exception as error:
        evaluation_metadata["status"] = "failed"
        evaluation_metadata["failed_at"] = datetime.now().isoformat()
        evaluation_metadata["error_type"] = type(error).__name__
        evaluation_metadata["error_message"] = str(error)
        write_json(metadata_path, evaluation_metadata)
        raise


if __name__ == "__main__":
    main()
