import argparse
from datetime import datetime
import json
from pathlib import Path

import torch
from peft import PeftModel
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from data_utils import AlignmentDataset, MassiveDataset
from models import CustomModel


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
    return argparse.Namespace(**experiment_config)


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
def collect_alignment_embeddings(model, dataloader):
    groups = {}
    input_device = get_model_input_device(model)
    number_of_batches = len(dataloader)

    for batch_index, batch in enumerate(dataloader, start=1):
        model_batch = move_tensors_to_device(batch, input_device)
        outputs = model(
            forward_type="alignment",
            alignment=model_batch,
        )

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

        # Rank 1 means that no candidate has a strictly higher cosine score.
        ranks = (
            (similarity > correct_scores.unsqueeze(1)).sum(dim=1)
            + 1
        )

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
                    "lang": language,
                    "utt": item["utt"],
                    "target": dataset.extract_slots(item["annot_utt"]),
                }
            )

    return samples


def collate_massive_evaluation_samples(batch):
    return {
        "lang": [item["lang"] for item in batch],
        "utt": [item["utt"] for item in batch],
        "target": [item["target"] for item in batch],
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
def generate_massive_predictions(
    model,
    tokenizer,
    dataset,
    dataloader,
    max_new_tokens,
):
    predictions = []
    input_device = get_model_input_device(model)
    number_of_batches = len(dataloader)
    original_padding_side = tokenizer.padding_side

    try:
        # Decoder-only batched generation must be left padded so generation
        # starts after the last prompt token for every sample in the batch.
        tokenizer.padding_side = "left"

        for batch_index, batch in enumerate(dataloader, start=1):
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

            for item_index, generated_answer in enumerate(
                generated_answers
            ):
                target = batch["target"][item_index]
                predicted_slots = parse_slot_items(generated_answer)
                target_slots = parse_slot_items(target)

                predictions.append(
                    {
                        "lang": batch["lang"][item_index],
                        "utt": batch["utt"][item_index],
                        "target": target,
                        "prediction": generated_answer.strip(),
                        "target_slots": target_slots,
                        "predicted_slots": predicted_slots,
                    }
                )

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
    dataloader = DataLoader(
        dataset,
        batch_size=args.alignment_batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )
    embedding_groups = collect_alignment_embeddings(model, dataloader)
    metrics = evaluate_alignment_retrieval(
        embedding_groups,
        chunk_size=args.retrieval_chunk_size,
        device=get_model_input_device(model),
    )

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
    )
    metrics = evaluate_massive_predictions(predictions)

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
    checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()

    if not checkpoint_path.is_dir():
        raise NotADirectoryError(
            f"Checkpoint directory not found: {checkpoint_path}"
        )

    experiment_config = load_experiment_config(checkpoint_path)
    model, tokenizer = build_model(
        checkpoint_path,
        experiment_config,
    )

    scopes = get_language_scopes(args.language_scope)
    evaluation_root = get_evaluation_root(args, checkpoint_path)
    split_output_dir = evaluation_root / args.split
    split_output_dir.mkdir(parents=True, exist_ok=True)

    evaluation_metadata = {
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

            if "alignment" in args.tasks:
                scope_results["alignment"] = evaluate_alignment_scope(
                    args=args,
                    experiment_config=experiment_config,
                    model=model,
                    tokenizer=tokenizer,
                    scope=scope,
                    scope_output_dir=scope_output_dir,
                )

            if "massive" in args.tasks:
                scope_results["massive"] = evaluate_massive_scope(
                    args=args,
                    experiment_config=experiment_config,
                    model=model,
                    tokenizer=tokenizer,
                    scope=scope,
                    scope_output_dir=scope_output_dir,
                )

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
