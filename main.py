from datetime import datetime
from importlib import metadata
import json
import os
from pathlib import Path
import re

import torch
import wandb
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed,
)

from config import parse_args
from custom_trainer import AlternativeRoutingTrainer
from data_utils import AlignmentDataset, CombinedDataset, MassiveDataset
from models import CustomModel
from utils import resolve_lora_target_modules


def path_safe_name(value, field_name):
    safe_value = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-")
    if not safe_value:
        raise ValueError(f"{field_name} must contain a path-safe character.")
    return safe_value


def build_run_name(args):
    if args.wandb_run_name:
        return path_safe_name(args.wandb_run_name, "wandb_run_name")

    model_tag = path_safe_name(
        args.model_name.split("/")[-1],
        "model_name",
    )
    in_languages = [
        args.training_anchor_langs,
        *args.training_lang,
    ]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        f"{model_tag}__{args.training_type}"
        f"__in_{'-'.join(in_languages)}"
        f"__out_{'-'.join(args.out_inference_lang)}"
        f"__{timestamp}"
    )


def planned_objective_updates(training_type, num_steps):
    if training_type == "transfer_only":
        return {"alignment": 0, "downstream": num_steps}
    if training_type == "contrastive_only":
        return {"alignment": num_steps, "downstream": 0}
    if training_type == "contrastive_then_transfer":
        alignment_steps = num_steps // 2
        return {
            "alignment": alignment_steps,
            "downstream": num_steps - alignment_steps,
        }
    if training_type == "alternative":
        return {
            "alignment": (num_steps + 1) // 2,
            "downstream": num_steps // 2,
        }
    raise ValueError(f"Unknown training_type: {training_type}")


def opus_config_name(first_language, second_language):
    """Resolve an OPUS-100 config name for a language pair.

    OPUS-100 ships exactly one direction per pair and always names it with the
    two codes in alphabetical order. Anchoring on the training language would
    therefore miss every language sorting before the anchor: with anchor "en",
    "en-de" does not exist and the corpus lives under "de-en".
    """
    return "-".join(sorted([first_language, second_language]))


def build_eval_datasets(args, tokenizer):
    """One validation dataset per objective and language.

    Trainer.evaluate handles a dict of eval datasets natively and prefixes each
    metric with its key, so this yields eval_massive_in_ko_loss and friends
    without any custom aggregation.

    Each dataset carries a single objective. That matters for alignment: InfoNCE
    uses in-batch negatives, so a loss measured over a batch mixing en-ko with
    en-ja describes a mixed negative pool rather than either language pair.
    """
    anchor_language = args.training_anchor_langs
    eval_datasets = {}

    language_scopes = (
        ("in", [anchor_language, *args.training_lang]),
        ("out", list(args.out_inference_lang)),
    )

    for scope, languages in language_scopes:
        if args.eval_language_scope != "both" and scope != args.eval_language_scope:
            continue

        split = f"{scope}_validation"

        for language in languages:
            eval_datasets[f"massive_{scope}_{language}"] = CombinedDataset(
                downstream_dataset=MassiveDataset(
                    args,
                    tokenizer=tokenizer,
                    split=split,
                    languages=[language],
                ),
            )

        for language in languages:
            # The anchor language has no parallel corpus with itself.
            if language == anchor_language:
                continue

            language_pair = opus_config_name(anchor_language, language)
            eval_datasets[f"align_{scope}_{language_pair}"] = CombinedDataset(
                alignment_dataset=AlignmentDataset(
                    args,
                    tokenizer=tokenizer,
                    split=split,
                    lang_pairs=[language_pair],
                ),
            )

    return eval_datasets


def lora_coverage(model):
    """How much of the backbone LoRA actually reached.

    Qwen3.5 interleaves linear_attention and full_attention layers and only the
    latter carry q_proj/v_proj, so the default target list reaches a quarter of
    its layers while covering all of Llama's and Qwen2.5's. Recording this keeps
    the resulting difference in adapter capacity visible instead of implicit.
    """
    decoder_layers = None
    for attribute in ("model", "base_model"):
        candidate = getattr(model, attribute, None)
        while candidate is not None and not hasattr(candidate, "layers"):
            candidate = getattr(candidate, "model", None)
        if candidate is not None:
            decoder_layers = candidate.layers
            break

    adapted_module_names = [
        name for name, module in model.named_modules()
        if hasattr(module, "lora_A")
    ]

    coverage = {
        "target_modules": None,
        "num_adapted_modules": len(adapted_module_names),
        "num_decoder_layers": None,
        "num_adapted_layers": None,
        "layer_coverage": None,
    }

    if decoder_layers is not None:
        coverage["num_decoder_layers"] = len(decoder_layers)
        adapted_layers = set()
        for name in adapted_module_names:
            parts = name.split(".")
            for index, part in enumerate(parts):
                if part == "layers" and index + 1 < len(parts):
                    adapted_layers.add(parts[index + 1])
                    break
        coverage["num_adapted_layers"] = len(adapted_layers)
        if len(decoder_layers):
            coverage["layer_coverage"] = (
                len(adapted_layers) / len(decoder_layers)
            )

    return coverage


def package_versions():
    versions = {}
    for package_name in (
        "accelerate",
        "bitsandbytes",
        "datasets",
        "peft",
        "torch",
        "transformers",
        "wandb",
    ):
        try:
            versions[package_name] = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            versions[package_name] = None
    return versions


def cuda_metadata():
    return {
        "available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
        "devices": [
            {
                "index": device_index,
                "name": torch.cuda.get_device_name(device_index),
                "capability": list(
                    torch.cuda.get_device_capability(device_index)
                ),
            }
            for device_index in range(torch.cuda.device_count())
        ],
    }


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(
            payload,
            output_file,
            ensure_ascii=False,
            indent=2,
            default=str,
        )


def configure_wandb(args, model_tag):
    os.environ["WANDB_PROJECT"] = args.wandb_project_name
    os.environ["WANDB_NAME"] = args.wandb_run_name
    os.environ["WANDB_MODE"] = args.wandb_mode

    in_language_tag = "-".join(
        [args.training_anchor_langs, *args.training_lang]
    )
    out_language_tag = "-".join(args.out_inference_lang)
    os.environ.setdefault(
        "WANDB_RUN_GROUP",
        f"{model_tag}__in_{in_language_tag}__out_{out_language_tag}",
    )
    os.environ.setdefault(
        "WANDB_TAGS",
        ",".join(
            [
                args.training_type,
                model_tag,
                f"in:{in_language_tag}",
                f"out:{out_language_tag}",
            ]
        ),
    )

    # Model checkpoints are stored on the experiment server. Avoid uploading
    # every 500-step checkpoint as a duplicate W&B artifact by default.
    os.environ.setdefault("WANDB_LOG_MODEL", "false")
    os.environ.setdefault("WANDB_WATCH", "false")


def main():
    args = parse_args()
    args.wandb_run_name = build_run_name(args)

    model_folder = path_safe_name(
        args.model_name.replace("/", "__"),
        "model_name",
    )
    output_dir = (
        Path(args.output_root)
        / model_folder
        / args.wandb_run_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    args.effective_output_dir = str(output_dir)

    configure_wandb(args, model_folder)

    # Trainer.__init__ calls set_seed(args.seed) too, but that happens after the
    # model exists. LoRA's A matrices are initialised inside get_peft_model
    # below, so without this call they are drawn from an unseeded global RNG and
    # differ on every run even for a fixed --training_seed. B is zero-initialised
    # so step 0 is unaffected, but the whole trajectory afterwards is not.
    set_seed(args.training_seed)

    run_metadata_path = output_dir / "run_metadata.json"
    run_metadata = {
        "status": "initializing",
        "created_at": datetime.now().isoformat(),
        "run_name": args.wandb_run_name,
        "output_dir": str(output_dir),
        "experiment_config": vars(args).copy(),
        "planned_objective_updates": planned_objective_updates(
            args.training_type,
            args.num_steps,
        ),
        "environment": {
            "packages": package_versions(),
            "cuda": cuda_metadata(),
        },
    }
    write_json(run_metadata_path, run_metadata)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.peft_target_modules is None:
        model_type = AutoConfig.from_pretrained(args.model_name).model_type
        args.peft_target_modules = resolve_lora_target_modules(model_type)
        print(
            f"[LoRA] target_modules resolved for model_type '{model_type}': "
            f"{args.peft_target_modules}"
        )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.peft_lora_r,
        lora_alpha=args.peft_lora_alpha,
        lora_dropout=args.peft_lora_dropout,
        target_modules=list(args.peft_target_modules),
    )

    quantization_config = None
    if args.quantization_load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=(
                args.quantization_use_double_quant
            ),
            bnb_4bit_quant_type=args.quantization_type,
            bnb_4bit_compute_dtype=getattr(
                torch,
                args.quantization_compute_dtype,
            ),
        )

    if not torch.cuda.is_available():
        raise EnvironmentError(
            "CUDA is not available. Please ensure you have a compatible "
            "GPU and the necessary drivers installed."
        )

    basemodel = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quantization_config,
        device_map="auto",
    )
    if args.quantization_load_in_4bit:
        basemodel = prepare_model_for_kbit_training(basemodel)
    basemodel = get_peft_model(basemodel, peft_config)
    model = CustomModel(config=args, basemodel=basemodel)

    adapter_coverage = lora_coverage(basemodel)
    adapter_coverage["target_modules"] = list(args.peft_target_modules)
    if adapter_coverage["layer_coverage"] is not None and (
        adapter_coverage["layer_coverage"] < 1.0
    ):
        print(
            f"[LoRA][WARNING] target_modules {args.peft_target_modules} reached "
            f"{adapter_coverage['num_adapted_layers']}/"
            f"{adapter_coverage['num_decoder_layers']} decoder layers "
            f"({adapter_coverage['layer_coverage']:.0%}). Adapter capacity is "
            "not comparable with models at full coverage."
        )

    alignment_dataset = AlignmentDataset(
        args,
        tokenizer=tokenizer,
        split="train",
    )
    massive_dataset = MassiveDataset(
        args,
        tokenizer=tokenizer,
        split="train",
    )
    

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    num_total_examples = (
        args.batch_size
        * world_size
        * args.accumulative_steps
        * args.num_steps
    )
    combined_dataset = CombinedDataset(
        alignment_dataset,
        massive_dataset,
        num_total_examples,
    )
    eval_datasets = build_eval_datasets(args, tokenizer)

    trainable_parameters = model.num_parameters(only_trainable=True)
    total_parameters = model.num_parameters()
    derived_metadata = {
        "world_size": world_size,
        "per_device_batch_size": args.batch_size,
        "effective_global_batch_size": (
            args.batch_size
            * world_size
            * args.accumulative_steps
        ),
        "combined_dataset_size": len(combined_dataset),
        "alignment_dataset_size": len(alignment_dataset),
        "alignment_pair_sizes": {
            language_pair: len(dataset)
            for language_pair, dataset
            in alignment_dataset.all_data.items()
        },
        "downstream_dataset_size": len(massive_dataset),
        "downstream_language_sizes": {
            language: len(dataset)
            for language, dataset
            in massive_dataset.all_data.items()
        },
        "eval_dataset_sizes": {
            name: len(dataset)
            for name, dataset in eval_datasets.items()
        },
        "lora_coverage": adapter_coverage,
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
        "trainable_parameter_ratio": (
            trainable_parameters / total_parameters
            if total_parameters
            else 0.0
        ),
    }
    run_metadata["derived"] = derived_metadata
    run_metadata["status"] = "ready"
    write_json(run_metadata_path, run_metadata)

    # WandbCallback reads this at train start, after the derived values above
    # are available.
    model.config.experiment_config = {
        **vars(args),
        "derived": derived_metadata,
        "planned_objective_updates": run_metadata[
            "planned_objective_updates"
        ],
    }

    report_to = (
        []
        if args.wandb_mode == "disabled"
        else ["wandb"]
    )
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        run_name=args.wandb_run_name,
        report_to=report_to,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        # transformers v5 dropped warmup_ratio and keeps only warmup_steps.
        # Converting here rather than exposing steps directly keeps the warmup
        # a constant fraction of training, which matters because num_steps
        # differs between the 50k and 100k methods.
        warmup_steps=int(args.warmup_ratio * args.num_steps),
        gradient_accumulation_steps=args.accumulative_steps,
        max_steps=args.num_steps,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        logging_first_step=True,
        logging_nan_inf_filter=False,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_only_model=False,
        save_on_each_node=False,
        seed=args.training_seed,
        data_seed=args.training_seed,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        # `evaluation_strategy` was removed in transformers v5.
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        per_device_eval_batch_size=args.eval_batch_size,
        # Never gather predictions: model outputs carry string metadata and
        # [B, T, vocab] logits, neither of which survives accumulation.
        prediction_loss_only=True,
        # Gives every run an untrained baseline row at step 0.
        eval_on_start=True,
        # `metric_for_best_model` is deliberately unset. A macro over languages
        # is not one of the emitted keys, and `_determine_best_metric` raises on
        # a missing key. Checkpoint selection happens offline from
        # trainer_state.json instead, which also keeps the selection rule
        # changeable without retraining.
    )

    trainer = AlternativeRoutingTrainer(
        model=model,
        args=training_args,
        train_dataset=combined_dataset,
        data_collator=combined_dataset.collate_fn,
        processing_class=tokenizer,
        training_type=args.training_type,
        total_steps=args.num_steps,
        eval_dataset=eval_datasets,
        eval_sample_log_limit=args.eval_sample_log_limit,
        # `compute_metrics` stays unset on purpose: it would switch
        # evaluation_loop out of prediction_loss_only mode and start
        # accumulating logits.
    )

    print("\n" + "*" * 60)
    print(f"Run name: {args.wandb_run_name}")
    print(f"Output directory: {output_dir}")
    print(f"Training type: {args.training_type}")
    print(f"Initial objective: {trainer.objective_for_step()}")
    print(f"Effective global batch size: {derived_metadata['effective_global_batch_size']}")
    print(f"Total optimizer steps: {args.num_steps}")
    print(f"Logging interval: {args.logging_steps}")
    print(f"Checkpoint interval: {args.save_steps}")
    print(f"Validation interval: {args.eval_steps}")
    print(f"Validation datasets: {len(eval_datasets)} (scope: {args.eval_language_scope})")
    print(
        f"Learning rate: {args.learning_rate} ({args.lr_scheduler_type}, "
        f"warmup {int(args.warmup_ratio * args.num_steps)} steps "
        f"= {args.warmup_ratio:g} x {args.num_steps})"
    )
    print("*" * 60 + "\n")

    try:
        run_metadata["status"] = "training"
        write_json(run_metadata_path, run_metadata)

        train_result = trainer.train(
            resume_from_checkpoint=args.resume_from_checkpoint,
        )
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        # The last evaluation round may have buffered samples that no
        # subsequent evaluate() call would write out.
        trainer.flush_eval_samples()

        run_metadata["status"] = "completed"
        run_metadata["completed_at"] = datetime.now().isoformat()
        run_metadata["train_metrics"] = train_result.metrics
        write_json(run_metadata_path, run_metadata)
    except Exception as error:
        run_metadata["status"] = "failed"
        run_metadata["failed_at"] = datetime.now().isoformat()
        run_metadata["error_type"] = type(error).__name__
        run_metadata["error_message"] = str(error)
        write_json(run_metadata_path, run_metadata)
        raise
    finally:
        if wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
