import argparse


WMT23_ACCESSIBLE_EXCLUSIONS = {
    "Neulab-tedtalks_train-1-eng-heb": "Official URL returns HTML instead of the corpus archive.",
    "ELRC-wikipedia_health-1-eng-heb": "Direct ELRC server certificate has expired.",
}

WMT23_PARTITIONS = {
    "full_parallel": (("de", "he", "ja"), ("zh", "ru", "uk")),
    "accessible_parallel": (("de", "he", "ja"), ("zh", "ru", "uk")),
    "alma_ja_opus": (("de", "cs", "ja"), ("zh", "ru", "uk")),
}


ALIGNMENT_LOSSES = (
    "infonce", "gap_consistency", "gap_distance_infonce",
    "centered_infonce", "gap_direction_infonce",
)
PAIR_BATCH_LOSSES = ALIGNMENT_LOSSES[1:]
ALIGNMENT_REFERENCES = {
    "infonce": None,
    "gap_consistency": "microbatch_mean_distance",
    "gap_distance_infonce": "microbatch_mean_distance",
    "centered_infonce": "microbatch_language_centroids",
    "gap_direction_infonce": "microbatch_mean_gap_vector",
}


def resolve_alignment_loss(config):
    """Old experiment configurations used InfoNCE implicitly."""
    value = (
        config.get("alignment_loss", "infonce")
        if isinstance(config, dict)
        else getattr(config, "alignment_loss", "infonce")
    )
    if value not in ALIGNMENT_LOSSES:
        raise ValueError(f"Unsupported alignment_loss: {value!r}")
    return value


def validate_gap_config(args):
    """Validate before loading datasets or model weights."""
    if (getattr(args, "downstream_task", "massive") in {"wmt23", "wmt25"}
            and args.training_type == "alternative" and args.num_steps % 2):
        raise ValueError("WMT alternating training requires even num_steps.")
    loss_type = resolve_alignment_loss(args)
    for name, default in (
        ("train_sample_log_interval", 1000),
        ("train_sample_log_limit", 8),
        ("eval_sample_log_limit", 64),
    ):
        if getattr(args, name, default) < 0:
            raise ValueError(f"{name} must be nonnegative.")
    if loss_type != "gap_consistency" and not getattr(args, "alignment_temperature", 0.05) > 0:
        raise ValueError("alignment_temperature must be positive.")
    if loss_type == "gap_distance_infonce" and not getattr(args, "alignment_gap_scale", 1.0) > 0:
        raise ValueError("alignment_gap_scale must be positive.")
    if loss_type not in PAIR_BATCH_LOSSES:
        return
    if getattr(args, "eval_batch_size", 16) < 2:
        raise ValueError(f"{loss_type} requires eval_batch_size >= 2.")
    if getattr(args, "training_type", "transfer_only") != "transfer_only":
        if getattr(args, "alignment_batching", "mixed") != "same_pair":
            raise ValueError(
                f"Training with {loss_type} requires alignment_batching=same_pair."
            )
        if getattr(args, "batch_size", 32) < 2:
            raise ValueError(f"{loss_type} requires training batch_size >= 2.")


def parse_args():
    parser = argparse.ArgumentParser(description="Configuration for the 2026Oct_ARR project.")
    
    # Add arguments here
    parser.add_argument('--alignment_data', type=str, default='Helsinki-NLP/opus-100', help='Path to the dataset.')
    parser.add_argument('--downstream_task_data', type=str, default='AmazonScience/massive', help='Path to the downstream task dataset.')
    parser.add_argument('--downstream_task', choices=['massive', 'wmt25', 'wmt23'], default='massive', help='Supervised downstream task; alignment remains OPUS-100.')
    parser.add_argument('--wmt25_data_dir', type=str, default=None, help='Directory produced by scripts/prepare_wmt25.py. Its split seed and corpus profile must match this run.')
    parser.add_argument('--wmt25_corpus_profile', choices=['full_recipe', 'ted'], default='full_recipe', help='Required prepared-data profile. Reject a TED-only directory when full_recipe is requested.')
    parser.add_argument('--wmt25_downstream_sampling', choices=['proportional', 'language_balanced'], default='proportional', help='proportional shuffles the combined target-language pool into mixed-language batches; language_balanced selects one target language per optimizer update.')
    parser.add_argument('--wmt23_data_dir', type=str)
    parser.add_argument('--wmt23_manifest_sha256', type=str)
    parser.add_argument('--wmt23_corpus_profile', choices=list(WMT23_PARTITIONS), default='alma_ja_opus', help='alma_ja_opus uses ALMA German/Czech, a documented Japanese human-parallel extension, HF WMT23-Test, and separate OPUS alignment. Legacy recipe profiles remain readable.')
    parser.add_argument('--wmt23_downstream_sampling', choices=['proportional', 'balanced_mixed'], default='balanced_mixed', help='balanced_mixed equalizes seen-language example exposure over the run while shuffling mixed-language batches and cycling each full pool; proportional preserves historical runs.')
    parser.add_argument('--downstream_micro_batch_size', type=int, default=0, help='Split SFT only into smaller forwards/backwards within one optimizer update; 0 keeps the full batch. Alignment batch size is unchanged.')
    parser.add_argument('--alignment_num_samples_per_lang', type=int, default=10000, help='Number of samples per language for alignment data.')
    parser.add_argument('--alignment_sampling_seed', type=int, default=42, help='Random seed for sampling alignment data.')
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.2-1B', help='Name of the model to use.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Peak learning rate. Note that LoRA scales its update by alpha/r, so the effective step here is doubled at alpha=32, r=16. QLoRA reports 2e-4 for 7B/13B.')
    parser.add_argument('--lr_scheduler_type', type=str, default='linear', help='Learning rate schedule. Any name accepted by TrainingArguments (linear, cosine, constant, constant_with_warmup, ...). A global decay makes the two phases of contrastive_then_transfer see different average learning rates; constant removes the schedule as a variable.')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Fraction of total steps spent warming up from 0 to the peak learning rate. Zero, the historical default, starts the first step at the full rate.')
    # parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs for training.')
    parser.add_argument('--num_steps', type=int, default=100000, help='Number of training steps.')
    parser.add_argument('--accumulative_steps', type=int, default=1, help='Number of steps to accumulate gradients before updating model parameters.')
    parser.add_argument('--training_type', type=str, default='transfer_only', help='Type of training to use.', choices=['transfer_only', 'contrastive_only', 'alternative', 'contrastive_then_transfer'])
    parser.add_argument('--alignment_loss', choices=ALIGNMENT_LOSSES, default='infonce', help='Loss used by alignment updates and alignment validation. gap_consistency minimizes raw Euclidean distance variance within a single language-pair microbatch. transfer_only performs no alignment updates.')
    parser.add_argument('--alignment_batching', choices=['mixed', 'same_pair'], default='mixed', help='Alignment batch composition. mixed keeps the existing loader; same_pair uses one language pair per alignment update (single process/GPU).')
    parser.add_argument('--training_seed', type=int, default=42, help='Random seed for model training and data loading.')

    # PEFT (LoRA)
    parser.add_argument('--peft_lora_r', type=int, default=8, help='Rank of the LoRA update matrices.')
    parser.add_argument('--peft_lora_alpha', type=int, default=32, help='LoRA scaling factor.')
    parser.add_argument('--peft_lora_dropout', type=float, default=0.1, help='Dropout probability for LoRA layers.')
    parser.add_argument('--peft_target_modules', type=str, nargs='+', default=None, help='Module names LoRA attaches to. Resolved from utils.LORA_TARGET_MODULES by model_type when omitted, which adapts every linear layer in the transformer block per the QLoRA recommendation.')

    # Quantization
    parser.add_argument('--quantization_load_in_4bit', action=argparse.BooleanOptionalAction, default=True, help='Whether to load the model in 4-bit precision.')
    parser.add_argument('--quantization_use_double_quant', action=argparse.BooleanOptionalAction, default=True, help='Whether to use nested quantization for 4-bit weights.')
    parser.add_argument('--quantization_type', type=str, default='nf4', choices=['nf4', 'fp4'], help='4-bit quantization data type.')
    parser.add_argument('--quantization_compute_dtype', type=str, default='float16', choices=['float16', 'bfloat16', 'float32'], help='Compute dtype used by 4-bit layers.')
    
    # Multi-Language can be selected by list type
    parser.add_argument('--training_anchor_langs', type=str, default='en', help='Anchor languages for training, separated by commas.')
    parser.add_argument('--training_lang', type=str, nargs='+', default=['ko', 'ja', 'es'], help='List of training languages.')
    parser.add_argument('--out_inference_lang', type=str, nargs='+', default=['fr', 'de', 'it'], help='List of output inference languages.')
    
    parser.add_argument('--alignment_hidden_state_layer', type=int, default=-1, help='Layer of the model to use for alignment hidden states.')
    parser.add_argument('--alignment_hidden_state_position', type=str, default='last_token', help='Position of the hidden state to use for alignment.', choices=['last_token', 'mean']) 
    parser.add_argument("--alignment_temperature", type=float, default=0.05, help="Temperature for all InfoNCE variants; unused by gap_consistency.")
    parser.add_argument("--alignment_gap_scale", type=float, default=1.0, help="Fixed distance scale for gap_distance_infonce: score = -((distance - positive_mean_distance) / scale)^2.")
    parser.add_argument('--alignment_max_length', type=int, default=None, help='Maximum alignment sequence length. Falls back to tokenizer.model_max_length when omitted, which is the historical behaviour.')

    # Validation
    parser.add_argument('--eval_steps', type=int, default=2500, help='Number of optimizer updates between validations. Should be a multiple of save_steps so that every evaluated step has a checkpoint.')
    parser.add_argument('--eval_batch_size', type=int, default=16, help='Per-device validation batch size. 16 divides the 2000-example OPUS validation splits exactly, which keeps the InfoNCE negative pool constant across batches.')
    parser.add_argument('--wmt25_eval_batch_size', type=int, default=8, help='WMT25 validation batch size; independent of OPUS/MASSIVE.')
    parser.add_argument('--wmt25_eval_chunk_size', type=int, default=512, help='Tokens per FP32 CE chunk in WMT25 validation; no sequence truncation.')
    parser.add_argument('--eval_sample_log_limit', type=int, default=64, help='Per-sample validation records saved per language per round. Set 0 to disable.')
    parser.add_argument('--eval_language_scope', type=str, default='both', choices=['in', 'out', 'both'], help="Which language scopes to build validation datasets for. Diagnostic runs that only need in-language signal can halve the validation cost with 'in'. Out-language data must not be used for model selection either way.")

    # Logging and checkpointing
    parser.add_argument('--output_root', type=str, default='./results', help='Root directory for experiment outputs.')
    parser.add_argument('--logging_steps', type=int, default=10, help='Number of optimizer updates between metric logs.')
    parser.add_argument('--train_sample_log_interval', type=int, default=1000, help='Alignment optimizer updates between detailed training sample records. Set 0 to disable; each microbatch of a selected update is eligible.')
    parser.add_argument('--train_sample_log_limit', type=int, default=8, help='Maximum detailed records per selected alignment microbatch. Set 0 to disable.')
    parser.add_argument('--save_steps', type=int, default=500, help='Number of optimizer updates between checkpoints.')
    parser.add_argument('--save_total_limit', type=int, default=None, help='Maximum number of checkpoints to retain. Keep all when omitted.')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='Checkpoint directory from which to resume training.')

    # Weights & Biases
    parser.add_argument('--wandb_project_name', type=str, default='Oct_ARR', help='Weights & Biases project name.')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='Weights & Biases run name and output directory name.')
    parser.add_argument('--wandb_mode', type=str, default='online', choices=['online', 'offline', 'disabled'], help='Weights & Biases logging mode.')
    
    args = parser.parse_args()
    return args
