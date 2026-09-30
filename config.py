import argparse


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
