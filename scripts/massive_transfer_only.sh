#!/usr/bin/env bash
# nohup bash scripts/massive_transfer_only.sh >> logs/transfer_only.log 2>&1 &
# nohup bash scripts/massive_transfer_only.sh > logs/same_pair_contrastive_then_transfer.log 2>&1 &
# nohup bash scripts/massive_transfer_only.sh > logs/transfer_and_cTT_same_pair_proposed.log 2>&1 &
# nohup bash scripts/massive_transfer_only.sh > logs/0921_Qwen3.5-2B-resumed_TransferandCTT.log 2>&1 &
# nohup bash scripts/massive_transfer_only.sh >> logs/llama_gap_distance_then_transfer.log 2>&1 &

# nohup bash scripts/massive_transfer_only.sh > logs/0922_Proposed.log 2>&1 &
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
set -euo pipefail
# if [[ "${DRY_RUN:-0}" != "1" ]]; then
#     sleep 8h
# fi
# "Qwen/Qwen3.5-2B" contrastive_then_transfer 다시 돌려야됨

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_ROOT}"

training_types=(
#   "transfer_only"
  "contrastive_then_transfer"
)

model_names=(
    "meta-llama/Llama-3.2-1B-Instruct"
    "Qwen/Qwen2.5-1.5B-Instruct"
    "Qwen/Qwen3.5-2B"
    # "Qwen/Qwen2.5-3B-Instruct"
    # "meta-llama/Llama-3.2-3B-Instruct"
    # "Qwen/Qwen3.5-4B"
    )
if [[ -n "${MODEL_NAME:-}" ]]; then
    model_names=("${MODEL_NAME}")
fi
alignment_num_samples_per_lang=10000
batch_size=16
learning_rate=0.0001
warmup_ratio=0.1
accumulative_steps=1
logging_steps=10
save_steps=1000
output_root="${OUTPUT_ROOT:-./results}"
alignment_losses=(
    "infonce"
    # "gap_distance_infonce"
    "centered_infonce"
    # "gap_direction_infonce"
)
# Set ALIGNMENT_LOSS to select one alignment loss (validation only for transfer_only).
if [[ -n "${ALIGNMENT_LOSS:-}" ]]; then
    alignment_losses=("${ALIGNMENT_LOSS}")
fi
train_sample_log_interval="${TRAIN_SAMPLE_LOG_INTERVAL:-1000}"
train_sample_log_limit="${TRAIN_SAMPLE_LOG_LIMIT:-8}"
alignment_batching="${ALIGNMENT_BATCHING:-same_pair}"

training_anchor_langs=en
training_lang=(ko ja es)
out_inference_lang=(fr de it)

training_seed="${TRAINING_SEED:-42}"

# Must stay identical to massive_contrastive_only.sh: the value is written into
# experiment_config.json and decides which layer evaluate.py extracts
# representations from, so a mismatch makes the retrieval table incomparable.
alignment_hidden_state_layer="${ALIGNMENT_HIDDEN_STATE_LAYER:--1}"
alignment_hidden_state_position=last_token
alignment_temperature=0.05 # Used by all InfoNCE variants.
eval_batch_size=16
eval_steps=2500
eval_language_scope="${TRAIN_EVAL_LANGUAGE_SCOPE:-both}"
eval_sample_log_limit=64

# Optional positional arguments select individual modes from this queue.
if (( $# > 0 )); then
    training_types=("$@")
fi
for mode in "${training_types[@]}"; do
    case "${mode}" in
        transfer_only|contrastive_then_transfer) ;;
        *) printf 'Unsupported training mode for this script: %s\n' "${mode}" >&2; exit 2 ;;
    esac
done

python_bin="${PYTHON_BIN:-/home/nlplab/anaconda3/envs/octarr/bin/python}"
log_dir="${PROJECT_ROOT}/logs"
dry_run="${DRY_RUN:-0}"
failed_runs=0
if [[ ! -x "${python_bin}" ]]; then
    printf 'Python executable not found: %s\n' "${python_bin}" >&2
    exit 2
fi
for alignment_loss in "${alignment_losses[@]}"; do
    case "${alignment_loss}" in
        infonce|gap_consistency|gap_distance_infonce|centered_infonce|gap_direction_infonce) ;;
        *) printf 'Unknown alignment loss: %s\n' "${alignment_loss}" >&2; exit 2 ;;
    esac
    if [[ "${alignment_loss}" != "infonce" && "${alignment_batching}" != "same_pair" ]]; then
        printf 'Gap/centered experiments require ALIGNMENT_BATCHING=same_pair.\n' >&2
        exit 2
    fi
done
if [[ ! "${CUDA_VISIBLE_DEVICES}" =~ ^[0-9]+$ ]]; then
    printf 'Select one GPU index with CUDA_VISIBLE_DEVICES.\n' >&2
    exit 2
fi
if [[ "${dry_run}" != "1" ]]; then
    mkdir -p "${log_dir}"
    exec 9>"${log_dir}/train-gpu-${CUDA_VISIBLE_DEVICES}.lock"
    if ! flock -n 9; then
        printf 'A training queue already owns GPU %s.\n' "${CUDA_VISIBLE_DEVICES}" >&2
        exit 1
    fi
fi

run_experiment() {
    if [[ "${dry_run}" == "1" ]]; then
        printf 'CUDA_VISIBLE_DEVICES=%q ' "${CUDA_VISIBLE_DEVICES}"
        printf '%q ' "${python_bin}" -u "$@"
        printf '\n'
        return
    fi
    local run_log="${log_dir}/${run_name}.log"
    printf '[START] GPU=%s loss=%s run=%s log=%s\n' \
        "${CUDA_VISIBLE_DEVICES}" "${alignment_loss}" "${run_name}" "${run_log}"
    if "${python_bin}" -u "$@" >"${run_log}" 2>&1; then
        printf '[DONE] %s\n' "${run_name}"
    else
        local status=$?
        printf '[FAILED] exit=%s run=%s log=%s\n' "${status}" "${run_name}" "${run_log}" >&2
        return "${status}"
    fi
}

for training_type in "${training_types[@]}"; do
  run_alignment_losses=("${alignment_losses[@]}")
  if [[ "${training_type}" == "transfer_only" ]]; then
    num_steps=50000
    # CE-only training runs once per model. Keep the first loss for validation.
    run_alignment_losses=("${alignment_losses[0]}")
  else
    num_steps=100000
  fi
  for alignment_loss in "${run_alignment_losses[@]}"; do
    for model_name in "${model_names[@]}"; do
    model_tag="${model_name##*/}"
    training_lang_tag="$(IFS=-; printf '%s' "${training_lang[*]}")"
    out_lang_tag="$(IFS=-; printf '%s' "${out_inference_lang[*]}")"

    project_name="${WANDB_PROJECT:-Oct_ARR}"
    wandb_mode="${WANDB_MODE:-online}"
    timestamp="$(date +'%Y%m%d_%H%M%S')"

    run_name="${model_tag}__${training_type}__${alignment_loss}__alignmentBatching_${alignment_batching}__${alignment_hidden_state_position}__${alignment_hidden_state_layer}__in_${training_anchor_langs}-${training_lang_tag}__out_${out_lang_tag}__seed${training_seed}__${timestamp}"

    run_experiment main.py \
        --downstream_task massive \
        --model_name "${model_name}" \
        --alignment_num_samples_per_lang "${alignment_num_samples_per_lang}" \
        --batch_size "${batch_size}" \
        --learning_rate "${learning_rate}" \
        --warmup_ratio "${warmup_ratio}" \
        --num_steps "${num_steps}" \
        --accumulative_steps "${accumulative_steps}" \
        --logging_steps "${logging_steps}" \
        --save_steps "${save_steps}" \
        --output_root "${output_root}" \
        --training_type "${training_type}" \
        --alignment_loss "${alignment_loss}" \
        --train_sample_log_interval "${train_sample_log_interval}" \
        --train_sample_log_limit "${train_sample_log_limit}" \
        --training_anchor_langs "${training_anchor_langs}" \
        --alignment_batching "${alignment_batching}" \
        --training_lang "${training_lang[@]}" \
        --out_inference_lang "${out_inference_lang[@]}" \
        --alignment_hidden_state_layer "${alignment_hidden_state_layer}" \
        --alignment_hidden_state_position "${alignment_hidden_state_position}" \
        --alignment_temperature "${alignment_temperature}" \
        --eval_batch_size "${eval_batch_size}" \
        --eval_steps "${eval_steps}" \
        --eval_language_scope "${eval_language_scope}" \
        --eval_sample_log_limit "${eval_sample_log_limit}" \
        --training_seed "${training_seed}" \
        --wandb_project_name "${project_name}" \
        --wandb_run_name "${run_name}" \
        --wandb_mode "${wandb_mode}" \
        --peft_lora_r 16 \
        --peft_lora_alpha 32 \
        --peft_lora_dropout 0.1 \
        --quantization_load_in_4bit \
        --quantization_use_double_quant \
        --quantization_compute_dtype bfloat16 \
        --quantization_type nf4 \
      || { failed_runs=$((failed_runs + 1)); continue; }
    done
  done
done

printf '[QUEUE FINISHED] GPU=%s failed_runs=%s\n' "${CUDA_VISIBLE_DEVICES}" "${failed_runs}"
(( failed_runs == 0 ))
