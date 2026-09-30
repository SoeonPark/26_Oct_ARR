#!/usr/bin/env bash
# conda activate octarr
# mkdir -p logs
# nohup bash scripts/eval.sh > logs/eval.log 2>&1 &
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
python_bin="${PYTHON_BIN:-python3}"
results_root="${RESULTS_ROOT:-${PROJECT_ROOT}/results}"

# List "model_folder/run_name" paths relative to results/ in evaluation order.
# A run directory selects the final saved model; append /checkpoint-N to select
# a specific checkpoint. Absolute paths are also supported.
run_paths=(
    # "Qwen__Qwen3.5-2B/your_run_name"
    # "meta-llama__Llama-3.2-1B-Instruct/your_run_name"
    # "Qwen__Qwen3.5-2B/your_run_name/checkpoint-50000"
)

split="${EVAL_SPLIT:-test}"
language_scope="${EVAL_LANGUAGE_SCOPE:-both}"
read -r -a tasks <<< "${EVAL_TASKS:-alignment massive}"
alignment_batch_size="${EVAL_ALIGNMENT_BATCH_SIZE:-32}"
massive_batch_size="${EVAL_MASSIVE_BATCH_SIZE:-32}"
retrieval_chunk_size="${EVAL_RETRIEVAL_CHUNK_SIZE:-256}"
max_new_tokens="${EVAL_MAX_NEW_TOKENS:-128}"
# Number of samples to save per task and language/pair; set 0 to disable.
eval_sample_log_limit="${EVAL_SAMPLE_LOG_LIMIT:-64}"
# Set true to also save all alignment tensors alongside the sample JSON/pickle.
save_alignment_embeddings="${EVAL_SAVE_ALIGNMENT_EMBEDDINGS:-false}"
# Full scalar diagnostics are independent of the bounded embedding records.
save_alignment_sample_metrics="${EVAL_SAVE_ALIGNMENT_SAMPLE_METRICS:-true}"

# --dry-run: validate paths and print commands without loading models.
# Positional path arguments override the run_paths array above.
# bash scripts/eval.sh --dry-run "model_folder/run_name" "model_folder/another_run"
dry_run=false
if [[ "${1:-}" == "--dry-run" ]]; then
    dry_run=true
    shift
fi
if (( $# > 0 )); then
    run_paths=("$@")
fi
if (( ${#run_paths[@]} == 0 )); then
    printf 'Add model_folder/run_name entries to run_paths in scripts/eval.sh, or pass them as arguments.\n' >&2
    exit 1
fi

# Validate every path and required model file before starting the first evaluation.
checkpoint_paths=()
for run_path in "${run_paths[@]}"; do
    if [[ "${run_path}" == /* ]]; then
        checkpoint_path="${run_path}"
    else
        checkpoint_path="${results_root}/${run_path}"
    fi
    for required_file in experiment_config.json adapter_config.json; do
        if [[ ! -f "${checkpoint_path}/${required_file}" ]]; then
            printf 'Required evaluation file not found: %s/%s\n' "${checkpoint_path}" "${required_file}" >&2
            exit 1
        fi
    done
    if [[ ! -f "${checkpoint_path}/adapter_model.safetensors" && ! -f "${checkpoint_path}/adapter_model.bin" ]]; then
        printf 'Adapter weights not found: %s\n' "${checkpoint_path}" >&2
        exit 1
    fi
    checkpoint_paths+=("${checkpoint_path}")
done

for index in "${!checkpoint_paths[@]}"; do
    checkpoint_path="${checkpoint_paths[index]}"
    command=(
        "${python_bin}" -u "${PROJECT_ROOT}/evaluate.py"
        --checkpoint_path "${checkpoint_path}"
        --split "${split}"
        --language_scope "${language_scope}"
        --tasks "${tasks[@]}"
        --alignment_batch_size "${alignment_batch_size}"
        --massive_batch_size "${massive_batch_size}"
        --retrieval_chunk_size "${retrieval_chunk_size}"
        --max_new_tokens "${max_new_tokens}"
        --eval_sample_log_limit "${eval_sample_log_limit}"
    )
    if [[ -n "${EVAL_OUTPUT_DIR:-}" ]]; then
        command+=(--output_dir "${EVAL_OUTPUT_DIR}")
    fi
    if [[ "${save_alignment_embeddings}" == true ]]; then
        command+=(--save_alignment_embeddings)
    else
        command+=(--no-save_alignment_embeddings)
    fi
    if [[ "${save_alignment_sample_metrics}" == true ]]; then
        command+=(--save_alignment_sample_metrics)
    else
        command+=(--no-save_alignment_sample_metrics)
    fi

    printf '\n[%d/%d] GPU=%s checkpoint=%s\n' \
        "$((index + 1))" "${#checkpoint_paths[@]}" "${CUDA_VISIBLE_DEVICES}" "${checkpoint_path}"
    if [[ "${dry_run}" == true ]]; then
        printf '%q ' "${command[@]}"
        printf '\n'
    else
        "${command[@]}"
    fi
done
