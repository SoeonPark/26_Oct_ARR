#!/usr/bin/env bash
# mkdir -p logs
# DRY_RUN=1 bash scripts/wmt23_contrastive_only.sh contrastive_only
# nohup bash scripts/wmt23_contrastive_only.sh contrastive_only > logs/wmt23_contrastive_only.log 2>&1 &
# Model/loss/mode arrays mirror the corresponding MASSIVE queue at creation.
# contrastive_only updates OPUS alignment only; WMT validation is diagnostic.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
# Alternating variable-length validation/training can fragment the CUDA cache.
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_ROOT}"

# ALMA de/cs + Japanese human-parallel extension; OPUS alignment; HF WMT23 tests.
wmt23_data_dir="${WMT23_DATA_DIR:-${PROJECT_ROOT}/data/wmt23_alma_ja_opus}"
wmt23_corpus_profile="${WMT23_CORPUS_PROFILE:-alma_ja_opus}"
wmt23_downstream_sampling="${WMT23_DOWNSTREAM_SAMPLING:-balanced_mixed}"

wmt23_manifest_sha256="${WMT23_MANIFEST_SHA256:-}"

training_types=(
    # "contrastive_only"
    "alternative"
)

model_names=(
    "meta-llama/Llama-3.2-1B-Instruct"
    # "Qwen/Qwen2.5-1.5B-Instruct"
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
output_root="${OUTPUT_ROOT:-./results_wmt23}"
alignment_losses=(
    # "infonce"
    "gap_distance_infonce"
    # "centered_infonce"
    # "gap_direction_infonce"
)
# Set ALIGNMENT_LOSS to select one loss instead of the active array above.
if [[ -n "${ALIGNMENT_LOSS:-}" ]]; then
    alignment_losses=("${ALIGNMENT_LOSS}")
fi
train_sample_log_interval="${TRAIN_SAMPLE_LOG_INTERVAL:-1000}"
train_sample_log_limit="${TRAIN_SAMPLE_LOG_LIMIT:-8}"
alignment_batching="${ALIGNMENT_BATCHING:-same_pair}"

training_anchor_langs=en
training_lang=(de cs ja)
out_inference_lang=(zh ru uk)

training_seed="${TRAINING_SEED:-42}"

# Must stay identical to wmt23_transfer_only.sh: the value is written into
# experiment_config.json and decides which layer evaluate.py extracts
# representations from, so a mismatch makes the retrieval table incomparable.
alignment_hidden_state_layer=-1 # 8
alignment_hidden_state_position=last_token
alignment_temperature=0.05 # Used by all InfoNCE variants.
eval_batch_size=16
eval_steps=2500
eval_sample_log_limit=64
eval_language_scope="${TRAIN_EVAL_LANGUAGE_SCOPE:-in}"

project_name="${WANDB_PROJECT:-Oct_ARR_WMT23}"
wandb_mode="${WANDB_MODE:-online}"

# Optional positional arguments select individual modes from this queue.
if (( $# > 0 )); then
    training_types=("$@")
fi
for mode in "${training_types[@]}"; do
    case "${mode}" in
        contrastive_only|alternative) ;;
        *) printf 'Unsupported training mode for this script: %s\n' "${mode}" >&2; exit 2 ;;
    esac
done

python_bin="${PYTHON_BIN:-/home/nlplab/anaconda3/envs/octarr/bin/python}"
log_dir="${PROJECT_ROOT}/logs"
dry_run="${DRY_RUN:-0}"
failed_runs=0
if [[ "${dry_run}" != "1" && ! -f "${wmt23_data_dir}/manifest.json" ]]; then
    printf 'WMT23 manifest not found: %s/manifest.json. Run scripts/prepare_wmt23.py or set WMT23_DATA_DIR.\n' "${wmt23_data_dir}" >&2
    exit 2
fi
if [[ ! -x "${python_bin}" ]]; then
    printf 'Python executable not found: %s\n' "${python_bin}" >&2
    exit 2
fi
if [[ "${dry_run}" != "1" && ! "${wmt23_manifest_sha256}" =~ ^[0-9a-f]{64}$ ]]; then
    printf 'Set WMT23_MANIFEST_SHA256 to the hash printed by prepare_wmt23.py.\n' >&2
    exit 2
fi
for alignment_loss in "${alignment_losses[@]}"; do
    case "${alignment_loss}" in
        infonce|gap_consistency|gap_distance_infonce|gap_distance_rms|gap_distance_detach|centered_infonce|gap_direction_infonce) ;;
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

for alignment_loss in "${alignment_losses[@]}"; do
for model_name in "${model_names[@]}"; do
    downstream_micro_batch_size=0
    if [[ "${model_name}" == "Qwen/Qwen3.5-2B" ]]; then
        downstream_micro_batch_size=8
    fi
    downstream_micro_batch_size="${DOWNSTREAM_MICRO_BATCH_SIZE:-${downstream_micro_batch_size}}"
    for training_type in "${training_types[@]}"; do
        # Same objective budgets as MASSIVE: 50k alignment updates in both modes.
        if [[ "${training_type}" == "contrastive_only" ]]; then
            num_steps=50000
        elif [[ "${training_type}" == "alternative" ]]; then
            num_steps=100000
        fi
        model_tag="${model_name##*/}"
        training_lang_tag="$(IFS=-; printf '%s' "${training_lang[*]}")"
        out_lang_tag="$(IFS=-; printf '%s' "${out_inference_lang[*]}")"
        timestamp="$(date +'%Y%m%d_%H%M%S')"

        run_name="${model_tag}__task_wmt23__corpus_${wmt23_corpus_profile}__sampling_${wmt23_downstream_sampling}__${training_type}__${alignment_loss}__alignmentBatching_${alignment_batching}__${alignment_hidden_state_position}__${alignment_hidden_state_layer}__in_${training_anchor_langs}-${training_lang_tag}__out_${out_lang_tag}__seed${training_seed}__${timestamp}"

        run_experiment main.py \
            --downstream_task wmt23 \
            --wmt23_data_dir "${wmt23_data_dir}" \
            --wmt23_manifest_sha256 "${wmt23_manifest_sha256}" \
            --wmt23_corpus_profile "${wmt23_corpus_profile}" \
            --wmt23_downstream_sampling "${wmt23_downstream_sampling}" \
            --model_name "${model_name}" \
            --alignment_num_samples_per_lang "${alignment_num_samples_per_lang}" \
            --batch_size "${batch_size}" \
            --downstream_micro_batch_size "${downstream_micro_batch_size}" \
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
            --peft_lora_r 16 \
            --peft_lora_alpha 32 \
            --peft_lora_dropout 0.1 \
            --quantization_load_in_4bit \
            --quantization_use_double_quant \
            --quantization_compute_dtype bfloat16 \
            --quantization_type nf4 \
            --wandb_project_name "${project_name}" \
            --wandb_run_name "${run_name}" \
            --wandb_mode "${wandb_mode}" \
            || { failed_runs=$((failed_runs + 1)); continue; }
    done
done
done

printf '[QUEUE FINISHED] GPU=%s failed_runs=%s\n' "${CUDA_VISIBLE_DEVICES}" "${failed_runs}"
(( failed_runs == 0 ))
