#!/usr/bin/env bash
# nohup bash scripts/contrastive_only.sh >> logs/contrastive_only.log 2>&1 &
export CUDA_VISIBLE_DEVICES=1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_ROOT}"

training_type=(
    "contrastive_only"
    "alternative"
)

model_names=(
    "Qwen/Qwen2.5-3B-Instruct"
    "meta-llama/Llama-3.2-3B-Instruct"
    "meta-llama/Llama-3.2-1B-Instruct"
    "Qwen/Qwen3.5-4B"
    "Qwen/Qwen3.5-2B"
    "Qwen/Qwen2.5-1.5B-Instruct"
    )
alignment_num_samples_per_lang=10000
batch_size=16
learning_rate=0.0001
warmup_ratio=0.1
accumulative_steps=1
logging_steps=10
save_steps=500
output_root="./results"

training_anchor_langs=en
training_lang=(ko ja es)
out_inference_lang=(fr de it)

training_seed=42

# Must stay identical to transfer_only.sh: the value is written into
# experiment_config.json and decides which layer evaluator.py extracts
# representations from, so a mismatch makes the retrieval table incomparable.
alignment_hidden_state_layer=8
alignment_hidden_state_position=last_token
alignment_temperature=0.05

project_name="${WANDB_PROJECT:-Oct_ARR}"
wandb_mode="${WANDB_MODE:-online}"

for training_type in "${training_type[@]}"; do
    # One step is one optimizer update of one objective, so the step budget is
    # not the same as the objective budget. contrastive_only spends every step
    # on alignment; alternative splits them 1:1. Both land on 50k alignment
    # updates and 50k task updates, matching transfer_only (50k task) and
    # contrastive_then_transfer (50k + 50k).
    if [[ "${training_type}" == "contrastive_only" ]]; then
        num_steps=50000
    elif [[ "${training_type}" == "alternative" ]]; then
        num_steps=100000
    fi
    for model_name in "${model_names[@]}"; do

        model_tag="${model_name##*/}"
        training_lang_tag="$(IFS=-; printf '%s' "${training_lang[*]}")"
        out_lang_tag="$(IFS=-; printf '%s' "${out_inference_lang[*]}")"
        timestamp="$(date +'%Y%m%d_%H%M%S')"

        run_name="${model_tag}__${training_type}__${alignment_hidden_state_position}__${alignment_hidden_state_layer}__in_${training_anchor_langs}-${training_lang_tag}__out_${out_lang_tag}__seed${training_seed}__${timestamp}"

        python3 main.py \
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
            --training_anchor_langs "${training_anchor_langs}" \
            --training_lang "${training_lang[@]}" \
            --out_inference_lang "${out_inference_lang[@]}" \
            --alignment_hidden_state_layer "${alignment_hidden_state_layer}" \
            --alignment_hidden_state_position "${alignment_hidden_state_position}" \
            --alignment_temperature "${alignment_temperature}" \
            --training_seed "${training_seed}" \
            --peft_lora_r 16 \
            --peft_lora_alpha 32 \
            --peft_lora_dropout 0.1 \
            --quantization_load_in_4bit \
            --quantization_use_double_quant \
            --quantization_compute_dtype float16 \
            --quantization_type nf4 \
            --wandb_project_name "${project_name}" \
            --wandb_run_name "${run_name}" \
            --wandb_mode "${wandb_mode}" \
            || { echo "!!! RUN FAILED: ${run_name}" >&2; continue; }
    done
done