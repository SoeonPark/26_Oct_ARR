#!/usr/bin/env bash
# nohup bash scripts/lr_sweep.sh > logs/lr_sweep.log 2>&1 &
#
# Does the learning rate explain how much InfoNCE-only damages the task?
#
# A contrastive_only run at lr=1e-3 reached align_in 2.95 -> 0.70 by step 2500
# while massive_in went 4.60 -> 9.70. Two readings fit that: the rate is too
# aggressive, or naive InfoNCE genuinely destroys task ability. They differ in
# what a lower rate does. If alignment still reaches ~0.7 with less task damage
# the rate was the cause; if the damage tracks the alignment gain it is a real
# property of the objective, which is what the proposed loss is meant to fix.
#
# Same model as that observation so the numbers stay comparable. In-language
# scope only: out-language data is reserved for final evaluation.
export CUDA_VISIBLE_DEVICES=0
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_ROOT}"

training_type=contrastive_only

model_name="Qwen/Qwen2.5-3B-Instruct"
learning_rates=(0.001 0.0005 0.0002)
warmup_ratio=0.1

alignment_num_samples_per_lang=10000
batch_size=16
num_steps=2000
accumulative_steps=1
logging_steps=10
save_steps=1000
eval_steps=1000
eval_batch_size=16
output_root="./results_lr_sweep"

training_anchor_langs=en
training_lang=(ko ja es)
out_inference_lang=(fr de it)

training_seed=42

alignment_hidden_state_layer=8
alignment_hidden_state_position=last_token
alignment_temperature=0.05

project_name="${WANDB_PROJECT:-Oct_ARR_diag}"
wandb_mode="${WANDB_MODE:-online}"

for learning_rate in "${learning_rates[@]}"; do
    model_tag="${model_name##*/}"
    training_lang_tag="$(IFS=-; printf '%s' "${training_lang[*]}")"
    out_lang_tag="$(IFS=-; printf '%s' "${out_inference_lang[*]}")"
    timestamp="$(date +'%Y%m%d_%H%M%S')"

    run_name="${model_tag}__${training_type}__${alignment_hidden_state_position}__${alignment_hidden_state_layer}__lr${learning_rate}__in_${training_anchor_langs}-${training_lang_tag}__out_${out_lang_tag}__seed${training_seed}__${timestamp}"

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
        --eval_steps "${eval_steps}" \
        --eval_batch_size "${eval_batch_size}" \
        --eval_language_scope in \
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
