#!/usr/bin/env bash
# nohup bash scripts/contrastive_then_transfer.sh > logs/contrastive_then_transfer.log 2>&1 &
export CUDA_VISIBLE_DEVICES=1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_ROOT}"

training_type=contrastive_then_transfer

model_name="meta-llama/Llama-3.2-1B"
alignment_num_samples_per_lang=10000
batch_size=32
learning_rate=0.0001
warmup_ratio=0.1
num_steps=100000
accumulative_steps=1
logging_steps=1
save_steps=500
output_root="./results"

training_anchor_langs=en
training_lang=(ko ja es)
out_inference_lang=(fr de it)

alignment_hidden_state_layer=-1
alignment_hidden_state_position=last_token
alignment_temperature=0.05

project_name="${WANDB_PROJECT:-Oct_ARR}"
wandb_mode="${WANDB_MODE:-online}"
timestamp="$(date +'%Y%m%d_%H%M%S')"

model_tag="${model_name##*/}"
training_lang_tag="$(IFS=-; printf '%s' "${training_lang[*]}")"
out_lang_tag="$(IFS=-; printf '%s' "${out_inference_lang[*]}")"

run_name="${model_tag}__${training_type}__${alignment_hidden_state_position}__${alignment_hidden_state_layer}__in_${training_anchor_langs}-${training_lang_tag}__out_${out_lang_tag}__${timestamp}"

exec python3 main.py \
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
  --peft_lora_r 8 \
  --peft_lora_alpha 32 \
  --peft_lora_dropout 0.1 \
  --quantization_load_in_4bit \
  --quantization_use_double_quant \
  --quantization_compute_dtype float16 \
  --quantization_type nf4 \
  --wandb_project_name "${project_name}" \
  --wandb_run_name "${run_name}" \
  --wandb_mode "${wandb_mode}"
