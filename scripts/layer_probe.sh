#!/usr/bin/env bash
# nohup bash scripts/layer_probe.sh > logs/layer_probe.log 2>&1 &
#
# Which hidden layer gives usable sentence representations before training?
#
# At layer 8 the pretrained Qwen2.5-3B scored align_in 2.95 at step 0, above the
# ln(16) = 2.77 chance level for a batch of 16. That layer sits at 22% depth in a
# 36-layer stack, which may simply be too early to be semantic.
#
# eval_on_start evaluates before the first optimizer step, so this needs no
# training at all: one step per configuration is enough to read the pretrained
# alignment loss off eval_align_in_*. Compare the step-0 numbers across layers
# and pick the layer on that evidence rather than by assumption. Apply the same
# layer to every baseline afterwards.
export CUDA_VISIBLE_DEVICES=1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_ROOT}"

training_type=contrastive_only

model_name="Qwen/Qwen2.5-3B-Instruct"
# Qwen2.5-3B has 36 layers, so hidden_states has 37 entries: 8 is 22% depth,
# 18 is 50%, 27 is 75%, and -1 is the final layer.
layers=(8 18 27 -1)
positions=(last_token mean)

alignment_num_samples_per_lang=10000
batch_size=16
learning_rate=0.0001
num_steps=1
accumulative_steps=1
logging_steps=1
save_steps=1000000
eval_steps=1000000
eval_batch_size=16
output_root="./results_layer_probe"

training_anchor_langs=en
training_lang=(ko ja es)
out_inference_lang=(fr de it)

training_seed=42
alignment_temperature=0.05

project_name="${WANDB_PROJECT:-Oct_ARR_diag}"
wandb_mode="${WANDB_MODE:-offline}"

for alignment_hidden_state_position in "${positions[@]}"; do
  for alignment_hidden_state_layer in "${layers[@]}"; do
    model_tag="${model_name##*/}"
    training_lang_tag="$(IFS=-; printf '%s' "${training_lang[*]}")"
    out_lang_tag="$(IFS=-; printf '%s' "${out_inference_lang[*]}")"
    timestamp="$(date +'%Y%m%d_%H%M%S')"

    run_name="probe__${model_tag}__${alignment_hidden_state_position}__${alignment_hidden_state_layer}__in_${training_anchor_langs}-${training_lang_tag}__out_${out_lang_tag}__seed${training_seed}__${timestamp}"

    python3 main.py \
      --model_name "${model_name}" \
      --alignment_num_samples_per_lang "${alignment_num_samples_per_lang}" \
      --batch_size "${batch_size}" \
      --learning_rate "${learning_rate}" \
      --num_steps "${num_steps}" \
      --accumulative_steps "${accumulative_steps}" \
      --logging_steps "${logging_steps}" \
      --save_steps "${save_steps}" \
      --eval_steps "${eval_steps}" \
      --eval_batch_size "${eval_batch_size}" \
      --eval_language_scope in \
      --eval_sample_log_limit 0 \
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
