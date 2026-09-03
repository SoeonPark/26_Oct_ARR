export CUDA_VISIBLE_DEVICES=0
# nohup bash scripts/evaluate_gpu0.sh >> logs/evaluate_gpu0.log 2>&1 &

# python3 evaluator.py \
#   --checkpoint_path results/meta-llama__Llama-3.2-1B/Llama-3.2-1B__contrastive_only__last_token__8__in_en-ko-ja-es__out_fr-de-it__20260902_005235/checkpoint-5000 \
#   --split validation \
#   --language_scope both \
#   --tasks massive \
#   --massive_batch_size 32 \
#   --max_new_tokens 64

python3 evaluator.py \
  --checkpoint_path results/meta-llama__Llama-3.2-1B/Llama-3.2-1B__contrastive_only__last_token__8__in_en-ko-ja-es__out_fr-de-it__20260902_005235/checkpoint-5000 \
  --split validation \
  --language_scope both \
  --tasks massive \
  --massive_batch_size 32 \
  --max_new_tokens 128