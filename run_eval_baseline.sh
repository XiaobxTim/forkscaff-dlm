#!/bin/bash
set -e

export HF_ALLOW_CODE_EVAL=1

MODEL="/root/autodl-tmp/LLaDA-8B-Instruct"
RESULT_DIR="/root/autodl-tmp/results_baseline"

mkdir -p $RESULT_DIR

# echo "===== Running GSM8K ====="
# accelerate launch --num_processes 1 \
#   dllm/pipelines/llada/eval.py \
#   --tasks gsm8k_cot \
#   --model llada \
#   --apply_chat_template \
#   --num_fewshot 8 \
#   --model_args "pretrained=$MODEL,config=baseline_default" \
#   --output_path $RESULT_DIR/gsm8k

# echo "===== Running MATH500 ====="
# accelerate launch --num_processes 1 \
#   dllm/pipelines/llada/eval.py \
#   --tasks hendrycks_math500 \
#   --model llada \
#   --apply_chat_template \
#   --num_fewshot 4 \
#   --model_args "pretrained=$MODEL,config=baseline_default" \
#   --output_path $RESULT_DIR/math500

# echo "===== Running HumanEval ====="
# accelerate launch --num_processes 1 \
#   dllm/pipelines/llada/eval.py \
#   --tasks humaneval \
#   --model llada \
#   --apply_chat_template \
#   --model_args "pretrained=$MODEL,config=baseline_default" \
#   --confirm_run_unsafe_code \
#   --output_path $RESULT_DIR/humaneval

# echo "===== Running MBPP ====="
# accelerate launch --num_processes 1 \
#   dllm/pipelines/llada/eval.py \
#   --tasks mbpp \
#   --model llada \
#   --apply_chat_template \
#   --num_fewshot 3 \
#   --model_args "pretrained=$MODEL,config=baseline_default" \
#   --confirm_run_unsafe_code \
#   --output_path $RESULT_DIR/mbpp

echo "===== Running GSM8K pass@k ====="
accelerate launch --num_processes 1 \
  dllm/pipelines/llada/eval.py \
  --include_path ./custom_lm \
  --tasks gsm8k_cot_passk \
  --model llada \
  --apply_chat_template \
  --num_fewshot 8 \
  --model_args "pretrained=$MODEL,config=baseline_default" \
  --log_samples \
  --output_path $RESULT_DIR/gsm8k_passk

echo "===== Running MATH500 pass@k ====="
accelerate launch --num_processes 1 \
  dllm/pipelines/llada/eval.py \
  --include_path ./custom_lm \
  --tasks hendrycks_math500_passk \
  --model llada \
  --apply_chat_template \
  --num_fewshot 4 \
  --model_args "pretrained=$MODEL,config=baseline_default" \
  --log_samples \
  --output_path $RESULT_DIR/math500_passk

echo "===== Running HumanEval pass@k ====="
accelerate launch --num_processes 1 \
  dllm/pipelines/llada/eval.py \
  --include_path ./custom_lm \
  --tasks humaneval_passk \
  --model llada \
  --apply_chat_template \
  --model_args "pretrained=$MODEL,config=baseline_default" \
  --confirm_run_unsafe_code \
  --log_samples \
  --output_path $RESULT_DIR/humaneval_passk

echo "===== Running MBPP pass@k ====="
accelerate launch --num_processes 1 \
  dllm/pipelines/llada/eval.py \
  --include_path ./custom_lm \
  --tasks mbpp_passk \
  --model llada \
  --apply_chat_template \
  --num_fewshot 3 \
  --model_args "pretrained=$MODEL,config=baseline_default" \
  --confirm_run_unsafe_code \
  --log_samples \
  --output_path $RESULT_DIR/mbpp_passk