#!/bin/bash
set -e

export HF_ALLOW_CODE_EVAL=1

MODEL="/root/autodl-tmp/LLaDA-8B-Instruct"
RESULT_DIR="/root/autodl-tmp/results_forkaware"

mkdir -p $RESULT_DIR

echo "===== Running GSM8K ====="
accelerate launch --num_processes 1 \
  dllm/pipelines/llada/forkaware_eval.py \
  --tasks gsm8k_cot \
  --model forkaware \
  --apply_chat_template \
  --num_fewshot 8 \
  --model_args "pretrained=$MODEL,config=forkaware_default" \
  --output_path $RESULT_DIR/gsm8k

echo "===== Running MATH500 ====="
accelerate launch --num_processes 1 \
  dllm/pipelines/llada/forkaware_eval.py \
  --tasks hendrycks_math500 \
  --model forkaware \
  --apply_chat_template \
  --num_fewshot 4 \
  --model_args "pretrained=$MODEL,config=forkaware_default" \
  --output_path $RESULT_DIR/math500

echo "===== Running HumanEval ====="
accelerate launch --num_processes 1 \
  dllm/pipelines/llada/forkaware_eval.py \
  --tasks humaneval \
  --model forkaware \
  --apply_chat_template \
  --model_args "pretrained=$MODEL,config=forkaware_default" \
  --confirm_run_unsafe_code \
  --output_path $RESULT_DIR/humaneval

echo "===== Running MBPP ====="
accelerate launch --num_processes 1 \
  dllm/pipelines/llada/forkaware_eval.py \
  --tasks mbpp \
  --model forkaware \
  --apply_chat_template \
  --num_fewshot 3 \
  --model_args "pretrained=$MODEL,config=forkaware_default" \
  --confirm_run_unsafe_code \
  --output_path $RESULT_DIR/mbpp

echo "===== Summarizing Results ====="
python summarize_results.py $RESULT_DIR