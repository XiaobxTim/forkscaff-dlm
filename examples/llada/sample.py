"""
python -u examples/llada/sample.py --model_name_or_path "YOUR_MODEL_PATH"
"""

from dataclasses import dataclass

import transformers

import dllm

import time

import torch


@dataclass
class ScriptArguments:
    model_name_or_path: str = "GSAI-ML/LLaDA-8B-Instruct"
    seed: int = 42
    visualize: bool = True

    def __post_init__(self):
        self.model_name_or_path = dllm.utils.resolve_with_base_env(
            self.model_name_or_path, "BASE_MODELS_DIR"
        )


@dataclass
class SamplerConfig(dllm.core.samplers.MDLMSamplerConfig):
    steps: int = 128
    max_new_tokens: int = 128
    block_size: int = 32
    temperature: float = 0.0
    remasking: str = "low_confidence"

def print_metrics(title, outputs):
    print(f"\n[{title}]")
    if not hasattr(outputs, "metrics"):
        print("No metrics found in outputs.")
        return

    metrics = outputs.metrics
    for k, v in metrics.items():
        print(f"{k}: {v}")

def mean_list(xs):
    return sum(xs) / len(xs)

def compute_stable_ratio_gen(histories, prompt_lens, max_new_tokens):
    """
    histories: list of [B, T] tensors, one per decoding step
    prompt_lens: list[int]
    max_new_tokens: int

    Returns:
        stable_ratios: list[float] for each sample
    """
    if len(histories) <= 1:
        B = histories[0].shape[0]
        return [1.0] * B

    H = torch.stack(histories, dim=0)  # [S, B, T]
    S, B, T = H.shape

    stable_ratios = []

    for j in range(B):
        start = prompt_lens[j]
        end = min(prompt_lens[j] + max_new_tokens, T)

        if end <= start:
            stable_ratios.append(1.0)
            continue

        # compare adjacent steps only on generation region
        same = (H[1:, j, start:end] == H[:-1, j, start:end]).float()

        ratio = same.mean().item()
        stable_ratios.append(ratio)

    return stable_ratios

parser = transformers.HfArgumentParser((ScriptArguments, SamplerConfig))
script_args, sampler_config = parser.parse_args_into_dataclasses()
transformers.set_seed(script_args.seed)

# Load model & tokenizer
model = dllm.utils.get_model(model_args=script_args).eval()
tokenizer = dllm.utils.get_tokenizer(model_args=script_args)
sampler = dllm.core.samplers.MDLMSampler(model=model, tokenizer=tokenizer)
terminal_visualizer = dllm.utils.TerminalVisualizer(tokenizer=tokenizer)

# --- Example 1: Batch sampling ---
print("\n" + "=" * 80)
print("TEST: llada.sample()".center(80))
print("=" * 80)

messages = [
    [{"role": "user", "content": "Lily runs 12 km/h for 4 hours. How far in 8 hours?"}],
    [{"role": "user", "content": "Please write an educational python function."}],
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
)

start = time.time()
outputs = sampler.sample(inputs, sampler_config, return_dict=True)
end = time.time()
sequences = dllm.utils.sample_trim(tokenizer, outputs.sequences.tolist(), inputs)

elapsed = end - start

print_metrics("Sample Metrics", outputs)

for iter, s in enumerate(sequences):
    print("\n" + "-" * 80)
    print(f"[Case {iter}]")
    print("-" * 80)
    print(s.strip() if s.strip() else "<empty>")
print("\n" + "=" * 80 + "\n")

print("mean_commit_stage_total_tps:",
      mean_list(outputs.metrics["commit_stage_total_tps"]))
print("mean_commit_stage_struct_efficiency:",
      mean_list(outputs.metrics["commit_stage_struct_efficiency"]))
print("mean_structural_commit_ratio:",
      mean_list(outputs.metrics["structural_commit_ratio"]))
print("mean_avg_attempt:",
      mean_list(outputs.metrics["avg_attempt"]))
print("mean_max_attempt:",
      mean_list(outputs.metrics["max_attempt"]))

stable_ratios = compute_stable_ratio_gen(
    histories=outputs.histories,
    prompt_lens=[len(x) for x in inputs],
    max_new_tokens=sampler_config.max_new_tokens,
)

print("mean_stable_ratio:", sum(stable_ratios) / len(stable_ratios))

generated_token_counts = [
    len(tokenizer.encode(s, add_special_tokens=False))
    for s in sequences
]
total_generated_tokens = sum(generated_token_counts)

decode_tps = total_generated_tokens / elapsed
print("total_generated_tokens:", total_generated_tokens)
print("decode_tps:", decode_tps)

def extract_first_unmask_order(histories, tokenizer, mask_token_id):
    """
    histories: list[Tensor], each [B, T]
    returns:
        results[b] = [(step, pos, token_id, token_str), ...] sorted by first unmask step
    """
    if histories is None or len(histories) <= 1:
        return []

    B, T = histories[0].shape
    results = [[] for _ in range(B)]
    first_seen = [dict() for _ in range(B)]

    for t in range(1, len(histories)):
        prev_state = histories[t - 1]
        curr_state = histories[t]

        for b in range(B):
            changed = (prev_state[b] != curr_state[b]).nonzero(as_tuple=False).squeeze(-1)

            for pos in changed.tolist():
                prev_id = int(prev_state[b, pos].item())
                curr_id = int(curr_state[b, pos].item())

                # 只统计从 mask -> token 的首次变化
                if prev_id == mask_token_id and curr_id != mask_token_id:
                    if pos not in first_seen[b]:
                        token_str = tokenizer.decode([curr_id])
                        first_seen[b][pos] = (t, pos, curr_id, token_str)

    for b in range(B):
        results[b] = sorted(first_seen[b].values(), key=lambda x: x[0])

    return results

mask_token_id = tokenizer.mask_token_id
first_orders = extract_first_unmask_order(
    histories=outputs.histories,
    tokenizer=tokenizer,
    mask_token_id=mask_token_id,
)

for sample_id, order in enumerate(first_orders):
    print(f"\n===== Sample {sample_id} First-Unmask Order =====")
    for step, pos, token_id, token_str in order:
        print(f"step {step:3d} | pos {pos:3d} | token {repr(token_str)}")

if script_args.visualize:
    terminal_visualizer.visualize(outputs.histories, rich=True)