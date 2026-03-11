"""
python -u examples/llada/forkaware_sample.py --model_name_or_path "YOUR_MODEL_PATH"
"""

from dataclasses import dataclass

import transformers

import dllm

import time


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
class SamplerConfig(dllm.core.samplers.ForkAwareMDLMSamplerConfig):
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

parser = transformers.HfArgumentParser((ScriptArguments, SamplerConfig))
script_args, sampler_config = parser.parse_args_into_dataclasses()
transformers.set_seed(script_args.seed)

# Load model & tokenizer
model = dllm.utils.get_model(model_args=script_args).eval()
tokenizer = dllm.utils.get_tokenizer(model_args=script_args)
sampler = dllm.core.samplers.ForkAwareMDLMSampler(model=model, tokenizer=tokenizer)
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
print("mean_avg_revisit:",
      mean_list(outputs.metrics["avg_revisit"]))
print("mean_max_revisit:",
      mean_list(outputs.metrics["max_revisit"]))
print("mean_ready_coverage:",
      mean_list(outputs.metrics["ready_coverage"]))
print("mean_scheduler_activation_rate:",
      mean_list(outputs.metrics["scheduler_activation_rate"]))

total_tokens = len(inputs) * sampler_config.max_new_tokens

decode_tps = total_tokens / elapsed

print("decode_tps:", decode_tps)

if script_args.visualize:
    terminal_visualizer.visualize(outputs.histories, rich=True)
