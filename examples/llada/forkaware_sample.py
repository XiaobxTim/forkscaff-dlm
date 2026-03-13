"""
python -u examples/llada/forkaware_sample.py --model_name_or_path "YOUR_MODEL_PATH"
"""

from dataclasses import dataclass

import transformers

import dllm

import time

import json
from pathlib import Path
import math
import numpy as np
import matplotlib.pyplot as plt

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

print("mean_commit_stage_struct_efficiency:",
      mean_list(outputs.metrics["commit_stage_struct_efficiency"]))
print("mean_structural_commit_ratio:",
      mean_list(outputs.metrics["structural_commit_ratio"]))
print("mean_avg_attempt:",
      mean_list(outputs.metrics["avg_attempt"]))
print("mean_max_attempt:",
      mean_list(outputs.metrics["max_attempt"]))
print("mean_ready_coverage:",
      mean_list(outputs.metrics["ready_coverage"]))
print("mean_scheduler_activation_rate:",
      mean_list(outputs.metrics["scheduler_activation_rate"]))

generated_token_counts = [
    len(tokenizer.encode(s, add_special_tokens=False))
    for s in sequences
]
total_generated_tokens = sum(generated_token_counts)

decode_tps = total_generated_tokens / elapsed
print("total_generated_tokens:", total_generated_tokens)
print("decode_tps:", decode_tps)

attempt_list = outputs.metrics["avg_attempt"]
nfe_list = outputs.metrics["nfe"]

attempt_ratio_list = [
    a / n if n > 0 else 0.0
    for a, n in zip(attempt_list, nfe_list)
]

mean_attempt_ratio = mean_list(attempt_ratio_list)

print("mean_attempt_ratio:", mean_attempt_ratio)

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

def save_first_unmask_order(save_path, first_unmask_orders):
    """
    first_unmask_orders:
        list over samples, each sample is
        [(step, pos, token_id, token_str), ...]
    """
    payload = {}

    for i, sample_order in enumerate(first_unmask_orders):
        payload[f"sample_{i}"] = [
            {
                "step": int(step),
                "pos": int(pos),
                "token_id": int(token_id),
                "token": str(token),
            }
            for step, pos, token_id, token in sample_order
        ]

    Path(save_path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[Saved] {save_path}")

# mask_token_id = tokenizer.mask_token_id
# forkaware_first_orders = extract_first_unmask_order(
#     histories=outputs.histories,
#     tokenizer=tokenizer,
#     mask_token_id=tokenizer.mask_token_id,
# )

# save_first_unmask_order(
#     "forkaware_first_unmask.json",
#     forkaware_first_orders,
# )

def build_score_commit_pairs_from_metrics(metrics):
    """
    metrics["position_precommit_max_struct_score"]: list[B][T]
    metrics["position_first_commit_step"]: list[B][T]

    returns:
        pairs: list of (score, first_commit_step)
    """
    if "position_precommit_max_struct_score" not in metrics:
        raise KeyError("Missing metrics['position_precommit_max_struct_score']")
    if "position_first_commit_step" not in metrics:
        raise KeyError("Missing metrics['position_first_commit_step']")

    all_scores = metrics["position_precommit_max_struct_score"]
    all_steps = metrics["position_first_commit_step"]

    pairs = []

    for b in range(len(all_scores)):
        scores = all_scores[b]
        steps = all_steps[b]

        for pos in range(len(scores)):
            score = scores[pos]
            step = steps[pos]

            # 过滤无效值
            if step is None or int(step) < 0:
                continue
            if score is None:
                continue

            try:
                score = float(score)
            except Exception:
                continue

            if math.isnan(score) or math.isinf(score):
                continue

            pairs.append((score, int(step)))

    return pairs

def save_score_commit_pairs(save_path, pairs):
    payload = [
        {"score": float(score), "first_commit_step": int(step)}
        for score, step in pairs
    ]
    Path(save_path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[Saved] {save_path}")

def compute_score_commit_correlation(pairs):
    if len(pairs) < 3:
        print("Not enough pairs for correlation.")
        return None

    scores = np.array([p[0] for p in pairs], dtype=float)
    steps = np.array([p[1] for p in pairs], dtype=float)

    pearson = np.corrcoef(scores, steps)[0, 1]

    score_ranks = scores.argsort().argsort().astype(float)
    step_ranks = steps.argsort().argsort().astype(float)
    spearman = np.corrcoef(score_ranks, step_ranks)[0, 1]

    print(f"Pearson(score, first_commit_step): {pearson:.6f}")
    print(f"Spearman(score, first_commit_step): {spearman:.6f}")

    return pearson, spearman

def plot_score_commit_scatter(
    pairs,
    save_path="influence_commit_scatter.png",
    title="Structural Score vs First Commit Step",
):
    if len(pairs) == 0:
        print("No pairs to plot.")
        return

    scores = np.array([p[0] for p in pairs], dtype=float)
    steps = np.array([p[1] for p in pairs], dtype=float)

    plt.figure(figsize=(6, 5))
    plt.scatter(scores, steps, alpha=0.45, s=16)
    plt.xlabel("Pre-commit max structural score")
    plt.ylabel("First commit step")
    plt.title(title)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close()

    print(f"[Saved] {save_path}")

def plot_score_commit_binned(
    pairs,
    save_path="influence_commit_binned.png",
    title="Influence-Commit Trend",
    num_bins=5,
):
    if len(pairs) == 0:
        print("No pairs to plot.")
        return

    scores = np.array([p[0] for p in pairs], dtype=float)
    steps = np.array([p[1] for p in pairs], dtype=float)

    # 分位数分桶
    edges = np.quantile(scores, np.linspace(0.0, 1.0, num_bins + 1))
    edges = np.unique(edges)

    if len(edges) < 3:
        print("Not enough score variation for binning.")
        return

    xs, ys, yerr, labels = [], [], [], []

    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]

        if i == len(edges) - 2:
            mask = (scores >= lo) & (scores <= hi)
        else:
            mask = (scores >= lo) & (scores < hi)

        if mask.sum() == 0:
            continue

        xs.append((lo + hi) / 2.0)
        ys.append(steps[mask].mean())
        yerr.append(steps[mask].std())
        labels.append(f"[{lo:.3f}, {hi:.3f}]")

    plt.figure(figsize=(6, 5))
    plt.errorbar(xs, ys, yerr=yerr, marker="o", capsize=4)
    plt.xlabel("Structural score (binned)")
    plt.ylabel("Average first commit step")
    plt.title(title)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close()

    print(f"[Saved] {save_path}")
    print("Bin labels:", labels)

def build_score_commit_pairs_per_sample(metrics):
    if "position_precommit_max_struct_score" not in metrics:
        raise KeyError("Missing metrics['position_precommit_max_struct_score']")
    if "position_first_commit_step" not in metrics:
        raise KeyError("Missing metrics['position_first_commit_step']")

    all_scores = metrics["position_precommit_max_struct_score"]
    all_steps = metrics["position_first_commit_step"]

    results = []

    for b in range(len(all_scores)):
        pairs = []
        scores = all_scores[b]
        steps = all_steps[b]

        for pos in range(len(scores)):
            score = scores[pos]
            step = steps[pos]

            if step is None or int(step) < 0:
                continue
            if score is None:
                continue

            try:
                score = float(score)
            except Exception:
                continue

            if math.isnan(score) or math.isinf(score):
                continue

            pairs.append((score, int(step)))

        results.append(pairs)

    return results

## Compute Structure Score
# pairs = build_score_commit_pairs_from_metrics(outputs.metrics)
# print("num_score_commit_pairs:", len(pairs))

# save_score_commit_pairs("forkaware_score_commit_pairs.json", pairs)

# compute_score_commit_correlation(pairs)

# plot_score_commit_scatter(
#     pairs,
#     save_path="forkaware_influence_commit_scatter.png",
#     title="ForkAware: Structural Score vs First Commit Step",
# )

# plot_score_commit_binned(
#     pairs,
#     save_path="forkaware_influence_commit_binned.png",
#     title="ForkAware: Structural Score vs First Commit Step (Binned)",
# )

# pairs_per_sample = build_score_commit_pairs_per_sample(outputs.metrics)
# for sample_idx, sample_pairs in enumerate(pairs_per_sample):
#     print(f"[Sample {sample_idx}] num_pairs={len(sample_pairs)}")
#     if len(sample_pairs) == 0:
#         continue

#     compute_score_commit_correlation(sample_pairs)

    # plot_score_commit_scatter(
    #     sample_pairs,
    #     save_path=f"forkaware_sample{sample_idx}_scatter.png",
    #     title=f"Sample {sample_idx}: Structural Score vs First Commit Step",
    # )

    # plot_score_commit_binned(
    #     sample_pairs,
    #     save_path=f"forkaware_sample{sample_idx}_binned.png",
    #     title=f"Sample {sample_idx}: Structural Score vs First Commit Step (Binned)",
    # )

if script_args.visualize:
    terminal_visualizer.visualize(outputs.histories, rich=True)