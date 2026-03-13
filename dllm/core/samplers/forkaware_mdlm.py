"""
reference: https://github.com/ML-GSAI/LLaDA/blob/main/generate.py
"""

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from dllm.core.samplers.base import BaseSampler, BaseSamplerConfig, BaseSamplerOutput
from dllm.core.samplers.utils import add_gumbel_noise, get_num_transfer_tokens

from collections import deque, Counter, defaultdict

import time

def _sync_time():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


@dataclass
class ForkAwareMDLMSamplerConfig(BaseSamplerConfig):
    max_new_tokens: int = 128
    max_length: int = (
        None  # There's no explicit length_limit except for the tokenizer/model context
    )
    block_size: int = 128
    steps: int = 128
    temperature: float = 0.0
    remasking: str = "low_confidence"
    stochastic_transfer: bool = False
    cfg_scale: float = 0.0
    cfg_keep_tokens: list[int] | None = None
    suppress_tokens: list[int] | None = None
    begin_suppress_tokens: list[int] | None = None
    right_shift_logits: bool = False
    downstream_window: int = 16
    probe_top_m: int = 4
    downstream_top_r: int = 2
    probe_divergence: str = "js"
    structural_top_k: int = 2
    branch_top_k_values: int = 1   # 最小版只用 top-1
    probe_interval: int = 8
    alpha_struct: float = 1.0      # I_down 权重
    beta_branch: float = 0.5       # I_branch 权重
    consistency_window: int = 4
    min_history_for_consistency: int = 3
    entropy_threshold: float = 4.5
    consistency_threshold: float = 0.75
    lambda_confidence: float = 1.0
    lambda_structural: float = 0.5
    commit_top_m: int = 1
    readiness_window: int = 4
    readiness_boost: float = 1.0
    readiness_decay: float = 0.8
    lambda_readiness: float = 0.5
    epsilon_ready: float = 0.3
    enable_ready_scheduler: bool = True


@dataclass
class ForkAwareMDLMSampler(BaseSampler):
    def __init__(self, model, tokenizer):
        super().__init__(model=model, tokenizer=tokenizer)
        self.pred_history = {}
        self.branch_history = {}

        self.readiness_map = None

        self.metrics_state = None

    def reset_histories(self):
        self.pred_history = {}
        self.branch_history = {}
    
    def reset_scheduler_state(self):
        self.readiness_map = None

    def init_readiness_map(
        self,
        batch_size,
        seq_len,
        device,
    ):
        self.readiness_map = torch.zeros(
            (batch_size, seq_len),
            dtype=torch.float32,
            device=device,
        )

    def reset_metrics_state(self):
        self.metrics_state = None


    def init_metrics_state(
        self,
        batch_size,
        seq_len,
        device,
    ):
        self.metrics_state = {
            # efficiency
            "nfe": torch.zeros(batch_size, dtype=torch.long, device=device),

            # commitment
            "struct_commit_total": torch.zeros(batch_size, dtype=torch.long, device=device),
            "total_commit_total": torch.zeros(batch_size, dtype=torch.long, device=device),
            "eligible_total": torch.zeros(batch_size, dtype=torch.long, device=device),

            # scheduler
            "scheduler_activation_steps": torch.zeros(batch_size, dtype=torch.long, device=device),
            "scheduler_total_steps": torch.zeros(batch_size, dtype=torch.long, device=device),
            "ready_coverage_sum": torch.zeros(batch_size, dtype=torch.float32, device=device),
            "ready_coverage_count": torch.zeros(batch_size, dtype=torch.long, device=device),

            "attempt_counts": torch.zeros((batch_size, seq_len), dtype=torch.long, device=device),

            # commit trajectory
            "commit_trajectory": [[] for _ in range(batch_size)],
            "commit_phase_time_sum": 0.0,
            "commit_phase_struct_time_sum": 0.0,

            "position_first_seen_struct_score": torch.full(
                (batch_size, seq_len),
                float("nan"),
                dtype=torch.float32,
                device=device,
            ),
            
            "position_precommit_max_struct_score": torch.full(
                (batch_size, seq_len),
                float("-inf"),
                dtype=torch.float32,
                device=device,
            ),
            
            "position_commit_step_struct_score": torch.full(
                (batch_size, seq_len),
                float("nan"),
                dtype=torch.float32,
                device=device,
            ),
            
            "position_first_commit_step": torch.full(
                (batch_size, seq_len),
                -1,
                dtype=torch.long,
                device=device,
            ),
        }

    def update_position_score_traces(
        self,
        candidate_indices,
        structural_source_scores,
        mask_index,
    ):
        if self.metrics_state is None:
            return
    
        first_seen = self.metrics_state["position_first_seen_struct_score"]
        precommit_max = self.metrics_state["position_precommit_max_struct_score"]
        first_commit_step = self.metrics_state["position_first_commit_step"]
    
        B, K = candidate_indices.shape
    
        for j in range(B):
            for k in range(K):
                pos = int(candidate_indices[j, k].item())
                if pos < 0:
                    continue
                if not bool(mask_index[j, pos].item()):
                    continue
    
                score = structural_source_scores[j, k].float()
    
                # 第一次看到这个位置时的结构分数
                if torch.isnan(first_seen[j, pos]):
                    first_seen[j, pos] = score
    
                # 仅在它尚未 first-commit 前，累计 precommit max
                if int(first_commit_step[j, pos].item()) < 0:
                    precommit_max[j, pos] = torch.maximum(precommit_max[j, pos], score)

    def compute_ready_mask(
        self,
        mask_index,
        epsilon_ready,
    ):
        """
        Compute token-level ready mask from readiness_map.
        Only currently masked positions can be ready.
        """
        if self.readiness_map is None:
            return mask_index.clone()

        ready_mask = self.readiness_map >= epsilon_ready
        ready_mask = ready_mask & mask_index
        return ready_mask
    
    def get_step_state(
        self,
        x,
        attention_mask,
        temperature,
        cfg_scale=0.0,
        unmasked_index=None,
        suppress_tokens=None,
        right_shift_logits=False,
        begin_suppress_tokens=None,
    ):
        mask_id = self.tokenizer.mask_token_id

        mask_index = x == mask_id  # current global mask map

        if cfg_scale > 0.0:
            un_x = x.clone()
            un_x[unmasked_index] = mask_id
            x_ = torch.cat([x, un_x], dim=0)
            logits = self.model(
                x_, attention_mask=attention_mask.repeat(2, 1)
            ).logits
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
        else:
            logits = self.model(
                x, attention_mask=attention_mask
            ).logits  # Use attention mask here

        if suppress_tokens is not None and len(suppress_tokens) > 0:
            for token_id in suppress_tokens:
                logits[:, :, token_id] = -torch.inf

        if right_shift_logits:
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

        # Argmax decoding with optional Gumbel-Max noise for exploration
        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        x0 = torch.argmax(
            logits_with_noise, dim=-1
        )  # [B, T] predicted token ids

        if begin_suppress_tokens is not None and len(begin_suppress_tokens) > 0:
            for token_id in begin_suppress_tokens:
                logits[:, :, token_id] = -torch.inf
        
        return logits, x0, mask_index
    
    def count_block_masked_positions(self, mask_index, prompt_lens, block_idx, block_size):
        B = mask_index.shape[0]
        counts = []

        for j in range(B):
            start = prompt_lens[j] + block_idx * block_size
            end = min(start + block_size, mask_index.shape[1])
            if start >= end:
                counts.append(0)
            else:
                counts.append(int(mask_index[j, start:end].sum().item()))

        return counts
    
    def apply_readiness_bias_to_entropy(
        self,
        entropy,
        lambda_readiness,
    ):
        """
        Add readiness bonus to entropy-based preselection score.

        Args:
            entropy: Tensor [B, T]
            lambda_readiness: float

        Returns:
            biased_entropy: Tensor [B, T]
        """
        if self.readiness_map is None:
            return entropy

        return entropy + lambda_readiness * self.readiness_map.to(entropy.dtype)
    
    def update_scheduler_metrics(
        self,
        original_candidate_mask,
        ready_mask,
        candidate_mask,
    ):
        if self.metrics_state is None:
            return

        B = candidate_mask.shape[0]
        self.metrics_state["scheduler_total_steps"] += 1

        for j in range(B):
            final_count = int(candidate_mask[j].sum().item())
            if final_count == 0:
                continue

            ready_count = int((candidate_mask[j] & ready_mask[j]).sum().item())
            coverage = ready_count / final_count

            self.metrics_state["ready_coverage_sum"][j] += float(coverage)
            self.metrics_state["ready_coverage_count"][j] += 1

            original_count = int(original_candidate_mask[j].sum().item())
            if final_count < original_count:
                self.metrics_state["scheduler_activation_steps"][j] += 1

    def preselect_candidates(
        self,
        logits,
        mask_index,
        prompt_lens,
        block_idx,
        block_size,
        max_new_tokens,
        top_m,
        config=None,
    ):
        """
        在当前 block 内，从 still-masked positions 中选出高熵候选。
        这里加入:
        1) readiness soft bias (3.5)
        2) readiness-guided scheduler gating (3.6 minimal)

        Args:
            logits: [B, T, V]
            mask_index: [B, T] bool
            prompt_lens: list[int]
            block_idx: 当前 block 编号
            block_size: 当前 block 大小
            top_m: 每个 sample 最多保留多少个候选
            config: sampler config

        Returns:
            candidate_indices: [B, K]
            candidate_scores:  [B, K]
        """
        probs = F.softmax(logits, dim=-1)  # [B, T, V]
        entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1)  # [B, T]

        # Step 3.5: readiness soft bias
        lambda_readiness = getattr(config, "lambda_readiness", 0.0) if config is not None else 0.0
        biased_entropy = self.apply_readiness_bias_to_entropy(
            entropy=entropy,
            lambda_readiness=lambda_readiness,
        )

        B, T = entropy.shape
        candidate_mask = torch.zeros_like(mask_index, dtype=torch.bool)

        # original block-local candidate window
        for j in range(B):
            start = prompt_lens[j] + block_idx * block_size
            end = min(start + block_size, prompt_lens[j] + max_new_tokens, T)
            if start < end:
                candidate_mask[j, start:end] = mask_index[j, start:end]

        original_candidate_mask = candidate_mask.clone()

        # ------------------------------------------------------------------
        # Step 3.6: readiness-guided online scheduler (minimal token-level)
        # ------------------------------------------------------------------
        if config is not None and getattr(config, "enable_ready_scheduler", False):
            ready_mask = self.compute_ready_mask(
                mask_index=mask_index,
                epsilon_ready=config.epsilon_ready,
            )

            candidate_mask = candidate_mask & ready_mask

            for j in range(B):
                if candidate_mask[j].sum() == 0:
                    start = prompt_lens[j] + block_idx * block_size
                    end = min(start + block_size, prompt_lens[j] + max_new_tokens, T)
                    if start < end:
                        candidate_mask[j, start:end] = mask_index[j, start:end]

            self.update_scheduler_metrics(
                original_candidate_mask=original_candidate_mask,
                ready_mask=ready_mask,
                candidate_mask=candidate_mask,
            )

        # Apply candidate mask after scheduler
        masked_entropy = torch.where(
            candidate_mask,
            biased_entropy,
            torch.full_like(biased_entropy, float("-inf")),
        )

        valid_counts = candidate_mask.sum(dim=1)  # [B]
        max_k = int(valid_counts.max().item()) if valid_counts.numel() > 0 else 0
        k = min(top_m, max_k)

        if k == 0:
            empty_idx = torch.empty((B, 0), dtype=torch.long, device=logits.device)
            empty_scores = torch.empty((B, 0), dtype=logits.dtype, device=logits.device)
            return empty_idx, empty_scores

        candidate_scores, candidate_indices = torch.topk(masked_entropy, k=k, dim=1)

        return candidate_indices, candidate_scores
    
    def update_attempt_metrics(self, candidate_indices):
        if self.metrics_state is None:
            return

        B, K = candidate_indices.shape
        for j in range(B):
            for pos in candidate_indices[j].tolist():
                pos = int(pos)
                if pos >= 0:
                    self.metrics_state["attempt_counts"][j, pos] += 1
    
    def distribution_distance(
        self,
        p_base,
        p_cf,
        mode="l1",
        eps=1e-8,
    ):
        """
        p_base, p_cf: [N, V]
        return: [N]
        """
        p_base = p_base.clamp_min(eps)
        p_cf = p_cf.clamp_min(eps)

        if mode == "l1":
            return torch.abs(p_base - p_cf).sum(dim=-1)

        if mode == "kl":
            # KL(p_base || p_cf)
            return (p_base * (p_base.log() - p_cf.log())).sum(dim=-1)

        if mode == "js":
            m = 0.5 * (p_base + p_cf)
            return 0.5 * (
                (p_base * (p_base.log() - m.log())).sum(dim=-1)
                + (p_cf * (p_cf.log() - m.log())).sum(dim=-1)
            )

        raise NotImplementedError(f"Unknown divergence mode: {mode}")
        
    def get_block_candidate_mask(
        self,
        mask_index,
        prompt_lens,
        block_idx,
        block_size,
        max_new_tokens,
    ):
        """
        返回当前 block 内仍为 mask 的位置布尔掩码 [B, T]
        """
        B, T = mask_index.shape
        block_mask = torch.zeros_like(mask_index, dtype=torch.bool)

        for j in range(B):
            start = prompt_lens[j] + block_idx * block_size
            end = min(start + block_size, prompt_lens[j] + max_new_tokens, T)
            if start < end:
                block_mask[j, start:end] = mask_index[j, start:end]

        return block_mask

    def get_local_downstream_indices(
        self,
        block_mask_row: torch.Tensor,
        candidate_pos: int,
        window: int,
    ) -> torch.Tensor:
        """
        Select a local unresolved neighborhood for downstream influence.
        We only keep the nearest `window` masked positions after candidate_pos.
        """
        masked_pos = torch.nonzero(block_mask_row, as_tuple=False).squeeze(-1)
    
        if masked_pos.numel() == 0:
            return masked_pos
    
        future_pos = masked_pos[masked_pos > candidate_pos]
    
        if future_pos.numel() == 0:
            return future_pos
    
        return future_pos[:window]

    def select_downstream_candidates(
        self,
        candidate_indices: torch.Tensor,   # [B, K]
        candidate_scores: torch.Tensor,    # [B, K]
        top_r: int,
    ):
        """
        Select a small subset of candidates for downstream probing.
        Assumes larger candidate_scores means higher priority.
        Returns:
            selected_indices: [B, R]
            selected_pos: [B, R] positions in the original candidate list
        """
        B, K = candidate_indices.shape
        R = min(top_r, K)
    
        top_vals, top_pos = torch.topk(candidate_scores, k=R, dim=-1)
        selected_indices = torch.gather(candidate_indices, dim=1, index=top_pos)
    
        return selected_indices, top_pos

    def scatter_downstream_scores(
        self,
        small_scores: torch.Tensor,   # [B, R]
        selected_pos: torch.Tensor,   # [B, R]
        full_k: int,
    ):
        """
        Scatter downstream scores back to [B, K], filling unselected positions with 0.
        """
        B, R = small_scores.shape
        full_scores = torch.zeros(
            (B, full_k),
            dtype=small_scores.dtype,
            device=small_scores.device,
        )
        full_scores.scatter_(dim=1, index=selected_pos, src=small_scores)
        return full_scores
        
    def estimate_downstream_influence(
        self,
        x,
        attention_mask,
        logits,
        candidate_indices,
        mask_index,
        prompt_lens,
        block_idx,
        block_size,
        max_new_tokens,
        cfg_scale=0.0,
        unmasked_index=None,
        suppress_tokens=None,
        begin_suppress_tokens=None,
        right_shift_logits=False,
        divergence_mode="l1",
        downstream_window=16,
    ):
        """
        For each candidate position s, perform one top-1 hypothetical commit,
        re-run forward, and measure its average distributional effect on a LOCAL
        unresolved neighborhood in the current block.
    
        Args:
            x: [B, T]
            attention_mask: [B, T]
            logits: [B, T, V] current base logits
            candidate_indices: [B, K]
            mask_index: [B, T] bool
            prompt_lens: list[int]
            block_idx, block_size, max_new_tokens: current block info
            downstream_window: number of nearby unresolved positions to inspect
    
        Returns:
            influence_scores: [B, K]
        """
        B, T, V = logits.shape
        K = candidate_indices.shape[1]
    
        # base probs
        base_probs = F.softmax(logits, dim=-1)  # [B, T, V]
    
        # masked positions in the current block
        block_mask = self.get_block_candidate_mask(
            mask_index=mask_index,
            prompt_lens=prompt_lens,
            block_idx=block_idx,
            block_size=block_size,
            max_new_tokens=max_new_tokens,
        )  # [B, T]
    
        # top-1 hypothetical commit token
        top1_tokens = torch.argmax(logits, dim=-1)  # [B, T]
    
        influence_scores = torch.zeros(
            (B, K), dtype=logits.dtype, device=logits.device
        )
    
        for j in range(B):
            for k in range(K):
                s = int(candidate_indices[j, k].item())
    
                # invalid or already unmasked
                if not (0 <= s < T) or not bool(mask_index[j, s].item()):
                    influence_scores[j, k] = 0.0
                    continue
    
                # build counterfactual input
                x_cf = x.clone()
                x_cf[j, s] = top1_tokens[j, s]
    
                # counterfactual forward
                cf_logits, _, cf_mask_index = self.get_step_state(
                    x=x_cf,
                    attention_mask=attention_mask,
                    temperature=0.0,
                    cfg_scale=cfg_scale,
                    unmasked_index=unmasked_index,
                    suppress_tokens=suppress_tokens,
                    begin_suppress_tokens=begin_suppress_tokens,
                    right_shift_logits=right_shift_logits,
                )
                cf_probs = F.softmax(cf_logits, dim=-1)  # [B, T, V]
    
                # local unresolved neighborhood after s
                target_indices = self.get_local_downstream_indices(
                    block_mask_row=block_mask[j],
                    candidate_pos=s,
                    window=downstream_window,
                )
    
                if target_indices.numel() == 0:
                    influence_scores[j, k] = 0.0
                    continue
    
                p_base = base_probs[j, target_indices, :]  # [N, V]
                p_cf = cf_probs[j, target_indices, :]      # [N, V]
    
                dist = self.distribution_distance(
                    p_base,
                    p_cf,
                    mode=divergence_mode,
                )  # [N]
    
                influence_scores[j, k] = dist.mean()
    
        return influence_scores
    
    def get_topk_candidate_values(
        self,
        logits,
        candidate_indices,
        top_k_values=2,
    ):
        """
        logits: [B, T, V]
        candidate_indices: [B, K]
        return:
            topk_token_ids: [B, K, top_k_values]
        """
        B, K = candidate_indices.shape
        topk_token_ids = torch.zeros(
            (B, K, top_k_values),
            dtype=torch.long,
            device=logits.device,
        )

        for j in range(B):
            for k in range(K):
                s = int(candidate_indices[j, k].item())
                token_ids = torch.topk(logits[j, s], k=top_k_values, dim=-1).indices
                topk_token_ids[j, k] = token_ids

        return topk_token_ids
    
    def estimate_branch_sensitivity(
        self,
        x,
        attention_mask,
        logits,
        candidate_indices,
        mask_index,
        prompt_lens,
        block_idx,
        block_size,
        max_new_tokens,
        cfg_scale=0.0,
        unmasked_index=None,
        suppress_tokens=None,
        begin_suppress_tokens=None,
        right_shift_logits=False,
        divergence_mode="l1",
        top_k_values=2,
        branch_window=16,
    ):
        """
        For each candidate position s:
        take top-2 plausible values, perform two hypothetical commits,
        and compare the difference they induce on a LOCAL unresolved neighborhood.
    
        Returns:
            branch_scores: [B, K]
        """
        B, T, V = logits.shape
        K = candidate_indices.shape[1]
    
        if K == 0:
            return torch.empty((B, 0), dtype=logits.dtype, device=logits.device)
    
        # masked positions in the current block
        block_mask = self.get_block_candidate_mask(
            mask_index=mask_index,
            prompt_lens=prompt_lens,
            block_idx=block_idx,
            block_size=block_size,
            max_new_tokens=max_new_tokens,
        )  # [B, T]
    
        # top-2 plausible values for each candidate
        topk_token_ids = self.get_topk_candidate_values(
            logits=logits,
            candidate_indices=candidate_indices,
            top_k_values=top_k_values,
        )  # [B, K, 2]
    
        branch_scores = torch.zeros((B, K), dtype=logits.dtype, device=logits.device)
    
        for j in range(B):
            for k in range(K):
                s = int(candidate_indices[j, k].item())
    
                if not (0 <= s < T) or not bool(mask_index[j, s].item()):
                    branch_scores[j, k] = 0.0
                    continue
    
                if top_k_values < 2:
                    branch_scores[j, k] = 0.0
                    continue
    
                v1 = int(topk_token_ids[j, k, 0].item())
                v2 = int(topk_token_ids[j, k, 1].item())
    
                # if top-2 collapse to the same token, no branch difference
                if v1 == v2:
                    branch_scores[j, k] = 0.0
                    continue
    
                # counterfactual #1
                x_cf1 = x.clone()
                x_cf1[j, s] = v1
                cf1_logits, _, _ = self.get_step_state(
                    x=x_cf1,
                    attention_mask=attention_mask,
                    temperature=0.0,
                    cfg_scale=cfg_scale,
                    unmasked_index=unmasked_index,
                    suppress_tokens=suppress_tokens,
                    begin_suppress_tokens=begin_suppress_tokens,
                    right_shift_logits=right_shift_logits,
                )
                cf1_probs = F.softmax(cf1_logits, dim=-1)
    
                # counterfactual #2
                x_cf2 = x.clone()
                x_cf2[j, s] = v2
                cf2_logits, _, _ = self.get_step_state(
                    x=x_cf2,
                    attention_mask=attention_mask,
                    temperature=0.0,
                    cfg_scale=cfg_scale,
                    unmasked_index=unmasked_index,
                    suppress_tokens=suppress_tokens,
                    begin_suppress_tokens=begin_suppress_tokens,
                    right_shift_logits=right_shift_logits,
                )
                cf2_probs = F.softmax(cf2_logits, dim=-1)
    
                # local unresolved neighborhood after s
                target_indices = self.get_local_downstream_indices(
                    block_mask_row=block_mask[j],
                    candidate_pos=s,
                    window=branch_window,
                )
    
                if target_indices.numel() == 0:
                    branch_scores[j, k] = 0.0
                    continue
    
                p1 = cf1_probs[j, target_indices, :]   # [N, V]
                p2 = cf2_probs[j, target_indices, :]   # [N, V]
    
                dist = self.distribution_distance(
                    p1,
                    p2,
                    mode=divergence_mode,
                )  # [N]
    
                branch_scores[j, k] = dist.mean()
    
        return branch_scores
    
    def get_branch_plausibility_weight(
        self,
        logits,
        candidate_indices,
        eps=1e-8,
    ):
        """
        根据 top1/top2 概率差，给 I_branch 一个 plausibility weight。
        返回 [B, K]
        """
        probs = F.softmax(logits, dim=-1)
        B, K = candidate_indices.shape

        weights = torch.zeros((B, K), dtype=logits.dtype, device=logits.device)

        for j in range(B):
            for k in range(K):
                s = int(candidate_indices[j, k].item())
                top2_probs = torch.topk(probs[j, s], k=2, dim=-1).values
                p1 = top2_probs[0]
                p2 = top2_probs[1]

                # 一个简单稳定的版本：用 top2 mass 或者 (1 - margin)
                margin = p1 - p2
                weights[j, k] = torch.clamp(1.0 - margin, min=0.0)

        return weights
    
    def fuse_structural_scores(
        self,
        i_down_scores,
        i_branch_scores,
        alpha=1.0,
        beta=1.0,
    ):
        """
        S_struct = alpha * I_down + beta * I_branch
        """
        return alpha * i_down_scores + beta * i_branch_scores
    
    def select_structural_candidates(
        self,
        candidate_indices,
        structural_source_scores,
        top_k,
    ):
        """
        candidate_indices: [B, K_probe]
        structural_source_scores: [B, K_probe]  # 现在可以是 S_struct
        """
        B, K_probe = candidate_indices.shape

        if K_probe == 0:
            empty_idx = torch.empty((B, 0), dtype=torch.long, device=candidate_indices.device)
            empty_scores = torch.empty((B, 0), dtype=structural_source_scores.dtype, device=structural_source_scores.device)
            return empty_idx, empty_scores

        k = min(top_k, K_probe)
        structural_scores, select_pos = torch.topk(structural_source_scores, k=k, dim=1)
        structural_indices = torch.gather(candidate_indices, dim=1, index=select_pos)

        return structural_indices, structural_scores
    
    def topk_from_scores(self, candidate_indices, score_tensor, top_k):
        """
        candidate_indices: [B, K_probe]
        score_tensor:      [B, K_probe]
        return:
            top_indices: [B, K]
            top_scores:  [B, K]
        """
        B, K_probe = candidate_indices.shape

        if K_probe == 0:
            empty_idx = torch.empty((B, 0), dtype=torch.long, device=candidate_indices.device)
            empty_scores = torch.empty((B, 0), dtype=score_tensor.dtype, device=score_tensor.device)
            return empty_idx, empty_scores

        k = min(top_k, K_probe)
        top_scores, select_pos = torch.topk(score_tensor, k=k, dim=1)
        top_indices = torch.gather(candidate_indices, dim=1, index=select_pos)
        return top_indices, top_scores

    def update_prediction_history(
        self,
        x0,
        mask_index,
        maxlen=4,
    ):
        """
        Record top-1 predictions of masked positions.

        Args:
            x0: Tensor [B, T]
                current top-1 predictions
            mask_index: Tensor [B, T]
                which positions are still masked
            maxlen: history window size
        """
        B, T = x0.shape

        for j in range(B):
            # 只遍历仍 masked 的位置
            positions = torch.nonzero(mask_index[j], as_tuple=False).squeeze(-1)

            for pos in positions.tolist():

                key = (j, int(pos))

                # 如果这个位置第一次出现
                if key not in self.pred_history:
                    self.pred_history[key] = deque(maxlen=maxlen)

                # 当前 step 的 top-1 prediction
                pred_token = int(x0[j, pos].item())

                self.pred_history[key].append(pred_token)
    
    def compute_consistency_scores(
        self,
        structural_indices,
        min_history=2,
    ):
        """
        Compute cross-step consistency scores for structural candidates.

        Args:
            structural_indices: Tensor [B, K]
            min_history: minimum history length required to trust consistency

        Returns:
            consistency_scores: Tensor [B, K]
        """
        B, K = structural_indices.shape
        device = structural_indices.device

        scores = torch.zeros((B, K), dtype=torch.float32, device=device)

        for j in range(B):
            for k in range(K):
                pos = int(structural_indices[j, k].item())

                # 跳过无效位置（如果你后面有 -1 padding）
                if pos < 0:
                    scores[j, k] = 0.0
                    continue

                key = (j, pos)

                if key not in self.pred_history:
                    scores[j, k] = 0.0
                    continue

                hist = list(self.pred_history[key])

                # 冷启动保护：history 太短时不认为它稳定
                if len(hist) < min_history:
                    scores[j, k] = 0.0
                    continue

                current_pred = hist[-1]
                same_count = sum(1 for v in hist if v == current_pred)

                scores[j, k] = same_count / len(hist)

        return scores
    
    def compute_candidate_entropy(
        self,
        logits,
        structural_indices,
    ):
        """
        Compute entropy for current structural candidates.

        Args:
            logits: Tensor [B, T, V]
            structural_indices: Tensor [B, K]

        Returns:
            entropy_scores: Tensor [B, K]
        """
        B, K = structural_indices.shape
        device = logits.device
        dtype = logits.dtype

        # [B, T, V]
        probs = F.softmax(logits, dim=-1)

        # [B, T]
        token_entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1)

        # [B, K]
        entropy_scores = torch.zeros((B, K), dtype=token_entropy.dtype, device=device)

        for j in range(B):
            for k in range(K):
                pos = int(structural_indices[j, k].item())

                # 如果后面你有 -1 padding，先兼容一下
                if pos < 0:
                    entropy_scores[j, k] = 0.0
                    continue

                entropy_scores[j, k] = token_entropy[j, pos]

        return entropy_scores
    
    def get_eligible_candidates_minimal(
        self,
        structural_indices,
        entropy_scores,
        consistency_scores,
        entropy_threshold,
        consistency_threshold,
    ):
        """
        Minimal eligibility gate:
        low entropy + high consistency
        """
        eligible_mask = (
            (entropy_scores <= entropy_threshold)
            & (consistency_scores >= consistency_threshold)
        )
        return eligible_mask


    def finalize_eligible_candidates(
        self,
        structural_indices,
        eligible_mask,
    ):
        """
        Finalize eligibility output for downstream usage.
        """
        eligible_indices = torch.full_like(structural_indices, fill_value=-1)
        eligible_indices[eligible_mask] = structural_indices[eligible_mask]

        eligible_counts = eligible_mask.sum(dim=-1)

        return eligible_indices, eligible_counts


    def compute_candidate_confidence(
        self,
        logits,
        eligible_indices,
    ):
        """
        Compute local confidence C(s;t) = max_v p_theta(x_s=v | x^(t))
        for eligible candidates.

        Args:
            logits: Tensor [B, T, V]
            eligible_indices: Tensor [B, K], invalid positions are -1

        Returns:
            confidence_scores: Tensor [B, K]
        """
        probs = torch.softmax(logits, dim=-1)          # [B, T, V]
        max_probs = probs.max(dim=-1).values           # [B, T]

        B, K = eligible_indices.shape
        confidence_scores = torch.zeros(
            (B, K),
            dtype=max_probs.dtype,
            device=logits.device,
        )

        for j in range(B):
            for k in range(K):
                pos = int(eligible_indices[j, k].item())
                if pos < 0:
                    confidence_scores[j, k] = 0.0
                    continue

                confidence_scores[j, k] = max_probs[j, pos]

        return confidence_scores

    def normalize_structural_scores_on_eligible(
        self,
        structural_scores,
        eligible_mask,
    ):
        """
        Normalize structural scores within eligible candidates.

        Args:
            structural_scores: Tensor [B, K]
            eligible_mask: Tensor [B, K] bool

        Returns:
            normalized_scores: Tensor [B, K]
        """
        B, K = structural_scores.shape
        normalized_scores = torch.zeros_like(structural_scores)

        for j in range(B):
            valid_mask = eligible_mask[j]
            valid_count = int(valid_mask.sum().item())

            if valid_count == 0:
                continue

            vals = structural_scores[j][valid_mask]
            vmin = vals.min()
            vmax = vals.max()

            if float(vmax - vmin) < 1e-8:
                normalized_scores[j][valid_mask] = 1.0
            else:
                normalized_scores[j][valid_mask] = (vals - vmin) / (vmax - vmin)

        return normalized_scores

    def compute_commit_priority_minimal(
        self,
        confidence_scores,
        normalized_structural_scores,
        eligible_mask,
        lambda_confidence,
        lambda_structural,
    ):
        """
        Compute minimal commit priority on eligible candidates.

        P(s;t) = lambda_C * confidence + lambda_S * normalized_structural_score

        Args:
            confidence_scores: Tensor [B, K]
            normalized_structural_scores: Tensor [B, K]
            eligible_mask: Tensor [B, K] bool
            lambda_confidence: float
            lambda_structural: float

        Returns:
            priority_scores: Tensor [B, K]
        """
        priority_scores = (
            lambda_confidence * confidence_scores
            + lambda_structural * normalized_structural_scores
        )

        priority_scores = priority_scores.masked_fill(~eligible_mask, float("-inf"))
        return priority_scores
    
    def select_top_commit_candidates(
        self,
        eligible_indices,
        priority_scores,
        top_m,
    ):
        """
        Select Top-M commit candidates from eligible set.

        Args:
            eligible_indices: Tensor [B, K], invalid positions are -1
            priority_scores: Tensor [B, K], invalid positions should already be -inf
            top_m: int

        Returns:
            commit_indices: Tensor [B, K], selected positions kept, others -1
            commit_mask: Tensor [B, K] bool
        """
        B, K = eligible_indices.shape
        commit_mask = torch.zeros_like(eligible_indices, dtype=torch.bool)

        for j in range(B):
            valid_mask = eligible_indices[j] >= 0
            valid_count = int(valid_mask.sum().item())

            if valid_count == 0:
                continue

            m = min(top_m, valid_count)

            topk_idx = torch.topk(priority_scores[j], k=m, dim=-1).indices
            commit_mask[j, topk_idx] = True

        commit_indices = torch.full_like(eligible_indices, fill_value=-1)
        commit_indices[commit_mask] = eligible_indices[commit_mask]

        return commit_indices, commit_mask
    
    def commit_indices_to_mask(
        self,
        commit_indices,
        seq_len,
    ):
        """
        Convert commit_indices [B, K] into a boolean mask [B, T].

        Args:
            commit_indices: Tensor [B, K], invalid positions are -1
            seq_len: int

        Returns:
            commit_transfer_mask: Tensor [B, T] bool
            commit_counts: Tensor [B]
        """
        B, K = commit_indices.shape
        device = commit_indices.device

        commit_transfer_mask = torch.zeros((B, seq_len), dtype=torch.bool, device=device)

        for j in range(B):
            for pos in commit_indices[j].tolist():
                pos = int(pos)
                if pos >= 0:
                    commit_transfer_mask[j, pos] = True

        commit_counts = commit_transfer_mask.sum(dim=-1)
        return commit_transfer_mask, commit_counts
    
    def update_readiness_from_commits(
        self,
        commit_indices,
        mask_index,
        readiness_window,
        readiness_boost,
        readiness_decay,
    ):
        """
        Update readiness_map using newly committed structural positions.

        Args:
            commit_indices: Tensor [B, K], invalid positions are -1
            mask_index: Tensor [B, T], current masked positions
        """
        if self.readiness_map is None:
            raise RuntimeError("readiness_map is not initialized.")

        B, K = commit_indices.shape
        _, T = mask_index.shape

        for j in range(B):
            for pos in commit_indices[j].tolist():
                pos = int(pos)
                if pos < 0:
                    continue

                # propagate to local downstream masked region
                for d in range(1, readiness_window + 1):
                    nxt = pos + d
                    if nxt >= T:
                        break

                    # only unresolved/masked positions receive readiness boost
                    if not bool(mask_index[j, nxt].item()):
                        continue

                    self.readiness_map[j, nxt] += readiness_boost * (readiness_decay ** (d - 1))
    
    def update_nfe_metric(self):
        if self.metrics_state is None:
            return
        self.metrics_state["nfe"] += 1

    def update_eligible_metric(
        self,
        eligible_counts,
    ):
        if self.metrics_state is None:
            return
        self.metrics_state["eligible_total"] += eligible_counts.to(self.metrics_state["eligible_total"].dtype)

    def update_commit_metrics(
        self,
        commit_indices,
        commit_counts,
        transfer_index,
    ):
        if self.metrics_state is None:
            return

        self.metrics_state["struct_commit_total"] += commit_counts.to(
            self.metrics_state["struct_commit_total"].dtype
        )

        total_commit_counts = transfer_index.sum(dim=-1)
        self.metrics_state["total_commit_total"] += total_commit_counts.to(
            self.metrics_state["total_commit_total"].dtype
        )

        B, K = commit_indices.shape
        for j in range(B):
            committed = [int(v) for v in commit_indices[j].tolist() if int(v) >= 0]
            self.metrics_state["commit_trajectory"][j].append(committed)

    def finalize_metrics(self):
        if self.metrics_state is None:
            return {}

        ready_cov_count = self.metrics_state["ready_coverage_count"].clamp_min(1)
        ready_coverage = self.metrics_state["ready_coverage_sum"] / ready_cov_count.float()

        batch_size = self.metrics_state["total_commit_total"].shape[0]

        commit_phase_time_sum = float(self.metrics_state["commit_phase_time_sum"])
        latency_commit_phase_per_sample = commit_phase_time_sum / max(batch_size, 1)

        commit_stage_total_tps = (
            self.metrics_state["total_commit_total"].float()
            / max(latency_commit_phase_per_sample, 1e-12)
        )

        commit_stage_struct_efficiency = (
            self.metrics_state["struct_commit_total"].float()
            / max(latency_commit_phase_per_sample, 1e-12)
        )

        structural_commit_ratio = (
            self.metrics_state["struct_commit_total"].float()
            / self.metrics_state["total_commit_total"].clamp_min(1).float()
        )

        attempt_counts = self.metrics_state["attempt_counts"]
        avg_attempt = attempt_counts.float().mean(dim=-1)
        max_attempt = attempt_counts.max(dim=-1).values

        metrics = {
            "nfe": self.metrics_state["nfe"].tolist(),

            "struct_commit_total": self.metrics_state["struct_commit_total"].tolist(),
            "total_commit_total": self.metrics_state["total_commit_total"].tolist(),
            "eligible_total": self.metrics_state["eligible_total"].tolist(),

            "scheduler_activation_steps": self.metrics_state["scheduler_activation_steps"].tolist(),
            "scheduler_total_steps": self.metrics_state["scheduler_total_steps"].tolist(),
            "scheduler_activation_rate": (
                self.metrics_state["scheduler_activation_steps"].float()
                / self.metrics_state["scheduler_total_steps"].clamp_min(1).float()
            ).tolist(),

            "ready_coverage": ready_coverage.tolist(),

            "avg_attempt": avg_attempt.tolist(),
            "max_attempt": max_attempt.tolist(),

            "commit_trajectory": self.metrics_state["commit_trajectory"],

            "commit_phase_time_sum": commit_phase_time_sum,
            "latency_commit_phase_per_sample": latency_commit_phase_per_sample,

            "commit_stage_struct_efficiency": commit_stage_struct_efficiency.tolist(),
            "structural_commit_ratio": structural_commit_ratio.tolist(),

            "position_first_seen_struct_score": self.metrics_state["position_first_seen_struct_score"].tolist(),
            "position_precommit_max_struct_score": self.metrics_state["position_precommit_max_struct_score"].tolist(),
            "position_commit_step_struct_score": self.metrics_state["position_commit_step_struct_score"].tolist(),
            "position_first_commit_step": self.metrics_state["position_first_commit_step"].tolist(),
        }
        return metrics
    
    def extract_effective_positions(self, commit_steps):
        """
        Flatten commit trajectory and remove consecutive duplicates.
        Empty steps are ignored.
        """
        raw_positions = []
        for step in commit_steps:
            for pos in step:
                if pos >= 0:
                    raw_positions.append(int(pos))
    
        if not raw_positions:
            return []
    
        effective_positions = [raw_positions[0]]
        for pos in raw_positions[1:]:
            if pos != effective_positions[-1]:
                effective_positions.append(pos)
    
        return effective_positions

    def compute_effective_jump_distance(self, commit_steps):
        positions = self.extract_effective_positions(commit_steps)
    
        if len(positions) <= 1:
            return 0.0
    
        jumps = [abs(positions[i] - positions[i - 1]) for i in range(1, len(positions))]
        return sum(jumps) / len(jumps)

    def update_first_commit_traces(
        self,
        transfer_index,
        current_step,
        candidate_indices=None,
        structural_source_scores=None,
    ):
        if self.metrics_state is None:
            return
    
        first_commit_step = self.metrics_state["position_first_commit_step"]
        commit_step_score = self.metrics_state["position_commit_step_struct_score"]
    
        B, T = transfer_index.shape
    
        # 可选：如果这一步某些被 commit 的位置也在 candidate list 里，就记下该步 score
        score_maps = []
        if candidate_indices is not None and structural_source_scores is not None:
            for j in range(B):
                mp = {}
                for k in range(candidate_indices.shape[1]):
                    pos = int(candidate_indices[j, k].item())
                    if pos >= 0:
                        mp[pos] = float(structural_source_scores[j, k].item())
                score_maps.append(mp)
        else:
            score_maps = [None for _ in range(B)]
    
        for j in range(B):
            commit_pos = torch.nonzero(transfer_index[j], as_tuple=False).squeeze(-1)
            for pos_t in commit_pos.tolist():
                pos = int(pos_t)
    
                if int(first_commit_step[j, pos].item()) < 0:
                    first_commit_step[j, pos] = current_step
    
                    mp = score_maps[j]
                    if mp is not None and pos in mp and torch.isnan(commit_step_score[j, pos]):
                        commit_step_score[j, pos] = mp[pos]

    @torch.no_grad()
    def sample(
        self,
        inputs: list[torch.Tensor | list],
        config: ForkAwareMDLMSamplerConfig | None = None,
        **kwargs,
    ) -> BaseSamplerOutput | torch.Tensor:
        """
        Generate text using masked diffusion language modeling.

        Iteratively unmasks tokens over multiple diffusion steps, starting from
        fully masked sequences appended to the input prompts.

        Args:
            inputs: List of input prompts (token tensors or lists of token IDs).
            config: Sampler configuration, or None to use defaults.
            **kwargs: Override specific config parameters.

        Returns:
            BaseSamplerOutput with generated sequences, or raw tensor if return_dict=False.
        """
        self.reset_histories()
        self.reset_scheduler_state()
        self.reset_metrics_state()

        if config is None:
            config = ForkAwareMDLMSamplerConfig()

        # ----- pull args from config, allow kwargs to override -----
        steps = kwargs.get("steps", config.steps)
        max_new_tokens = kwargs.get("max_new_tokens", config.max_new_tokens)
        max_length = kwargs.get("max_length", config.max_length)
        block_size = kwargs.get("block_size", config.block_size)
        temperature = kwargs.get("temperature", config.temperature)
        cfg_scale = kwargs.get("cfg_scale", config.cfg_scale)
        cfg_keep_tokens = kwargs.get("cfg_keep_tokens", config.cfg_keep_tokens)
        remasking = kwargs.get("remasking", config.remasking)
        suppress_tokens = kwargs.get("suppress_tokens", config.suppress_tokens)
        stochastic_transfer = kwargs.get(
            "stochastic_transfer", config.stochastic_transfer
        )
        return_dict = kwargs.get("return_dict", config.return_dict)
        right_shift_logits = kwargs.get("right_shift_logits", config.right_shift_logits)
        begin_suppress_tokens = kwargs.get(
            "begin_suppress_tokens", config.begin_suppress_tokens
        )

        assert 1 <= block_size
        assert 1 <= steps
        mask_id = self.tokenizer.mask_token_id
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id

        # ----- Shape bookkeeping: per-sample prompt lengths and final canvas width -----
        # If right_shift_logits is true and a sequence has length 0, replace that sequence with [bos].
        if right_shift_logits:
            inputs = [
                [bos_id] if isinstance(p, list) and len(p) == 0 else p for p in inputs
            ]

        if isinstance(inputs[0], list):
            inputs = [
                torch.as_tensor(p, dtype=torch.long, device=self.model.device)
                for p in inputs
            ]
        prompt_lens = [p.shape[0] for p in inputs]

        if max_new_tokens:
            max_length = max_new_tokens + max(prompt_lens)
        else:
            max_new_tokens = max_length - max(prompt_lens)

        B = len(inputs)
        T = max_length

        # ----- Initialize canvas with EOS, copy inputs, and append mask tail -----
        x = torch.full((B, T), eos_id, dtype=torch.long, device=self.model.device)
        for i, p in enumerate(inputs):
            x[i, : prompt_lens[i]] = p  # keep original prompt tokens
            x[i, prompt_lens[i] : prompt_lens[i] + max_new_tokens] = (
                mask_id  # append `max_new_tokens` masks to be generated
            )
        attention_mask = torch.zeros((B, T), dtype=torch.long, device=self.model.device)
        for i, pl in enumerate(prompt_lens):
            valid_end = min(pl + max_new_tokens, T)
            attention_mask[i, :valid_end] = 1

        # Tokens that were *given* at the start (non-mask, non-EOS).
        # These will be masked in the unconditional forward pass for CFG.
        # Tokens from `cfg_keep_tokens` should *not* be treated as "given" for CFG
        unmasked_index = (x != mask_id) & attention_mask.bool()
        if not (cfg_keep_tokens is None or len(cfg_keep_tokens) == 0):
            keep_mask = torch.isin(
                x, torch.as_tensor(cfg_keep_tokens, device=self.model.device)
            )
            unmasked_index = unmasked_index & ~keep_mask

        # ----- Block scheduling over the appended mask tail -----
        num_blocks = math.ceil(max_new_tokens / block_size)
        steps = math.ceil(steps / num_blocks)  # per-block step budget
        histories = [x.clone()] if return_dict else None

        if self.readiness_map is None:
            self.init_readiness_map(
                batch_size=x.shape[0],
                seq_len=x.shape[1],
                device=x.device,
            )

        if self.metrics_state is None:
            self.init_metrics_state(
                batch_size=x.shape[0],
                seq_len=x.shape[1],
                device=x.device,
            )

        for b in range(num_blocks):
            # Build a per-sample mask *within this block* (aligned to each prompt's tail)
            block_mask_index = torch.zeros(
                (B, block_size), dtype=torch.bool, device=x.device
            )

            for j in range(B):
                start = prompt_lens[j] + b * block_size
                end = min(start + block_size, prompt_lens[j] + max_new_tokens, T)
                if start < end:
                    width = end - start
                    block_mask_index[j, :width] = (
                        x[j, start:end] == mask_id
                    )  # which positions in this block are still masked

            # Decide how many tokens to reveal per step in this block
            num_transfer_tokens = get_num_transfer_tokens(
                mask_index=block_mask_index,
                steps=steps,
                scheduler=self.scheduler,
                stochastic=stochastic_transfer,
            )

            # Some steps may be skipped if there are no transfers
            effective_steps = num_transfer_tokens.size(1)


            # Cache heavy probe results across steps
            cached_candidate_indices = None
            cached_candidate_scores = None
            cached_i_down_scores = None
            cached_i_branch_scores = None
            cached_structural_indices = None
            cached_structural_scores = None
            cached_structural_source_scores = None

            # ----- Iterative reveal inside the current block -----
            for i in range(effective_steps):
            
                logits, x0, mask_index = self.get_step_state(
                    x=x,
                    attention_mask=attention_mask,
                    temperature=temperature,
                    cfg_scale=cfg_scale,
                    unmasked_index=unmasked_index,
                    suppress_tokens=suppress_tokens,
                    begin_suppress_tokens=begin_suppress_tokens,
                    right_shift_logits=right_shift_logits,
                )
            
                self.update_nfe_metric()
            
                self.update_prediction_history(
                    x0=x0,
                    mask_index=mask_index,
                    maxlen=config.consistency_window,
                )
            
                # ------------------------------------------------------------
                # Heavy structural probe is no longer executed every step.
                # We only refresh it periodically and reuse cached results.
                # ------------------------------------------------------------
                need_probe = (
                    i == 0
                    or cached_structural_indices is None
                    or (i % config.probe_interval == 0)
                )
            
                if need_probe:
                    candidate_indices, candidate_scores = self.preselect_candidates(
                        logits=logits,
                        mask_index=mask_index,
                        prompt_lens=prompt_lens,
                        block_idx=b,
                        block_size=block_size,
                        max_new_tokens=max_new_tokens,
                        top_m=config.probe_top_m,
                        config=config,
                    )
            
            
                    downstream_candidate_indices, downstream_selected_pos = self.select_downstream_candidates(
                        candidate_indices=candidate_indices,
                        candidate_scores=candidate_scores,
                        top_r=config.downstream_top_r,
                    )
                    
                    i_down_scores_small = self.estimate_downstream_influence(
                        x=x,
                        attention_mask=attention_mask,
                        logits=logits,
                        candidate_indices=downstream_candidate_indices,
                        mask_index=mask_index,
                        prompt_lens=prompt_lens,
                        block_idx=b,
                        block_size=block_size,
                        max_new_tokens=max_new_tokens,
                        cfg_scale=cfg_scale,
                        unmasked_index=unmasked_index,
                        suppress_tokens=suppress_tokens,
                        begin_suppress_tokens=begin_suppress_tokens,
                        right_shift_logits=right_shift_logits,
                        divergence_mode=config.probe_divergence,
                        downstream_window=config.downstream_window,
                    )
                    
                    i_down_scores = self.scatter_downstream_scores(
                        small_scores=i_down_scores_small,
                        selected_pos=downstream_selected_pos,
                        full_k=candidate_indices.shape[1],
                    )
            
                    i_branch_scores = self.estimate_branch_sensitivity(
                        x=x,
                        attention_mask=attention_mask,
                        logits=logits,
                        candidate_indices=candidate_indices,
                        mask_index=mask_index,
                        prompt_lens=prompt_lens,
                        block_idx=b,
                        block_size=block_size,
                        max_new_tokens=max_new_tokens,
                        cfg_scale=cfg_scale,
                        unmasked_index=unmasked_index,
                        suppress_tokens=suppress_tokens,
                        begin_suppress_tokens=begin_suppress_tokens,
                        right_shift_logits=right_shift_logits,
                        divergence_mode=config.probe_divergence,
                        top_k_values=config.branch_top_k_values,
                        branch_window=config.downstream_window,   # 或 config.branch_window
                    )
            
                    branch_weights = self.get_branch_plausibility_weight(
                        logits=logits,
                        candidate_indices=candidate_indices,
                    )
            
                    i_branch_scores = i_branch_scores * branch_weights
            
                    structural_source_scores = self.fuse_structural_scores(
                        i_down_scores=i_down_scores,
                        i_branch_scores=i_branch_scores,
                        alpha=config.alpha_struct,
                        beta=config.beta_branch,
                    )

                    self.update_position_score_traces(
                        candidate_indices=candidate_indices,
                        structural_source_scores=structural_source_scores,
                        mask_index=mask_index,
                    )
            
                    structural_indices, structural_scores = self.select_structural_candidates(
                        candidate_indices=candidate_indices,
                        structural_source_scores=structural_source_scores,
                        top_k=config.structural_top_k,
                    )
            
                    # Update cache
                    cached_candidate_indices = candidate_indices
                    cached_candidate_scores = candidate_scores
                    cached_i_down_scores = i_down_scores
                    cached_i_branch_scores = i_branch_scores
                    cached_structural_indices = structural_indices
                    cached_structural_scores = structural_scores
                    cached_structural_source_scores = structural_source_scores
            
                else:
                    # Reuse previous heavy structural probe results
                    candidate_indices = cached_candidate_indices
                    candidate_scores = cached_candidate_scores
                    i_down_scores = cached_i_down_scores
                    i_branch_scores = cached_i_branch_scores
                    structural_indices = cached_structural_indices
                    structural_scores = cached_structural_scores
                    structural_source_scores = cached_structural_source_scores

                self.update_attempt_metrics(candidate_indices=candidate_indices)
            
                # ------------------------------------------------------------
                # Lightweight per-step filtering / commitment still runs every step
                # ------------------------------------------------------------
                entropy_scores = self.compute_candidate_entropy(
                    logits=logits,
                    structural_indices=structural_indices,
                )
            
                consistency_scores = self.compute_consistency_scores(
                    structural_indices=structural_indices,
                    min_history=config.min_history_for_consistency,
                )
            
                eligible_mask = self.get_eligible_candidates_minimal(
                    structural_indices=structural_indices,
                    entropy_scores=entropy_scores,
                    consistency_scores=consistency_scores,
                    entropy_threshold=config.entropy_threshold,
                    consistency_threshold=config.consistency_threshold,
                )
            
                eligible_indices, eligible_counts = self.finalize_eligible_candidates(
                    structural_indices=structural_indices,
                    eligible_mask=eligible_mask,
                )
            
                self.update_eligible_metric(
                    eligible_counts=eligible_counts,
                )
            
                confidence_scores = self.compute_candidate_confidence(
                    logits=logits,
                    eligible_indices=eligible_indices,
                )
            
                normalized_structural_scores = self.normalize_structural_scores_on_eligible(
                    structural_scores=structural_scores,
                    eligible_mask=eligible_mask,
                )
            
                priority_scores = self.compute_commit_priority_minimal(
                    confidence_scores=confidence_scores,
                    normalized_structural_scores=normalized_structural_scores,
                    eligible_mask=eligible_mask,
                    lambda_confidence=config.lambda_confidence,
                    lambda_structural=config.lambda_structural,
                )
            
                commit_indices, commit_mask = self.select_top_commit_candidates(
                    eligible_indices=eligible_indices,
                    priority_scores=priority_scores,
                    top_m=config.commit_top_m,
                )
            
                # Per-position confidence used for fallback selection
                if remasking == "low_confidence":
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                    )  # [B, T]
                elif remasking == "random":
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                else:
                    raise NotImplementedError(remasking)
            
                # Restrict selection window to the current block's tail region
                for j in range(B):
                    x0_p[j, prompt_lens[j] + (b + 1) * block_size :] = -np.inf
            
                # Only allow updates at currently masked positions; keep others fixed
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)

                commit_start = _sync_time()
            
                # Convert [B, K] commit_indices -> [B, T] bool mask
                commit_transfer_mask, commit_counts = self.commit_indices_to_mask(
                    commit_indices=commit_indices,
                    seq_len=x0.shape[1],
                )
            
                # Safety: only keep currently masked positions
                commit_transfer_mask = commit_transfer_mask & mask_index
            
                # Safety: restrict to current block window as well
                for j in range(B):
                    commit_transfer_mask[j, prompt_lens[j] + (b + 1) * block_size :] = False
            
                # Start transfer set from structural commit candidates
                transfer_index = commit_transfer_mask.clone()
            
                # Fill the remaining quota with original fallback selection
                for j in range(B):
                    budget = int(num_transfer_tokens[j, i])
            
                    already_selected = int(transfer_index[j].sum().item())
                    remaining = max(0, budget - already_selected)
            
                    if remaining == 0:
                        continue
            
                    # Exclude already selected positions from fallback
                    fallback_conf = confidence[j].clone()
                    fallback_conf[transfer_index[j]] = -np.inf
            
                    # Only select if there are still valid positions
                    valid_count = int(torch.isfinite(fallback_conf).sum().item())
                    if valid_count == 0:
                        continue
            
                    k_select = min(remaining, valid_count)
                    _, select_index = torch.topk(fallback_conf, k=k_select)
                    transfer_index[j, select_index] = True
            
                # Commit chosen predictions into the canvas
                x[transfer_index] = x0[transfer_index]

                global_step = b * effective_steps + i

                self.update_first_commit_traces(
                    transfer_index=transfer_index,
                    current_step=global_step,
                    candidate_indices=candidate_indices if need_probe else cached_candidate_indices,
                    structural_source_scores=structural_source_scores if need_probe else None,
                )
                        
                self.update_commit_metrics(
                    commit_indices=commit_indices,
                    commit_counts=commit_counts,
                    transfer_index=transfer_index,
                )
            
                self.update_readiness_from_commits(
                    commit_indices=commit_indices,
                    mask_index=mask_index,
                    readiness_window=config.readiness_window,
                    readiness_boost=config.readiness_boost,
                    readiness_decay=config.readiness_decay,
                )

                commit_end = _sync_time()
                commit_time = commit_end - commit_start
                self.metrics_state["commit_phase_time_sum"] += commit_time
            
                if histories is not None:
                    histories.append(x.clone())

        metrics = self.finalize_metrics()

        commit_traj = metrics["commit_trajectory"]

        jump_distances = [
            self.compute_effective_jump_distance(traj)
            for traj in commit_traj
        ]

        print("\n[Average Jump Distance]")

        for i, j in enumerate(jump_distances):
            print(f"sample {i}: {j:.2f}")

        print("mean:", sum(jump_distances) / len(jump_distances))

        # ----- Output format -----
        if not return_dict:
            return x
        else:
            output = BaseSamplerOutput(sequences=x, histories=histories)
            output.metrics = metrics
            return output

    @torch.no_grad()
    def infill(
        self, inputs: list[torch.Tensor | list], config, **kwargs
    ) -> BaseSamplerOutput | torch.Tensor:
        """
        Fill in-place the <|mdm_mask|> tokens contained in `inputs`.
        The whole (padded) sequence is split into block windows of length
        `block_size`; within each window we progressively "unmask" positions
        according to the scheduler and chosen remasking strategy.

        Notes:
        - Right padding uses EOS.
        - CFG masks out *originally known* (non-mask, non-EOS) tokens in the
        unconditional branch, identical to `generate`.
        - Only masked positions are ever updated; non-mask tokens are left intact.
        """
        # ----- pull args from config, allow kwargs to override -----
        steps = kwargs.get("steps", config.steps)
        block_size = kwargs.get("block_size", config.block_size)
        temperature = kwargs.get("temperature", config.temperature)
        cfg_scale = kwargs.get("cfg_scale", config.cfg_scale)
        cfg_keep_tokens = kwargs.get("cfg_keep_tokens", config.cfg_keep_tokens)
        remasking = kwargs.get("remasking", config.remasking)
        suppress_tokens = kwargs.get("suppress_tokens", config.suppress_tokens)
        stochastic_transfer = kwargs.get(
            "stochastic_transfer", config.stochastic_transfer
        )
        return_dict = kwargs.get("return_dict", config.return_dict)
        right_shift_logits = kwargs.get("right_shift_logits", config.right_shift_logits)
        begin_suppress_tokens = kwargs.get(
            "begin_suppress_tokens", config.begin_suppress_tokens
        )

        mask_id = self.tokenizer.mask_token_id
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id

        # ----- Build canvas: right-pad with EOS to the max length in the batch -----
        # If right_shift_logits is true and a sequence has length 0, replace that sequence with [bos].
        if right_shift_logits:
            inputs = [
                [bos_id] if isinstance(p, list) and len(p) == 0 else p for p in inputs
            ]

        if isinstance(inputs[0], list):
            inputs = [
                torch.as_tensor(p, dtype=torch.long, device=self.model.device)
                for p in inputs
            ]

        B = len(inputs)
        seq_lens = [t.shape[0] for t in inputs]
        T = max(seq_lens)

        # Default to a single block spanning the whole sequence
        if block_size is None:
            block_size = T

        assert 1 <= block_size
        assert 1 <= steps

        x = torch.full((B, T), eos_id, dtype=torch.long, device=self.model.device)
        for i, t in enumerate(inputs):
            x[i, : seq_lens[i]] = t

        attention_mask = torch.zeros((B, T), dtype=torch.long, device=self.model.device)
        for i, L in enumerate(seq_lens):
            if L > 0:
                attention_mask[i, :L] = 1

        # Tokens that were *given* at the start (non-mask, non-EOS).
        # These will be masked in the unconditional forward pass for CFG.
        # Tokens from `cfg_keep_tokens` should *not* be treated as "given" for CFG
        unmasked_index = (x != mask_id) & attention_mask.bool()
        if not (cfg_keep_tokens is None or len(cfg_keep_tokens) == 0):
            keep_mask = torch.isin(
                x, torch.as_tensor(cfg_keep_tokens, device=self.model.device)
            )
            unmasked_index = unmasked_index & ~keep_mask

        # ----- Blockwise schedule over the *entire* (padded) sequence -----
        num_blocks = math.ceil(T / block_size)
        steps_per_block = math.ceil(steps / num_blocks)
        histories = [x.clone()] if return_dict else None

        for b in range(num_blocks):
            start = b * block_size
            stop = min(start + block_size, T)

            # Per-sample view of which positions in this block are masks
            block_mask_index = torch.zeros(
                (B, block_size), dtype=torch.bool, device=self.model.device
            )
            widths = []
            for j in range(B):
                # Width limited by sample's true length and sequence end
                width = max(0, min(seq_lens[j], stop) - start)
                widths.append(width)
                if width > 0:
                    block_mask_index[j, :width] = x[j, start : start + width] == mask_id

            # Decide how many tokens to reveal at each step in this block
            num_transfer_tokens = get_num_transfer_tokens(
                mask_index=block_mask_index,
                steps=steps_per_block,
                scheduler=self.scheduler,
                stochastic=stochastic_transfer,
            )

            # Some blocks may have no masks => effective_steps == 0
            effective_steps = num_transfer_tokens.size(1)

            for s in range(effective_steps):
                mask_index_full = x == mask_id

                # ----- Forward pass (+ optional CFG) -----
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[unmasked_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = self.model(
                        x_, attention_mask=attention_mask.repeat(2, 1)
                    ).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = self.model(
                        x, attention_mask=attention_mask
                    ).logits  # Use attention mask here

                if suppress_tokens is not None and len(suppress_tokens) > 0:
                    for token_id in suppress_tokens:
                        logits[:, :, token_id] = -torch.inf

                if right_shift_logits:
                    logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

                # Greedy with optional Gumbel-Max noise
                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)  # [B, T]

                if begin_suppress_tokens is not None and len(begin_suppress_tokens) > 0:
                    for token_id in begin_suppress_tokens:
                        logits[:, :, token_id] = -torch.inf

                # Confidence used for choosing which masks to commit this step
                if remasking == "low_confidence":
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(
                        -1
                    )  # [B, T]
                elif remasking == "random":
                    x0_p = torch.rand((B, T), device=self.model.device)
                else:
                    raise NotImplementedError(remasking)

                # Restrict selection to the *current* block only
                for j in range(B):
                    end_j = start + widths[j]
                    # Outside current block => impossible to select
                    x0_p[j, :start] = -np.inf
                    x0_p[j, end_j:] = -np.inf

                # Only consider currently-masked positions as candidates
                x0 = torch.where(mask_index_full, x0, x)
                confidence = torch.where(mask_index_full, x0_p, -np.inf)

                # Pick exactly num_transfer_tokens[j, s] positions per sample
                transfer_index = torch.zeros_like(x, dtype=torch.bool)
                for j in range(B):
                    k = int(num_transfer_tokens[j, s].item())
                    if k > 0:
                        _, select_idx = torch.topk(confidence[j], k=k)
                        transfer_index[j, select_idx] = True

                # Commit selected predictions into the canvas
                x[transfer_index] = x0[transfer_index]
                if histories is not None:
                    histories.append(x.clone())

        # ----- Output format -----
        if not return_dict:
            return x
        else:
            return BaseSamplerOutput(sequences=x, histories=histories)