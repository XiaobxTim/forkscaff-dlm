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

from collections import deque


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
    probe_top_m: int = 16
    probe_divergence: str = "l1"
    structural_top_k: int = 4
    branch_top_k_values: int = 2   # 最小版只用 top-2
    alpha_struct: float = 1.0      # I_down 权重
    beta_branch: float = 0.5       # I_branch 权重
    consistency_window: int = 4
    min_history_for_consistency: int = 3
    entropy_threshold: float = 4.5
    consistency_threshold: float = 0.75


@dataclass
class ForkAwareMDLMSampler(BaseSampler):
    def __init__(self, model, tokenizer):
        super().__init__(model=model, tokenizer=tokenizer)
        self.pred_history = {}
        self.branch_history = {}

    def reset_histories(self):
        self.pred_history = {}
        self.branch_history = {}

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
    
    def preselect_candidates(
        self,
        logits,
        mask_index,
        prompt_lens,
        block_idx,
        block_size,
        max_new_tokens,
        top_m,
    ):
        """
        在当前 block 内，从 still-masked positions 中选出高熵候选。

        Args:
            logits: [B, T, V]
            mask_index: [B, T] bool
            prompt_lens: list[int]
            block_idx: 当前 block 编号
            block_size: 当前 block 大小
            top_m: 每个 sample 最多保留多少个候选

        Returns:
            candidate_indices: [B, K]
            candidate_scores:  [B, K]
        """
        probs = F.softmax(logits, dim=-1)  # [B, T, V]
        entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1)  # [B, T]

        B, T = entropy.shape
        candidate_mask = torch.zeros_like(mask_index, dtype=torch.bool)

        for j in range(B):
            start = prompt_lens[j] + block_idx * block_size
            end = min(start + block_size, prompt_lens[j] + max_new_tokens, T)
            if start < end:
                candidate_mask[j, start:end] = mask_index[j, start:end]

        masked_entropy = torch.where(
            candidate_mask,
            entropy,
            torch.full_like(entropy, float("-inf")),
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
    ):
        """
        对每个 candidate position s，做一次 top-1 hypothetical commit，
        重新 forward，并计算它对当前 block 内其他 masked positions 的平均分布影响。

        Args:
            x: [B, T]
            attention_mask: [B, T]
            logits: [B, T, V]   当前 base logits
            candidate_indices: [B, K]
            mask_index: [B, T] bool
            prompt_lens: list[int]
            block_idx, block_size, max_new_tokens: 当前 block 信息

        Returns:
            influence_scores: [B, K]
        """
        B, T, V = logits.shape
        K = candidate_indices.shape[1]

        # base probs
        base_probs = F.softmax(logits, dim=-1)  # [B, T, V]

        # 当前 block 内仍 masked 的位置
        block_mask = self.get_block_candidate_mask(
            mask_index=mask_index,
            prompt_lens=prompt_lens,
            block_idx=block_idx,
            block_size=block_size,
            max_new_tokens=max_new_tokens,
        )  # [B, T]

        # 候选位置的 top-1 token，作为 hypothetical commit
        top1_tokens = torch.argmax(logits, dim=-1)  # [B, T]

        influence_scores = torch.zeros(
            (B, K), dtype=logits.dtype, device=logits.device
        )

        for j in range(B):
            for k in range(K):
                s = int(candidate_indices[j, k].item())

                # 如果这个位置不是有效 mask，直接记 0
                if not (0 <= s < T) or not bool(mask_index[j, s].item()):
                    influence_scores[j, k] = 0.0
                    continue

                # 构造 counterfactual x_cf
                x_cf = x.clone()
                x_cf[j, s] = top1_tokens[j, s]

                # 重新 forward
                cf_logits, _, cf_mask_index = self.get_step_state(
                    x=x_cf,
                    attention_mask=attention_mask,
                    temperature=0.0,  # 这里只需要稳定分布，不要噪声
                    cfg_scale=cfg_scale,
                    unmasked_index=unmasked_index,
                    suppress_tokens=suppress_tokens,
                    begin_suppress_tokens=begin_suppress_tokens,
                    right_shift_logits=right_shift_logits,
                )
                cf_probs = F.softmax(cf_logits, dim=-1)  # [B, T, V]

                # 只看 sample j 当前 block 内“其他”仍 masked 的位置
                target_mask = block_mask[j].clone()  # [T]
                target_mask[s] = False  # 不比较自己

                target_indices = torch.nonzero(target_mask, as_tuple=False).squeeze(-1)

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
    ):
        """
        对每个 candidate position s：
        取 top-2 plausible values，分别做两次 hypothetical commit，
        比较两种 commit 对当前 block 内其他 masked positions 的未来分布差异。

        返回:
            branch_scores: [B, K]
        """
        B, T, V = logits.shape
        K = candidate_indices.shape[1]

        if K == 0:
            return torch.empty((B, 0), dtype=logits.dtype, device=logits.device)

        # 当前 block 内仍 masked 的位置
        block_mask = self.get_block_candidate_mask(
            mask_index=mask_index,
            prompt_lens=prompt_lens,
            block_idx=block_idx,
            block_size=block_size,
            max_new_tokens=max_new_tokens,
        )  # [B, T]

        # 每个 candidate 的 top-2 plausible values
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

                # 如果 top-2 恰好一样，直接记 0
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

                # 只看 sample j 当前 block 内“其他”仍 masked 的位置
                target_mask = block_mask[j].clone()
                target_mask[s] = False
                target_indices = torch.nonzero(target_mask, as_tuple=False).squeeze(-1)

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
        eligible if entropy is low enough and consistency is high enough.

        Args:
            structural_indices: Tensor [B, K]
            entropy_scores: Tensor [B, K]
            consistency_scores: Tensor [B, K]
            entropy_threshold: float
            consistency_threshold: float

        Returns:
            eligible_indices: Tensor [B, K]
            eligible_mask: Tensor [B, K] (bool)
        """
        eligible_mask = (
            (entropy_scores <= entropy_threshold)
            & (consistency_scores >= consistency_threshold)
        )

        eligible_indices = torch.full_like(structural_indices, fill_value=-1)
        eligible_indices[eligible_mask] = structural_indices[eligible_mask]

        return eligible_indices, eligible_mask
    
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

                self.update_prediction_history(
                    x0=x0,
                    mask_index=mask_index,
                    maxlen=config.consistency_window,
                )

                candidate_indices, candidate_scores = self.preselect_candidates(
                    logits=logits,
                    mask_index=mask_index,
                    prompt_lens=prompt_lens,
                    block_idx=b,
                    block_size=block_size,
                    max_new_tokens=max_new_tokens,
                    top_m=config.probe_top_m,
                )

                i_down_scores = self.estimate_downstream_influence(
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

                structural_indices, structural_scores = self.select_structural_candidates(
                    candidate_indices=candidate_indices,
                    structural_source_scores=structural_source_scores,
                    top_k=config.structural_top_k,
                )

                entropy_scores = self.compute_candidate_entropy(
                    logits=logits,
                    structural_indices=structural_indices,
                )

                consistency_scores = self.compute_consistency_scores(
                    structural_indices=structural_indices,
                    min_history=config.min_history_for_consistency,
                )

                eligible_indices, eligible_mask = self.get_eligible_candidates_minimal(
                    structural_indices=structural_indices,
                    entropy_scores=entropy_scores,
                    consistency_scores=consistency_scores,
                    entropy_threshold=config.entropy_threshold,
                    consistency_threshold=config.consistency_threshold,
                )


                # Per-position confidence used to pick which masks to commit this step
                if remasking == "low_confidence":
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                    )  # [B, T] confidence of predicted token
                elif remasking == "random":
                    x0_p = torch.rand(
                        (x0.shape[0], x0.shape[1]), device=x0.device
                    )  # random scores
                else:
                    raise NotImplementedError(remasking)

                # Restrict selection window to the *current block's* tail region
                for j in range(B):
                    x0_p[j, prompt_lens[j] + (b + 1) * block_size :] = -np.inf

                # Only allow updates at currently masked positions; keep others fixed
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(
                    mask_index, x0_p, -np.inf
                )  # consider masked positions only

                # Pick exactly `num_transfer_tokens[j, i]` highest-confidence positions per sample
                transfer_index = torch.zeros_like(
                    x0, dtype=torch.bool, device=x0.device
                )
                for j in range(confidence.shape[0]):
                    _, select_index = torch.topk(
                        confidence[j], k=num_transfer_tokens[j, i]
                    )
                    transfer_index[j, select_index] = True

                # Commit chosen predictions into the canvas
                x[transfer_index] = x0[transfer_index]
                if histories is not None:
                    histories.append(x.clone())

        # ----- Output format -----
        if not return_dict:
            return x
        else:
            return BaseSamplerOutput(sequences=x, histories=histories)

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
