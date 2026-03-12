# dllm/core/samplers/config_builders.py
from dataclasses import fields

from dllm.core.samplers import MDLMSamplerConfig
from dllm.core.samplers import ForkAwareMDLMSamplerConfig   # 改成你的真实路径


def _cast_value(raw_value: str, current_value):
    if raw_value == "None":
        return None
    if isinstance(current_value, bool):
        return raw_value.lower() in ("1", "true", "yes", "y")
    if isinstance(current_value, int):
        return int(raw_value)
    if isinstance(current_value, float):
        return float(raw_value)
    if isinstance(current_value, list):
        if raw_value.strip() == "":
            return []
        return [int(x) for x in raw_value.split(",")]
    return raw_value


def apply_overrides(cfg, overrides: dict):
    for k, v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, _cast_value(v, getattr(cfg, k)))
    return cfg


def get_named_sampler_config(config_name: str):
    if config_name == "baseline_default":
        return MDLMSamplerConfig()

    if config_name == "forkaware_default":
        return ForkAwareMDLMSamplerConfig()

    if config_name == "forkaware_fast":
        return ForkAwareMDLMSamplerConfig(
            block_size=32,
            steps=128,
            probe_top_m=8,
            structural_top_k=2,
            consistency_window=2,
            min_history_for_consistency=2,
            readiness_window=2,
            enable_ready_scheduler=True,
        )

    raise ValueError(f"Unknown config: {config_name}")