from dataclasses import dataclass

from lm_eval.__main__ import cli_evaluate
from lm_eval.api.registry import register_model

from dllm.core.eval import MDLMEvalConfig, MDLMEvalHarness
from dllm.core.samplers.config_builders import (
    get_named_sampler_config,
    apply_overrides,
)

from dllm.core.samplers import ForkAwareMDLMSampler, ForkAwareMDLMSamplerConfig


@dataclass
class ForkAwareEvalConfig(MDLMEvalConfig):
    """Eval config for fork-aware decoding on top of LLaDA-style models."""
    max_length: int = 4096


@register_model("forkaware")
class ForkAwareEvalHarness(MDLMEvalHarness):
    def __init__(
        self,
        eval_config: ForkAwareEvalConfig | None = None,
        sampler_config: ForkAwareMDLMSamplerConfig | None = None,
        sampler_cls: type[ForkAwareMDLMSampler] = ForkAwareMDLMSampler,
        **kwargs,
    ):
        eval_config = eval_config or ForkAwareEvalConfig()

        if sampler_config is None:
            config_name = kwargs.pop("config", "forkaware_default")
            sampler_config = get_named_sampler_config(config_name)
            sampler_config = apply_overrides(sampler_config, kwargs)

        super().__init__(
            eval_config=eval_config,
            sampler_config=sampler_config,
            sampler_cls=sampler_cls,
            **kwargs,
        )


if __name__ == "__main__":
    cli_evaluate()