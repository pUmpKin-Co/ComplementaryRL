from typing import Dict, List

import gem
from transformers import PreTrainedTokenizer

from roll.distributed.scheduler.generate_scheduler import RequestScheduler
from roll.pipeline.agentic.agentic_config import LLMProxyConfig
from roll.pipeline.agentic.llm_proxy.base_llm_proxy import (LLM_PROXY_REGISTRY,
                                                            BaseLLMProxy,
                                                            register_llm_proxy)
from roll.pipeline.agentic.llm_proxy.openai_proxy import OpenAIProxy
from roll.pipeline.agentic.llm_proxy.policy_proxy import PolicyProxy
from roll.pipeline.agentic.llm_proxy.random_proxy import RandomProxy


def create_llm_proxy(
        generate_scheduler: RequestScheduler,
        llm_proxy_config: LLMProxyConfig,
        tokenizer: PreTrainedTokenizer,
        env: gem.Env) -> BaseLLMProxy:
    proxy_type = llm_proxy_config.proxy_type
    if proxy_type in LLM_PROXY_REGISTRY:
        cls = LLM_PROXY_REGISTRY[proxy_type]
        return cls(generate_scheduler, llm_proxy_config, tokenizer, env)
    else:
        raise ValueError(f"Unknown proxy type: {proxy_type}")

