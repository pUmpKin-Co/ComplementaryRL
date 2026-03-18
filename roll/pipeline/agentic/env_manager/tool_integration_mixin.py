from typing import TYPE_CHECKING, Any, Dict, Optional

from roll.pipeline.agentic.tools.tool_env_wrapper import tool_wrapper
from roll.utils.logging import get_logger

logger = get_logger()

if TYPE_CHECKING:
    from roll.pipeline.agentic.memory.memory_manager import MemoryManager


class ToolIntegrationMixin:
    """
    Mixin that provides tool integration for environment managers.
    """

    def wrap_env_with_tools(
        self,
        env: Any,
        env_config: Dict,
        memory_manager: Optional["MemoryManager"] = None,
    ) -> Any:
        """
        Wrap the environment with tools if configured.

        Args:
            env: The environment instance to wrap
            env_config: Environment configuration dictionary
            memory_manager: Optional memory manager (required for some tools like search_and_ask)

        Returns:
            The wrapped environment or original environment
        """
        if "tool_wrapper" in env_config:
            logger.info(f"Wrapping environment {env_config.get('env_id', 'unknown')} with tools...")
            env_name = env_config["env_class"]
            if "env_type" in env_config["config"]:
                env_name = f"{env_name}_{env_config['config']['env_type']}"
            env = tool_wrapper(
                env,
                wrapper_args=env_config.tool_wrapper.wrapper_args,
                tool_configs=env_config.tool_wrapper.tool_configs,
                memory_manager=memory_manager,
                env_name=env_name,
            )

        return env
