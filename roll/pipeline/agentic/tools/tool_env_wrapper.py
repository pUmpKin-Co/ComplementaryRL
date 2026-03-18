from typing import TYPE_CHECKING, Any, Dict, List, Optional, SupportsFloat, Tuple

from gem import Env
from gem.tools.tool_env_wrapper import ToolEnvWrapper as GEMToolEnvWrapper
from omegaconf import OmegaConf

from roll.pipeline.agentic.tools.registration import make_tool

if TYPE_CHECKING:
    from roll.pipeline.agentic.memory.memory_manager import MemoryManager


class ToolEnvWrapper(GEMToolEnvWrapper):
    def _overwrite_GEMToolEnvWrapper_reset(self, seed: Optional[int] = None, **kwargs) -> Tuple[str, dict[str, Any]]:
        prev_ep_tool_uses = self.tool_use_counter
        prev_ep_tool_success = self.tool_success_counter
        self.tool_use_counter = 0
        self.tool_success_counter = 0
        obs, info = self.env.reset(seed=seed, **kwargs)
        if obs is None:
            return None, None

        obs_string = f"{obs}"

        if len(self.tools) > 0:
            obs_string += "\nAvailable tools and their instructions:\n"
            for idx, tool in enumerate(self.tools):
                obs_string += f"***Tool {idx + 1}***\nTool Type: {tool.tool_type}\nTool Instruction: {tool.instruction_string()}\n"

        info["goal"] = obs
        info["tool_use_counter"] = self.tool_use_counter
        info["prev_ep_tool_use_counter"] = prev_ep_tool_uses
        info["tool_success_counter"] = self.tool_success_counter
        info["prev_ep_tool_success_counter"] = prev_ep_tool_success
        info["use_tool"] = False  # The initial context is not a tool result

        obs = obs_string
        return obs, info

    def _overwrite_GEMToolEnvWrapper_step(
        self, action: str, verbose: bool = False, **kwargs
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        # try to execute the action with each tool
        tool_parsed = False
        if self.tool_use_counter < self.max_tool_uses:
            for tool in self.tools:
                tool_parsed, tool_execute_error, observation, parsed_action = tool.execute_action(action)
                if tool_parsed and (not tool_execute_error):
                    break

        reward = 0
        if tool_parsed:
            self.tool_use_counter += 1
            if self.tool_use_counter == self.max_tool_uses:
                observation = (
                    f"{observation}\n\nReached the maximum number of tool use. Please do not use any tools anymore."
                )
            reward += self.tool_reward
            terminated, truncated = False, False
            info = {"parsed_action": parsed_action, "tool_type": tool.tool_type}
            if verbose:
                print(f"Tool parsed: {tool.name}, tool use count: {self.tool_use_counter}")
            if not tool_execute_error:
                self.tool_success_counter += 1
                reward += self.tool_success_reward
                if hasattr(tool, "triggered_interactions") and tool.triggered_interactions:
                    info["triggered_interactions"] = tool.triggered_interactions
                if verbose:
                    print(f"Tool executed: {tool.name}, tool use count: {self.tool_use_counter}")
        else:
            observation, reward, terminated, truncated, info = self.env.step(action, **kwargs)

        info["tool_use_counter"] = self.tool_use_counter
        info["tool_success_counter"] = self.tool_success_counter
        info["use_tool"] = tool_parsed
        return observation, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None, **kwargs) -> Tuple[str, dict[str, Any]]:
        # observation, info = super().reset(seed=seed)
        observation, info = self._overwrite_GEMToolEnvWrapper_reset(seed, **kwargs)
        if observation is None:
            return None, None

        metrics = {
            "tool_use_counter": info.pop("tool_use_counter"),
            "tool_success_counter": info.pop("tool_success_counter"),
        }
        metrics_agg_mode = {
            "tool_use_counter": "last",
            "tool_success_counter": "last",
        }
        metrics.update(info.pop("metrics", {}))
        metrics_agg_mode.update(info.pop("metrics_agg_mode", {}))
        info["metrics"] = metrics
        info["metrics_agg_mode"] = metrics_agg_mode
        return observation, info

    def step(
        self,
        action: str,
        verbose: bool = False,
        **kwargs,
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        # observation, reward, terminated, truncated, info = super().step(action, verbose)
        observation, reward, terminated, truncated, info = self._overwrite_GEMToolEnvWrapper_step(
            action, verbose, **kwargs
        )
        metrics = {
            "tool_use_counter": info.pop("tool_use_counter"),
            "tool_success_counter": info.pop("tool_success_counter"),
        }
        metrics_agg_mode = {
            "tool_use_counter": "last",
            "tool_success_counter": "last",
        }
        metrics.update(info.pop("metrics", {}))
        metrics_agg_mode.update(info.pop("metrics_agg_mode", {}))
        info["metrics"] = metrics
        info["metrics_agg_mode"] = metrics_agg_mode
        return observation, reward, terminated, truncated, info


def tool_wrapper(
    env: Env,
    wrapper_args: Dict,
    tool_configs: List[Dict],
    memory_manager: Optional["MemoryManager"] = None,
    env_name: Optional[str] = None,
):
    tools = []
    for tool_config in tool_configs:
        tool_config = (
            tool_config
            if not OmegaConf or not OmegaConf.is_config(tool_config)
            else OmegaConf.to_container(tool_config, resolve=True)
        )
        tool_id = tool_config["tool_id"]
        raw_tool_args = tool_config.get("tool_args", {})
        if OmegaConf and OmegaConf.is_config(raw_tool_args):
            tool_args = OmegaConf.to_container(raw_tool_args, resolve=True)
        elif isinstance(raw_tool_args, dict):
            tool_args = dict(raw_tool_args)
        else:
            tool_args = raw_tool_args

        if tool_id == "knowledge_search_and_ask":
            assert memory_manager is not None, "memory_manager is required for knowledge_search_and_ask tool"
            tool_args["memory_manager"] = memory_manager
            tool_args["env_name"] = env_name

        tools.append(make_tool(tool_id=tool_id, **tool_args))

    if OmegaConf and OmegaConf.is_config(wrapper_args):
        wrapper_args_dict = OmegaConf.to_container(wrapper_args, resolve=True)
    elif isinstance(wrapper_args, dict):
        wrapper_args_dict = dict(wrapper_args)
    else:
        wrapper_args_dict = wrapper_args
    tool_env_wrapper = ToolEnvWrapper(env, tools, **wrapper_args_dict)
    return tool_env_wrapper
