import random
import string
from typing import Any, Optional, Union

from gem import Env
from webshop_minimal import WebAgentTextEnv, init_basedir

init_basedir()  # init DEFAULT_FILE_PATH, hardcoded dataset to small
from webshop_minimal.utils import DEFAULT_FILE_PATH

from roll.pipeline.agentic.env.parse_action_utils import default_parser_action_func
from roll.pipeline.agentic.utils import all_seed


class WebShopEnv(Env, WebAgentTextEnv):
    def __init__(
        self,
        observation_mode: str = "text",
        file_path: str = DEFAULT_FILE_PATH,
        server: Any = None,
        filter_goals: Any = None,
        limit_goals: int = -1,
        num_products: int = None,
        human_goals: bool = False,
        show_attrs: bool = False,
        max_steps: int = 10,
        env_instruction: str = None,
        format_penalty=0.0,
        action_pattern: str = r"<answer>(.*?)</answer>",
        special_token_list: list = ("<|im_start|>", "<|im_end|>"),
        **kwargs: any,
    ) -> None:
        """
        Adapter for WebAgentTextEnv to conform to the BaseLanguageBasedEnv interface.
        """
        self.env_instruction = "Your task is to {task_description}."
        if env_instruction is not None:
            self.env_instruction = env_instruction

        self.observation_mode = observation_mode
        self.file_path = file_path
        self.server = server
        self.filter_goals = filter_goals
        self.limit_goals = limit_goals
        self.num_products = num_products
        self.human_goals = human_goals
        self.show_attrs = show_attrs
        self.render_cache = None
        self.max_steps = max_steps
        self.action_pattern = action_pattern
        self.special_token_list = special_token_list
        self.format_penalty = format_penalty
        # WebAgentTextEnv.__init__ calls self.reset(), so initialize these first.
        self.step_count = 0
        self._scoped_session_id = f"env-{id(self)}"

        WebAgentTextEnv.__init__(
            self,
            observation_mode=self.observation_mode,
            file_path=self.file_path,
            server=self.server,
            filter_goals=self.filter_goals,
            limit_goals=self.limit_goals,
            num_products=self.num_products,
            human_goals=self.human_goals,
            show_attrs=self.show_attrs,
            **kwargs,
        )

    def reset(
        self, seed=None, session: Optional[Union[str, int]] = None, instruction_text: Optional[str] = None
    ) -> any:
        self.step_count = 0
        session_int = None
        session_id = session
        if not hasattr(self, "_scoped_session_id"):
            self._scoped_session_id = f"env-{id(self)}"

        if session is None:
            with all_seed(seed):
                session_int = random.randint(0, len(self.server.weights) - 1)
            session_id = self._scoped_session_id
        elif isinstance(session, int):
            session_int = session
            session_id = self._scoped_session_id

        obs, _ = WebAgentTextEnv.reset(
            self,
            session=session_id,
            instruction_text=instruction_text,
            session_int=session_int,
        )

        task_description = (self.get_instruction_text() + obs).replace("Instruction: ", "").strip()
        instruction = self.env_instruction.format(task_description=task_description)
        self.prepare_render_cache(instruction)
        self.obs_with_actions = self._attach_actions(instruction)

        info = {
            "env_instruction": "",
            "goal": task_description,
        }
        return self.obs_with_actions, info

    def step(self, action):
        metrics_agg_mode = {
            "action_is_effective": "mean",
            "action_is_valid": "mean",
            "success": "last",
            "format_penalty": "mean",
        }

        self.step_count += 1
        action_info = self.parse_action(action)
        if action_info["action"] is None:
            action_desc = f"At turn {self.step_count}, You did not provide a valid action."

            metrics = {
                "action_is_effective": False,
                "action_is_valid": False,
                "success": False,
                "format_penalty": self.format_penalty,
            }
            info = {
                "metrics": metrics,
                "metrics_agg_mode": metrics_agg_mode,
                "action_desc": action_desc,
            }
            info.update(action_info)
            truncated = self.step_count >= self.max_steps
            return (
                action_desc + " Please try again." + "\n" + self.render(),
                self.format_penalty,
                truncated,
                truncated,
                info,
            )

        state, reward, done, info = WebAgentTextEnv.step(self, action_info["action"])
        self.prepare_render_cache(self.observation)
        metrics = {
            "action_is_effective": tuple(self.get_available_actions())
            == ("click[back to search]", "click[< prev]", "click[next >]"),
            "action_is_valid": True,
            "success": done,
            "format_penalty": 0,
        }
        info = {"metrics": metrics, "metrics_agg_mode": metrics_agg_mode}
        info.update(action_info)
        obs_with_actions = self._attach_actions(state)
        terminated, truncated = done, False
        if terminated:
            if not metrics["success"] and self.step_count >= self.max_steps:
                truncated = True
        return obs_with_actions, reward, terminated, truncated, info

    def _attach_actions(self, observation: str) -> str:
        actions = ", ".join(self.get_available_actions())
        return observation + "\n" + "Available actions: " + actions

    def parse_action(self, text):
        return default_parser_action_func(text, self.action_pattern, None, None)

    def render(self, mode=None):
        """
        Render the environment.
        """
        return self.render_cache

    def close(self):
        """
        Close the environment.
        """
        WebAgentTextEnv.close(self)

    def prepare_render_cache(self, observation: str):
        """
        Prepare the render cache for the environment.
        """
        available_actions = self.get_available_actions()
        self.render_cache = observation + "\n" + "Available actions: " + ", ".join(available_actions)

    def get_available_actions(self):
        """
        Parse the available actions in the environment to a list of strings.
        """
        orig_available_actions = WebAgentTextEnv.get_available_actions(self)
        available_actions = []

        if orig_available_actions["has_search_bar"]:
            available_actions.append("search[<content>]")

        for clickable in orig_available_actions["clickables"]:
            if clickable != "search":
                available_actions.append(f"click[{clickable}]")
        # TODO: we may need to purge the case when available_actions == ['click[back to search]', 'click[< prev]', 'click[next >]']
        return available_actions


if __name__ == "__main__":
    env = WebShopEnv()
    print(env.reset())
    while True:
        print(env.observation)
        print(f"Available actions: {env.get_available_actions()}")
        # action = input("Enter action: ")
        action = "<answer>search[men's t-shirts & tanks with needle sleeve, classic fit with color: brown, and fit type]</answer>"
        if action == "q":
            break
        obs, reward, truncated, done, info = env.step(action)
        print(obs, reward, done, info)
    env.close()
