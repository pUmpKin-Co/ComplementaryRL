import copy
import hashlib
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import ray
from gem import Env

from roll.datasets.global_dataset import GlobalDataset, GlobalDatasetManager
from roll.pipeline.agentic.env.swe_env.util.define import action
from roll.pipeline.agentic.env.swe_env.util.define.action import Action
from roll.pipeline.agentic.env.swe_env.util.define.observation import Observation
from roll.pipeline.agentic.env.swe_env.util.repo_env import RepoClient
from roll.pipeline.agentic.env.swe_env.utils import (
    Colors,
    MultiprocessSafeLogger,
    _lazy_load_jsonl_lines_spec_idx,
    pretty_print,
    write_data_json,
)
from roll.utils.constants import RAY_NAMESPACE

DEBUG = True
SAVE_TRAJ = False


class SWEEnv(Env, gym.Env):
    def __init__(
        self,
        render_mode: str = "text",
        max_steps: int = 50,
        max_reset_retry_times: int = 20,
        format_penalty=0.0,
        mode: str = "train",  # train, val, spec-xx
        data_path: str = "data/part_0.jsonl",
        train_idx_range: Tuple[int, int] = (0, 4577),  # 训练集任务ID范围
        val_idx_range: Tuple[int, int] = (0, 128),  # 验证集任务ID范围
        tools: list[str] = [
            "swe_env/util/tools/search.py",
            "swe_env/util/tools/file_editor.py",
            "swe_env/util/tools/execute_bash.py",
            "swe_env/util/tools/finish.py",
        ],
        action_pattern="^<answer>(.*?)</answer>$",
        special_token_list=("<think>", "</think>", "<answer>", "</answer>", "<|im_start|>", "<|im_end|>"),
        swe_rex_host="https://your-swe-rex-endpoint.com/docker",
        traj_dir: str = "./traj/trainset/",
        swe_requirment_dir: str = "/mnt/dataset/swe/requirements",
        base_dir: str = "./logs",
        max_execute_time: float = 300.0,
        max_execute_retry: int = 10,
        timeout: int = 180,
        max_env_time: float = 60 * 60,
        base_agent: str = "swe",
        sanbox_mode: str = "http",
        golden_patch: bool = False,
        user_id=None,
        experiment_id=None,
        xrl_authorization="t-qx3yxhe61uxxx",
        xrl_cluster="nt-a",
        clear_time=60 * 60,
        **kwargs,
    ):
        self.sanbox_mode = sanbox_mode
        self.action_pattern = action_pattern
        self.special_token_list = special_token_list
        self.format_penalty = format_penalty
        self.base_dir = base_dir
        # print(f"*************** SWEEnv的参数 ***************")
        # print(f'swe_rex_host: {swe_rex_host}')
        # print(f'base_dir: {base_dir}')
        # print(f'max_execute_time: {max_execute_time}')
        # print(f'max_execute_retry: {max_execute_retry}')
        # print(f'timeout: {timeout}')
        # print(f"*************** SWEEnv的参数 ***************")
        # rock参数
        self.user_id = user_id
        self.experiment_id = experiment_id
        self.xrl_authorization = xrl_authorization
        self.xrl_cluster = xrl_cluster
        self.clear_time = clear_time

        # 环境信息(不变)
        self.swe_rex_host = swe_rex_host
        self.mode = mode
        self.train_idx_range = train_idx_range
        self.val_idx_range = val_idx_range
        self.max_execute_time = max_execute_time if max_execute_time else 300.0
        self.max_execute_retry = max_execute_retry if max_execute_retry else 10
        self.timeout = timeout if timeout else 720
        self.base_agent = base_agent if base_agent else "swe"
        self.data_path = data_path

        # 基本参数(不变)
        self.max_reset_retry_times = max_reset_retry_times
        self.max_steps = max_steps
        current_file_path = Path(__file__).resolve()
        self.tools = [f"{current_file_path.parent.parent}/{data_path}" for data_path in tools]
        self.traj_dir = traj_dir
        os.environ["SWE_REQUIREMENT_DIR"] = swe_requirment_dir

        # 当前参数(会更新)
        self.global_step = 0
        self.retry_time = 0
        self.turn_count = 0
        self.task_idx = None
        self.history = []
        self.data_line = {}
        self.container_name = None
        self.sandbox_id = None
        self.problem_statement = None
        self.issue = None
        self.metrics = {}
        self.terminate = False
        self.truncated = False
        self.env_timeout = False
        self.env_failed = False
        self.reward = 0
        self.action_is_valid_lst = []  # metric
        self.action_is_effective_lst = []  # metric
        self.current_step = -1
        self.reach_max_length = False
        self.unittest_output = ""
        self.is_closed = False
        self.reach_max_turn = False
        self.if_sandbox_failed = 0
        self.logger = None  # 初始化 logger 为 None
        self.golden_patch = golden_patch  # 是否apply golden
        if self.golden_patch:
            print("self.golden_patch: True")
        else:
            print("self.golden_patch: False")
        # 时间参数
        self.traj_reset_time = 0
        self.traj_step_time = 0
        self.traj_reward_time = 0
        self.traj_env_time = 0
        self.traj_rollout_time = 0
        self.time_start = time.time()
        self.max_env_time = max_env_time  # 单环境最长rollout40min, 超时则return mask。

        # 数据
        if "part_" in self.data_path:
            dataset_name = self.data_path.replace("/part_0.jsonl", f"")
        else:
            dataset_name = self.data_path

        # Convert train/val mode to sample/traversal for GlobalDataset
        global_dataset_mode = "sample" if self.mode == "train" else "traversal"

        # 检查是否使用 LOCAL 模式（为每个 task 创建独立的 Actor）
        use_local_mode = os.environ.get("LOCAL", "").lower() in ("1", "true", "yes")

        if use_local_mode:
            # 【本地评测使用】LOCAL 模式：使用 task_idx 作为唯一标识（eval 场景每个 task 需要独立的 filter）
            if self.mode == "val" and self.val_idx_range:
                task_idx = self.val_idx_range[0]
                actor_name_suffix = f"_task{task_idx}"
            elif self.mode == "train" and self.train_idx_range:
                if self.train_idx_range[0] == self.train_idx_range[1]:
                    task_idx = self.train_idx_range[0]
                    actor_name_suffix = f"_task{task_idx}"
                else:
                    actor_name_suffix = f"_pid{os.getpid()}"
            else:
                actor_name_suffix = f"_pid{os.getpid()}"
            # LOCAL 模式：对 dataset_name 进行哈希处理，避免路径过长
            dataset_name_hash = hashlib.md5(dataset_name.encode()).hexdigest()[:8]
            actor_name = f"{self.mode}_{dataset_name_hash}{actor_name_suffix}"
            manager_name = f"{self.mode}_dataset_manager{actor_name_suffix}"
        else:
            # 【星云上使用】直接使用 dataset_name（不进行哈希）
            actor_name = f"{self.mode}_{dataset_name}"
            manager_name = f"{self.mode}_dataset_manager"
        print(f"actor_name: {actor_name}, manager_name: {manager_name}, use_local_mode: {use_local_mode}")
        self.dataset = GlobalDataset.options(name=actor_name, get_if_exists=True, namespace=RAY_NAMESPACE).remote(
            dataset_name=dataset_name, mode=global_dataset_mode
        )
        self.dataset_manager = GlobalDatasetManager.options(
            name=manager_name, get_if_exists=True, namespace=RAY_NAMESPACE
        ).remote()
        ray.get(self.dataset_manager.register.remote(dataset_name=dataset_name, dataset_ref=self.dataset))
        data_ranges = self.train_idx_range
        if self.mode == "val":
            data_ranges = self.val_idx_range
        ray.get(
            self.dataset.filter.remote(
                filter_name="filter_idx_range", function=lambda x: data_ranges[0] <= int(x["idx"]) <= data_ranges[1]
            )
        )

        print(
            f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}[GSTEP{self.global_step}][base_dir]{self.base_dir}, [sandbox_mode]{self.sanbox_mode}, [max_env_time]{max_env_time}'
        )

    def get_task_suffix(self) -> Any:
        problem_statement, issue = self.get_instruction()
        return problem_statement

    def get_task_idx_and_data(self, seed):
        data_item: Optional[Dict] = ray.get(self.dataset.get_data_item.remote(seed=seed))
        # print(f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}[GSTEP{self.global_step}]#### data_item: {data_item}')
        if data_item is None:
            return None, None
        idx = data_item["idx"]
        if self.mode == "val":
            print(
                f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}[GSTEP{self.global_step}][GEN_IDX]mode: {self.mode}, self.val_idx_range: {self.val_idx_range}, seed: {seed}, cur_task_idx: {idx}"
            )
        elif self.mode == "train":
            print(
                f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}[GSTEP{self.global_step}][GEN_IDX]mode: {self.mode}, self.train_idx_range: {self.train_idx_range}, seed: {seed}, cur_task_idx: {idx}"
            )
        else:
            print(
                f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}[GSTEP{self.global_step}][GEN_IDX][Attention] mode: {self.mode}, val_idx_range: {self.val_idx_range}, train_idx_range: {self.train_idx_range}, seed: {seed}, cur_task_idx: {idx}"
            )
        return idx, data_item

    def gen_sandbox_id(self, data_line):
        if "docker_image" in data_line:
            sandbox_id = f"{self.task_idx}_{data_line['docker_image']}"
        elif "docker_image" not in data_line:
            sandbox_id = f"{self.task_idx}"
        return sandbox_id

    def get_instruction(self):
        problem_statement = self.data_line["problem_statement"]
        try:
            issue = re.search(r"\[ISSUE\](.*)\[/ISSUE\]", problem_statement, re.DOTALL).group(1)  # r2e-gym trainset
        except:
            issue = problem_statement  # swe-bench-verified
        return problem_statement, issue

    def reset(self, seed=None, step=None):
        st = time.time()
        self.time_start = st
        self.global_step = step
        self.clean_record()
        failed_commands = []

        # gen task_idx and load data_line
        task_idx, self.data_line = self.get_task_idx_and_data(seed)
        if self.data_line is None:
            print(
                f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}[GSTEP{self.global_step}][ATTENTION][ENV][RESET][ERROR]reset failed, data_line is None.(seed: {seed}, mode: {self.mode})"
            )
            return None, {}
        self.docker_image = self.data_line.get("docker_image", None)
        self.data_line["swe_rex_host"] = self.swe_rex_host
        self.task_idx = task_idx

        try:
            self.project_path = json.loads(self.data_line["image_info"])["project_path"]
        except Exception as e:
            self.project_path = "/testbed"

        # init logger - 只在第一次创建，后续 reset 只更新文件路径
        time_str = time.strftime("%m%d%H%M%S", time.localtime())
        log_path = os.path.join(
            self.base_dir,
            f"log/step{step}/env",
            f"{self.mode}_idx{self.task_idx}_seed{seed}-{time_str}-{time.time_ns()}.log",
        )
        if self.logger is None:
            self.logger = MultiprocessSafeLogger(path=log_path)
        else:
            self.logger.update_log_path(path=log_path)
        self.logger.info(f"start reset, task_idx: {self.task_idx}, golden_patch: {self.golden_patch}")

        # init repo_env
        self.repo_env = RepoClient(
            logger=self.logger,
            max_execute_time=self.max_execute_time,
            max_execute_retry=self.max_execute_retry,
            timeout=self.timeout,
            max_env_time=self.max_env_time,
            sanbox_mode=self.sanbox_mode,
            swe_rex_host=self.swe_rex_host,
            golden_patch=self.golden_patch,
            user_id=self.user_id,
            experiment_id=self.experiment_id,
            xrl_authorization=self.xrl_authorization,
            xrl_cluster=self.xrl_cluster,
            clear_time=self.clear_time,
            task_idx=self.task_idx,
        )

        self.logger.info(
            f"\n\n*************** 环境初始化 start reset, task_idx: {self.task_idx}, seed: {seed}, global_step: {self.global_step}, docker_image: {self.docker_image}***************"
        )
        self.logger.info(
            f"\n[ENV][RESET]start reset, task_idx: {self.task_idx}, global_step: {self.global_step}, mode: {self.mode}"
        )

        print(
            f'\n{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}[GSTEP{self.global_step}][ENV][RESET]🟢 starting reset.... retry_times: {self.retry_times}/{self.max_reset_retry_times}, used_time: {round(time.time() - st, 4)}/{self.max_env_time}, task_idx: {self.task_idx}, mode: {self.mode}'
        )
        # init docker_runtime
        reset_info, failed_commands = self.repo_env.reset(
            self.data_line,
            max_execute_time=60 * 30,
            max_execute_retry=10,
            timeout=60 * 15,
        )  # Returns RunStatus object with: state, sandbox_id, retry_times, error_message, session_name, host_ip
        # Access RunStatus attributes directly (it's a Pydantic model, not a dict)

        # setup_env_result is determined by setup_env() call, not from reset_info
        self.setup_env_result = "ERROR" if reset_info.state == "error" else "SUCCESS"
        self.container_name = reset_info.sandbox_id if hasattr(reset_info, "sandbox_id") else None  # 可以去掉
        self.sandbox_id = reset_info.sandbox_id if hasattr(reset_info, "sandbox_id") else None
        self.retry_times += reset_info.retry_times if hasattr(reset_info, "retry_times") else 1
        self.reset_error_message = reset_info.retry_times if hasattr(reset_info, "error_message") else ""

        if time.time() - self.time_start > self.max_env_time:
            self.env_timeout = True
            self.env_failed = True
            print(
                f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}[ENV_RESET][GSTEP{self.global_step}]'
                f"❌ reset timeout, task_idx: {self.task_idx}, docker_image: {self.docker_image}, sandbox_id: {self.sandbox_id}, failed_commands: {failed_commands}"
            )
            self.logger.info(
                f"[ENV_RESET][GSTEP{self.global_step}]"
                f"❌ reset timeout, task_idx: {self.task_idx}, docker_image: {self.docker_image}, sandbox_id: {self.sandbox_id}, failed_commands: {failed_commands}"
            )
        elif reset_info.state == "success" or self.sandbox_id:
            self.env_failed = False
            print(
                f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}[ENV_RESET][GSTEP{self.global_step}]'
                f"✅ reset success, task_idx: {self.task_idx}, docker_image: {self.docker_image}, sandbox_id: {self.sandbox_id}, failed_commands: {failed_commands}"
            )
            self.logger.info(
                f"[ENV_RESET][GSTEP{self.global_step}]"
                f"✅ reset success, task_idx: {self.task_idx}, docker_image: {self.docker_image}, sandbox_id: {self.sandbox_id}, failed_commands: {failed_commands}"
            )
        else:
            self.env_failed = True
            print(
                f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}[ENV_RESET][GSTEP{self.global_step}]'
                f"❌ reset failed, retry_times: {self.retry_times}/{self.max_reset_retry_times}, used_time: {round(time.time() - st, 4)}/{self.max_env_time}, "
                f"task_idx: {self.task_idx}, docker_image: {self.docker_image}, sandbox_id: {self.sandbox_id}, reset_info: {reset_info}, failed_commands: {failed_commands}"
            )
            self.logger.info(
                f"[ENV_RESET][GSTEP{self.global_step}]"
                f"❌ reset failed, retry_times: {self.retry_times}/{self.max_reset_retry_times}, used_time: {round(time.time() - st, 4)}/{self.max_env_time}, "
                f"task_idx: {self.task_idx}, docker_image: {self.docker_image}, sandbox_id: {self.sandbox_id}, reset_info: {reset_info}, failed_commands: {failed_commands}"
            )

        self.traj_reset_time = round(time.time() - st, 4)
        # 环境初始化失败
        if self.env_timeout or self.env_failed:
            return "reset failed", {
                "suffix": self.project_path,
                "metrics": {
                    "task_idx": self.task_idx,
                    "traj_reset_time": time.time() - self.time_start,
                    "env_timeout": True,
                    "env_failed": True,
                    "failed_commands": failed_commands,
                },
                "messages": [],
                "prompt_ids": [],
                "response_ids": [],
                "reward": 0.0,
                "penalty": 0.0,
                "llm_response": "",
            }

        # add tools to repo_env
        self.data_line["container_name"] = self.container_name
        self.repo_env.add_commands(self.tools)
        self.problem_statement, self.issue = self.get_instruction()
        self.traj_reset_time = round(time.time() - st, 4)
        # add history
        self.history.append({"role": "user", "content": self.problem_statement})
        self.allowed_cmds = self.repo_env.get_available_cmds()

        print(
            f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}[ENV_RESET][GSTEP{self.global_step}]'
            f"✅ successfully reset, task_idx: {self.task_idx}, retry_times: {self.retry_times}, used_time: {self.traj_reset_time}s. sandbox_id: {self.sandbox_id}"
        )
        self.logger.info(
            f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}[ENV_RESET][GSTEP{self.global_step}]'
            f"✅ successfully reset, task_idx: {self.task_idx}, retry_times: {self.retry_times}, used_time: {self.traj_reset_time}s. sandbox_id: {self.sandbox_id}"
        )
        return self.problem_statement, {
            "suffix": self.project_path,
            "task_idx": self.task_idx,
            "traj_reset_time": self.traj_reset_time,
            "metrics": {
                "task_idx": self.task_idx,
                "traj_reset_time": time.time() - self.time_start,
                "env_timeout": False,
                "env_failed": False,
                "failed_commands": failed_commands,
            },
            "messages": [],
            "prompt_ids": [],
            "response_ids": [],
            "reward": 0.0,
            "penalty": 0.0,
            "llm_response": "",
        }

    def step(self, action: str):
        """
        @input:
            action: <answer>Right</answer>
        @output:
            [obs] At turn 1, you moved Down, which is effective.
            [reward] 0.0
            [terminate] False
            [truncated] False
            [info] {'suffix': 'Here is the current state of the FrozenLake:\n____\n_OP_\n___O\nGO__\n', 'metrics': {'action_is_effective': True, 'action_is_valid': True, 'success': False, 'format_penalty': 0.0}, 'action': 1, 'action_content': 'Down', 'think_content': ''}
        """
        # step默认参数
        st = time.time()
        obs = ""
        model_response = copy.deepcopy(action)
        info = {"suffix": "", "metrics": ""}
        bash_output, exit_code, execute_time = "", "", 0
        action_is_valid, action_is_effective = False, False

        # 确保 logger 已初始化
        if self.logger is None:
            time_str = time.strftime("%m%d%H%M%S", time.localtime())
            log_path = os.path.join(
                self.base_dir,
                f"log/step{self.global_step}/env",
                f"{self.mode}_idx{self.task_idx}_seed{getattr(self, 'current_seed', 0)}-{time_str}-{time.time_ns()}.log",
            )
            self.logger = MultiprocessSafeLogger(path=log_path)

        print(
            f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}[ENV_STEP][GSTEP{self.global_step}]'
            f"🟢 mode: {self.mode}, task_idx: {self.task_idx}, current_turn: {self.turn_count}/{self.max_steps}, sandbox_id: {self.sandbox_id}, docker_image: {self.docker_image}, model_input: {[model_response]}"
        )
        self.logger.info(
            f"[ENV_STEP][GSTEP{self.global_step}]"
            f"🟢 mode: {self.mode}, task_idx: {self.task_idx}, current_turn: {self.turn_count}/{self.max_steps}, sandbox_id: {self.sandbox_id}, docker_image: {self.docker_image}, model_input: {[model_response]}"
        )

        # update history/turn_count/if_sandbox_failed
        self.turn_count += 1
        self.if_sandbox_failed = self.repo_env.get_status("sandbox_failed_times")
        self.history.append({"role": "assistant", "content": f"{model_response}"})

        # 环境初始化失败
        if self.env_failed or not self.sandbox_id or self.env_timeout:
            self.truncated = True
            self.metrics = {
                "env_timeout": self.env_timeout,
                "env_failed": self.env_failed,
                "reach_max_length": self.reach_max_length,
                "reach_max_turn": self.reach_max_turn,
                "if_sandbox_failed": self.if_sandbox_failed,
                "reward": self.reward,
                "success": self.terminate,
                "truncated": self.truncated,
                "format_penalty": self.format_penalty,
                "action_is_valid": (
                    round(sum(self.action_is_valid_lst) / len(self.action_is_valid_lst), 4)
                    if self.action_is_valid_lst
                    else 0.0
                ),
                "action_is_effective": (
                    round(sum(self.action_is_effective_lst) / len(self.action_is_effective_lst), 4)
                    if self.action_is_effective_lst
                    else 0.0
                ),
                "turn_count": int(self.turn_count),
                "retry_times": int(self.retry_times),
                "traj_reset_time": round(self.traj_reset_time, 4),
                "traj_step_time": round(self.traj_step_time + round(time.time() - st, 4), 4),
                "traj_reward_time": round(self.traj_reward_time, 4),
                "traj_env_time": round(self.traj_reset_time + self.traj_step_time + self.traj_reward_time, 4),
                "traj_rollout_time": round(time.time() - self.time_start, 4),
                "task_idx": int(self.task_idx),  # for rollout_log
            }
            info["metrics"] = self.metrics
            if self.env_failed or not self.sandbox_id:
                obs = f"ERROR: Start container failed, the container name is {self.container_name}"
                print(
                    f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}[ENV_STEP][GSTEP{self.global_step}]❌[容器初始化失败] task_idx: {self.task_idx}, current_turn: {self.turn_count}/{self.max_steps}, sandbox_id: {self.sandbox_id}, docker_image: {self.docker_image}, retry_times: {self.retry_times}'
                )
                self.logger.info(
                    f"[ENV][STEP][容器初始化失败] task_idx: {self.task_idx}, current_turn: {self.turn_count}/{self.max_steps}, sandbox_id: {self.sandbox_id}, docker_image: {self.docker_image}, retry_times: {self.retry_times}, obs: {obs}, info: {info}"
                )
            elif self.env_timeout:
                obs = f"ERROR: Environment timeout, used time: {round(time.time() - self.time_start, 4)}s (>{self.max_env_time/60}min)"
                print(
                    f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}[ENV_STEP][GSTEP{self.global_step}]❌[ENV_TIMEOUT]交互时间超过{(self.max_env_time)/60}min (task_idx: {self.task_idx}, current_turn: {self.turn_count}/{self.max_steps}, sandbox_id: {self.sandbox_id}, docker_image: {self.docker_image}, retry_times: {self.retry_times})'
                )
                self.logger.info(
                    f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}[ENV_STEP][GSTEP{self.global_step}]❌[ENV_TIMEOUT]交互时间超过{(self.max_env_time)/60}min (task_idx: {self.task_idx}, current_turn: {self.turn_count}/{self.max_steps}, sandbox_id: {self.sandbox_id}, docker_image: {self.docker_image}, retry_times: {self.retry_times})'
                )
            return obs, self.reward, self.terminate, self.truncated, info

        # split think content
        if "</think>" in action:
            # action = action.split("</think>")[-1].strip()
            actionlst = action.split("</think>")
            if len(actionlst) > 2:
                print(f"⚠️ 这里有多个think标签: {[action]}")
            action = actionlst[1].strip()

        # parse action from response & format exam
        action_info = self.parse_action(action)
        format_info = self.format_exam(action_info)  # error, error_msg
        self.logger.info(
            f"[ENV][STEP][解析Action]\n"
            f'****** think_content ******\n{action_info["think_content"]}\n\n'
            f'****** action_content ******\n{action_info["action_content"]}\n\n'
        )
        self.logger.info(f"[ENV][STEP][检查格式]{format_info}")
        # if format_info["error"]:
        # print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}[GSTEP{self.global_step}][ENV][STEP][format_error][action]{[action_info]}\n[ENV][STEP][检查格式]{format_info}")

        # run action and get obs
        # 任务主动完成：finish, submit
        if (
            "finish" in action_info["action"].function_name.lower()
            or "submit" in action_info["action"].function_name.lower()
        ):
            self.terminate = True
            action_is_valid, action_is_effective = True, True
            self.logger.info(f"[ENV][STEP][terminate]主动结束")
            obs = "<<< Finished >>>"
            print(
                f"[ENV_STEP][GSTEP{self.global_step}]✅[terminate]主动结束, task_idx: {self.task_idx}, current_turn: {self.turn_count}/{self.max_steps}, sandbox_id: {self.sandbox_id}, docker_image: {self.docker_image}, retry_times: {self.retry_times}"
            )
            self.logger.info(
                f"[ENV_STEP][GSTEP{self.global_step}]✅[terminate]主动结束, task_idx: {self.task_idx}, current_turn: {self.turn_count}/{self.max_steps}, sandbox_id: {self.sandbox_id}, docker_image: {self.docker_image}, retry_times: {self.retry_times}"
            )
        elif not format_info["error"]:
            remain_time = self.max_env_time - (time.time() - self.time_start)
            max_execute_time = min(remain_time, 60 * 15)
            print(
                f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}[ENV_STEP][GSTEP{self.global_step}]'
                f'🟢 starting run action ... mode: {self.mode}, task_idx: {self.task_idx}, current_turn: {self.turn_count}/{self.max_steps}, sandbox_id: {self.sandbox_id}, action: {[action_info["action"]]}'
            )
            bash_output, exit_code, execute_time = self.repo_env.run_action(
                action_info["action"],
                timeout=60 * 15,
                base_agent=self.base_agent,
                max_execute_time=max_execute_time,
            )
            if str(exit_code) == "-200":
                self.env_failed = True
                print(
                    f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}[ENV_STEP][GSTEP{self.global_step}]❌[ENV_FAILED] run action failed because sandbox is not alive, used_time: {(time.time() - self.time_start)/60}min, task_idx: {self.task_idx}, current_turn: {self.turn_count}/{self.max_steps}, sandbox_id: {self.sandbox_id}, sandbox_failed_times: {self.if_sandbox_failed}'
                )
                self.logger.info(
                    f"[ENV_STEP][GSTEP{self.global_step}]❌[ENV_FAILED] run action failed because sandbox is not alive, used_time: {(time.time() - self.time_start)/60}min, task_idx: {self.task_idx}, current_turn: {self.turn_count}/{self.max_steps}, sandbox_id: {self.sandbox_id}, sandbox_failed_times: {self.if_sandbox_failed}"
                )
            obs = str(Observation(bash_output, exit_code, action_info["action"]))
        else:
            obs = format_info.get("error_msg", "")
            print(
                f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}[ENV_STEP][GSTEP{self.global_step}]'
                f"❌ format error, sandbox_id: {self.sandbox_id}, obs: {obs}"
            )

        self.if_sandbox_failed = self.repo_env.get_status("sandbox_failed_times")
        if self.if_sandbox_failed > 20:
            print(
                f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}[GSTEP{self.global_step}]❌[ENV][STEP]注意：sandbox_failed_times: {self.if_sandbox_failed} > 20, task_idx: {self.task_idx}, current_turn: {self.turn_count}/{self.max_steps}, docker_image: {self.docker_image}, sandbox_id: {self.sandbox_id}."
            )
            self.logger.info(
                f"[ENV][STEP]注意：sandbox_failed_times: {self.if_sandbox_failed} > 10, task_idx: {self.task_idx}, current_turn: {self.turn_count}/{self.max_steps}, sandbox_id: {self.sandbox_id}, docker_image: {self.docker_image}, sandbox_id: {self.sandbox_id}, retry_times: {self.retry_times}"
            )

        if "Service unavailable: Upstream server is not reachable" in obs:
            self.env_failed = True
        if time.time() - self.time_start > self.max_env_time:
            self.env_timeout = True

        # 环境失败
        self.traj_env_time = round(self.traj_reset_time + self.traj_step_time + self.traj_reward_time, 4)
        self.traj_rollout_time = round(time.time() - self.time_start, 4)
        if self.env_failed or not self.sandbox_id or self.env_timeout:
            self.truncated = True
            self.metrics = {
                "env_timeout": self.env_timeout,
                "env_failed": self.env_failed,
                "reach_max_length": self.reach_max_length,
                "reach_max_turn": self.reach_max_turn,
                "if_sandbox_failed": self.if_sandbox_failed,
                "reward": self.reward,
                "success": self.terminate,
                "truncated": self.truncated,
                "format_penalty": self.format_penalty,
                "action_is_valid": (
                    round(sum(self.action_is_valid_lst) / len(self.action_is_valid_lst), 4)
                    if self.action_is_valid_lst
                    else 0.0
                ),
                "action_is_effective": (
                    round(sum(self.action_is_effective_lst) / len(self.action_is_effective_lst), 4)
                    if self.action_is_effective_lst
                    else 0.0
                ),
                "turn_count": int(self.turn_count),
                "retry_times": int(self.retry_times),
                "traj_reset_time": round(self.traj_reset_time, 4),
                "traj_step_time": round(self.traj_step_time + round(time.time() - st, 4), 4),
                "traj_reward_time": round(self.traj_reward_time, 4),
                "traj_env_time": self.traj_env_time,
                "traj_rollout_time": self.traj_rollout_time,
                "task_idx": int(self.task_idx),  # for rollout_log
            }
            info["metrics"] = self.metrics
            obs = f"ERROR: Start container failed, the container name is {self.container_name}"
            print(
                f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}[ENV_STEP][GSTEP{self.global_step}]❌[容器失败]traj_env_time: {round(self.traj_env_time/60, 2)} min, traj_rollout_time: {round(self.traj_rollout_time/60, 2)} / {round(self.max_env_time/60, 2)}min, task_idx: {self.task_idx}, current_turn: {self.turn_count}/{self.max_steps}, sandbox_id: {self.sandbox_id}, docker_image: {self.docker_image}, sandbox_failed_times: {self.if_sandbox_failed}'
            )
            self.logger.info(
                f"[ENV][STEP][容器失败]traj_env_time: {self.traj_env_time}, traj_rollout_time: {self.traj_rollout_time}, task_idx: {self.task_idx}, current_turn: {self.turn_count}/{self.max_steps}, sandbox_id: {self.sandbox_id}, docker_image: {self.docker_image}, retry_times: {self.retry_times}, obs: {obs}, info: {info}"
            )
            return obs, self.reward, self.terminate, self.truncated, info

        # invalid action
        try:
            if not ("Invalid Action" in str(exit_code)) and not format_info["error"]:  # 无效动作
                action_is_valid = True
            if not format_info["error"] and str(exit_code) == "0":  # 动作执行成功
                action_is_effective = True
        except Exception as e:
            print(
                f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}[ENV_STEP][GSTEP{self.global_step}]❌[注意env里的解析BUG][error in action valid]{e}'
            )
            action_is_valid = False
            action_is_effective = False

        # 超过最大轮数强制结束
        if not self.terminate and self.turn_count >= self.max_steps:
            self.truncated = True
            self.reach_max_turn = True
            self.logger.info(f"[ENV][STEP][truncated]超过最大轮数强制结束")
            print(
                f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}[ENV_STEP][GSTEP{self.global_step}]❌[ENV][STEP]超过最大轮数强制结束, traj_env_time: {round(self.traj_env_time/60, 2)} min, traj_rollout_time: {round(self.traj_rollout_time/60, 2)} / {round(self.max_env_time/60, 2)}min, task_idx: {self.task_idx}, turn_count: {self.turn_count} (max_steps: {self.max_steps}))'
            )
            self.logger.info(
                f"[ENV_STEP][GSTEP{self.global_step}]❌[ENV][STEP]超过最大轮数强制结束, traj_env_time: {round(self.traj_env_time/60, 2)} min, traj_rollout_time: {round(self.traj_rollout_time/60, 2)} / {round(self.max_env_time/60, 2)}min, task_idx: {self.task_idx}, turn_count: {self.turn_count} (max_steps: {self.max_steps}))"
            )

        # update history
        self.history.append(
            {
                "role": "assistant_parsed",
                "content": f'{action_info["action_content"]}',
                "action_is_valid": action_is_valid,
                "action_is_effective": action_is_effective,
            }
        )
        self.history.append({"role": "user", "content": f"{obs}"})

        # update action_valid_list / action_is_effective_lst / traj_step_time
        self.action_is_valid_lst.append(1 if action_is_valid else 0)
        self.action_is_effective_lst.append(1 if action_is_effective else 0)
        self.traj_step_time += round(time.time() - st, 4)
        self.traj_env_time = self.traj_reset_time + self.traj_step_time + self.traj_reward_time

        # update metrics
        self.metrics = {
            "env_timeout": self.env_timeout,
            "env_failed": self.env_failed,
            "reach_max_length": self.reach_max_length,
            "reach_max_turn": self.reach_max_turn,
            "if_sandbox_failed": self.if_sandbox_failed,  # 一条轨迹中是否存在过失败现象
            "reward": self.reward,
            "success": self.terminate,
            "truncated": self.truncated,
            "format_penalty": self.format_penalty,
            "action_is_valid": (
                round(sum(self.action_is_valid_lst) / len(self.action_is_valid_lst), 4)
                if self.action_is_valid_lst
                else 0.0
            ),
            "action_is_effective": (
                round(sum(self.action_is_effective_lst) / len(self.action_is_effective_lst), 4)
                if self.action_is_effective_lst
                else 0.0
            ),
            "turn_count": int(self.turn_count),
            "retry_times": int(self.retry_times),
            "traj_reset_time": round(self.traj_reset_time, 4),
            "traj_step_time": round(self.traj_step_time, 4),
            "traj_reward_time": round(self.traj_reward_time, 4),
            "traj_env_time": round(self.traj_env_time, 4),
            "traj_rollout_time": round(time.time() - self.time_start, 4),
            "task_idx": int(self.task_idx),
        }
        info = {
            "suffix": obs,
            "metrics": self.metrics,
        }
        self.logger.info(
            f"\n*************** 环境交互输出【正常】***************\nstep: {self.turn_count}\nobs: {obs}\nreward: {self.reward}\nterminate: {self.terminate}\ntruncated: {self.truncated}\ninfo: {info}"
        )
        print(
            f'\n{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}[ENV_STEP][GSTEP{self.global_step}]✅(used_time: {round((time.time() - st),2)}s)'
            f"[INPUT]task_idx:{self.task_idx}, currenct_turns: {self.turn_count}/{self.max_steps}, sandbox_id: {self.sandbox_id}, model_input: {[model_response]} "
            f"[OUTPUT]obs: {[obs]}"
            f"[METRIC]reward: {self.reward}, terminated: {self.terminate}, truncated: {self.truncated}, reach_max_length: {self.reach_max_length}, reach_max_turn: {self.reach_max_turn}."
            f"[TIME]traj_rollout_time: {round((time.time() - self.time_start)/60,2)}min, traj_env_time: {round((self.traj_env_time)/60,2)}min, traj_reset_time: {round((self.traj_reset_time)/60,2)}min, traj_step_time: {round((self.traj_step_time)/60,2)}min, traj_reward_time: {round((self.traj_reward_time)/60,2)}min, sandbox_failed_times: {self.if_sandbox_failed}"
        )
        return obs, self.reward, self.terminate, self.truncated, info

    def parse_action(self, response_text):
        """
        Extracts:
        - thought: everything before the first <function=...> block
        - action: the entire first <function=...></function> block
        Returns (thought, action).
        """
        # Regex to match (non-greedily) from `<function=` up to the first `</function>`
        pattern = re.compile(r"(?s)(<function=.*?</function>)")
        match = pattern.search(response_text)

        if match:
            action = match.group(1)  # The entire <function=...></function> block
            thought = response_text[: match.start()]  # Everything before the block
        else:
            # If no match, treat entire text as "thought"
            thought = response_text
            action = ""

        # Strip leading/trailing whitespace
        thought = thought.strip()
        action = action.strip()

        # convert action to Action object
        action_obj = Action.from_string(action)
        # print(f'[ATTENTION][response_text]{response_text}\n[before] action: {action}\n[after]action_obj: {action_obj}')

        action_info = {
            "response": response_text,
            "action": action_obj,  # Action object
            "action_content": action,  # string
            "think_content": thought,  # string
        }
        return action_info

    def format_exam(self, action_info: dict):
        """ """
        thought, action, response = action_info["think_content"], action_info["action"], action_info["action_content"]

        # format error type1
        action_dict = action.to_dict()
        if action_dict["function"] == "":
            return {
                "error": True,
                "error_msg": "ERROR: The model output is illegal, please check it carefully. Tips: It should be '<function=func1><parameter=param1>xxx</parameter><parameter=param2>xxx</parameter></function>'.",
            }
        # format error type2
        elif "<parameter>" in response:
            return {
                "error": True,
                "error_msg": "ERROR: The model output is illegal, please check it carefully. Tips: It should be '<parameter=xxxx>', not '<parameter>xxxx>.'",
            }
        elif re.search(r"<parameter\s*=\s*[^>]*<", response):
            return {
                "error": True,
                "error_msg": "ERROR: The model output is illegal, malformed parameter tag detected. Tips: use '<parameter=param1>value</parameter>'.",
            }
        elif "<function>" in response:
            return {
                "error": True,
                "error_msg": "ERROR: The model output is illegal, please check it carefully. Tips: It should be '<function=func1>', not '<function>func1>.'",
            }
        elif "<parameter" in response and "</parameter>" not in response:
            return {
                "error": True,
                "error_msg": "ERROR: The model output is illegal, please check it carefully. Tips: It should be '<parameter=param1>xxx</parameter>'. Do not forget to close the parameter tag.",
            }
        elif "<function" in response and "</function>" not in response:
            return {
                "error": True,
                "error_msg": "ERROR: The model output is illegal, please check it carefully. Tips: It should be '<function=func1>xxxx</function>'. Do not forget to close the function tag.",
            }
        elif not action:
            return {"error": True, "error_msg": "ERROR: The model output is illegal, please check it carefully."}
        # format error type3: function name
        function_name = action_dict["function"]
        if function_name not in self.allowed_cmds:
            return {
                "error": True,
                "error_msg": f"Invalid Action: input action must be one of allowed actions. Allowed actions: {self.allowed_cmds}. Current input action: {function_name}. ",
            }
        params = action_dict.get("parameters", {})
        if function_name == "execute_bash" and not ({"command", "cmd"} & set(params.keys())):
            return {
                "error": True,
                "error_msg": "Invalid Action: execute_bash requires '<parameter=command>...</parameter>' (or cmd).",
            }
        if function_name == "str_replace_editor" and not {"command", "path"}.issubset(set(params.keys())):
            return {
                "error": True,
                "error_msg": "Invalid Action: str_replace_editor requires both '<parameter=command>' and '<parameter=path>'.",
            }

        return {"error": False, "error_msg": ""}

    def calculate_reward(self, stop_reason: str = "", mode: str = ""):
        st_reward = time.time()
        # 如果 metrics 为 None，先初始化它
        print(
            f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}[ENV][GSTEP{self.global_step}]'
            f"🟢 start calculate reward ... mode: {mode}, task_idx: {self.task_idx}, terminate: {self.terminate}, truncated: {self.truncated}, reach_max_length: {self.reach_max_length}, stop_reason: {stop_reason}, used_time: {(round(time.time() - self.time_start, 4))/60}min. max_env_time: {self.max_env_time/60}min."
        )

        if self.metrics is None:
            self.metrics = {
                "env_timeout": self.env_timeout,
                "env_failed": self.env_failed,
                "reach_max_length": self.reach_max_length,
                "reach_max_turn": self.reach_max_turn,
                "if_sandbox_failed": self.if_sandbox_failed,
                "reward": self.reward,
                "success": self.terminate,
                "truncated": self.truncated,
                "format_penalty": self.format_penalty,
                "action_is_valid": (
                    round(sum(self.action_is_valid_lst) / len(self.action_is_valid_lst), 4)
                    if self.action_is_valid_lst
                    else 0.0
                ),
                "action_is_effective": (
                    round(sum(self.action_is_effective_lst) / len(self.action_is_effective_lst), 4)
                    if self.action_is_effective_lst
                    else 0.0
                ),
                "turn_count": int(self.turn_count),
                "retry_times": int(self.retry_times),
                "traj_reset_time": round(self.traj_reset_time, 4),
                "traj_step_time": round(self.traj_step_time, 4),
                "traj_reward_time": round(self.traj_reward_time, 4),
                "traj_env_time": round(self.traj_reset_time + self.traj_step_time + self.traj_reward_time, 4),
                "traj_rollout_time": round(time.time() - self.time_start, 4),
                "task_idx": int(self.task_idx) if self.task_idx is not None else 0,
            }

        if self.env_timeout or self.env_failed or (time.time() - self.time_start) > self.max_env_time:
            msg = "ERROR: Environment timeout or failed"
            info = {
                "suffix": msg if not self.unittest_output else self.unittest_output,
                "metrics": self.metrics,
            }
            print(
                f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}[ENV][GSTEP{self.global_step}]'
                f"❌[calcucate_reward][{self.mode}]{msg}, task {self.task_idx}, terminate: {self.terminate}, truncated: {self.truncated}, reach_max_length: {self.reach_max_length}, stop_reason: {stop_reason}, used_time: {(round(time.time() - self.time_start, 4))/60}min. max_env_time: {self.max_env_time/60}min."
            )
            return msg, 0, info

        # 更新metrics
        if stop_reason in ("reach_max_length", "max_length"):
            self.reach_max_length = True
        print(
            f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}[ENV][GSTEP{self.global_step}]'
            f"🟢[calcucate_reward][{self.mode}]task {self.task_idx} start calculate reward ... (terminate: {self.terminate}, truncated: {self.truncated}, reach_max_length: {self.reach_max_length}, stop_reason: {stop_reason})"
        )
        # 计算reward
        self.logger.info(
            f"[ENV][{self.mode}]task {self.task_idx} start calculate reward ... (terminate: {self.terminate}, truncated: {self.truncated}, reach_max_length: {self.reach_max_length}, stop_reason: {stop_reason})"
        )
        self.reward, self.unittest_output = self.repo_env.calculate_reward(
            max_execute_time=min(self.max_env_time - (time.time() - self.time_start), 10 * 60), max_execute_retry=3
        )
        self.unittest_output = "UNITTEST OUTPUT: \n" + self.unittest_output
        self.traj_reward_time = round(time.time() - st_reward, 4)
        self.traj_rollout_time = round(time.time() - self.time_start, 4)
        self.logger.info(
            f"[ENV][STEP][calcucate_reward]task {self.task_idx} reward: {self.reward}, stop_reason: {stop_reason}, traj_reward_time: {round(self.traj_reward_time/60, 2)} min, traj_env_time: {round(self.traj_env_time/60, 2)} min, traj_rollout_time: {round(self.traj_rollout_time/60, 2)} min, unittest_output: {[self.unittest_output]}"
        )
        print(
            f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}[ENV][GSTEP{self.global_step}]'
            f"✅[calcucate_reward][{self.mode}]task {self.task_idx}, reward: {self.reward}, stop_reason: {stop_reason}, traj_reward_time: {self.traj_reward_time}s, traj_env_time: {self.traj_env_time}s, traj_rollout_time: {self.traj_rollout_time}s, sandbox_failed_times: {self.if_sandbox_failed}"
        )
        # 更新history
        self.history.append({"role": "reward", "content": f"{self.unittest_output}", "reward": self.reward})
        # 更新metrics
        self.metrics["reach_max_length"] = self.reach_max_length
        self.metrics["traj_reward_time"] = round(self.traj_reward_time, 4)
        self.metrics["traj_env_time"] = round(self.traj_env_time, 4)
        self.metrics["traj_rollout_time"] = round(self.traj_rollout_time, 4)
        info = {
            "suffix": self.unittest_output,
            "metrics": self.metrics,
        }
        return self.unittest_output, self.reward, info

    def sample_random_action(self):
        actions = []
        actions.append(
            [
                "\nI'll work in this environment by creating a local copy of the file to edit it.\n\n<tool_call>\n<function=execute_bash>\n<parameter=command>\ncd /testbed && cp src/_pytest/logging.py /tmp/logging.py\n\n\n"
            ]
        )
        # # 执行正确动作
        # actions.append(
        #     "Now let's look for the @return_future decorator. Based on the issue description, it's likely in one of the tornado modules. Let's search for it:\n\n<function=search>\n  <parameter=search_term>return_future</parameter>\n  <parameter=path>/testbed</parameter>\n</function>"
        # )
        # actions.append(
        #     "Now let's look for the @return_future decorator. Based on the issue description, it's likely in one of the tornado modules. Let's search for it:\n\n<function=search>\n  <parameter=search_term>return_future</parameter>\n  <parameter=path>/testbed</parameter>\n</function>"
        # )
        # actions.append(
        #     "Now let's look for the @return_future decorator. Based on the issue description, it's likely in one of the tornado modules. Let's search for it:\n\n<function=search>\n  <parameter=search_term>return_future</parameter>\n  <parameter=path>/testbed</parameter>\n</function>"
        # )
        # # 执行不允许的动作（exit code: 1）
        actions.append(
            "the issue:\n\n<function=execute_bash>\n  <parameter=cmd>cd /testbed && python reproduce_issue.py</parameter>\n</function>"
        )
        # 执行错误动作（exit code: 2）
        actions.append("<function=execute_bash>\n  <parameter=cmd>python reproduce_issue.py</parameter>\n</function>")
        # # 提交
        actions.append(
            'tness.\n\n<function=finish>\n  <parameter=command>submit</parameter>\n  <parameter=result>I have successfully implemented a fix for the GitHub issue "Callback Receives Unexpected Argument When No Result is Provided" in the `@return_future` decorator.\n\n**Changes Made:**\n1. Added `import inspect` to `/testbed/tornado/concurrent.py`\n2. Modified the `run_callback` function to intelligently handle callbacks when no result is provided:\n   - When `result is None`, inspect the callback signature\n   - If callback accepts no parameters, call it without arguments\n   - Otherwise, call it with the `None` result as before\n   - Includes robust fallback mechanism for cases where signature inspection fails\n\n**Key Benefits:**\n- \u2705 Fixes the original issue: callbacks expecting no arguments work correctly\n- \u2705 Maintains backward compatibility: all existing tests pass\n- \u2705 Handles edge cases: works with lambdas, default parameters, *args, **kwargs\n- \u2705 Robust: graceful fallback when signature inspection fails\n- \u2705 Minimal impact: only changes behavior when result is `None`\n\nThe fix ensures that when a `@return_future` decorated function calls its callback without arguments, the client callback is invoked appropriately based on its signature, resolving the TypeError while preserving all existing functionality.</parameter>\n</function>'
        )
        return random.choice(actions)

    def close(self, stop_reason=None):
        self.logger.info(f"*************** task_idx:{self.task_idx} SWEEnv.close(start) ***************")
        if self.is_closed:
            self.logger.info(f"[重复close]task_idx: {self.task_idx}")
            return
        if self.env_timeout:
            stop_reason = "env_timeout"
        elif self.env_failed:
            stop_reason = "env_failed"
        elif self.reach_max_turn:
            stop_reason = "reach_max_turn"
        elif self.reach_max_length:
            stop_reason = "reach_max_length"
        elif stop_reason:
            stop_reason = stop_reason
        elif self.truncated:
            stop_reason = "truncated"
        elif self.terminate:
            stop_reason = "terminate"
        else:
            stop_reason = "unknown"
        self.history.append({"role": "reward", "content": f"{self.unittest_output}", "reward": self.reward})

        save = {
            "env_timeout": self.env_timeout,
            "env_failed": self.env_failed,
            "reach_max_length": self.reach_max_length,
            "reach_max_turn": self.reach_max_turn,
            "if_sandbox_failed": self.if_sandbox_failed,  # 一条轨迹中是否存在过失败现象
            "global_step": self.global_step,
            "task_idx": int(self.task_idx),
            "reward": self.reward,
            "unittest_output": self.unittest_output,
            "terminate": self.terminate,
            "truncated": self.truncated,
            "stop_reason": stop_reason,
            "container_name": self.container_name,
            "sandbox_id": self.sandbox_id,
            # "problem_statement": self.problem_statement,
            "retry_times": self.retry_times,
            "docker_image": self.docker_image,
            "history": self.history,
            "metrics": self.metrics,
            "retry_times": self.retry_times,
        }
        valid_score = (
            round(sum(self.action_is_valid_lst) / len(self.action_is_valid_lst), 1)
            if self.action_is_valid_lst
            else 0.0
        )
        effective_score = (
            round(sum(self.action_is_effective_lst) / len(self.action_is_effective_lst), 1)
            if self.action_is_effective_lst
            else 0.0
        )
        time_str = time.strftime("%m%d%H%M%S", time.localtime())
        log_path = os.path.join(
            self.traj_dir,
            f"step{self.global_step}/env_traj",
            f"re{self.reward}-{stop_reason}-{self.task_idx}-{time_str}-v{valid_score}_e{effective_score}_tc{self.turn_count}_time{self.traj_env_time}.json",
        )
        if SAVE_TRAJ:
            os.makedirs(self.traj_dir, exist_ok=True)
            write_data_json(save, log_path)
        # close
        self.repo_env.close()
        # 这里不能clean_record，否则返回就不对了。
        print(
            f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}[GSTEP{self.global_step}]🟢[ENV CLOSE][global_step:{self.global_step}][task_idx:{self.task_idx}][Step:{self.turn_count}][Rollout时间消耗]{round((time.time() - self.time_start)/60,2)}min. [服务交互消耗]{round((self.traj_env_time)/60,2)}min. [Metrics]{self.metrics} (sandbox_id: {self.sandbox_id})"
        )
        self.logger.info(
            f"\n*************** 【释放环境({stop_reason})】 ***************\n[global_step:{self.global_step}][task_idx:{self.task_idx}][Step:{self.turn_count}][Rollout时间消耗]{round((time.time() - self.time_start)/60,2)}min. [服务交互消耗]{round((self.traj_env_time)/60,2)}min. [Metrics]{self.metrics} (sandbox_id: {self.sandbox_id}), log: {log_path}"
        )
        self.logger.save()
        self.history = []
        self.data_line = {}
        self.is_closed = True

    def clean_record(self):
        print(
            f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}[GSTEP{self.global_step}]🟢[DEBUG][参数重置 ...]"
        )
        if self.logger:
            self.logger.close()
        self.logger = None
        self.history = []
        self.data_line = {}
        self.container_name = None
        self.sandbox_id = None
        self.task_idx = None
        self.turn_count = 0
        self.retry_times = 0
        self.traj_reset_time = 0
        self.traj_step_time = 0
        self.traj_reward_time = 0
        self.traj_env_time = 0
        self.problem_statement, self.issue = "", ""
        self.reward, self.terminate, self.truncated = 0, False, False
        self.env_timeout = False
        self.env_failed = False
        self.reach_max_length = False
        self.reach_max_turn = False
        self.unittest_output = ""
        self.is_closed = False
        self.if_sandbox_failed = 0  # 一条轨迹中是否存在过失败现象

    def get_history(self):
        return self.history

    def get_key_params(self):
        return {
            "task_idx": int(self.task_idx),
            "container_name": self.container_name,
            "sandbox_id": self.sandbox_id,
            "problem_statement": self.problem_statement,
            "retry_times": self.retry_times,
            "docker_image": self.docker_image,
        }
