import ast
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict

import gym

from roll.pipeline.agentic.env.swe_env.util.define.action import Action
from roll.pipeline.agentic.env.swe_env.util.define.commands import ParseCommandBash
from roll.pipeline.agentic.env.swe_env.util.define.log import get_logger
from roll.pipeline.agentic.env.swe_env.util.get_requirment.swebench_test_spec import make_test_spec
from roll.pipeline.agentic.env.swe_env.util.sanbox_server_sdk import ExecuteObservation, RunStatus
from roll.pipeline.agentic.env.swe_env.util.sanbox_server_sdk import SWERexClientSDK as SWERexClient
from roll.pipeline.agentic.env.swe_env.util.spec.swe_reward import SweReward

DEBUG = False

"""
调用sanbox_server执行命令，init_env
"""


class RepoClient(gym.Env):
    def __init__(
        self,
        logger=None,
        max_execute_time=60 * 20,
        max_execute_retry=3,
        timeout=360,
        max_env_time=60 * 60,
        sanbox_mode="http",  # 可以去掉了
        swe_rex_host="https://your-swe-rex-endpoint.com/docker",
        golden_patch=False,  # 是否使用golden patch
        user_id="374702",
        experiment_id=None,
        xrl_authorization=None,
        xrl_cluster="nt-a",
        clear_time=60 * 60,  # 60 minutes
        default_timeout=720,
        default_max_execute_time=720.0,
        default_max_execute_retry=3,
        task_idx: str = "",
    ):
        # Get the logger
        if logger is None:
            self.logger = get_logger("RepoClient")  # Pass the module name for clarity
        else:
            self.logger = logger
        self.task_idx = task_idx
        self.start_time = time.time()

        # sandbox相关的参数
        self.user_id = user_id
        self.experiment_id = experiment_id
        self.xrl_authorization = xrl_authorization
        self.xrl_cluster = xrl_cluster
        self.clear_time = clear_time
        self.session_name = "agent"
        self.timeout = default_timeout
        self.max_execute_time = default_max_execute_time
        self.max_execute_retry = default_max_execute_retry
        self.sanbox_mode = sanbox_mode
        self.runtime = SWERexClient(
            user_id=self.user_id,
            experiment_id=self.experiment_id,
            xrl_authorization=self.xrl_authorization,
            xrl_cluster=self.xrl_cluster,
            clear_time=self.clear_time,
            default_timeout=self.timeout,
            default_max_execute_time=self.max_execute_time,
            default_max_execute_retry=self.max_execute_retry,
            session_name=self.session_name,
            logger=self.logger,
            task_idx=self.task_idx,
        )
        self.sandbox_id = ""

        # init base path
        self.repo_path = "/testbed"
        self.alt_path = "/root"
        self.cmd_parser = ParseCommandBash()

        # init swe reward
        self.swe_reward = SweReward(logger=self.logger)

        # init for spec
        self.session = None
        self.container_name = None
        self.retry_times = None
        self.observation = None
        self.done = False
        self.state = "init"
        self.commands = []  # 初始化 commands 属性为空列表
        self.max_env_time = max_env_time  # env交互时间
        self.env_start_time = time.time()  # 记录最开始服务时间
        self.available_tools_name = []

        # 是否使用golden patch
        self.golden_patch = golden_patch
        if isinstance(golden_patch, str):
            self.golden_patch = ast.literal_eval(golden_patch)

        # for syn
        self.script_folder = ""
        self.project_path = ""
        print(f"[RepoClient]max_env_time: {max_env_time}, golden_patch: {golden_patch}")

    def init_data(self, ds):
        self.ds = ds
        self.docker_image = self.ds["docker_image"]
        self.category = (
            self.ds["category"] if self.ds.get("category") else ""
        )  # TODO: 文件夹名称， e.g. swebench-verified
        self.repo_name = self.ds["repo"] if self.ds.get("repo") else self.ds["repo_name"]
        self.task_idx = (
            self.ds.get("task_idx", None) if self.ds.get("task_idx") else self.ds.get("idx", None)
        )  # TODO: 任务索引， e.g. 1
        self.run_tests_regression = self.ds.get("run_tests_regression", None)

        # 后面处理下数据集
        self.source = ""
        if "swebench" in self.docker_image:
            self.source = "swebench"
        elif "vpc.cn" in self.docker_image:
            self.source = "r2e"
        else:
            self.source = "syn"
        self.docker_image = self.docker_image.replace("rex-registry-vpc", "rex-registry")

        self.swebench_verified = "swebench" in self.docker_image
        if self.source == "swebench":
            # also create a test spec for swebench verified dockers (useful for grading)
            self.test_spec = make_test_spec(self.ds)
        print(
            f"[RepoEnv]✅init_data完成! task_idx: {self.task_idx}, docker_image: {self.docker_image}, source: {self.source}"
        )

    def get_status(self, mode: str = "sandbox_failed_times"):
        if mode == "sandbox_failed_times":
            return self.runtime.get_sandbox_failed_times()
        elif mode == "sandbox_status":
            return self.runtime.get_sandbox_status()
        else:
            return self.runtime.get_sandbox_failed_times()

    def read_file(self, rel_file_path: str) -> str:
        # alt_path = /root
        result = self.runtime.run_in_session(
            command=f"cat {rel_file_path}", timeout=720, mode="nohup", max_execute_time=720, max_execute_retry=3
        )
        return result.output

    def reset(
        self,
        ds,
        clear_time: int = 120 * 60,  # 60 minutes
        timeout: int = 60 * 10,
        max_execute_time: float = 60 * 30,
        max_execute_retry: int = 10,
        update_route_key_interval: int = 11,
    ) -> Dict[str, Any]:
        """
        Resets the environment and returns an initial observation.
        return:
            {
                - state: "success" or "error" # Attention
                - error_message: error_message if state is "error".
                - container_name: only one container_name.
                - retry_times: final retry_times.
                - session: final session.
            }
        """
        # init data: docker_image, swebench_verified, repo_name, parsed_commit, expected_json, run_tests_regression
        self.init_data(ds)
        failed = False

        # post for start container
        self.logger.info(
            f"[RepoEnv]开始重置环境 ... "
            f"task_idx: {self.task_idx}, docker_image: {self.docker_image}, clear_time: {clear_time/60}min, max_env_time: {round(self.max_env_time,2)/60} min, max_execute_time: {max_execute_time} seconds, max_execute_retry: {max_execute_retry}"
        )
        print(
            f"[RepoEnv]🟢 开始重置环境 ... task_idx: {self.task_idx}, docker_image: {self.docker_image}, clear_time: {clear_time/60}min, max_env_time: {round(self.max_env_time,2)/60} min, max_execute_time: {max_execute_time} seconds, max_execute_retry: {max_execute_retry}"
        )

        # start container
        max_execute_time = min(self.max_env_time - (time.time() - self.env_start_time), max_execute_time)
        try:
            reset_info = self.runtime.start_container(
                docker_image=self.docker_image,
                clear_time=clear_time,  # attention
                timeout=timeout,
                max_execute_time=max_execute_time,
                max_execute_retry=max_execute_retry,
            )
        except Exception as e:
            error_message = f"Unexpected exception in start_container: {repr(e)}"
            self.logger.error(f"[RepoEnv][RESET][EXCEPTION] {error_message}")
            reset_info = RunStatus(
                state="error",
                sandbox_id="",
                retry_times=max_execute_retry,
                error_message=error_message,
                session_name=self.runtime.session_name if hasattr(self.runtime, "session_name") else "agent",
                host_ip="",
            )
            print(
                f"[RepoEnv]❌重置镜像失败! task_idx: {self.task_idx}, docker_image: {self.docker_image}, reset_info: {reset_info}, error_message: {error_message}"
            )
            failed = True
        self.state = reset_info.state if hasattr(reset_info, "state") else reset_info.get("state", "error")
        self.sandbox_id = (
            reset_info.sandbox_id if hasattr(reset_info, "sandbox_id") else reset_info.get("sandbox_id", None)
        )
        self.retry_times = (
            reset_info.retry_times if hasattr(reset_info, "retry_times") else reset_info.get("retry_times", 1)
        )
        self.error_message = (
            reset_info.error_message if hasattr(reset_info, "error_message") else reset_info.get("error_message", None)
        )
        self.session_name = (
            reset_info.session_name
            if hasattr(reset_info, "session_name")
            else reset_info.get("session_name", self.runtime.session_name)
        )
        if self.state == "error":
            failed = True

        if failed:
            failed_commands = ["start_container"]
            print(
                f"[RepoEnv]❌重置镜像失败! start_container failed, task_idx: {self.task_idx}, docker_image: {self.docker_image}, reset_info: {reset_info}"
            )
            self.logger.error(
                f"[RepoEnv]❌重置镜像失败! start_container failed, task_idx: {self.task_idx}, docker_image: {self.docker_image}, reset_info: {reset_info}"
            )
            return reset_info, failed_commands
        print(
            f"[RepoEnv]✅重置镜像完成! start_container success, task_idx: {self.task_idx}, docker_image: {self.docker_image}, reset_info: {reset_info}"
        )
        self.logger.info(
            f"[RepoEnv]✅重置镜像完成! start_container success, task_idx: {self.task_idx}, docker_image: {self.docker_image}, reset_info: {reset_info}"
        )

        # setup env
        failed, failed_commands = self.setup_env()
        if failed:
            print(
                f"[RepoEnv]❌setup env阶段失败! setup env failed, task_idx: {self.task_idx}, source: {self.source}, failed_commands: {failed_commands}"
            )
            self.logger.error(
                f"[RepoEnv]❌setup env阶段失败! setup env failed, task_idx: {self.task_idx}, source: {self.source}, failed_commands: {failed_commands}"
            )
            reset_info.error_message = f"setup env阶段失败, failed_commands: {failed_commands}"
            reset_info.state = "error_setup_env"
            return reset_info, failed_commands
        print(
            f"[RepoEnv]✅setup env阶段完成! setup env success, task_idx: {self.task_idx}, docker_image: {self.docker_image}, reset_info: {reset_info}"
        )
        self.logger.info(
            f"[RepoEnv]✅setup env阶段完成! setup env success, task_idx: {self.task_idx}, docker_image: {self.docker_image}, reset_info: {reset_info}"
        )
        return reset_info, failed_commands

    def setup_env(self):
        self.logger.info(f"[RepoEnv]开始setup环境 .... docker_image: {self.docker_image}")
        failed_commands = ["no_sandbox_id"]
        if not self.sandbox_id:
            return False, failed_commands
        # return_info = self.runtime.run_in_session("pip config set global.index-url https://your-pypi-mirror.com/1/pypi/simple")
        if self.source == "swebench":
            failed, failed_commands = self.setup_env_swebench()
        elif self.source == "r2e":
            failed, failed_commands = self.setup_env_trainset_r2e()
        else:
            failed, failed_commands = self.setup_env_trainset_syn()
        return failed, failed_commands

    def setup_env_swebench(self):
        self.alt_path = "/"  # the run_test is in the "/" directory for swebench dockers
        self.logger.info(f"[RepoEnv]开始准备swebench的环境 ...")
        failed = False
        failed_commands = []
        setup_commands = [
            "ln -s /opt/miniconda3/envs/testbed /root/.venv",
            "python -m pip install chardet -i https://mirrors.aliyun.com/pypi/simple/",
            "chmod +x /run_tests.sh",
        ]
        for command in setup_commands:
            return_info = self.runtime.run_in_session(
                command, timeout=720, mode="nohup", max_execute_time=720, max_execute_retry=3
            )
            if return_info.exit_code != 0:
                failed = True
                self.logger.error(f"[RepoEnv]swebench的环境准备失败!")
                print(
                    f"[RepoEnv]❌swebench的环境准备失败!, task_idx: {self.task_idx}, docker_image: {self.docker_image}, error command: {[command]}"
                )
                failed_commands.append(command)
        if failed:
            print(f"[RepoEnv]❌swebench的环境准备失败!, task_idx: {self.task_idx}, docker_image: {self.docker_image}")
            self.logger.info(
                f"[RepoEnv]swebench的环境准备失败!, task_idx: {self.task_idx}, docker_image: {self.docker_image}"
            )
            return failed, failed_commands
        else:
            self.logger.info(
                f"[RepoEnv]swebench的环境准备完成!, task_idx: {self.task_idx}, docker_image: {self.docker_image}"
            )
            print(f"[RepoEnv]✅swebench的环境准备完成!, task_idx: {self.task_idx}, docker_image: {self.docker_image}")
            return failed, failed_commands

    def setup_env_trainset_r2e(self):
        self.logger.info(f"[RepoEnv]开始准备r2e trainset的环境 ...")
        failed = False
        failed_commands = []
        setup_commands = [
            # '/testbed' -> '/root'
            "ln -s /testbed/.venv /root/.venv",
            "mkdir -p /root/.local/bin",
            "ln -s /testbed/.venv/bin/python /root/.local/bin/python",
            "ln -s /testbed/.venv/bin/python /root/.local/bin/python3",
            "mv /r2e_tests /root/r2e_tests",
            "mv /testbed/run_tests.sh /root/run_tests.sh",
            "sed -i 's|\.venv/bin/python|/testbed/.venv/bin/python|g' /root/run_tests.sh",
            "sed -i 's|r2e_tests|/root/r2e_tests|g' /root/run_tests.sh",
            "chmod +x /root/run_tests.sh",
        ]
        for command in setup_commands:
            return_info = self.runtime.run_in_session(
                command, timeout=720, mode="normal", max_execute_time=720, max_execute_retry=3
            )
            if return_info.exit_code != 0:
                failed = True
                self.logger.error(f"[RepoEnv]r2e trainset的环境准备失败!")
                failed_commands.append(command)

        setup_commands = [
            "uv pip install chardet -i https://mirrors.aliyun.com/pypi/simple/  --trusted-host mirrors.aliyun.com",
        ]
        for command in setup_commands:
            return_info = self.runtime.run_in_session(
                command, timeout=720, mode="nohup", max_execute_time=720, max_execute_retry=3
            )
            if return_info.exit_code != 0:
                failed = True
                self.logger.error(f"[RepoEnv]r2e trainset的环境准备失败!")
        self.logger.info("[RepoEnv]r2e trainset的环境准备完成 ...")
        if failed:
            print(
                f"[RepoEnv]❌r2e trainset的环境准备失败!, task_idx: {self.task_idx}, docker_image: {self.docker_image}, failed_commands: {failed_commands}"
            )
            self.logger.error(
                f"[RepoEnv]❌r2e trainset的环境准备失败!, task_idx: {self.task_idx}, docker_image: {self.docker_image}, failed_commands: {failed_commands}"
            )
            return failed, failed_commands
        else:
            print(
                f"[RepoEnv]✅r2e trainset的环境准备完成!, task_idx: {self.task_idx}, docker_image: {self.docker_image}"
            )
            self.logger.info(
                f"[RepoEnv]✅r2e trainset的环境准备完成!, task_idx: {self.task_idx}, docker_image: {self.docker_image}"
            )
            return failed, failed_commands

    def setup_env_trainset_syn(self):
        self.logger.info(f"[RepoEnv]开始准备syn trainset的环境 ...")
        image_info = json.loads(self.ds["image_info"])
        self.script_folder = image_info["script_folder"]  # 单测目录
        self.project_path = image_info["project_path"]  # 项目目录
        failed = False
        failed_commands = []

        setup_commands = [
            # "apt update",
            # "apt install -y python3-pip",
            "pip install chardet -i https://mirrors.aliyun.com/pypi/simple/  --trusted-host mirrors.aliyun.com",
        ]
        for command in setup_commands:
            return_info = self.runtime.run_in_session(
                command, timeout=720, mode="nohup", max_execute_time=720, max_execute_retry=3
            )
            if return_info.exit_code != 0:
                self.logger.error(f"[RepoEnv]syn trainset的环境准备失败!")
                print(
                    f"[RepoEnv]❌syn trainset的环境准备失败!, task_idx: {self.task_idx}, docker_image: {self.docker_image}, error command: {[command]}"
                )
                failed = True
                failed_commands.append(command)
        setup_commands = [
            f"chmod +x {self.script_folder}apply_test_patch.sh",
            f"chmod +x {self.script_folder}run.sh",
            f"chmod +x {self.script_folder}apply_fix_patch.sh",
            f"bash -c cd {self.project_path}",
        ]
        for command in setup_commands:
            return_info = self.runtime.run_in_session(
                command, timeout=720, mode="normal", max_execute_time=720, max_execute_retry=3
            )
            if return_info.exit_code != 0:
                self.logger.error(f"[RepoEnv]syn trainset的环境准备失败!")
                print(
                    f"[RepoEnv]❌syn trainset的环境准备失败!, task_idx: {self.task_idx}, docker_image: {self.docker_image}, error command: {[command]}"
                )
                failed = True
                failed_commands.append(command)
        self.logger.info("[RepoEnv]syn trainset的环境准备完成 ...")
        if failed:
            print(
                f"[RepoEnv]❌syn trainset的环境准备失败!, task_idx: {self.task_idx}, docker_image: {self.docker_image}"
            )
            self.logger.error(
                f"[RepoEnv]❌syn trainset的环境准备失败!, task_idx: {self.task_idx}, docker_image: {self.docker_image}, failed_commands: {failed_commands}"
            )
            return failed, failed_commands
        else:
            print(
                f"[RepoEnv]✅syn trainset的环境准备完成!, task_idx: {self.task_idx}, docker_image: {self.docker_image}"
            )
            self.logger.info(
                f"[RepoEnv]✅syn trainset的环境准备完成!, task_idx: {self.task_idx}, docker_image: {self.docker_image}"
            )
            return failed, failed_commands

    def calculate_reward(self, max_execute_time: int = 10 * 60, max_execute_retry: int = 3) -> int:
        """
        Basic reward calculation based on command success.
        """
        if not self.sandbox_id:
            return 0, "ERROR: Sandbox ID is None"
        if self.source == "swebench":
            reward, output = self._calculate_reward_swebench(max_execute_time, max_execute_retry)
        elif self.source == "r2e":
            reward, output = self._calculate_reward_r2e(max_execute_time, max_execute_retry)
        else:
            reward, output = self._calculate_reward_syn(max_execute_time, max_execute_retry)
        emoji = "😊" if reward > 0 else "😭"
        print(f"[RepoEnv]{emoji}计算reward: {reward}, task_idx: {self.task_idx}")
        self.logger.info(f"[RepoEnv]{emoji}计算reward: {reward}, output: {[output]}, task_idx: {self.task_idx}")
        return reward, output

    def _calculate_reward_r2e(self, max_execute_time: int = 10 * 60, max_execute_retry: int = 3) -> int:
        self.logger.info(f"[RepoEnv]计算r2e trainset的reward start ...")
        self.expected_json = self.ds.get("expected_output_json", "")

        # run_tests.sh
        result = self.runtime.run_in_session(
            command=f"/root/run_tests.sh",
            timeout=max_execute_time,
            mode="nohup",
            max_execute_time=max_execute_time,
            max_execute_retry=max_execute_retry,
        )
        output = result.output

        reward = self.swe_reward.calculate_reward_r2e(
            output=output,
            ds=self.ds,
            repo_name=self.repo_name,
            expected_json=self.expected_json,
            get_test_output=False,
        )
        print(
            f"[RepoEnv]✅计算r2e trainset的reward完成! task_idx: {self.task_idx}, docker_image: {self.docker_image}, reward: {reward}"
        )
        return reward, output

    def _calculate_reward_syn(self, max_execute_time: int = 1800, max_execute_retry: int = 3) -> int:
        self.logger.info(f"[RepoEnv]计算syn trainset的reward start ...")
        setup_commands = [
            f"bash -c cd {self.script_folder}" f"chmod +x {self.script_folder}apply_test_patch.sh",
            f"chmod +x {self.script_folder}run.sh",
            f"chmod +x {self.script_folder}apply_fix_patch.sh",
        ]
        for command in setup_commands:
            execute_observation = self.runtime.run_in_session(
                command=command,
                timeout=300,
                mode="normal",
                max_execute_time=300,
                max_execute_retry=3,
                wait_interval=10,
                response_limited_bytes_in_nohup=1024 * 1024 * 64,
            )
            if execute_observation.exit_code != 0:
                self.logger.error(f"[RepoEnv]计算reward时，syn trainset的环境准备失败!")

        # apply_test_patch.sh
        result = self.runtime.run_in_session(
            command=f"sh {self.script_folder}apply_test_patch.sh",
            timeout=360,
            mode="nohup",
            max_execute_time=360,
            max_execute_retry=3,
        )
        if self.golden_patch:
            result = self.runtime.run_in_session(
                command=f"sh {self.script_folder}apply_fix_patch.sh",
                timeout=max_execute_time,
                mode="nohup",
                max_execute_time=max_execute_time,
                max_execute_retry=max_execute_retry,
            )

        # run.sh
        result = self.runtime.run_in_session(
            command=f"sh {self.script_folder}run.sh",
            timeout=max_execute_time,
            mode="nohup",
            max_execute_time=max_execute_time,
            max_execute_retry=3,
        )
        output = result.output
        if "all test cases run successfully" in output:
            reward = 1.0
        else:
            reward = 0.0
        print(
            f"[RepoEnv]✅计算syn trainset的reward完成! task_idx: {self.task_idx}, docker_image: {self.docker_image}, reward: {reward}"
        )
        return reward, output

    def _calculate_reward_swebench(self, max_execute_time: int = 360, max_execute_retry: int = 3) -> int:
        # 覆盖run_tests.sh 文件，并运行单测
        if self.golden_patch:
            print(f"[RepoEnv]使用golden patch: {[self.ds['run_tests_golden']]}")
            self.runtime.create_file(file_path="/run_tests.sh", content=self.ds["run_tests_golden"])
        else:
            self.runtime.create_file(file_path="/run_tests.sh", content=self.ds["run_tests"])
        chmod_result = self.runtime.run_in_session(
            command="chmod +x /run_tests.sh", timeout=360, mode="normal", max_execute_time=360, max_execute_retry=3
        )
        if chmod_result.exit_code != 0:
            self.logger.warning(f"Failed to chmod /run_tests.sh: {chmod_result.error_message}")
        # run the tests after applying the patch
        result = self.runtime.run_in_session(
            command="/run_tests.sh",
            timeout=max_execute_time,
            mode="nohup",
            max_execute_time=max_execute_time,
            max_execute_retry=3,
        )
        out = result.output
        # parse eval logs & calculate reward
        reward = self.swe_reward.calculate_reward_swebench(self.ds, out, get_test_output=False)
        print(
            f"[RepoEnv]✅计算swebench的reward完成! task_idx: {self.task_idx}, docker_image: {self.docker_image}, reward: {reward}"
        )
        return reward, out

    def get_system_prompt_in_iflow(self):
        """后续改下"""
        stdout, exit_code = self._execute_command("iflow --sysinfo 目录下有哪些文件")
        text = stdout
        try:
            stdout = stdout.split("You are iFlow CLI, ")[-1].split("query is completely resolved")[0]
            text = f"You are iFlow CLI, {stdout}query is completely resolved"
        except Exception as e:
            print(f"[RepoEnv][ERROR][get_system_prompt_in_iflow error]: {e}")
        return text

    def add_commands(self, cmd_files: list[str]):
        """
        Adds command files and tool names to the environment.

        Elements ending with '.py' are treated as file paths and parsed normally. copying them to the Docker container, and making them executable or sourced.
        Elements not ending with '.py' are treated as tool names and registered directly.

        Args:
            cmd_files: List of paths to command files or tool names.
        """
        from roll.pipeline.agentic.env.swe_env.util.define.commands import Command

        cmds = []
        for cmd_file in cmd_files:
            # Check if it's a tool name (not ending with .py)
            if not cmd_file.endswith(".py"):
                # Register as a tool name directly
                name = cmd_file.split("/")[-1]
                tool_cmd = Command(
                    name=name,
                    command=f"# Tool: {name}",
                    docstring=f"Tool command: {name}",
                    arguments=None,
                    signature=None,
                )
                cmds.append(tool_cmd)
                self.logger.info(f"Registered tool: {cmd_file}")
                continue

            # Process as a file path
            current_file_path = Path(__file__).resolve()
            cmd_file = cmd_file.replace("./", f"{current_file_path.parent.parent.parent.parent.parent}/")
            # Parse commands from file
            parsed_commands = self.cmd_parser.parse_command_file(cmd_file)
            cmds.extend(parsed_commands)
            # Process the command file
            self._process_command_file(cmd_file)

        # Add to existing commands
        self.commands = cmds  # name, code,
        self.logger.info(f"Added {len(cmds)} commands to the environment.")
        print(f"Added {len(cmds)} commands to the environment.")
        self.available_tools_name = self.get_available_cmds()

    def get_available_cmds(self) -> list[str]:
        return [x.name for x in self.commands]

    def _process_command_file(self, cmd_file: str):
        """
        Process a single command file by copying it to the container and setting appropriate permissions.

        Args:
            cmd_file: Path to the command file to process.
        """
        # Determine the file extension and get base name
        _, ext = os.path.splitext(cmd_file)
        cmd_name = os.path.basename(cmd_file)

        # Determine container command name and path
        if ext == ".py" or self._is_shebang_script(cmd_file):
            # Python script or shebang script: strip .py extension if applicable
            container_cmd_name = cmd_name[:-3] if ext == ".py" else cmd_name
            container_path = f"/usr/local/bin/{container_cmd_name}"
            upload_result = self.runtime.upload_file(file_path=cmd_file, target_path=container_path)
            if upload_result.exit_code != 0:
                self.logger.error(f"Failed to upload {cmd_file} to {container_path}: {upload_result.error_message}")
            # 修改 Python 脚本的 shebang 指向正确的 Python 环境 # TODO
            # if self.swebench_verified:
            #     # swebench 环境使用 conda 环境
            #     python_path = "/opt/miniconda3/envs/testbed/bin/python"
            # else:
            #     # trainset 环境使用 /testbed/.venv/bin/python
            #     python_path = "/testbed/.venv/bin/python"
            # 使用 sed 修改第一行的 shebang
            # sed_result = self.runtime.run_in_session(command=f"sed -i '1s|^#!.*|#!{python_path}|' {container_path}",timeout=720,mode='normal',max_execute_time=720,max_execute_retry=3)
            # if sed_result.exit_code != 0:
            #     self.logger.warning(f"Failed to update shebang: {sed_result.error_message}")
            chmod_result = self.runtime.run_in_session(
                command=f"chmod +x {container_path}",
                timeout=720,
                mode="normal",
                max_execute_time=720,
                max_execute_retry=3,
            )
            if chmod_result.exit_code != 0:
                self.logger.warning(f"Failed to chmod {container_path}: {chmod_result.error_message}")
                print(
                    f"[REPOENV][PROCESS COMMAND FILE]❌failed to chmod {container_path}: {chmod_result.error_message}, task_idx: {self.task_idx}"
                )
        else:
            # Bash script: copy, chmod, and source it
            container_cmd_name = cmd_name
            container_path = f"/usr/local/bin/{container_cmd_name}"
            upload_result = self.runtime.upload_file(file_path=cmd_file, target_path=container_path)
            if upload_result.exit_code != 0:
                self.logger.error(f"Failed to upload {cmd_file} to {container_path}: {upload_result.error_message}")
                print(
                    f"[REPOENV][PROCESS COMMAND FILE]❌failed to upload {cmd_file} to {container_path}: {upload_result.error_message}, task_idx: {self.task_idx}"
                )
            chmod_result = self.runtime.run_in_session(
                command=f"chmod +x {container_path}",
                timeout=720,
                mode="normal",
                max_execute_time=720,
                max_execute_retry=3,
            )
            if chmod_result.exit_code != 0:
                self.logger.warning(f"Failed to chmod {container_path}: {chmod_result.error_message}")
                print(
                    f"[REPOENV][PROCESS COMMAND FILE]❌failed to chmod {container_path}: {chmod_result.error_message}, task_idx: {self.task_idx}"
                )
            # Source the script inside the container
            source_result = self.runtime.run_in_session(
                command=f"bash -c 'source {container_path}'",
                timeout=720,
                mode="normal",
                max_execute_time=720,
                max_execute_retry=3,
            )
            if source_result.exit_code != 0:
                self.logger.warning(f"Failed to source {container_path}: {source_result.error_message}")
                print(
                    f"[REPOENV][PROCESS COMMAND FILE]❌failed to source {container_path}: {source_result.error_message}, task_idx: {self.task_idx}"
                )

    def _is_shebang_script(self, cmd_file: str) -> bool:
        """
        Checks if the given file starts with a shebang (#!).

        Args:
            cmd_file: Path to the command file.

        Returns:
            True if the file starts with a shebang, False otherwise.
        """
        with open(cmd_file, "r") as file:
            first_line = file.readline().strip()
        return first_line.startswith("#!")

    def _should_use_nohup(self, command: str) -> bool:
        """
        判断命令是否需要使用 nohup 执行

        Args:
            command: 要执行的命令

        Returns:
            True if the command should use nohup, False otherwise
        """
        # 排除不应该使用 nohup 的命令
        exclude_patterns = [
            r"^export\s+",  # export 命令
            r"^mkdir\s+",  # mkdir 命令
            r"^chmod\s+",  # chmod 命令
            r"^ln\s+-s",  # 符号链接命令
            r"^mv\s+",  # 移动文件命令
            r"^echo\s+",  # echo 命令
            r"^bash\s+-c",  # bash -c 命令
        ]

        command_lower = command.lower().strip()
        for pattern in exclude_patterns:
            if re.search(pattern, command_lower):
                return False

        # 需要 nohup 的命令模式
        nohup_patterns = [
            r"pip\s+install",
            r"uv\s+pip\s+install",
            r"python\s+-m\s+pip\s+install",
            r"poetry\s+install",
            r"pipenv\s+install",
            r"setup\.py\s+install",
            r"python\s+setup\.py",
            r"conda\s+activate",
            r"source\s+.*activate",
        ]

        command_lower = command.lower().strip()
        for pattern in nohup_patterns:
            if re.search(pattern, command_lower):
                return True
        return False

    def _convert_command_and_run_in_session(
        self,
        command: str,
        timeout: int = 180,
        mode: str = "nohup",  # nohup, or normal
        max_execute_time: float = 300.0,
        max_execute_retry: int = 3,
        wait_interval: int = 10,
        response_limited_bytes_in_nohup: int = 1024 * 1024 * 64,
    ) -> ExecuteObservation:
        """
        执行命令，根据命令类型决定是否使用 nohup

        Args:
            command: 要执行的命令
            timeout: 超时时间

        Returns:
            (stdout, exit_code) 元组
        """
        command_converted = command

        # TODO: 是否必要：为 r2e trainset环境设置执行前缀
        self.logger.info(f"[RepoEnv]self.available_tools_name: {self.available_tools_name}")
        if command_converted.split(" ")[0] in self.available_tools_name and not self.swebench_verified:
            command_converted = "/usr/local/bin/" + command_converted
            self.logger.info(f"[CONVERT_COMMAND]之前:{[command_converted]}, 之后:{[command_converted]}")

        # TODO: 是否必要：execute_bash + cd 命令的特殊处理
        if command_converted.startswith("execute_bash") and "cd " in command_converted:
            cmd_match = re.search(r"--cmd\s+'([^']+)'", command_converted)
            if cmd_match:
                actual_cmd = cmd_match.group(1)
                # 直接使用 bash -c 执行包含 cd 的命令，避免 execute_bash 工具的限制
                command_converted = f"bash -c '{actual_cmd}'"
            self.logger.info(f"[CONVERT_COMMAND]之前:{[command_converted]}, 之后:{[command_converted]}")

        # run in session
        return self.runtime.run_in_session(
            command=command_converted,
            timeout=timeout,
            mode=mode,  # nohup, or normal
            max_execute_time=max_execute_time,
            max_execute_retry=max_execute_retry,
            wait_interval=wait_interval,
            response_limited_bytes_in_nohup=response_limited_bytes_in_nohup,
        )

    def run_action(
        self,
        action: Action,
        timeout: int,
        base_agent: str = "swe",
        mode: str = "nohup",  # nohup, or normal
        max_execute_time: float = 300.0,  # 这里在env.py中设置
        max_execute_retry: int = 3,  # 默认
        wait_interval: int = 10,  # 默认
        response_limited_bytes_in_nohup: int = 1024 * 1024 * 64,  # 默认
    ):
        """
        @input:
            - action: 模型的输入
        @output:
            - bash_output, exit_code: 环境执行反馈
        """
        start_time = time.time()
        bash_cmd, exit_code, bash_output = action.function_name, "-100", ""

        # if base_agent == 'swe':
        allowed_cmds = [x.name for x in self.commands]
        # if base_agent == 'iflow':
        # allowed_cmds = allowed_cmds + ['list_directory','read_file','write_file','replace','multi_edit','search_file_content','web_search','todo_write','todo_read','glob','run_shell_command','web_fetch']
        # check for empty or no function call / action
        if not action.function_name:
            bash_output = (
                f"Invalid Action: input action must be one of allowed actions \n Allowed actions: {allowed_cmds}\n."
            )
            exit_code = -100
            execute_observation = ExecuteObservation(
                output=bash_output, exit_code=exit_code, error_message=bash_output
            )
        # Check if action is in allowed actions/commands
        elif action.function_name not in allowed_cmds:
            bash_output = f"Invalid Action: input action must be one of allowed actions \n Allowed actions: {allowed_cmds}\n. Input action: {action.function_name}\t"
            exit_code = -100
            execute_observation = ExecuteObservation(
                output=bash_output, exit_code=exit_code, error_message=bash_output
            )
        # Run action and return
        elif base_agent == "swe":
            bash_cmd = action.to_bashcmd()
            if "python -c" in bash_cmd:
                self.runtime.run_in_session(
                    command="set +H",
                    timeout=3 * 60,
                    mode="normal",
                    max_execute_time=5 * 60,
                    max_execute_retry=2,
                    wait_interval=10,
                    response_limited_bytes_in_nohup=1024 * 1024 * 64,
                )
            execute_observation = self._convert_command_and_run_in_session(
                bash_cmd,
                timeout,
                mode,
                max_execute_time,
                max_execute_retry,
                wait_interval,
                response_limited_bytes_in_nohup,
            )
            if "python -c" in bash_cmd:
                self.runtime.run_in_session(
                    command="set +H",
                    timeout=3 * 60,
                    mode="normal",
                    max_execute_time=5 * 60,
                    max_execute_retry=2,
                    wait_interval=10,
                    response_limited_bytes_in_nohup=1024 * 1024 * 64,
                )
        elif base_agent == "iflow":
            bash_cmd = action.to_iflowcmd(tool_id=str(uuid.uuid4().hex) + "_" + str(time.time()))
            print(f"[run_action: iflow]bash_cmd: {bash_cmd}")
            execute_observation = self._convert_command_and_run_in_session(
                bash_cmd,
                timeout,
                mode,
                max_execute_time,
                max_execute_retry,
                wait_interval,
                response_limited_bytes_in_nohup,
            )
        else:
            bash_output = f"Unknown base_agent: {base_agent}"
            exit_code = -200
            execute_observation = ExecuteObservation(
                output=bash_output, exit_code=exit_code, error_message=bash_output
            )

        if execute_observation.exit_code != 0:
            if_alive = self.runtime.check_alive()
            if not if_alive:
                bash_output = f"sandbox is not alive"
                exit_code = -200
                execute_observation = ExecuteObservation(
                    output=bash_output, exit_code=exit_code, error_message=bash_output
                )
                return execute_observation.output, execute_observation.exit_code, time.time() - start_time
        # self.logger.info(
        #     f"[RepoEnv][执行命令]{[bash_cmd]}\n[RepoEnv][执行反馈]exit_code: {exit_code}\nbash_output: \n{bash_output}"
        # )
        # if len(bash_output) > 500:
        #     print(f"[RepoEnv][执行命令]{[bash_cmd]}\n[RepoEnv][执行反馈]exit_code: {exit_code}, bash_output: {[bash_output[:500]]} ... (truncated)")
        # else:
        #     print(f"[RepoEnv][执行命令]{[bash_cmd]}\n[RepoEnv][执行反馈]exit_code: {exit_code}, bash_output: {[bash_output]}")
        return execute_observation.output, execute_observation.exit_code, time.time() - start_time

    def close(self):
        try:
            if self.sandbox_id:
                self.runtime.stop_container()
        except Exception as e:
            self.logger.warning(f"关闭容器时出错: {e}")
            print(f"[ERROR]关闭容器时出错: {e}")

    def _rm_conda_in_swebench(self):
        """
        注释掉 /run_tests.sh 中与环境创建相关的命令
        包括 conda activate、source activate、pip install 等
        """
        try:
            # 读取当前的 /run_tests.sh 文件
            self.logger.info(f"[rm_conda_in_swebench]在创建文件之前，cat /run_tests.sh的内容: ")
            result = self.runtime.run_in_session(
                command="cat /run_tests.sh", timeout=300, mode="normal", max_execute_time=300, max_execute_retry=3
            )
            out_file = result.output
            if not out_file:
                self.logger.error("[ERROR]无法读取 /run_tests.sh 文件")
                return False

            # 定义需要注释的环境相关命令
            env_commands_to_comment = [
                # conda 相关
                r"^conda activate",
                r"^source /opt/miniconda3/bin/activate",
                r"^source.*activate",
                # pip install 相关
                r"^python -m pip install",
                r"^pip install",
                r"^uv pip install",
                r"^poetry install",
                r"^pipenv install",
                # setup.py install 相关
                r"^python setup\.py install",
                r"^setup\.py install",
                r"^python.*setup\.py.*install",
                # 错误命令
                r"^cat: 300: No such file or directory",
            ]

            lines = out_file.split("\n")
            modified_lines = []
            commented_count = 0

            for line in lines:
                should_comment = False

                # 检查是否需要注释
                for pattern in env_commands_to_comment:
                    if re.search(pattern, line.strip()):
                        should_comment = True
                        break

                if should_comment and line.strip() and not line.strip().startswith("#"):
                    # 添加注释符号
                    modified_lines.append(f"# {line}")
                    commented_count += 1
                    # self.logger.info(f"注释环境命令: {line.strip()}")
                else:
                    modified_lines.append(line)

            modified_lines.insert(1, "export GIT_PAGER=cat")
            if len(modified_lines) > 0 and "#!/bin/bash" not in modified_lines[0]:
                modified_lines.insert(0, "#!/bin/bash")
            modified_content = "\n".join(modified_lines)
            # 创建临时文件
            temp_content = f"""{modified_content}"""
            # if self.sanbox_mode=='sdk':
            # 使用 cat 和 heredoc 语法避免 bash 历史扩展问题
            # self.runtime.run(f"cat > /run_tests.sh << 'EOF'\n{temp_content}\nEOF", timeout=300)
            # else:
            # self.logger.info(f"[rm_conda_in_swebench]在创建文件之前，cat /run_tests.sh的内容: ")
            # out_file, _ = self.runtime.run("cat /run_tests.sh", timeout=300)
            # self.logger.info(f"[rm_conda_in_swebench]创建文件时的temp_content: \n{temp_content}")
            # self.runtime.create_file(file_path="/tmp/run_tests_modified.sh", content=temp_content)
            # self.runtime.run("cp /tmp/run_tests_modified.sh /run_tests.sh", timeout=300, arun=False)
            create_result = self.runtime.create_file(file_path="/run_tests.sh", content=temp_content)
            if create_result.exit_code != 0:
                self.logger.error(f"Failed to create /run_tests.sh: {create_result.error_message}")
                return False
            chmod_result = self.runtime.run_in_session(
                command="chmod +x /run_tests.sh", timeout=300, mode="normal", max_execute_time=300, max_execute_retry=3
            )
            if chmod_result.exit_code != 0:
                self.logger.warning(f"Failed to chmod /run_tests.sh: {chmod_result.error_message}")
            self.logger.info(f"成功注释环境中的耗时命令，耗时命令数量: {commented_count}, 当前的单测文件：\n")
            result = self.runtime.run_in_session(
                command="cat /run_tests.sh", timeout=300, mode="normal", max_execute_time=300, max_execute_retry=3
            )
            out_file = result.output
            return True

        except Exception as e:
            self.logger.info(f"[ERROR]注释环境命令时出错: {repr(e)}")
            return False
