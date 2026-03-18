"""
与sanbox交互的底层client写在这里
sdk交互方式。
"""

import asyncio
import os
import re
import time

from rock.sdk.sandbox.client import Sandbox
from rock.sdk.sandbox.config import SandboxConfig
from rock.sdk.sandbox.request import CreateBashSessionRequest
from rock.sdk.sandbox.response import Observation

from roll.pipeline.agentic.env.swe_env.util.define.log import get_logger

DEBUG = False
from pydantic import BaseModel
from typing import Literal

# os.environ['XRL_AUTHORIZATION'] = 't-qx3yxhe61uo88fbz'
# os.environ['XRL_CLUSTER'] = 'nt-a'

class RunStatus(BaseModel):
    state: str = "error"
    session_name: str = ""
    retry_times: int = 0
    error_message: str = ""
    sandbox_id: str = ""
    host_ip: str = ""

class ExecuteObservation(BaseModel):
    session_type: Literal["bash"] = "bash"
    output: str = ""
    exit_code: int | None = None
    failure_reason: str = ""
    expect_string: str = ""
    sandbox_failed_times: int = 0
    error_message: str = ""


class SWERexClientSDK:
    def __init__(self, 
            logger=None, 
            user_id="374702", 
            experiment_id=None,
            xrl_authorization='t-qx3yxhe61uxxx',
            xrl_cluster="nt-a",
            clear_time = 60 * 60, # 60 minutes
            default_timeout = 180, # 3 minutes
            default_max_execute_time = 300.0, # 5 minutes
            default_max_execute_retry = 3,
            session_name:str = 'agent',
            task_idx: str = ""
            ):

        # sanbox基础信息
        self.sandbox = None
        # 确保 user_id 是字符串类型（SandboxConfig 要求字符串）
        self.user_id = str(user_id) if user_id is not None else "default"
        self.session_name = session_name
        self.experiment_id = experiment_id
        self.xrl_authorization = xrl_authorization
        self.xrl_cluster = xrl_cluster
        self.clear_time = clear_time
        self.default_timeout = default_timeout
        self.default_max_execute_time = default_max_execute_time
        self.default_max_execute_retry = default_max_execute_retry

        self.start_time = time.time()

        self.task_idx = task_idx
        self.host_ip = ''

        # 后续会更新的信息
        self.sandbox_id = ""
        self.sandbox_failed_times = 0
        self.url = ''

        if logger == None:
            self.logger = get_logger("SWERexClinet")
        else:
            self.logger = logger
        elapsed_min = round((time.time() - self.start_time) / 60, 2)
        print(f'[SWERexClientSDK] init success, user_id: {self.user_id}, experiment_id: {self.experiment_id}, xrl_authorization: {self.xrl_authorization}, xrl_cluster: {self.xrl_cluster}, session_name: {self.session_name}, task_idx: {self.task_idx}, elapsed_time: {elapsed_min}min')

    def get_sandbox_failed_times(self):
        # 监控某env中sandbox是否存在异常情况
        return self.sandbox_failed_times
    
    def get_sandbox_status(self):
        # TODO
        # status_lst = [success_init, success_execute, failed_execute, ....]
        # error_msg = []
        # success_msd = []
        # current_status = status_lst[-1]
        pass

    def _start_sandbox(self):
        # init sandbox
        start = time.time()
        try:
            config = SandboxConfig(
                image=self.docker_image,
                auto_clear_seconds=self.clear_time, # seconds
                startup_timeout=self.startup_timeout,
                xrl_authorization=self.xrl_authorization,  # FIXME: change dynamic author_key
                user_id=str(self.user_id),
                experiment_id=self.experiment_id,
                cluster=self.xrl_cluster # zb-a, zb-b
            )
            self.sandbox = Sandbox(config)
            self.logger.info(
                f"[SANBOX SDK][INIT CONFIG SUCCESS]✅"
                f"sandbox created config success, auto_clear_seconds: {self.clear_time/60} min, task_idx: {self.task_idx}",
            )
            elapsed_min = round((time.time() - self.start_time) / 60, 2)
            print(
                f"[SANBOX SDK][INIT CONFIG SUCCESS]✅"
                f"sandbox created config success, auto_clear_seconds: {self.clear_time/60} min, task_idx: {self.task_idx}, elapsed_time: {elapsed_min}min",
            )
            asyncio.run(self.sandbox.start())
            cost = time.time() - start
            self.logger.info(f"docker_image: {self.docker_image}, sandbox_id:{self.sandbox.sandbox_id}, sandbox ip: {self.sandbox.host_ip},  start sandbox cost:{cost}, task_idx: {self.task_idx}")
            self.sandbox_id = self.sandbox.sandbox_id
            return True, self.sandbox.host_ip
        except Exception as e:
            self.logger.error(f"docker_image: {self.docker_image}, error_massage:{e}, task_idx: {self.task_idx}")
            time.sleep(20.0)
            print(f"[SANBOX SDK][START SANDBOX ERROR]❌ Failed to start sandbox, error_message: {e}, task_idx: {self.task_idx}, docker_image: {self.docker_image}")
            return False, None


    def start_container(
        self,
        docker_image: str = None,
        clear_time: int = 120 * 60, # 60 minutes
        timeout=360,
        max_execute_time: float = 20 * 60, # 20 minutes
        max_execute_retry: int = 10,
    ):
        """请求远程服务init_env, pull docker image, 并返回RunStatus。
        @input:
            - route_key: if None, will be auto-generated.
            - docker_image:
            - clear_time: (s)
            - timeout: (s)
        @output:
            - return_info:
                - state: "success" or "error" # Attention
                - error_message: error_message if state is "error".
                - route_key: final route_key.
                - sandbox_id: only one sandbox_id.
                - retry_times: final retry_times.
                - session: final session.
        """
        st, execute_time, retry_times = time.time(), 0, 1
        error_message = ""
        
        # 清理时间
        if clear_time is None:
            self.clear_time = self.default_clear_time
        else:
            self.clear_time = clear_time
        self.docker_image = docker_image
        self.startup_timeout = timeout
        execute_time = round(time.time() - st, 4)

        while execute_time < max_execute_time and retry_times < max_execute_retry and (time.time() - self.start_time) < self.clear_time:
            elapsed_min = round((time.time() - self.start_time) / 60, 2)
            print(f"[SANBOX SDK][START CONTAINER]🟢 starting start sandbox.... retry_times:{retry_times}/{max_execute_retry}, execute_time: {execute_time}/{max_execute_time}, task_idx: {self.task_idx}, docker_image: {self.docker_image}, timeout: {timeout}.")
            # start sandbox
            success, host_ip = self._start_sandbox()
            if success and host_ip:
                self.host_ip = host_ip
            execute_time = round(time.time() - st,2)
            if not success:
                error_message = "Failed to start sandbox"
                self.logger.error(
                    f"[SANBOX SDK][START SANBOX ERROR]❌(retry_times:{retry_times}/{max_execute_retry}, execute_time: {execute_time}/{max_execute_time}) docker_image: {self.docker_image}, error: {error_message}, task_idx: {self.task_idx}"
                )
                elapsed_min = round((time.time() - self.start_time) / 60, 2)
                print(f"[SANBOX SDK][START SANBOX ERROR]❌Failed to start sandbox, retry_times: {retry_times}/{max_execute_retry}, execute_time: {execute_time}/{max_execute_time}, docker_image: {self.docker_image}, error_message: {error_message}, task_idx: {self.task_idx}, elapsed_time: {elapsed_min}min (PARAMS: timeout: {timeout}, auto_clear_time: {self.clear_time/60} min)")
                retry_times += 1
                time.sleep(120)
                continue
            print(f"[SANBOX SDK][START SANDBOX SUCCESS]✅ Successfully started sandbox, retry_times: {retry_times}/{max_execute_retry}, execute_time: {execute_time}/{max_execute_time}, docker_image: {self.docker_image}, sandbox_id: {self.sandbox_id}, host_ip: {self.host_ip}, task_idx: {self.task_idx}, elapsed_time: {elapsed_min}min (PARAMS: timeout: {timeout}, auto_clear_time: {self.clear_time/60} min)")

            print(f"[SANBOX SDK][START CONTAINER]🟢 starting check sandbox status.... retry_times:{retry_times}/{max_execute_retry}, execute_time: {execute_time}/{max_execute_time}, task_idx: {self.task_idx}, docker_image: {self.docker_image}, timeout: {timeout}.")
            try:
                is_alive_response = asyncio.run(self.sandbox.is_alive())
            except (Exception, KeyboardInterrupt, SystemExit) as alive_e:
                error_msg = f"Failed to check sandbox status: {repr(alive_e)}"
                elapsed_min = round((time.time() - self.start_time) / 60, 2)
                print(f"[SANBOX SDK][START SANBOX ERROR]❌ Failed to start sandbox, retry_times: {retry_times}/{max_execute_retry}, execute_time: {execute_time}/{max_execute_time}, docker_image: {self.docker_image}, error_message: {error_message}, error_messages: {error_msg}, task_idx: {self.task_idx}, elapsed_time: {elapsed_min}min (PARAMS: timeout: {timeout}, auto_clear_time: {self.clear_time/60} min)")
            if not is_alive_response.is_alive:
                retry_times += 1
                time.sleep(120)
                continue
            print(f"[SANBOX SDK][START SANDBOX SUCCESS]✅ Successfully checked sandbox status, retry_times: {retry_times}/{max_execute_retry}, execute_time: {execute_time}/{max_execute_time}, docker_image: {self.docker_image}, sandbox_id: {self.sandbox_id}, host_ip: {self.host_ip}, task_idx: {self.task_idx}, elapsed_time: {elapsed_min}min (PARAMS: timeout: {timeout}, auto_clear_time: {self.clear_time/60} min)")
            
            # crate session
            try:
                self.url = self.sandbox._url
                self.sandbox_id = self.sandbox.sandbox_id
                self.host_ip = self.sandbox.host_ip
                # 创建session
                asyncio.run(self.sandbox.create_session(CreateBashSessionRequest(session=self.session_name,env_enable=True)))
                execute_time = round(time.time() - st, 2)
                if self.sandbox_id:
                    run_status = RunStatus(state="success", sandbox_id=self.sandbox.sandbox_id, retry_times=retry_times, error_message="", session_name=self.session_name, host_ip=self.sandbox.host_ip)
                    self.logger.info(
                        f"[SANBOX SDK][CREATE SESSION SUCCESS]✅ retry_times:{retry_times}/{max_execute_retry}, execute_time: {execute_time}/{max_execute_time}, docker_image: {self.docker_image}, sandbox_id: {self.sandbox_id}, host_ip: {self.host_ip}, task_idx: {self.task_idx}, (PARAMS: timeout: {timeout})"
                    )
                    elapsed_min = round((time.time() - self.start_time) / 60, 2)
                    print(f"[SANBOX SDK][CREATE SESSION SUCCESS]✅ Successfully create session, retry_times: {retry_times}/{max_execute_retry}, execute_time: {execute_time}/{max_execute_time}, docker_image: {self.docker_image}, sandbox_id: {self.sandbox_id}, host_ip: {self.host_ip}, task_idx: {self.task_idx}, elapsed_time: {elapsed_min}min (PARAMS: timeout: {timeout}, auto_clear_time: {self.clear_time/60} min)")
                    return run_status
                else:
                    error_message = "Failed to create session: no sandbox id"
                    self.logger.error(
                        f"[SANBOX SDK][CREATE SESSION FAILED]{retry_times}/{max_execute_retry}, execute_time: {execute_time}/{max_execute_time}, docker_image: {self.docker_image}, sandbox_id: {self.sandbox_id}, error_message: {error_message}, task_idx: {self.task_idx}"
                    )
                    elapsed_min = round((time.time() - self.start_time) / 60, 2)
                    print(f"[SANBOX SDK][CREATE SESSION FAILED]❌ Failed to create session, retry_times: {retry_times}/{max_execute_retry}, execute_time: {execute_time}/{max_execute_time}, docker_image: {self.docker_image}, sandbox_id: {self.sandbox_id}, error_message: {error_message}, task_idx: {self.task_idx}, elapsed_time: {elapsed_min}min (PARAMS: timeout: {timeout}, auto_clear_time: {self.clear_time/60} min)")
                    retry_times += 1
            except Exception as start_e:
                error_message = f"Failed to create session: {repr(start_e)}"
                self.logger.error(
                    f"[SANBOX SDK][START CONTAINER ERROR]❌(retry_times:{retry_times}/{max_execute_retry}, execute_time: {execute_time}/{max_execute_time}) docker_image: {self.docker_image}, error: {error_message}, task_idx: {self.task_idx}"
                )
                elapsed_min = round((time.time() - self.start_time) / 60, 2)
                print(f"[SANBOX SDK][CREATE SESSION FAILED]❌ Failed to create session, retry_times: {retry_times}/{max_execute_retry}, execute_time: {execute_time}/{max_execute_time}, docker_image: {self.docker_image}, error_message: {error_message}, task_idx: {self.task_idx}, elapsed_time: {elapsed_min}min (PARAMS: timeout: {timeout}, auto_clear_time: {self.clear_time/60} min)")
                retry_times += 1
            time.sleep(120)
            execute_time = round(time.time() - st, 4)
        elapsed_min = round((time.time() - self.start_time) / 60, 2)
        print(
            f"[SANBOX SDK][START CONTAINER ERROR](final)❌ (retry_times: {retry_times}/{max_execute_retry}, execute_time: {execute_time}/{max_execute_time})"
            f"docker_image: {self.docker_image}, timeout: {timeout},max_execute_time: {max_execute_time}, max_execute_retry: {max_execute_retry}, sandbox_id: {self.sandbox_id}, error: {error_message}, task_idx: {self.task_idx}, elapsed_time: {elapsed_min}min",
        )
        self.logger.error(
            f"[SANBOX SDK][START CONTAINER ERROR](final)❌ (retry_times: {retry_times}/{max_execute_retry}, execute_time: {execute_time}/{max_execute_time})"
            f"docker_image: {self.docker_image}, timeout: {timeout}, sandbox_id: {self.sandbox_id}, error: {error_message}, task_idx: {self.task_idx}"
        )
        run_status = RunStatus(state="error", sandbox_id="", retry_times=retry_times, error_message=error_message, session_name=self.session_name, host_ip=self.sandbox.host_ip)
        return run_status

    def stop_container(self):
        elapsed_min = round((time.time() - self.start_time) / 60, 2)
        print(f"[SANBOX SDK][STOP CONTAINNER]🟢 start ...... sandbox_id: {self.sandbox_id}, docker_image: {self.docker_image}, task_idx: {self.task_idx}, elapsed_time: {elapsed_min}min") # 这里可以注释
        if self.sandbox and self.sandbox_id:
            is_alive_response = asyncio.run(self.sandbox.is_alive())
            if not is_alive_response.is_alive:
                elapsed_min = round((time.time() - self.start_time) / 60, 2)
                print(f"[SANBOX SDK][STOP CONTAINNER FAILED]❌ sandbox is not alive"
                    f"sandbox_id: {self.sandbox_id}, docker_image: {self.docker_image}, task_idx: {self.task_idx}, elapsed_time: {elapsed_min}min"
                )
                self.logger.error(f"[SANBOX SDK][STOP CONTAINNER FAILED]❌ sandbox is not alive"
                    f"sandbox_id: {self.sandbox_id}, docker_image: {self.docker_image}, task_idx: {self.task_idx}"
                )
                return False
            try:
                asyncio.run(self.sandbox.stop())
                elapsed_min = round((time.time() - self.start_time) / 60, 2)
                print(f"[SANBOX SDK][STOP CONTAINNER SUCCESS]✅ Successfully stopped sandbox_id: {self.sandbox_id}, docker_image: {self.docker_image}, task_idx: {self.task_idx}, elapsed_time: {elapsed_min}min") # 这里可以注释
                self.logger.info(f"[SANBOX SDK][STOP CONTAINNER SUCCESS]✅ Successfully stopped sandbox_id: {self.sandbox_id}, docker_image: {self.docker_image}, task_idx: {self.task_idx}")
                return True
            except (Exception, KeyboardInterrupt, SystemExit) as stop_e:
                error_msg = f"Failed to stop sandbox via asyncio.run: {repr(stop_e)}"
                self.sandbox_failed_times += 1
                elapsed_min = round((time.time() - self.start_time) / 60, 2)
                print(
                    "[SANBOX SDK][STOP CONTAINNER ERROR]❌"
                    f"[{os.getpid()}]sandbox_id: {self.sandbox_id}, docker_image: {self.docker_image}, error message: {error_msg}, task_idx: {self.task_idx}, elapsed_time: {elapsed_min}min",
                )
                self.logger.error(
                    f"[SANBOX SDK][STOP CONTAINNER ERROR]❌"
                    f"[{os.getpid()}]sandbox_id: {self.sandbox_id}, docker_image: {self.docker_image}, error message: {error_msg}, task_idx: {self.task_idx}"
                )
                return False
    def run_in_session(
        self,
        command: str = [],
        timeout: int = 720,
        mode: str = "nohup", # nohup, or normal
        max_execute_time: float = 720.0,
        max_execute_retry: int = 3,
        wait_interval:int = 10,
        response_limited_bytes_in_nohup:int = 1024 * 1024 * 20
    ):
        """ """
        retry_times, execute_time = 0, 0
        st = time.time()
        error_message = ""
        response = ExecuteObservation(output="", exit_code=-1, failure_reason="",sandbox_failed_times=0, error_message="")
        execute_time = round(time.time() - st,2)
        is_dns_error = False  # 在循环外初始化，避免未定义错误

        while execute_time < max_execute_time and retry_times < max_execute_retry and (time.time() - self.start_time) < self.clear_time:
            execute_time = round(time.time() - st,2)
            print(f"[SANBOX SDK][RUN IN SESSION]🟢 starting run in session.... retry_times: {retry_times}/{max_execute_retry}, execute_time: {execute_time}/{max_execute_time}, task_idx: {self.task_idx}, sandbox_id: {self.sandbox_id}, mode: {mode}, command: {[command]}")
            # try:
            #     is_alive_response = asyncio.run(self.sandbox.is_alive())
            #     if not is_alive_response.is_alive:
            #         error_msg = f"sandbox is not alive"
            #         elapsed_min = round((time.time() - self.start_time) / 60, 2)
            #         print(f"[SANBOX SDK][RUN IN SESSION ERROR]❌ sandbox is not alive, task_idx: {self.task_idx}, elapsed_time: {elapsed_min}min")
            #         self.logger.error(f"[SANBOX SDK][RUN IN SESSION ERROR]❌ sandbox is not alive, task_idx: {self.task_idx}")
            #         self.sandbox_failed_times += 1
            #         return ExecuteObservation(output="sandbox is not alive", exit_code=-1, failure_reason=error_msg, sandbox_failed_times=self.sandbox_failed_times, error_message=error_msg)
            # except (Exception, KeyboardInterrupt, SystemExit) as alive_e:
            #     error_msg = f"Failed to check sandbox status: {repr(alive_e)}"
            #     elapsed_min = round((time.time() - self.start_time) / 60, 2)
            #     print(f"[SANBOX SDK][RUN IN SESSION ERROR]❌ Failed to check sandbox status: {repr(alive_e)}, task_idx: {self.task_idx}, elapsed_time: {elapsed_min}min")
            #     self.logger.error(f"[SANBOX SDK][RUN IN SESSION ERROR]❌ Failed to check sandbox status: {repr(alive_e)}, task_idx: {self.task_idx}")
            #     self.sandbox_failed_times += 1
            #     return ExecuteObservation(output="sandbox is not alive", exit_code=-1, failure_reason=error_msg, sandbox_failed_times=self.sandbox_failed_times, error_message=error_msg)

            # 使用 try-except 包装 asyncio.run，确保异常被正确捕获
            try:
                response: Observation = asyncio.run(self.sandbox.arun(
                        command, 
                        session=self.session_name,
                        mode=mode,
                        wait_timeout=timeout,
                        wait_interval=wait_interval, 
                        response_limited_bytes_in_nohup=response_limited_bytes_in_nohup)
                )
                # exit_code: 0, success
                # exit_code: -1, failed，命令执行失败
                # exit_code: 1, failed，sdk问题（超时也为1）
                execute_time = time.time() - st
                elapsed_min = round((time.time() - self.start_time) / 60, 2)
                if "cannot schedule new futures after interpreter shutdown" in response.output or "cannot schedule new futures after interpreter shutdown" in response.failure_reason:
                    attention_message = "[ATTENTION] cannot schedule new futures after interpreter shutdown. "
                    print(
                        f"[SANBOX SDK][RUN IN SESSION FAILED]❌{attention_message}"
                        f"(retry_times:{retry_times}/{max_execute_retry}, execute_time: {execute_time}/{max_execute_time})"
                        f'sandbox_id: {self.sandbox_id}, mode: {mode}, command: {[command]}, '
                        f'exit_code: {response.exit_code}, failure_reason: {[response.failure_reason]}, output: {[response.output]}, task_idx: {self.task_idx}, elapsed_time: {elapsed_min}min'
                    )
                    self.logger.error(
                        f"[SANBOX SDK][RUN IN SESSION FAILED]❌{attention_message}"
                        f"(retry_times:{retry_times}/{max_execute_retry}, execute_time: {execute_time}/{max_execute_time})"
                        f'sandbox_id: {self.sandbox_id}, mode: {mode}, command: {[command]}, '
                        f'exit_code: {response.exit_code}, failure_reason: {[response.failure_reason]}, output: {[response.output]}, task_idx: {self.task_idx}'
                    )
                    self.sandbox_failed_times += 1
                    return ExecuteObservation(output=response.output, exit_code=response.exit_code, failure_reason=response.failure_reason,sandbox_failed_times=self.sandbox_failed_times, error_message=error_message)

                elif "Service unavailable: Upstream server is not reachable" in response.output or "Service unavailable: Upstream server is not reachable" in response.failure_reason:
                    attention_message = "[ATTENTION] Service unavailable: Upstream server is not reachable. "
                    print(
                        f"[SANBOX SDK][RUN IN SESSION FAILED]❌{attention_message}"
                        f"(retry_times:{retry_times}/{max_execute_retry}, execute_time: {execute_time}/{max_execute_time})"
                        f'sandbox_id: {self.sandbox_id}, mode: {mode}, command: {[command]}, '
                        f'exit_code: {response.exit_code}, failure_reason: {[response.failure_reason]}, output: {[response.output]}, task_idx: {self.task_idx}, elapsed_time: {elapsed_min}min'
                    )
                    self.logger.error(
                        f"[SANBOX SDK][RUN IN SESSION FAILED]❌{attention_message}"
                        f"(retry_times:{retry_times}/{max_execute_retry}, execute_time: {execute_time}/{max_execute_time})"
                        f'sandbox_id: {self.sandbox_id}, mode: {mode}, command: {[command]}, '
                        f'exit_code: {response.exit_code}, failure_reason: {[response.failure_reason]}, output: {[response.output]}, task_idx: {self.task_idx}'
                    )
                    self.sandbox_failed_times += 1
                    return ExecuteObservation(output=response.output, exit_code=response.exit_code, failure_reason=response.failure_reason,sandbox_failed_times=self.sandbox_failed_times, error_message=error_message)
                
                elif "Failed to submit command, nohup output: " in response.output:
                    attention_message = "[ATTENTION] Failed to submit command. "
                    print(
                        f"[SANBOX SDK][RUN IN SESSION FAILED]❌{attention_message}"
                        f"(retry_times:{retry_times}/{max_execute_retry}, execute_time: {execute_time}/{max_execute_time})"
                        f'sandbox_id: {self.sandbox_id}, mode: {mode}, command: {[command]}, '
                        f'exit_code: {response.exit_code}, failure_reason: {[response.failure_reason]}, output: {[response.output]}, task_idx: {self.task_idx}, elapsed_time: {elapsed_min}min'
                    )
                    self.sandbox_failed_times += 1
                    return ExecuteObservation(output=response.output, exit_code=response.exit_code, failure_reason=response.failure_reason,sandbox_failed_times=self.sandbox_failed_times, error_message=error_message)
                elif response.exit_code == 0:
                    # 命令执行正常
                    if response.failure_reason.strip() != "":
                        attention_message = "[ATTENTION] failure reason is not empty. "
                    else:
                        attention_message = ""
                    print(
                        f"[SANBOX SDK][RUN IN SESSION SUCCESS]✅{attention_message}"
                        f"(retry_times:{retry_times}/{max_execute_retry}, execute_time: {execute_time}/{max_execute_time}), "
                        f'docker_image: {self.docker_image}, sandbox_id: {self.sandbox_id}, mode: {mode}, command: {[command]}, '
                        f'exit_code: {response.exit_code}, failure_reason: {[response.failure_reason]}, output: {[response.output]}, task_idx: {self.task_idx}, elapsed_time: {elapsed_min}min'
                    )
                    self.logger.info(
                        f"[SANBOX SDK][RUN IN SESSION SUCCESS]✅{attention_message}"
                        f"(retry_times:{retry_times}/{max_execute_retry}, execute_time: {execute_time}/{max_execute_time}), "
                        f'docker_image: {self.docker_image}, sandbox_id: {self.sandbox_id}, mode: {mode}, command: {[command]}, '
                        f'exit_code: {response.exit_code}, failure_reason: {[response.failure_reason]}, output: {[response.output]}, task_idx: {self.task_idx}'
                    )
                    if response.output == response.failure_reason:
                        output = response.output
                    else:
                        output = response.output+'\n'+response.failure_reason.strip()
                    return ExecuteObservation(output=output, exit_code=response.exit_code, failure_reason=response.failure_reason,sandbox_failed_times=self.sandbox_failed_times, error_message=error_message)
                elif response.exit_code == -1:
                    # 命令执行错误
                    if response.output == response.failure_reason:
                        output = response.output
                    else:
                        output = response.output+'\n'+response.failure_reason.strip()
                    print(
                        f"[SANBOX SDK][RUN IN SESSION ERROR]❌{attention_message}"
                        f"(retry_times:{retry_times}/{max_execute_retry}, execute_time: {execute_time}/{max_execute_time})"
                        f'sandbox_id: {self.sandbox_id}, mode: {mode}, command: {[command]}, response: {response}, '
                        f'failure_reason: {[response.failure_reason]}, task_idx: {self.task_idx}, elapsed_time: {elapsed_min}min'
                    )
                    self.logger.info(
                        f"[SANBOX SDK][RUN IN SESSION ERROR]❌{attention_message}"
                        f"(retry_times:{retry_times}/{max_execute_retry}, execute_time: {execute_time}/{max_execute_time})"
                        f'sandbox_id: {self.sandbox_id}, mode: {mode}, command: {[command]}, response: {response}, '
                        f'failure_reason: {[response.failure_reason]}, task_idx: {self.task_idx}'
                    )
                    return ExecuteObservation(output=output, exit_code=response.exit_code, failure_reason=response.failure_reason,sandbox_failed_times=self.sandbox_failed_times, error_message=error_message)
                else:
                    if "Failed to execute nohup command" in response.failure_reason and "/bin/bash: " in response.failure_reason:
                        output=response.output+response.failure_reason if hasattr(response, 'output') and hasattr(response, 'failure_reason') else str(response)
                        attention_message = "[ATTENTION] Failed to execute nohup command & /bin/bash."
                        error_message = attention_message
                        print(
                            f"[SANBOX SDK][RUN IN SESSION ERROR]❌{attention_message}"
                            f"(retry_times:{retry_times}/{max_execute_retry}, execute_time: {execute_time}/{max_execute_time})"
                            f'sandbox_id: {self.sandbox_id}, mode: {mode}, command: {[command]}, '
                            f'exit_code: {response.exit_code}, failure_reason: {[response.failure_reason]}, output: {[response.output]}, task_idx: {self.task_idx}, elapsed_time: {elapsed_min}min'
                        )
                        self.logger.error(
                            f"[SANBOX SDK][RUN IN SESSION ERROR]❌{attention_message}"
                            f"(retry_times:{retry_times}/{max_execute_retry}, execute_time: {execute_time}/{max_execute_time})"
                            f'sandbox_id: {self.sandbox_id}, mode: {mode}, command: {[command]}, '
                            f'exit_code: {response.exit_code}, failure_reason: {[response.failure_reason]}, output: {[response.output]}, task_idx: {self.task_idx}'
                        )
                        self.sandbox_failed_times += 1
                        return ExecuteObservation(output=output, exit_code=response.exit_code, failure_reason=response.failure_reason,sandbox_failed_times=self.sandbox_failed_times, error_message=error_message)
                    # 服务端错误：包括timeout
                    attention_message = "[ATTENTION] server error. "
                    print(
                        f"[SANBOX SDK][RUN IN SESSION ERROR]❌{attention_message}"
                        f"(retry_times:{retry_times}/{max_execute_retry}, execute_time: {execute_time}/{max_execute_time})"
                        f'sandbox_id: {self.sandbox_id}, mode: {mode}, command: {[command]}, '
                        f'exit_code: {response.exit_code}, failure_reason: {[response.failure_reason]}, output: {[response.output]}, task_idx: {self.task_idx}, elapsed_time: {elapsed_min}min'
                    )
                    self.logger.error(
                        f"[SANBOX SDK][RUN IN SESSION ERROR]❌{attention_message}"
                        f"(retry_times:{retry_times}/{max_execute_retry}, execute_time: {execute_time}/{max_execute_time})"
                        f'sandbox_id: {self.sandbox_id}, mode: {mode}, command: {[command]}, '
                        f'exit_code: {response.exit_code}, failure_reason: {[response.failure_reason]}, output: {[response.output]}, task_idx: {self.task_idx}'
                    )
                    self.sandbox_failed_times += 1
            except (Exception, KeyboardInterrupt, SystemExit) as e:
                # 捕获所有可能的异常，包括 KeyboardInterrupt 和 SystemExit
                error_message = repr(e)
                elapsed_min = round((time.time() - self.start_time) / 60, 2)
                execute_time = time.time() - st
                # 检查 response 是否存在且有 exit_code 属性
                if hasattr(response, 'exit_code') and response.exit_code == -1:
                    # normal模式会抛出异常，nohup模式不会抛出异常. normal模式下，cp/export类命令异常
                    # 命令执行错误
                    if hasattr(response, 'output') and hasattr(response, 'failure_reason'):
                        if response.output == response.failure_reason:
                            output = response.output
                        elif response.failure_reason.strip() != "":
                            output = response.output+'\n'+response.failure_reason
                        else:
                            output = response.output
                        print(
                            f"[SANBOX SDK][RUN IN SESSION SUCCESS]❌[ATTENTION]exception: {error_message}"
                            f"(retry_times:{retry_times}/{max_execute_retry}, execute_time: {execute_time}/{max_execute_time})"
                            f'sandbox_id: {self.sandbox_id}, mode: {mode}, command: {[command]}, response: {response}, '
                            f'failure_reason: {[response.failure_reason]}, task_idx: {self.task_idx}, elapsed_time: {elapsed_min}min'
                        )
                        self.logger.info(
                            f"[SANBOX SDK][RUN IN SESSION SUCCESS]❌[ATTENTION]exception: {error_message}"
                            f"(retry_times:{retry_times}/{max_execute_retry}, execute_time: {execute_time}/{max_execute_time})"
                            f'sandbox_id: {self.sandbox_id}, mode: {mode}, command: {[command]}, response: {response}, '
                            f'failure_reason: {[response.failure_reason]}, task_idx: {self.task_idx}'
                        )
                        return ExecuteObservation(output=output, exit_code=response.exit_code, failure_reason=response.failure_reason,sandbox_failed_times=self.sandbox_failed_times, error_message=error_message)
                
                # 检查是否是 DNS 错误
                is_dns_error = "name resolution" in str(e).lower() or "Temporary failure" in str(e) or "ConnectError" in str(type(e).__name__)
                
                # 如果 response 不存在或没有必要的属性，创建一个默认的
                if not hasattr(response, 'exit_code') or not hasattr(response, 'output'):
                    response = ExecuteObservation(
                        output=f"Exception occurred: {error_message}",
                        exit_code=-1,
                        failure_reason=error_message,
                        sandbox_failed_times=self.sandbox_failed_times,
                        error_message=error_message
                    )
                elapsed_min = round((time.time() - self.start_time) / 60, 2)
                print(
                    "[SANBOX SDK][RUN IN SESSION ERROR]❌(Exception)"
                    f"(retry_times:{retry_times}, execute_time: {execute_time}, sandbox_failed_times: {self.sandbox_failed_times})"
                    f'sandbox_id: {self.sandbox_id}, mode: {mode}, command: {[command]}, '
                    f'exit_code: {response.exit_code}, failure_reason: {[response.failure_reason]}, output: {[response.output]},'
                    f'error message: {repr(e)}, is_dns_error: {is_dns_error}, task_idx: {self.task_idx}, elapsed_time: {elapsed_min}min',
                )
                self.logger.error(
                    f"[SANBOX SDK][RUN IN SESSION ERROR]❌(Exception)"
                    f"(retry_times:{retry_times}, execute_time: {execute_time}, sandbox_failed_times: {self.sandbox_failed_times})"
                    f'sandbox_id: {self.sandbox_id}, mode: {mode}, command: {[command]}, '
                    f'exit_code: {response.exit_code}, failure_reason: {[response.failure_reason]}, output: {[response.output]},'
                    f'error message: {repr(e)}, is_dns_error: {is_dns_error}, task_idx: {self.task_idx}'
                )
                self.sandbox_failed_times += 1
            retry_times += 1
            execute_time = round(time.time() - st, 4)
            time.sleep(120)
        
        # 确保 response 有必要的属性
        if not hasattr(response, 'exit_code') or not hasattr(response, 'output') or not hasattr(response, 'failure_reason'):
            response = ExecuteObservation(
                output=f"Max retries exceeded. Last error: {error_message}",
                exit_code=-1,
                failure_reason=error_message if error_message else "Max retries exceeded",
                sandbox_failed_times=self.sandbox_failed_times,
                error_message=error_message if error_message else "Max retries exceeded"
            )
        elapsed_min = round((time.time() - self.start_time) / 60, 2)
        print(
            f"[SANBOX SDK][RUN IN SESSION ERROR]❌(final)(retry_times:{retry_times}, execute_time: {execute_time}, sandbox_failed_times: {self.sandbox_failed_times})"
            f'sandbox_id: {self.sandbox_id}, mode: {mode}, command: {[command]}, '
            f'exit_code: {response.exit_code}, failure_reason: {[response.failure_reason]}, output: {[response.output]}, '
            f'error message: {error_message}, is_dns_error: {is_dns_error}, task_idx: {self.task_idx}, elapsed_time: {elapsed_min}min'
        )
        self.logger.error(
            f"[SANBOX SDK][RUN IN SESSION ERROR]❌(final)(retry_times:{retry_times}, execute_time: {execute_time}, sandbox_failed_times: {self.sandbox_failed_times})"
            f'sandbox_id: {self.sandbox_id}, mode: {mode}, command: {[command]}, '
            f'exit_code: {response.exit_code}, failure_reason: {[response.failure_reason]}, output: {[response.output]}, '
            f'error message: {error_message}, is_dns_error: {is_dns_error}, task_idx: {self.task_idx}'
        )
        return ExecuteObservation(
            output=response.output+response.failure_reason if hasattr(response, 'output') and hasattr(response, 'failure_reason') else str(response),
            exit_code=response.exit_code if hasattr(response, 'exit_code') else -1,
            failure_reason=response.failure_reason if hasattr(response, 'failure_reason') else error_message,
            sandbox_failed_times=self.sandbox_failed_times,
            error_message=error_message
        )

    def upload_file(self, file_path: str, target_path: str, max_execute_retry: int = 3, max_execute_time: float = 360.0):
        """Upload a file to the sandbox"""
        st = time.time()
        execute_time, retry_times = 0, 0
        error_message = ""  # 初始化 error_message，避免未定义错误
        response = ExecuteObservation(output="", exit_code=-1, failure_reason="",sandbox_failed_times=0, error_message="")

        while execute_time < max_execute_time and retry_times < max_execute_retry and (time.time() - self.start_time) < self.clear_time:
            print(f"[SANBOX SDK][UPLOAD FILE]🟢 starting upload file.... retry_times: {retry_times}/{max_execute_retry}, execute_time: {execute_time}/{max_execute_time}, task_idx: {self.task_idx}, sandbox_id: {self.sandbox_id}, file_path: {file_path}, target_path: {target_path}")
            try:
                is_alive_response = asyncio.run(self.sandbox.is_alive())
                if not is_alive_response.is_alive:
                    error_msg = f"sandbox is not alive"
                    elapsed_min = round((time.time() - self.start_time) / 60, 2)
                    print(f"[SANBOX SDK][UPLOAD FILE ERROR]❌ sandbox is not alive, task_idx: {self.task_idx}, elapsed_time: {elapsed_min}min")
                    self.logger.error(f"[SANBOX SDK][UPLOAD FILE ERROR]❌ sandbox is not alive, task_idx: {self.task_idx}")
                    self.sandbox_failed_times += 1
                    return ExecuteObservation(output="sandbox is not alive", exit_code=-1, failure_reason=error_msg, sandbox_failed_times=self.sandbox_failed_times, error_message=error_msg)
            except (Exception, KeyboardInterrupt, SystemExit) as alive_e:
                error_msg = f"Failed to check sandbox status: {repr(alive_e)}"
                elapsed_min = round((time.time() - self.start_time) / 60, 2)
                print(f"[SANBOX SDK][UPLOAD FILE ERROR]❌ Failed to check sandbox status: {repr(alive_e)}, task_idx: {self.task_idx}, elapsed_time: {elapsed_min}min")
                self.logger.error(f"[SANBOX SDK][UPLOAD FILE ERROR]❌ Failed to check sandbox status: {repr(alive_e)}, task_idx: {self.task_idx}")
                self.sandbox_failed_times += 1
                return ExecuteObservation(output="sandbox is not alive", exit_code=-1, failure_reason=error_msg, sandbox_failed_times=self.sandbox_failed_times, error_message=error_msg)
            self.logger.info(
                f"[SANDBOX SDK][UPLOAD FILE]({retry_times}/{max_execute_retry}, execute_time: {execute_time})"
                f"docker_image:{self.docker_image}, sandbox_id:{getattr(self.sandbox, 'sandbox_id', '') if self.sandbox else ''}, file_path:{file_path} target_path: {target_path}, task_idx: {self.task_idx}")
            # 使用 try-except 包装 asyncio.run，确保异常被正确捕获
            try:
                upload_response = asyncio.run(self.sandbox.aupload(str(file_path), target_path))
            except (Exception, KeyboardInterrupt, SystemExit) as async_e:
                # 捕获 asyncio.run 可能抛出的所有异常
                error_message = f"asyncio.run failed: {repr(async_e)}"
                self.logger.error(
                    f"[SANDBOX SDK][UPLOAD FILE ERROR]❌(retry_times:{retry_times}/{max_execute_retry}, execute_time: {execute_time}/{max_execute_time})"
                    f"docker_image:{self.docker_image}, sandbox_id:{getattr(self.sandbox, 'sandbox_id', '') if self.sandbox else ''}, file_path:{file_path}, target_path: {target_path}, error: {error_message}, task_idx: {self.task_idx}"
                )
                elapsed_min = round((time.time() - self.start_time) / 60, 2)
                print(
                    f"[SANDBOX SDK][UPLOAD FILE ERROR]❌retry_times:{retry_times}/{max_execute_retry}, execute_time: {execute_time}/{max_execute_time})"
                    f"docker_image:{self.docker_image}, sandbox_id:{getattr(self.sandbox, 'sandbox_id', '') if self.sandbox else ''}, file_path:{file_path}, target_path: {target_path}, error: {error_message}, task_idx: {self.task_idx}, elapsed_time: {elapsed_min}min"
                )
                # 不重新抛出异常，而是继续重试循环
                self.sandbox_failed_times += 1
                retry_times += 1
                execute_time = round(time.time() - st, 4)
                time.sleep(120)
                continue
            if upload_response.success:
                elapsed_min = round((time.time() - self.start_time) / 60, 2)
                print(
                    f"[SANDBOX SDK][UPLOAD FILE SUCCESS]✅(retry_times:{retry_times}/{max_execute_retry}, execute_time: {execute_time}/{max_execute_time})"
                    f"docker_image:{self.docker_image}, sandbox_id:{getattr(self.sandbox, 'sandbox_id', '') if self.sandbox else ''}, file_path:{file_path}, target_path: {target_path}, message: {upload_response.message}, task_idx: {self.task_idx}, elapsed_time: {elapsed_min}min"
                )
                self.logger.info(
                    f"[SANDBOX SDK][UPLOAD FILE SUCCESS]✅(retry_times:{retry_times}/{max_execute_retry}, execute_time: {execute_time}/{max_execute_time})"
                    f"docker_image:{self.docker_image}, sandbox_id:{getattr(self.sandbox, 'sandbox_id', '') if self.sandbox else ''}, file_path:{file_path}, target_path: {target_path}, message: {upload_response.message}, task_idx: {self.task_idx}"
                )
                return ExecuteObservation(output=upload_response.message, exit_code=0, failure_reason="",sandbox_failed_times=self.sandbox_failed_times, error_message="")
            else:
                error_message = getattr(upload_response, 'message', '') or getattr(upload_response, 'error_message', 'Upload failed')
                elapsed_min = round((time.time() - self.start_time) / 60, 2)
                print(
                    "[SANDBOX SDK][UPLOAD FILE ERROR]❌"
                    f"(retry_times:{retry_times}/{max_execute_retry}, execute_time: {execute_time}/{max_execute_time})"
                    f"docker_image:{self.docker_image}, sandbox_id:{getattr(self.sandbox, 'sandbox_id', '') if self.sandbox else ''}, file_path:{file_path}, target_path: {target_path}, error_message: {error_message}, task_idx: {self.task_idx}, elapsed_time: {elapsed_min}min"
                )
                self.logger.error(
                    f"[SANDBOX SDK][UPLOAD FILE ERROR]❌"
                    f"(retry_times:{retry_times}/{max_execute_retry}, execute_time: {execute_time}/{max_execute_time})"
                    f"docker_image:{self.docker_image}, sandbox_id:{getattr(self.sandbox, 'sandbox_id', '') if self.sandbox else ''}, file_path:{file_path}, target_path: {target_path}, error_message: {error_message}, task_idx: {self.task_idx}"
                )
            self.sandbox_failed_times += 1
            retry_times += 1
            execute_time = round(time.time() - st, 4)
            time.sleep(120)
        
        # 确保 error_message 有值
        if not error_message:
            error_message = "Max retries exceeded or upload failed"
        
        elapsed_min = round((time.time() - self.start_time) / 60, 2)
        print(
            f"[SANDBOX SDK][UPLOAD FILE ERROR]❌(final)(retry_times:{retry_times}/{max_execute_retry}, execute_time: {execute_time}/{max_execute_time})"
            f"docker_image:{self.docker_image}, sandbox_id:{getattr(self.sandbox, 'sandbox_id', '') if self.sandbox else ''}, file_path:{file_path}, target_path: {target_path}, error_message: {error_message}, task_idx: {self.task_idx}, elapsed_time: {elapsed_min}min"
        )
        self.logger.error(
            f"[SANDBOX SDK][UPLOAD FILE ERROR]❌(final)(retry_times:{retry_times}/{max_execute_retry}, execute_time: {execute_time}/{max_execute_time})"
            f"docker_image:{self.docker_image}, sandbox_id:{getattr(self.sandbox, 'sandbox_id', '') if self.sandbox else ''}, file_path:{file_path}, target_path: {target_path}, error_message: {error_message}, task_idx: {self.task_idx}"
        )
        return ExecuteObservation(output="", exit_code=-1, failure_reason=error_message, sandbox_failed_times=self.sandbox_failed_times, error_message=error_message)

    def uplpad_dir(self, dir_path: str, target_path: str, max_execute_retry: int = 3, max_execute_time: float = 360.0):
        """Upload a directory to the sandbox"""
        pass

    def check_alive(self):
        print(f"[SANBOX SDK][CHECK ALIVE]🟢 starting check alive.... task_idx: {self.task_idx}, sandbox_id: {self.sandbox_id}")
        try:
            is_alive_response = asyncio.run(self.sandbox.is_alive())
            if not is_alive_response.is_alive:
                error_msg = f"sandbox is not alive"
                elapsed_min = round((time.time() - self.start_time) / 60, 2)
                print(f"[SANBOX SDK][CHECK ALIVE ERROR]❌ sandbox is not alive, task_idx: {self.task_idx}, elapsed_time: {elapsed_min}min")
                self.logger.error(f"[SANBOX SDK][CHECK ALIVE ERROR]❌ sandbox is not alive, task_idx: {self.task_idx}")
                self.sandbox_failed_times += 1
                return False
            return True
        except Exception as e:
            error_msg = f"Failed to check sandbox status: {repr(e)}"
            elapsed_min = round((time.time() - self.start_time) / 60, 2)
            print(f"[SANBOX SDK][CHECK ALIVE ERROR]❌ Failed to check sandbox status: {repr(e)}, task_idx: {self.task_idx}, elapsed_time: {elapsed_min}min")
            self.logger.error(f"[SANBOX SDK][CHECK ALIVE ERROR]❌ Failed to check sandbox status: {repr(e)}, task_idx: {self.task_idx}")
            self.sandbox_failed_times += 1
            return False

    def create_file(
        self,
        file_path: str,
        content: str,
        max_execute_time: float = 360,
        max_execute_retry: int = 10,
    ):
        st, execute_time, retry_times = time.time(), 0, 0
        timeout, error_message = 180, ""

        execute_obs = ExecuteObservation(output="env_timeout", exit_code=-1, failure_reason="env_timeout", sandbox_failed_times=self.sandbox_failed_times, error_message="env_timeout")

        while retry_times < max_execute_retry and execute_time < max_execute_time and (time.time() - self.start_time) < self.clear_time:
            print(f"[SANBOX SDK][CREATE FILE]🟢 starting create file.... retry_times: {retry_times}/{max_execute_retry}, execute_time: {execute_time}/{max_execute_time}, task_idx: {self.task_idx}, sandbox_id: {self.sandbox_id}, file_path: {file_path}, content: {[content]}")
            try:
                is_alive_response = asyncio.run(self.sandbox.is_alive())
                if not is_alive_response.is_alive:
                    error_msg = f"sandbox is not alive"
                    elapsed_min = round((time.time() - self.start_time) / 60, 2)
                    print(f"[SANBOX SDK][CREATE FILE ERROR]❌ sandbox is not alive, task_idx: {self.task_idx}, elapsed_time: {elapsed_min}min")
                    self.logger.error(f"[SANBOX SDK][CREATE FILE ERROR]❌ sandbox is not alive, task_idx: {self.task_idx}")
                    self.sandbox_failed_times += 1
                    return ExecuteObservation(output="sandbox is not alive", exit_code=-1, failure_reason=error_msg, sandbox_failed_times=self.sandbox_failed_times, error_message=error_msg)
            except (Exception, KeyboardInterrupt, SystemExit) as alive_e:
                error_msg = f"Failed to check sandbox status: {repr(alive_e)}"
                elapsed_min = round((time.time() - self.start_time) / 60, 2)
                print(f"[SANBOX SDK][CREATE FILE ERROR]❌ Failed to check sandbox status: {repr(alive_e)}, task_idx: {self.task_idx}, elapsed_time: {elapsed_min}min")
                self.logger.error(f"[SANBOX SDK][RUN IN SESSION ERROR]❌ Failed to check sandbox status: {repr(alive_e)}, task_idx: {self.task_idx}")
                self.sandbox_failed_times += 1
                return ExecuteObservation(output="sandbox is not alive", exit_code=-1, failure_reason=error_msg, sandbox_failed_times=self.sandbox_failed_times, error_message=error_msg)
            
            if (time.time() - self.start_time) > self.clear_time:
                print(f"[SANBOX SDK][CREATE FILE ERROR]❌ env_timeout, task_idx: {self.task_idx}, elapsed_time: {elapsed_min}min, ")
                return execute_obs

            try:
                # FIX ME: use sdk when sdk support wrire_file
                response = asyncio.run(self.sandbox.write_file(content=content, path=file_path))
                if response.success:
                    response_data = response.message
                    if DEBUG:
                        elapsed_min = round((time.time() - self.start_time) / 60, 2)
                        print("[SANBOX SDK][CREATE FILE SUCCESS]", f"response_data: {response_data}, elapsed_time: {elapsed_min}min")
                    self.logger.info(
                        f"[SANBOX SDK][CREATE FILE SUCCESS]✅"
                        f"(retry_times:{retry_times}/{max_execute_retry}, execute_time: {execute_time}/{max_execute_time})"
                        f"sandbox_id: {self.sandbox_id}, docker_image: {self.docker_image}, file_path: {file_path}, content: {[content]}, task_idx: {self.task_idx}"
                        f"response: {response}"
                    )
                    return ExecuteObservation(output=response.message, exit_code=0, failure_reason="",sandbox_failed_times=self.sandbox_failed_times, error_message="")
                else:
                    error_message = getattr(response, 'message', 'Write file failed')
                    elapsed_min = round((time.time() - self.start_time) / 60, 2)
                    print(
                        "[SANBOX SDK][CREATE FILE ERROR]❌"
                        f"(retry_times:{retry_times}/{max_execute_retry}, execute_time: {execute_time}/{max_execute_time})"
                        f"sandbox_id: {self.sandbox_id}, docker_image: {self.docker_image}, file_path: {file_path}, content: {[content]}, task_idx: {self.task_idx}, elapsed_time: {elapsed_min}min"
                        f"response: {response}"
                    )
                    self.logger.error(
                        f"[SANBOX SDK][CREATE FILE ERROR]❌"
                        f"(retry_times:{retry_times}/{max_execute_retry}, execute_time: {execute_time}/{max_execute_time})"
                        f"sandbox_id: {self.sandbox_id}, docker_image: {self.docker_image}, file_path: {file_path}, content: {[content]}, task_idx: {self.task_idx}"
                        f"response: {response}"
                    )
            except (Exception, KeyboardInterrupt, SystemExit) as async_e:
                # 捕获 asyncio.run 可能抛出的所有异常
                error_message = f"asyncio.run failed: {repr(async_e)}"
                self.logger.error(
                    f"[SANBOX SDK][CREATE FILE ERROR]❌"
                    f"(retry_times:{retry_times}/{max_execute_retry}, execute_time: {execute_time}/{max_execute_time})"
                    f"sandbox_id: {self.sandbox_id}, docker_image: {self.docker_image}, file_path: {file_path}, error: {error_message}, task_idx: {self.task_idx}"
                )
                elapsed_min = round((time.time() - self.start_time) / 60, 2)
                print(
                    f"[SANBOX SDK][CREATE FILE ERROR]❌"
                    f"(retry_times:{retry_times}/{max_execute_retry}, execute_time: {execute_time}/{max_execute_time})"
                    f"sandbox_id: {self.sandbox_id}, docker_image: {self.docker_image}, file_path: {file_path}, error: {error_message}, task_idx: {self.task_idx}, elapsed_time: {elapsed_min}min"
                )
            

            self.sandbox_failed_times += 1
            retry_times += 1
            execute_time = round(time.time() - st, 4)
            time.sleep(120)
        
        # 确保 error_message 有值
        if not error_message:
            error_message = "Max retries exceeded or create file failed"
        
        self.logger.error(
            f"[SANBOX SDK][CREATE FILE ERROR]❌(final)"
            f"(retry_times:{retry_times}/{max_execute_retry}, execute_time: {execute_time}/{max_execute_time})"
            f"sandbox_id: {self.sandbox_id}, docker_image: {self.docker_image}, file_path: {file_path}, content: {[content]}, error_message: {error_message}, task_idx: {self.task_idx}"
        )
        elapsed_min = round((time.time() - self.start_time) / 60, 2)
        print(
            f"[SANBOX SDK][CREATE FILE ERROR]❌(final)(retry_times:{retry_times}/{max_execute_retry}, execute_time: {execute_time}/{max_execute_time})"
            f"sandbox_id: {self.sandbox_id}, docker_image: {self.docker_image}, file_path: {file_path}, error_message: {error_message}, task_idx: {self.task_idx}, elapsed_time: {elapsed_min}min"
        )
        return ExecuteObservation(output="", exit_code=-1, failure_reason=error_message, sandbox_failed_times=self.sandbox_failed_times, error_message=error_message)

    def clean_stdout(self, stdout):
        # 清理空的STDOUT和STDERR标签
        # 删除空的STDOUT标签（如果后面没有内容）
        stdout = re.sub(r"\[STDOUT\]\n\n \n\n\[STDERR\]", "[STDERR]", stdout)
        stdout = re.sub(r"\[STDOUT\]\n\n\[STDERR\]", "[STDERR]", stdout)
        stdout = re.sub(r"\[STDOUT\]\n\n \n", "", stdout)
        stdout = re.sub(r"\[STDOUT\]\n\n", "", stdout)
        # 清理多余的换行符
        stdout = re.sub(r"\n\n\[STDERR\]\n\n", "\n[STDERR]\n", stdout)
        stdout = re.sub(r"\n\n\[STDERR\]", "\n[STDERR]", stdout)
        # 如果STDERR也是空的，删除整个标签
        stdout = re.sub(r"\[STDERR\]\n\n$", "", stdout)
        stdout = re.sub(r"\[STDERR\]\n$", "", stdout)
        return stdout

    def _should_use_nohup(self, command: str) -> bool:
        """
        判断命令是否需要使用 nohup 执行

        Args:
            command: 要执行的命令

        Returns:
            True if the command should use nohup, False otherwise
        """
        # 需要 nohup 的命令模式
        nohup_patterns = [
            # 下载
            r"pip\s+install",
            r"uv\s+pip\s+install",
            r"python\s+-m\s+pip\s+install",
            r"poetry\s+install",
            r"pipenv\s+install",
            r"setup\.py\s+install",
            r"python\s+setup\.py",
            r"conda\s+activate",
            r"source\s+.*activate",
            # 运行单测
            r"run_tests\.sh",
        ]
        command_lower = command.lower().strip()
        for pattern in nohup_patterns:
            if re.search(pattern, command_lower):
                return True
        return False

    def run(
        self, code: str, args: str = "", mode: str = "nohup", timeout: int = None, max_execute_time: int = 60 * 4, max_execute_retry: int = 10, wait_interval: int = 10, response_limited_bytes_in_nohup: int = 1024 * 1024 * 10
    ) -> tuple[str, str]:
        """
        General method to execute code or commands in the container, with a timeout.

        :param code: The code or command to execute.
        :param args: Arguments to pass to the code/script.
        :param workdir: The working directory inside the container (optional).
        :return: A tuple containing (stdout, error_ecode). If no error, error_message is the exit code (str).
            - "124": post timeout
            - "-1": post fialed
            - "0": execute success
            - "1": post success, but get execute error
        """
        if not timeout:
            timeout = self.timeout
        DOCKER_PATH = "/root/.venv/bin:/root/.local/bin:/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        # command = f"timeout {timeout} export PATH={DOCKER_PATH} && {code} {args}"
        command = f"timeout {timeout} {code} {args}"

        # try:
        # with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        # Notice we do NOT set tty=True here
        # future = executor.submit(
        exec_result = self.run_in_session(
            command=command,
            timeout=timeout,
            mode="nohup", # nohup, or normal
            max_execute_time=max_execute_time,
            max_execute_retry=self.max_execute_retry,
            wait_interval=10,
            response_limited_bytes_in_nohup=response_limited_bytes_in_nohup
        )
        if DEBUG:
            elapsed_min = round((time.time() - self.start_time) / 60, 2)
            print("[SANBOX SDK][RUN]exec_result: ", f"{exec_result}, task_idx: {self.task_idx}, elapsed_time: {elapsed_min}min")
        self.logger.info(f"[SANBOX SDK][RUN]exec_result: {exec_result}, task_idx: {self.task_idx}")
        # exec_result = future.result(timeout=timeout)
        # Retrieve output and exit code
        exit_code = exec_result.exit_code
        stdout = exec_result.output

        failure_reason = exec_result.failure_reason
        sandbox_failed_times = exec_result.sandbox_failed_times
        error_message = exec_result.error_message
        clean_stdout = self.clean_stdout(stdout)

        # TODO: 当前sandbox仅0/-1两种退出码，表示sandbox状态。需要具体根据stdout提取命令的退出码。
        # return stdout & return exit_code
        # return_stdout, return_code = "", ""
        # if exit_code == 124:
        #     return_stdout, return_code = f"The command took too long to execute (>{timeout}s)", "-1"
        # # elif exit_code == -100:
        # #     return_stdout, return_code = f"Error: network connection timeout, no response", "-1"
        # elif exit_code != 0 and exit_code != 1:
        #     # output: [{'stdout': '', 'stderr': '/bin/sh: 1: cannot open /parameter: No such file\n', 'exit_code': 2}]  # TODO：这里可以设计不同的reward
        #     return_stdout, return_code = stdout, str(exit_code)
        # else:
        #     # success post: Remove ANSI escape codes and \r characters
        #     return_stdout, return_code = re.sub(r"\x1b\[[0-9;]*m|\r", "", stdout), str(exit_code)
        # if exit_code == 0 and return_stdout.strip() == "":
        #     return_stdout = "success"

        self.logger.info(
            f"[SANBOX执行命令的返回]"
            f"(输入)command: {[command]}, sandbox_id: {self.sandbox_id}, task_idx: {self.task_idx},"
            f"(输出)exit_code: {exit_code}, stdout: {clean_stdout}, failure_reason: {failure_reason}, sandbox_failed_times: {sandbox_failed_times}, error_message: {error_message}"
        )
        return clean_stdout, str(exit_code)