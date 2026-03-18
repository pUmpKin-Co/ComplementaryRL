from typing import Optional, Union, Dict, List, Any
import json
import os
import re
import time
import torch
import requests
import numpy as np
from functools import partial
from tensordict import TensorDict
from roll.configs.worker_config import WorkerConfig
from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.decorator import Dispatch, register
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.strategy.factory import create_strategy
from roll.distributed.strategy.strategy import InferenceStrategy, TrainStrategy
from roll.models.model_providers import default_tokenizer_provider, default_reward_model_provider
from roll.utils.context_managers import state_offload_manger
from roll.utils.prompt import *


def split_list(input_list, n):
    """
    将输入列表按照长度 n 等分为多个子列表
    
    :param input_list: 输入的列表
    :param n: 每个子列表的长度
    :return: 包含多个子列表的列表
    """
    return [input_list[i:i + n] for i in range(0, len(input_list)-n+1)]

def check_repetition(text, n):
    textidict = {}
    splited_text = split_list(input_list=text, n=n)
    for text_i in splited_text:
        texti = tuple(text_i)
        if texti in textidict.keys():
            textidict[texti] += 1
        else:
            textidict[texti] = 1
    if not textidict:
        return 1
    max_repetition_num = max(textidict.values())
    return max_repetition_num


class LLMJudgeRewardWorker(Worker):
    """
    Reward Worker that uses LLM-as-judge to compute rewards.
    """

    def __init__(self, worker_config: WorkerConfig):
        super().__init__(worker_config=worker_config)
        self.rank_info.dp_rank = self.rank_info.rank
        self.rank_info.dp_size = self.rank_info.world_size
        self.tokenizer = None
        self.strategy: Optional[Union[InferenceStrategy, TrainStrategy]] = None

        # LLM judge相关配置
        self.judge_prompt = self.worker_config.judge_prompt if hasattr(self.worker_config, "judge_prompt") else None
        self.judge_prompt = prompt_maps[self.judge_prompt]
        self.judge_model_type = (
            self.worker_config.judge_model_type if hasattr(self.worker_config, "judge_model_type") else "api"
        )
        self.judge_model_name = (
            self.worker_config.judge_model_name if hasattr(self.worker_config, "judge_model_name") else None
        )
        self.judge_api_url = self.worker_config.judge_api_url if hasattr(self.worker_config, "judge_api_url") else None
        self.judge_api_key = self.worker_config.judge_api_key if hasattr(self.worker_config, "judge_api_key") else None
        self.judge_api_location= self.worker_config.judge_api_location if hasattr(self.worker_config, "judge_api_location") else None

        # log the start time with yy-mm-dd HH:MM:SS
        print(f"[DEBUG] Reward Endpoint start time is {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
        self.logger.info(f"Reward Endpoint start time is {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")


    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self, pipeline_config):
        super().initialize(pipeline_config)
        if self.judge_model_type == "api":
            self.tokenizer = default_tokenizer_provider(model_args=self.worker_config.model_args)
            print(f"{self.worker_name} initialized with API model")

        elif self.judge_model_type == "inference":
            self.strategy = create_strategy(worker=self)
            self.strategy.initialize(model_provider=default_reward_model_provider)
            self.tokenizer = self.strategy.tokenizer
            print(f"{self.worker_name} initialized with inference model")
            self.strategy.offload_states()
        else:
            raise ValueError(f"Unsupported model type: {self.judge_model_type}")

    def _call_api_model(self, messages: Dict, retry_times=3) -> str:
        from openai import OpenAI

        ouput = ""
        if not self.judge_api_url or not self.judge_api_key:
            raise ValueError("API URL and API key must be provided for API model type")
        while retry_times > 0:
            retry_times -= 1
            try:
                client = OpenAI(
                    api_key=self.judge_api_key,
                    base_url=self.judge_api_url,
                )
                completion = client.chat.completions.create(model=self.judge_model_name, messages=messages)
                output = completion.choices[0].message.content
                total_tokens = completion.usage.total_tokens
                prompt_token = completion.usage.prompt_tokens
                completion_token = completion.usage.completion_tokens
                token_info = {
                    "total_tokens": total_tokens,
                    "prompt_token": prompt_token,
                    "completion_token": completion_token,
                }
                print(token_info)
                if output != None and output != "":
                    break
            except Exception as e:
                print(e)
                continue
        self.logger.info(f"judge model api output: {str(output)}")
        return output

    def _run_whales_hz(self, messages: Dict) -> str:
        """Example method for calling external API - configure with your own endpoint."""
        api_url = os.environ.get("JUDGE_API_URL", "http://localhost:8000/v1/chat/completions")
        api_key = os.environ.get("JUDGE_API_KEY", "your-api-key")
        auth_token = os.environ.get("JUDGE_AUTH_TOKEN", "your-auth-token")
        headers = {
            'ApiKey': api_key,
            'Authorization': f'Bearer {auth_token}',
            'Content-Type': 'application/json'
        }
        data = {
            'model': os.environ.get("JUDGE_MODEL", 'qwen2.5-7B-Instruct'),
            "top_p": 0.8,
            "top_k": 1,
            "temperature": 0.8,
            "max_tokens": 200,
            'messages': messages
        }
        res = requests.post(api_url, json=data, headers=headers)
        res = json.loads(res.text)
        valid_response=res["choices"][0]["message"]["content"]
        return valid_response
    
    def _run_whales_zhangjiakou(self, messages: Dict) -> str:
        """Example method for calling external API - configure with your own endpoint."""
        api_url = os.environ.get("JUDGE_API_URL_ZJK", "http://localhost:8000/v1/chat/completions")
        api_key = os.environ.get("JUDGE_API_KEY", "your-api-key")
        auth_token = os.environ.get("JUDGE_AUTH_TOKEN", "your-auth-token")
        headers = {
            'ApiKey': api_key,
            'Authorization': f'Bearer {auth_token}',
            'Content-Type': 'application/json'
        }
        data = {
            'model': os.environ.get("JUDGE_MODEL", 'qwen2.5-7B-Instruct'),
            "top_p": 0.8,
            "top_k": 1,
            "temperature": 0.8,
            "max_tokens": 200,
            'messages': messages
        }
        res = requests.post(api_url, json=data, headers=headers)
        res = json.loads(res.text)
        valid_response=res["choices"][0]["message"]["content"]
        return valid_response

    def _run_whales_production(self, messages: Dict) -> str:
        """Example method for calling external API - configure with your own endpoint."""
        address = os.environ.get("JUDGE_API_PROD_URL", "https://your-api-endpoint.com/v1/chat/completions")
        api_key = os.environ.get("JUDGE_API_KEY", "your-api-key")

        headers={"Content-Type": "application/json",
                "Authorization": f"bearer {api_key}"}

        function_request = {
            "model": os.environ.get("JUDGE_MODEL", 'qwen2.5-7B-Instruct'),
            "top_p": 0.8,
            "top_k": 1,
            "temperature": 0.8,
            "max_tokens": 200,
            "messages": messages,
        }
        res = requests.post(f"{address}", json=function_request, headers=headers)
        res = json.loads(res.text)
        valid_response = res["choices"][0]["message"]["content"]
        return valid_response
    
    def _run_local_inference(self, messages: Dict) -> str:
        if not self.strategy:
            raise ValueError("Strategy not initialized for local inference")
        
        from roll.third_party.llmtuner.data import get_template_and_fix_tokenizer
        
        template_name = self.worker_config.data_args.template
        template = get_template_and_fix_tokenizer(self.tokenizer, name=template_name, processor=None)
        # template.default_system=""
        input_ids = template._encode(self.tokenizer, messages, None, None)
        input_ids_tensor = torch.tensor(input_ids, device='cuda')
        # self.logger.info(f"qiyang check input_ids_tensor shape is {input_ids_tensor.shape}")
        # self.logger.info(f"qiyang start checking")
        check_response = self.tokenizer.decode(input_ids_tensor[0], skip_special_tokens=False)
        logging_dict={}
        logging_dict["original_message"]=messages[0]["content"]
        logging_dict["check_response"] =check_response
        # self.logger.info(f"qiyang finish checking")

        generation_config = self.worker_config.generating_args.to_dict()
        generation_config["eos_token_id"] = [self.tokenizer.eos_token_id] + self.tokenizer.additional_special_tokens_ids
        generation_config["eos_token_id"] = [self.tokenizer.eos_token_id]
        generation_config["pad_token_id"] = self.tokenizer.pad_token_id

        attention_mask = torch.ones_like(input_ids_tensor, dtype=torch.long, device='cuda')
        position_ids = torch.clip(torch.cumsum(attention_mask, dim=-1) - 1, min=0, max=None)

        data = DataProto(batch=TensorDict({"input_ids": input_ids_tensor, "attention_mask": attention_mask, "position_ids": position_ids}, batch_size=input_ids_tensor.shape[0]))
        data = data.to('cuda')
        data.meta_info = {'micro_batch_size': self.worker_config.infer_batch_size}

        with torch.no_grad():
            output = self.strategy.generate(batch=data, generation_config=generation_config)
            print("output shape:", output.shape)
            if isinstance(output, torch.Tensor):
                generate_ids = output[:, len(input_ids[0]):]
            else:
                generate_ids = output.batch["input_ids"][:, len(input_ids[0]):]
        # self.logger.info(f"generate_ids shape is {generate_ids.shape}")

        print("There is no error here")

        # response_log_probs = results['log_probs']
        response = self.tokenizer.decode(generate_ids[0], skip_special_tokens=True)
        self.logger.info(f"response is {response}")
        logging_dict["response"]=response
        # self.logger.info(f"qiyang check logging dict in inference is {logging_dict}")
        print("judge model inference output: ", response)
        # print("judge model inference output probs: ", response_log_probs)
        return  response.strip()

    # def _run_local_inference(self, messages: Dict) -> str:
    #     if not self.strategy:
    #         raise ValueError("Strategy not initialized for local inference")

    #     template_name = self.worker_config.data_args.template
    #     chat_template_func = get_chat_template(template_name, self.tokenizer)
    #     text = chat_template_func(messages)

    #     tokenized = self.tokenizer(text, return_tensors="pt")
    #     input_ids = tokenized["input_ids"].to("cuda")
    #     attention_mask = tokenized["attention_mask"].to("cuda")

    #     generation_config = self.worker_config.generating_args.to_dict()
    #     generation_config["eos_token_id"] = [self.tokenizer.eos_token_id]
    #     generation_config["pad_token_id"] = self.tokenizer.pad_token_id

    #     data = DataProto(
    #         batch=TensorDict({"input_ids": input_ids, "attention_mask": attention_mask}, batch_size=input_ids.shape[0])
    #     )
    #     data = data.to("cuda")
    #     data.meta_info = {"micro_batch_size": self.worker_config.infer_batch_size}

    #     with torch.no_grad():
    #         output = self.strategy.generate(batch=data, generation_config=generation_config)
    #         if isinstance(output, torch.Tensor):
    #             generate_ids = output[:, len(input_ids[0]) :]
    #         else:
    #             generate_ids = output.batch["input_ids"][:, len(input_ids[0]) :]

    #     output = self.tokenizer.decode(generate_ids[0], skip_special_tokens=True)
    #     self.logger.info(f"judge model inference output: {str(output)}")
    #     return output.strip()

    def _extract_score(self, response: str) -> float:
        try:
            match = re.search("Score: ([0-9.]+)", response)
            if match:
                score = float(match.group(1))
                normalized_score = score / 10
                return normalized_score
            else:
                self.logger.warning(f"Could not extract score from response: {response}")
                return 0.5
        except Exception as e:
            self.logger.error(f"Error extracting score: {e}")
            return 0.5

    def _extract_score_v2(self, response: str) -> float:
        response = response.lower()
        try:
            if "yes" in response:
                return 1
            elif "no" in response:
                return 0
            else:
                self.logger.warning(f"Could not extract score from response: {response}")
                return 0
        except Exception as e:
            self.logger.error(f"Error extracting score: {e}")
            return 0
        
    def match_and_convert(self, s):
        # 正则表达式匹配开头的数字，包括整数和小数
        match = re.match(r"^(\d*\.\d+|\d+)", s)
        if match:
            # 提取匹配到的数字并转换为浮点数
            number = float(match.group(1))

            return number
        else:
            # 如果没有匹配到数字，返回None或者抛出异常
            return None

    def extract_boxed_content3(self,text): #提前最后一个boxed, 且考虑多层嵌套结果
        last_match = None
        i = 0
        n = len(text)
        boxed_start = r"\boxed{"
        boxed_len = len(boxed_start)
        format_flag=False

        while i < n:
            # 查找 \boxed{ 的起始位置
            if text[i:i+boxed_len] == boxed_start:
                start = i + boxed_len
                depth = 1
                j = start

                # 遍历直到匹配最外层闭合的 }
                while j < n and depth > 0:
                    if text[j] == '{':
                        depth += 1
                    elif text[j] == '}':
                        depth -= 1
                    j += 1

                if depth == 0:
                    # 更新为最后一个匹配的内容
                    last_match = text[start:j-1]
                    i = j  # 跳过已处理部分
                    continue
            i += 1
        if last_match:
            format_flag=True
            return last_match,format_flag
        else:
            return "",format_flag
        

    def customize_judge_prompt(self, prompt: str, response: str, reference: str = None) -> str:
        my_prompt = '''
        # Overview

        Evaluate the accuracy of the model-generated answer base on the given Question. The response should mainly be reasonable in the setting of the Question. The response should also align with the reference answer, cover key details, and avoid speculative or fabricated claims. 

        Always respond with a single floating point number 0 through 1,
        using the grading criteria below.

        ## Grading Criteria:
        - **1.0**: The model answer is fully aligned with the reference and factually correct.
        - **0.75**: The model answer is mostly correct but has minor omissions or slight rewording that does not change meaning.
        - **0.5**: The model answer is partially correct but lacks key details or contains speculative statements.
        - **0.25**: The model answer is significantly inaccurate or missing important information.
        - **0.0**: The model answer is completely incorrect, hallucinates details, or is irrelevant.



        Question : {question}
        Reference Answer: {reference}
        Model Answer: {response}
        '''
        boxed_response,format_flag=self.extract_boxed_content3(response)
        new_prompt=my_prompt.format(
                question=prompt,
                response=boxed_response,
                reference = reference
            )

        # self.logger.info(f"qiyang check new prompt is {new_prompt}")

        messages=[
           {"role": "user", "content": new_prompt}
         ]
        
        return messages,format_flag

    def _get_llm_judgment(self, prompt_id: str, prompt: str, response: str, reference: str = None) -> float:
        messages,format_flag = self.customize_judge_prompt(prompt, response, reference)
        logging_dict={}
        logging_dict["prompt_id"] = prompt_id
        logging_dict["customized_prompt"]=messages[0]["content"]
        logging_dict["format_values"]=format_flag
        logging_dict["original_response"]=response

        if self.judge_model_type == "api":
            if self.judge_api_location == "hz":
                llm_response = self._run_whales_hz(messages)
            elif self.judge_api_location == "zjk":
                llm_response = self._run_whales_zhangjiakou(messages)
            else:
                llm_response = self._run_whales_production(messages)
        elif self.judge_model_type == "inference":
            llm_response = self._run_local_inference(messages)
        else:
            raise ValueError(f"Unsupported model type: {self.judge_model_type}")
        
        logging_dict["llm_response"] = llm_response
        llm_reward = self.match_and_convert(llm_response)
        logging_dict["llm_reward"] = llm_reward
        # self.logger.info(f"qiyang check llm_reward is {llm_reward}")
        if not llm_reward:
            llm_reward = 0.0
        self.logger.info(f"logging dict is {logging_dict}")
        format_value = 1 if format_flag else 0
        return llm_reward, format_value

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE)
    def compute_rewards(self, data: DataProto):
        is_offload_states = data.meta_info.get("is_offload_states", True)
        metrics = {}

        if self.judge_model_type == "inference" and self.strategy:
            with state_offload_manger(
                strategy=self.strategy,
                metrics=metrics,
                metric_infix=f"{self.cluster_name}/compute_rewards",
                is_offload_states=is_offload_states,
            ):
                return self._compute_rewards_impl(data, metrics)
        else:
            return self._compute_rewards_impl(data, metrics)

    def _compute_rewards_impl(self, data: DataProto, metrics: Dict):
        prompts_text_list = self.tokenizer.batch_decode(data.batch["prompts"], skip_special_tokens=True)
        response_text_list = self.tokenizer.batch_decode(data.batch["responses"], skip_special_tokens=False)

        scores = []
        total_reward=[]
        format_values=[]
        chenghuan_values=[]
        for prompt_id, prompt_txt, response, reference in zip(
            data.non_tensor_batch["id"], prompts_text_list, response_text_list, data.non_tensor_batch["ground_truth"]
        ):
            try:
                print(f"Computing reward for prompt_id: {prompt_id}")
                resp_text_without_sptoken = response.replace("<|endoftext|>", "").replace("<pad>", "").replace("<|im_end|>", "")
                llm_reward, format_value = self._get_llm_judgment(prompt_id, prompt_txt, resp_text_without_sptoken, reference)
                resp_tokens_without_sptoken =self.tokenizer(resp_text_without_sptoken)["input_ids"]
                nums_repetitions=check_repetition(resp_tokens_without_sptoken,15)
                repetition_penalty=0
                if nums_repetitions>5:
                    repetition_penalty=-0.3
                chenghuan_value = 0 if repetition_penalty==0 else 1
                score= 1.0 if llm_reward>=0.5 else 0
                format_values.append(format_value)
                chenghuan_values.append(chenghuan_value)
                scores.append(score)
                total_reward.append(llm_reward+repetition_penalty)
                print(f"Reward of prompt_id {prompt_id}: {score}")
            except Exception as e:
                self.logger.error(f"Error computing reward for prompt_id {prompt_id}: {e}")
                scores.append(0)
                total_reward.append(0)
                chenghuan_values.append(0)
                format_values.append(0)    

        scores_tensor = torch.tensor(scores, dtype=torch.float16)
        token_level_rewards = torch.zeros_like(data.batch["responses"], dtype=torch.float16)
        response_level_rewards = torch.tensor(total_reward, dtype=torch.float16)
        repetition_penalty_rewards = torch.zeros_like(scores_tensor, dtype=torch.float16)
        response_length_rewards = torch.zeros_like(scores_tensor, dtype=torch.float16)
        format_values = torch.tensor(format_values, dtype=torch.float16)
        correct_values = scores_tensor
        chenghuan_values = torch.tensor(chenghuan_values, dtype=torch.float16)
        
        output = DataProto.from_dict(
            tensors={
                "token_level_rewards": token_level_rewards,
                "response_level_rewards": response_level_rewards,
                "scores": scores_tensor,
                'repetition_penalty_rewards': repetition_penalty_rewards,
                'response_length_rewards': response_length_rewards,
                'format_values': format_values,
                'correct_values': correct_values,
                'chenghuan_values':chenghuan_values
            }
        )
        output.meta_info = {"metrics": metrics}
        print(f"Computed rewards for {len(scores)} samples")
        return output
