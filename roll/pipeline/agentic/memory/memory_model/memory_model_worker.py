import threading
import time
from typing import Dict

import numpy as np
import openai
import ray
import torch
from tqdm import tqdm

from roll.configs.worker_config import WorkerConfig
from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.decorator import Dispatch, register
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.strategy.factory import create_strategy
from roll.models.model_providers import default_actor_model_provider
from roll.pipeline.agentic.memory.memory_config import MemoryModelConfig, MemoryModelType, MemoryType
from roll.pipeline.base_worker import ActorWorker
from roll.utils.checkpoint_manager import download_model
from roll.utils.context_managers import state_offload_manger
from roll.utils.functionals import GenerateRequestType, agg_loss, append_to_dict, compute_approx_kl, masked_mean
from roll.utils.logging import get_logger
from roll.utils.offload_states import OffloadStateType

logger = get_logger()


def get_memory_scores(batch: DataProto) -> torch.Tensor:
    # Group by request_id (string), then take one score per request_id group.
    # NOTE: This requires `non_tensor_batch` to be present on every Megatron rank.
    # The pipeline should set `batch.meta_info["_broadcast_non_tensor_batch"] = True`.
    batch_group_by_traj: Dict[str, DataProto] = batch.group_by(keys="request_id")
    scores = []
    for _, traj_batch in batch_group_by_traj.items():
        v = traj_batch.non_tensor_batch["memory_scores"][0]
        if isinstance(v, (list, tuple, np.ndarray)):
            scores.append(float(v[0]) if len(v) > 0 else 0.0)
        else:
            scores.append(float(v))
    return torch.tensor(scores, dtype=torch.float32)


class MemoryModelWorker(Worker):
    """
    Memory model worker for inference memory model.
    """

    def __init__(self, worker_config: MemoryModelConfig):
        super().__init__(worker_config=worker_config)
        self.model_name = self.worker_config.model_args.model_name_or_path
        self.model = None
        self.tokenizer = None
        self.thread_server = None
        self.response_call_back_fns = {}
        self.response_callback_refs = []
        self.server_metrics = {}
        self.offload_manager = None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self):
        if self.worker_config.memory_model_type == MemoryModelType.api_model:
            self.model = openai.OpenAI(
                api_key=self.worker_config.api_key,
                base_url=self.worker_config.api_url,
            )
        elif self.worker_config.memory_model_type == MemoryModelType.local_model:
            model_name = self.worker_config.model_args.model_name_or_path
            if model_name:
                self.worker_config.model_args.model_name_or_path = download_model(model_name)

            if self.worker_config.resume_from_checkpoint:
                self.logger.info(f"resume_from_checkpoint: {self.worker_config.resume_from_checkpoint}")

            self.model = create_strategy(worker=self)
            self.model.initialize(model_provider=default_actor_model_provider)
            self.tokenizer = self.model.tokenizer
            self.strategy = self.model  # backward compatibility

            self.model.offload_states()
            torch.cuda.init()
        else:
            raise ValueError(f"Unsupported memory model type: {self.worker_config.memory_model_type}")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL_ONE)
    @torch.no_grad()
    def start_server(self, data: DataProto):
        # API-based memory model does not run a local inference server.
        if self.worker_config.memory_model_type != MemoryModelType.local_model:
            return

        if self.thread_server is not None:
            return

        global_step = data.meta_info.get("global_step", 0)
        is_offload_states = data.meta_info.get("is_offload_states", True)
        self.logger.info(f"{self.worker_name} generate server global step {global_step}")

        self.response_call_back_fns = {}
        self.response_callback_refs = []
        self.server_metrics = {}

        self.offload_manager = state_offload_manger(
            strategy=self.strategy,
            metrics=self.server_metrics,
            metric_infix=f"{self.cluster_name}/generate",
            is_offload_states=is_offload_states,
            load_kwargs={"include": [OffloadStateType.model_params]},
        )
        self.offload_manager.__enter__()
        self.thread_server = threading.Thread(
            target=self.model.start_server, kwargs=dict(data=data, request_complete_callback=self.request_complete)
        )
        self.thread_server.start()
        while not self.model.running:
            time.sleep(0.1)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL_ONE)
    def stop_server(self, data: DataProto = None):
        if self.worker_config.memory_model_type != MemoryModelType.local_model:
            return

        if self.thread_server == None:
            return

        self.model.add_request(command=GenerateRequestType.STOP, data=None)
        self.thread_server.join()
        self.thread_server = None
        self.response_call_back_fns.clear()
        if self.offload_manager is not None:
            self.offload_manager.__exit__(None, None, None)
            self.offload_manager = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
        ray.get(self.response_callback_refs)
        self.response_callback_refs.clear()

        return DataProto(meta_info={"metrics": self.server_metrics})

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, clear_cache=False)
    def add_request(self, command, data: DataProto):
        if command == GenerateRequestType.ALIVE_CHECK:
            if self.thread_server is not None:
                if not self.thread_server.is_alive():
                    raise Exception("thread server has stopped unexpectedly. check stderr for more info.")
            output = DataProto(meta_info={"request_counts": len(self.response_call_back_fns)})
            return output
        elif command == GenerateRequestType.ADD:
            assert "response_callback_fn" in data.meta_info, "response_callback_fn is not in data.meta_info"
            is_num_return_sequences_expand = data.meta_info.get("is_num_return_sequences_expand", False)
            if "generation_config" not in data.meta_info:
                generation_config = self.worker_config.generating_args.to_dict()
                if is_num_return_sequences_expand:
                    self.worker_config.generating_args.num_return_sequences = 1
                    generation_config["num_return_sequences"] = 1
                    self.logger.info(f"is_num_return_sequences_expand is True, set num_return_sequences to 1.")
            else:
                generation_config = data.meta_info["generation_config"]
            generation_config["eos_token_id"] = [
                self.tokenizer.eos_token_id
            ] + self.tokenizer.additional_special_tokens_ids
            generation_config["pad_token_id"] = self.tokenizer.pad_token_id
            data.meta_info["generation_config"] = generation_config
            self.response_call_back_fns[data.meta_info["request_id"]] = data.meta_info.pop("response_callback_fn")
        self.model.add_request(command=command, data=data)
        return DataProto(meta_info={"request_counts": len(self.response_call_back_fns)})

    def request_complete(self, data: DataProto):
        """Called when vllm/sglang completes a generation request."""
        data.meta_info["eos_token_id"] = self.tokenizer.eos_token_id
        data.meta_info["pad_token_id"] = self.tokenizer.pad_token_id
        response_call_back_fn = self.response_call_back_fns.pop(data.meta_info["request_id"])
        self.response_callback_refs.append(response_call_back_fn(data))

    @register(dispatch_mode=Dispatch.ONE_TO_ALL_ONE)
    def generate_with_memory_model(self, data: DataProto):
        if self.worker_config.memory_model_type == MemoryModelType.api_model:
            messages = data.meta_info.get("messages", [])
            generation_config = data.meta_info.get("generation_config", {})

            response = self.model.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=generation_config.get("temperature", 0.7),
                max_tokens=generation_config.get("max_new_tokens", 65536),
                top_p=generation_config.get("top_p", 1.0),
            )

            output_text = response.choices[0].message.content

            output = DataProto()
            output.meta_info["response_text"] = output_text
            output.meta_info["model"] = self.model_name
            return output

        elif self.worker_config.memory_model_type == MemoryModelType.local_model:
            raise ValueError(
                "For local_model, use add_request() with async callback pattern, " "not generate_with_memory_model()"
            )


class MemoryActorWorker(ActorWorker):
    """
    Memory Actor Worker for training memory models with separate config parameters.
    Uses memory-specific hyperparameters from MemoryActorConfig (memory_pg_clip, memory_loss_agg_mode, etc.)
    """

    def __init__(self, worker_config: WorkerConfig):
        super().__init__(worker_config=worker_config)
        self.is_memory_model = True
        self._topr_sample_logged = False

    @register(dispatch_mode=Dispatch.DP_MP_DISPATCH_FIRST)
    def train_step(self, data: DataProto):
        """
        return DataProto(meta_info={'metrics': metrics})
        """
        global_step = data.meta_info.get("global_step", 0)
        is_offload_states = data.meta_info.get("is_offload_states", True)
        metrics = {}
        self.logger.info(f"{self.worker_name} generate global step {global_step}")

        with state_offload_manger(
            strategy=self.strategy,
            metrics=metrics,
            metric_infix=f"{self.cluster_name}/train_step",
            is_offload_states=is_offload_states,
            load_kwargs={"include": [OffloadStateType.model_params, OffloadStateType.other_params]},
        ):
            data = data.to("cuda")
            data = self.strategy.get_data_input(data)
            per_device_train_batch_size = self.worker_config.training_args.per_device_train_batch_size
            backward_batch_size = (
                per_device_train_batch_size * self.worker_config.training_args.gradient_accumulation_steps
            )

            dataloader = data.make_iterator(
                mini_batch_size=backward_batch_size,
                epochs=self.pipeline_config.memory_ppo_epochs,
                seed=self.pipeline_config.seed,
                dataloader_kwargs={"shuffle": True},
            )

            for batch_idx, data in tqdm(
                enumerate(dataloader),
                desc=f"{self.worker_name} train global step {global_step}",
                total=data.batch.batch_size[0] * self.pipeline_config.memory_ppo_epochs // backward_batch_size,
            ):
                pg_metrics = self.strategy.train_step(batch=data, loss_func=self.loss_func)
                append_to_dict(metrics, pg_metrics)

            lr_metric_key = "memory_actor/lr" if self.is_memory_model else "actor/lr"
            metrics[lr_metric_key] = self.strategy.scheduler.get_last_lr()[0]
            data.to("cpu")

        output = DataProto(meta_info={"metrics": metrics})
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self, pipeline_config):
        super().initialize(pipeline_config)
        if hasattr(self.pipeline_config, "memory_sequence_length"):
            self.strategy.seq_length = self.pipeline_config.memory_sequence_length
        else:
            self.strategy.seq_length = self.pipeline_config.sequence_length

    def loss_func(self, data: DataProto, output_tensor: torch.Tensor):
        """
        Memory actor loss function using memory-specific config parameters.
        Uses memory_pg_clip, memory_loss_agg_mode, memory_dual_clip_loss, etc. from pipeline_config.

        Args:
            data: DataProto containing batch data with response_mask, old_log_probs, advantages, etc.
            output_tensor: Model forward output logits tensor

        Returns:
            total_loss: Computed loss for backpropagation
            pg_metrics: Dictionary of training metrics
        """
        response_mask = data.batch["response_mask"][:, 1:].long()
        ref_log_probs = data.batch.get("ref_log_probs", None)
        old_log_probs = data.batch["old_log_probs"]
        advantages = data.batch["advantages"]

        log_probs = self.strategy.op_compute_log_probs(
            logits=output_tensor, input_ids=data.batch["input_ids"], attention_mask=data.batch["response_mask"]
        )

        pg_variant = self.pipeline_config.memory_pg_variant or "vanilla"
        ratio = (log_probs - old_log_probs).exp()

        if pg_variant in {"vanilla", "ppo", None}:
            # Use memory-specific clipping parameters
            pg_clip_low = (
                self.pipeline_config.memory_pg_clip_low
                if self.pipeline_config.memory_use_pg_clip_range
                else self.pipeline_config.memory_pg_clip
            )
            pg_clip_high = (
                self.pipeline_config.memory_pg_clip_high
                if self.pipeline_config.memory_use_pg_clip_range
                else self.pipeline_config.memory_pg_clip
            )

            surr1 = ratio * advantages
            surr2 = ratio.clamp(1 - pg_clip_low, 1 + pg_clip_high) * advantages
            pg_loss = -torch.min(surr1, surr2)

            # Use memory-specific dual clip loss setting
            if self.pipeline_config.memory_dual_clip_loss:
                dual_clip_loss = -torch.max(-pg_loss, (1 + self.pipeline_config.memory_pg_clip * 2) * advantages)
                pg_loss = torch.where(advantages < 0, dual_clip_loss, pg_loss)

            clipped_low = (ratio < 1 - pg_clip_low).float()
            clipped_high = (ratio > 1 + pg_clip_high).float()
            clipped = (clipped_low + clipped_high).float()

            # Memory-specific metrics
            pg_metrics = {
                "memory_actor/ppo_ratio_high_clipfrac": clipped_high.mean().detach().item(),
                "memory_actor/ppo_ratio_low_clipfrac": clipped_low.mean().detach().item(),
                "memory_actor/ppo_ratio_clipfrac": clipped.mean().detach().item(),
                "memory_actor/clipfrac": agg_loss(
                    loss_mat=torch.lt(surr2, surr1).float(),
                    loss_mask=response_mask,
                    loss_agg_mode=self.pipeline_config.memory_loss_agg_mode,
                )
                .detach()
                .item(),
            }

        elif pg_variant == "tis":
            tis_lower_bound = float(getattr(self.pipeline_config, "memory_tis_lower_bound", 0.0))
            tis_upper_bound = float(getattr(self.pipeline_config, "memory_tis_upper_bound", 1.0))

            clipped_ratio = torch.clamp(ratio, min=tis_lower_bound, max=tis_upper_bound)
            pg_loss = -(clipped_ratio.detach() * advantages * log_probs)

            lower_clipped = (ratio < tis_lower_bound).float()
            upper_clipped = (ratio > tis_upper_bound).float()
            total_clipped = lower_clipped + upper_clipped

            pg_metrics = {
                "memory_actor/tis_lower_bound": tis_lower_bound,
                "memory_actor/tis_upper_bound": tis_upper_bound,
                "memory_actor/tis_lower_clipfrac": masked_mean(lower_clipped, response_mask).detach().item(),
                "memory_actor/tis_upper_clipfrac": masked_mean(upper_clipped, response_mask).detach().item(),
                "memory_actor/tis_total_clipfrac": masked_mean(total_clipped, response_mask).detach().item(),
                "memory_actor/tis_clipped_ratio_mean": masked_mean(clipped_ratio, response_mask).detach().item(),
            }

        elif pg_variant == "topr":
            # Use memory-specific TOPR loss
            topr_positive_weight = self.pipeline_config.memory_topr_positive_weight
            topr_negative_weight = self.pipeline_config.memory_topr_negative_weight

            scores = get_memory_scores(data)
            positive_mask = (scores > 0).float().to(log_probs.device)
            negative_mask = (scores <= 0).float().to(log_probs.device)

            if not self._topr_sample_logged:
                total_samples = len(scores)
                positive_count = positive_mask.sum().item()
                negative_count = negative_mask.sum().item()
                self.logger.info(
                    f"TOPR样本分布 - 总样本: {total_samples}, 正样本: {positive_count} ({positive_count/total_samples*100:.1f}%), 负样本: {negative_count} ({negative_count/total_samples*100:.1f}%)"
                )
                self.logger.info(
                    f"TOPR奖励统计 - 平均: {scores.mean().item():.4f}, 标准差: {scores.std().item():.4f}, 最大: {scores.max().item():.4f}, 最小: {scores.min().item():.4f}"
                )
                self.logger.info(
                    f"TOPR权重配置 - 正样本权重: {topr_positive_weight}, 负样本权重: {topr_negative_weight}"
                )
                self._topr_sample_logged = True

            positive_token_mask = positive_mask.unsqueeze(-1).expand_as(log_probs)
            negative_token_mask = negative_mask.unsqueeze(-1).expand_as(log_probs)

            positive_loss = -advantages * log_probs * positive_token_mask

            clipped_ratio = torch.clamp(ratio, min=0.0, max=1.0).detach()
            negative_loss = -clipped_ratio * advantages * log_probs * negative_token_mask

            weighted_positive_loss = topr_positive_weight * positive_loss
            weighted_negative_loss = topr_negative_weight * negative_loss

            pg_loss = weighted_positive_loss + weighted_negative_loss
            negative_lower_clipped = ((ratio < 0.0) & (negative_token_mask > 0)).float()
            negative_upper_clipped = ((ratio > 1.0) & (negative_token_mask > 0)).float()
            negative_total_clipped = negative_lower_clipped + negative_upper_clipped

            pg_metrics = {
                # "memory_actor/topr_positive_loss": positive_loss.cpu().detach().item(),
                "memory_actor/topr_positive_loss": agg_loss(
                    loss_mat=positive_loss,
                    loss_mask=response_mask,
                    loss_agg_mode=self.pipeline_config.memory_loss_agg_mode,
                ).item(),
                # "memory_actor/topr_negative_loss": negative_loss.cpu().detach().item(),
                "memory_actor/topr_negative_loss": agg_loss(
                    loss_mat=negative_loss,
                    loss_mask=response_mask,
                    loss_agg_mode=self.pipeline_config.memory_loss_agg_mode,
                ).item(),
                # "memory_actor/topr_weighted_positive_loss": weighted_positive_loss,
                "memory_actor/topr_weighted_positive_loss": agg_loss(
                    loss_mat=weighted_positive_loss,
                    loss_mask=response_mask,
                    loss_agg_mode=self.pipeline_config.memory_loss_agg_mode,
                ).item(),
                # "memory_actor/topr_weighted_negative_loss": weighted_negative_loss.cpu().detach().item(),
                "memory_actor/topr_weighted_negative_loss": agg_loss(
                    loss_mat=weighted_negative_loss,
                    loss_mask=response_mask,
                    loss_agg_mode=self.pipeline_config.memory_loss_agg_mode,
                ).item(),
                "memory_actor/topr_positive_samples": positive_mask.sum().detach().item(),
                "memory_actor/topr_negative_samples": negative_mask.sum().detach().item(),
                "memory_actor/topr_positive_ratio": (positive_mask.sum() / (positive_mask.size(0) + 1e-8))
                .detach()
                .item(),
                "memory_actor/topr_negative_ratio": (negative_mask.sum() / (negative_mask.size(0) + 1e-8))
                .detach()
                .item(),
                "memory_actor/topr_negative_lower_clipfrac": negative_lower_clipped.mean().detach().item(),
                "memory_actor/topr_negative_upper_clipfrac": negative_upper_clipped.mean().detach().item(),
                "memory_actor/topr_negative_total_clipfrac": negative_total_clipped.mean().detach().item(),
                "memory_actor/topr_scores_mean": scores.mean().detach().item(),
                "memory_actor/topr_scores_std": scores.std().detach().item(),
            }

        elif pg_variant == "cispo":
            epsilon_low = float(getattr(self.pipeline_config, "memory_cispo_epsilon_low", 0.1))
            epsilon_high = float(getattr(self.pipeline_config, "memory_cispo_epsilon_high", 0.1))
            use_unified_mask = bool(getattr(self.pipeline_config, "memory_cispo_use_unified_mask", False))

            clip_lower = 1.0 - epsilon_low
            clip_upper = 1.0 + epsilon_high

            if not self._cispo_config_logged:
                self.logger.info(
                    f"[memory_actor] CISPO config - epsilon_low: {epsilon_low}, epsilon_high: {epsilon_high}"
                )
                self.logger.info(f"[memory_actor] CISPO clip range: [{clip_lower:.3f}, {clip_upper:.3f}]")
                self.logger.info(f"[memory_actor] CISPO unified mask enabled: {use_unified_mask}")
                self._cispo_config_logged = True

            clipped_ratio = torch.clamp(ratio, min=clip_lower, max=clip_upper)
            lower_clipped = (ratio < clip_lower).float()
            upper_clipped = (ratio > clip_upper).float()
            total_clipped = lower_clipped + upper_clipped

            if use_unified_mask:
                positive_advantages = advantages > 0
                negative_advantages = advantages < 0
                mask_positive = positive_advantages & (ratio > clip_upper)
                mask_negative = negative_advantages & (ratio < clip_lower)
                token_mask = ~(mask_positive | mask_negative)
                pg_loss = -(clipped_ratio.detach() * advantages * log_probs * token_mask.float())
            else:
                mask_positive = None
                mask_negative = None
                token_mask = None
                pg_loss = -(clipped_ratio.detach() * advantages * log_probs)

            pg_metrics = {
                "memory_actor/cispo_epsilon_low": epsilon_low,
                "memory_actor/cispo_epsilon_high": epsilon_high,
                "memory_actor/cispo_clip_lower": clip_lower,
                "memory_actor/cispo_clip_upper": clip_upper,
                "memory_actor/cispo_use_unified_mask": float(use_unified_mask),
                "memory_actor/cispo_lower_clipfrac": masked_mean(lower_clipped, response_mask).detach().item(),
                "memory_actor/cispo_upper_clipfrac": masked_mean(upper_clipped, response_mask).detach().item(),
                "memory_actor/cispo_total_clipfrac": masked_mean(total_clipped, response_mask).detach().item(),
                "memory_actor/cispo_clipped_ratio_mean": masked_mean(clipped_ratio, response_mask).detach().item(),
            }
            if use_unified_mask:
                pg_metrics.update(
                    {
                        "memory_actor/cispo_masked_positive_tokens": masked_mean(mask_positive.float(), response_mask)
                        .detach()
                        .item(),
                        "memory_actor/cispo_masked_negative_tokens": masked_mean(mask_negative.float(), response_mask)
                        .detach()
                        .item(),
                        "memory_actor/cispo_kept_tokens": masked_mean(token_mask.float(), response_mask)
                        .detach()
                        .item(),
                    }
                )

        else:
            raise ValueError(f"Unsupported memory_pg_variant: {pg_variant}")

        # Use memory-specific loss aggregation mode
        pg_loss = agg_loss(
            loss_mat=pg_loss, loss_mask=response_mask, loss_agg_mode=self.pipeline_config.memory_loss_agg_mode
        )

        # KL loss with reference model using memory_kl_coef
        if ref_log_probs is not None:
            kl_loss = compute_approx_kl(
                log_probs=log_probs, log_probs_base=ref_log_probs, action_mask=response_mask, kl_penalty="k3"
            )
            kl_loss = agg_loss(
                loss_mat=kl_loss, loss_mask=response_mask, loss_agg_mode=self.pipeline_config.memory_loss_agg_mode
            )

        # Compute metrics
        approxkl = compute_approx_kl(
            log_probs=log_probs, log_probs_base=old_log_probs, action_mask=response_mask, kl_penalty="mse"
        )
        policykl = compute_approx_kl(
            log_probs=log_probs, log_probs_base=old_log_probs, action_mask=response_mask, kl_penalty="kl"
        )

        entropy = self.strategy.op_compute_entropy(logits=output_tensor, attention_mask=data.batch["response_mask"])
        entropy_loss = agg_loss(
            loss_mat=entropy,
            loss_mask=response_mask,
            loss_agg_mode=self.pipeline_config.memory_loss_agg_mode,
        )

        # Total loss computation with memory_kl_coef
        if self.pipeline_config.memory_use_kl_loss and ref_log_probs is not None:
            total_loss = pg_loss + kl_loss * self.pipeline_config.memory_kl_coef
        else:
            total_loss = pg_loss

        # Add entropy loss if configured
        if self.pipeline_config.memory_entropy_loss_coef > 0:
            total_loss = total_loss - entropy_loss * self.pipeline_config.memory_entropy_loss_coef

        pg_metrics.update(
            {
                "memory_actor/pg_loss": pg_loss.detach().item(),
                "memory_actor/total_loss": total_loss.detach().item(),
                "memory_actor/approxkl": agg_loss(
                    loss_mat=approxkl, loss_mask=response_mask, loss_agg_mode=self.pipeline_config.memory_loss_agg_mode
                )
                .detach()
                .item(),
                "memory_actor/policykl": agg_loss(
                    loss_mat=policykl, loss_mask=response_mask, loss_agg_mode=self.pipeline_config.memory_loss_agg_mode
                )
                .detach()
                .item(),
                "memory_actor/entropy": entropy_loss.detach().item(),
                "memory_actor/ratio_mean": masked_mean(ratio, response_mask, dim=-1).mean().detach().item(),
                "memory_actor/ratio_max": torch.max(ratio * response_mask).detach().item(),
                "memory_actor/ratio_min": torch.min(ratio * response_mask + (1 - response_mask) * 1e10)
                .detach()
                .item(),
            }
        )

        if ref_log_probs is not None:
            pg_metrics["memory_actor/kl_loss"] = kl_loss.detach().item()

        return total_loss, pg_metrics
