import os
import threading
import time
from typing import Dict, Optional, Union

import numpy as np
import ray
import torch
from codetiming import Timer
from tqdm import tqdm

from roll.configs.worker_config import WorkerConfig
from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.decorator import Dispatch, register
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.strategy.factory import create_strategy
from roll.distributed.strategy.strategy import InferenceStrategy, TrainStrategy
from roll.models.model_providers import (
    default_actor_model_provider,
    default_reward_model_provider,
    default_value_model_provider,
)
from roll.utils.actor_pg_losses import (
    compute_cispo_pg_loss_mat_and_metrics,
    compute_ppo_pg_loss_mat_and_metrics,
    compute_tis_pg_loss_mat_and_metrics,
    compute_topr_pg_loss_mat_and_metrics,
)
from roll.utils.bool_utils import to_bool_tensor
from roll.utils.checkpoint_manager import download_model
from roll.utils.context_managers import state_offload_manger
from roll.utils.functionals import (
    GenerateRequestType,
    agg_loss,
    append_to_dict,
    compute_approx_kl,
    masked_mean,
    postprocess_generate,
)
from roll.utils.offload_states import OffloadStateType


def _scalarize_score(v) -> float:
    if isinstance(v, (list, tuple, np.ndarray)):
        return float(v[0]) if len(v) > 0 else 0.0
    return float(v)


def _get_topr_per_sample_scores(data: DataProto, response_mask: torch.Tensor, device: torch.device) -> torch.Tensor:
    if "scores" in data.batch:
        scores_tensor = data.batch["scores"]
        if scores_tensor.dim() == 2:
            return (scores_tensor[:, 1:] * response_mask.float()).sum(dim=-1).to(device=device, dtype=torch.float32)
        if scores_tensor.dim() == 1:
            return scores_tensor.to(device=device, dtype=torch.float32)
        raise ValueError(f"Unsupported scores tensor dim={scores_tensor.dim()} for TOPR.")

    if not (hasattr(data, "non_tensor_batch") and data.non_tensor_batch is not None):
        raise ValueError(
            "TOPR requires scores. Provide `data.batch['scores']` or `data.non_tensor_batch['episode_scores']`."
        )

    if "episode_scores" not in data.non_tensor_batch:
        raise ValueError(
            "TOPR requires scores. Provide `data.batch['scores']` or `data.non_tensor_batch['episode_scores']`."
        )

    raw_traj_ids = data.non_tensor_batch.get("traj_id", None)
    raw_episode_scores = data.non_tensor_batch["episode_scores"]

    if raw_traj_ids is not None:
        traj_to_score: Dict[str, float] = {}
        grouped = data.group_by(keys="traj_id")
        for traj_id, traj_batch in grouped.items():
            traj_scores = traj_batch.non_tensor_batch["episode_scores"][0]
            traj_to_score[str(traj_id)] = _scalarize_score(traj_scores)
        score_list = [traj_to_score.get(str(traj_id), 0.0) for traj_id in raw_traj_ids]
    else:
        score_list = [_scalarize_score(v) for v in raw_episode_scores]

    return torch.tensor(score_list, dtype=torch.float32, device=device)


class ActorWorker(Worker):
    def __init__(self, worker_config: WorkerConfig):
        super().__init__(worker_config=worker_config)
        self.tokenizer = None
        self.strategy: Optional[Union[InferenceStrategy, TrainStrategy]] = None
        self.response_call_back_fns = {}
        self.response_callback_refs = []
        self.server_metrics = {}
        self.thread_server = None
        self.offload_manager = None
        self.is_memory_model = getattr(worker_config, "is_memory_model", False)
        self._topr_sample_logged = False
        self._cispo_config_logged = False

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self, pipeline_config):
        super().initialize(pipeline_config)

        self.strategy = create_strategy(worker=self)

        self.strategy.initialize(model_provider=default_actor_model_provider)
        self.tokenizer = self.strategy.tokenizer
        if self.pipeline_config.resume_from_checkpoint:
            load_dir = download_model(self.pipeline_config.resume_from_checkpoint)
            self.strategy.load_checkpoint(load_dir=load_dir, tag="checkpoint")
        self.logger.info(f"{self.worker_name} initialized")

        self.strategy.offload_states()

        # Cuda must have been initialized when calling torch.cuda.reset_max_memory_allocated
        # with arguments (inside state_offload_manager). We explicitly init cuda here because
        # current process is used as engine client when using vllm v1 engine, and
        # there is no chance to init cuda context.
        torch.cuda.init()

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
                epochs=self.pipeline_config.ppo_epochs,
                seed=self.pipeline_config.seed,
                dataloader_kwargs={"shuffle": True},
            )

            for batch_idx, data in tqdm(
                enumerate(dataloader),
                desc=f"{self.worker_name} train global step {global_step}",
                total=data.batch.batch_size[0] * self.pipeline_config.ppo_epochs // backward_batch_size,
            ):
                pg_metrics = self.strategy.train_step(batch=data, loss_func=self.loss_func)
                append_to_dict(metrics, pg_metrics)

            lr_metric_key = "memory_actor/lr" if self.is_memory_model else "actor/lr"
            metrics[lr_metric_key] = self.strategy.scheduler.get_last_lr()[0]
            data.to("cpu")

        output = DataProto(meta_info={"metrics": metrics})
        return output

    @register(dispatch_mode=Dispatch.DP_MP_DISPATCH_FIRST)
    def train_sft_step(self, data: DataProto):
        """
        SFT training step using full batch without mini-batching.
        Uses SFT loss (cross-entropy on response tokens).
        
        Args:
            data: DataProto with batch containing 'input_ids', 'attention_mask', 
                  'position_ids', 'prompt_mask', 'response_mask', and 'labels'
                  
        Returns:
            DataProto with meta_info containing SFT metrics
        """
        global_step = data.meta_info.get("global_step", 0)
        is_offload_states = data.meta_info.get("is_offload_states", True)
        metrics = {}
        self.logger.info(f"{self.worker_name} SFT training global step {global_step}")

        with state_offload_manger(
            strategy=self.strategy,
            metrics=metrics,
            metric_infix=f"{self.cluster_name}/sft_train_step",
            is_offload_states=is_offload_states,
            load_kwargs={"include": [OffloadStateType.model_params, OffloadStateType.other_params]},
        ):
            data = data.to("cuda")
            data = self.strategy.get_data_input(data)

            # Use full batch - no mini-batching. Strategy handles gradient accumulation internally.
            sft_metrics = self.strategy.train_step(batch=data, loss_func=self.sft_loss_func)
            append_to_dict(metrics, sft_metrics)

            # Log SFT specific metrics
            metrics["sft/lr"] = self.strategy.scheduler.get_last_lr()[0]
            data.to("cpu")

        output = DataProto(meta_info={"metrics": metrics})
        return output

    def sft_loss_func(self, data: DataProto, output_tensor: torch.Tensor):
        """
        SFT loss function - cross-entropy on response tokens only.
        Labels should have IGNORE_INDEX (-100) for prompt tokens.
        """
        labels = data.batch["labels"]
        return self.strategy.op_compute_language_loss(output_tensor, labels)

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE)
    @torch.no_grad()
    def generate(self, data: DataProto):
        """
        batch = TensorDict(
            {
                'prompts': idx,
                'responses': response,
                'input_ids': seq,  # here input_ids become the whole sentences
                'attention_mask': attention_mask,
                'position_ids': position_ids,
                'old_log_probs': log_probs,
            },
            batch_size=batch_size)
        return DataProto(batch=batch)
        """
        if "generation_config" not in data.meta_info:
            generation_config = self.worker_config.generating_args.to_dict()
        else:
            generation_config = data.meta_info["generation_config"]

        generation_config["eos_token_id"] = [
            self.tokenizer.eos_token_id
        ] + self.tokenizer.additional_special_tokens_ids
        generation_config["pad_token_id"] = self.tokenizer.pad_token_id

        global_step = data.meta_info.get("global_step", 0)
        is_offload_states = data.meta_info.get("is_offload_states", True)
        self.logger.info(f"{self.worker_name} generate global step {global_step}")

        metrics = {}
        with state_offload_manger(
            strategy=self.strategy,
            metrics=metrics,
            metric_infix=f"{self.cluster_name}/generate",
            is_offload_states=is_offload_states,
        ):
            data = data.to("cuda")
            data.meta_info["micro_batch_size"] = self.worker_config.infer_batch_size

            output = self.strategy.generate(batch=data, generation_config=generation_config)
            output = postprocess_generate(
                prompts=data,
                output=output,
                num_return_sequences=generation_config["num_return_sequences"],
                sequence_length=self.pipeline_config.sequence_length,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            data.to("cpu")
            output = output.to("cpu")

        output.meta_info = {"metrics": metrics}
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL_ONE)
    @torch.no_grad()
    def start_server(self, data: DataProto):
        """
        解决dp generate的长尾问题，async+ load balance
        """
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
            target=self.strategy.start_server, kwargs=dict(data=data, request_complete_callback=self.request_complete)
        )
        self.thread_server.start()
        while not self.strategy.running:
            time.sleep(0.1)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL_ONE)
    def stop_server(self, data: DataProto = None):
        if self.thread_server == None:
            return

        self.strategy.add_request(command=GenerateRequestType.STOP, data=None)
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

    @register(dispatch_mode=Dispatch.DP_MP_DISPATCH_FIRST)
    def compute_log_probs(self, data: DataProto):
        """
        return DataProto.from_dict(tensors={'log_probs': output})
        """
        data = self.strategy.get_data_input(data)
        global_step = data.meta_info.get("global_step", 0)
        is_offload_states = data.meta_info.get("is_offload_states", True)
        metrics = {}
        with state_offload_manger(
            strategy=self.strategy,
            metrics=metrics,
            metric_infix=f"{self.cluster_name}/compute_log_probs",
            is_offload_states=is_offload_states,
            load_kwargs={"include": [OffloadStateType.model_params]},
        ):
            data = data.to("cuda")
            data.meta_info["micro_batch_size"] = self.worker_config.infer_batch_size
            with torch.no_grad():
                results: Dict[str, torch.Tensor] = self.strategy.forward_step(
                    batch=data, forward_func=self.forward_func_log_probs
                )
            if results is None:
                return DataProto(batch=None, meta_info={"metrics": metrics})
            if "question_entropy" in results:
                output = DataProto.from_dict(
                    tensors={
                        "log_probs": results["log_probs"],
                        "entropy": results["entropy"],
                        "question_entropy": results["question_entropy"],
                    }
                )
            else:
                output = DataProto.from_dict(
                    tensors={"log_probs": results["log_probs"], "entropy": results["entropy"]}
                )
            output = output.to("cpu")
            data.to("cpu")
        output.meta_info = {"metrics": metrics}
        return output

    def forward_func_log_probs(self, data: DataProto, output_tensor: torch.Tensor):
        """
        forward func 接口定义:
            data: DataProto, 由forward_step透传
            output_tensor: torch.Tensor, model.forward()的输出Tensor
        """
        log_probs = self.strategy.op_compute_log_probs(
            logits=output_tensor, input_ids=data.batch["input_ids"], attention_mask=data.batch["response_mask"]
        )
        entropy = self.strategy.op_compute_entropy(logits=output_tensor, attention_mask=data.batch["response_mask"])
        if "question_mask" in data.batch:
            question_entropy = self.strategy.op_compute_entropy(
                logits=output_tensor, attention_mask=data.batch["question_mask"]
            )
            return log_probs, {
                "log_probs": log_probs.clone().detach(),
                "entropy": entropy.clone().detach(),
                "question_entropy": question_entropy.clone().detach(),
            }
        else:
            return log_probs, {"log_probs": log_probs.clone().detach(), "entropy": entropy.clone().detach()}

    def loss_func(self, data: DataProto, output_tensor: torch.Tensor):
        """
        loss func接口定义:
            data: DataProto, 由train_step透传
            output_tensor: torch.Tensor, model.forward()的输出Tensor
        """

        response_mask = data.batch["response_mask"][:, 1:].long()
        ref_log_probs = data.batch.get("ref_log_probs", None)
        old_log_probs = data.batch["old_log_probs"]
        advantages = data.batch["advantages"]

        log_probs = self.strategy.op_compute_log_probs(
            logits=output_tensor, input_ids=data.batch["input_ids"], attention_mask=data.batch["response_mask"]
        )

        ratio = (log_probs - old_log_probs).exp()

        metric_prefix = "memory_actor" if self.is_memory_model else "actor"

        if self.is_memory_model:
            pg_clip = self.pipeline_config.memory_pg_clip
            pg_clip_low = self.pipeline_config.memory_pg_clip_low
            pg_clip_high = self.pipeline_config.memory_pg_clip_high
            use_pg_clip_range = self.pipeline_config.memory_use_pg_clip_range
            dual_clip_loss_enabled = self.pipeline_config.memory_dual_clip_loss
            loss_agg_mode = self.pipeline_config.memory_loss_agg_mode
            use_kl_loss = self.pipeline_config.memory_use_kl_loss
            kl_loss_coef = self.pipeline_config.memory_kl_loss_coef
            entropy_loss_coef = self.pipeline_config.memory_entropy_loss_coef
        else:
            pg_clip = self.pipeline_config.pg_clip
            pg_clip_low = self.pipeline_config.pg_clip_low
            pg_clip_high = self.pipeline_config.pg_clip_high
            use_pg_clip_range = self.pipeline_config.use_pg_clip_range
            dual_clip_loss_enabled = self.pipeline_config.dual_clip_loss
            loss_agg_mode = self.pipeline_config.loss_agg_mode
            use_kl_loss = self.pipeline_config.use_kl_loss
            kl_loss_coef = self.pipeline_config.kl_loss_coef
            entropy_loss_coef = self.pipeline_config.entropy_loss_coef

        grouped_cfg = self.pipeline_config.grouped_actor_pg
        grouped_enabled = (not self.is_memory_model) and grouped_cfg.enable

        if grouped_enabled:
            group_key = grouped_cfg.group_key
            strict = grouped_cfg.strict
            log_group_metrics = grouped_cfg.log_metrics

            raw_group = None
            if hasattr(data, "non_tensor_batch") and data.non_tensor_batch is not None:
                raw_group = data.non_tensor_batch.get(group_key, None)
            if raw_group is None:
                if strict:
                    raise KeyError(f"grouped_actor_pg.group_key={group_key!r} not found in non_tensor_batch")
                w_memory_mask = torch.zeros(data.batch.batch_size[0], dtype=torch.bool, device=log_probs.device)
                missing_group_key = True
            else:
                w_memory_mask = to_bool_tensor(raw_group, strict=strict, device=log_probs.device)
                if w_memory_mask.numel() != data.batch.batch_size[0]:
                    raise ValueError(
                        f"non_tensor_batch[{group_key!r}] length {w_memory_mask.numel()} "
                        f"!= batch size {data.batch.batch_size[0]}"
                    )
                missing_group_key = False

            wo_memory_mask = ~w_memory_mask
            w_count = int(w_memory_mask.sum().item())
            wo_count = int(wo_memory_mask.sum().item())

            pg_variant_wo_memory = grouped_cfg.pg_variant_wo_memory
            pg_variant_w_memory = grouped_cfg.pg_variant_w_memory

            variants = {pg_variant_wo_memory, pg_variant_w_memory}
            topr_scores: torch.Tensor | None = None
            if "topr" in variants:
                topr_scores = _get_topr_per_sample_scores(
                    data=data,
                    response_mask=response_mask,
                    device=log_probs.device,
                )
                if not self._topr_sample_logged:
                    total_samples = int(topr_scores.numel())
                    positive_count = float((topr_scores > 0).sum().item())
                    negative_count = float((topr_scores <= 0).sum().item())
                    self.logger.info(
                        f"TOPR sample distribution - total: {total_samples}, "
                        f"positive: {positive_count} ({positive_count/max(total_samples,1)*100:.1f}%), "
                        f"negative: {negative_count} ({negative_count/max(total_samples,1)*100:.1f}%)"
                    )
                    self.logger.info(
                        f"TOPR score stats - mean: {topr_scores.mean().item():.4f}, "
                        f"std: {topr_scores.std(unbiased=False).item():.4f}, "
                        f"max: {topr_scores.max().item():.4f}, min: {topr_scores.min().item():.4f}"
                    )
                    self.logger.info(
                        "TOPR weights - "
                        f"positive_weight: {self.worker_config.topr_positive_weight}, "
                        f"negative_weight: {self.worker_config.topr_negative_weight}"
                    )
                    self._topr_sample_logged = True

            if "cispo" in variants and not self._cispo_config_logged:
                epsilon_low = float(self.worker_config.cispo_epsilon_low)
                epsilon_high = float(self.worker_config.cispo_epsilon_high)
                use_unified_mask = bool(self.worker_config.cispo_use_unified_mask)
                clip_lower = 1.0 - epsilon_low
                clip_upper = 1.0 + epsilon_high
                self.logger.info(f"CISPO config - epsilon_low: {epsilon_low}, epsilon_high: {epsilon_high}")
                self.logger.info(f"CISPO clip range: [{clip_lower:.3f}, {clip_upper:.3f}]")
                self.logger.info(f"CISPO unified mask enabled: {use_unified_mask}")
                self._cispo_config_logged = True

            def _prefix_group_metrics(prefix: str, m: Dict[str, float]) -> Dict[str, float]:
                return {f"{prefix}/{k}": v for k, v in m.items()}

            def _compute_pg_loss_mat_and_metrics(
                *,
                pg_variant: str,
                metric_prefix_group: str,
                sample_mask: torch.Tensor,
                token_mask: torch.Tensor,
            ) -> tuple[torch.Tensor, Dict[str, float]]:
                if pg_variant in {"vanilla", "ppo"}:
                    pg_loss_local, m = compute_ppo_pg_loss_mat_and_metrics(
                        ratio=ratio,
                        advantages=advantages,
                        pg_clip=float(pg_clip),
                        pg_clip_low=float(pg_clip_low),
                        pg_clip_high=float(pg_clip_high),
                        use_pg_clip_range=bool(use_pg_clip_range),
                        dual_clip_loss_enabled=bool(dual_clip_loss_enabled),
                        loss_mask=token_mask,
                        loss_agg_mode=loss_agg_mode,
                        clipfrac_mask=token_mask,
                    )
                    return pg_loss_local, _prefix_group_metrics(metric_prefix_group, m)

                if pg_variant == "tis":
                    pg_loss_local, m = compute_tis_pg_loss_mat_and_metrics(
                        ratio=ratio,
                        log_probs=log_probs,
                        advantages=advantages,
                        tis_lower_bound=float(self.worker_config.tis_lower_bound),
                        tis_upper_bound=float(self.worker_config.tis_upper_bound),
                        loss_mask=token_mask,
                    )
                    return pg_loss_local, _prefix_group_metrics(metric_prefix_group, m)

                if pg_variant == "topr":
                    assert topr_scores is not None, "TOPR scores must be precomputed"
                    pg_loss_local, m = compute_topr_pg_loss_mat_and_metrics(
                        ratio=ratio,
                        log_probs=log_probs,
                        advantages=advantages,
                        scores=topr_scores,
                        topr_positive_weight=float(self.worker_config.topr_positive_weight),
                        topr_negative_weight=float(self.worker_config.topr_negative_weight),
                        loss_mask=token_mask,
                        loss_agg_mode=loss_agg_mode,
                        sample_mask=sample_mask,
                        clipfrac_mask=token_mask,
                    )
                    return pg_loss_local, _prefix_group_metrics(metric_prefix_group, m)

                if pg_variant == "cispo":
                    pg_loss_local, m = compute_cispo_pg_loss_mat_and_metrics(
                        ratio=ratio,
                        log_probs=log_probs,
                        advantages=advantages,
                        epsilon_low=float(self.worker_config.cispo_epsilon_low),
                        epsilon_high=float(self.worker_config.cispo_epsilon_high),
                        use_unified_mask=bool(self.worker_config.cispo_use_unified_mask),
                        loss_mask=token_mask,
                    )
                    return pg_loss_local, _prefix_group_metrics(metric_prefix_group, m)

                raise ValueError(f"Unsupported pg_variant: {pg_variant}")

            wo_token_mask = response_mask * wo_memory_mask.unsqueeze(-1).long()
            w_token_mask = response_mask * w_memory_mask.unsqueeze(-1).long()

            wo_metric_prefix = f"{metric_prefix}/grouped_pg/wo_memory"
            w_metric_prefix = f"{metric_prefix}/grouped_pg/w_memory"

            pg_loss_wo_mat = torch.zeros_like(ratio)
            pg_loss_w_mat = torch.zeros_like(ratio)
            grouped_metrics: Dict[str, float] = {}
            if missing_group_key:
                grouped_metrics[f"{metric_prefix}/grouped_pg/missing_group_key"] = 1.0

            if log_group_metrics and wo_count > 0:
                pg_loss_wo_mat, wo_variant_metrics = _compute_pg_loss_mat_and_metrics(
                    pg_variant=pg_variant_wo_memory,
                    metric_prefix_group=wo_metric_prefix,
                    sample_mask=wo_memory_mask,
                    token_mask=wo_token_mask,
                )
                grouped_metrics.update(wo_variant_metrics)

            if log_group_metrics and w_count > 0:
                pg_loss_w_mat, w_variant_metrics = _compute_pg_loss_mat_and_metrics(
                    pg_variant=pg_variant_w_memory,
                    metric_prefix_group=w_metric_prefix,
                    sample_mask=w_memory_mask,
                    token_mask=w_token_mask,
                )
                grouped_metrics.update(w_variant_metrics)

            if (not log_group_metrics) and wo_count > 0:
                pg_loss_wo_mat, _ = _compute_pg_loss_mat_and_metrics(
                    pg_variant=pg_variant_wo_memory,
                    metric_prefix_group=wo_metric_prefix,
                    sample_mask=wo_memory_mask,
                    token_mask=wo_token_mask,
                )
            if (not log_group_metrics) and w_count > 0:
                pg_loss_w_mat, _ = _compute_pg_loss_mat_and_metrics(
                    pg_variant=pg_variant_w_memory,
                    metric_prefix_group=w_metric_prefix,
                    sample_mask=w_memory_mask,
                    token_mask=w_token_mask,
                )

            pg_loss_mat = torch.where(w_memory_mask.unsqueeze(-1), pg_loss_w_mat, pg_loss_wo_mat)
            pg_loss = agg_loss(loss_mat=pg_loss_mat, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

            kl_loss_mat = None
            kl_loss = None
            if ref_log_probs is not None:
                kl_loss_mat = compute_approx_kl(
                    log_probs=log_probs, log_probs_base=ref_log_probs, action_mask=response_mask, kl_penalty="k3"
                )
                kl_loss = agg_loss(loss_mat=kl_loss_mat, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

            approxkl = compute_approx_kl(
                log_probs=log_probs, log_probs_base=old_log_probs, action_mask=response_mask, kl_penalty="mse"
            )
            policykl = compute_approx_kl(
                log_probs=log_probs, log_probs_base=old_log_probs, action_mask=response_mask, kl_penalty="kl"
            )

            entropy = self.strategy.op_compute_entropy(
                logits=output_tensor, attention_mask=data.batch["response_mask"]
            )
            entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

            if use_kl_loss and ref_log_probs is not None:
                total_loss = pg_loss + kl_loss * kl_loss_coef
            else:
                total_loss = pg_loss
            if entropy_loss_coef > 0:
                total_loss = total_loss - entropy_loss * entropy_loss_coef

            pg_metrics = {
                f"{metric_prefix}/ratio_mean": masked_mean(ratio, response_mask, dim=-1).mean().detach().item(),
                f"{metric_prefix}/ratio_max": torch.max(ratio * response_mask).detach().item(),
                f"{metric_prefix}/ratio_min": torch.min(ratio * response_mask + (1 - response_mask) * 1e10)
                .detach()
                .item(),
                f"{metric_prefix}/pg_loss": pg_loss.detach().item(),
                f"{metric_prefix}/total_loss": total_loss.detach().item(),
                f"{metric_prefix}/approxkl": agg_loss(
                    loss_mat=approxkl, loss_mask=response_mask, loss_agg_mode=loss_agg_mode
                )
                .detach()
                .item(),
                f"{metric_prefix}/policykl": agg_loss(
                    loss_mat=policykl, loss_mask=response_mask, loss_agg_mode=loss_agg_mode
                )
                .detach()
                .item(),
            }

            if ref_log_probs is not None:
                pg_metrics[f"{metric_prefix}/kl_loss"] = kl_loss.detach().item()

            if log_group_metrics:
                for group_prefix, group_mask, group_token_mask, group_pg_loss_mat in (
                    (wo_metric_prefix, wo_memory_mask, wo_token_mask, pg_loss_wo_mat),
                    (w_metric_prefix, w_memory_mask, w_token_mask, pg_loss_w_mat),
                ):
                    if group_mask.any() and group_token_mask.sum() > 0:
                        group_pg_loss = agg_loss(
                            loss_mat=group_pg_loss_mat, loss_mask=group_token_mask, loss_agg_mode=loss_agg_mode
                        )
                        group_entropy_loss = agg_loss(
                            loss_mat=entropy, loss_mask=group_token_mask, loss_agg_mode=loss_agg_mode
                        )
                        group_total_loss = group_pg_loss
                        if kl_loss_mat is not None:
                            group_kl_loss = agg_loss(
                                loss_mat=kl_loss_mat, loss_mask=group_token_mask, loss_agg_mode=loss_agg_mode
                            )
                            pg_metrics[f"{group_prefix}/kl_loss"] = group_kl_loss.detach().item()
                            if use_kl_loss:
                                group_total_loss = group_total_loss + group_kl_loss * kl_loss_coef
                        if entropy_loss_coef > 0:
                            group_total_loss = group_total_loss - group_entropy_loss * entropy_loss_coef

                        pg_metrics.update(
                            {
                                f"{group_prefix}/ratio_mean": masked_mean(ratio, group_token_mask, dim=-1)
                                .mean()
                                .detach()
                                .item(),
                                f"{group_prefix}/ratio_max": torch.max(ratio * group_token_mask).detach().item(),
                                f"{group_prefix}/ratio_min": torch.min(
                                    ratio * group_token_mask + (1 - group_token_mask) * 1e10
                                )
                                .detach()
                                .item(),
                                f"{group_prefix}/pg_loss": group_pg_loss.detach().item(),
                                f"{group_prefix}/total_loss": group_total_loss.detach().item(),
                                f"{group_prefix}/approxkl": agg_loss(
                                    loss_mat=approxkl, loss_mask=group_token_mask, loss_agg_mode=loss_agg_mode
                                )
                                .detach()
                                .item(),
                                f"{group_prefix}/policykl": agg_loss(
                                    loss_mat=policykl, loss_mask=group_token_mask, loss_agg_mode=loss_agg_mode
                                )
                                .detach()
                                .item(),
                            }
                        )
                    else:
                        pg_metrics.update(
                            {
                                f"{group_prefix}/ratio_mean": 0.0,
                                f"{group_prefix}/ratio_max": 0.0,
                                f"{group_prefix}/ratio_min": 0.0,
                                f"{group_prefix}/pg_loss": 0.0,
                                f"{group_prefix}/total_loss": 0.0,
                                f"{group_prefix}/approxkl": 0.0,
                                f"{group_prefix}/policykl": 0.0,
                            }
                        )

                pg_metrics.update(grouped_metrics)

            return total_loss, pg_metrics

        pg_variant = (
            self.pipeline_config.memory_pg_variant if self.is_memory_model else self.worker_config.pg_variant
        ) or "vanilla"
        variant_metrics: Dict[str, float] = {}

        if pg_variant in {"vanilla", "ppo", None}:
            pg_loss, m = compute_ppo_pg_loss_mat_and_metrics(
                ratio=ratio,
                advantages=advantages,
                pg_clip=float(pg_clip),
                pg_clip_low=float(pg_clip_low),
                pg_clip_high=float(pg_clip_high),
                use_pg_clip_range=bool(use_pg_clip_range),
                dual_clip_loss_enabled=bool(dual_clip_loss_enabled),
                loss_mask=response_mask,
                loss_agg_mode=loss_agg_mode,
                clipfrac_mask=None,
            )
            variant_metrics = {f"{metric_prefix}/{k}": v for k, v in m.items()}
        elif pg_variant == "tis":
            pg_loss, m = compute_tis_pg_loss_mat_and_metrics(
                ratio=ratio,
                log_probs=log_probs,
                advantages=advantages,
                tis_lower_bound=float(self.worker_config.tis_lower_bound),
                tis_upper_bound=float(self.worker_config.tis_upper_bound),
                loss_mask=response_mask,
            )
            variant_metrics = {f"{metric_prefix}/{k}": v for k, v in m.items()}
        elif pg_variant == "topr":
            topr_positive_weight = float(self.worker_config.topr_positive_weight)
            topr_negative_weight = float(self.worker_config.topr_negative_weight)

            scores = _get_topr_per_sample_scores(data=data, response_mask=response_mask, device=log_probs.device)

            if not self._topr_sample_logged:
                total_samples = int(scores.numel())
                positive_count = float((scores > 0).sum().item())
                negative_count = float((scores <= 0).sum().item())
                self.logger.info(
                    f"TOPR sample distribution - total: {total_samples}, "
                    f"positive: {positive_count} ({positive_count/max(total_samples,1)*100:.1f}%), "
                    f"negative: {negative_count} ({negative_count/max(total_samples,1)*100:.1f}%)"
                )
                self.logger.info(
                    f"TOPR score stats - mean: {scores.mean().item():.4f}, std: {scores.std(unbiased=False).item():.4f}, "
                    f"max: {scores.max().item():.4f}, min: {scores.min().item():.4f}"
                )
                self.logger.info(
                    f"TOPR weights - positive_weight: {topr_positive_weight}, negative_weight: {topr_negative_weight}"
                )
                self._topr_sample_logged = True

            pg_loss, m = compute_topr_pg_loss_mat_and_metrics(
                ratio=ratio,
                log_probs=log_probs,
                advantages=advantages,
                scores=scores,
                topr_positive_weight=topr_positive_weight,
                topr_negative_weight=topr_negative_weight,
                loss_mask=response_mask,
                loss_agg_mode=loss_agg_mode,
                sample_mask=None,
                clipfrac_mask=None,
            )
            variant_metrics = {f"{metric_prefix}/{k}": v for k, v in m.items()}
        elif pg_variant == "cispo":
            epsilon_low = float(self.worker_config.cispo_epsilon_low)
            epsilon_high = float(self.worker_config.cispo_epsilon_high)
            use_unified_mask = bool(self.worker_config.cispo_use_unified_mask)

            clip_lower = 1.0 - epsilon_low
            clip_upper = 1.0 + epsilon_high

            if not self._cispo_config_logged:
                self.logger.info(f"CISPO config - epsilon_low: {epsilon_low}, epsilon_high: {epsilon_high}")
                self.logger.info(f"CISPO clip range: [{clip_lower:.3f}, {clip_upper:.3f}]")
                self.logger.info(f"CISPO unified mask enabled: {use_unified_mask}")
                self._cispo_config_logged = True
            pg_loss, m = compute_cispo_pg_loss_mat_and_metrics(
                ratio=ratio,
                log_probs=log_probs,
                advantages=advantages,
                epsilon_low=epsilon_low,
                epsilon_high=epsilon_high,
                use_unified_mask=use_unified_mask,
                loss_mask=response_mask,
            )
            variant_metrics = {f"{metric_prefix}/{k}": v for k, v in m.items()}
        else:
            raise ValueError(f"Unsupported pg_variant: {pg_variant}")

        pg_loss = agg_loss(loss_mat=pg_loss, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

        if ref_log_probs is not None:
            kl_loss = compute_approx_kl(
                log_probs=log_probs, log_probs_base=ref_log_probs, action_mask=response_mask, kl_penalty="k3"
            )
            kl_loss = agg_loss(loss_mat=kl_loss, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

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
            loss_agg_mode=loss_agg_mode,
        )

        if use_kl_loss and ref_log_probs is not None:
            total_loss = pg_loss + kl_loss * kl_loss_coef
        else:
            total_loss = pg_loss
        if entropy_loss_coef > 0:
            total_loss = total_loss - entropy_loss * entropy_loss_coef

        pg_metrics = {
            f"{metric_prefix}/ratio_mean": masked_mean(ratio, response_mask, dim=-1).mean().detach().item(),
            f"{metric_prefix}/ratio_max": torch.max(ratio * response_mask).detach().item(),
            f"{metric_prefix}/ratio_min": torch.min(ratio * response_mask + (1 - response_mask) * 1e10)
            .detach()
            .item(),
            f"{metric_prefix}/pg_loss": pg_loss.detach().item(),
            f"{metric_prefix}/total_loss": total_loss.detach().item(),
            f"{metric_prefix}/approxkl": agg_loss(
                loss_mat=approxkl, loss_mask=response_mask, loss_agg_mode=loss_agg_mode
            )
            .detach()
            .item(),
            f"{metric_prefix}/policykl": agg_loss(
                loss_mat=policykl, loss_mask=response_mask, loss_agg_mode=loss_agg_mode
            )
            .detach()
            .item(),
        }

        pg_metrics.update(variant_metrics)

        if ref_log_probs is not None:
            pg_metrics[f"{metric_prefix}/kl_loss"] = kl_loss.detach().item()

        return total_loss, pg_metrics

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def do_checkpoint(self, global_step):
        with Timer("do_checkpoint") as total_timer:
            ckpt_id = f"checkpoint-{global_step}"
            # actor train是直接存在save dir目录下的，其他role是存在save_dir/cluster_name下的
            save_dir = os.path.join(self.pipeline_config.output_dir, self.worker_name, ckpt_id)
            self.logger.info(f"save checkpoint-{global_step} to {save_dir}")

            exec_metrics: Dict = self.strategy.save_checkpoint(save_dir, global_step, ckpt_id)

        metrics = {
            f"time/{self.cluster_name}/do_checkpoint/total": total_timer.last,
        }
        metric_prefix = f"time/{self.cluster_name}/do_checkpoint"
        metrics.update({f"{metric_prefix}/{k}": v for k, v in exec_metrics.items()})
        output = DataProto(meta_info={"metrics": metrics})
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, clear_cache=False)
    def add_request(self, command, data: DataProto):
        """
        data req meta_info里需要包含:
            request_id: str
            response_callback_fn: callable
        generation_config, 按request设置
        """
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
        self.strategy.add_request(command=command, data=data)
        return DataProto(meta_info={"request_counts": len(self.response_call_back_fns)})

    def request_complete(self, data: DataProto):
        data.meta_info["eos_token_id"] = self.tokenizer.eos_token_id
        data.meta_info["pad_token_id"] = self.tokenizer.pad_token_id
        response_call_back_fn = self.response_call_back_fns.pop(data.meta_info["request_id"])
        self.response_callback_refs.append(response_call_back_fn(data))


class CriticWorker(Worker):

    def __init__(self, worker_config: WorkerConfig):
        super().__init__(worker_config=worker_config)
        self.tokenizer = None
        self.strategy: Optional[Union[InferenceStrategy, TrainStrategy]] = None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self, pipeline_config):
        super().initialize(pipeline_config)

        self.strategy = create_strategy(worker=self)

        self.strategy.initialize(model_provider=default_value_model_provider)
        self.tokenizer = self.strategy.tokenizer

        if self.pipeline_config.resume_from_checkpoint:
            load_dir = os.path.join(download_model(self.pipeline_config.resume_from_checkpoint), self.cluster_name)
            self.strategy.load_checkpoint(load_dir=load_dir, tag="checkpoint")

        self.logger.info(f"{self.worker_name} initialized")

        self.strategy.offload_states()

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE)
    def compute_values(self, data: DataProto):
        """
        return DataProto.from_dict(tensors={'values': values})
        """
        global_step = data.meta_info.get("global_step", 0)
        is_offload_states = data.meta_info.get("is_offload_states", True)
        metrics = {}
        with state_offload_manger(
            strategy=self.strategy,
            metrics=metrics,
            metric_infix=f"{self.cluster_name}/compute_values",
            is_offload_states=is_offload_states,
            load_kwargs={"include": [OffloadStateType.model_params]},
        ):
            data = data.to("cuda")
            data.meta_info["micro_batch_size"] = self.worker_config.infer_batch_size
            with torch.no_grad():
                results: Dict[str, torch.Tensor] = self.strategy.forward_step(
                    batch=data, forward_func=self.forward_func_values
                )

            output = DataProto.from_dict(tensors={"values": results["values"]})
            data.to("cpu")
            output = output.to("cpu")

        output.meta_info = {"metrics": metrics}
        return output

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE)
    def train_step(self, data: DataProto):
        """
        return DataProto(meta_info={'metrics': metrics})
        """
        global_step = data.meta_info.get("global_step", 0)
        is_offload_states = data.meta_info.get("is_offload_states", True)
        metrics = {}
        with state_offload_manger(
            strategy=self.strategy,
            metrics=metrics,
            metric_infix=f"{self.cluster_name}/train_step",
            is_offload_states=is_offload_states,
            load_kwargs={"include": [OffloadStateType.model_params, OffloadStateType.other_params]},
        ):
            data = data.to("cuda")
            per_device_train_batch_size = self.worker_config.training_args.per_device_train_batch_size
            backward_batch_size = (
                per_device_train_batch_size * self.worker_config.training_args.gradient_accumulation_steps
            )

            dataloader = data.make_iterator(
                mini_batch_size=backward_batch_size,
                epochs=1,
                seed=self.pipeline_config.seed,
                dataloader_kwargs={"shuffle": True},
            )

            for batch_idx, data in tqdm(
                enumerate(dataloader),
                desc=f"{self.worker_name} train global step {global_step}",
                total=data.batch.batch_size[0] * self.pipeline_config.ppo_epochs // backward_batch_size,
            ):
                vf_metrics = self.strategy.train_step(batch=data, loss_func=self.loss_func)
                append_to_dict(metrics, vf_metrics)

            data.to("cpu")
            metrics["critic/lr"] = self.strategy.scheduler.get_last_lr()[0]

        output = DataProto(meta_info={"metrics": metrics}).to("cpu")

        return output

    def loss_func(self, data: DataProto, output_tensor: torch.Tensor):
        """
        loss func接口定义:
            data: DataProto, 由train_step透传
            output_tensor: torch.Tensor, model.forward()的输出Tensor
        """
        response_mask = data.batch["response_mask"][:, 1:]
        old_values = data.batch["values"]
        returns = data.batch["returns"]

        values, _ = self.forward_func_values(data=data, output_tensor=output_tensor)

        if self.pipeline_config.value_clip is not None:
            values_clipped = torch.clip(
                values,
                old_values - self.pipeline_config.value_clip,
                old_values + self.pipeline_config.value_clip,
            )
            surr1 = (values - returns) ** 2
            surr2 = (values_clipped - returns) ** 2
            vf_clipfrac = masked_mean(torch.gt(surr2, surr1).float(), response_mask, dim=-1).mean()
            loss = torch.max(surr1, surr2)
        else:
            loss = (values - returns) ** 2
            vf_clipfrac = masked_mean(loss, response_mask, dim=-1).mean()

        vf_loss = 0.5 * masked_mean(loss, response_mask, dim=-1).mean()

        vf_metrics = {
            "critic/loss": vf_loss.detach().item(),
            "critic/value": (masked_mean(old_values, response_mask, dim=-1)).mean().detach().item(),
            "critic/vpred": (masked_mean(values, response_mask, dim=-1)).mean().detach().item(),
            "critic/clipfrac": vf_clipfrac.detach().item(),
            "critic/error": masked_mean((values - returns) ** 2, response_mask, dim=-1).mean().detach().item(),
        }

        return vf_loss, vf_metrics

    def forward_func_values(self, data: DataProto, output_tensor: torch.Tensor):
        values = output_tensor[:, :-1]
        values = values.squeeze(dim=-1)
        return values, {"values": values.clone().detach()}

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def do_checkpoint(self, global_step):
        with Timer("do_checkpoint") as total_timer:
            ckpt_id = f"checkpoint-{global_step}"
            save_dir = os.path.join(self.pipeline_config.output_dir, self.worker_name, ckpt_id, self.cluster_name)
            critic_save_dir = os.path.join(self.pipeline_config.output_dir, self.worker_name, ckpt_id)
            self.logger.info(f"save checkpoint-{global_step} to {save_dir}")
            exec_metrics: Dict = self.strategy.save_checkpoint(
                save_dir, global_step, ckpt_id, local_state_path=critic_save_dir
            )

        metrics = {
            f"time/{self.cluster_name}/do_checkpoint/total": total_timer.last,
        }
        metric_prefix = f"time/{self.cluster_name}/do_checkpoint"
        metrics.update({f"{metric_prefix}/{k}": v for k, v in exec_metrics.items()})
        output = DataProto(meta_info={"metrics": metrics})
        return output


class RewardWorker(Worker):
    """
    Reward Model 使用 AutoModelForSequenceClassification 协议
    """

    def __init__(self, worker_config: WorkerConfig):
        super().__init__(worker_config=worker_config)
        self.tokenizer = None
        self.strategy: Optional[Union[InferenceStrategy, TrainStrategy]] = None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self, pipeline_config):
        super().initialize(pipeline_config)

        self.strategy = create_strategy(worker=self)

        self.strategy.initialize(model_provider=default_reward_model_provider)
        self.tokenizer = self.strategy.tokenizer

        self.logger.info(f"{self.worker_name} initialized")
        self.strategy.offload_states()

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE, clear_cache=False)
    def compute_rewards(self, data: DataProto):
        """
        return DataProto.from_dict(tensors={'rewards': rewards})
        """
        global_step = data.meta_info.get("global_step", 0)
        is_offload_states = data.meta_info.get("is_offload_states", True)
        metrics = {}
        with state_offload_manger(
            strategy=self.strategy,
            metrics=metrics,
            metric_infix=f"{self.cluster_name}/compute_rewards",
            is_offload_states=is_offload_states,
        ):
            data = data.to("cuda")

            # TODO: _switch_chat_template, 异构reward model

            data.meta_info["micro_batch_size"] = self.worker_config.infer_batch_size
            with torch.no_grad():
                results: Dict[str, torch.Tensor] = self.strategy.forward_step(
                    batch=data, forward_func=self.forward_func_values
                )
            token_level_rewards = results["values"]  # (bsz, input_ids.shape[1]-1)
            input_ids = data.batch["input_ids"][:, 1:]
            seq_lengths = torch.eq(input_ids, self.tokenizer.pad_token_id).int().argmax(-1) - 1
            seq_lengths = (seq_lengths % input_ids.shape[-1]).to(token_level_rewards.device)
            response_level_rewards = token_level_rewards[
                torch.arange(seq_lengths.shape[0], device=token_level_rewards.device), seq_lengths
            ]

            output = DataProto.from_dict(
                tensors={"token_level_rewards": token_level_rewards, "response_level_rewards": response_level_rewards}
            )

            data.to("cpu")
            output = output.to("cpu")

        output.meta_info = {"metrics": metrics}
        return output

    def forward_func_values(self, data: DataProto, output_tensor: torch.Tensor):
        values = output_tensor[:, 1:]
        values = values.squeeze(dim=-1)
        return values, {"values": values.clone().detach()}
