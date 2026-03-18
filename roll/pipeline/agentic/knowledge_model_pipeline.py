import json
import os
import os.path
import random
import traceback
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import ray
import torch
from codetiming import Timer
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from ray.util.timer import _Timer
from tensordict import TensorDict

from roll.datasets.global_dataset import GlobalDatasetManager
from roll.distributed.executor.cluster import Cluster
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.scheduler.resource_manager import ResourceManager
from roll.distributed.scheduler.rollout_scheduler import RolloutScheduler
from roll.models.model_providers import default_tokenizer_provider
from roll.pipeline.agentic.agentic_pipeline import AgenticPipeline
from roll.pipeline.agentic.multi_agentic_config import MemoryActorConfig
from roll.pipeline.agentic.utils import (
    compute_discounted_returns,
    compute_response_level_rewards,
    dump_rollout_trajectories,
)
from roll.pipeline.base_pipeline import BasePipeline
from roll.utils.constants import RAY_NAMESPACE
from roll.utils.functionals import (
    RunningMoments,
    agg_loss,
    apply_kl_penalty,
    compute_advantage,
    compute_clip_fraction,
    masked_mean,
    pad_to_length,
    reduce_metrics,
)
from roll.utils.kl_controller import get_kl_controller
from roll.utils.logging import get_logger

logger = get_logger()


@dataclass
class MemoryTrainingBufferEntry:
    request_id: str
    interaction: DataProto
    score_sum: float = 0.0
    score_count: int = 0

    def add(self, interaction: DataProto, score_sum: float, score_count: int) -> None:
        self.interaction = interaction  # keep latest representative
        self.score_sum += float(score_sum)
        self.score_count += int(score_count)

    @property
    def score_mean(self) -> float:
        return self.score_sum / max(self.score_count, 1)


class MemoryTrainingBuffer:
    def __init__(self, max_size: int):
        self.max_size = int(max_size)
        self._entries: "OrderedDict[str, MemoryTrainingBufferEntry]" = OrderedDict()

    def __len__(self) -> int:
        return len(self._entries)

    def add(self, request_id: str, interaction: DataProto, score_sum: float, score_count: int) -> None:
        request_id = str(request_id)
        if request_id in self._entries:
            self._entries[request_id].add(interaction=interaction, score_sum=score_sum, score_count=score_count)
            return

        # Evict oldest if at capacity.
        if self.max_size > 0 and len(self._entries) >= self.max_size:
            self._entries.popitem(last=False)

        entry = MemoryTrainingBufferEntry(
            request_id=request_id,
            interaction=interaction,
            score_sum=float(score_sum),
            score_count=int(score_count),
        )
        self._entries[request_id] = entry

    def ready(self, batch_size: int) -> bool:
        return len(self._entries) >= int(batch_size)

    def pop_batch(self, batch_size: int) -> List[MemoryTrainingBufferEntry]:
        batch_size = int(batch_size)
        batch_size = min(batch_size, len(self._entries))
        out: List[MemoryTrainingBufferEntry] = []
        for _ in range(batch_size):
            _, entry = self._entries.popitem(last=False)
            out.append(entry)
        return out


class RefactoredMemoryActorPipeline(AgenticPipeline):
    """
    Refactored Memory Actor Pipeline with phased training:

    1. Warmup Phase: Actor collects batches and records statistics, but doesn't add to buffer
       - Controlled by memory_config.memory_warmup_interval
       - Consistent with async_memory_manager's could_begin_interaction()
       - Memory actor remains offloaded
       - Memory updates continue in background (no flush)
    2. Buffer Phase: After memory_warmup_interval, start adding to buffer
    3. Training Phase: Only after buffer is ready, memory model updates
    4. Validation: Actor validates at the beginning and after memory model updates

    Step Tracking:
    - basic_step: Controls the main loop iteration
    - true_step: Counts memory actor updates
    """

    def __init__(self, pipeline_config: MemoryActorConfig):
        self.pipeline_config = pipeline_config
        self.pipeline_config.set_max_steps(max_steps=self.pipeline_config.max_steps)
        self.resource_manager = ResourceManager(
            num_nodes=self.pipeline_config.num_nodes,
            num_gpus_per_node=self.pipeline_config.num_gpus_per_node,
        )

        BasePipeline.__init__(self, pipeline_config)
        self.memory_model_tokenizer = default_tokenizer_provider(
            model_args=self.pipeline_config.memory_config.memory_actor_train.model_args
        )

        # Create memory-specific clusters
        self.memory_actor_train: Any = Cluster(
            name=self.pipeline_config.memory_config.memory_actor_train.name,
            worker_cls=self.pipeline_config.memory_config.memory_actor_train.worker_cls,
            resource_manager=self.resource_manager,
            worker_config=self.pipeline_config.memory_config.memory_actor_train,
        )

        # CRITICAL: This cluster must be initialized BEFORE super().__init__()
        self.memory_model_infer: Any = Cluster(
            name=self.pipeline_config.memory_config.memory_model.name,
            worker_cls=self.pipeline_config.memory_config.memory_model.worker_cls,
            resource_manager=self.resource_manager,
            worker_config=self.pipeline_config.memory_config.memory_model,
        )

        if self.pipeline_config.reference_for_memory:
            self.reference_for_memory: Any = Cluster(
                name=self.pipeline_config.memory_reference.name,
                worker_cls=self.pipeline_config.memory_reference.worker_cls,
                resource_manager=self.resource_manager,
                worker_config=self.pipeline_config.memory_reference,
            )

        self.memory_kl_ctrl = get_kl_controller(
            init_kl_coef=self.pipeline_config.memory_kl_coef,
            target_kl=self.pipeline_config.memory_target_kl,
            kl_horizon=self.pipeline_config.memory_kl_horizon,
        )

        download_clusters = [
            self.memory_actor_train,
            self.memory_model_infer,
        ]
        if self.pipeline_config.reference_for_memory:
            download_clusters.append(self.reference_for_memory)
        self.download_models(*download_clusters)

        # Initialize clusters and wait for completion
        refs: List[ray.ObjectRef] = []
        refs.extend(self.memory_actor_train.initialize(pipeline_config=self.pipeline_config, blocking=False))
        memory_infer_refs = self.memory_model_infer.initialize(blocking=False)
        if memory_infer_refs is not None:
            refs.extend(memory_infer_refs)
        if self.pipeline_config.reference_for_memory:
            refs.extend(self.reference_for_memory.initialize(pipeline_config=self.pipeline_config, blocking=False))
        ray.get(refs)
        logger.info("Memory actor train and memory model infer clusters initialized")

        self.actor_infer: Any = Cluster(
            name=self.pipeline_config.actor_infer.name,
            worker_cls=self.pipeline_config.actor_infer.worker_cls,
            resource_manager=self.resource_manager,
            worker_config=self.pipeline_config.actor_infer,
        )
        download_clusters = [
            self.actor_infer,
        ]

        self.download_models(*download_clusters)
        self.tokenizer = default_tokenizer_provider(model_args=self.pipeline_config.actor_infer.model_args)

        # Initialize global memory manager if enabled
        self.global_memory_manager = None
        if (
            hasattr(self.pipeline_config, "memory_config")
            and self.pipeline_config.memory_config
            and self.pipeline_config.memory_config.enable
        ):
            self._initialize_global_memory_manager()

        # Initialize memory training buffer
        self._memory_training_buffer: MemoryTrainingBuffer | None = None
        if getattr(self.pipeline_config, "memory_training_buffer_enable", False):
            self._memory_training_buffer = MemoryTrainingBuffer(
                max_size=int(getattr(self.pipeline_config, "memory_training_buffer_size", 0))
            )

        # Initialize rollout schedulers
        self.train_rollout_scheduler = (
            ray.remote(RolloutScheduler)
            .options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=ray.get_runtime_context().get_node_id(),
                    soft=False,
                )
            )
            .remote(
                config=self.pipeline_config,
                env_manager_config=self.pipeline_config.train_env_manager,
                resource_manager=self.resource_manager,
                infer_cluster=self.actor_infer,
                global_memory_manager=self.global_memory_manager,
                mode="train",
            )
        )
        self.val_rollout_scheduler = (
            ray.remote(RolloutScheduler)
            .options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=ray.get_runtime_context().get_node_id(),
                    soft=False,
                )
            )
            .remote(
                config=self.pipeline_config,
                env_manager_config=self.pipeline_config.val_env_manager,
                resource_manager=self.resource_manager,
                infer_cluster=self.actor_infer,
                global_memory_manager=self.global_memory_manager,
                mode="val",
            )
        )
        self.val_dataset_manager = GlobalDatasetManager.options(
            name=f"val_dataset_manager",
            get_if_exists=True,
            namespace=RAY_NAMESPACE,
        ).remote()
        self.actor_infer.initialize(pipeline_config=self.pipeline_config, blocking=True)

        self.running = RunningMoments()

        self.set_model_update_pair(
            src_cluster=self.memory_actor_train,
            tgt_cluster=self.memory_model_infer,
            frequency=self.pipeline_config.memory_config.memory_actor_train.model_update_frequency,
        )
        self.set_checkpoint_clusters(self.memory_actor_train)

        # Track training phases
        self._warmup_interval = getattr(self.pipeline_config.memory_config, "memory_warmup_interval", 0)
        self._current_warmup_step = 0
        self._warmup_complete = False
        self._memory_model_updated = False

        # Track step counters
        self._basic_step = 0  # Controls main loop iteration
        self._true_step = 0  # Counts memory actor updates

        # Track if we've done the first flush before training

        logger.info(f"RefactoredMemoryActorPipeline initialized with warmup_interval={self._warmup_interval}")

    def compute_non_memory_group_avg_scores(self, batch: DataProto) -> Dict[str, float]:
        """
        Compute average episode scores for each trajectory group where evolve_with_memory = False.

        Args:
            batch: DataProto containing:
                - batch.batch["scores"]: tensor of shape [batch_size, seq_len]
                - batch.non_tensor_batch["traj_group_id"]: numpy array of trajectory group IDs
                - batch.non_tensor_batch["episode_scores"]: numpy array of episode scores
                - batch.non_tensor_batch["evolve_with_memory"]: numpy array of booleans

        Returns:
            Dict mapping traj_group_id (str) -> average_score (float) for non-memory trajectories
        """
        traj_group_ids = batch.non_tensor_batch["traj_group_id"]
        episode_scores = batch.non_tensor_batch["episode_scores"]
        evolve_with_memory = batch.non_tensor_batch.get("evolve_with_memory", None)

        if evolve_with_memory is None:
            logger.warning("evolve_with_memory not found in batch, returning empty dict")
            return {}

        # Group scores by traj_group_id for non-memory trajectories
        group_scores = defaultdict(list)

        for i in range(len(traj_group_ids)):
            # Filter: only include trajectories where evolve_with_memory = False
            if not evolve_with_memory[i]:
                traj_group_id = str(traj_group_ids[i])  # Convert to string for consistent key type
                score = float(episode_scores[i])
                group_scores[traj_group_id].append(score)

        # Compute average score per trajectory group
        group_avg_scores = {}
        for traj_group_id, scores in group_scores.items():
            group_avg_scores[traj_group_id] = np.mean(scores)

        # Log statistics
        if group_avg_scores:
            logger.info(f"Non-memory trajectory group average scores: {group_avg_scores}")
            logger.info(f"Total non-memory trajectories: {sum(len(scores) for scores in group_scores.values())}")
        else:
            logger.info("No non-memory trajectories found in batch")

        return group_avg_scores

    def _collect_triggered_interactions_to_buffer(self, batch: DataProto) -> int:
        """
        Collect triggered interactions into the request_id-keyed memory training buffer.

        Returns:
            number of unique request_ids updated in this call
        """
        if self._memory_training_buffer is None:
            return 0

        triggered_interactions = batch.non_tensor_batch.pop("triggered_interactions", None)
        if triggered_interactions is None:
            return 0

        per_req_score_sum: Dict[str, float] = defaultdict(float)
        per_req_score_count: Dict[str, int] = defaultdict(int)
        per_req_interaction: Dict[str, DataProto] = {}

        for interaction in triggered_interactions:
            req_id = interaction.meta_info.get("request_id", None)
            if req_id is None:
                continue
            req_id = str(req_id)
            score = interaction.non_tensor_batch.get("episode_score", None)
            if score is not None:
                score_val = float(score)
                if score_val == 0:
                    score_val = -1
            else:
                score_val = 0.0

            per_req_score_sum[req_id] += score_val
            per_req_score_count[req_id] += 1
            per_req_interaction[req_id] = interaction  # keep latest

        for req_id, interaction in per_req_interaction.items():
            self._memory_training_buffer.add(
                request_id=req_id,
                interaction=interaction,
                score_sum=per_req_score_sum.get(req_id, 0.0),
                score_count=per_req_score_count.get(req_id, 0),
            )

        return len(per_req_interaction)

    def _record_batch_statistics(self, batch: DataProto) -> Dict[str, Any]:
        """
        Record batch statistics into memory_manager without adding to buffer.

        Args:
            batch: DataProto containing batch data

        Returns:
            Dictionary of recorded statistics
        """
        stats = {}

        # 1. Compute the Avg Score for Non-Memory Trajectories (per group)
        group_avg_scores = self.compute_non_memory_group_avg_scores(batch)
        stats["group_avg_scores"] = group_avg_scores

        # 2. Compute sum score and traj_count according to each env_tag
        tags = batch.non_tensor_batch.get("tags", None)
        env_tag_stats = {}
        if tags is not None:
            batch_grouped_by_tag = batch.group_by(keys="tags")
            for tag, tag_batch in batch_grouped_by_tag.items():
                episode_scores = tag_batch.non_tensor_batch.get("episode_scores", None)
                if episode_scores is not None:
                    score_sum = float(np.sum([float(s) for s in episode_scores]))
                    traj_count = len(episode_scores)
                    avg_score = score_sum / traj_count if traj_count > 0 else 0.0
                    # Use flat key format: "env_tag/{tag}/{metric_type}"
                    env_tag_stats[f"{tag}"] = {
                        "score_sum": score_sum,
                        "traj_count": traj_count,
                        "avg_score": avg_score,
                    }
            if env_tag_stats:
                logger.info(f"Env tag stats: {env_tag_stats}")

        stats["env_tag_stats"] = env_tag_stats

        # Record to memory manager
        if self.global_memory_manager is not None:
            self.global_memory_manager.record_actor_performance(env_tag_stats)

        return stats

    def _build_memory_batch_from_buffer_entries(
        self, entries: List[MemoryTrainingBufferEntry], global_step: int
    ) -> DataProto:
        """
        Build memory training batch from buffer entries.

        Similar to concat_triggered_interaction but uses buffer entries.
        """
        input_ids_list = []
        attention_mask_list = []
        prompt_mask_list = []
        response_mask_list = []
        infer_log_probs_list = []
        scores = []
        traj_group_ids = []
        env_ids = []
        request_ids = []
        has_infer_log_probs = False

        for entry in entries:
            interaction = entry.interaction
            if interaction.batch is None:
                continue
            input_ids_list.append(interaction.batch["input_ids"].squeeze(0))
            attention_mask_list.append(interaction.batch["attention_mask"].squeeze(0))
            prompt_mask_list.append(interaction.batch["prompt_mask"].squeeze(0))
            response_mask_list.append(interaction.batch["response_mask"].squeeze(0))
            scores.append(float(entry.score_mean))

            traj_group_ids.append(interaction.non_tensor_batch.get("traj_group_id", None))
            env_ids.append(interaction.non_tensor_batch.get("env_id", None))
            request_ids.append([str(entry.request_id)])

            if "infer_log_probs" in interaction.batch:
                has_infer_log_probs = True
                infer_log_probs_list.append(interaction.batch["infer_log_probs"].squeeze(0))

        if not input_ids_list:
            return None

        max_length_in_batch = max(ids.shape[0] for ids in input_ids_list)
        if (
            self.pipeline_config.memory_sequence_length is not None
            and max_length_in_batch > self.pipeline_config.memory_sequence_length
        ):
            max_length_in_batch = int(self.pipeline_config.memory_sequence_length)

        context_parallel_size = (
            self.pipeline_config.memory_config.memory_actor_train.strategy_args.strategy_config.get(
                "context_parallel_size", 1
            )
        )
        if context_parallel_size > 1:
            effective_cp_size = 2 * context_parallel_size
            pad_length = ((max_length_in_batch + effective_cp_size - 1) // effective_cp_size) * effective_cp_size
        else:
            pad_length = max_length_in_batch

        if (
            self.pipeline_config.memory_sequence_length is not None
            and pad_length > self.pipeline_config.memory_sequence_length
        ):
            if context_parallel_size > 1:
                effective_cp_size = 2 * context_parallel_size
                pad_length = (
                    int(self.pipeline_config.memory_sequence_length) // effective_cp_size
                ) * effective_cp_size
                pad_length = max(pad_length, 1)
            else:
                pad_length = int(self.pipeline_config.memory_sequence_length)

        # usage stats for repeat-control weights keyed by request_id (1:1 with UID)
        usage_stats = {}
        if (
            getattr(self.pipeline_config, "memory_repeat_control_enable", True)
            and self.global_memory_manager is not None
        ):
            try:
                usage_stats = self.global_memory_manager.get_usage_stats([str(e.request_id) for e in entries])
            except Exception as e:
                logger.warning(f"Failed to fetch memory usage stats: {e}")
                usage_stats = {}

        padded_input_ids = []
        padded_attention_mask = []
        padded_prompt_mask = []
        padded_response_mask = []
        padded_position_ids = []
        padded_scores = []
        padded_infer_log_probs = []
        sample_weights = []

        for i, (input_ids, attention_mask, prompt_mask, response_mask, score) in enumerate(
            zip(input_ids_list, attention_mask_list, prompt_mask_list, response_mask_list, scores)
        ):
            if input_ids.shape[0] > pad_length:
                input_ids = input_ids[:pad_length]
                attention_mask = attention_mask[:pad_length]
                prompt_mask = prompt_mask[:pad_length]
                response_mask = response_mask[:pad_length]

            position_ids = attention_mask.cumsum(dim=-1)
            score_tensor = torch.zeros(input_ids.shape[0], dtype=torch.float)
            score_tensor[-1] = float(score)

            req_id = str(entries[i].request_id)
            w = 1.0
            if getattr(self.pipeline_config, "memory_repeat_control_enable", True):
                power = float(getattr(self.pipeline_config, "memory_repeat_weight_power", 0.0))
                cooldown = int(getattr(self.pipeline_config, "memory_repeat_cooldown_steps", 0))
                train_count = int(usage_stats.get(req_id, {}).get("train_count", 0))
                last_step = int(usage_stats.get(req_id, {}).get("last_trained_step", -1))
                if cooldown > 0 and last_step >= 0 and (global_step - last_step) < cooldown:
                    w = 0.0
                elif power > 0:
                    w = float((1.0 + train_count) ** (-power))
            sample_weights.append(w)

            pad_token_id = self.memory_model_tokenizer.pad_token_id
            padded_input_ids.append(pad_to_length(input_ids.unsqueeze(0), length=pad_length, pad_value=pad_token_id))
            padded_attention_mask.append(pad_to_length(attention_mask.unsqueeze(0), length=pad_length, pad_value=0))
            padded_position_ids.append(pad_to_length(position_ids.unsqueeze(0), length=pad_length, pad_value=0))
            padded_prompt_mask.append(pad_to_length(prompt_mask.unsqueeze(0), length=pad_length, pad_value=0))
            padded_response_mask.append(pad_to_length(response_mask.unsqueeze(0), length=pad_length, pad_value=0))
            padded_scores.append(pad_to_length(score_tensor.unsqueeze(0), length=pad_length, pad_value=0))

            if has_infer_log_probs:
                inf = infer_log_probs_list[i]
                if inf.shape[0] > pad_length:
                    inf = inf[:pad_length]
                padded_infer_log_probs.append(pad_to_length(inf.unsqueeze(0), length=pad_length, pad_value=0))

        batch_dict = {
            "input_ids": torch.cat(padded_input_ids, dim=0),
            "attention_mask": torch.cat(padded_attention_mask, dim=0),
            "position_ids": torch.cat(padded_position_ids, dim=0),
            "prompt_mask": torch.cat(padded_prompt_mask, dim=0),
            "response_mask": torch.cat(padded_response_mask, dim=0),
            "scores": torch.cat(padded_scores, dim=0),
            "sample_weight": torch.tensor(sample_weights, dtype=torch.float32),
        }
        if has_infer_log_probs:
            batch_dict["infer_log_probs"] = torch.cat(padded_infer_log_probs, dim=0)

        memory_batch = DataProto()
        memory_batch.batch = TensorDict(batch_dict, batch_size=batch_dict["input_ids"].shape[0])
        memory_batch.non_tensor_batch = {
            "traj_group_id": np.array(traj_group_ids, dtype=object),
            "env_ids": np.array(env_ids, dtype=object),
            "memory_scores": np.array(scores, dtype=object),
            "request_id": np.array(request_ids, dtype=object),
        }
        memory_batch.meta_info = {"global_step": int(global_step)}
        return memory_batch

    def _train_memory_model(self, train_batch: DataProto, global_step: int) -> Dict[str, Any]:
        """
        Train memory model with the given batch.

        Args:
            train_batch: DataProto containing training data
            global_step: Current global step

        Returns:
            Dictionary of training metrics
        """
        metrics = {}

        # memory reference (consistent with legacy metrics/timers)
        if self.pipeline_config.reference_for_memory:
            with Timer(name="cal_ref_log_probs_memory", logger=None) as cal_timer:
                ref_log_probs_refs: List[ray.ObjectRef] = self.reference_for_memory.compute_log_probs(
                    train_batch, blocking=False
                )
                ref_log_probs = DataProto.materialize_concat(data_refs=ref_log_probs_refs)
                ref_log_probs.rename(old_keys="log_probs", new_keys="ref_log_probs")
                train_batch = train_batch.union(ref_log_probs)
                avg_ref_log_prob = masked_mean(
                    train_batch.batch["ref_log_probs"],
                    train_batch.batch["response_mask"][:, 1:],
                )
                metrics.update(reduce_metrics(ref_log_probs.meta_info.pop("metrics", {})))
                metrics.update({"critic/ref_log_prob_memory/mean": avg_ref_log_prob.item()})
            metrics["time/ref_log_probs_values_reward_memory"] = cal_timer.last

        # memory old log probability (consistent with legacy metrics/timers)
        with Timer(name="cal_old_log_probs_values_memory", logger=None) as cal_old_logpb_timer:
            train_batch.meta_info["is_offload_states"] = False
            old_log_probs_refs: List[ray.ObjectRef] = self.memory_actor_train.compute_log_probs(
                train_batch, blocking=False
            )
            memory_old_log_probs = DataProto.materialize_concat(data_refs=old_log_probs_refs)
            train_batch.batch["old_log_probs"] = memory_old_log_probs.batch["log_probs"]
            avg_old_log_prob = masked_mean(
                train_batch.batch["old_log_probs"],
                train_batch.batch["response_mask"][:, 1:],
            )
            metrics.update({"critic/old_log_prob_memory/mean": avg_old_log_prob.item()})
            agg_entropy = agg_loss(
                loss_mat=memory_old_log_probs.batch["entropy"],
                loss_mask=train_batch.batch["response_mask"][:, 1:],
                loss_agg_mode="token-mean",
            )
            metrics.update({"critic/entropy_memory/mean": agg_entropy.item()})
            metrics.update(reduce_metrics(memory_old_log_probs.meta_info.pop("metrics", {})))
        metrics["time/old_log_probs_values_memory"] = cal_old_logpb_timer.last

        # advantage + reward shaping (consistent with timer/metrics)
        with Timer(name="adv_memory", logger=None) as timer:
            train_batch = compute_response_level_rewards(
                batch=train_batch, pipeline_config=self.pipeline_config, is_memory_batch=True
            )
            metrics.update(reduce_metrics(train_batch.meta_info.pop("metrics", {})))

            if self.pipeline_config.reward_clip:
                reward_clip_frac = compute_clip_fraction(
                    values=train_batch.batch["response_level_rewards"],
                    clip_max=self.pipeline_config.reward_clip,
                    clip_min=-self.pipeline_config.reward_clip,
                )
                train_batch.batch["response_level_rewards"] = torch.clamp(
                    train_batch.batch["response_level_rewards"],
                    min=-self.pipeline_config.reward_clip,
                    max=self.pipeline_config.reward_clip,
                )
                metrics["critic/reward_clip_frac_memory"] = reward_clip_frac

            train_batch, memory_kl_metrics = apply_kl_penalty(
                data=train_batch,
                kl_ctrl=self.memory_kl_ctrl,
                kl_penalty=self.pipeline_config.kl_penalty,
            )
            train_batch = compute_advantage(
                data=train_batch,
                gamma=self.pipeline_config.gamma,
                lambd=self.pipeline_config.lambd,
                adv_estimator=self.pipeline_config.memory_adv_estimator,
                advantage_clip=self.pipeline_config.memory_advantage_clip,
                whiten_advantages=self.pipeline_config.memory_whiten_advantages,
                whiten_rewards=self.pipeline_config.memory_whiten_rewards,
                whiten_advantages_shift_mean=self.pipeline_config.memory_whiten_advantages_shift_mean,
            )
            metrics.update(reduce_metrics(train_batch.meta_info.pop("metrics", {})))

            if "sample_weight" in train_batch.batch and "advantages" in train_batch.batch:
                w = train_batch.batch["sample_weight"].to(train_batch.batch["advantages"].dtype)
                print("[DEBUG] apply sample weight to advantages: ", w)
                train_batch.batch["advantages"] = train_batch.batch["advantages"] * w.unsqueeze(1)

        if self.pipeline_config.reference_for_memory:
            metrics.update(memory_kl_metrics)
        metrics["time/adv_memory"] = timer.last

        # Megatron TP/PP/CP broadcast: `non_tensor_batch` (e.g., request_id) is NOT broadcast
        # unless explicitly enabled. `MemoryActorWorker.loss_func` groups by request_id, so we
        # must broadcast non-tensor fields for correctness.
        train_batch.meta_info["_broadcast_non_tensor_batch"] = True
        actor_train_metrics_refs = self.memory_actor_train.train_step(train_batch, blocking=False)
        actor_train_metrics: DataProto = DataProto.materialize_concat(data_refs=actor_train_metrics_refs)
        metrics.update(reduce_metrics(actor_train_metrics.meta_info.pop("metrics", {})))
        memory_data_metrics = compute_data_metrics_for_memory(batch=train_batch)
        metrics.update(memory_data_metrics)

        if self.global_memory_manager is not None:
            try:
                weights = None
                if "sample_weight" in train_batch.batch:
                    weights = train_batch.batch["sample_weight"].detach().cpu().tolist()
                rid_list = []
                for obj in train_batch.non_tensor_batch.get("request_id", []):
                    if isinstance(obj, (list, tuple, np.ndarray)):
                        rid_list.append([str(obj[0])])
                    else:
                        rid_list.append([str(obj)])
                self.global_memory_manager.record_training(
                    uids_list=rid_list,
                    global_step=int(global_step),
                    weights=weights,
                )
            except Exception as e:
                logger.warning(f"Failed to record memory training stats: {e}")

        return metrics

    def adjust_memory_batch(self, data: DataProto, mode="auto") -> DataProto:
        """
        Adjust memory batch size to match training requirements.

        ref: https://github.com/langfengQ/verl-agent/blob/e03bd502667c45172e8c093cc506db8438ae8ab5/agent_system/multi_turn_rollout/utils.py#L86
        """
        # Calculate required batch sizes
        memory_actor_train_train_bsz = (
            self.pipeline_config.memory_config.memory_actor_train.training_args.per_device_train_batch_size
            * self.pipeline_config.memory_config.memory_actor_train.training_args.gradient_accumulation_steps
            * self.memory_actor_train.dp_size
        )

        if self.pipeline_config.reference_for_memory:
            ref_infer_bsz = self.pipeline_config.memory_reference.infer_batch_size * self.reference_for_memory.dp_size
        else:
            ref_infer_bsz = 1

        # Calculate LCM to find common divisor
        size_divide = np.lcm.reduce(
            np.array(
                [
                    memory_actor_train_train_bsz,
                    ref_infer_bsz,
                ]
            )
        ).item()

        batch_size = data.batch.batch_size[0]
        threshold = batch_size % size_divide

        if threshold == 0:
            return data

        if mode == "auto":
            if threshold >= 0.5 * batch_size or batch_size // size_divide == 0:
                mode = "copy"
            else:
                mode = "delete"

        # Initialize metrics
        metrics = data.meta_info.get("metrics", {})
        metrics["system/memory_batch_add_count"] = 0
        metrics["system/memory_batch_remove_count"] = 0

        if mode == "delete":
            remove_indices = np.random.choice(batch_size, threshold, replace=False)
            remove_indices = np.sort(remove_indices)
            keep_mask = np.ones(batch_size, dtype=bool)
            keep_mask[remove_indices] = False
            keep_mask_tensor = torch.tensor(
                keep_mask,
                dtype=torch.bool,
                device=data.batch["input_ids"].device,
            )
            tensor_data = data.batch[keep_mask_tensor]
            non_tensor_data = {key: val[keep_mask] for key, val in data.non_tensor_batch.items()}
            adjusted_batch = DataProto(
                batch=tensor_data,
                non_tensor_batch=non_tensor_data,
                meta_info=data.meta_info,
            )
            metrics["system/memory_batch_remove_count"] = len(remove_indices)

        elif mode == "copy":
            to_add = size_divide - threshold
            dup_indices = (
                np.random.choice(batch_size, to_add, replace=True)
                if to_add > batch_size
                else np.random.choice(batch_size, to_add, replace=False)
            )
            dup_proto = data.select_idxs(dup_indices)
            adjusted_batch = DataProto.concat([data, dup_proto])
            metrics["system/memory_batch_add_count"] = to_add

        else:
            raise ValueError(f"Unsupported mode: {mode}. Must be 'delete', 'copy', or 'auto'")

        adjusted_batch.meta_info["metrics"] = metrics

        return adjusted_batch

    @torch.no_grad()
    def run(self):
        tps_timer = _Timer(window_size=5)

        # Check if warmup is already complete (e.g., from checkpoint resume)
        if self.global_memory_manager is not None and self.global_memory_manager.could_begin_interaction():
            self._warmup_complete = True
            logger.info("Warmup already complete (resumed from checkpoint or pre-warmed memory manager)")

        memory_actor_should_update = False
        for basic_step in range(self.pipeline_config.max_steps):
            if basic_step <= self.state.step:
                continue

            self._basic_step = basic_step
            logger.info(f"Pipeline basic_step {basic_step} start...")

            # Cache can_begin_interaction once per iteration to avoid redundant calls
            can_begin_interaction = (
                self.global_memory_manager is not None and self.global_memory_manager.could_begin_interaction()
            )
            logger.info(f"Phase: {'WARMUP' if not can_begin_interaction else 'BUFFER/TRAINING'}")

            metrics = {}

            with tps_timer:
                # Offload strategy: Only offload memory_actor_train when we're about to use it
                # During warmup, keep it offloaded (no offload needed)
                # During training, offload before model update
                offload_refs = []
                if can_begin_interaction:
                    # First time entering training phase: offload memory actor
                    offload_refs.extend(self.memory_actor_train.offload_states(blocking=False))
                    logger.info("Offloading memory actor states before first training")

                # Flush strategy: Only flush when can_begin_interaction and before first model update
                # During warmup, keep updates in background (no flush)
                flush_ref = None
                if can_begin_interaction:
                    logger.info("Flushing all pending memory updates before first training...")
                    flush_ref = self.global_memory_manager.flush_pending_updates_async(timeout=None)

                if flush_ref is not None:
                    flushed_count = ray.get(flush_ref)
                    logger.info(f"Flushed {flushed_count} pending memory updates before first training")

                    if self.global_memory_manager.memory_config.dedup_similarity_threshold is not None:
                        dedup_statistic = self.global_memory_manager.dedup_memory(
                            dry_run=False,
                        )
                        logger.info(f"Dedup memory statistics: {dedup_statistic}")
                    if (
                        self.pipeline_config.memory_config.memory_model is not None
                        and self.pipeline_config.memory_config.memory_model.enable_merging
                        and (
                            (
                                self.pipeline_config.memory_config.memory_model.merging_interval is not None
                                and basic_step % self.pipeline_config.memory_config.memory_model.merging_interval == 0
                            )
                            or self.pipeline_config.memory_config.memory_model.merging_size is not None
                        )
                    ):
                        logger.info(f"Merging memory at basic_step {basic_step}")
                        self.global_memory_manager.merge_memory()

                if offload_refs:
                    ray.get(offload_refs)

                # Memory model update (only after buffer is ready)
                if self.global_memory_manager is not None and memory_actor_should_update:
                    logger.info("Suspending rollout for memory model update...")
                    ray.get(self.train_rollout_scheduler.suspend.remote())
                    if self.pipeline_config.async_generation_ratio > 0:
                        self.actor_infer.stop_server()

                    self.global_memory_manager.suspend(basic_step)
                    model_update_metrics: Dict = self.model_update(basic_step)
                    metrics.update(model_update_metrics)

                    if self.pipeline_config.async_generation_ratio > 0:
                        self.actor_infer.start_server(
                            data=DataProto(
                                meta_info={
                                    "global_step": basic_step,
                                    "is_offload_states": False,
                                }
                            )
                        )
                    else:
                        self.actor_infer.start_server(
                            data=DataProto(
                                meta_info={
                                    "global_step": basic_step,
                                    "is_offload_states": True,
                                }
                            )
                        )

                    self.global_memory_manager.resume(basic_step)
                    memory_actor_should_update = False
                    self._memory_model_updated = True

                    # Phase 4: Validate after memory model update
                    logger.info("Phase 4: Running validation after memory model update...")
                    val_metrics = self.val(global_step=basic_step)
                    metrics.update(val_metrics)
                    logger.info(f"Post-update validation metrics: {val_metrics}")

                # Phase 1: Initial validation at the beginning
                self.actor_infer.start_server(
                    data=DataProto(
                        meta_info={
                            "global_step": basic_step,
                            "is_offload_states": True,
                        }
                    )
                )

                if basic_step == 0:
                    logger.info("Phase 1: Running initial validation...")
                    initial_val_metrics = self.val(global_step=0)
                    metrics.update(initial_val_metrics)
                    logger.info(f"Initial validation metrics: {initial_val_metrics}")

                batch: DataProto = DataProto()
                batch.meta_info = {"global_step": basic_step}

                # Regular validation based on eval_steps
                if basic_step % self.pipeline_config.eval_steps == 0:
                    val_metrics = self.val(global_step=basic_step)
                    metrics.update(val_metrics)

                # Rollout
                with Timer(name="rollout", logger=None) as rollout_timer:
                    batch.meta_info["is_offload_states"] = True
                    batch = ray.get(
                        self.train_rollout_scheduler.get_batch.remote(batch, self.pipeline_config.rollout_batch_size)
                    )
                    dump_rollout_trajectories(
                        self.pipeline_config.rollout_dump_dir,
                        basic_step,
                        batch,
                    )
                metrics["time/rollout"] = rollout_timer.last
                metrics.update(reduce_metrics(batch.meta_info.pop("metrics", {})))
                if not (self.pipeline_config.async_generation_ratio > 0):
                    self.actor_infer.stop_server()

                # Phase 1 & 2: Always record statistics
                batch_stats = self._record_batch_statistics(batch)
                metrics["phase"] = "warmup" if not can_begin_interaction else "buffer"

                # Phase 2: Warmup progress and buffer collection
                if not can_begin_interaction:
                    self._current_warmup_step += 1
                    logger.info(f"Warmup step {self._current_warmup_step}/{self._warmup_interval}")
                    if self.global_memory_manager is not None:
                        self.global_memory_manager.notify_memory_warmup()
                    if self.global_memory_manager.could_begin_interaction():
                        self._warmup_complete = True
                        logger.info("Warmup complete, notified memory manager, starting to add to buffer...")
                else:
                    # Phase 2: Add triggered interactions to buffer
                    if self._memory_training_buffer is not None:
                        updated = self._collect_triggered_interactions_to_buffer(batch)
                        metrics["memory/buffer_size"] = len(self._memory_training_buffer)
                        metrics["memory/buffer_updated_keys"] = updated
                        logger.info(f"Buffer size: {metrics['memory/buffer_size']}, added: {updated}")

                # Phase 3: Train memory model when buffer is ready
                if self._memory_training_buffer is not None and can_begin_interaction:
                    batch_size = int(getattr(self.pipeline_config, "memory_training_batch_size", 0))
                    if batch_size > 0 and self._memory_training_buffer.ready(batch_size):
                        logger.info(
                            f"Buffer ready with {len(self._memory_training_buffer)} entries, training memory model..."
                        )
                        buffer_entries = self._memory_training_buffer.pop_batch(batch_size)
                        train_batch = self._build_memory_batch_from_buffer_entries(
                            buffer_entries, global_step=basic_step
                        )
                        if train_batch is not None:
                            train_batch = self.adjust_memory_batch(
                                train_batch, mode=self.pipeline_config.batch_adjust_mode
                            )

                            # Train the memory model
                            memory_train_metrics = self._train_memory_model(train_batch, basic_step)
                            metrics.update(memory_train_metrics)

                            # Increment true_step (memory actor update counter)
                            self._true_step += 1
                            metrics["step/true_step"] = self._true_step

                            # Schedule model update for next iteration
                            memory_actor_should_update = True

                tps_timer.push_units_processed(n=torch.sum(batch.batch["attention_mask"]).detach().item())

            metrics["system/tps"] = tps_timer.mean_throughput
            metrics["system/samples"] = (basic_step + 1) * self.pipeline_config.rollout_batch_size
            metrics["step/basic_step"] = basic_step
            metrics["step/true_step"] = self._true_step
            metrics["phase/warmup_complete"] = can_begin_interaction
            metrics["phase/memory_model_updated"] = self._memory_model_updated

            self.state.step = basic_step
            self.state.log_history.append(metrics)
            self.do_checkpoint(global_step=basic_step)
            self.tracker.log(values=metrics, step=basic_step)
            if self.global_memory_manager is not None:
                self.global_memory_manager.notify_update()

            if (basic_step + 1) % self.pipeline_config.memory_config.memory_save_interval == 0:
                save_paths = os.path.join(
                    self.pipeline_config.memory_config.memory_save_path,
                    f"memory_{basic_step + 1}.json",
                )
                try:
                    success = self.global_memory_manager.save_state(save_paths)
                    if success:
                        logger.info(f"Memory saved to {save_paths}")
                    else:
                        logger.warning(f"Error saving memory")
                except Exception as e:
                    logger.warning(f"Error saving memory: {e}")

            if basic_step % self.pipeline_config.logging_steps == 0:
                if int(os.environ.get("RAY_PROFILING", "0")):
                    timeline_dir = os.path.join(self.pipeline_config.profiler_output_dir, "timeline")
                    os.makedirs(timeline_dir, exist_ok=True)
                    ray.timeline(
                        filename=os.path.join(
                            timeline_dir,
                            f"timeline-step-{basic_step}.json",
                        ),
                    )

                # Log batch samples
                log_res = []
                batch_grouped = batch.group_by(keys="traj_id")
                for group_name, group_batch in batch_grouped.items():
                    prompt_mask = group_batch.batch["prompt_mask"]
                    non_prompt_mask = (
                        torch.logical_not(group_batch.batch["prompt_mask"]) * group_batch.batch["attention_mask"]
                    )
                    input_ids = group_batch.batch["input_ids"]
                    prompt_ids_list = [input_ids[i][mask.bool()] for i, mask in enumerate(prompt_mask)]
                    response_ids_list = [input_ids[i][mask.bool()] for i, mask in enumerate(non_prompt_mask)]
                    prompts = self.tokenizer.batch_decode(prompt_ids_list, skip_special_tokens=False)
                    responses = self.tokenizer.batch_decode(response_ids_list, skip_special_tokens=False)
                    episode_scores = group_batch.non_tensor_batch["episode_scores"].tolist()
                    step_scores = group_batch.non_tensor_batch["step_scores"].tolist()
                    if not isinstance(step_scores[0], float):
                        step_scores = [t.tolist() for t in step_scores]

                    log_item = []
                    for prompt, response, episode_score, step_score in zip(
                        prompts, responses, episode_scores, step_scores
                    ):
                        log_item.append(
                            {
                                "prompt": prompt,
                                "response": response,
                                "episode_score": episode_score,
                                "step_score": step_score,
                            }
                        )
                    log_res.append(log_item)
                    if len(log_res) >= 10:
                        break
                logger.info(json.dumps(log_res, ensure_ascii=False))
                logger.info(json.dumps(metrics, ensure_ascii=False))

            logger.info(f"Pipeline basic_step {basic_step} finished")

        # Cleanup
        ray.get(
            [
                self.train_rollout_scheduler.shutdown.remote(),
                self.val_rollout_scheduler.shutdown.remote(),
            ]
        )

        if self.global_memory_manager is not None:
            try:
                self.global_memory_manager.shutdown()
                logger.info("Global memory manager cleaned up successfully")
            except Exception as e:
                logger.warning(f"Error cleaning up global memory manager: {e}")

        logger.info("Pipeline complete!")


def get_memory_scores(batch: DataProto) -> torch.Tensor:
    batch_group_by_traj: Dict[str, DataProto] = batch.group_by(keys="traj_group_id")
    scores = []
    for traj_id, traj_batch in batch_group_by_traj.items():
        episode_scores = traj_batch.non_tensor_batch["memory_scores"][0]
        scores.append(episode_scores)
    return torch.tensor(scores, dtype=torch.float32)


def compute_data_metrics_for_memory(batch):
    memory_scores = get_memory_scores(batch)
    sequence_reward = batch.batch["token_level_rewards"].sum(-1)
    advantages = batch.batch["advantages"]

    # fix: https://github.com/volcengine/verl/pull/60
    prompt_mask = batch.batch["prompt_mask"].bool()
    response_mask = batch.batch["response_mask"][:, 1:].bool()
    prompt_lengths = prompt_mask.sum(-1).float()  # (batch_size,)
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    returns = batch.batch["returns"]
    non_prompt_mask = torch.logical_not(batch.batch["prompt_mask"]).float().sum(-1)

    metrics = {
        # score, sequence_score from env
        "critic/memory_scores/mean": torch.mean(memory_scores).detach().item(),
        "critic/memory_scores/max": torch.max(memory_scores).detach().item(),
        "critic/memory_scores/min": torch.min(memory_scores).detach().item(),
        # reward
        "critic/memory_rewards/mean": torch.mean(sequence_reward).detach().item(),
        "critic/memory_rewards/max": torch.max(sequence_reward).detach().item(),
        "critic/memory_rewards/min": torch.min(sequence_reward).detach().item(),
        # adv
        "critic/memory_advantages/mean": masked_mean(advantages, response_mask).detach().item(),
        "critic/memory_advantages/max": (
            torch.max(advantages[response_mask]).detach().item() if response_mask.sum() > 0 else 0.0
        ),
        "critic/memory_advantages/min": (
            torch.min(advantages[response_mask]).detach().item() if response_mask.sum() > 0 else 0.0
        ),
        # returns
        "critic/memory_returns/mean": masked_mean(returns, response_mask).detach().item(),
        "critic/memory_returns/max": (
            torch.max(returns[response_mask]).detach().item() if response_mask.sum() > 0 else 0.0
        ),
        "critic/memory_returns/min": (
            torch.min(returns[response_mask]).detach().item() if response_mask.sum() > 0 else 0.0
        ),
        # response length
        "tokens/memory_response_length/mean": torch.mean(response_length).detach().item(),
        "tokens/memory_response_length/max": torch.max(response_length).detach().item(),
        "tokens/memory_response_length/min": torch.min(response_length).detach().item(),
        # prompt length
        "tokens/memory_prompt_length/mean": torch.mean(prompt_lengths).detach().item(),
        "tokens/memory_prompt_length/max": torch.max(prompt_lengths).detach().item(),
        "tokens/memory_prompt_length/min": torch.min(prompt_lengths).detach().item(),
        # non-prompt length
        "tokens/memory_non_prompt_length/mean": torch.mean(non_prompt_mask).detach().item(),
        "tokens/memory_non_prompt_length/max": torch.max(non_prompt_mask).detach().item(),
        "tokens/memory_non_prompt_length/min": torch.min(non_prompt_mask).detach().item(),
    }

    return metrics
