import json
import math
import os
import random
import traceback
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import ray
import torch
from codetiming import Timer
from ray.util.timer import _Timer
from tensordict import TensorDict

from roll.distributed.executor.cluster import Cluster
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.scheduler.resource_manager import ResourceManager
from roll.models.model_providers import default_tokenizer_provider
from roll.pipeline.agentic.agentic_pipeline import AgenticPipeline, compute_data_metrics
from roll.pipeline.agentic.multi_agentic_config import MemoryActorConfig
from roll.pipeline.agentic.utils import (
    compute_discounted_returns,
    compute_response_level_rewards,
    dump_rollout_trajectories,
)
from roll.utils.bool_utils import to_bool_numpy, to_bool_tensor
from roll.utils.constants import IGNORE_INDEX
from roll.utils.functionals import (
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


def _convert_to_json_serializable(obj):
    """Convert numpy arrays and other non-serializable types to JSON serializable Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.detach().cpu().item()
        return obj.detach().cpu().tolist()
    elif isinstance(obj, dict):
        return {k: _convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return [_convert_to_json_serializable(item) for item in obj]
    return obj


def _lcm(a: int, b: int) -> int:
    if a <= 0 or b <= 0:
        raise ValueError(f"lcm inputs must be positive, got {a=}, {b=}")
    return (a // math.gcd(a, b)) * b


@dataclass
class MemoryTrainingBufferEntry:
    request_id: str
    interaction: DataProto
    score_sum: float = 0.0
    score_count: int = 0

    def add(self, interaction: DataProto, score_sum: float, score_count: int) -> None:
        self.interaction = interaction
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


class MemoryActorPipeline(AgenticPipeline):
    def __init__(self, pipeline_config: MemoryActorConfig):
        self.pipeline_config = pipeline_config
        self.pipeline_config.set_max_steps(max_steps=self.pipeline_config.max_steps)
        self.resource_manager = ResourceManager(
            num_nodes=self.pipeline_config.num_nodes,
            num_gpus_per_node=self.pipeline_config.num_gpus_per_node,
        )

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

        init_refs: List[ray.ObjectRef] = []
        memory_infer_refs = self.memory_model_infer.initialize(blocking=False)
        if memory_infer_refs is not None:
            init_refs.extend(memory_infer_refs)
        if init_refs:
            ray.get(init_refs)

        # After the vLLM-based memory-model is fully initialized, it is safe to initialize the remaining clusters.
        init_refs = []
        init_refs.extend(self.memory_actor_train.initialize(pipeline_config=self.pipeline_config, blocking=False))
        if self.pipeline_config.reference_for_memory:
            init_refs.extend(
                self.reference_for_memory.initialize(pipeline_config=self.pipeline_config, blocking=False)
            )
        if init_refs:
            ray.get(init_refs)
        logger.info("Memory model infer and memory actor train clusters initialized")

        super().__init__(pipeline_config)

        self.set_model_update_pair(
            src_cluster=self.memory_actor_train,
            tgt_cluster=self.memory_model_infer,
            frequency=self.pipeline_config.max_steps,
        )
        self.set_checkpoint_clusters(self.memory_actor_train)

        self._memory_training_buffer: MemoryTrainingBuffer | None = None
        if getattr(self.pipeline_config, "memory_training_buffer_enable", False):
            self._memory_training_buffer = MemoryTrainingBuffer(
                max_size=int(getattr(self.pipeline_config, "memory_training_buffer_size", 0))
            )

        self._warmup_interval = getattr(self.pipeline_config.memory_config, "memory_warmup_interval", 0)
        self._warmup_complete = False

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

    def extract_successful_memory_trajectories(
        self,
        batch: DataProto,
        group_avg_scores: Dict[str, float],
    ) -> Optional[DataProto]:
        """
        Extract memory trajectories with scores > non-memory average within same group.

        Args:
            batch: DataProto containing trajectories
            group_avg_scores: Dict mapping traj_group_id -> avg score of non-memory trajectories

        Returns:
            DataProto with successful memory trajectories (memory still attached),
            or None if no successful trajectories found
        """
        traj_group_ids = batch.non_tensor_batch.get("traj_group_id", None)
        evolve_with_memory = batch.non_tensor_batch.get("evolve_with_memory", None)
        episode_scores = batch.non_tensor_batch.get("episode_scores", None)

        if evolve_with_memory is None or episode_scores is None or traj_group_ids is None:
            logger.warning("Missing required fields for extraction")
            return None

        # Find indices where evolve_with_memory=True AND score > group_avg
        successful_indices = []
        for i in range(len(batch)):
            if evolve_with_memory[i]:
                group_id = str(traj_group_ids[i])
                avg_score = group_avg_scores.get(group_id, 0.0)
                if float(episode_scores[i]) > avg_score:
                    successful_indices.append(i)

        if not successful_indices:
            logger.info("No successful memory trajectories found above group average")
            return None

        logger.info(f"Extracted {len(successful_indices)} successful memory trajectories")
        return batch.select_idxs(successful_indices)

    def strip_memory_from_batch(self, batch: DataProto) -> DataProto:
        """
        Strip memory tokens from trajectories by using pre-computed stripped tensors.

        The env_manager already creates stripped versions of input_ids and masks
        when evolve_with_memory=True. This method simply swaps the original tensors
        with the stripped ones.

        Args:
            batch: DataProto with memory trajectories (must contain stripped_* tensors)

        Returns:
            DataProto with memory stripped from trajectories
        """
        if "stripped_input_ids" not in batch.batch:
            logger.warning(
                "stripped_input_ids not found in batch, cannot strip memory. "
                "Make sure the env_manager is creating stripped tensors."
            )
            return batch

        # Create new batch with stripped tensors
        stripped_batch = DataProto()
        stripped_batch.batch = batch.batch.clone()
        stripped_batch.non_tensor_batch = batch.non_tensor_batch.copy()
        stripped_batch.meta_info = batch.meta_info.copy()

        # Replace original tensors with stripped versions
        stripped_batch.batch["input_ids"] = batch.batch["stripped_input_ids"]
        stripped_batch.batch["attention_mask"] = batch.batch["stripped_attention_mask"]
        stripped_batch.batch["position_ids"] = batch.batch["stripped_position_ids"]
        stripped_batch.batch["prompt_mask"] = batch.batch["stripped_prompt_mask"]
        stripped_batch.batch["response_mask"] = batch.batch["stripped_response_mask"]
        # Keep original question_mask and scores (rewards) unchanged

        # Mark as stripped
        stripped_batch.non_tensor_batch["memory_stripped"] = np.array([True] * len(batch), dtype=object)

        logger.info(f"Stripped memory from {len(batch)} trajectories using pre-computed tensors")
        return stripped_batch

    def prepare_sft_batch(self, batch: DataProto) -> DataProto:
        """
        Prepare batch for SFT training by creating labels.

        Labels are created with IGNORE_INDEX (-100) for:
        - Prompt tokens (marked by prompt_mask=1)
        - Padding tokens (marked by attention_mask=0)

        Actual token IDs are used for response tokens only.

        Args:
            batch: DataProto containing 'input_ids', 'prompt_mask', 'attention_mask'

        Returns:
            DataProto with 'labels' added to batch
        """
        input_ids = batch.batch["input_ids"]
        prompt_mask = batch.batch["prompt_mask"]  # 1 for prompt, 0 for response
        attention_mask = batch.batch["attention_mask"]  # 1 for valid tokens, 0 for padding

        # Create a combined mask: 1 for tokens that should be IGNORED (prompt OR padding)
        # prompt_mask=1 indicates prompt, attention_mask=0 indicates padding
        ignore_mask = prompt_mask.bool() | (~attention_mask.bool())

        # Create labels: IGNORE_INDEX for prompt/padding, input_ids for response
        labels = torch.where(ignore_mask, torch.full_like(input_ids, IGNORE_INDEX), input_ids)

        batch.batch["labels"] = labels
        return batch

    def _adjust_batch_for_dp(self, batch: DataProto, dp_size: int, mode: str = "auto") -> DataProto:
        """
        Adjust batch size to be divisible by DP size for even distribution.

        Args:
            batch: DataProto to adjust
            dp_size: Data parallelism size
            mode: Adjustment mode - "auto", "copy", or "delete"

        Returns:
            Adjusted DataProto with size divisible by dp_size
        """
        if dp_size <= 1:
            return batch

        batch_size = len(batch)
        remainder = batch_size % dp_size

        if remainder == 0:
            return batch

        # Calculate adjustment
        to_add = dp_size - remainder

        if mode == "auto":
            # If remainder is large or batch is small, copy; otherwise delete
            if remainder >= 0.5 * batch_size or batch_size // dp_size == 0:
                mode = "copy"
            else:
                mode = "delete"

        if mode == "copy":
            # Duplicate random samples
            dup_indices = (
                np.random.choice(batch_size, to_add, replace=True)
                if to_add > batch_size
                else np.random.choice(batch_size, to_add, replace=False)
            )
            dup_proto = batch.select_idxs(dup_indices)
            adjusted_batch = DataProto.concat([batch, dup_proto])
            logger.info(
                f"Adjusted batch for DP: copied {len(dup_indices)} samples, size {batch_size} -> {len(adjusted_batch)}"
            )

        elif mode == "delete":
            # Remove random samples
            remove_indices = np.random.choice(batch_size, remainder, replace=False)
            remove_indices = np.sort(remove_indices)
            keep_mask = np.ones(batch_size, dtype=bool)
            keep_mask[remove_indices] = False
            keep_mask_tensor = torch.tensor(
                keep_mask,
                dtype=torch.bool,
                device=batch.batch["input_ids"].device,
            )
            tensor_data = batch.batch[keep_mask_tensor]
            non_tensor_data = {key: val[keep_mask] for key, val in batch.non_tensor_batch.items()}
            adjusted_batch = DataProto(
                batch=tensor_data,
                non_tensor_batch=non_tensor_data,
                meta_info=batch.meta_info,
            )
            logger.info(
                f"Adjusted batch for DP: removed {len(remove_indices)} samples, size {batch_size} -> {len(adjusted_batch)}"
            )
        else:
            raise ValueError(f"Unsupported mode: {mode}. Must be 'copy', 'delete', or 'auto'")

        return adjusted_batch

    def compute_base_batch_group_avg_entropy(self, batch: DataProto) -> Tuple[Dict[str, float], Dict[str, float]]:
        group_by_traj_group_id = batch.group_by(keys="traj_group_id")
        group_non_memory_entropies = {}
        traj_memory_entropies = {}
        for traj_id, traj_group in group_by_traj_group_id.items():
            traj_group_by_memory = traj_group.group_by(keys="evolve_with_memory")

            # Calculate baseline entropy (without memory)
            if "False" in traj_group_by_memory:
                non_memory_group = traj_group_by_memory["False"]
                old_question_entropy = non_memory_group.batch["old_question_entropy"]
                group_non_memory_entropies[traj_id] = agg_loss(
                    loss_mat=old_question_entropy,
                    loss_mask=non_memory_group.batch["question_mask"][:, 1:],
                    loss_agg_mode="token-mean",
                )

            # Calculate memory entropy (with memory)
            if "True" in traj_group_by_memory:
                memory_group = traj_group_by_memory["True"]
                evolve_with_memory_group = memory_group.group_by(keys="env_ids")
                for env_id, env_group in evolve_with_memory_group.items():
                    # Use old_question_entropy instead of question_entropy as it was renamed in batch
                    question_entropy = env_group.batch["old_question_entropy"]
                    traj_memory_entropies[env_id] = agg_loss(
                        loss_mat=question_entropy,
                        loss_mask=env_group.batch["question_mask"][:, 1:],
                        loss_agg_mode="token-mean",
                    )

        return group_non_memory_entropies, traj_memory_entropies

    def concat_triggered_interaction(self, batch: DataProto, group_avg_scores: Dict[str, float]) -> DataProto:
        """
        Concat the triggered interaction, pad to max_length, compute scores and position_ids.

        Args:
            batch: DataProto containing triggered_interactions in non_tensor_batch
            group_avg_scores: Dict mapping traj_group_id to average scores from non-memory trajectories

        Returns:
            DataProto with padded and concatenated memory interactions, or None if no interactions
        """
        triggered_interactions = batch.non_tensor_batch.pop("triggered_interactions", None)
        if triggered_interactions is None:
            logger.info("No triggered interactions found in batch")
            return None

        request_to_scores = defaultdict(list)
        request_to_traj_group_ids = defaultdict(list)
        request_to_env_ids = defaultdict(list)
        request_to_triggered_interactions = {}

        for interaction in triggered_interactions:
            score = interaction.non_tensor_batch.get("episode_score", None)
            traj_group_id = interaction.non_tensor_batch.get("traj_group_id", None)
            req_id = interaction.meta_info.get("request_id", None)
            env_id = interaction.non_tensor_batch.get("env_id", None)

            assert req_id is not None, "Request ID not found in batch"
            request_to_scores[req_id].append(score)
            request_to_traj_group_ids[req_id].append(traj_group_id)
            request_to_triggered_interactions[req_id] = interaction
            request_to_env_ids[req_id].append(env_id)

        assert (
            len(request_to_scores) == len(request_to_traj_group_ids) == len(request_to_triggered_interactions)
        ), "Request IDs, scores, and triggered interactions are not the same length"

        # Extract tensors from each triggered interaction
        input_ids_list = []
        attention_mask_list = []
        prompt_mask_list = []
        response_mask_list = []
        infer_log_probs_list = []
        scores = []
        traj_group_ids = []
        env_ids = []
        request_id = []
        has_infer_log_probs = "infer_log_probs" in triggered_interactions[0].batch

        for req_id, interaction in request_to_triggered_interactions.items():
            input_ids_list.append(interaction.batch["input_ids"].squeeze(0))  # Remove batch dim
            attention_mask_list.append(interaction.batch["attention_mask"].squeeze(0))
            prompt_mask_list.append(interaction.batch["prompt_mask"].squeeze(0))
            response_mask_list.append(interaction.batch["response_mask"].squeeze(0))
            scores.append(np.mean(request_to_scores[req_id]))
            traj_group_ids.append(request_to_traj_group_ids[req_id])
            env_ids.append(request_to_env_ids[req_id])
            request_id.append(req_id)
            if has_infer_log_probs:
                infer_log_probs_list.append(interaction.batch["infer_log_probs"].squeeze(0))

        max_length_in_batch = max(ids.shape[0] for ids in input_ids_list)

        context_parallel_size = (
            self.pipeline_config.memory_config.memory_actor_train.strategy_args.strategy_config.get(
                "context_parallel_size", 1
            )
        )
        tensor_model_parallel_size = (
            self.pipeline_config.memory_config.memory_actor_train.strategy_args.strategy_config.get(
                "tensor_model_parallel_size", 1
            )
        )
        cp_size = int(context_parallel_size)
        tp_size = int(tensor_model_parallel_size)
        effective_cp_size = 2 * cp_size if cp_size > 1 else 1
        seq_align = _lcm(effective_cp_size, max(tp_size * cp_size, 1))
        pad_length = ((max_length_in_batch + seq_align - 1) // seq_align) * seq_align

        padded_input_ids = []
        padded_attention_mask = []
        padded_prompt_mask = []
        padded_response_mask = []
        padded_position_ids = []
        padded_scores = []
        padded_infer_log_probs = []
        memory_scores = []

        for i, (
            input_ids,
            attention_mask,
            prompt_mask,
            response_mask,
            score,
            traj_group_id,
            env_id,
        ) in enumerate(
            zip(
                input_ids_list,
                attention_mask_list,
                prompt_mask_list,
                response_mask_list,
                scores,
                traj_group_ids,
                env_ids,
            )
        ):
            # Compute position_ids from attention_mask (before padding)
            position_ids = attention_mask.cumsum(dim=-1)

            # Create score tensor: all zeros except last position = adjusted episode_score
            seq_len = input_ids.shape[0]
            score_tensor = torch.zeros(seq_len, dtype=torch.float)

            traj_group_id_str = [str(i[0]) for i in traj_group_id]
            avg_score = np.mean([group_avg_scores.get(i, 0.0) for i in traj_group_id_str])
            # adjusted_score = (
            #     float(score)
            #     - avg_score
            #     - self.pipeline_config.memory_score_penalty
            # )
            current_score = float(score)
            if current_score < avg_score:
                adjusted_score = -1
            elif current_score > avg_score:
                adjusted_score = 1
            else:
                adjusted_score = 0
            score_tensor[-1] = adjusted_score

            assert (
                self.memory_model_tokenizer.pad_token_id is not None
            ), "Pad token ID not found in memory model tokenizer"
            pad_token_id = self.memory_model_tokenizer.pad_token_id

            memory_scores.append(adjusted_score)
            padded_input_ids.append(
                pad_to_length(
                    input_ids.unsqueeze(0),
                    length=pad_length,
                    pad_value=pad_token_id,
                )
            )
            padded_attention_mask.append(
                pad_to_length(
                    attention_mask.unsqueeze(0),
                    length=pad_length,
                    pad_value=0,
                )
            )
            padded_position_ids.append(
                pad_to_length(
                    position_ids.unsqueeze(0),
                    length=pad_length,
                    pad_value=0,
                )
            )
            padded_prompt_mask.append(pad_to_length(prompt_mask.unsqueeze(0), length=pad_length, pad_value=0))
            padded_response_mask.append(
                pad_to_length(
                    response_mask.unsqueeze(0),
                    length=pad_length,
                    pad_value=0,
                )
            )
            padded_scores.append(
                pad_to_length(
                    score_tensor.unsqueeze(0),
                    length=pad_length,
                    pad_value=0,
                )
            )

            if has_infer_log_probs:
                padded_infer_log_probs.append(
                    pad_to_length(
                        infer_log_probs_list[i].unsqueeze(0),
                        length=pad_length,
                        pad_value=0,
                    )
                )

        # Concatenate all samples into batch dimension
        batch_input_ids = torch.cat(padded_input_ids, dim=0)
        batch_attention_mask = torch.cat(padded_attention_mask, dim=0)
        batch_position_ids = torch.cat(padded_position_ids, dim=0)
        batch_prompt_mask = torch.cat(padded_prompt_mask, dim=0)
        batch_response_mask = torch.cat(padded_response_mask, dim=0)
        batch_scores = torch.cat(padded_scores, dim=0)
        memory_scores = np.array(memory_scores, dtype=object)

        # Create TensorDict for batch
        batch_dict = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "position_ids": batch_position_ids,
            "prompt_mask": batch_prompt_mask,
            "response_mask": batch_response_mask,
            "scores": batch_scores,
        }

        if has_infer_log_probs:
            batch_infer_log_probs = torch.cat(padded_infer_log_probs, dim=0)
            batch_dict["infer_log_probs"] = batch_infer_log_probs

        # Create DataProto
        memory_batch = DataProto()
        memory_batch.batch = TensorDict(batch_dict, batch_size=batch_input_ids.shape[0])

        # Add non_tensor_batch fields
        memory_batch.non_tensor_batch = {
            "traj_group_id": np.array(traj_group_ids, dtype=object),
            "env_ids": np.array(env_ids, dtype=object),
            "memory_scores": memory_scores,
            "request_id": np.array(request_id, dtype=object),
        }

        return memory_batch

    def adjust_memory_batch(self, data: DataProto, mode="auto") -> DataProto:
        """
        ref: https://github.com/langfengQ/verl-agent/blob/e03bd502667c45172e8c093cc506db8438ae8ab5/agent_system/multi_turn_rollout/utils.py#L86
        """
        # Calculate required batch sizes
        memory_actor_train_train_bsz = (
            self.pipeline_config.memory_config.memory_actor_train.training_args.per_device_train_batch_size
            * self.pipeline_config.memory_config.memory_actor_train.training_args.gradient_accumulation_steps
            * self.memory_actor_train.dp_size
        )
        memory_actor_train_infer_bsz = (
            self.pipeline_config.memory_config.memory_model.infer_batch_size * self.memory_model_infer.dp_size
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
                    memory_actor_train_infer_bsz,
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

    def _compute_entropy_bonus(
        self,
        entropy_gain: float,
        entropy_wo_mem: float,
        entropy_w_mem: float,
    ) -> float:
        """
        Compute stabilized entropy bonus using various normalization strategies.

        Args:
            entropy_gain: Raw entropy reduction (entropy_wo_mem - entropy_w_mem)
            entropy_wo_mem: Entropy without memory (baseline)
            entropy_w_mem: Entropy with memory

        Returns:
            Stabilized entropy bonus in range compatible with 0-1 rewards
        """
        weight = self.pipeline_config.entropy_reduction_weight
        strategy = self.pipeline_config.entropy_bonus_strategy

        if strategy == "relative":
            # Relative Percentage Reduction (recommended for stability)
            # Normalizes by baseline entropy, making it scale-invariant
            # Example: 50% reduction → 0.5, 20% increase → -0.2
            if entropy_wo_mem > 1e-6:  # Avoid division by zero
                relative_reduction = entropy_gain / entropy_wo_mem
                relative_reduction = np.clip(
                    relative_reduction,
                    self.pipeline_config.entropy_bonus_clip_min,
                    self.pipeline_config.entropy_bonus_clip_max,
                )
                bonus = weight * relative_reduction
            else:
                bonus = 0.0

        elif strategy == "tanh":
            # Tanh normalization: smooth non-linear scaling, bounded to [-1, 1]
            # Good for naturally handling outliers without hard clipping
            bonus = weight * np.tanh(entropy_gain)

        elif strategy == "sigmoid":
            # Sigmoid with adjustable temperature: bounded to [0, 1] range
            # Emphasizes positive gains, de-emphasizes negative ones
            # Output range scaled to [-1, 1] for consistency
            temperature = self.pipeline_config.entropy_sigmoid_temperature
            sigmoid_val = 1.0 / (1.0 + np.exp(-entropy_gain / temperature))
            bonus = weight * (sigmoid_val - 0.5) * 2  # Scale to [-1, 1]

        elif strategy == "asymmetric_clip":
            # Asymmetric clipping: reward positive gains more than punish negative
            # Example: clips negative to -0.2 but positive to 1.0
            # Useful when you want to encourage exploration without heavy penalty
            bonus = weight * np.clip(
                entropy_gain,
                self.pipeline_config.entropy_asymmetric_negative_clip,
                self.pipeline_config.entropy_bonus_clip_max,
            )

        elif strategy == "log_space":
            # Log-space transformation for very large entropy values
            # Compresses large values while preserving sign
            # Good when entropy magnitudes vary widely
            sign = np.sign(entropy_gain)
            bonus = weight * sign * np.log1p(abs(entropy_gain))

        else:
            raise ValueError(f"Unknown entropy_bonus_strategy: {strategy}")

        return bonus

    def compute_entropy_memory_scores(self, memory_batch: DataProto, base_batch: DataProto) -> DataProto:
        """
        Compute entropy-based memory scores for memory co-training.

        Returns:
            Updated memory_batch with entropy-based scores
        """
        # Check minimum sample requirement
        batch_size = (
            memory_batch.batch.batch_size[0]
            if hasattr(memory_batch.batch, "batch_size")
            else len(memory_batch.batch.get("input_ids", []))
        )
        if batch_size < self.pipeline_config.entropy_min_samples:
            logger.warning(
                f"Insufficient samples for entropy computation: {batch_size} < {self.pipeline_config.entropy_min_samples}"
            )
            return memory_batch

        assert "old_question_entropy" in base_batch.batch, "old_question_entropy not found in base_batch"
        assert "question_mask" in base_batch.batch, "question_mask not found in base_batch"

        logger.info("Computing entropy-based memory scores")
        group_avg_entropies_wo_mem, traj_avg_entropies_w_mem = self.compute_base_batch_group_avg_entropy(base_batch)

        try:
            env_ids = memory_batch.non_tensor_batch.get("env_ids", None)
            traj_group_ids = memory_batch.non_tensor_batch.get("traj_group_id", None)

            if env_ids is None or traj_group_ids is None:
                logger.warning("env_ids or traj_group_id not found in memory_batch, skipping entropy reward")
                return memory_batch

            if isinstance(env_ids, np.ndarray):
                env_ids = env_ids.tolist()
            if isinstance(traj_group_ids, np.ndarray):
                traj_group_ids = traj_group_ids.tolist()

            batch_size = (
                memory_batch.batch.batch_size[0] if hasattr(memory_batch.batch, "batch_size") else len(env_ids)
            )
            entropy_gains = []
            entropy_bonus_list = []

            for i in range(batch_size):
                request_id = str(memory_batch.non_tensor_batch["request_id"][i][0])
                sub_env_ids = str(j for j in env_ids[i])
                sub_traj_group_ids = str(j for j in traj_group_ids[i])

                entropy_wo_mem = []
                entropy_w_mem = []

                for sub_env_id in sub_env_ids:
                    entropy_w_mem.append(traj_avg_entropies_w_mem.get(sub_env_id, 0.0))
                for sub_traj_group_id in sub_traj_group_ids:
                    entropy_wo_mem.append(group_avg_entropies_wo_mem.get(sub_traj_group_id, 0.0))

                entropy_wo_mem = np.mean(entropy_wo_mem)
                entropy_w_mem = np.mean(entropy_w_mem)

                if entropy_wo_mem is not None and entropy_w_mem is not None:
                    entropy_gain = float(entropy_wo_mem - entropy_w_mem)
                    entropy_gains.append(entropy_gain)

                    entropy_bonus = self._compute_entropy_bonus(
                        entropy_gain=entropy_gain,
                        entropy_wo_mem=entropy_wo_mem,
                        entropy_w_mem=entropy_w_mem,
                    )
                    entropy_bonus_list.append(entropy_bonus)
                else:
                    entropy_gains.append(0.0)
                    entropy_bonus_list.append(0.0)
                    if entropy_wo_mem is None or entropy_w_mem is None:
                        logger.debug(f"No entropy_wo_mem or entropy_w_mem found for request_id {request_id}")

            entropy_bonus_tensor = torch.tensor(
                entropy_bonus_list,
                dtype=memory_batch.batch["scores"].dtype,
                device=memory_batch.batch["scores"].device,
            )

            scores = memory_batch.batch["scores"]
            attention_mask = memory_batch.batch["attention_mask"]

            for i in range(batch_size):
                valid_positions = (attention_mask[i] == 1).nonzero(as_tuple=True)[0]
                if len(valid_positions) > 0:
                    last_pos = valid_positions[-1]
                    scores[i, last_pos] += entropy_bonus_tensor[i]

            # Update scores in batch
            memory_batch.batch["scores"] = scores

            avg_entropy_gain = np.mean(entropy_gains) if entropy_gains else 0.0
            avg_entropy_bonus = np.mean(entropy_bonus_list) if entropy_bonus_list else 0.0
            max_entropy_gain = np.max(entropy_gains) if entropy_gains else 0.0
            min_entropy_gain = np.min(entropy_gains) if entropy_gains else 0.0
            positive_gain_rate = np.sum(np.array(entropy_gains) > 0) / len(entropy_gains) if entropy_gains else 0.0

            logger.info(
                f"Entropy reward stats - Avg gain: {avg_entropy_gain:.4f}, "
                f"Avg bonus: {avg_entropy_bonus:.4f}, "
                f"Min gain: {min_entropy_gain:.4f}, Max gain: {max_entropy_gain:.4f}, "
                f"Positive rate: {positive_gain_rate:.2%}"
            )

            if not hasattr(memory_batch, "meta_info") or memory_batch.meta_info is None:
                memory_batch.meta_info = {}
            if "metrics" not in memory_batch.meta_info:
                memory_batch.meta_info["metrics"] = {}

            memory_batch.meta_info["metrics"].update(
                {
                    "memory/avg_entropy_gain": avg_entropy_gain,
                    "memory/avg_entropy_bonus": avg_entropy_bonus,
                    "memory/max_entropy_gain": max_entropy_gain,
                    "memory/min_entropy_gain": min_entropy_gain,
                    "memory/positive_gain_rate": positive_gain_rate,
                }
            )

        except Exception as e:
            logger.error(f"Error computing entropy memory scores: {e}")
            logger.error(traceback.format_exc())

        return memory_batch

    def _collect_triggered_interactions_to_buffer(self, batch: DataProto) -> int:
        if self._memory_training_buffer is None:
            return 0

        triggered_interactions = batch.non_tensor_batch.pop("triggered_interactions", None)
        if triggered_interactions is None:
            return 0

        logger.info(f"Size of triggered_interactions: {len(triggered_interactions)}")
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
            per_req_interaction[req_id] = interaction

        logger.info(f"Size of per_req_interaction: {len(per_req_interaction)}")
        for req_id, interaction in per_req_interaction.items():
            self._memory_training_buffer.add(
                request_id=req_id,
                interaction=interaction,
                score_sum=per_req_score_sum.get(req_id, 0.0),
                score_count=per_req_score_count.get(req_id, 0),
            )

        return len(per_req_interaction)

    def _build_memory_batch_from_buffer_entries(
        self, entries: List[MemoryTrainingBufferEntry], global_step: int
    ) -> DataProto:
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
        tensor_model_parallel_size = (
            self.pipeline_config.memory_config.memory_actor_train.strategy_args.strategy_config.get(
                "tensor_model_parallel_size", 1
            )
        )
        cp_size = int(context_parallel_size)
        tp_size = int(tensor_model_parallel_size)
        effective_cp_size = 2 * cp_size if cp_size > 1 else 1
        seq_align = _lcm(effective_cp_size, max(tp_size * cp_size, 1))
        pad_length = ((max_length_in_batch + seq_align - 1) // seq_align) * seq_align

        if (
            self.pipeline_config.memory_sequence_length is not None
            and pad_length > self.pipeline_config.memory_sequence_length
        ):
            max_len = int(self.pipeline_config.memory_sequence_length)
            pad_length = (max_len // seq_align) * seq_align
            pad_length = max(pad_length, seq_align)

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
        metrics = {}

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
                train_batch.batch["advantages"] = train_batch.batch["advantages"] * w.unsqueeze(1)

        if self.pipeline_config.reference_for_memory:
            metrics.update(memory_kl_metrics)
        metrics["time/adv_memory"] = timer.last

        train_batch.meta_info["_broadcast_non_tensor_batch"] = True
        actor_train_metrics_refs = self.memory_actor_train.train_step(train_batch, blocking=False)
        actor_train_metrics: DataProto = DataProto.materialize_concat(data_refs=actor_train_metrics_refs)
        metrics.update(reduce_metrics(actor_train_metrics.meta_info.pop("metrics", {})))

        return metrics

    def _update_memory_model(self, global_step: int) -> Dict[str, Any]:
        """
        Synchronize memory actor weights (train cluster) to memory model (infer cluster).


        """
        metrics: Dict[str, Any] = {}

        for group in self.model_update_groups:
            if group.src_cluster != self.memory_actor_train or group.tgt_cluster != self.memory_model_infer:
                continue

            # Keep group.frequency high to avoid automatic updates each step, but force an update
            # here whenever we explicitly train the memory actor.
            original_frequency = group.frequency
            try:
                group.frequency = 1
                group_metrics = group.model_update(step=global_step)
            finally:
                group.frequency = original_frequency

            metrics.update(group_metrics)
            break

        return metrics

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

    @torch.no_grad()
    def run(self):
        tps_timer = _Timer(window_size=5)

        if self.global_memory_manager is not None and self.global_memory_manager.could_begin_interaction():
            self._warmup_complete = True
            logger.info("Warmup already complete (resumed from checkpoint or pre-warmed memory manager)")

        for global_step in range(self.pipeline_config.max_steps):
            if global_step <= self.state.step:
                global_step += 1
                continue

            if not self._warmup_complete and global_step > self._warmup_interval:
                self._warmup_complete = True
                if not self._warmup_interval <= 0:
                    assert (
                        self.global_memory_manager.could_begin_interaction()
                    ), "Warmup complete but memory manager could not begin interaction"
                else:
                    self.global_memory_manager.notify_memory_warmup()
                logger.info(f"Warmup complete at step {global_step}, buffer collection starting...")

            logger.info(f"pipeline rollout global step {global_step} start...")
            metrics = {}
            with tps_timer:
                # Overlap model offloading (IO bound) with memory flushing (waiting for background processing)
                offload_refs = []
                if self.pipeline_config.adv_estimator == "gae":
                    offload_refs.extend(self.critic.offload_states(blocking=False))
                offload_refs.extend(self.actor_train.offload_states(blocking=False))

                will_train_memory = (
                    self._memory_training_buffer is not None
                    and self._warmup_complete
                    and self._memory_training_buffer.ready(self.pipeline_config.memory_training_batch_size)
                )

                if not will_train_memory:
                    offload_refs.extend(self.memory_actor_train.offload_states(blocking=False))

                # Sync Model Weight & Do Eval
                # CRITICAL: Flush pending memory updates before suspending to avoid data loss
                # We explicitly pass timeout=None to wait indefinitely until all memory updates are processed
                # This ensures strict consistency: memory state is fully updated before the next rollout search begins
                flush_ref = None
                if self.global_memory_manager is not None:
                    # if self.global_memory_manager is not None and self._warmup_complete:
                    flush_ref = self.global_memory_manager.flush_pending_updates_async(timeout=None)

                if flush_ref is not None:
                    flushed_count = ray.get(flush_ref)
                    logger.info(f"Flushed {flushed_count} pending memory updates before suspend")
                    if self.global_memory_manager.memory_config.dedup_similarity_threshold is not None:
                        dedup_statistic = self.global_memory_manager.dedup_memory(
                            dry_run=False,
                        )
                        logger.info(f"Dedup memory statistics: {dedup_statistic}")

                    if (
                        self.pipeline_config.memory_config.memory_model is not None
                        and self.pipeline_config.memory_config.memory_model.enable_merging
                        and global_step % self.pipeline_config.memory_config.memory_model.merging_interval == 0
                    ):
                        logger.info(f"Merging memory at global step {global_step}")
                        self.global_memory_manager.merge_memory()

                if offload_refs:
                    ray.get(offload_refs)

                ray.get(self.train_rollout_scheduler.suspend.remote())
                if self.pipeline_config.async_generation_ratio > 0:
                    self.actor_infer.stop_server()

                model_update_metrics: Dict = self.model_update(global_step)
                metrics.update(model_update_metrics)
                if self.pipeline_config.async_generation_ratio > 0:
                    self.actor_infer.start_server(
                        data=DataProto(
                            meta_info={
                                "global_step": global_step,
                                "is_offload_states": False,
                            }
                        )
                    )
                else:
                    self.actor_infer.start_server(
                        data=DataProto(
                            meta_info={
                                "global_step": global_step,
                                "is_offload_states": True,
                            }
                        )
                    )

                batch: DataProto = DataProto()
                batch.meta_info = {"global_step": global_step}

                # Resume memory manager after validation (searches continue during suspend)
                if self.global_memory_manager is not None:
                    self.global_memory_manager.resume(global_step)

                if global_step % self.pipeline_config.eval_steps == 0:
                    metrics.update(self.val(global_step=global_step))

                with Timer(name="rollout", logger=None) as rollout_timer:
                    batch.meta_info["is_offload_states"] = True
                    batch = ray.get(
                        self.train_rollout_scheduler.get_batch.remote(batch, self.pipeline_config.rollout_batch_size)
                    )
                    dump_rollout_trajectories(
                        self.pipeline_config.rollout_dump_dir,
                        global_step,
                        batch,
                    )
                metrics["time/rollout"] = rollout_timer.last
                metrics.update(reduce_metrics(batch.meta_info.pop("metrics", {})))
                if not (self.pipeline_config.async_generation_ratio > 0):
                    self.actor_infer.stop_server()

                # 1. Compute the Avg Score for Non-Memory Trajectories (per group), save it as a dict
                group_avg_scores = self.compute_non_memory_group_avg_scores(batch)
                batch_stats = self._record_batch_statistics(batch)

                # 1.5. Extract successful memory trajectories (score > non-memory avg) for training
                if getattr(self.pipeline_config, "extract_successful_memory", False):
                    successful_memory_batch = self.extract_successful_memory_trajectories(batch, group_avg_scores)
                    if successful_memory_batch is not None:
                        # Strip memory tokens if configured
                        if getattr(self.pipeline_config, "strip_memory_for_extraction", True):
                            successful_memory_batch = self.strip_memory_from_batch(successful_memory_batch)

                        metrics["training/successful_memory_extracted"] = len(successful_memory_batch)

                        # Add to batch based on configuration
                        use_mode = getattr(self.pipeline_config, "use_extracted_for_training", "concat")
                        if use_mode == "concat":
                            batch = DataProto.concat([batch, successful_memory_batch])
                            logger.info(f"Concatenated successful memory batch. New batch size: {len(batch)}")
                        elif use_mode == "separate":
                            # Adjust batch size to be divisible by DP size for even distribution
                            dp_size = self.actor_train.dp_size
                            adjusted_batch = self._adjust_batch_for_dp(successful_memory_batch, dp_size)

                            # Store for separate processing - could be used in a different training step
                            self._successful_memory_batch = adjusted_batch
                            logger.info(
                                f"Stored successful memory batch for separate training: "
                                f"original={len(successful_memory_batch)}, adjusted={len(adjusted_batch)}"
                            )
                        # "none" mode: just log metrics but don't use for training
                else:
                    metrics["training/successful_memory_extracted"] = 0

                # 2. Buffer collection (only after warmup)
                if self._warmup_complete and self._memory_training_buffer is not None:
                    updated = self._collect_triggered_interactions_to_buffer(batch)
                    metrics["memory/buffer_size"] = len(self._memory_training_buffer)
                    metrics["memory/buffer_updated_keys"] = updated
                    logger.info(f"Buffer size: {metrics['memory/buffer_size']}, added: {updated}")
                else:
                    metrics["memory/buffer_size"] = 0
                    metrics["memory/buffer_updated_keys"] = 0

                # Usually, just for GiGPO, we do not use this but keep for backward compatibility
                batch = compute_discounted_returns(
                    batch,
                    self.pipeline_config.adv_estimator,
                    self.pipeline_config.step_reward_gamma,
                )
                batch = self.adjust_batch(batch, mode=self.pipeline_config.batch_adjust_mode)

                # ---------------- Reference Log Probability ----------------
                metrics.update(reduce_metrics(batch.meta_info.pop("metrics", {})))
                with Timer(name="cal_ref_log_probs", logger=None) as cal_timer:
                    ref_log_probs_refs: List[ray.ObjectRef] = self.reference.compute_log_probs(batch, blocking=False)
                    ref_log_probs = DataProto.materialize_concat(data_refs=ref_log_probs_refs)
                    ref_log_probs.rename(old_keys="log_probs", new_keys="ref_log_probs")
                    batch = batch.union(ref_log_probs)
                    avg_ref_log_prob = masked_mean(
                        batch.batch["ref_log_probs"],
                        batch.batch["response_mask"][:, 1:],
                    )
                    metrics.update(reduce_metrics(ref_log_probs.meta_info.pop("metrics", {})))
                    metrics.update({"critic/ref_log_prob/mean": avg_ref_log_prob.item()})
                metrics["time/ref_log_probs_values_reward"] = cal_timer.last

                # ---------------- Old Log Probability ----------------
                with Timer(name="cal_old_log_probs_values", logger=None) as cal_old_logpb_timer:
                    # TODO: use engine log_probs as old_log_probs
                    batch.meta_info["is_offload_states"] = False
                    old_log_probs_refs: List[ray.ObjectRef] = self.actor_train.compute_log_probs(batch, blocking=False)
                    if self.pipeline_config.adv_estimator == "gae":
                        values_refs: List[ray.ObjectRef] = self.critic.compute_values(batch, blocking=False)
                    base_old_log_probs = DataProto.materialize_concat(data_refs=old_log_probs_refs)
                    if self.pipeline_config.adv_estimator == "gae":
                        values = DataProto.materialize_concat(data_refs=values_refs)
                        batch = batch.union(values)
                        metrics.update(reduce_metrics(values.meta_info.pop("metrics", {})))
                    batch.batch["old_log_probs"] = base_old_log_probs.batch["log_probs"]
                    if "question_entropy" in base_old_log_probs.batch:
                        batch.batch["old_question_entropy"] = base_old_log_probs.batch["question_entropy"]
                    avg_old_log_prob = masked_mean(
                        batch.batch["old_log_probs"],
                        batch.batch["response_mask"][:, 1:],
                    )
                    metrics.update({"critic/old_log_prob/mean": avg_old_log_prob.item()})

                    agg_entropy = agg_loss(
                        loss_mat=base_old_log_probs.batch["entropy"],
                        loss_mask=batch.batch["response_mask"][:, 1:],
                        loss_agg_mode="token-mean",
                    )
                    metrics.update({"critic/entropy/mean": agg_entropy.item()})

                    metrics.update(reduce_metrics(base_old_log_probs.meta_info.pop("metrics", {})))
                metrics["time/old_log_probs_values"] = cal_old_logpb_timer.last

                with Timer(name="adv", logger=None) as timer:
                    batch = compute_response_level_rewards(batch=batch, pipeline_config=self.pipeline_config)
                    metrics.update(reduce_metrics(batch.meta_info.pop("metrics", {})))

                    if self.pipeline_config.reward_clip:
                        reward_clip_frac = compute_clip_fraction(
                            values=batch.batch["response_level_rewards"],
                            clip_max=self.pipeline_config.reward_clip,
                            clip_min=-self.pipeline_config.reward_clip,
                        )
                        batch.batch["response_level_rewards"] = torch.clamp(
                            batch.batch["response_level_rewards"],
                            min=-self.pipeline_config.reward_clip,
                            max=self.pipeline_config.reward_clip,
                        )
                        metrics["critic/reward_clip_frac"] = reward_clip_frac

                    batch, kl_metrics = apply_kl_penalty(
                        data=batch,
                        kl_ctrl=self.kl_ctrl,
                        kl_penalty=self.pipeline_config.kl_penalty,
                    )
                    batch = compute_advantage(
                        data=batch,
                        gamma=self.pipeline_config.gamma,
                        lambd=self.pipeline_config.lambd,
                        adv_estimator=self.pipeline_config.adv_estimator,
                        advantage_clip=self.pipeline_config.advantage_clip,
                        whiten_advantages=self.pipeline_config.whiten_advantages,
                        whiten_rewards=self.pipeline_config.whiten_rewards,
                        whiten_advantages_shift_mean=self.pipeline_config.whiten_advantages_shift_mean,
                    )
                    metrics.update(reduce_metrics(batch.meta_info.pop("metrics", {})))

                metrics.update(kl_metrics)
                metrics["time/adv"] = timer.last

                grouped_cfg = self.pipeline_config.grouped_actor_pg
                if grouped_cfg.enable:
                    group_key = grouped_cfg.group_key
                    strict = grouped_cfg.strict
                    raw_group = batch.non_tensor_batch.get(group_key, None)
                    if raw_group is None:
                        if strict:
                            raise KeyError(f"grouped_actor_pg.group_key={group_key!r} not found in non_tensor_batch")
                        metrics["system/grouped_actor_pg/missing_group_key"] = 1.0
                        w_mask_np = np.zeros(len(batch), dtype=np.bool_)
                    else:
                        w_mask_np = to_bool_numpy(raw_group, strict=strict)
                        if len(w_mask_np) != len(batch):
                            raise ValueError(
                                f"non_tensor_batch[{group_key!r}] length {len(w_mask_np)} != batch size {len(batch)}"
                            )

                    w_count = int(w_mask_np.sum())
                    wo_count = int(len(w_mask_np) - w_count)
                    metrics["system/grouped_actor_pg/w_memory/samples"] = w_count
                    metrics["system/grouped_actor_pg/wo_memory/samples"] = wo_count

                    if "response_mask" in batch.batch.keys():
                        response_tokens = batch.batch["response_mask"][:, 1:].long().sum(dim=-1)
                        w_mask_t = to_bool_tensor(w_mask_np, device=response_tokens.device)
                        metrics["system/grouped_actor_pg/w_memory/response_tokens"] = int(
                            response_tokens[w_mask_t].sum().item()
                        )
                        metrics["system/grouped_actor_pg/wo_memory/response_tokens"] = int(
                            response_tokens[~w_mask_t].sum().item()
                        )

                if self.pipeline_config.adv_estimator == "gae":
                    critic_train_metrics_refs: List[ray.ObjectRef] = self.critic.train_step(batch, blocking=False)

                if self.pipeline_config.critic_warmup <= global_step and self._warmup_complete:
                    # if self.pipeline_config.critic_warmup <= global_step:
                    batch.meta_info["_broadcast_non_tensor_batch"] = True
                    actor_train_metrics_refs = self.actor_train.train_step(batch, blocking=False)
                    actor_train_metrics: DataProto = DataProto.materialize_concat(data_refs=actor_train_metrics_refs)
                    metrics.update(reduce_metrics(actor_train_metrics.meta_info.pop("metrics", {})))

                # SFT training on stripped successful memory trajectories (separate mode)
                if (
                    getattr(self.pipeline_config, "use_extracted_for_training", "none") == "separate"
                    and hasattr(self, "_successful_memory_batch")
                    and self._successful_memory_batch is not None
                    and self._warmup_complete
                ):
                    logger.info(
                        f"Performing SFT training on {len(self._successful_memory_batch)} "
                        f"stripped trajectories ({self.pipeline_config.sft_epochs} epochs)"
                    )

                    # Prepare SFT batch (create labels)
                    sft_batch = self.prepare_sft_batch(self._successful_memory_batch)
                    sft_batch = self.adjust_batch(sft_batch, mode=self.pipeline_config.batch_adjust_mode)

                    # Perform SFT for configured epochs
                    for epoch in range(self.pipeline_config.sft_epochs):
                        sft_batch.meta_info["is_offload_states"] = False
                        sft_batch.meta_info["global_step"] = global_step

                        # Call dedicated SFT train step
                        sft_metrics_refs = self.actor_train.train_sft_step(sft_batch, blocking=False)
                        sft_metrics = DataProto.materialize_concat(data_refs=sft_metrics_refs)

                        # Prefix metrics with sft/epoch{epoch}/
                        epoch_metrics = {
                            f"sft/epoch{epoch}/{k}": v for k, v in sft_metrics.meta_info.pop("metrics", {}).items()
                        }
                        metrics.update(epoch_metrics)

                    logger.info(f"SFT training completed for {len(self._successful_memory_batch)} trajectories")

                    # Clear the batch after training
                    self._successful_memory_batch = None

                # Buffer-based memory training (only when buffer is ready)
                memory_train_metrics = {}
                if self._memory_training_buffer is not None and self._warmup_complete:
                    batch_size = int(getattr(self.pipeline_config, "memory_training_batch_size", 0))
                    if batch_size > 0 and self._memory_training_buffer.ready(batch_size):
                        logger.info(
                            f"Buffer ready with {len(self._memory_training_buffer)} entries, training memory model..."
                        )
                        buffer_entries = self._memory_training_buffer.pop_batch(batch_size)
                        train_batch = self._build_memory_batch_from_buffer_entries(
                            buffer_entries, global_step=global_step
                        )

                        if train_batch is not None:
                            train_batch = self.adjust_memory_batch(
                                train_batch, mode=self.pipeline_config.batch_adjust_mode
                            )

                            # When memory infer and memory actor train share GPUs, we must ensure the memory
                            # local-model server is stopped (and offloaded) before loading the training model.
                            # This mirrors the actor_train/actor_infer colocation pattern used in other pipelines.
                            if self.global_memory_manager is not None:
                                self.global_memory_manager.suspend(global_step)

                            self.memory_actor_train.load_states(blocking=True)

                            memory_train_metrics = self._train_memory_model(train_batch, global_step)
                            metrics.update(memory_train_metrics)

                            memory_update_metrics = self._update_memory_model(global_step)
                            metrics.update(memory_update_metrics)

                            memory_data_metrics = compute_data_metrics_for_memory(batch=train_batch)
                            metrics.update(memory_data_metrics)

                            self.memory_actor_train.offload_states(blocking=True)

                            if self.global_memory_manager is not None:
                                self.global_memory_manager.resume(global_step)

                    else:
                        metrics["memory/buffer_ready"] = False
                        metrics["memory/buffer_size"] = len(self._memory_training_buffer)
                else:
                    metrics["memory/buffer_ready"] = False
                    metrics["memory/buffer_size"] = 0

                if self.pipeline_config.adv_estimator == "gae":
                    critic_train_metrics = DataProto.materialize_concat(data_refs=critic_train_metrics_refs)
                    metrics.update(reduce_metrics(critic_train_metrics.meta_info.pop("metrics", {})))
                tps_timer.push_units_processed(n=torch.sum(batch.batch["attention_mask"]).detach().item())

            metrics["phase/warmup_complete"] = self._warmup_complete
            metrics["memory/buffer_size"] = len(self._memory_training_buffer) if self._memory_training_buffer else 0

            if self._memory_training_buffer is not None and self._warmup_complete:
                batch_size = int(getattr(self.pipeline_config, "memory_training_batch_size", 0))
                metrics["memory/buffer_ready"] = self._memory_training_buffer.ready(batch_size)
            else:
                metrics["memory/buffer_ready"] = False

            if not self._warmup_complete:
                logger.info(f"Warmup step {global_step} / {self._warmup_interval}")
                if self.global_memory_manager is not None:
                    self.global_memory_manager.notify_memory_warmup()

            data_metrics = compute_data_metrics(batch=batch)
            metrics.update(data_metrics)
            metrics["system/tps"] = tps_timer.mean_throughput
            metrics["system/samples"] = (global_step + 1) * self.pipeline_config.rollout_batch_size

            self.state.step = global_step
            self.state.log_history.append(metrics)
            self.do_checkpoint(global_step=global_step)
            self.tracker.log(values=metrics, step=global_step)
            if self.global_memory_manager is not None:
                self.global_memory_manager.notify_update()

            if (global_step + 1) % self.pipeline_config.memory_config.memory_save_interval == 0:
                save_paths = os.path.join(
                    self.pipeline_config.memory_config.memory_save_path,
                    f"memory_{global_step + 1}.json",
                )
                try:
                    success = self.global_memory_manager.save_state(save_paths)
                    if success:
                        logger.info(f"Memory saved to {save_paths}")
                    else:
                        logger.warning(f"Error saving memory")
                except Exception as e:
                    logger.warning(f"Error saving memory: {e}")

            if global_step % self.pipeline_config.logging_steps == 0:
                if int(os.environ.get("RAY_PROFILING", "0")):
                    timeline_dir = os.path.join(self.pipeline_config.profiler_output_dir, "timeline")
                    os.makedirs(timeline_dir, exist_ok=True)
                    ray.timeline(
                        filename=os.path.join(
                            timeline_dir,
                            f"timeline-step-{global_step}.json",
                        ),
                    )

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
                # Convert metrics to JSON serializable format before logging
                serializable_metrics = _convert_to_json_serializable(metrics)
                logger.info(json.dumps(serializable_metrics, ensure_ascii=False))

            logger.info(f"pipeline step {global_step} finished")
            global_step += 1
            logger.info(f"epoch {global_step} finished")

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

        logger.info("pipeline complete!")


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

    # Add entropy-based metrics if available from compute_entropy_memory_scores
    if hasattr(batch, "meta_info") and batch.meta_info is not None and "metrics" in batch.meta_info:
        entropy_metrics = batch.meta_info.get("metrics", {})

        # Add entropy reward metrics computed in compute_entropy_memory_scores
        if "memory/avg_entropy_gain" in entropy_metrics:
            metrics.update(
                {
                    "entropy/avg_entropy_gain": entropy_metrics.get("memory/avg_entropy_gain", 0.0),
                    "entropy/avg_entropy_bonus": entropy_metrics.get("memory/avg_entropy_bonus", 0.0),
                    "entropy/max_entropy_gain": entropy_metrics.get("memory/max_entropy_gain", 0.0),
                    "entropy/min_entropy_gain": entropy_metrics.get("memory/min_entropy_gain", 0.0),
                    "entropy/positive_gain_rate": entropy_metrics.get("memory/positive_gain_rate", 0.0),
                }
            )

    return metrics
