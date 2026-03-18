import json
import os.path
from typing import Any, Dict, List

import numpy as np
import ray
import torch
from codetiming import Timer
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from ray.util.timer import _Timer

from roll.datasets.global_dataset import GlobalDatasetManager
from roll.distributed.executor.cluster import Cluster
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.scheduler.rollout_scheduler import RolloutScheduler
from roll.models.model_providers import default_tokenizer_provider
from roll.pipeline.agentic.agentic_config import AgenticConfig
from roll.pipeline.agentic.multi_agentic_config import MemoryActorConfig
from roll.pipeline.agentic.utils import (
    dump_rollout_render,
    dump_rollout_trajectories,
)
from roll.pipeline.base_pipeline import BasePipeline
from roll.utils.constants import RAY_NAMESPACE
from roll.utils.functionals import reduce_metrics
from roll.utils.logging import get_logger

logger = get_logger()


def get_episode_scores(batch: DataProto) -> torch.Tensor:
    """Get episode scores from batch."""
    batch_group_by_traj = batch.group_by(keys="traj_id")
    scores = []
    for traj_id, traj_batch in batch_group_by_traj.items():
        episode_scores = traj_batch.non_tensor_batch["episode_scores"][0]
        scores.append(episode_scores)
    return torch.tensor(scores, dtype=torch.float32)


class BaselineEvalPipeline(BasePipeline):
    """
    Baseline Evaluation Pipeline - Actor Inference Only.

    This pipeline provides a naive baseline by:
    1. Using only actor_infer model (no memory model, no training)
    2. Running validation to evaluate model performance
    3. Using the same config structure as knowledge_model_pipeline.py
    4. Producing comparable scores for baseline comparison

    This allows you to compare:
    - Naive baseline performance (this pipeline)
    - Memory-enhanced performance (knowledge_model_pipeline.py)
    """

    def __init__(self, pipeline_config: AgenticConfig | MemoryActorConfig):
        super().__init__(pipeline_config)
        self.pipeline_config: AgenticConfig | MemoryActorConfig

        self.pipeline_config.set_max_steps(max_steps=self.pipeline_config.max_steps)

        # Initialize only actor_infer cluster
        self.actor_infer: Any = Cluster(
            name=self.pipeline_config.actor_infer.name,
            worker_cls=self.pipeline_config.actor_infer.worker_cls,
            resource_manager=self.resource_manager,
            worker_config=self.pipeline_config.actor_infer,
        )

        self.download_models(self.actor_infer)
        self.tokenizer = default_tokenizer_provider(model_args=self.pipeline_config.actor_infer.model_args)

        # Initialize rollout schedulers WITHOUT memory_manager
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
                global_memory_manager=None,  # No memory manager for baseline
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
                global_memory_manager=None,  # No memory manager for baseline
                mode="val",
            )
        )
        self.val_dataset_manager = GlobalDatasetManager.options(
            name=f"val_dataset_manager",
            get_if_exists=True,
            namespace=RAY_NAMESPACE,
        ).remote()

        # Initialize actor_infer cluster
        self.actor_infer.initialize(pipeline_config=self.pipeline_config, blocking=True)

        logger.info("BaselineEvalPipeline initialized (actor_infer only, no training)")

    def val(self, global_step: int) -> Dict[str, float]:
        """
        Run validation with actor_infer only.

        This is identical to the validation logic in AgenticPipeline.val()
        to ensure scores are comparable.

        Note: Server lifecycle is managed by the run() method, not here.
        """
        batch = DataProto()
        metrics = {}
        batch.meta_info["is_offload_states"] = False
        batch.meta_info["global_step"] = global_step

        # Reset validation dataset
        ray.get(self.val_dataset_manager.reset.remote())

        # Get validation batch (server should already be started by run())
        eval_batch = ray.get(self.val_rollout_scheduler.get_batch.remote(batch, self.pipeline_config.val_batch_size))

        # Dump trajectories
        dump_rollout_trajectories(self.pipeline_config.rollout_dump_dir, global_step, eval_batch)

        # Dump renders if configured
        if self.pipeline_config.render_save_dir:
            try:
                batch_group_by_traj = eval_batch.group_by(keys="traj_id")
            except Exception as e:
                logger.warning(f"Failed to group eval batch for render dump: {e}")
            else:
                frames: List[List] = []
                env_ids: List = []
                tags: List = []
                episode_scores: List[float] = []
                for traj_id, traj_batch in batch_group_by_traj.items():
                    traj_frames_array = traj_batch.non_tensor_batch.get("frames")
                    if traj_frames_array is None or len(traj_frames_array) == 0:
                        continue
                    traj_frames = traj_frames_array[0]
                    if isinstance(traj_frames, np.ndarray):
                        traj_frames = traj_frames.tolist()
                    if len(traj_frames) == 0:
                        continue
                    frames.append(traj_frames)

                    traj_env_ids = traj_batch.non_tensor_batch.get("env_ids")
                    traj_tags = traj_batch.non_tensor_batch.get("tags")
                    traj_scores = traj_batch.non_tensor_batch.get("episode_scores")

                    env_ids.append(traj_env_ids[0] if traj_env_ids is not None and len(traj_env_ids) > 0 else traj_id)
                    tags.append(traj_tags[0] if traj_tags is not None and len(traj_tags) > 0 else traj_id)
                    if traj_scores is not None and len(traj_scores) > 0:
                        episode_scores.append(float(traj_scores[0]))
                    else:
                        episode_scores.append(0.0)

                if frames:
                    try:
                        dump_rollout_render(
                            self.pipeline_config.render_save_dir,
                            global_step,
                            frames,
                            env_ids,
                            tags,
                            episode_scores,
                        )
                    except Exception as e:
                        logger.warning(f"Failed to dump validation rollout render: {e}")

        # Compute evaluation metrics
        eval_metrics = reduce_metrics(eval_batch.meta_info.get("metrics", {}))

        # Safety check: skip score computation if batch is empty
        if eval_batch.batch is not None and eval_batch.batch.batch_size[0] > 0:
            eval_score = get_episode_scores(eval_batch)
            if len(eval_score) > 0:
                eval_metrics["score/mean"] = torch.mean(eval_score).detach().item()
                eval_metrics["score/max"] = torch.max(eval_score).detach().item()
                eval_metrics["score/min"] = torch.min(eval_score).detach().item()
            else:
                logger.warning("Eval batch has no trajectories, skipping score computation")
        else:
            logger.warning("Eval batch is empty, skipping score computation")

        # Group by tags (environment types) for per-environment metrics
        batch_grouped = eval_batch.group_by(keys="tags")
        for group_name, group_batch in batch_grouped.items():
            traj_group_scores = []
            batch_traj_grouped = group_batch.group_by(keys="traj_group_id")
            for batch_traj_group_name, batch_traj_group in batch_traj_grouped.items():
                traj_group_score = get_episode_scores(batch_traj_group)
                traj_group_scores.append(traj_group_score.mean().item())
            eval_score = torch.tensor(traj_group_scores, dtype=torch.float)
            eval_metrics[f"{group_name}/score/mean"] = torch.mean(eval_score).detach().item()
            eval_metrics[f"{group_name}/score/max"] = torch.max(eval_score).detach().item()
            eval_metrics[f"{group_name}/score/min"] = torch.min(eval_score).detach().item()

        metrics.update({f"val/{k}": v for k, v in eval_metrics.items()})
        logger.info(f"val_batch_size: {len(eval_batch)}")
        logger.info(f"val metrics: {metrics}")

        return metrics

    @torch.no_grad()
    def run(self):
        """
        Run baseline evaluation pipeline.

        This pipeline:
        1. Runs initial validation at step 0
        2. Periodically runs validation based on eval_steps
        3. Collects rollout data (but does NOT train)
        4. Logs metrics for comparison with memory-enhanced version
        """
        tps_timer = _Timer(window_size=5)

        # Start actor_infer server for initial validation
        if self.pipeline_config.async_generation_ratio > 0:
            self.actor_infer.start_server(
                data=DataProto(
                    meta_info={
                        "global_step": 0,
                        "is_offload_states": False,
                    }
                )
            )
        else:
            self.actor_infer.start_server(
                data=DataProto(
                    meta_info={
                        "global_step": 0,
                        "is_offload_states": True,
                    }
                )
            )

        # Phase 1: Initial validation at the beginning
        logger.info("Phase 1: Running initial validation...")
        initial_val_metrics = self.val(global_step=0)
        logger.info(f"Initial validation metrics: {initial_val_metrics}")

        for basic_step in range(self.pipeline_config.max_steps):
            if basic_step <= self.state.step:
                continue

            logger.info(f"Pipeline baseline step {basic_step} start...")
            metrics = {}

            with tps_timer:
                # Server is already started from previous iteration (or initial validation)
                # No need to start it again here

                batch: DataProto = DataProto()
                batch.meta_info = {"global_step": basic_step}

                # Regular validation based on eval_steps
                if basic_step % self.pipeline_config.eval_steps == 0:
                    logger.info(f"Running validation at step {basic_step}...")
                    val_metrics = self.val(global_step=basic_step)
                    metrics.update(val_metrics)

                # Rollout (no training)
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

                # Stop actor_infer server only if not async (async server stays running)
                if not (self.pipeline_config.async_generation_ratio > 0):
                    self.actor_infer.stop_server()

                    # Start server again for next iteration if there is one
                    # Only restart if this is not the last iteration
                    if basic_step + 1 < self.pipeline_config.max_steps:
                        self.actor_infer.start_server(
                            data=DataProto(
                                meta_info={
                                    "global_step": basic_step + 1,
                                    "is_offload_states": True,
                                }
                            )
                        )

                tps_timer.push_units_processed(n=torch.sum(batch.batch["attention_mask"]).detach().item())

            metrics["system/tps"] = tps_timer.mean_throughput
            metrics["system/samples"] = (basic_step + 1) * self.pipeline_config.rollout_batch_size
            metrics["step/basic_step"] = basic_step
            metrics["pipeline_type"] = "baseline"  # Mark as baseline for comparison

            self.state.step = basic_step
            self.state.log_history.append(metrics)
            self.tracker.log(values=metrics, step=basic_step)

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

            logger.info(f"Pipeline baseline step {basic_step} finished")

        # Cleanup
        # Stop actor_infer server if it's running
        if self.pipeline_config.async_generation_ratio > 0:
            self.actor_infer.stop_server()

        ray.get(
            [
                self.train_rollout_scheduler.shutdown.remote(),
                self.val_rollout_scheduler.shutdown.remote(),
            ]
        )

        logger.info("Baseline evaluation pipeline complete!")