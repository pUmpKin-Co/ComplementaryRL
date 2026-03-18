import os
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Literal, Optional

from omegaconf import DictConfig

from roll.pipeline.agentic.agentic_config import AgenticConfig, EnvManagerConfig, RewardNormalizationConfig
from roll.pipeline.agentic.memory.memory_config import MemoryModelWorkerConfigBase
from roll.utils.logging import get_logger

logger = get_logger()


@dataclass
class MemoryActorConfig(AgenticConfig):
    memory_pg_variant: Optional[str] = field(default=None, metadata={"help": "The variant of the pg."})
    memory_topr_positive_weight: Optional[float] = field(
        default=1.0, metadata={"help": "The weight of the positive samples in the TOPR loss."}
    )
    memory_topr_negative_weight: Optional[float] = field(
        default=1.0, metadata={"help": "The weight of the negative samples in the TOPR loss."}
    )

    # Off-policy PG variants for memory actor training.
    memory_tis_lower_bound: float = field(
        default=0.0,
        metadata={"help": "Lower bound for truncated importance sampling (TIS) ratio clipping (memory actor)."},
    )
    memory_tis_upper_bound: float = field(
        default=1.0,
        metadata={"help": "Upper bound for truncated importance sampling (TIS) ratio clipping (memory actor)."},
    )
    memory_cispo_epsilon_low: float = field(
        default=0.1,
        metadata={"help": "CISPO epsilon_low for IS-ratio clipping (memory actor): clip_lower = 1 - epsilon_low."},
    )
    memory_cispo_epsilon_high: float = field(
        default=0.1,
        metadata={"help": "CISPO epsilon_high for IS-ratio clipping (memory actor): clip_upper = 1 + epsilon_high."},
    )
    memory_cispo_use_unified_mask: bool = field(
        default=False,
        metadata={"help": "Use CISPO unified token mask (memory actor)."},
    )
    memory_sequence_length: int = field(
        default=32768,
        metadata={"help": "The sequence length for the memory actor."},
    )

    memory_training_buffer_enable: bool = field(
        default=False,
        metadata={
            "help": "Enable request_id-keyed memory training buffer (train memory actor only when buffer is ready)."
        },
    )
    memory_training_buffer_size: int = field(
        default=4096,
        metadata={"help": "Capacity of memory training buffer (unique request_ids)."},
    )
    memory_training_batch_size: int = field(
        default=64,
        metadata={"help": "Train memory actor when buffer has this many unique request_ids; pop exactly this many."},
    )
    reference_for_memory: bool = field(
        default=False,
        metadata={"help": "Whether to use a separate reference for the memory actor."},
    )
    memory_reference: MemoryModelWorkerConfigBase = field(
        default_factory=MemoryModelWorkerConfigBase,
        metadata={"help": "The reference model for the memory."},
    )
    memory_ppo_epochs: int = field(
        default=1,
        metadata={"help": "The number of PPO epochs for the memory actor."},
    )
    memory_kl_coef: float = field(
        default=0.2,
        metadata={"help": "The KL coefficient for the memory actor."},
    )
    memory_target_kl: float = field(
        default=None,
        metadata={"help": "The target KL for the memory actor."},
    )
    memory_kl_horizon: int = field(
        default=10000,
        metadata={"help": "The horizon for the memory actor."},
    )
    memory_advantage_clip: Optional[float] = field(
        default=None,
        metadata={"help": "The advantage clip for the memory actor."},
    )
    memory_pg_clip: float = field(default=0.2, metadata={"help": "The PGLIP for the memory actor."})
    memory_pg_clip_low: float = field(
        default=0.2,
        metadata={"help": "The PGLIP low for the memory actor."},
    )
    memory_pg_clip_high: float = field(
        default=0.2,
        metadata={"help": "The PGLIP high for the memory actor."},
    )
    memory_use_pg_clip_range: bool = field(
        default=False,
        metadata={"help": "Whether to use PGLIP range for the memory actor."},
    )
    memory_loss_agg_mode: Literal[
        "token-mean",
        "seq-mean-token-sum",
        "seq-mean-token-mean",
        "seq-mean-token-sum-norm",
    ] = field(
        default="seq-mean-token-mean",
        metadata={"help": "The loss aggregation mode for the memory actor."},
    )
    memory_dual_clip_loss: bool = field(
        default=False,
        metadata={"help": "Whether to use dual clip loss for the memory actor."},
    )
    memory_adv_estimator: Literal["reinforce", "grpo"] = field(
        default="reinforce",
        metadata={"help": "The advantage estimator for the memory actor."},
    )
    memory_reward_normalization: RewardNormalizationConfig = field(
        default_factory=RewardNormalizationConfig,
        metadata={"help": "The reward normalization for the memory actor."},
    )
    memory_whiten_advantages: bool = field(
        default=False,
        metadata={"help": "Whether to whiten the advantages for the memory actor."},
    )
    memory_whiten_rewards: bool = field(
        default=False,
        metadata={"help": "Whether to whiten the rewards for the memory actor."},
    )
    memory_entropy_loss_coef: float = field(
        default=0.0,
        metadata={"help": "The entropy loss coefficient for the memory actor."},
    )
    memory_use_kl_loss: bool = field(
        default=False,
        metadata={"help": "Whether to use KL loss for the memory actor."},
    )
    memory_kl_loss_coef: float = field(
        default=0.0,
        metadata={"help": "The KL loss coefficient for the memory actor."},
    )
    memory_whiten_advantages_shift_mean: bool = field(
        default=False,
        metadata={"help": "Whether to shift the mean of the advantages for the memory actor."},
    )

    no_memory_traj_per_env_group: int = field(
        default=1,
        metadata={"help": "The number of trajectories per environment group without memory."},
    )
    memory_score_penalty: float = field(
        default=0.0,
        metadata={"help": "Penalty to subtract from memory trajectory scores."},
    )

    # -------------------------
    # Repeat-control for memory training (dedup / weighting / cooldown)
    # -------------------------
    memory_repeat_control_enable: bool = field(
        default=True,
        metadata={"help": "Enable repeat-control when training memory actor (dedup + weighting + cooldown)."},
    )
    memory_repeat_cooldown_steps: int = field(
        default=0,
        metadata={"help": "Cooldown steps: if a memory UID was trained recently, downweight/skip its samples."},
    )
    memory_repeat_weight_power: float = field(
        default=0.5,
        metadata={"help": "Weight = (1 + max_train_count(uid))^(-power). 0 disables weighting."},
    )

    # Entropy-based memory co-training configuration
    use_entropy_memory: bool = field(
        default=False,
        metadata={"help": "Whether to use entropy-based memory co-training."},
    )
    entropy_reduction_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for entropy reduction in memory reward calculation."},
    )
    entropy_computation_frequency: int = field(
        default=1,
        metadata={"help": "Frequency of entropy computation (every N steps)."},
    )
    entropy_min_samples: int = field(
        default=1,
        metadata={"help": "Minimum number of samples required for entropy computation."},
    )
    entropy_bonus_strategy: Literal["relative", "tanh", "sigmoid", "asymmetric_clip", "log_space"] = field(
        default="relative",
        metadata={
            "help": "Strategy for computing entropy bonus: 'relative' (% reduction, stable), 'tanh' (smooth), 'sigmoid' (bounded [0,1]), 'asymmetric_clip' (reward>punish), 'log_space' (large values)."
        },
    )
    entropy_bonus_clip_min: float = field(
        default=-1.0,
        metadata={"help": "Minimum value for entropy bonus clipping (for relative strategy)."},
    )
    entropy_bonus_clip_max: float = field(
        default=1.0,
        metadata={"help": "Maximum value for entropy bonus clipping (for relative strategy)."},
    )
    entropy_sigmoid_temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature for sigmoid strategy (higher = less steep)."},
    )
    entropy_asymmetric_negative_clip: float = field(
        default=-0.2,
        metadata={"help": "Negative clip value for asymmetric strategy (punish less)."},
    )
    question_on_all_turns: bool = field(
        default=False,
        metadata={"help": "Whether to use question mask on all turns."},
    )

    # Successful memory trajectory extraction configuration
    extract_successful_memory: bool = field(
        default=False,
        metadata={"help": "Extract memory trajectories with scores above non-memory group average for training."},
    )
    strip_memory_for_extraction: bool = field(
        default=True,
        metadata={"help": "Strip memory tokens from extracted successful trajectories before training."},
    )
    use_extracted_for_training: Literal["concat", "separate", "none"] = field(
        default="concat",
        metadata={
            "help": "How to use extracted trajectories: 'concat' (add to batch), 'separate' (store separately), 'none' (metrics only)."
        },
    )

    # SFT training on stripped successful memory trajectories
    sft_epochs: int = field(
        default=1,
        metadata={"help": "Number of epochs to train on stripped data per RL step."},
    )
    sft_learning_rate: Optional[float] = field(
        default=None,
        metadata={"help": "Learning rate for SFT training. Uses actor LR if None."},
    )

    def __post_init__(self):
        AgenticConfig.__post_init__(self)

        if self.memory_config.memory_actor_enable_training is not True:
            self.memory_config.memory_actor_enable_training = True

        if self.memory_config.memory_save_path is None:
            self.memory_config.memory_save_path = os.path.join(self.output_dir, "memory")
            os.makedirs(self.memory_config.memory_save_path, exist_ok=True)
            logger.info(
                f"memory_save_path is not set, so memory_save_path will be set to {self.memory_config.memory_save_path}"
            )
        else:
            logger.info(f"memory_save_path is set to {self.memory_config.memory_save_path}")

        if not self.reference_for_memory:
            self.memory_reference = None
        else:
            if self.memory_reference.worker_cls is None:
                self.memory_reference.worker_cls = "roll.pipeline.base_worker.ActorWorker"
            if self.memory_reference.name is None:
                self.memory_reference.name = f"memory_reference"

        if self.memory_advantage_clip is False:
            self.memory_advantage_clip = None

        if self.memory_config is not None:
            setattr(
                self.memory_config.memory_model,
                "checkpoint_config",
                self.checkpoint_config,
            )
            setattr(
                self.memory_config.memory_actor_train,
                "checkpoint_config",
                self.checkpoint_config,
            )
            if self.memory_config.memory_actor_train.worker_cls is None:
                self.memory_config.memory_actor_train.worker_cls = (
                    "roll.pipeline.agentic.memory.memory_model.memory_model_worker.MemoryActorWorker"
                )

    def make_env_configs(self, env_manager_config: EnvManagerConfig):
        # construct env configs
        env_configs = defaultdict(defaultdict)
        done_groups = 0
        env_manager_config.env_configs = {}
        group_seeds = {}
        max_env_num_per_worker = env_manager_config.max_env_num_per_worker
        apply_eval_memory_split = self._should_split_eval_memory(env_manager_config)
        eval_memory_env_ids = set()
        if apply_eval_memory_split:
            eval_memory_env_ids = self._get_eval_memory_env_ids(env_manager_config)
            logger.info(
                "Eval memory split enabled in multi-agentic config: with_memory=%s, without_memory=%s",
                len(eval_memory_env_ids),
                env_manager_config.num_env_groups * env_manager_config.group_size - len(eval_memory_env_ids),
            )
        # Per group initialize environment, each group have same seed
        for tag, n_group in zip(env_manager_config.tags, env_manager_config.num_groups_partition):
            for env_id in range(
                done_groups * env_manager_config.group_size,
                (done_groups + n_group) * env_manager_config.group_size,
            ):
                cfg_template = self.custom_envs[tag]
                env_class = cfg_template.env_type

                group_id = env_id // env_manager_config.group_size

                if "env_config" not in cfg_template:
                    cfg_template.env_config = {}
                env_config = {**cfg_template.env_config}

                if group_id not in group_seeds:
                    group_seeds[group_id] = random.randint(0, 2**32 - 1)

                # Calculate position within the group for memory evolution logic
                env_position_in_group = env_id % env_manager_config.group_size
                evolve_with_memory = False
                if apply_eval_memory_split:
                    evolve_with_memory = env_id in eval_memory_env_ids
                elif hasattr(self, "no_memory_traj_per_env_group"):
                    # First no_memory_traj_per_env_group envs have evolve_with_memory=False, others True
                    if (
                        env_position_in_group >= self.no_memory_traj_per_env_group
                        or env_manager_config.group_size <= self.no_memory_traj_per_env_group
                    ):
                        evolve_with_memory = True

                entry = {}
                entry.update(cfg_template)
                entry.pop("env_config", None)
                entry.update(
                    {
                        "tag": tag,
                        "group_id": group_id,
                        "env_id": env_id,
                        "config": env_config,
                        "env_class": env_class,
                        "env_manager_cls": cfg_template.get(
                            "env_manager_cls",
                            "roll.pipeline.agentic.env_manager.traj_env_manager.TrajEnvManager",
                        ),
                        "group_seed": group_seeds[group_id],
                        "evolve_with_memory": evolve_with_memory,
                    }
                )
                worker_rank = env_id // max_env_num_per_worker
                env_configs[worker_rank][env_id] = DictConfig(entry)
                logger.info(
                    f"[ENV CONFIG] tag: {tag}, group_id: {group_id}, group_seeds: {group_seeds[group_id]}, env_id: {env_id}"
                )
            done_groups += n_group
        assert done_groups == env_manager_config.num_env_groups
        env_manager_config.env_configs = env_configs

    def set_max_steps(self, max_steps: int):
        memory_actor_backward_batch_size = (
            self.memory_config.memory_actor_train.training_args.per_device_train_batch_size
            * self.memory_config.memory_actor_train.training_args.gradient_accumulation_steps
        )
        actor_backward_batch_size = (
            self.actor_train.training_args.per_device_train_batch_size
            * self.actor_train.training_args.gradient_accumulation_steps
        )
        critic_backward_batch_size = (
            self.critic.training_args.per_device_train_batch_size
            * self.critic.training_args.gradient_accumulation_steps
        )

        self.memory_config.memory_actor_train.training_args.max_steps = max_steps * (
            self.rollout_batch_size
            * self.memory_config.searcher.memory_fetch_num
            * self.memory_config.memory_model.generating_args.num_return_sequences
            * self.ppo_epochs
            // memory_actor_backward_batch_size
        )
        self.actor_train.training_args.max_steps = max_steps * (
            self.rollout_batch_size
            * self.actor_infer.generating_args.num_return_sequences
            * self.ppo_epochs
            // actor_backward_batch_size
        )
        self.critic.training_args.max_steps = max_steps * (
            self.rollout_batch_size
            * self.actor_infer.generating_args.num_return_sequences
            // critic_backward_batch_size
        )

        logger.info(f"pipeline max_steps: {self.max_steps} to {max_steps}")
        logger.info(f"actor train max_steps without dp_size: {self.actor_train.training_args.max_steps}")
        logger.info(f"critic train max_steps without dp_size: {self.critic.training_args.max_steps}")
        self.max_steps = max_steps
