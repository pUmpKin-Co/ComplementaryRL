from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Union

from roll.configs import DataArguments, GeneratingArguments, ModelArguments
from roll.configs.training_args import TrainingArguments
from roll.utils.logging import get_logger

logger = get_logger()


@dataclass
class StrategyArguments:
    strategy_name: Literal[
        "deepspeed_train",
        "hf_infer",
        "deepspeed_infer",
        "vllm",
        "sglang",
        "megatron_infer",
        "megatron_train",
    ] = field(
        default="deepspeed_train",
        metadata={
            "help": "The name of the strategy. Options: 'deepspeed_train', 'hf_infer', 'deepspeed_infer', 'vllm', 'sglang', "
            "'megatron_infer', 'megatron_train'."
        },
    )
    strategy_config: Optional[Dict] = field(
        default_factory=dict,
        metadata={"help": "Configuration dictionary for the strategy."},
    )


@dataclass
class WorkerConfig:
    name: str = field(
        default=None,
        metadata={"help": "name of this role."},
    )
    worker_cls: Optional[str] = field(
        default=None, metadata={"help": "The class of the worker."}
    )
    pg_variant: Optional[str] = field(
        default=None,
        metadata={"help": "The variant of the policy gradient."},
    )
    model_args: ModelArguments = field(
        default_factory=ModelArguments,
        metadata={
            "help": "The arguments for the model, encapsulated in a ModelArguments object."
        },
    )
    training_args: TrainingArguments = field(
        default_factory=TrainingArguments,
        metadata={"help": "Training-related arguments."},
    )
    data_args: DataArguments = field(
        default=None,
        metadata={
            "help": "Data-related arguments; optional and can be None."
        },
    )
    generating_args: GeneratingArguments = field(
        default=None,
        metadata={
            "help": "Arguments for generating output; optional and can be None."
        },
    )
    strategy_args: StrategyArguments = field(
        default=None,
        metadata={
            "help": "The strategy configuration, encapsulated in a StrategyArguments object."
        },
    )
    world_size: int = field(
        default=None, metadata={"help": "The number of role clusters."}
    )
    device_mapping: Union[List[int], str] = field(
        default=None,
        metadata={
            "help": "The list of device ids to use when training. "
            "Configure it as a string that can be evaluated as List[int], such as 'list(range(0, 8))'."
            "If device_mapping is None, the worker uses cpu only."
        },
    )
    num_gpus_per_worker: int = field(
        default=1, metadata={"help": "The number of gpu per worker."}
    )
    num_workers_per_gpu: int = field(
        default=1,
        metadata={
            "help": "The number of workers sharing the same GPU. Only applicable when num_gpus_per_worker=1."
        },
    )
    model_update_frequency: int = field(
        default=1, metadata={"help": "Frequency of model updates."}
    )
    infer_batch_size: int = field(
        default=16, metadata={"help": "Batch size for inference."}
    )
    backend_timeout: int = field(
        default=30,
        metadata={"help": "minutes for dist backend communicating."},
    )
    system_envs: dict = field(
        default_factory=dict,
        metadata={"help": "system environment variables for this worker."},
    )
    topr_positive_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for positive samples in TOPR loss."},
    )
    topr_negative_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for negative samples in TOPR loss."},
    )

    # -------------------------
    # Off-policy PG variants
    # -------------------------
    tis_lower_bound: float = field(
        default=0.0,
        metadata={"help": "Lower bound for truncated importance sampling (TIS) ratio clipping."},
    )
    tis_upper_bound: float = field(
        default=1.0,
        metadata={"help": "Upper bound for truncated importance sampling (TIS) ratio clipping."},
    )
    cispo_epsilon_low: float = field(
        default=0.1,
        metadata={"help": "CISPO epsilon_low for IS-ratio clipping: clip_lower = 1 - epsilon_low."},
    )
    cispo_epsilon_high: float = field(
        default=0.1,
        metadata={"help": "CISPO epsilon_high for IS-ratio clipping: clip_upper = 1 + epsilon_high."},
    )
    cispo_use_unified_mask: bool = field(
        default=False,
        metadata={"help": "Use CISPO unified token mask (paper Eq.7) in addition to clipped-ratio stop-gradient."},
    )

    seed: int = field(
        default=42, metadata={"help": "Random seed for initializations."}
    )

    def __post_init__(self):

        if self.strategy_args is not None:
            if (
                self.strategy_args.strategy_name
                not in ["hf_infer", "vllm", "sglang"]
                and self.num_gpus_per_worker > 1
            ):
                logger.info(
                    f"strategy_name={self.strategy_args.strategy_name}, force set num_gpus_per_worker={self.num_gpus_per_worker} to 1."
                )
                self.num_gpus_per_worker = 1
            if self.strategy_args.strategy_name == "vllm":
                strategy_config = self.strategy_args.strategy_config
                tensor_parallel_size = strategy_config.get(
                    "tensor_parallel_size", 1
                )
                pipeline_parallel_size = strategy_config.get(
                    "pipeline_parallel_size", 1
                )
                self.num_gpus_per_worker = (
                    tensor_parallel_size * pipeline_parallel_size
                )
                logger.info(
                    f"set vllm num_gpus_per_worker to {self.num_gpus_per_worker}, "
                    f"tensor_parallel_size: {tensor_parallel_size}, "
                    f"pipeline_parallel_size: {pipeline_parallel_size}"
                )

        if self.device_mapping is not None:
            self.device_mapping = eval(self.device_mapping)

            # Validate num_workers_per_gpu
            if self.num_workers_per_gpu > 1:
                assert self.num_gpus_per_worker == 1, (
                    f"num_workers_per_gpu={self.num_workers_per_gpu} is only supported when num_gpus_per_worker=1. "
                    f"Got num_gpus_per_worker={self.num_gpus_per_worker}."
                )
                # When multiple workers share GPUs, world_size is explicitly set
                # and device_mapping specifies which GPUs to use (can be repeated)
                if self.world_size is None:
                    # If world_size not set, calculate from device_mapping
                    # Each unique GPU can host num_workers_per_gpu workers
                    unique_gpus = len(set(self.device_mapping))
                    self.world_size = unique_gpus * self.num_workers_per_gpu
                    logger.info(
                        f"Calculated world_size={self.world_size} from {unique_gpus} unique GPUs "
                        f"× {self.num_workers_per_gpu} workers per GPU"
                    )
            else:
                # Original logic: each worker gets dedicated GPU(s)
                assert (
                    len(self.device_mapping) % self.num_gpus_per_worker == 0
                ), f"len(device_mapping)={len(self.device_mapping)} must be divisible by num_gpus_per_worker={self.num_gpus_per_worker}."
                self.world_size = (
                    len(self.device_mapping) // self.num_gpus_per_worker
                )
        else:
            self.num_gpus_per_worker = 0

        self.resource_placement_groups: Optional[List[Dict]] = None
        self.checkpoint_config: Optional[Dict] = None

        if hasattr(self, "model_args"):
            if self.model_args.dtype == "bf16":
                self.training_args.bf16 = True
            elif self.model_args.dtype == "fp16":
                self.training_args.fp16 = True
