import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from roll.configs.worker_config import WorkerConfig
from roll.utils.logging import get_logger

logger = get_logger()


class StrEnum(str, Enum):
    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"'{str(self)}'"


class MemoryType(StrEnum):
    # case bese
    case_memory = "case_memory"
    case_embedding_memory = "case_embedding_memory"
    trajectory_memory = "trajectory_memory"


class MemoryStructure(StrEnum):
    tabuler = "tabuler"
    multi_task_tabuler = "multi_task_tabuler"


class SampleStrategy(StrEnum):
    simple_similarity = "simple_similarity"
    embedding_similarity = "embedding_similarity"
    random = "random"
    most_recent = "most_recent"
    faiss_embedding_similarity = "faiss"


class UpdateStrategy(StrEnum):
    lru = "lru"
    random = "random"
    fifo = "fifo"


class MemoryModelType(StrEnum):
    api_model = "api_model"
    local_model = "local_model"


class MemoryIntegrationStrategy(StrEnum):
    """Strategy for when to search and update memory in the environment.

    - TURN_BASED: Search and update every turn (episodic memory)
    - TRAJECTORY_BASED: Search at first turn, update at trajectory end (procedural memory)
    - BOTH: Combine both strategies
    """

    turn_based = "turn_based"
    trajectory_based = "trajectory_based"
    both = "both"


class PoolType(StrEnum):
    """
    Pool type for embedding model.
    - cls: CLS token pooling
    - last_token: Last token pooling
    - mean: Mean pooling
    """

    cls = "cls"
    last_token = "last_token"
    mean = "mean"


@dataclass
class SearcherConfig(WorkerConfig):
    memory_search_strategy: SampleStrategy = field(
        default=SampleStrategy.simple_similarity,
        metadata={"help": "The strategy to use for searching the memory."},
    )
    memory_fetch_num: int = field(
        default=3,
        metadata={"help": "The number of entries to fetch from the memory."},
    )
    memory_searcher_state_path: Optional[str] = field(
        default=None, metadata={"help": "The path to the searcher state."}
    )

    # Optional: diversify retrieval results to reduce repeated top-1 memories.
    diversity_enable: bool = field(
        default=False,
        metadata={"help": "Whether to re-rank retrieved UIDs using usage penalties to increase diversity."},
    )
    diversity_lambda: float = field(
        default=0.0,
        metadata={"help": "Penalty strength for frequently retrieved memories (log1p(retrieve_count))."},
    )
    diversity_recent_seconds: float = field(
        default=30.0,
        metadata={"help": "Recency window (seconds) to penalize recently retrieved memories."},
    )
    diversity_dropout_p: float = field(
        default=0.0,
        metadata={"help": "Probability to demote the top UID if it was retrieved very recently."},
    )

    diversity_candidate_multiplier: int = field(
        default=2,
        metadata={
            "help": "When diversity is enabled, fetch up to (multiplier * memory_fetch_num) candidates, "
            "rerank with penalties, then return the final memory_fetch_num."
        },
    )


@dataclass
class UpdaterConfig(WorkerConfig):
    memory_update_strategy: UpdateStrategy = field(
        default=UpdateStrategy.lru,
        metadata={"help": "The strategy to use for updating the memory."},
    )
    memory_size: Optional[int] = field(default=None, metadata={"help": "The size of the memory."})


@dataclass
class MemoryModelWorkerConfigBase(WorkerConfig):
    is_memory_model: bool = field(default=True, metadata={"help": "Whether to use the memory batch."})

    def __post_init__(self):
        super().__post_init__()
        if not hasattr(self, "is_memory_model") or self.is_memory_model is None:
            self.is_memory_model = True


@dataclass
class MemoryModelConfig(MemoryModelWorkerConfigBase):
    memory_model_type: MemoryModelType = field(
        default=MemoryModelType.api_model,
        metadata={"help": "The type of the memory model."},
    )
    memory_model_system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "The system prompt of the memory model."},
    )
    memory_model_user_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "The user prompt of the memory model."},
    )
    tool_use_system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "The system prompt of the tool use."},
    )
    tool_use_user_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "The user prompt of the tool use."},
    )
    tool_use_parsing_pattern: Optional[str] = field(
        default=r"<answer>(.*?)</answer>",
        metadata={"help": "The regex pattern to parse the tool use from the memory model."},
    )
    memory_model_build_message_pattern: Optional[str] = field(
        default=r"<answer>(.*?)</answer>",
        metadata={"help": "Regex pattern to extract answer from model response."},
    )
    memory_model_response_pattern: Optional[str] = field(
        default=r"<answer>(.*?)</answer>",
        metadata={"help": "Regex pattern to extract answer from model response."},
    )
    memory_model_with_functional_operations: bool = field(
        default=False,
        metadata={"help": "Whether to use functional operations in the memory model."},
    )
    memory_model_max_functional_operations: int = field(
        default=2,
        metadata={"help": "The maximum number of functional operations in the memory model."},
    )
    enable_merging: bool = field(default=False, metadata={"help": "Whether to enable memory merging."})
    merging_interval: Optional[int] = field(
        default=None, metadata={"help": "Interval (in steps) to perform memory merging."}
    )
    merging_size: Optional[Dict[str, int]] = field(
        default_factory=dict,
        metadata={"help": "Merging when the memory size is larger than the merging size."},
    )
    max_merging_item_per_call: int = field(
        default=5, metadata={"help": "Maximum number of memory items to merge in a single LLM call."}
    )
    enable_actor_critic: bool = field(default=True, metadata={"help": "Whether to enable the actor critic."})
    merging_system_prompt: Optional[str] = field(default=None, metadata={"help": "System prompt for memory merging."})
    merging_user_prompt: Optional[str] = field(default=None, metadata={"help": "User prompt for memory merging."})
    api_url: Optional[str] = field(default=None, metadata={"help": "The url of the api."})
    api_key: Optional[str] = field(default=None, metadata={"help": "The key of the api."})
    resume_from_checkpoint: Optional[str] = field(default=None, metadata={"help": "The path to the checkpoint."})
    memory_model_interaction_output_dir: Optional[str] = field(
        default=None, metadata={"help": "The directory to save the interaction outputs."}
    )


@dataclass
class EmbeddingModelConfig(WorkerConfig):
    pool_type: PoolType = field(
        default=PoolType.cls,
        metadata={"help": "The type of the pooling."},
    )


@dataclass
class MemoryConfig:
    # basic config
    enable: bool = field(default=False, metadata={"help": "Whether to enable the memory."})
    memory_type: MemoryType = field(
        default=MemoryType.case_memory,
        metadata={"help": "The type of the memory."},
    )
    memory_structure: MemoryStructure = field(
        default=MemoryStructure.tabuler,
        metadata={"help": "The structure of the memory."},
    )
    memory_integration_strategy: MemoryIntegrationStrategy = field(
        default=MemoryIntegrationStrategy.turn_based,
        metadata={"help": "Strategy for when to search/update memory."},
    )
    memory_format_pattern: Optional[str] = field(
        default=r"##\s*Title:\s*(?P<title>.+?)(?=\n|##|$).*?##\s*Content:\s*(?P<content>.+?)(?=##|$)",
        metadata={"help": "Regex pattern used to validate individual memory entries."},
    )
    use_memory_title_for_key: bool = field(
        default=False,
        metadata={"help": "Whether to derive the memory key from the extracted title of the memory entry."},
    )
    memory_load_path: Optional[str] = field(default=None, metadata={"help": "The path to the memory."})
    memory_save_path: Optional[str] = field(default=None, metadata={"help": "The path to the memory."})
    memory_save_interval: int = field(default=1000, metadata={"help": "The interval to save the memory."})
    memory_should_save: bool = field(default=True, metadata={"help": "Whether to save the memory."})
    memory_flush_timeout: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Timeout in seconds for flushing pending memory updates before suspend. "
                "When None (default), waits indefinitely until all updates complete. "
                "When set, the flush returns after the timeout and any remaining updates "
                "continue running in the background on the memory actor."
            )
        },
    )
    searcher: SearcherConfig = field(
        default_factory=SearcherConfig,
        metadata={"help": "The searcher config."},
    )
    updater: Optional[UpdaterConfig] = field(default=None, metadata={"help": "The updater config."})
    embedding_model: Optional[EmbeddingModelConfig] = field(
        default=None,
        metadata={"help": "The embedding model config."},
    )
    memory_model: Optional[MemoryModelConfig] = field(default=None, metadata={"help": "The memory model config."})
    memory_actor_train: Optional[MemoryModelWorkerConfigBase] = field(
        default=None, metadata={"help": "The memory actor train config."}
    )
    memory_actor_enable_training: bool = field(
        default=False,
        metadata={"help": "Whether to enable training the memory actor."},
    )

    # Optimization
    query_batch_size: int = field(
        default=16,
        metadata={"help": "The batch size for querying the memory."},
    )
    query_batch_timeout: float = field(
        default=0.01,
        metadata={"help": "The timeout for querying the memory."},
    )
    query_cache_size: int = field(
        default=1000,
        metadata={"help": "The cache size for querying the memory."},
    )
    faiss_rebuild_threshold: int = field(
        default=100,
        metadata={"help": "The threshold for rebuilding the FAISS index."},
    )

    # Deduplication config
    dedup_similarity_threshold: Optional[float] = field(
        default=None,
        metadata={
            "help": "Similarity threshold for memory deduplication. Entries with similarity >= threshold will be considered duplicates."
        },
    )
    dedup_batch_size: int = field(
        default=500,
        metadata={"help": "Batch size for deduplication to avoid memory overflow when processing large tasks."},
    )
    dedup_device: str = field(
        default="cuda",
        metadata={"help": "Device to use for deduplication computation (cuda or cpu)."},
    )
    memory_warmup_interval: int = field(
        default=0,
        metadata={"help": "The interval to warmup the memory."},
    )

    actor_critic_sys_prompt: Optional[str] = field(
        default=None, metadata={"help": "The system prompt of the actor critic."}
    )
    actor_critic_user_prompt: Optional[str] = field(
        default=None, metadata={"help": "The user prompt of the actor critic."}
    )
    trajectory_memory_injection: Optional[str] = field(
        default="system", metadata={"help": "The injection mode of the trajectory memory."}
    )

    def __post_init__(self):
        self._compiled_memory_format_pattern = None
        if self.memory_format_pattern:
            try:
                self._compiled_memory_format_pattern = re.compile(
                    self.memory_format_pattern, re.IGNORECASE | re.DOTALL
                )
            except re.error as exc:
                raise ValueError(f"Invalid memory_format_pattern '{self.memory_format_pattern}': {exc}") from exc

        if self.use_memory_title_for_key:
            if not self.memory_format_pattern:
                raise ValueError("use_memory_title_for_key requires memory_format_pattern to be provided.")
            if "title" not in self._compiled_memory_format_pattern.groupindex:
                raise ValueError(
                    "memory_format_pattern must define a named capturing group 'title' when use_memory_title_for_key is enabled."
                )

        if self.memory_should_save and self.memory_save_path is None:
            if self.memory_load_path is not None:
                self.memory_save_path = self.memory_load_path
                logger.warning(
                    f"memory_should_save is True, but memory_save_path is not set, so memory_save_path will be set to {self.memory_save_path}"
                )
            else:
                logger.warning(
                    "memory_should_save is True, but memory_load_path is not set, so memory_should_save will be ignored"
                )

        if self.searcher.worker_cls is None:
            self.searcher.worker_cls = "roll.pipeline.agentic.memory.searcher.search_worker.MemorySearcherWorker"

        if self.searcher.name is None:
            self.searcher.name = f"memory_{self.searcher.memory_search_strategy}_searcher"

        if self.updater is not None:
            if self.updater.worker_cls is None:
                self.updater.worker_cls = "roll.pipeline.agentic.memory.updater.update_worker.MemoryUpdaterWorker"
            if self.updater.name is None:
                self.updater.name = f"memory_{self.updater.memory_update_strategy}_updater"

        if self.embedding_model is not None:
            if self.embedding_model.worker_cls is None:
                self.embedding_model.worker_cls = (
                    "roll.pipeline.agentic.memory.embedding.embedding_worker.EmbeddingWorker"
                )
            if self.embedding_model.name is None:
                self.embedding_model.name = f"memory_embedding"

        if self.memory_model is not None:
            if self.memory_model.worker_cls is None:
                self.memory_model.worker_cls = (
                    "roll.pipeline.agentic.memory.memory_model.memory_model_worker.MemoryModelWorker"
                )

        if self.memory_actor_train is not None:
            if self.memory_actor_train.worker_cls is None:
                self.memory_actor_train.worker_cls = (
                    "roll.pipeline.agentic.memory.memory_model.memory_model_worker.MemoryActorWorker"
                )

        if self.memory_model is not None:
            if self.memory_model.name is None:
                self.memory_model.name = f"memory_model"

        if self.memory_actor_train is not None:
            if self.memory_actor_train.name is None:
                self.memory_actor_train.name = f"memory_actor_train"

        if self.updater is not None:
            self.memory_should_update = True
        else:
            self.memory_should_update = False
