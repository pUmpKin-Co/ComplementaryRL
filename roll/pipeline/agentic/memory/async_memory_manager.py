import asyncio
import hashlib
import json
import logging
import math
import os
import random
import re
import time
import uuid as uuid_module
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import ray
import torch
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from tensordict import TensorDict

from roll.distributed.executor.cluster import Cluster
from roll.distributed.scheduler.generate_scheduler import RequestScheduler
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.scheduler.resource_manager import ResourceManager
from roll.models.model_providers import default_processor_provider, default_tokenizer_provider
from roll.pipeline.agentic.env_manager.token_mask_utils import custom_apply_chat_template, token_ids_to_assistant_mask
from roll.pipeline.agentic.memory.memory_config import MemoryConfig, MemoryModelType, SampleStrategy
from roll.pipeline.agentic.memory.merge_utils import (
    GLOBAL_MERGE_SCOPE_ID,
    build_multitask_merge_targets,
    build_tabular_merge_targets,
    resolve_tabular_merging_threshold,
)
from roll.pipeline.agentic.memory.memory_structure import MultiTaskTabulerMemory, TabulerMemory, memory_structure_dict
from roll.pipeline.agentic.memory.rwlock import AsyncRWLock
from roll.utils.constants import RAY_NAMESPACE
from roll.utils.logging import get_logger

logger = get_logger()


def parser_answer_func(text: str, action_pattern: str) -> Optional[Dict[str, str]]:
    matches = list(re.finditer(action_pattern, text, re.DOTALL | re.IGNORECASE))
    if not matches:
        return None

    match = matches[-1]
    groups = match.groups()

    think_content = ""
    if not groups:
        # No capture groups – fall back to the whole match.
        action_content = match.group(0).strip()
    elif len(groups) == 1:
        action_content = groups[0].strip()
    else:
        think_content = groups[0].strip()
        action_content = groups[-1].strip()

    action_info = {
        "action": None,
        "action_content": action_content,
        "think_content": think_content,
    }
    return action_info


def parse_answer_to_list(content: str) -> Optional[list]:
    """
    Parse merged knowledge items from LLM response using strict delimiter format.

    Only accepts the format: <<<KNOWLEDGE_ITEM>>>...<<<END_ITEM>>>
    Special-case support:
      - If the model outputs "<answer>return</answer>" (case-insensitive), treat as a valid no-op and return [].

    If parsing fails, returns None so the caller can keep original entries.

    Returns:
        List of knowledge strings, [] for explicit no-op "return", or None if parsing fails
    """
    if not content:
        logger.warning("parse_answer_to_list: Content is empty")
        return None

    # Extract content from <answer> tags (required)
    answer_pattern = r"<answer>(.*?)</answer>"
    answer_info = parser_answer_func(content, answer_pattern)

    if not answer_info:
        logger.warning("parse_answer_to_list: <answer> tags not found")
        return None

    answer_text = answer_info["action_content"]
    if not answer_text:
        logger.warning("parse_answer_to_list: Content within <answer> tags is empty")
        return None

    if answer_text.strip().lower() == "return":
        logger.info("parse_answer_to_list: Received explicit 'return' (no-op) from merge model")
        return []

    stripped = answer_text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            obj = json.loads(stripped)
            if isinstance(obj, dict) and str(obj.get("function_name", "")).lower() == "return":
                logger.info("parse_answer_to_list: Received JSON return (no-op) from merge model")
                return []
        except Exception:
            pass

    # Parse using delimiter format only
    pattern = r"<<<KNOWLEDGE_ITEM>>>(.*?)<<<END_ITEM>>>"
    matches = re.findall(pattern, answer_text, re.DOTALL)

    if not matches:
        logger.warning("parse_answer_to_list: No <<<KNOWLEDGE_ITEM>>>...<<<END_ITEM>>> blocks found")
        return None

    result = []
    for match_content in matches:
        cleaned = match_content.strip()
        if cleaned and len(cleaned) > 30 and not _is_placeholder(cleaned):
            result.append(cleaned)
        else:
            logger.warning(f"parse_answer_to_list: Rejected item (too short or placeholder): {cleaned[:50]}...")

    if not result:
        logger.warning("parse_answer_to_list: All parsed items were filtered out")
        return None

    logger.info(f"parse_answer_to_list: Successfully parsed {len(result)} knowledge items")
    return result


def _is_placeholder(text: str) -> bool:
    """Check if text is likely a placeholder rather than real content."""
    text_lower = text.lower().strip()

    # Common placeholder patterns
    placeholder_patterns = [
        r"^string\d*$",  # "string1", "string2"
        r"^item\s*\d*$",  # "item 1", "item2"
        r"^knowledge\s*\d*$",  # "knowledge1"
        r"^entry\s*\d*$",  # "entry 1"
        r"^your\s+(comprehensive\s+)?knowledge",  # "your comprehensive knowledge"
        r"^\[.*placeholder.*\]$",  # "[placeholder]"
        r"^<.*>$",  # "<content>"
        r"^\.\.\.$",  # "..."
        r"^xxx+$",  # "xxx"
        r"^first\s+(merged\s+)?knowledge",  # "first merged knowledge"
        r"^second\s+(merged\s+)?knowledge",  # "second merged knowledge"
    ]

    for pattern in placeholder_patterns:
        if re.match(pattern, text_lower):
            return True

    generic_words = {"example", "sample", "test", "placeholder", "todo", "tbd", "na", "none"}
    if text_lower in generic_words:
        return True

    return False


@ray.remote
class AsyncMemoryManager:
    def __init__(
        self,
        memory_config: MemoryConfig,
        mode: str = "train",
        resource_manager: Optional[ResourceManager] = None,
        memory_model_cluster: Optional[Cluster] = None,
    ):
        self.memory_config = memory_config
        self.memory_warmup_interval = memory_config.memory_warmup_interval
        self.current_memory_warmup_interval = 0
        self.begin_interaction = False
        self.mode = mode
        self.resource_manager = resource_manager
        # Batch processing configuratiocachen
        self.query_batch_size = getattr(memory_config, "query_batch_size", 16)
        self.query_batch_timeout = getattr(memory_config, "query_batch_timeout", 0.01)  # 10ms default
        self.query_cache_size = getattr(memory_config, "query_cache_size", 1000)
        self.query_cache_enabled = self.query_cache_size > 0

        # Query cache: LRU cache for query embeddings
        # Key: hash of query string, Value: (embedding, timestamp)

        self.query_cache = OrderedDict()
        self.query_cache_hits = 0
        self.query_cache_misses = 0

        # pending queries
        self.pending_queries = []
        self.query_lock = asyncio.Lock()
        self.batch_processor_task = None
        self.batch_event = asyncio.Event()
        self.first_query_timestamp = None

        # Separate flag for search operations (should continue during suspend)
        # self.running controls updates, self.running_searches controls searches
        self.running_searches = True

        # Batch processing statistics
        self.batch_stats = {
            "total_batches": 0,
            "total_queries_batched": 0,
            "timeout_batches": 0,
            "full_batches": 0,
            "avg_batch_size": 0.0,
        }

        # -------------------------
        # Usage stats (repeat-control)
        # -------------------------
        # Track how often a memory UID is retrieved / used for training.
        # Kept separate from the memory entry itself to avoid write-locking the
        # memory structure on every search.
        self._usage_lock = asyncio.Lock()
        # uid -> {"retrieve_count": int, "train_count": int, "last_retrieved_ts": float, "last_trained_ts": float, "last_trained_step": int}
        self._uid_usage: Dict[str, Dict[str, Any]] = {}

        # -------------------------
        # Per-tag performance tracking
        # -------------------------
        self._tag_performance_lock = asyncio.Lock()
        self._tag_performance: Dict[str, Dict[str, Any]] = {}

        # Setup dedicated logging for memory manager (Ray actor)
        self._setup_memory_manager_logging()

        # Initialize memory structure
        self.memory = memory_structure_dict[memory_config.memory_structure](memory_config)

        # Initialize searcher cluster (distributed workers)
        if memory_config.searcher is not None:
            self.searcher_cluster = Cluster(
                name=memory_config.searcher.name,
                worker_cls=memory_config.searcher.worker_cls,
                resource_manager=resource_manager,
                worker_config=memory_config.searcher,
            )

        if self.memory_config.searcher.memory_search_strategy in [
            SampleStrategy.embedding_similarity,
            SampleStrategy.faiss_embedding_similarity,
        ]:
            self.embedding_cluster = Cluster(
                name=memory_config.embedding_model.name,
                worker_cls=memory_config.embedding_model.worker_cls,
                resource_manager=resource_manager,
                worker_config=memory_config.embedding_model,
            )

        # Memory model cluster can be provided externally (e.g., by pipeline for training)
        # or created internally (backward compatible)
        if self.memory_config.memory_model is not None:
            if memory_model_cluster is not None:
                # Use externally provided cluster (e.g., from pipeline)
                self.memory_model_cluster = memory_model_cluster
                self.owns_memory_model_cluster = False
                logger.info(f"Using externally provided memory_model_cluster")
            else:
                # Create our own cluster (backward compatible)
                self.memory_model_cluster = Cluster(
                    name=memory_config.memory_model.name,
                    worker_cls=memory_config.memory_model.worker_cls,
                    resource_manager=resource_manager,
                    worker_config=memory_config.memory_model,
                )
                self.owns_memory_model_cluster = True
                logger.info(f"Created internal memory_model_cluster")
        else:
            self.owns_memory_model_cluster = False

        # Initialize updater cluster (typically world_size=1 for thread-safety)
        if memory_config.updater is not None:
            self.updater_cluster = Cluster(
                name=memory_config.updater.name,
                worker_cls=memory_config.updater.worker_cls,
                resource_manager=resource_manager,
                worker_config=memory_config.updater,
            )

        # Round-robin index for load balancing across searcher workers
        self._searcher_worker_index = 0
        self._embedding_worker_index = 0

        # 1. Multiple searches can run concurrently (all acquire read lock)
        # 2. Updates are exclusive (acquire write lock, block all searches)
        # 3. Writer priority: pending updates block new searches
        self.memory_lock = AsyncRWLock()

        # Update concurrency control
        # We want to allow multiple generations to run in parallel
        concurrency_limit = 1
        if hasattr(self, "memory_model_cluster") and self.memory_model_cluster is not None:
            concurrency_limit = max(1, self.memory_model_cluster.world_size * 4)

        self.update_semaphore = asyncio.Semaphore(concurrency_limit)
        self.active_update_tasks = set()
        logger.info(f"Update concurrency limit set to: {concurrency_limit}")

        # Update queue for background processing
        self.update_queue = asyncio.Queue()

        # Performance tracking
        self.performance_stats = {
            "concurrent_searches": 0,
            "pending_updates": 0,
            "total_operations": 0,
            "average_operation_time": 0.0,
            "update_wait_time": 0.0,
        }

        # Background task for processing updates
        self.update_processor_task = None
        self.running = True

        logger.info(f"AsyncMemoryManager initialized with {memory_config.memory_structure} structure")
        if memory_config.updater is not None:
            logger.info(f"Updater cluster will use world_size={memory_config.updater.world_size}")
        if memory_config.updater is not None and memory_config.updater.world_size > 1:
            logger.warning(
                "Updater world_size > 1 detected. Ensure your updater is thread-safe "
                "or use world_size=1 to avoid race conditions!"
            )

    def _get_uid_usage_ref(self, uid: str) -> Dict[str, Any]:
        stat = self._uid_usage.get(uid)
        if stat is None:
            stat = {
                "retrieve_count": 0,
                "train_count": 0,
                "last_retrieved_ts": 0.0,
                "last_trained_ts": 0.0,
                "last_trained_step": -1,
            }
            self._uid_usage[uid] = stat
        return stat

    async def _record_retrieval_uids(self, uids: List[str]) -> None:
        if not uids:
            return
        now = time.time()
        async with self._usage_lock:
            for uid in uids:
                stat = self._get_uid_usage_ref(uid)
                stat["retrieve_count"] = int(stat.get("retrieve_count", 0)) + 1
                stat["last_retrieved_ts"] = now

    async def _record_training_uids(
        self,
        uids_list: List[List[str]],
        global_step: int,
        weights: Optional[List[float]] = None,
    ) -> None:
        if not uids_list:
            return
        now = time.time()
        async with self._usage_lock:
            for i, uids in enumerate(uids_list):
                if not uids:
                    continue
                w = 1.0
                if weights is not None and i < len(weights) and weights[i] is not None:
                    w = float(weights[i])
                if w <= 0:
                    continue
                for uid in uids:
                    stat = self._get_uid_usage_ref(uid)
                    stat["train_count"] = int(stat.get("train_count", 0)) + 1
                    stat["last_trained_ts"] = now
                    stat["last_trained_step"] = max(int(stat.get("last_trained_step", -1)), int(global_step))

    async def _get_usage_snapshot(self, uids: List[str]) -> Dict[str, Dict[str, Any]]:
        async with self._usage_lock:
            out: Dict[str, Dict[str, Any]] = {}
            for uid in uids:
                out[uid] = self._get_uid_usage_ref(uid).copy()
            return out

    def _diversify_uids_by_usage(
        self,
        uids: List[str],
        usage_snapshot: Dict[str, Dict[str, Any]],
        diversity_lambda: float,
        diversity_recent_seconds: float,
        diversity_dropout_p: float,
    ) -> List[str]:
        if not uids:
            return uids

        now = time.time()
        scored = []
        for rank, uid in enumerate(uids):
            stat = usage_snapshot.get(uid, {})
            retrieve_count = int(stat.get("retrieve_count", 0))
            last_retrieved_ts = float(stat.get("last_retrieved_ts", 0.0))
            is_recent = (now - last_retrieved_ts) < float(diversity_recent_seconds) if last_retrieved_ts > 0 else False

            base_rank_score = -rank
            penalty = float(diversity_lambda) * math.log1p(retrieve_count) + (1.0 if is_recent else 0.0)
            score = base_rank_score - penalty
            scored.append((score, rank, uid))

        # Stable-ish: keep original order when scores tie.
        scored.sort(key=lambda x: (x[0], -x[1]), reverse=True)
        reranked = [uid for _, _, uid in scored]

        # Optional stochastic dropout of the most recently used top result.
        if diversity_dropout_p and reranked:
            top_uid = reranked[0]
            stat = usage_snapshot.get(top_uid, {})
            last_retrieved_ts = float(stat.get("last_retrieved_ts", 0.0))
            if last_retrieved_ts > 0 and (now - last_retrieved_ts) < float(diversity_recent_seconds):
                if random.random() < float(diversity_dropout_p):
                    reranked = reranked[1:] + [top_uid]

        return reranked

    async def get_usage_stats(self, uids: List[str]) -> Dict[str, Dict[str, Any]]:
        if uids is None:
            uids = []
        return await self._get_usage_snapshot(uids)

    async def record_training(
        self,
        uids_list: List[List[str]],
        global_step: int,
        weights: Optional[List[float]] = None,
    ) -> bool:
        await self._record_training_uids(uids_list=uids_list, global_step=global_step, weights=weights)
        return True

    def _hash_query(self, query: str) -> str:
        """Create stable hash for query caching"""
        return hashlib.md5(query.encode("utf-8")).hexdigest()

    def _safe_log(self, level: str, message: str, *args, **kwargs):
        try:
            log_fn = getattr(logger, level.lower())
            log_fn(message, *args, **kwargs)
        except (OSError, IOError) as e:
            # Silently suppress I/O errors during logging to prevent crashes
            # In production, the system should continue working even if logging fails
            pass
        except Exception:
            # Catch any other unexpected logging errors
            pass

    def _setup_memory_manager_logging(self):
        global logger

        log_dir = os.environ.get("ROLL_LOG_DIR", "./output/logs")
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pid = os.getpid()
        log_filename = f"memory_manager_pid{pid}_{timestamp}.log"
        log_path = os.path.join(log_dir, log_filename)

        try:
            # Create file handler with better error handling
            file_handler = logging.FileHandler(log_path, mode="a")
            file_handler.setLevel(logging.INFO)

            # Create detailed formatter for memory manager logs
            formatter = logging.Formatter(
                fmt="[%(asctime)s] [%(filename)s:%(lineno)d] [%(levelname)s] [MemoryManager PID:%(process)d] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            file_handler.setFormatter(formatter)

            # Store handler reference for cleanup during shutdown
            self._log_file_handler = file_handler

            logger.addHandler(file_handler)

            root_logger = logging.getLogger()
            root_logger.addHandler(file_handler)

            logger.info(f"=== Memory Manager Logging Initialized ===")
            logger.info(f"Log file: {log_path}")
            logger.info(f"Process ID: {pid}")
            logger.info(f"Logger details: name={logger.name}, level={logging.getLevelName(logger.level)}")

        except Exception as e:
            self._log_file_handler = None

    async def initialize(self):
        if self.memory_config.searcher is not None:
            self.searcher_cluster.initialize(self.memory_config)
            logger.info(f"Searcher cluster initialized with {self.searcher_cluster.world_size} workers")

        if self.memory_config.updater is not None:
            self.updater_cluster.initialize(self.memory_config)
            logger.info(f"Updater cluster initialized with {self.updater_cluster.world_size} workers")

        if self.memory_config.searcher is not None and self.memory_config.searcher.memory_search_strategy in [
            SampleStrategy.embedding_similarity,
            SampleStrategy.faiss_embedding_similarity,
        ]:
            self.embedding_cluster.initialize()
            logger.info(f"Embedding cluster initialized with {self.embedding_cluster.world_size} workers")
            self.memory.init_embedding_for_entry(self.embedding_cluster)

        if self.memory_config.memory_model is not None:
            # Only initialize if we own the cluster (not provided externally)
            if self.owns_memory_model_cluster:
                self.memory_model_cluster.initialize()
                logger.info(f"Memory model cluster initialized with {self.memory_model_cluster.world_size} workers")
            else:
                logger.info(
                    f"Using pre-initialized memory model cluster with {self.memory_model_cluster.world_size} workers"
                )

            if self.memory_config.memory_model.memory_model_type == MemoryModelType.local_model:
                self.tokenizer = default_tokenizer_provider(model_args=self.memory_config.memory_model.model_args)
                # dummy data for starting server
                data = DataProto()
                await asyncio.gather(
                    *[
                        asyncio.wrap_future(ref.obj_ref.future())
                        for ref in self.memory_model_cluster.start_server(data, blocking=False)
                    ],
                )

                logger.info("Memory local model started server")

                self.generate_scheduler = RequestScheduler.options(
                    scheduling_strategy=NodeAffinitySchedulingStrategy(
                        node_id=ray.get_runtime_context().get_node_id(),
                        soft=False,
                    ),
                    max_concurrency=1000,
                ).remote(
                    infer_cluster=self.memory_model_cluster,
                    pipeline_config=self.memory_config,
                )

        # Start background update processor
        self.update_processor_task = asyncio.create_task(self._process_updates())

        # Start background batch processor for query batching
        self.batch_processor_task = asyncio.create_task(self._batch_processor_loop())

        # The memory_structure will load the memory from the load_path when initialized
        if self.memory_config.memory_load_path:
            if self.memory_config.searcher is not None:
                self.searcher_cluster.load_state(self.memory_config.searcher.memory_searcher_state_path)
            logger.info("Searcher state loaded across all workers")

        logger.info("AsyncMemoryManager fully initialized")

    async def search(self, query: str, task_id: str = None) -> Tuple[str, List[str], List[DataProto]]:
        """
        Async search operation with batching and proper synchronization.

        Batching Strategy:
        -----------------
        1. Queries accumulate until batch_size is reached OR timeout expires
        2. Background task processes batches automatically
        3. Each query gets its result via a Future

        Args:
            query: Search query string
            task_id: Optional task ID for task-specific search (required for MultiTaskTabulerMemory)

        Returns:
            Tuple of (message, uids, triggered_interactions)
        """
        future = asyncio.Future()
        current_time = time.time()

        async with self.query_lock:
            # Add query to pending batch with task_id
            # Batch structure: (query, task_id, future, timestamp)
            self.pending_queries.append((query, task_id, future, current_time))

            # Track timestamp of first query in batch
            if self.first_query_timestamp is None:
                self.first_query_timestamp = current_time

            # Trigger immediate processing if batch is full
            if len(self.pending_queries) >= self.query_batch_size:
                self.batch_event.set()

        return await future

    async def _batch_processor_loop(self):
        """
        Background task that processes query batches.

        Triggers batch processing when:
        1. Batch size threshold is reached (immediate)
        2. Timeout expires since first query (time-based)

        Note: Uses self.running_searches instead of self.running so searches
        can continue during suspend (which only stops updates).
        """
        while self.running_searches:
            try:
                # Wait for either batch_size trigger or timeout
                await asyncio.wait_for(
                    self.batch_event.wait(),
                    timeout=self.query_batch_timeout,
                )
                self.batch_event.clear()
            except asyncio.TimeoutError:
                # Timeout expired - process partial batch if any
                pass

            # Check if we should process a batch
            async with self.query_lock:
                if not self.pending_queries:
                    continue

                # Check if timeout has expired since first query
                current_time = time.time()
                time_since_first = current_time - self.first_query_timestamp if self.first_query_timestamp else 0

                should_process = (
                    len(self.pending_queries) >= self.query_batch_size  # Full batch
                    or time_since_first >= self.query_batch_timeout  # Timeout
                )

                if not should_process:
                    continue

                # Extract batch to process
                batch_size = min(len(self.pending_queries), self.query_batch_size)
                batch = self.pending_queries[:batch_size]
                self.pending_queries = self.pending_queries[batch_size:]

                # Reset timestamp if queue is empty
                if not self.pending_queries:
                    self.first_query_timestamp = None
                else:
                    # Update timestamp to first remaining query
                    # Batch structure: (query, task_id, future, timestamp)
                    self.first_query_timestamp = self.pending_queries[0][3]

            # Process batch outside the lock
            if batch:
                asyncio.create_task(self._process_query_batch(batch))

    def _get_cached_embedding(self, query: str):
        """
        Get cached embedding for query.

        Returns:
            embedding if found, None otherwise
        """
        if not self.query_cache_enabled:
            return None

        query_hash = self._hash_query(query)
        if query_hash in self.query_cache:
            self.query_cache.move_to_end(query_hash)
            embedding, _ = self.query_cache[query_hash]
            self.query_cache_hits += 1
            return embedding

        self.query_cache_misses += 1
        return None

    def _cache_embedding(self, query: str, embedding):
        """
        Cache embedding for query (LRU eviction).

        Args:
            query: Query string
            embedding: Query embedding tensor
        """
        if not self.query_cache_enabled:
            return

        query_hash = self._hash_query(query)
        current_time = time.time()

        self.query_cache[query_hash] = (embedding, current_time)

        if len(self.query_cache) > self.query_cache_size:
            self.query_cache.popitem(last=False)

    def clear_query_cache(self):
        """
        Clear the query embedding cache.

        This should be called when the memory structure changes significantly
        (e.g., after loading new state) to ensure cache consistency.
        """
        if self.query_cache_enabled:
            cache_size = len(self.query_cache)
            self.query_cache.clear()
            logger.info(f"Cleared query cache ({cache_size} entries)")

    async def _process_query_batch(self, batch: List[Tuple[str, Optional[str], asyncio.Future, float]]):
        """
        Process a batch of queries with caching.

        Args:
            batch: List of (query, task_id, future, timestamp) tuples
        """
        start_time = time.time()
        batch_size = len(batch)

        # Batch structure: (query, task_id, future, timestamp)
        queries = [q for q, _, _, _ in batch]
        task_ids = [t for _, t, _, _ in batch]
        futures = [f for _, _, f, _ in batch]

        try:
            # STEP 1: Get search info (read lock - allows concurrent searches)
            # For MultiTaskTabulerMemory, we need to get search_info per task_id
            # For regular TabulerMemory, we get all search_info once
            async with self.memory_lock.reader():
                # Check if memory is MultiTaskTabulerMemory
                is_multi_task = hasattr(self.memory, "task_to_uid")

                if is_multi_task:
                    # For multi-task memory, we need to handle each query separately
                    # since they may have different task_ids
                    # We'll get search_info per query in the search loop below
                    search_info_map = {}  # Map task_id -> search_info
                else:
                    # For regular memory, get all search_info once
                    search_info = self.memory.get_search_info()

            diversity_enable = bool(getattr(self.memory_config.searcher, "diversity_enable", False))
            base_k = int(getattr(self.memory_config.searcher, "memory_fetch_num", 0))
            # STEP 2: Check cache and batch encode queries (if using embeddings)
            if self.memory_config.searcher is not None and self.memory_config.searcher.memory_search_strategy in [
                SampleStrategy.embedding_similarity,
                SampleStrategy.faiss_embedding_similarity,
            ]:
                # Check cache for each query
                query_embeddings = []
                queries_to_encode = []
                cache_indices = []  # Indices of queries that need encoding

                for i, query in enumerate(queries):
                    cached_emb = self._get_cached_embedding(query)
                    if cached_emb is not None:
                        query_embeddings.append(cached_emb)
                    else:
                        query_embeddings.append(None)  # Placeholder
                        queries_to_encode.append(query)
                        cache_indices.append(i)

                # Batch encode only non-cached queries
                if queries_to_encode:
                    worker_index = self._embedding_worker_index % self.embedding_cluster.world_size
                    self._embedding_worker_index += 1
                    embedding_worker = self.embedding_cluster.workers[worker_index]

                    # CRITICAL: Batch encoding - process all queries in one GPU call
                    new_embeddings = await embedding_worker.encode.remote(queries_to_encode)

                    # Fill in the new embeddings and cache them
                    for idx, emb in zip(cache_indices, new_embeddings):
                        query_embeddings[idx] = emb
                        self._cache_embedding(queries[idx], emb)
            else:
                # For non-embedding strategies, queries are strings
                query_embeddings = queries

            # STEP 3: Batch search across workers (no lock needed here)
            if is_multi_task:
                # For multi-task: Group queries by task_id (which determines search_info)
                # Then batch search each group
                task_groups = {}  # task_id -> list of (query_idx, query_emb)
                for i, query_emb in enumerate(query_embeddings):
                    task_id = task_ids[i]
                    if task_id is None:
                        logger.warning(f"task_id is required for MultiTaskTabulerMemory but was None for query {i}")
                        # Use special key for None task_id
                        task_id = "__NONE__"

                    if task_id not in task_groups:
                        task_groups[task_id] = []
                    task_groups[task_id].append((i, query_emb))

                # Get search_info for each unique task_id (with read lock)
                async with self.memory_lock.reader():
                    for task_id_key in task_groups:
                        if task_id_key == "__NONE__":
                            search_info_map[task_id_key] = []
                        elif task_id_key not in search_info_map:
                            try:
                                search_info_map[task_id_key] = self.memory.get_search_info(task_id_key)
                            except Exception as e:
                                logger.warning(f"Failed to get search_info for task_id {task_id_key}: {e}")
                                search_info_map[task_id_key] = []

                # Process each task group with batch search
                searcher_tasks = []
                result_indices = []  # Track which result goes to which original query index

                for task_id_key, query_group in task_groups.items():
                    query_indices = [idx for idx, _ in query_group]
                    query_embs = [emb for _, emb in query_group]
                    group_search_info = search_info_map[task_id_key]

                    # Use round-robin to distribute groups across workers
                    worker_idx = (self._searcher_worker_index + len(searcher_tasks)) % self.searcher_cluster.world_size
                    searcher_worker = self.searcher_cluster.workers[worker_idx]

                    # Batch search for this group
                    if len(query_embs) == 1:
                        searcher_tasks.append(searcher_worker.search.remote(query_embs[0], group_search_info))
                        result_indices.append(query_indices[0])
                    else:
                        searcher_tasks.append(searcher_worker.search_batch.remote(query_embs, group_search_info))
                        result_indices.append(query_indices)

                # Gather all results
                search_results = await asyncio.gather(*searcher_tasks)

                # Reconstruct uids_list in original query order
                uids_list = [None] * batch_size
                for result, indices in zip(search_results, result_indices):
                    if isinstance(indices, int):
                        uids_list[indices] = result if isinstance(result, list) else []
                    else:
                        for idx, uids in zip(indices, result):
                            uids_list[idx] = uids if isinstance(uids, list) else []
            else:
                # For regular memory: All queries use same search_info, batch them all
                worker_idx = self._searcher_worker_index % self.searcher_cluster.world_size
                searcher_worker = self.searcher_cluster.workers[worker_idx]

                if len(query_embeddings) == 1:
                    # Single query - use search()
                    # Wrap in list to match batch format: [List[str]]
                    uids_list = [await searcher_worker.search.remote(query_embeddings[0], search_info)]
                else:
                    # Multiple queries - use batch search for efficiency
                    uids_list = await searcher_worker.search_batch.remote(query_embeddings, search_info)

            # STEP 4: Format results (read lock - allows concurrent searches)
            # NOTE: The search stage may run while updates happen, so UIDs returned by the
            # searcher can become stale. We filter to UIDs that still exist at formatting time,
            # so returned `uids` always corresponds to the memories actually shown.
            uids_for_access_time_update: List[str] = []
            async with self.memory_lock.reader():
                # We only want to count "retrievals" for the memories actually shown to the model,
                # i.e. after rerank+truncate to k (not the larger candidate set).
                retrieval_uids_to_record: List[str] = []
                for future, uids in zip(futures, uids_list):
                    if not future.done():  # Check if future wasn't cancelled
                        message = "[No Memory Found]"
                        triggered = []
                        uids_to_return: List[str] = []
                        if len(uids) > 0:
                            # Optional: diversify retrieval results (reduce repeated top-1 selection)
                            if diversity_enable:
                                usage_snapshot = await self._get_usage_snapshot(uids)
                                diversity_lambda = float(getattr(self.memory_config.searcher, "diversity_lambda", 0.0))
                                diversity_recent_seconds = float(
                                    getattr(self.memory_config.searcher, "diversity_recent_seconds", 30.0)
                                )
                                diversity_dropout_p = float(
                                    getattr(self.memory_config.searcher, "diversity_dropout_p", 0.0)
                                )
                                uids = self._diversify_uids_by_usage(
                                    uids=uids,
                                    usage_snapshot=usage_snapshot,
                                    diversity_lambda=diversity_lambda,
                                    diversity_recent_seconds=diversity_recent_seconds,
                                    diversity_dropout_p=diversity_dropout_p,
                                )
                                # After rerank, keep only top-k entries (the rest were only candidates)
                                if base_k > 0 and len(uids) > base_k:
                                    uids = uids[:base_k]

                            # Filter out stale UIDs before formatting (the search stage can race with updates).
                            uids_existing = [uid for uid in uids if self.memory.get_entry_by_uid(uid) is not None]
                            if uids_existing:
                                message, triggered = self.memory.search_entry(uids_existing)
                                uids_to_return = uids_existing
                            for interaction in triggered:
                                try:
                                    if interaction.non_tensor_batch is None:
                                        interaction.non_tensor_batch = {}
                                    # DataProto requires non_tensor_batch values to be np.ndarray(dtype=object)
                                    # with first dimension == batch_size. Most triggered interactions are batch_size=1.
                                    bs = 1
                                    try:
                                        if interaction.batch is not None:
                                            bs = int(interaction.batch.batch_size[0])
                                    except Exception:
                                        bs = 1
                                    arr = np.empty(bs, dtype=object)
                                    arr[:] = [list(uids_to_return)] * bs
                                    interaction.non_tensor_batch["last_turn_uids"] = arr
                                except Exception:
                                    pass
                            retrieval_uids_to_record.extend(uids_to_return)
                            uids_for_access_time_update.extend(uids_to_return)

                        future.set_result((message, uids_to_return, triggered))

            if retrieval_uids_to_record:
                await self._record_retrieval_uids(retrieval_uids_to_record)

            # STEP 5: Update access times (async, fire-and-forget)
            if self.mode == "train" and self.memory_config.updater is not None and self.updater_cluster.world_size > 0:
                all_uids = list(dict.fromkeys(uids_for_access_time_update))
                if all_uids:  # Only update if there are UIDs
                    updater_worker = self.updater_cluster.workers[0]
                    asyncio.create_task(self._update_access_times_async(updater_worker, all_uids))

            # Update statistics
            operation_time = time.time() - start_time
            self.performance_stats["total_operations"] += batch_size

            # Update batch statistics
            self.batch_stats["total_batches"] += 1
            self.batch_stats["total_queries_batched"] += batch_size

            if batch_size >= self.query_batch_size:
                self.batch_stats["full_batches"] += 1
            else:
                self.batch_stats["timeout_batches"] += 1

            self.batch_stats["avg_batch_size"] = (
                self.batch_stats["total_queries_batched"] / self.batch_stats["total_batches"]
            )

            # Update average operation time (per query, not per batch)
            avg_per_query = operation_time / batch_size
            total_ops = self.performance_stats["total_operations"]
            self.performance_stats["average_operation_time"] = (
                self.performance_stats["average_operation_time"] * (total_ops - batch_size) + operation_time
            ) / total_ops

        except Exception as e:
            logger.error(f"Error processing query batch: {e}")
            # Set exception on all futures in batch
            for future in futures:
                if not future.done():
                    future.set_exception(e)

    async def _update_access_times_async(self, updater_worker, uids: List[str]):
        await updater_worker.update_access_times.remote(uids)

    async def update(self, data: Any, mode: Optional[str] = None) -> bool:
        if mode is None:
            mode = self.mode

        if mode != "train":
            return True

        await self.update_queue.put((data, time.time()))
        self.performance_stats["pending_updates"] += 1

        return True

    def parse_memory_from_response(self, response: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """
        Parse structured memory from LM response using configurable pattern.

        Expected format (default):
        <answer>
        ## Title: <title>
        ## Description: <description>
        ## Content: <content>
        </answer>

        Args:
            response: Raw LM response text

        Returns:
            Formatted memory string if parsing succeeds, None otherwise
        """
        # Get pattern from config, use default if not set
        answer_pattern = (
            self.memory_config.memory_model.memory_model_response_pattern
            if self.memory_config.memory_model is not None
            else r"<answer>(.*?)</answer>"
        )
        answer_info = parser_answer_func(response, answer_pattern)
        if not answer_info:
            logger.warning(f"Pattern '{answer_pattern}' not found in response. Skipping memory update.")
            logger.debug(f"Response content: {response[:500]}...")
            return None, None

        answer_content = answer_info["action_content"]

        normalized_memory, parsed_fields = self._normalize_memory_text(answer_content)

        if parsed_fields is None:
            if self.memory_config.memory_format_pattern:
                logger.warning("Memory entry did not match expected format pattern. Using original content.")
                logger.debug(f"Response content: {response[:500]}...")
        elif not self.validate_memory_format(answer_content, parsed_fields=parsed_fields):
            logger.warning("Invalid memory format in response. Using original content.")
            logger.debug(f"Response content: {response[:500]}...")

        if not normalized_memory:
            logger.warning("Normalized memory content is empty. Skipping update.")
            return None, parsed_fields

        return normalized_memory, parsed_fields

    def _get_compiled_memory_format_pattern(self):
        pattern = getattr(self.memory_config, "_compiled_memory_format_pattern", None)
        if pattern is not None:
            return pattern

        raw_pattern = getattr(self.memory_config, "memory_format_pattern", None)
        if not raw_pattern:
            return None

        try:
            pattern = re.compile(raw_pattern, re.IGNORECASE | re.DOTALL)
        except re.error as exc:
            self._safe_log(
                "error",
                f"Failed to compile memory_format_pattern '{raw_pattern}': {exc}",
            )
            return None

        self.memory_config._compiled_memory_format_pattern = pattern
        return pattern

    def _extract_memory_fields(self, memory_str: str) -> Optional[Dict[str, Any]]:
        pattern = self._get_compiled_memory_format_pattern()
        if pattern is None or not memory_str:
            return None

        match = pattern.search(memory_str)
        if not match:
            return None

        groups = {}
        for key, value in match.groupdict().items():
            if isinstance(value, str):
                groups[key] = value.strip()
            else:
                groups[key] = value
        return groups

    def _normalize_memory_text(self, memory_text: str) -> Tuple[str, Optional[Dict[str, Any]]]:
        if not isinstance(memory_text, str):
            return memory_text, None

        normalized = memory_text.strip()
        parsed_fields = self._extract_memory_fields(normalized)
        if not parsed_fields:
            return normalized, None

        content = parsed_fields.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip(), parsed_fields

        return normalized, parsed_fields

    def _extract_memory_title(
        self,
        memory_str: str,
        parsed_fields: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        if parsed_fields is None:
            parsed_fields = self._extract_memory_fields(memory_str)
        if not parsed_fields:
            return None

        title = parsed_fields.get("title")
        if title is None:
            return None

        if isinstance(title, str):
            title = title.strip()
        return title or None

    async def _encode_text_key(self, text_key: str):
        worker_index = self._embedding_worker_index % self.embedding_cluster.world_size
        self._embedding_worker_index += 1
        embedding_worker = self.embedding_cluster.workers[worker_index]
        embedding = await embedding_worker.encode.remote(text_key)
        return embedding.squeeze(0)

    async def _apply_title_key_if_enabled(
        self,
        data: Dict[str, Any],
        memory_text: str,
        parsed_fields: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.memory_config.use_memory_title_for_key:
            return

        if not memory_text or not isinstance(memory_text, str):
            return

        # TODO: Adjust it to make it more alegent
        # title = self._extract_memory_title(memory_text, parsed_fields)
        title = memory_text
        if not title:
            return

        key_field = self.memory.memory_key_field
        existing_key = data.get(key_field)
        existing_text_key = data.get("text_key")

        if self.memory_config.searcher is not None and self.memory_config.searcher.memory_search_strategy in [
            SampleStrategy.embedding_similarity,
            SampleStrategy.faiss_embedding_similarity,
        ]:
            if existing_key is not None and not isinstance(existing_key, str) and existing_text_key == title:
                data["text_key"] = title
                return
            if (
                not hasattr(self, "embedding_cluster")
                or self.embedding_cluster is None
                or self.embedding_cluster.world_size == 0
            ):
                data[key_field] = title
                data["text_key"] = title
                return

            embedding = await self._encode_text_key(title)
            data[key_field] = embedding
            data["text_key"] = title
        else:
            if existing_key == title:
                data["text_key"] = title
                return
            data[key_field] = title
            data["text_key"] = title

    def validate_memory_format(
        self,
        memory_str: str,
        parsed_fields: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Validate that a memory string has the expected format.

        Args:
            memory_str: Memory string to validate

        Returns:
            bool: True if format is valid, False otherwise
        """
        if parsed_fields is None:
            parsed_fields = self._extract_memory_fields(memory_str)

        if parsed_fields is None:
            if not self.memory_config.memory_format_pattern:
                return True
            logger.warning("Memory entry did not match expected format pattern.")
            return False

        title = parsed_fields.get("title")
        content = parsed_fields.get("content")

        if title is not None and isinstance(title, str) and not title.strip():
            logger.warning("Memory has empty Title field")
            return False

        if content is not None and isinstance(content, str) and not content.strip():
            logger.warning("Memory has empty Content field")
            return False

        return True

    def parse_functional_operations_from_response(self, response: str) -> Optional[List[Dict[str, Any]]]:
        """
        Parse functional operations (add/upvote/downvote) from LM response.

        Expected format:
        <answer>
        [
            {"function_name": "add", "parameters": {"new_memory": "..."}},
            {"function_name": "upvote", "parameters": {"idx": [1, 2]}},
            {"function_name": "downvote", "parameters": {"idx": [3]}}
            {"function_name": "update", "parameters": {"idx": [3], "updated_memory": "..."}}
        ]
        </answer>

        Args:
            response: Raw LM response text

        Returns:
            List of operation dicts if parsing succeeds, None otherwise
        """
        # Extract content from <answer> tags
        answer_pattern = (
            self.memory_config.memory_model.memory_model_response_pattern
            if self.memory_config.memory_model is not None
            else r"<answer>(.*?)</answer>" or r"<answer>(.*?)</answer>"
        )
        answer_info = parser_answer_func(response, answer_pattern)

        if not answer_info:
            logger.warning(f"Pattern '{answer_pattern}' not found in response. Skipping functional operations.")
            logger.debug(f"Response content: {response[:500]}...")
            return None

        answer_content = answer_info["action_content"]

        # Try to parse JSON
        try:
            operations = json.loads(answer_content)

            if not isinstance(operations, list):
                if isinstance(operations, dict):
                    operations = [operations]
                else:
                    logger.warning(f"Functional operations response is not a list: {type(operations)}")
                    return None

            # Validate operation structure
            for op in operations:
                if not isinstance(op, dict):
                    logger.warning(f"Operation is not a dict: {op}")
                    return None

                if "function_name" not in op:
                    logger.warning(f"Operation missing required fields: {op}")
                    return None

                if op["function_name"] != "return" and "parameters" not in op:
                    logger.warning(f"Operation missing required parameters: {op}")
                    return None

                func_name = op["function_name"]
                if func_name != "return":
                    params = op["parameters"]

                # Validate each operation type
                if func_name == "add":
                    if "new_memory" not in params or not isinstance(params["new_memory"], str):
                        logger.warning(f"Invalid 'add' operation parameters: {params}")
                        return None
                    # Validate memory format
                    if not self.validate_memory_format(params["new_memory"]):
                        logger.warning(f"Invalid memory format in 'add' operation")
                        logger.debug(f"Memory content: {params['new_memory'][:200]}...")
                        return None

                elif func_name == "update":
                    if not isinstance(params, dict):
                        logger.warning(f"Invalid 'update' operation parameters: {params}")
                        return None

                    idx_name = "idx" if "idx" in params else ("target_idx" if "target_idx" in params else None)
                    if idx_name is None:
                        logger.warning(f"Invalid 'update' operation parameters: {params}")
                        return None

                    if "new_memory" not in params and "updated_memory" in params:
                        params["new_memory"] = params["updated_memory"]

                    if "new_memory" not in params:
                        logger.warning(f"Invalid 'update' operation parameters: {params}")
                        return None

                    if not isinstance(params[idx_name], list) or not isinstance(params["new_memory"], str):
                        logger.warning(f"'update' parameters must be list and string: {params}")
                        return None
                    if not all(isinstance(i, int) for i in params[idx_name]):
                        logger.warning(f"'update' idx must be list of integers: {params[idx_name]}")
                        return None
                    # Validate memory format
                    if not self.validate_memory_format(params["new_memory"]):
                        logger.warning(f"Invalid memory format in 'update' operation")
                        logger.debug(f"Memory content: {params['new_memory'][:200]}...")
                        return None

                elif func_name == "upvote":
                    if "idx" not in params or not isinstance(params["idx"], list):
                        logger.warning(f"Invalid 'upvote' operation parameters: {params}")
                        return None
                    if not all(isinstance(i, int) for i in params["idx"]):
                        logger.warning(f"'upvote' idx must be list of integers: {params['idx']}")
                        return None
                    if len(params["idx"]) == 0:
                        logger.warning(f"'upvote' idx list cannot be empty")
                        return None

                elif func_name == "downvote":
                    if "idx" not in params or not isinstance(params["idx"], list):
                        logger.warning(f"Invalid 'downvote' operation parameters: {params}")
                        return None
                    if not all(isinstance(i, int) for i in params["idx"]):
                        logger.warning(f"'downvote' idx must be list of integers: {params['idx']}")
                        return None
                    if len(params["idx"]) == 0:
                        logger.warning(f"'downvote' idx list cannot be empty")
                        return None
                elif func_name == "return":
                    pass
                else:
                    logger.warning(f"Unknown function name: {func_name}")
                    return None

            logger.info(f"Successfully parsed {len(operations)} functional operations")
            return operations

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON from answer content: {e}")
            logger.debug(f"Answer content: {answer_content[:500]}...")
            return None
        except Exception as e:
            logger.error(f"Unexpected error parsing functional operations: {e}")
            return None

    async def build_memory_model_messages(
        self,
        interaction_messages: List[Dict[str, Any]],
        task_goal: str,
        outcome: str,
        last_turn_uids: List[str],
        memory_system_prompt: str = None,
        memory_use_prompt: str = None,
        assistant_answer_pattern: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Build memory model messages from data.

        Args:
            interaction_messages: List of messages from the trajectory
            task_goal: The task goal
            outcome: Whether the outcome was successful (SUCCESS/FAILURE string or bool)
            assistant_answer_pattern: Optional regex pattern used to extract assistant
                responses when they are wrapped in tags (e.g. <answer>...</answer>).
        """
        if memory_system_prompt is None:
            memory_system_prompt = self.memory_config.memory_model.memory_model_system_prompt
        if memory_use_prompt is None:
            memory_use_prompt = self.memory_config.memory_model.memory_model_user_prompt

        interaction_message = ""
        for message in interaction_messages:
            if message["role"] == "system":
                role = "System"
            elif message["role"] == "user":
                role = "Environment"
            elif message["role"] == "tool":
                role = "Tool Environment"
            elif message["role"] == "assistant":
                role = "Agent"
            else:
                raise ValueError(f"Unknown role: {message['role']}")
            content = message.get("content", "")
            if message["role"] == "assistant" and assistant_answer_pattern and isinstance(content, str):
                answer_info = parser_answer_func(content, assistant_answer_pattern)
                if answer_info:
                    content = answer_info["action_content"]
            interaction_message += f"{role}: {content}\n"

        # Format outcome as SUCCESS/FAILURE
        if isinstance(outcome, bool):
            outcome_str = "SUCCESS" if outcome else "FAILURE"
        else:
            outcome_str = str(outcome)

        effective_last_turn_uids: List[str] = []
        prev_sections: List[str] = []
        missing = 0

        if last_turn_uids:
            async with self.memory_lock.reader():
                for uid in last_turn_uids:
                    memory_instance = self.memory.get_entry_by_uid(uid)
                    if memory_instance is None:
                        missing += 1
                        continue

                    memory_content = getattr(memory_instance, self.memory.memory_value_field, None)
                    effective_last_turn_uids.append(uid)
                    entry_idx = len(effective_last_turn_uids)  # 1-based, compact after filtering
                    prev_sections.append(f"**Memory Entry {entry_idx}**\n{memory_content}\n")

        if missing:
            logger.debug(
                f"build_memory_model_messages: {missing}/{len(last_turn_uids)} last_turn_uids not found (evicted/merged)"
            )

        prev_memory = "\n\n".join(prev_sections) if prev_sections else "[No previous memories]"

        memory_model_messages = [
            {
                "role": "system",
                "content": (
                    memory_system_prompt.format(
                        max_operations=self.memory_config.memory_model.memory_model_max_functional_operations
                    )
                    if self.memory_config.memory_model is not None
                    else memory_system_prompt
                ),
            },
            {
                "role": "user",
                "content": memory_use_prompt.format(
                    interaction_messages=interaction_message,
                    trajectory=interaction_message,  # Alias for functional operations prompt
                    task_goal=task_goal,
                    outcome=outcome_str,
                    prev_memory=prev_memory,
                    previous_memories=prev_memory,  # Alias for functional operations prompt
                    max_operations=(
                        self.memory_config.memory_model.memory_model_max_functional_operations
                        if self.memory_config.memory_model is not None
                        else None
                    ),
                ),
            },
        ]
        return memory_model_messages, effective_last_turn_uids

    async def _process_updates(self):
        """Background task to process update queue with resilient error handling"""
        while self.running:
            try:
                data, timestamp = await asyncio.wait_for(self.update_queue.get(), timeout=1.0)

                # Create task for processing this update
                task = asyncio.create_task(self._process_single_update_wrapper(data))
                self.active_update_tasks.add(task)
                task.add_done_callback(self.active_update_tasks.discard)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error retrieving update from queue: {e}")

    async def _process_single_update_wrapper(self, data):
        """Wrapper to handle semaphore and error catching for single update"""
        async with self.update_semaphore:
            await self._process_single_update(data)

    def _should_log_interaction(self, should_control_frequency: bool = False, operations: str = "update") -> bool:
        """
        Determine if the interaction should be logged.
        """
        if (
            self.memory_config.memory_model is None
            or self.memory_config.memory_model.memory_model_interaction_output_dir is None
        ):
            return False

        if not should_control_frequency:
            return True

        if operations == "update":
            if getattr(self, "last_update_logging", None) is None:
                self.last_update_logging = time.time()
                return True
            if time.time() - self.last_update_logging > 60:
                self.last_update_logging = time.time()
                return True
            return False
        elif operations == "merge":
            if getattr(self, "last_merge_logging", None) is None:
                self.last_merge_logging = time.time()
                return True
            if time.time() - self.last_merge_logging > 60:
                self.last_merge_logging = time.time()
                return True
            return False
        elif operations == "search":
            if getattr(self, "last_search_logging", None) is None:
                self.last_search_logging = time.time()
                return True
            if time.time() - self.last_search_logging > 60:
                self.last_search_logging = time.time()
                return True
            return False

    def _log_memory_model_interaction(
        self,
        operation: str,
        input_messages: List[Dict[str, Any]],
        output_response: str,
        additional_info: Optional[Dict[str, Any]] = None,
    ):
        """
        Log memory model interaction messages to file.

        Args:
            operation: Operation type ("update", "merge", or "search")
            input_messages: Input messages sent to the memory model
            output_response: Response text from the memory model
            additional_info: Optional additional information to include
        """
        if not self._should_log_interaction(should_control_frequency=True, operations=operation):
            return

        # Get save directory - use memory_model_interaction_output_dir or memory_model_save_dir
        save_dir = getattr(self.memory_config.memory_model, "memory_model_save_dir", None) or getattr(
            self.memory_config.memory_model, "memory_model_interaction_output_dir", None
        )

        if not save_dir:
            return

        operation_dir = os.path.join(save_dir, operation)
        os.makedirs(operation_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"interaction_{timestamp}.json"
        filepath = os.path.join(operation_dir, filename)

        interaction_data = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "input": {"messages": input_messages},
            "output": {"response": output_response},
        }

        if additional_info:
            interaction_data["additional_info"] = additional_info

        # Save to file
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(interaction_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to log memory model interaction to {filepath}: {e}")

    async def _process_single_update(self, data):
        """
        Process a single update item.
        This contains the logic previously in the _process_updates loop.
        """
        parsed_fields_for_key: Optional[Dict[str, Any]] = None
        title_applied = False

        if (
            self.memory_config.searcher.memory_search_strategy
            in [
                SampleStrategy.embedding_similarity,
                SampleStrategy.faiss_embedding_similarity,
            ]
            and not self.memory_config.use_memory_title_for_key
            and (self.memory.memory_key_field not in data or isinstance(data[self.memory.memory_key_field], str))
        ):
            text_key = data[self.memory.memory_key_field]
            embedding = await self._encode_text_key(text_key)
            data[self.memory.memory_key_field] = embedding.squeeze(0)
            data["text_key"] = text_key

        if self.memory_config.memory_model is not None:
            messages = data["messages"]
            task_goal = data.get("task_goal", "Unknown task")
            outcome = data.get("outcome", False)
            last_turn_uids = data.get("last_turn_uids", [])

            memory_model_messages, last_turn_uids = await self.build_memory_model_messages(
                messages,
                task_goal,
                outcome,
                last_turn_uids,
                assistant_answer_pattern=(
                    self.memory_config.memory_model.memory_model_build_message_pattern
                    if self.memory_config.memory_model is not None
                    else None
                ),
            )

            if self.memory.memory_uid_field not in data:
                uuid = str(uuid_module.uuid4())
                data[self.memory.memory_uid_field] = uuid
            else:
                uuid = data[self.memory.memory_uid_field]

            if self.memory_config.memory_model.memory_model_type == MemoryModelType.local_model:
                input_ids = custom_apply_chat_template(
                    messages=memory_model_messages,
                    tokenizer=self.tokenizer,
                    add_generation_prompt=True,
                )
                input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
                attention_mask = torch.tensor([1] * input_ids.shape[1], dtype=torch.long).unsqueeze(0)
                position_ids = attention_mask.cumsum(dim=-1)

                lm_input = DataProto()
                lm_input.batch = TensorDict(
                    {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "position_ids": position_ids,
                    },
                    batch_size=input_ids.shape[0],
                )
                generation_config = self.memory_config.memory_model.generating_args.to_dict()
                lm_input.meta_info["src_rank"] = uuid
                lm_input.meta_info["response_callback_fn"] = self.generate_scheduler.report_response.remote
                lm_input.meta_info["pad_to_seq_len"] = False
                lm_input.meta_info["generation_config"] = generation_config
                lm_output: DataProto = await self.generate_scheduler.generate_one_request.remote(data=lm_input)

                if lm_output is None:
                    logger.warning(f"Memory model generation returned None for UUID: {uuid}, skipping update")
                    self.performance_stats["pending_updates"] = max(
                        0,
                        self.performance_stats["pending_updates"] - 1,
                    )
                    return

                raw_responses = self.tokenizer.batch_decode(
                    lm_output.batch["responses"],
                    skip_special_tokens=True,
                )

                self._log_memory_model_interaction(
                    operation="update",
                    input_messages=memory_model_messages,
                    output_response=raw_responses[0] if raw_responses else "",
                    additional_info={"uuid": uuid, "model_type": "local_model"},
                )

                # Parse the structured memory from the response
                if self.memory_config.memory_model.memory_model_with_functional_operations:
                    operations = self.parse_functional_operations_from_response(raw_responses[0])
                    if operations is None:
                        logger.info(f"Skipping memory update due to parsing failure for UUID: {uuid}")
                        return

                    # Execute functional operations with write lock (exclusive)
                    async with self.memory_lock.writer():
                        success = await self._execute_functional_operations(
                            operations,
                            last_turn_uids,
                            data,
                            lm_output,
                        )

                    if success:
                        self.performance_stats["pending_updates"] = max(
                            0,
                            self.performance_stats["pending_updates"] - 1,
                        )
                    else:
                        logger.warning("Functional operations failed")

                    return
                else:
                    (
                        parsed_memory,
                        parsed_fields,
                    ) = self.parse_memory_from_response(raw_responses[0])
                    if parsed_memory is None:
                        # Skip this update if parsing failed
                        logger.info(f"Skipping memory update due to parsing failure for UUID: {uuid}")
                        return

                    data[self.memory.memory_value_field] = parsed_memory
                    parsed_fields_for_key = parsed_fields
                    await self._apply_title_key_if_enabled(data, parsed_memory, parsed_fields)
                    title_applied = True

            elif self.memory_config.memory_model.memory_model_type == MemoryModelType.api_model:
                lm_input = DataProto()
                lm_input.meta_info["messages"] = memory_model_messages
                generation_config = (
                    self.memory_config.memory_model.generating_args.to_dict()
                    if hasattr(
                        self.memory_config.memory_model,
                        "generating_args",
                    )
                    else {}
                )
                lm_input.meta_info["generation_config"] = generation_config

                lm_output = await asyncio.wrap_future(
                    self.memory_model_cluster.generate_with_memory_model(data=lm_input, blocking=False)[
                        0
                    ].obj_ref.future()
                )
                raw_response = lm_output.meta_info["response_text"]

                self._log_memory_model_interaction(
                    operation="update",
                    input_messages=memory_model_messages,
                    output_response=raw_response,
                    additional_info={"uuid": uuid, "model_type": "api_model"},
                )

                if self.memory_config.memory_model.memory_model_with_functional_operations:
                    operations = self.parse_functional_operations_from_response(raw_response)
                    if operations is None:
                        logger.info(f"Skipping memory update due to parsing failure for UUID: {uuid}")
                        return

                    # Execute functional operations with write lock (exclusive)
                    async with self.memory_lock.writer():
                        success = await self._execute_functional_operations(
                            operations, last_turn_uids, data, lm_output
                        )

                    if success:
                        self.performance_stats["pending_updates"] = max(
                            0,
                            self.performance_stats["pending_updates"] - 1,
                        )
                    else:
                        logger.warning("Functional operations failed")

                    # Skip the normal update flow for functional operations
                    return
                else:
                    (
                        parsed_memory,
                        parsed_fields,
                    ) = self.parse_memory_from_response(raw_response)
                    if parsed_memory is None:
                        # Skip this update if parsing failed
                        logger.info(f"Skipping memory update due to parsing failure for UUID: {uuid}")
                        return

                    data[self.memory.memory_value_field] = parsed_memory

        if (
            self.memory_config.use_memory_title_for_key
            and not title_applied
            and self.memory.memory_value_field in data
            and isinstance(data[self.memory.memory_value_field], str)
        ):
            await self._apply_title_key_if_enabled(
                data,
                data[self.memory.memory_value_field],
                parsed_fields_for_key,
            )

        # Process update with write lock (exclusive access)
        async with self.memory_lock.writer():
            success = await self._execute_update(data)

        if success:
            self.performance_stats["pending_updates"] = max(0, self.performance_stats["pending_updates"] - 1)
        else:
            logger.warning("Update operation failed")

    async def _execute_update(self, data: Any) -> bool:
        """
        Execute actual update operation on memory.

        Uses updater worker 0 (typically world_size=1 for thread-safety).
        Holds memory lock during entire update to prevent searches from
        accessing stale or inconsistent data.
        """
        updater_worker = self.updater_cluster.workers[0]

        success, new_memory = await updater_worker.update.remote(data, self.memory)
        if success:
            self.memory = new_memory

        if not success:
            logger.warning("Failed to update memory via updater worker")
            return False

        return True

    async def _execute_functional_operations(
        self,
        operations: List[Dict[str, Any]],
        last_turn_uids: List[str],
        data: Any,
        lm_output: DataProto,
    ) -> bool:
        """
        Execute functional memory operations (add/upvote/downvote).

        Args:
            operations: List of parsed operations from LLM
            last_turn_uids: UIDs from the previous turn's memory retrieval (for mapping idx)
            data: Original data dict (used for metadata when adding new memories)
            lm_output: LLM output data proto

        Returns:
            bool: True if all operations succeeded, False otherwise
        """
        try:
            # Track operations for logging
            add_count = 0
            upvote_count = 0
            downvote_count = 0
            update_count = 0

            for op in operations:
                func_name = op["function_name"]
                params = op["parameters"] if "parameters" in op else None

                if func_name == "return":
                    logger.info("Returning for return func call called")
                    return True

                elif func_name == "upvote":
                    # Map idx to UIDs and upvote
                    indices = params["idx"]
                    uids_to_upvote = []

                    for idx in indices:
                        # idx is 1-based in the prompt (Memory Entry 1, 2, 3...)
                        # Convert to 0-based for list access
                        list_idx = idx - 1

                        if 0 <= list_idx < len(last_turn_uids):
                            uids_to_upvote.append(last_turn_uids[list_idx])
                        else:
                            logger.warning(f"Upvote idx {idx} out of range (max: {len(last_turn_uids)})")

                    for uid in uids_to_upvote:
                        # Check if UID still exists
                        if uid not in self.memory.entries:
                            logger.info(
                                f"UID {uid[:8]}... already evicted/deleted, skipping upvote (expected in concurrent system)"
                            )
                            continue

                        # Get the entry and update vote count in metadata
                        entry = self.memory.entries[uid]
                        if not hasattr(entry, "metadata") or entry.metadata is None:
                            entry.metadata = {}

                        # Increment upvote count
                        entry.metadata["votes_count"] = entry.metadata.get("votes_count", 0) + 1
                        upvote_count += 1
                        logger.info(
                            f"Upvoted memory entry: {uid[:8]}... (votes_count: {entry.metadata['votes_count']})"
                        )

                elif func_name == "downvote":
                    # Map idx to UIDs and downvote
                    indices = params["idx"]
                    uids_to_downvote = []

                    for idx in indices:
                        # idx is 1-based in the prompt (Memory Entry 1, 2, 3...)
                        # Convert to 0-based for list access
                        list_idx = idx - 1

                        if 0 <= list_idx < len(last_turn_uids):
                            uids_to_downvote.append(last_turn_uids[list_idx])
                        else:
                            logger.warning(f"Downvote idx {idx} out of range (max: {len(last_turn_uids)})")

                    for uid in uids_to_downvote:
                        if uid not in self.memory.entries:
                            logger.info(
                                f"UID {uid[:8]}... already evicted/deleted, skipping downvote (expected in concurrent system)"
                            )
                            continue

                        # Get the entry and update vote count in metadata
                        entry = self.memory.entries[uid]
                        if not hasattr(entry, "metadata") or entry.metadata is None:
                            entry.metadata = {}

                        # Decrement vote count
                        entry.metadata["votes_count"] = entry.metadata.get("votes_count", 0) - 1
                        votes_count = entry.metadata["votes_count"]

                        if votes_count < 0:
                            self.memory.delete_entry(uid)
                            if self._uid_usage is not None and uid in self._uid_usage:
                                del self._uid_usage[uid]
                            logger.info(f"Deleted memory entry: {uid[:8]}... (votes_count: {votes_count})")
                        else:
                            downvote_count += 1
                            logger.info(f"Downvoted memory entry: {uid[:8]}... (votes_count: {votes_count})")

                elif func_name == "add":
                    # Add new memory entry
                    new_memory = params["new_memory"]

                    # Create a new data dict, preserving metadata from original data
                    new_data = {"triggered_interaction": lm_output, "votes_count": 0}
                    excluded_fields = {
                        self.memory.memory_uid_field,
                        self.memory.memory_value_field,
                        "messages",
                        "last_turn_uids",
                    }
                    if self.memory_config.use_memory_title_for_key:
                        excluded_fields.add(self.memory.memory_key_field)

                    for key, value in data.items():
                        if key not in excluded_fields:
                            new_data[key] = value

                    normalized_memory, parsed_fields = self._normalize_memory_text(new_memory)
                    new_data[self.memory.memory_value_field] = normalized_memory
                    new_uuid = str(uuid_module.uuid4())
                    new_data[self.memory.memory_uid_field] = new_uuid

                    await self._apply_title_key_if_enabled(new_data, normalized_memory, parsed_fields)

                    updater_worker = self.updater_cluster.workers[0]
                    success, new_memory_obj = await updater_worker.update.remote(new_data, self.memory)
                    if success:
                        self.memory = new_memory_obj
                        add_count += 1
                        logger.info(f"Added new memory entry: {new_uuid[:8]}...")
                    else:
                        logger.warning("Failed to add new memory entry")

                elif func_name == "update":
                    # Update = delete old entries + add new merged entry
                    idx_name = "idx" if "idx" in params else "target_idx"
                    indices = params[idx_name]
                    new_memory = params["new_memory"]

                    # First, delete the old entries
                    uids_to_delete = []
                    for idx in indices:
                        list_idx = idx - 1
                        if 0 <= list_idx < len(last_turn_uids):
                            uids_to_delete.append(last_turn_uids[list_idx])
                        else:
                            logger.warning(f"Update idx {idx} out of range (max: {len(last_turn_uids)})")

                    for uid in uids_to_delete:
                        if uid not in self.memory.entries:
                            logger.info(
                                f"UID {uid[:8]}... already evicted/deleted during update, skipping (expected in concurrent system)"
                            )
                            continue

                        success = self.memory.delete_entry(uid)
                        if success:
                            if self._uid_usage is not None and uid in self._uid_usage:
                                del self._uid_usage[uid]
                            logger.info(f"Deleted memory entry for update: {uid[:8]}...")

                    # Now add the new merged memory
                    # Create a new data dict, preserving metadata from original data
                    new_data = {"triggered_interaction": lm_output, "votes_count": 0}
                    excluded_fields = {
                        self.memory.memory_uid_field,
                        self.memory.memory_value_field,
                        "messages",
                        "last_turn_uids",
                    }
                    if self.memory_config.use_memory_title_for_key:
                        excluded_fields.add(self.memory.memory_key_field)

                    # Copy non-memory fields from original data (like timestamps, metadata)
                    for key, value in data.items():
                        if key not in excluded_fields:
                            new_data[key] = value

                    normalized_memory, parsed_fields = self._normalize_memory_text(new_memory)
                    new_data[self.memory.memory_value_field] = normalized_memory
                    # new uuid since may be multiple updates, but they share the same key field since all belong to same query
                    new_uuid = str(uuid_module.uuid4())
                    new_data[self.memory.memory_uid_field] = new_uuid

                    await self._apply_title_key_if_enabled(new_data, normalized_memory, parsed_fields)

                    updater_worker = self.updater_cluster.workers[0]
                    success, new_memory = await updater_worker.update.remote(new_data, self.memory)
                    if success:
                        self.memory = new_memory
                        update_count += 1
                        logger.info(f"Updated memory (merged {len(uids_to_delete)} entries into 1)")
                    else:
                        logger.warning("Failed to add merged memory after delete")

            logger.info(
                f"Functional operations completed: {add_count} adds, {upvote_count} upvotes, {downvote_count} downvotes, {update_count} updates"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to execute functional operations: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return False

    async def flush_pending_updates(self, timeout: Optional[float] = 30.0) -> int:
        """
        Wait for all pending updates in queue to be processed.

        This is critical to call before suspend/resume cycles to ensure
        memory updates from completed trajectories are not lost.

        Args:
            timeout: Maximum time to wait in seconds. If None, wait indefinitely.

        Returns:
            Number of updates successfully processed
        """
        start_time = time.time()
        initial_pending = self.update_queue.qsize() + len(self.active_update_tasks)

        if initial_pending == 0:
            logger.info("No pending updates to flush")
            return 0

        logger.info(f"Flushing {initial_pending} pending updates...")

        while self.update_queue.qsize() > 0 or len(self.active_update_tasks) > 0:
            elapsed = time.time() - start_time
            if timeout is not None and elapsed > timeout:
                remaining = self.update_queue.qsize() + len(self.active_update_tasks)
                logger.warning(
                    f"Timeout ({timeout}s) waiting for updates to complete. "
                    f"{remaining}/{initial_pending} still pending"
                )
                break

            # Log progress every 5 seconds
            if int(elapsed) % 5 == 0 and int(elapsed) > 0:
                remaining = self.update_queue.qsize() + len(self.active_update_tasks)
                logger.info(
                    f"Flush progress: {initial_pending - remaining}/{initial_pending} completed "
                    f"({elapsed:.1f}s elapsed)"
                )

            await asyncio.sleep(0.1)

        current_pending = self.update_queue.qsize() + len(self.active_update_tasks)
        processed = initial_pending - current_pending
        logger.info(f"Flushed {processed}/{initial_pending} pending updates in {time.time() - start_time:.2f}s")
        return processed

    async def get_memory_size(self) -> int:
        async with self.memory_lock.reader():
            return len(self.memory)

    async def get_memory_fields(self) -> Dict[str, str]:
        return {
            "uid_field": self.memory.memory_uid_field,
            "key_field": self.memory.memory_key_field,
            "value_field": self.memory.memory_value_field,
        }

    async def get_entry_by_uid(self, uid: str) -> Optional[Dict[str, Any]]:
        """
        Get memory entry by UID.

        Args:
            uid: Memory entry UID

        Returns:
            Dict containing entry data (uid, key, value, metadata) if found, None otherwise
        """
        async with self.memory_lock.reader():
            entry = self.memory.get_entry_by_uid(uid)
            if entry is None:
                return None

            result = {
                "uid": getattr(entry, self.memory.memory_uid_field, uid),
                "key": getattr(entry, self.memory.memory_key_field, None),
                "value": getattr(entry, self.memory.memory_value_field, None),
            }

            if hasattr(entry, "metadata") and entry.metadata:
                result["metadata"] = entry.metadata

            return result

    async def get_performance_stats(self) -> Dict:
        searcher_stats_results = []
        if self.searcher_cluster.workers:
            searcher_stats_refs = [worker.get_stats.remote() for worker in self.searcher_cluster.workers]
            searcher_stats_results = await asyncio.gather(*searcher_stats_refs, return_exceptions=True)

        combined_searcher_stats = {}
        for result in searcher_stats_results:
            if isinstance(result, Exception):
                logger.warning(f"Failed to get searcher stats: {result}")
                continue
            if result and isinstance(result, dict) and "meta_info" in result and "stats" in result["meta_info"]:
                combined_searcher_stats.update(result["meta_info"]["stats"])

        updater_stats_results = []
        if self.memory_config.updater is not None and self.updater_cluster.workers:
            updater_stats_refs = [worker.get_stats.remote() for worker in self.updater_cluster.workers]
            updater_stats_results = await asyncio.gather(*updater_stats_refs, return_exceptions=True)

        combined_updater_stats = {}
        for result in updater_stats_results:
            if isinstance(result, Exception):
                logger.warning(f"Failed to get updater stats: {result}")
                continue
            if result and isinstance(result, dict) and "meta_info" in result and "stats" in result["meta_info"]:
                combined_updater_stats.update(result["meta_info"]["stats"])

        # Calculate cache statistics
        total_cache_requests = self.query_cache_hits + self.query_cache_misses
        cache_hit_rate = self.query_cache_hits / total_cache_requests if total_cache_requests > 0 else 0.0

        return {
            "memory_manager": self.performance_stats.copy(),
            "batch_processing": self.batch_stats.copy(),
            "query_cache": {
                "enabled": self.query_cache_enabled,
                "size": len(self.query_cache),
                "max_size": self.query_cache_size,
                "hits": self.query_cache_hits,
                "misses": self.query_cache_misses,
                "hit_rate": cache_hit_rate,
            },
            "searcher": combined_searcher_stats,
            "updater": combined_updater_stats,
            "memory_size": await self.get_memory_size(),
            "queue_size": self.update_queue.qsize(),
            "pending_queries": len(self.pending_queries),
        }

    async def save_state(self, save_path: Optional[str] = None) -> bool:
        """Save memory state and searcher state"""
        try:
            async with self.memory_lock.reader():
                if self.memory_config.memory_should_save and self.memory_config.memory_save_path:
                    self.memory.save_memory(save_path)

            if self.searcher_cluster.workers:
                save_refs = [worker.save_state.remote(save_path) for worker in self.searcher_cluster.workers]
                save_results = await asyncio.gather(*save_refs, return_exceptions=True)

            if self._uid_usage is not None:
                usage_save_path = Path(save_path)
                if usage_save_path.is_dir():
                    usage_save_path = usage_save_path / "uid_usage.json"
                else:
                    usage_save_path = usage_save_path.parent / "uid_usage.json"
                with open(usage_save_path, "w", encoding="utf-8") as f:
                    json.dump(self._uid_usage, f, indent=2, ensure_ascii=False)
                logger.info(f"UID usage saved to {usage_save_path}")

            logger.info("Memory state saved successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to save memory state: {e}")
            return False

    def _cleanup_logging_handlers(self):
        """
        Properly flush and close all logging handlers to prevent I/O errors during shutdown.

        This prevents OSError: [Errno 5] Input/output error that can occur when Ray
        terminates the actor process while logging handlers are still trying to write.
        """
        try:
            if hasattr(self, "_log_file_handler") and self._log_file_handler:
                try:
                    self._log_file_handler.flush()
                    self._log_file_handler.close()
                    logger.removeHandler(self._log_file_handler)
                except Exception as e:
                    print(f"Warning: Error closing file handler: {e}")

            # Clean up all handlers from logger and root logger
            for handler in logger.handlers[:]:
                try:
                    handler.flush()
                    handler.close()
                    logger.removeHandler(handler)
                except Exception:
                    pass

            root_logger = logging.getLogger()
            for handler in root_logger.handlers[:]:
                try:
                    handler.flush()
                    handler.close()
                    root_logger.removeHandler(handler)
                except Exception:
                    pass

        except Exception as e:
            print(f"Warning: Error during logging cleanup: {e}")

    async def shutdown(self):
        self.running = False
        self.running_searches = False  # Also stop searches on shutdown

        # Stop batch processor
        if self.batch_processor_task:
            self.batch_event.set()  # Wake up the task
            try:
                await asyncio.wait_for(self.batch_processor_task, timeout=1.0)
            except asyncio.TimeoutError:
                self.batch_processor_task.cancel()
                try:
                    await self.batch_processor_task
                except asyncio.CancelledError:
                    pass

        # Stop update processor
        if self.update_processor_task:
            await self.update_processor_task

        if self.memory_config.memory_save_path is not None:
            save_path = os.path.join(self.memory_config.memory_save_path, "memory_final.json")
            await self.save_state(save_path)
            logger.info(f"Final memory state saved to {save_path}")

        if (
            self.memory_config.memory_model is not None
            and self.memory_config.memory_model.memory_model_type == MemoryModelType.local_model
        ):
            stop_server_tasks = [
                asyncio.wrap_future(ref.obj_ref.future())
                for ref in self.memory_model_cluster.stop_server(blocking=False)
            ]
            gen_metrics = await asyncio.gather(*stop_server_tasks)
            gen_metrics = gen_metrics[0]
            logger.info(f"Memory local model stop server metrics: {gen_metrics}")

        await self.save_state()

        logger.info("AsyncMemoryManager shutdown complete")

        self._cleanup_logging_handlers()

    async def suspend(self, global_step: int):
        if (
            self.memory_config.memory_model.memory_model_type != MemoryModelType.local_model
            or self.memory_model_cluster is None
        ):
            return {}

        if not self.running:
            return {}

        await self.generate_scheduler.suspend.remote()
        return await self._stop_update()

    async def resume(self, global_step: int):
        if (
            self.memory_config.memory_model.memory_model_type != MemoryModelType.local_model
            or self.memory_model_cluster is None
        ):
            return {}

        if self.running:
            return

        await self._start_update(global_step)

        await self.generate_scheduler.resume.remote()

    async def _stop_update(self):
        if not self.running:
            return

        self.running = False  # Stop update processing
        # Note: self.running_searches remains True - searches continue during suspend

        if self.memory_model_cluster is not None:
            stop_server_tasks = [
                asyncio.wrap_future(ref.obj_ref.future())
                for ref in self.memory_model_cluster.stop_server(blocking=False)
            ]

            gen_metrics = await asyncio.gather(*stop_server_tasks)
            gen_metrics = gen_metrics[0]
            if gen_metrics is not None:
                return gen_metrics.meta_info.pop("metrics", {})
            else:
                return {}

        return {}

    async def _start_update(self, global_step: int):
        if self.running:
            return

        self.running = True

        # The previous task has exited when self.running was False
        if self.update_processor_task is not None:
            # Cancel the old task if it's still somehow running
            if not self.update_processor_task.done():
                self.update_processor_task.cancel()
                try:
                    await self.update_processor_task
                except asyncio.CancelledError:
                    pass

        # Create a new background task
        self.update_processor_task = asyncio.create_task(self._process_updates())
        logger.info("Background update processor task restarted")

        # Note: Batch processor should continue running during suspend
        # (searches are read-only and don't need to be stopped)
        # Only restart if it somehow stopped (shouldn't happen normally)
        if self.batch_processor_task is None or self.batch_processor_task.done():
            # Only restart if it's not running
            self.running_searches = True
            self.batch_processor_task = asyncio.create_task(self._batch_processor_loop())
            logger.info("Background batch processor task restarted")

        data = DataProto()
        data.meta_info["global_step"] = global_step
        if self.memory_model_cluster is not None:
            await asyncio.gather(
                *[
                    asyncio.wrap_future(ref.obj_ref.future())
                    for ref in self.memory_model_cluster.start_server(data, blocking=False)
                ],
            )

    async def notify_embedding_index_update(self):
        """
        Notify searcher workers that the embedding-backed search index has changed.
        """

        if self.memory_config.searcher is None or self.memory_config.searcher.memory_search_strategy not in [
            SampleStrategy.embedding_similarity,
            SampleStrategy.faiss_embedding_similarity,
        ]:
            return

        if (
            not hasattr(self, "searcher_cluster")
            or self.searcher_cluster is None
            or not getattr(self.searcher_cluster, "workers", None)
        ):
            return

        notify_refs = self.searcher_cluster.notify_update(blocking=False)

        if not notify_refs:
            return

        if not isinstance(notify_refs, list):
            notify_refs = [notify_refs]

        awaitables = []
        for ref in notify_refs:
            if ref is None:
                continue
            if hasattr(ref, "obj_ref"):
                awaitables.append(asyncio.wrap_future(ref.obj_ref.future()))
            else:
                awaitables.append(asyncio.wrap_future(ref.future()))

        if not awaitables:
            return

        results = await asyncio.gather(*awaitables, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Searcher worker notify_update failed: {result}")

    async def merge_memory(self):
        """
        Merge memories for all tasks in MultiTaskTabulerMemory using the memory model.
        """
        if self.memory_config.memory_model is None:
            logger.warning("merge_memory requires memory_model configuration")
            return

        if not self.memory_config.memory_model.enable_merging:
            return

        is_multi_task = isinstance(self.memory, MultiTaskTabulerMemory)
        if not isinstance(self.memory, TabulerMemory):
            logger.warning("merge_memory is only supported for TabulerMemory/MultiTaskTabulerMemory")
            return

        # Snapshot scopes to merge (acquire read lock)
        if self.memory_config.memory_model.merging_interval is None:
            if self.memory_config.memory_model.merging_size is not None:
                if is_multi_task:
                    task_max_size = self.memory_config.memory_model.merging_size
                    global_max_size = None
                else:
                    task_max_size = None
                    global_max_size = resolve_tabular_merging_threshold(self.memory_config.memory_model.merging_size)
                    if global_max_size is None:
                        logger.warning(
                            "merge_memory for TabulerMemory requires a global merging_size key "
                            "('__global__'/'_global_'/'global'/'shared'/'all') or a single-entry dict"
                        )
                        return
            else:
                logger.warning("merge_memory requires merging_size or merging_interval configuration")
                return
        else:
            task_max_size = None
            global_max_size = None

        tasks_to_process = []
        async with self.memory_lock.reader():
            if is_multi_task:
                tasks_to_process = build_multitask_merge_targets(self.memory.task_to_uid, task_max_size)
            else:
                tasks_to_process = build_tabular_merge_targets(
                    self.memory.get_all_uids(),
                    threshold=global_max_size,
                    scope_id=GLOBAL_MERGE_SCOPE_ID,
                )

        if not tasks_to_process:
            return

        scope_name = "tasks" if is_multi_task else "shared scopes"
        logger.info(f"Starting memory merge for {len(tasks_to_process)} {scope_name}")
        print("[DEBUG] size before memory merge: ", self.memory.memory_size)

        concurrency_limit = 16
        if hasattr(self, "memory_model_cluster") and self.memory_model_cluster is not None:
            concurrency_limit = max(concurrency_limit, self.memory_model_cluster.world_size * 4)

        semaphore = asyncio.Semaphore(concurrency_limit)

        async def process_task(task_id, uids):
            async with semaphore:
                try:
                    return await self._merge_single_task(task_id, uids)
                except Exception as e:
                    logger.error(f"Failed to merge memory for task {task_id}: {e}")
                    import traceback

                    logger.error(traceback.format_exc())
                    return None

        # Run all merge tasks
        results = await asyncio.gather(*[process_task(t, u) for t, u in tasks_to_process])

        # Apply updates (acquire write lock)
        successful_merges = 0
        actually_merged = 0
        async with self.memory_lock.writer():
            for result in results:
                if result:
                    scope_id, uids_to_delete, new_entries_data_list, was_actually_merged = result

                    # Verify UIDs to delete still exist (optimistic locking check)
                    # We only delete entries that were actually replaced by merged outputs.
                    all_exist = all(uid in self.memory.entries for uid in uids_to_delete)
                    if not all_exist:
                        continue

                    # Delete only replaced entries
                    deleted_count = 0
                    for uid in uids_to_delete:
                        if self.memory.delete_entry(uid):
                            deleted_count += 1
                            if self._uid_usage is not None and uid in self._uid_usage:
                                del self._uid_usage[uid]

                    # Add new entries
                    added_count = 0
                    for new_entry_data in new_entries_data_list:
                        if is_multi_task:
                            new_entry_data["task"] = scope_id
                        new_uid = self.memory.add_entry(new_entry_data)
                        if new_uid:
                            added_count += 1

                    successful_merges += 1
                    if was_actually_merged:
                        actually_merged += 1

                    scope_label = f"Task {scope_id}" if is_multi_task else "Shared memory"
                    logger.info(
                        f"{scope_label}: deleted {deleted_count} entries, added {added_count} entries "
                        f"(merged: {was_actually_merged})"
                    )

        print("[DEBUG] size after memory merge: ", self.memory.memory_size)
        logger.info(
            f"Memory merge completed: {successful_merges}/{len(tasks_to_process)} scopes processed, "
            f"{actually_merged} actually merged (LLM combined items)"
        )

    async def _merge_single_task(
        self, task_id: str, uids: List[str]
    ) -> Optional[Tuple[str, List[str], List[Dict[str, Any]], bool]]:
        """
        Merge entries for a single task sequentially.
        Returns: (task_id, uids_to_delete, list_of_new_entry_data, was_actually_merged)
        """
        # 1. Get memory contents
        memories = []
        async with self.memory_lock.reader():  # Short read lock
            for uid in uids:
                entry = self.memory.get_entry_by_uid(uid)
                if entry:
                    memories.append(
                        {
                            "uid": uid,
                            "content": getattr(entry, self.memory.memory_value_field),
                        }
                    )

        if len(memories) < 2:
            return None

        # 2. Iterative Merging
        max_items = self.memory_config.memory_model.max_merging_item_per_call

        # Create initial chunks. Keep UID+content so "no-op" / parse-fail paths
        # preserve original entries instead of recreating them with new UIDs.
        chunk_wise_memory = []
        for i in range(0, len(memories), max_items):
            chunk_wise_memory.append(memories[i : i + max_items])

        last_triggered_interaction = None
        final_merged_items = []
        was_actually_merged = False  # Track if LLM actually merged anything
        uids_to_delete_set = set()

        for chunk_idx, chunk_memory in enumerate(chunk_wise_memory):
            # Call LLM to merge current chunk (model only sees text)
            chunk_contents = [item["content"] for item in chunk_memory]
            response_text, lm_output = await self._call_merge_llm(chunk_contents)

            if not response_text:
                logger.warning(f"Merge failed (empty response) for task {task_id}, chunk {chunk_idx}")
                final_merged_items.extend(chunk_memory)
                continue

            merged_items_list = None
            print("[DEBUG] response_text: ", response_text)
            merged_items_list = parse_answer_to_list(response_text)
            print("[DEBUG] merged_items_list: ", merged_items_list)

            if merged_items_list is None:
                logger.info(
                    f"Parsing failed for task {task_id}, chunk {chunk_idx}. "
                    f"Keeping {len(chunk_memory)} original items, processing next chunk independently."
                )
                final_merged_items.extend(chunk_memory)
                continue

            if isinstance(merged_items_list, list) and len(merged_items_list) == 0:
                logger.info(
                    f"Merge returned no-op for task {task_id}, chunk {chunk_idx}. "
                    f"Keeping {len(chunk_memory)} original items."
                )
                final_merged_items.extend(chunk_memory)
                continue

            # Validate merged outputs early (per-chunk) so we never delete originals unless we have
            # at least one valid replacement to add.
            if self.memory_config.memory_format_pattern:
                valid_merged_items_list = []
                for item_content in merged_items_list:
                    if self.validate_memory_format(item_content):
                        valid_merged_items_list.append(item_content)
                    else:
                        logger.warning(
                            f"Merged memory entry does not match format pattern; skipping: {item_content[:100]}..."
                        )

                if not valid_merged_items_list:
                    logger.warning(
                        f"All merged outputs invalid for task {task_id}, chunk {chunk_idx}; keeping originals."
                    )
                    final_merged_items.extend(chunk_memory)
                    continue

                merged_items_list = valid_merged_items_list

            was_actually_merged = True
            last_triggered_interaction = lm_output

            # This chunk's UID-backed entries are replaced by merged outputs.
            for item in chunk_memory:
                uid = item.get("uid")
                if uid:
                    uids_to_delete_set.add(uid)

            merged_items = [{"uid": None, "content": c} for c in merged_items_list]

            # Pass result to next chunk if exists
            if chunk_idx < len(chunk_wise_memory) - 1:
                chunk_wise_memory[chunk_idx + 1].extend(merged_items)
            else:
                # This is the last chunk, so this is the final result
                final_merged_items.extend(merged_items)

        if not final_merged_items:
            logger.warning(f"No merged items for task {task_id}")
            return None

        # 3. Prepare new entry data for each item in final list
        # Only create NEW entries for merged outputs (uid is None).
        new_item_contents = [it["content"] for it in final_merged_items if not it.get("uid")]

        # If nothing was replaced/created, skip apply
        if not new_item_contents and not uids_to_delete_set:
            return None

        new_entries_data_list = []

        for item_content in new_item_contents:
            new_entry_data = {
                self.memory.memory_value_field: item_content,
                "triggered_interaction": last_triggered_interaction,
            }
            if isinstance(self.memory, MultiTaskTabulerMemory):
                new_entry_data["task"] = task_id

            # Compute embedding (outside write lock)
            if self.memory_config.searcher and self.memory_config.searcher.memory_search_strategy in [
                SampleStrategy.embedding_similarity,
                SampleStrategy.faiss_embedding_similarity,
            ]:
                # Use title/key logic
                key_text = item_content
                if self.memory_config.use_memory_title_for_key:
                    # TODO: Temporary skip title for query
                    pass
                    # parsed_fields = self._extract_memory_fields(item_content)
                    # extracted_title = self._extract_memory_title(item_content, parsed_fields)
                    # if extracted_title:
                    # key_text = extracted_title

                # Encode
                embedding = await self._encode_text_key(key_text)
                new_entry_data[self.memory.memory_key_field] = embedding
                new_entry_data["text_key"] = key_text
            else:
                # Non-embedding strategy
                key_text = item_content
                if self.memory_config.use_memory_title_for_key:
                    pass
                    # parsed_fields = self._extract_memory_fields(item_content)
                    # extracted_title = self._extract_memory_title(item_content, parsed_fields)
                    # if extracted_title:
                    # key_text = extracted_title

                new_entry_data[self.memory.memory_key_field] = key_text
                new_entry_data["text_key"] = key_text

            new_entries_data_list.append(new_entry_data)

        if not new_entries_data_list:
            return None

        return task_id, list(uids_to_delete_set), new_entries_data_list, was_actually_merged

    async def _call_merge_llm(self, items: List[str]) -> Tuple[Optional[str], Optional[DataProto]]:
        """Call LLM to merge a list of memory items."""
        system_prompt = (
            self.memory_config.memory_model.merging_system_prompt
            or "You are a helpful assistant that merges memory entries."
        )
        user_prompt_template = (
            self.memory_config.memory_model.merging_user_prompt
            or "Please merge the following memory entries into a single coherent entry:\n{entries}\n\nMerged Entry:"
        )

        # Format entries with clear numbering and separation
        formatted_entries = "\n\n".join([f"--- Entry {i+1} ---\n{item}" for i, item in enumerate(items)])
        user_prompt = user_prompt_template.format(entries=formatted_entries)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        if self.memory_config.memory_model.memory_model_type == MemoryModelType.local_model:
            input_ids = custom_apply_chat_template(
                messages=messages,
                tokenizer=self.tokenizer,
                add_generation_prompt=True,
            )
            input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
            attention_mask = torch.tensor([1] * input_ids.shape[1], dtype=torch.long).unsqueeze(0)
            position_ids = attention_mask.cumsum(dim=-1)

            lm_input = DataProto()
            lm_input.batch = TensorDict(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                },
                batch_size=input_ids.shape[0],
            )
            # Use a temp uuid for callback routing
            req_uuid = str(uuid_module.uuid4())
            generation_config = self.memory_config.memory_model.generating_args.to_dict()
            lm_input.meta_info["src_rank"] = req_uuid
            lm_input.meta_info["response_callback_fn"] = self.generate_scheduler.report_response.remote
            lm_input.meta_info["pad_to_seq_len"] = False
            lm_input.meta_info["generation_config"] = generation_config

            lm_output: DataProto = await self.generate_scheduler.generate_one_request.remote(data=lm_input)

            if lm_output is None:
                return None, None

            raw_responses = self.tokenizer.batch_decode(
                lm_output.batch["responses"],
                skip_special_tokens=True,
            )
            response_text = raw_responses[0].strip() if raw_responses else ""

            self._log_memory_model_interaction(
                operation="merge",
                input_messages=messages,
                output_response=response_text,
                additional_info={"model_type": "local_model", "num_items": len(items)},
            )

            return response_text, lm_output

        elif self.memory_config.memory_model.memory_model_type == MemoryModelType.api_model:
            lm_input = DataProto()
            lm_input.meta_info["messages"] = messages
            generation_config = (
                self.memory_config.memory_model.generating_args.to_dict()
                if hasattr(self.memory_config.memory_model, "generating_args")
                else {}
            )
            lm_input.meta_info["generation_config"] = generation_config

            lm_output = await asyncio.wrap_future(
                self.memory_model_cluster.generate_with_memory_model(data=lm_input, blocking=False)[0].obj_ref.future()
            )
            response_text = lm_output.meta_info["response_text"].strip()

            # Log memory model interaction
            self._log_memory_model_interaction(
                operation="merge",
                input_messages=messages,
                output_response=response_text,
                additional_info={"model_type": "api_model", "num_items": len(items)},
            )

            return response_text, lm_output

    async def search_and_ask(
        self, query: str, return_interaction: bool = False, task_id: Optional[str] = None
    ) -> Tuple[str, List[str], List[DataProto]]:
        """
        Search memory and ask memory model for an answer.
        """
        # 1. Search memory
        search_message, search_uids, search_triggered = await self.search(query, task_id)

        if not search_uids:
            return "No relevant knowledge found.", []

        if self.memory_config.memory_model is None:
            return search_message, search_triggered

        # 2. Build messages
        system_prompt = self.memory_config.memory_model.tool_use_system_prompt
        if not system_prompt:
            system_prompt = (
                "You are a helpful assistant. Analysis the query and the search results, and output a more useful knowledge that could solve the problem or answer the query."
                "Format the think process and the output in the following format: <think>...</think> <answer>...</answer>"
            )

        user_prompt_tmpl = self.memory_config.memory_model.tool_use_user_prompt
        if not user_prompt_tmpl:
            # Fallback template
            user_prompt_tmpl = (
                "Query: {query}\n\nResults from knowledge base:\n{memories}\n\nPlease answer the query."
                "Format the think process and the output in the following format: <think>...</think> <answer>...</answer>"
            )

        user_content = user_prompt_tmpl.format(
            query=query,
            knowledge_base=search_message,
        )

        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]

        # 3. Call model
        response_text = ""
        raw_response = None

        if self.memory_config.memory_model.memory_model_type == MemoryModelType.local_model:
            input_ids = custom_apply_chat_template(
                messages=messages,
                tokenizer=self.tokenizer,
                add_generation_prompt=True,
            )
            input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
            attention_mask = torch.tensor([1] * input_ids.shape[1], dtype=torch.long).unsqueeze(0)
            position_ids = attention_mask.cumsum(dim=-1)

            lm_input = DataProto()
            lm_input.batch = TensorDict(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                },
                batch_size=input_ids.shape[0],
            )

            req_uuid = str(uuid_module.uuid4())
            generation_config = self.memory_config.memory_model.generating_args.to_dict()
            lm_input.meta_info["src_rank"] = req_uuid
            lm_input.meta_info["response_callback_fn"] = self.generate_scheduler.report_response.remote
            lm_input.meta_info["pad_to_seq_len"] = False
            lm_input.meta_info["generation_config"] = generation_config

            lm_output: DataProto = await self.generate_scheduler.generate_one_request.remote(data=lm_input)

            if lm_output is None:
                return "Failed to get response from local memory model", [], []

            raw_responses = self.tokenizer.batch_decode(
                lm_output.batch["responses"],
                skip_special_tokens=True,
            )
            response_text = raw_responses[0].strip()
            original_response = response_text

            self._log_memory_model_interaction(
                operation="search",
                input_messages=messages,
                output_response=original_response,
                additional_info={"model_type": "local_model", "query": query},
            )

            if self.memory_config.memory_model.tool_use_parsing_pattern:
                answer_info = parser_answer_func(
                    response_text, self.memory_config.memory_model.tool_use_parsing_pattern
                )
                if answer_info:
                    response_text = answer_info["action_content"]
            raw_response = lm_output

        elif self.memory_config.memory_model.memory_model_type == MemoryModelType.api_model:
            lm_input = DataProto()
            lm_input.meta_info["messages"] = messages
            generation_config = (
                self.memory_config.memory_model.generating_args.to_dict()
                if hasattr(self.memory_config.memory_model, "generating_args")
                else {}
            )
            lm_input.meta_info["generation_config"] = generation_config

            lm_output = await asyncio.wrap_future(
                self.memory_model_cluster.generate_with_memory_model(data=lm_input, blocking=False)[0].obj_ref.future()
            )
            response_text = lm_output.meta_info["response_text"].strip()
            original_response = response_text

            self._log_memory_model_interaction(
                operation="search",
                input_messages=messages,
                output_response=original_response,
                additional_info={"model_type": "api_model", "query": query},
            )

            if self.memory_config.memory_model.tool_use_parsing_pattern:
                answer_info = parser_answer_func(
                    response_text, self.memory_config.memory_model.tool_use_parsing_pattern
                )
                if answer_info:
                    response_text = answer_info["action_content"]
                    raw_response = [raw_response]
                else:
                    response_text = search_message
                    raw_response = search_triggered
            else:
                raw_response = [raw_response] if raw_response else []

        return response_text, raw_response

    def _compute_similarity_matrix_direct(self, embeddings: torch.Tensor, device: str) -> torch.Tensor:
        embeddings = embeddings.to(device)

        similarity_matrix = embeddings @ embeddings.T  # (N, N)

        similarity_matrix.fill_diagonal_(-1.0)

        return similarity_matrix

    def _select_entries_to_keep(
        self, duplicate_pairs: List[Tuple[int, int, float]], uids: List[str], entries: Dict[str, Any]
    ) -> set:
        """
        Select which entries to delete based on timestamp (keep newer ones).

        Args:
            duplicate_pairs: List of (i, j, similarity_score) tuples
            uids: List of UIDs corresponding to indices
            entries: Dictionary of UID -> entry

        Returns:
            Set of UIDs to delete
        """
        deleted = set()

        for i, j, score in duplicate_pairs:
            uid_i, uid_j = uids[i], uids[j]

            # Skip if either has already been marked for deletion
            if uid_i in deleted or uid_j in deleted:
                continue

            # Get timestamps
            entry_i = entries.get(uid_i)
            entry_j = entries.get(uid_j)

            if entry_i is None or entry_j is None:
                continue

            ts_i = self._get_entry_timestamp(entry_i)
            ts_j = self._get_entry_timestamp(entry_j)

            # Keep the newer entry (larger timestamp)
            # If timestamps are equal, keep the first one (uid_i)
            if ts_i < ts_j:
                deleted.add(uid_i)
            else:
                deleted.add(uid_j)

        return deleted

    def _get_entry_timestamp(self, entry) -> float:
        """
        Extract timestamp from memory entry.

        Priority order:
        1. metadata['timestamp']
        2. metadata['created_at']
        3. metadata['updated_at'] or metadata['last_updated']
        4. Return 0.0 if no timestamp found

        Args:
            entry: Memory entry object

        Returns:
            float: Timestamp value, or 0.0 if not found
        """
        if not hasattr(entry, "metadata") or entry.metadata is None:
            return 0.0

        metadata = entry.metadata

        # Try different timestamp fields
        for field in ["timestamp", "created_at", "updated_at", "last_updated"]:
            if field in metadata:
                ts = metadata[field]
                # Convert to float if needed
                if isinstance(ts, (int, float)):
                    return float(ts)
                elif isinstance(ts, str):
                    try:
                        # Try parsing as timestamp
                        from datetime import datetime

                        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        return dt.timestamp()
                    except:
                        pass

        return 0.0

    def _find_duplicate_pairs(self, similarity_matrix: torch.Tensor, threshold: float) -> List[Tuple[int, int, float]]:
        """
        Find all pairs of entries with similarity >= threshold.

        Args:
            similarity_matrix: Similarity matrix (N, N)
            threshold: Similarity threshold

        Returns:
            List of (i, j, similarity_score) tuples, sorted by similarity (descending)
        """
        triu = torch.triu(similarity_matrix, diagonal=1)

        indices = torch.where(triu >= threshold)

        if len(indices[0]) == 0:
            return []

        scores = triu[indices]

        pairs = [(int(i), int(j), float(s)) for i, j, s in zip(indices[0], indices[1], scores)]

        pairs.sort(key=lambda x: x[2], reverse=True)

        return pairs

    async def dedup_memory(
        self, similarity_threshold: Optional[float] = None, task_ids: Optional[List[str]] = None, dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Deduplicate memory entries based on embedding similarity.

        Args:
            similarity_threshold: Similarity threshold (default: from config)
            task_ids: List of task IDs to deduplicate (None = all tasks)
            dry_run: If True, only compute statistics without deleting

        Returns:
            Dictionary with deduplication statistics:
            {
                'total_tasks': int,
                'task_stats': {
                    'task_id': {
                        'original_count': int,
                        'final_count': int,
                        'removed_count': int,
                        'duplicate_pairs': int,
                        'removed_uids': List[str]
                    }
                },
                'total_removed': int,
                'execution_time': float,
                'device_used': str
            }
        """
        start_time = time.time()

        if similarity_threshold is None:
            similarity_threshold = self.memory_config.dedup_similarity_threshold

        device_name = self.memory_config.dedup_device
        if device_name == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            if device_name == "cuda":
                logger.warning("CUDA requested but not available, using CPU for deduplication")

        logger.info(f"Starting memory deduplication: threshold={similarity_threshold:.3f}, device={device}")

        is_multi_task = hasattr(self.memory, "task_to_uid")

        if is_multi_task:
            if task_ids is None:
                tasks_to_process = list(self.memory.task_to_uid.keys())
            else:
                tasks_to_process = [t for t in task_ids if t in self.memory.task_to_uid]

            if task_ids is not None and len(tasks_to_process) < len(task_ids):
                missing = set(task_ids) - set(tasks_to_process)
                logger.warning(f"Some task_ids not found: {missing}")
        else:
            tasks_to_process = ["_global_"]

        task_stats = {}
        total_removed = 0

        for task_id in tasks_to_process:
            logger.info(f"Processing task: {task_id}")

            if is_multi_task:
                search_info = self.memory.get_search_info(task_id)
            else:
                search_info = self.memory.get_search_info()

            if not search_info:
                logger.info(f"Task {task_id} has no entries, skipping")
                task_stats[task_id] = {
                    "original_count": 0,
                    "final_count": 0,
                    "removed_count": 0,
                    "duplicate_pairs": 0,
                    "removed_uids": [],
                }
                continue

            original_count = len(search_info)

            valid_entries = []
            for uid, search_field in search_info:
                if isinstance(search_field, torch.Tensor):
                    valid_entries.append((uid, search_field))
                else:
                    logger.warning(f"Entry {uid[:8]}... has no embedding (type: {type(search_field)}), skipping")

            if len(valid_entries) == 0:
                logger.info(f"Task {task_id} has no valid embeddings, skipping")
                task_stats[task_id] = {
                    "original_count": original_count,
                    "final_count": original_count,
                    "removed_count": 0,
                    "duplicate_pairs": 0,
                    "removed_uids": [],
                    "entries_without_embedding": original_count - len(valid_entries),
                }
                continue

            if len(valid_entries) < 2:
                logger.info(f"Task {task_id} has only {len(valid_entries)} entry, no deduplication needed")
                task_stats[task_id] = {
                    "original_count": original_count,
                    "final_count": original_count,
                    "removed_count": 0,
                    "duplicate_pairs": 0,
                    "removed_uids": [],
                }
                continue

            uids = [uid for uid, _ in valid_entries]
            embeddings = torch.stack([emb for _, emb in valid_entries])  # (N, D)

            logger.info(f"Task {task_id}: {len(uids)} entries with embeddings")

            with torch.no_grad():
                similarity_matrix = self._compute_similarity_matrix_direct(embeddings, device)

            duplicate_pairs = self._find_duplicate_pairs(similarity_matrix, similarity_threshold)

            logger.info(f"Task {task_id}: Found {len(duplicate_pairs)} duplicate pairs")

            if len(duplicate_pairs) == 0:
                task_stats[task_id] = {
                    "original_count": original_count,
                    "final_count": original_count,
                    "removed_count": 0,
                    "duplicate_pairs": 0,
                    "removed_uids": [],
                }
                continue

            uids_to_delete = self._select_entries_to_keep(duplicate_pairs, uids, self.memory.entries)

            logger.info(f"Task {task_id}: Selected {len(uids_to_delete)} entries for deletion")

            if not dry_run and len(uids_to_delete) > 0:
                async with self.memory_lock.writer():
                    deleted_count = 0
                    for uid in uids_to_delete:
                        if self.memory.delete_entry(uid):
                            deleted_count += 1
                            if self._uid_usage is not None and uid in self._uid_usage:
                                del self._uid_usage[uid]

                    logger.info(f"Task {task_id}: Successfully deleted {deleted_count}/{len(uids_to_delete)} entries")

            removed_count = len(uids_to_delete)
            final_count = original_count - removed_count

            task_stats[task_id] = {
                "original_count": original_count,
                "final_count": final_count,
                "removed_count": removed_count,
                "duplicate_pairs": len(duplicate_pairs),
                "removed_uids": list(uids_to_delete),
            }

            total_removed += removed_count

        execution_time = time.time() - start_time

        result = {
            "total_tasks": len(tasks_to_process),
            "task_stats": task_stats,
            "total_removed": total_removed,
            "execution_time": execution_time,
            "device_used": str(device),
            "similarity_threshold": similarity_threshold,
            "dry_run": dry_run,
        }

        logger.info(
            f"Deduplication complete: {total_removed} entries removed across {len(tasks_to_process)} tasks "
            f"in {execution_time:.2f}s (dry_run={dry_run})"
        )

        return result

    async def notify_memory_warmup(self):
        self.current_memory_warmup_interval += 1
        if self.current_memory_warmup_interval >= self.memory_warmup_interval:
            self.begin_interaction = True

    async def could_begin_interaction(self):
        return self.begin_interaction

    async def record_actor_performance(self, performance_metrics: Dict):
        if not performance_metrics:
            return

        async with self._tag_performance_lock:
            for key, value in performance_metrics.items():
                if key not in self._tag_performance:
                    self._tag_performance[key] = {
                        "score_sum": 0.0,
                        "traj_count": 0,
                    }
                for metric_key, metric_value in value.items():
                    if metric_key == "score_sum":
                        self._tag_performance[key]["score_sum"] += float(metric_value)
                    elif metric_key == "traj_count":
                        self._tag_performance[key]["traj_count"] += int(metric_value)

        logger.info(f"Updated tag performance: {self._tag_performance}")

    async def get_tag_performance(self, tag: Optional[str] = None) -> Dict[str, Any]:
        async with self._tag_performance_lock:
            if tag is not None:
                return self._tag_performance.get(tag, {}).copy()
            else:
                return {k: v.copy() for k, v in self._tag_performance.items()}


class AsyncMemoryManagerClient:
    """
    Client interface for AsyncMemoryManager that provides synchronous methods
    """

    def __init__(
        self,
        memory_config: MemoryConfig,
        mode: str = "train",
        resource_manager: Optional[ResourceManager] = None,
        memory_model_cluster: Optional[Cluster] = None,
    ):
        self.memory_config = memory_config
        self.mode = mode

        actor_options = {
            "name": f"memory_manager_{id(self)}",
            "namespace": RAY_NAMESPACE,
            "num_cpus": 1.0,
            "scheduling_strategy": NodeAffinitySchedulingStrategy(node_id=ray.nodes()[0]["NodeID"], soft=False),
        }

        self.actor = AsyncMemoryManager.options(**actor_options).remote(
            memory_config, mode, resource_manager, memory_model_cluster
        )

        ray.get(self.actor.initialize.remote())

        logger.info("AsyncMemoryManagerClient initialized")

    def search(self, query: str, task_id: str = None) -> Tuple[str, List[str], List[DataProto]]:
        """Synchronous search interface"""
        return ray.get(self.actor.search.remote(query, task_id))

    def update(self, data: Any, mode: Optional[str] = None) -> bool:
        """Synchronous update interface"""
        return ray.get(self.actor.update.remote(data, mode))

    @property
    def memory_size(self) -> int:
        """Get current memory size"""
        return ray.get(self.actor.get_memory_size.remote())

    @property
    def memory_key_field(self) -> str:
        """Get memory key field"""
        fields = ray.get(self.actor.get_memory_fields.remote())
        return fields["key_field"]

    @property
    def memory_value_field(self) -> str:
        """Get memory value field"""
        fields = ray.get(self.actor.get_memory_fields.remote())
        return fields["value_field"]

    @property
    def memory_uid_field(self) -> str:
        """Get memory UID field"""
        fields = ray.get(self.actor.get_memory_fields.remote())
        return fields["uid_field"]

    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        return ray.get(self.actor.get_performance_stats.remote())

    def get_entry_by_uid(self, uid: str) -> Optional[Dict[str, Any]]:
        """
        Get memory entry by UID.

        Args:
            uid: Memory entry UID

        Returns:
            Dict containing entry data (uid, key, value, metadata, formatted_message) if found, None otherwise
        """
        return ray.get(self.actor.get_entry_by_uid.remote(uid))

    def get_usage_stats(self, uids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get usage stats (retrieve/train counts) for a list of UIDs."""
        return ray.get(self.actor.get_usage_stats.remote(uids))

    def record_training(
        self,
        uids_list: List[List[str]],
        global_step: int,
        weights: Optional[List[float]] = None,
    ) -> bool:
        """Record that these UID lists were used for training at this step."""
        return ray.get(self.actor.record_training.remote(uids_list, global_step, weights))

    def record_actor_performance(self, performance_metrics: Dict) -> bool:
        """Record actor performance metrics, particularly per-tag average scores."""
        return ray.get(self.actor.record_actor_performance.remote(performance_metrics))

    def get_tag_performance(self, tag: Optional[str] = None) -> Dict[str, Any]:
        """
        Get per-tag performance statistics.

        Args:
            tag: Optional specific tag to query. If None, returns all tags.

        Returns:
            Dictionary containing performance statistics
        """
        return ray.get(self.actor.get_tag_performance.remote(tag))

    def save_state(self, save_path: Optional[str] = None) -> bool:
        """Save memory state to the given path"""
        return ray.get(self.actor.save_state.remote(save_path))

    def flush_pending_updates(self, timeout: Optional[float] = 30.0) -> int:
        """
        Wait for all pending updates in queue to be processed.

        Call this before suspend to ensure memory updates are not lost.

        Args:
            timeout: Maximum time to wait in seconds. If None, wait indefinitely.

        Returns:
            Number of updates successfully processed
        """
        return ray.get(self.actor.flush_pending_updates.remote(timeout))

    def flush_pending_updates_async(self, timeout: Optional[float] = 30.0):
        """
        Async version of flush_pending_updates. Returns an ObjectRef.
        """
        return self.actor.flush_pending_updates.remote(timeout)

    def suspend(self, global_step: int):
        return ray.get(self.actor.suspend.remote(global_step))

    def resume(self, global_step: int):
        return ray.get(self.actor.resume.remote(global_step))

    def shutdown(self):
        ray.get(self.actor.shutdown.remote())

        ray.kill(self.actor)

        logger.info("AsyncMemoryManagerClient shutdown complete")

    def notify_update(self):
        ray.get(self.actor.notify_embedding_index_update.remote())

    def notify_memory_warmup(self):
        ray.get(self.actor.notify_memory_warmup.remote())

    def merge_memory(self):
        ray.get(self.actor.merge_memory.remote())

    def search_and_ask(
        self, query: str, return_interaction: bool = False, env_name: Optional[str] = None
    ) -> Tuple[str, List[str], List[DataProto]]:
        return ray.get(self.actor.search_and_ask.remote(query, return_interaction, env_name))

    def dedup_memory(
        self, similarity_threshold: Optional[float] = None, task_ids: Optional[List[str]] = None, dry_run: bool = False
    ):
        return ray.get(self.actor.dedup_memory.remote(similarity_threshold, task_ids, dry_run))

    def could_begin_interaction(self):
        return ray.get(self.actor.could_begin_interaction.remote())

    def record_actor_performance(self, performance_metrics: Dict):
        ray.get(self.actor.record_actor_performance.remote(performance_metrics))

    def get_tag_performance(self, tag: Optional[str] = None) -> Dict[str, Any]:
        return ray.get(self.actor.get_tag_performance.remote(tag))
