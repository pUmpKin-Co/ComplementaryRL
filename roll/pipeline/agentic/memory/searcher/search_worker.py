import asyncio
import json
import os
import shutil
import time
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from tensordict import TensorDict

from roll.configs.worker_config import WorkerConfig
from roll.distributed.executor.cluster import Cluster
from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.decorator import Dispatch, register
from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.agentic.memory.memory_config import MemoryConfig, SearcherConfig
from roll.pipeline.agentic.memory.searcher import memory_searcher_dict
from roll.pipeline.agentic.memory.searcher.base_searcher import BaseSearcher
from roll.utils.logging import get_logger

logger = get_logger()


class MemorySearcherWorker(Worker):
    """
    Worker for managing distributed memory search operations.

    Note: Since memory structure is globally shared across all environments,
    some statistics (like TF-IDF cache in SimpleSimilaritySearcher) are saved
    only once globally rather than per env_id.

    For GPU-based embedding searchers, be aware of memory constraints:
    1. **Model Loading**: Each worker loads its own searcher instance.
       - For GPU models: Set world_size = number of available GPUs
       - Avoid multiple workers sharing the same GPU (causes OOM)
       - Use GPU placement via device_mapping in worker_config

    2. **Embedding Cache**: Implement bounded caches in your searcher:
       - Use LRU cache with max_size limit
       - Precompute and persist embeddings offline
       - Share embeddings across workers via shared storage

    3. **Batch Size**: Limit concurrent searches per worker:
       - Each search creates temporary GPU tensors
       - Use semaphores to limit concurrency if needed

    Example safe configuration:
        searcher:
          world_size: 4  # Match number of GPUs
          device_mapping: [0, 1, 2, 3]  # One GPU per worker
    """

    def __init__(self, worker_config: SearcherConfig):
        """
        Initialize the Memory Searcher Worker.

        Args:
            worker_config: SearcherConfig containing searcher configuration
        """
        super().__init__(worker_config=worker_config)
        self.worker_config: SearcherConfig = worker_config
        self.memory_config: Optional[MemoryConfig] = None
        self.searcher: Optional[BaseSearcher] = None
        self.stats = {
            "total_searches": 0,
            "average_search_time": 0.0,
            "last_search_time": None,
            "searches_by_rank": 0,
        }

        self.logger.info(f"MemorySearcherWorker initialized on rank {self.rank}/{self.world_size}")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, clear_cache=False)
    def notify_update(self):
        """
        Notify the searcher that the memory has been updated.
        """
        if self.searcher is not None and hasattr(self.searcher, "notify_update"):
            self.searcher.notify_update()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, clear_cache=False)
    def register_embedding_cluster(self, embedding_cluster: Cluster):
        """
        Register the embedding cluster for all searcher workers.

        Args:
            embedding_cluster: Cluster of embedding workers for encoding queries
        """
        self.embedding_cluster = embedding_cluster
        self.logger.info(
            f"Rank {self.rank}: Embedding cluster registered " f"with {embedding_cluster.world_size} workers"
        )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, clear_cache=False)
    def initialize(self, memory_config: MemoryConfig):
        """
        Initialize the searcher based on memory configuration.

        Args:
            memory_config: MemoryConfig containing searcher strategy and parameters
        """
        self.memory_config = memory_config

        searcher_strategy = memory_config.searcher.memory_search_strategy
        searcher_cls = memory_searcher_dict.get(searcher_strategy)

        if searcher_cls is None:
            raise ValueError(
                f"Unknown searcher strategy: {searcher_strategy}. "
                f"Available strategies: {list(memory_searcher_dict.keys())}"
            )

        self.searcher = searcher_cls(memory_config.searcher)

        try:
            diversity_enable = bool(getattr(memory_config.searcher, "diversity_enable", False))
            base_k = int(getattr(memory_config.searcher, "memory_fetch_num", 0))
            multiplier = int(getattr(memory_config.searcher, "diversity_candidate_multiplier", 2))
            if diversity_enable and base_k > 0 and multiplier > 1:
                candidate_k = base_k * multiplier
                if hasattr(self.searcher, "fetch_num"):
                    self.searcher.fetch_num = candidate_k
                    self.logger.info(
                        f"Diversity enabled: searcher.fetch_num set to {candidate_k} (base_k={base_k}, multiplier={multiplier})"
                    )
        except Exception as e:
            self.logger.warning(f"Failed to set diversity candidate fetch_num: {e}")

        self.logger.info(
            f"Searcher initialized on rank {self.rank}: "
            f"strategy={searcher_strategy}, "
            f"fetch_num={memory_config.searcher.memory_fetch_num}"
        )

        # Load state for stateful searchers
        # Note: All workers load the same state file to ensure consistency
        # Read-only file operations are safe for concurrent access across workers
        if hasattr(self.searcher, "load_state") and memory_config.searcher.memory_searcher_state_path:

            self.logger.info(
                f"Rank {self.rank}: Loading searcher state from: "
                f"{memory_config.searcher.memory_searcher_state_path}"
            )

            self.searcher.load_state(memory_config.searcher.memory_searcher_state_path)

            self.logger.info(f"Rank {self.rank}: Searcher state loaded successfully")

    def _prepare_queries(
        self, queries: Union[List[str], torch.Tensor, str]
    ) -> Tuple[Union[List[str], torch.Tensor], int]:
        """
        Normalize queries into a batched representation and return batch size.
        """
        if isinstance(queries, torch.Tensor):
            if queries.dim() == 1:
                queries = queries.unsqueeze(0)
            return queries, queries.shape[0]

        if isinstance(queries, list):
            if isinstance(queries[0], str):
                return queries, len(queries)
            elif isinstance(queries[0], torch.Tensor):
                return torch.stack(queries), len(queries)

        # Single string (or other scalar) query
        return [queries], 1

    def _execute_search(
        self,
        queries: Union[List[str], torch.Tensor, str],
        search_info: List[Tuple],
    ) -> Tuple[List[List[str]], float, int]:
        """
        Shared search implementation used by both search() and search_batch().
        Returns (uids_list, elapsed_time, batch_size).
        """
        if self.searcher is None:
            raise RuntimeError("Searcher not initialized. Call initialize() first.")

        normalized_queries, batch_size = self._prepare_queries(queries)
        start_time = time.time()

        try:
            if hasattr(self.searcher, "_get_search_result_batch"):
                uids_list = self.searcher._get_search_result_batch(normalized_queries, search_info)

                # Ensure list-of-lists even if searcher returns numpy/tensor
                if isinstance(uids_list, (torch.Tensor, np.ndarray)):
                    uids_list = uids_list.tolist()
            else:
                self.logger.debug(
                    f"Searcher {type(self.searcher).__name__} doesn't support "
                    "batch search, falling back to sequential execution."
                )
                uids_list = []

                if isinstance(normalized_queries, torch.Tensor):
                    for idx in range(batch_size):
                        single_query = normalized_queries[idx]
                        uids = self.searcher._get_search_result(single_query, search_info)
                        uids_list.append(uids)
                else:
                    for query in normalized_queries:
                        uids = self.searcher._get_search_result(query, search_info)
                        uids_list.append(uids)

            elapsed = time.time() - start_time
            return uids_list, elapsed, batch_size

        except Exception as exc:
            self.logger.error(f"Search execution failed for batch of size {batch_size}: {exc}")
            return [[] for _ in range(batch_size)], 0.0, batch_size

    def _update_stats(self, batch_size: int, elapsed: float) -> None:
        """
        Update performance statistics based on batch size and elapsed time.
        """
        if batch_size <= 0:
            return

        self.stats["searches_by_rank"] += batch_size
        self.stats["total_searches"] += batch_size
        per_query_time = elapsed / batch_size if batch_size else 0.0
        self.stats["last_search_time"] = per_query_time

        total = self.stats["total_searches"]
        if total > 0:
            prev_avg = self.stats["average_search_time"]
            previous_total = total - batch_size
            self.stats["average_search_time"] = ((prev_avg * previous_total) + elapsed) / total

    async def search(self, query: Union[str, torch.Tensor], search_info: List[Tuple]) -> List[str]:
        """
        Perform search operation on a single query.

        This method is async to support both CPU-based and GPU-based searchers:
        - For CPU-based searchers (TF-IDF): Runs synchronously in async wrapper
        - For GPU-based searchers (embeddings): Allows non-blocking GPU inference

        Args:
            query: Search query string or embedding tensor
            search_info: List of (uid, search_field) tuples from memory structure

        Returns:
            List of UIDs from search results
        """
        uids_list, elapsed, _ = self._execute_search(query, search_info)
        self._update_stats(1, elapsed)

        return uids_list[0] if uids_list else []

    async def search_batch(
        self,
        queries: Union[List[str], torch.Tensor],
        search_info: List[Tuple],
    ) -> List[List[str]]:
        """
        Perform batch search operation for multiple queries.

        This is more efficient than calling search() multiple times because:
        - For FAISS: Processes all queries in a single GPU call
        - For other searchers: Amortizes overhead across queries

        Args:
            queries: Batch of search queries (list of strings or tensor of embeddings)
            search_info: List of (uid, search_field) tuples from memory structure

        Returns:
            List of lists, where each inner list contains UIDs for one query

        Example:
            >>> queries = [torch.randn(768) for _ in range(16)]
            >>> results = await worker.search_batch(queries, search_info)
            >>> len(results)  # 16
            >>> len(results[0])  # top_k UIDs for first query
        """
        uids_list, elapsed, batch_size = self._execute_search(queries, search_info)
        self._update_stats(batch_size, elapsed)
        return uids_list

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, clear_cache=False)
    def save_state(self, save_path: Optional[str] = None) -> DataProto:
        """
        Save searcher state to disk.

        For stateful searchers (e.g., SimpleSimilaritySearcher with TF-IDF cache),
        all workers' states are merged before saving to ensure no cached data is lost.

        Process (Ray-based, no PyTorch distributed):
        1. All workers save their state_dict to rank-specific temporary files
        2. Rank 0 loads all temporary files and merges the states
        3. Rank 0 saves the merged state to the final location
        4. All workers load the merged state to stay synchronized (via next call)

        Args:
            save_path: Optional path to save state. If None, uses config path.

        Returns:
            DataProto with success status
        """
        if self.searcher is None or not hasattr(self.searcher, "save_state"):
            dummy_batch = TensorDict(source={"dummy": torch.zeros(1)}, batch_size=(1,))
            return DataProto(
                batch=dummy_batch,
                meta_info={"success": True, "message": "No state to save"},
            )

        actual_save_path = save_path or self.memory_config.searcher.memory_searcher_state_path

        if not actual_save_path:
            self.logger.warning("No save path configured for searcher state")
            dummy_batch = TensorDict(source={"dummy": torch.zeros(1)}, batch_size=(1,))
            return DataProto(
                batch=dummy_batch,
                meta_info={
                    "success": False,
                    "message": "No save path configured",
                },
            )

        if os.path.isdir(actual_save_path):
            time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            actual_save_path = os.path.join(actual_save_path, f"searcher_{time_str}.json")
        else:
            parent_dir = os.path.dirname(actual_save_path)
            if not os.path.isdir(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)
            time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            actual_save_path = os.path.join(parent_dir, f"searcher_{time_str}.json")

        try:
            # Step 1: Get local state from all workers
            if hasattr(self.searcher, "get_state_dict"):
                local_state = self.searcher.get_state_dict()
            else:
                local_state = {}

            # Step 2: All workers save their state to rank-specific temp files (Ray-compatible)
            if self.world_size > 1:
                temp_dir = os.path.join(os.path.dirname(actual_save_path), "temp_states")
                os.makedirs(temp_dir, exist_ok=True)

                # Each worker saves its state
                temp_state_path = os.path.join(temp_dir, f"rank_{self.rank}_state.json")
                if hasattr(self.searcher, "save_state") and local_state:
                    with open(temp_state_path, "w") as f:
                        json.dump(local_state, f)
                    self.logger.info(f"Rank {self.rank}: Saved temporary state to {temp_state_path}")

                # Step 3: Rank 0 waits briefly, then loads and merges all states
                if self.rank == 0:
                    # Wait for all workers to write their temp files
                    max_wait = 30  # 30 seconds max wait
                    wait_interval = 0.5
                    elapsed = 0

                    while elapsed < max_wait:
                        all_present = all(
                            os.path.exists(os.path.join(temp_dir, f"rank_{i}_state.json"))
                            for i in range(self.world_size)
                        )
                        if all_present:
                            break
                        time.sleep(wait_interval)
                        elapsed += wait_interval

                    if not all_present:
                        self.logger.warning(
                            f"Not all worker states available after {max_wait}s. " "Proceeding with available states."
                        )

                    self.logger.info(f"Merging states from {self.world_size} workers")

                    # Load and merge all states
                    for i in range(self.world_size):
                        if i == 0:
                            continue  # Skip rank 0 itself (already in searcher)

                        temp_state_path = os.path.join(temp_dir, f"rank_{i}_state.json")
                        if os.path.exists(temp_state_path):
                            try:
                                with open(temp_state_path, "r") as f:
                                    other_state = json.load(f)

                                if hasattr(self.searcher, "merge_state_dict") and other_state:
                                    self.logger.info(f"Merging state from rank {i}")
                                    self.searcher.merge_state_dict(other_state)
                            except Exception as e:
                                self.logger.error(f"Failed to load/merge state from rank {i}: {e}")

                    self.logger.info("All states merged successfully")

                    # Clean up temp files
                    try:
                        shutil.rmtree(temp_dir)
                        self.logger.info("Cleaned up temporary state files")
                    except Exception as e:
                        self.logger.warning(f"Failed to clean up temp directory: {e}")

            # Step 4: Rank 0 saves the merged state
            if self.rank == 0:
                self.logger.info(f"Saving merged searcher state to: {actual_save_path}")
                self.searcher.save_state(actual_save_path)
                self.logger.info("Searcher state saved successfully")

            # Note: For reloading merged state on other workers, call load_state separately
            # after this method completes. No distributed barrier available in Ray.

            dummy_batch = TensorDict(source={"dummy": torch.zeros(1)}, batch_size=(1,))
            return DataProto(
                batch=dummy_batch,
                meta_info={
                    "success": True,
                    "message": f"Rank {self.rank}: State saved" + (" and merged" if self.rank == 0 else ""),
                    "final_path": (actual_save_path if self.rank == 0 else None),
                },
            )

        except Exception as e:
            self.logger.error(f"Rank {self.rank}: Failed to save searcher state: {e}")
            self.logger.error(traceback.format_exc())
            dummy_batch = TensorDict(source={"dummy": torch.zeros(1)}, batch_size=(1,))
            return DataProto(
                batch=dummy_batch,
                meta_info={
                    "success": False,
                    "message": f"Rank {self.rank}: Save failed: {str(e)}",
                },
            )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, clear_cache=False)
    def load_state(self, load_path: Optional[str] = None) -> DataProto:
        """
        Load searcher state from disk.

        All workers load the state to ensure consistency across ranks.
        Read-only operations are safe for concurrent access.

        Args:
            load_path: Optional path to load state from. If None, uses config path.

        Returns:
            DataProto with success status
        """
        if self.searcher is None or not hasattr(self.searcher, "load_state"):
            dummy_batch = TensorDict(source={"dummy": torch.zeros(1)}, batch_size=(1,))
            return DataProto(
                batch=dummy_batch,
                meta_info={"success": True, "message": "No state to load"},
            )

        try:
            actual_load_path = load_path or self.memory_config.searcher.memory_searcher_state_path

            if actual_load_path:
                self.logger.info(f"Rank {self.rank}: Loading searcher state from: {actual_load_path}")

                # All ranks load (read-only operation, safe for concurrent access)
                self.searcher.load_state(actual_load_path)

                self.logger.info(f"Rank {self.rank}: Searcher state loaded successfully")
                dummy_batch = TensorDict(source={"dummy": torch.zeros(1)}, batch_size=(1,))
                return DataProto(
                    batch=dummy_batch,
                    meta_info={
                        "success": True,
                        "message": f"Rank {self.rank}: State loaded from {actual_load_path}",
                    },
                )
            else:
                self.logger.warning("No load path configured for searcher state")
                dummy_batch = TensorDict(source={"dummy": torch.zeros(1)}, batch_size=(1,))
                return DataProto(
                    batch=dummy_batch,
                    meta_info={
                        "success": False,
                        "message": "No load path configured",
                    },
                )

        except Exception as e:
            self.logger.error(f"Rank {self.rank}: Failed to load searcher state: {e}")
            dummy_batch = TensorDict(source={"dummy": torch.zeros(1)}, batch_size=(1,))
            return DataProto(
                batch=dummy_batch,
                meta_info={
                    "success": False,
                    "message": f"Rank {self.rank}: Load failed: {str(e)}",
                },
            )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, clear_cache=False)
    def get_stats(self) -> DataProto:
        """
        Get searcher performance statistics.

        Returns:
            DataProto containing performance statistics
        """
        dummy_batch = TensorDict(source={"dummy": torch.zeros(1)}, batch_size=(1,))
        return DataProto(
            batch=dummy_batch,
            meta_info={"stats": {f"rank_{self.rank}": self.stats.copy()}},
        )
