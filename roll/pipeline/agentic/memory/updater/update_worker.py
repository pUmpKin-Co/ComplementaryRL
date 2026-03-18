"""
Memory Updater Worker for distributed memory update operations.

This worker manages updater instances across multiple workers. However, unlike searchers,
most updaters maintain internal state (OrderedDict for LRU/FIFO) that is NOT safe for
concurrent updates without proper synchronization.
"""
import asyncio
import time
from typing import Any, Dict, List, Optional

import torch
from tensordict import TensorDict

from roll.configs.worker_config import WorkerConfig
from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.decorator import Dispatch, register
from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.agentic.memory.memory_config import (MemoryConfig,
                                                        UpdaterConfig)
from roll.pipeline.agentic.memory.updater import memory_updater_dict
from roll.pipeline.agentic.memory.updater.base_updater import BaseUpdater
from roll.utils.logging import get_logger

logger = get_logger()


class MemoryUpdaterWorker(Worker):
    """
    Worker for managing distributed memory update operations.
    
    CONCURRENCY SAFETY WARNING:
    ===========================
    Most updaters (LRU, FIFO) maintain internal state that is NOT thread-safe:
    - LRUUpdater: Uses OrderedDict.move_to_end() - NOT atomic
    - FIFOUpdater: Uses OrderedDict for timestamps - NOT atomic
    - RandomUpdater: Random selection during eviction - race conditions
    
    Safe Scenarios for world_size > 1:
    ------------------------------------
    1. **Pure Append-Only**: If memory structure supports atomic add without eviction
       - Set memory_size=None (no eviction)
       - Update operations are independent
    
    2. **Partitioned Updates**: Each worker handles different memory partitions
       - Requires custom partitioning logic
       - No shared state between workers
    
    Recommended Configuration:
    --------------------------
    For standard updaters (LRU/FIFO/Random):
        updater:
          world_size: 1  # Single worker to avoid race conditions
    
    For append-only scenarios (no eviction):
        updater:
          world_size: N  # Multiple workers OK
          memory_size: null  # No eviction needed
    """
    
    def __init__(self, worker_config: UpdaterConfig):
        """
        Initialize the Memory Updater Worker.
        
        Args:
            worker_config: UpdaterConfig containing updater configuration
        """
        super().__init__(worker_config=worker_config)
        self.worker_config: UpdaterConfig = worker_config
        self.memory_config: Optional[MemoryConfig] = None
        self.updater: Optional[BaseUpdater] = None
        
        self.update_lock = asyncio.Lock()
        
        self.stats = {
            "total_updates": 0,
            "average_update_time": 0.0,
            "last_update_time": None,
            "failed_updates": 0,
            "updates_by_rank": 0,
        }
        
        self.logger.info(f"MemoryUpdaterWorker initialized on rank {self.rank}/{self.world_size}")
        
        if self.world_size > 1:
            self.logger.warning(
                f"MemoryUpdaterWorker running with world_size={self.world_size}. "
                "Ensure your updater is thread-safe or uses world_size=1 to avoid race conditions!"
            )
    
    @register(dispatch_mode=Dispatch.ONE_TO_ALL, clear_cache=False)
    def initialize(self, memory_config: MemoryConfig):
        """
        Initialize the updater based on memory configuration.
        
        Args:
            memory_config: MemoryConfig containing updater strategy and parameters
        """
        self.memory_config = memory_config
        
        # Get updater class from the configured strategy
        updater_strategy = memory_config.updater.memory_update_strategy
        updater_cls = memory_updater_dict.get(updater_strategy)
        
        if updater_cls is None:
            raise ValueError(
                f"Unknown updater strategy: {updater_strategy}. "
                f"Available strategies: {list(memory_updater_dict.keys())}"
            )
        
        self.updater = updater_cls(memory_config.updater)
        
        self.logger.info(
            f"Updater initialized on rank {self.rank}: "
            f"strategy={updater_strategy}, "
            f"memory_size={memory_config.updater.memory_size}"
        )
        
        if self.world_size > 1 and updater_strategy in ['lru', 'fifo', 'random']:
            self.logger.warning(
                f"Rank {self.rank}: Using {updater_strategy} updater with world_size={self.world_size}. "
                "This may cause race conditions! Consider using world_size=1 or implementing "
                "a custom thread-safe updater."
            )
    
    async def update(self, data: Any, memory: Any) -> bool:
        """
        Perform update operation.
        
        This method is async to support:
        1. Non-blocking I/O operations (e.g., disk persistence)
        2. Proper async coordination in the memory manager
        
        The update_lock ensures that if multiple async tasks call this method,
        they won't corrupt the updater's internal state (OrderedDict, etc.).
        However, this only protects WITHIN a single worker. If world_size > 1,
        race conditions can still occur between workers.
        
        Args:
            data: Data entry to add to memory
            memory: Memory structure to update
                
        Returns:
            bool: True if update succeeded, False otherwise
        """
        if self.updater is None:
            raise RuntimeError(
                "Updater not initialized. Call initialize() first."
            )
        
        start_time = time.time()
        
        try:
            # Lock to prevent concurrent access to updater state within this worker
            async with self.update_lock:
                # Delegate to the actual updater which handles both addition and eviction
                self.updater.update(data, memory)
            
            update_time = time.time() - start_time
            self.stats["updates_by_rank"] += 1
            self.stats["total_updates"] += 1
            self.stats["last_update_time"] = update_time
            
            if self.stats["total_updates"] > 0:
                self.stats["average_update_time"] = (
                    (self.stats["average_update_time"] * (self.stats["total_updates"] - 1) 
                     + update_time) 
                    / self.stats["total_updates"]
                )
            
            return True, memory
            
        except Exception as e:
            self.stats["failed_updates"] += 1
            self.logger.error(f"Rank {self.rank}: Update failed: {e}")
            return False, memory
    
    async def update_access_times(self, uids: List[str]) -> bool:
        """
        Update access times for LRU updater.
        
        This is called after successful searches to mark entries as recently accessed.
        
        Args:
            uids: List of UIDs that were accessed
            
        Returns:
            bool: True if update succeeded, False if not supported
        """
        if self.updater is None:
            return False
        
        if not hasattr(self.updater, 'update_access_times'):
            return False
        
        try:
            async with self.update_lock:
                self.updater.update_access_times(uids)
            return True
            
        except Exception as e:
            self.logger.error(f"Rank {self.rank}: Failed to update access times: {e}")
            return False
    
    @register(dispatch_mode=Dispatch.ONE_TO_ALL, clear_cache=False)
    def get_stats(self) -> DataProto:
        """
        Get updater performance statistics.
        
        Returns:
            DataProto containing performance statistics
        """
        # Create a dummy batch with size 1 to avoid StopIteration in __len__
        # when DataProto has only meta_info and empty non_tensor_batch
        dummy_batch = TensorDict(
            source={"dummy": torch.zeros(1)},
            batch_size=(1,)
        )
        return DataProto(
            batch=dummy_batch,
            meta_info={
                "stats": {
                    f"rank_{self.rank}": self.stats.copy()
                }
            }
        )
    
    @register(dispatch_mode=Dispatch.ONE_TO_ALL, clear_cache=False)
    def get_updater_state(self) -> DataProto:
        """
        Get updater internal state for debugging/monitoring.
        
        Returns:
            DataProto containing updater state information
        """
        state_info = {
            "rank": self.rank,
            "world_size": self.world_size,
            "updater_type": type(self.updater).__name__ if self.updater else None,
        }
        
        if hasattr(self.updater, 'access_time_dict'):
            state_info["lru_size"] = len(self.updater.access_time_dict)
            state_info["lru_oldest"] = (
                next(iter(self.updater.access_time_dict.keys())) 
                if self.updater.access_time_dict else None
            )
        elif hasattr(self.updater, 'time_stamp_dict'):
            state_info["fifo_size"] = len(self.updater.time_stamp_dict)
            state_info["fifo_oldest"] = (
                next(iter(self.updater.time_stamp_dict.keys())) 
                if self.updater.time_stamp_dict else None
            )
        
        dummy_batch = TensorDict(
            source={"dummy": torch.zeros(1)},
            batch_size=(1,)
        )
        return DataProto(
            batch=dummy_batch,
            meta_info={
                "state": state_info
            }
        )

