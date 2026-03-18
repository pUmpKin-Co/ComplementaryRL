import random
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, List

from roll.pipeline.agentic.memory.memory_config import UpdaterConfig
from roll.pipeline.agentic.memory.memory_structure.base_memory import \
    BaseMemoryStructure
from roll.utils.logging import get_logger

logger = get_logger()


class BaseUpdater(ABC):
    def __init__(self, memory_config: UpdaterConfig):
        self.memory_config = memory_config
        self.max_size: int = memory_config.memory_size

    @abstractmethod
    def update(self, data: Any, memory: BaseMemoryStructure):
        pass


class RandomUpdater(BaseUpdater):
    def update(self, data: Any, memory: BaseMemoryStructure):
        memory.add_entry(data)

        if self.max_size is not None:
            while len(memory) > self.max_size:
                delete_entry_uid = random.choice(memory.get_all_uids())
                memory.delete_entry(delete_entry_uid)


class FIFOUpdater(BaseUpdater):
    """
    FIFO (First In, First Out) updater that maintains entries ordered by timestamp.
    Uses an OrderedDict to track insertion order and efficiently evicts oldest entries.
    """
    
    def __init__(self, memory_config: UpdaterConfig):
        super().__init__(memory_config)
        # OrderedDict to maintain timestamp ordering: {uid: timestamp}
        self.time_stamp_dict: OrderedDict[str, float] = OrderedDict()

    def update(self, data: Any, memory: BaseMemoryStructure):
        """
        Add new entry to memory and update timestamp tracking.
        Evicts oldest entries if memory exceeds max_size.
        """
        # Add entry to memory first and get the UID
        current_time = data.get("timestamp", time.time())
        data["timestamp"] = current_time

        added_uid = memory.add_entry(data)
        
        if added_uid is None:
            logger.warning("Failed to add entry to memory, skipping timestamp update")
            return
        
        self.time_stamp_dict[added_uid] = current_time
        self._sync_timestamp_dict(memory)
        
        # Evict entries if memory exceeds max size
        if self.max_size is not None:
            while len(memory) > self.max_size:
                self._evict_oldest_entry(memory)
    
    def _sync_timestamp_dict(self, memory: BaseMemoryStructure) -> None:
        """
        Synchronize timestamp dict with current memory UIDs using set operations.
        Adds missing UIDs and removes stale ones.
        """

        # Get current UIDs from memory
        memory_uids = set(memory.get_all_uids())
        timestamp_uids = set(self.time_stamp_dict.keys())
        
        # Find UIDs that are in memory but not in timestamp dict
        missing_uids = memory_uids - timestamp_uids
        
        # Find UIDs that are in timestamp dict but not in memory (stale)
        stale_uids = timestamp_uids - memory_uids
        
        # Add missing UIDs with current timestamp
        current_time = time.time()
        for uid in missing_uids:
            self.time_stamp_dict[uid] = current_time
            logger.debug(f"Added missing UID {uid[:8]}... to timestamp dict")
        
        # Remove stale UIDs from timestamp dict
        for uid in stale_uids:
            del self.time_stamp_dict[uid]
            logger.debug(f"Removed stale UID {uid[:8]}... from timestamp dict")
        
        # If we added entries, we need to maintain proper ordering
        # Move newly added entries to the end (most recent)
        if missing_uids:
            for uid in missing_uids:
                # Move to end to maintain proper insertion order
                self.time_stamp_dict.move_to_end(uid)
    
    def _evict_oldest_entry(self, memory: BaseMemoryStructure) -> None:
        """
        Evict the oldest entry (first in timestamp dict) from memory.
        """
        if not self.time_stamp_dict:
            logger.warning("No entries in timestamp dict for eviction")
            return
        
        # Get the first (oldest) entry from OrderedDict
        oldest_uid, oldest_timestamp = next(iter(self.time_stamp_dict.items()))
        
        # Remove from memory
        memory.delete_entry(oldest_uid)
        
        # Remove from timestamp dict
        del self.time_stamp_dict[oldest_uid]
        
        logger.debug(f"Evicted oldest entry {oldest_uid[:8]}... with timestamp {oldest_timestamp}")
    
    def get_timestamp_info(self) -> dict:
        """
        Get information about current timestamp tracking.
        Useful for debugging and monitoring.
        """
        if not self.time_stamp_dict:
            return {
                'total_tracked': 0,
                'oldest_timestamp': None,
                'newest_timestamp': None
            }
        
        timestamps = list(self.time_stamp_dict.values())
        return {
            'total_tracked': len(self.time_stamp_dict),
            'oldest_timestamp': min(timestamps),
            'newest_timestamp': max(timestamps),
            'oldest_uid': next(iter(self.time_stamp_dict.keys())),
            'newest_uid': next(reversed(self.time_stamp_dict.keys()))
        }


class LRUUpdater(BaseUpdater):
    """
    LRU (Least Recently Used) updater that maintains entries ordered by access time.
    Uses an OrderedDict to track access order and efficiently evicts least recently used entries.
    """
    
    def __init__(self, memory_config: UpdaterConfig):
        super().__init__(memory_config)
        # OrderedDict to maintain access order: {uid: last_access_timestamp}
        self.access_time_dict: OrderedDict[str, float] = OrderedDict()

    def update(self, data: Any, memory: BaseMemoryStructure):
        """
        Add new entry to memory and update access time tracking.
        Evicts least recently used entries if memory exceeds max_size.
        """
        # Add entry to memory first and get the UID
        current_time = data.get("timestamp", time.time())
        data["timestamp"] = current_time

        added_uid = memory.add_entry(data)
        
        if added_uid is None:
            logger.warning("Failed to add entry to memory, skipping access time update")
            return
        
        # Update access time dict with the new entry (most recent)
        self.access_time_dict[added_uid] = current_time
        # Move to end to mark as most recently used
        self.access_time_dict.move_to_end(added_uid)
        
        # Synchronize access time dict with memory UIDs
        self._sync_access_time_dict(memory)
        
        # Evict entries if memory exceeds max size
        if self.max_size is not None:
            while len(memory) > self.max_size:
                self._evict_lru_entry(memory)
    
    def update_access_time(self, uid: str) -> None:
        """
        Update access time for a specific UID to mark it as recently used.
        This should be called when an entry is accessed (e.g., during search).
        """
        if uid in self.access_time_dict:
            # Update timestamp and move to end (most recent)
            self.access_time_dict[uid] = time.time()
            self.access_time_dict.move_to_end(uid)
            logger.debug(f"Updated access time for UID {uid[:8]}...")
        else:
            logger.warning(f"Attempted to update access time for unknown UID {uid[:8]}...")
    
    def update_access_times(self, uids: List[str]) -> None:
        """
        Update access times for multiple UIDs (batch operation).
        Useful when multiple entries are accessed during a search operation.
        """
        current_time = time.time()
        for uid in uids:
            if uid in self.access_time_dict:
                self.access_time_dict[uid] = current_time
                self.access_time_dict.move_to_end(uid)
                logger.debug(f"Batch updated access time for UID {uid[:8]}...")
    
    def _sync_access_time_dict(self, memory: BaseMemoryStructure) -> None:
        """
        Synchronize access time dict with current memory UIDs using set operations.
        Adds missing UIDs and removes stale ones.
        """
        # Get current UIDs from memory
        memory_uids = set(memory.get_all_uids())
        access_time_uids = set(self.access_time_dict.keys())
        
        # Find UIDs that are in memory but not in access time dict
        missing_uids = memory_uids - access_time_uids
        
        # Find UIDs that are in access time dict but not in memory (stale)
        stale_uids = access_time_uids - memory_uids
        
        # Add missing UIDs with current timestamp (least recently used)
        current_time = time.time()
        for uid in missing_uids:
            # Add at beginning to mark as least recently used
            self.access_time_dict[uid] = current_time
            # Move to beginning (least recently used position)
            self.access_time_dict.move_to_end(uid, last=False)
            logger.debug(f"Added missing UID {uid[:8]}... to access time dict as LRU")
        
        # Remove stale UIDs from access time dict
        for uid in stale_uids:
            del self.access_time_dict[uid]
            logger.debug(f"Removed stale UID {uid[:8]}... from access time dict")
    
    def _evict_lru_entry(self, memory: BaseMemoryStructure) -> None:
        """
        Evict the least recently used entry (first in access time dict) from memory.
        """
        if not self.access_time_dict:
            logger.warning("No entries in access time dict for eviction")
            return
        
        # Get the first (least recently used) entry from OrderedDict
        lru_uid, lru_timestamp = next(iter(self.access_time_dict.items()))
        
        # Remove from memory
        memory.delete_entry(lru_uid)
        
        # Remove from access time dict
        del self.access_time_dict[lru_uid]
        
        logger.debug(f"Evicted LRU entry {lru_uid[:8]}... with access time {lru_timestamp}")
    
    def get_access_info(self) -> dict:
        """
        Get information about current access time tracking.
        Useful for debugging and monitoring LRU behavior.
        """
        if not self.access_time_dict:
            return {
                'total_tracked': 0,
                'oldest_access': None,
                'newest_access': None,
                'lru_uid': None,
                'mru_uid': None
            }
        
        access_times = list(self.access_time_dict.values())
        return {
            'total_tracked': len(self.access_time_dict),
            'oldest_access': min(access_times),
            'newest_access': max(access_times),
            'lru_uid': next(iter(self.access_time_dict.keys())),  # Least recently used
            'mru_uid': next(reversed(self.access_time_dict.keys())),  # Most recently used
            'access_order': list(self.access_time_dict.keys())  # Full order from LRU to MRU
        }