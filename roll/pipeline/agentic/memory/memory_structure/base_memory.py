from abc import ABC, abstractmethod
from typing import Any, List, Tuple

import ray
from tqdm import tqdm

from roll.pipeline.agentic.memory.memory_config import MemoryConfig
from roll.pipeline.agentic.memory.memory_type import memory_type_dict
from roll.utils.logging import get_logger

logger = get_logger()


class BaseMemoryStructure(ABC):
    def __init__(self, config: MemoryConfig) -> None:
        self.load_path = config.memory_load_path
        self.save_path = config.memory_save_path
        self.should_save = config.memory_should_save
        self.memory_type = memory_type_dict[config.memory_type]
        self.memory_size = 0
    
    def __len__(self):
        return self.memory_size

    @property
    def fields(self):
        """
        The fields of the memory structure
        """
        return self.memory_type.get_fields()
    
    @property
    def memory_uid_field(self):
        """
        The field name of the memory uid
        """
        return self.memory_type.uid_key
    
    @property
    def memory_key_field(self):
        """
        The field name of the memory key
        """
        return self.memory_type.search_key
    
    @property
    def memory_value_field(self):
        """
        The field name of the memory value
        """
        return self.memory_type.value_key

    @abstractmethod
    def init_memory(self):
        pass
    
    @abstractmethod
    def load_memory(self):
        pass

    @abstractmethod
    def save_memory(self):
        pass
    
    @abstractmethod
    def add_entry(self, item: Any):
        pass

    @abstractmethod
    def search_entry(self, uids: List[str]) -> Tuple[str, List[Any]]:
        pass

    @abstractmethod
    def delete_entry(self, uid: str):
        pass

    @abstractmethod
    def get_search_info(self) -> List[Tuple[str, Any]]:
        """
        (uid, search_field) for search in searcher
        """
        pass
    
    def init_embedding_for_entry(self, embedding_cluster, micro_batch_size: int = 32):
        """
        Initialize embeddings for all entries by converting text keys to embeddings.
        
        This method:
        1. Gets all (uid, text_key) pairs from search info
        2. Batch processes embeddings using multiple workers (round-robin distribution)
        3. Uses micro-batching to process large batches in smaller chunks
        4. Updates the key_field with embeddings and stores original text in text_key metadata
        
        Args:
            embedding_cluster: Cluster of embedding workers to use for encoding
            micro_batch_size: Size of micro-batches to process on each worker (default: 32)
        """
        old_search_info = self.get_search_info()
        
        if not old_search_info:
            return
        
        # Filter out entries that already have embeddings (non-string keys)
        text_entries = [(uid, text_key) for uid, text_key in old_search_info 
                        if isinstance(text_key, str)]
        
        if not text_entries:
            return
        
        # Create mapping from uid to text_key for later use
        text_key_map = {uid: text_key for uid, text_key in text_entries}
        
        # Group entries by worker for round-robin distribution
        num_workers = embedding_cluster.world_size
        worker_batches = [[] for _ in range(num_workers)]
        
        for idx, (uid, text_key) in enumerate(text_entries):
            worker_idx = idx % num_workers
            worker_batches[worker_idx].append((uid, text_key))
        
        # Process batches with progress bar and micro-batching
        all_embeddings = {}
        with tqdm(total=len(text_entries), desc="Initializing embeddings") as pbar:
            # Submit all micro-batch encoding tasks asynchronously
            remote_tasks = []
            
            for worker_idx, batch in enumerate(worker_batches):
                if not batch:
                    continue
                
                worker = embedding_cluster.workers[worker_idx]
                
                # Split worker batch into micro-batches
                for micro_batch_start in range(0, len(batch), micro_batch_size):
                    micro_batch_end = min(micro_batch_start + micro_batch_size, len(batch))
                    micro_batch = batch[micro_batch_start:micro_batch_end]
                    
                    uids, texts = zip(*micro_batch)
                    
                    # Submit encoding task asynchronously
                    remote_task = worker.encode.remote(list(texts))
                    remote_tasks.append((uids, remote_task, worker_idx))
            
            # Collect results as they complete
            for uids, remote_task, worker_idx in remote_tasks:
                try:
                    embeddings = ray.get(remote_task)  # Shape: (micro_batch_size, embedding_dim)
                    
                    # Map embeddings back to UIDs
                    for i, uid in enumerate(uids):
                        embedding = embeddings[i].cpu()  # Ensure on CPU
                        all_embeddings[uid] = embedding
                        pbar.update(1)
                except Exception as e:
                    logger.error(f"Failed to get embeddings for micro-batch on worker {worker_idx}: {e}")
                    # Update progress for failed micro-batch
                    pbar.update(len(uids))
        
        # Update entries with embeddings
        # Access entries based on implementation (works for TabulerMemory)
        if hasattr(self, 'entries'):
            for uid, embedding in all_embeddings.items():
                if uid in self.entries:
                    entry = self.entries[uid]
                    # Store original text key in metadata
                    if not hasattr(entry, 'metadata') or entry.metadata is None:
                        entry.metadata = {}
                    
                    # Get original text key from mapping
                    if uid in text_key_map:
                        entry.metadata["text_key"] = text_key_map[uid]
                    
                    # Update the key field with embedding
                    setattr(entry, self.memory_key_field, embedding)
        else:
            # Fallback: try to get entry by uid if method exists
            for uid, embedding in all_embeddings.items():
                if hasattr(self, 'get_entry_by_uid'):
                    entry = self.get_entry_by_uid(uid)
                    if entry:
                        # Store original text key in metadata
                        if not hasattr(entry, 'metadata') or entry.metadata is None:
                            entry.metadata = {}
                        
                        if uid in text_key_map:
                            entry.metadata["text_key"] = text_key_map[uid]
                        
                        # Update the key field with embedding
                        setattr(entry, self.memory_key_field, embedding)