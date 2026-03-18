import json
import os
import pickle
from collections import defaultdict
from datetime import datetime
from ntpath import isdir
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import ray
import torch
from tqdm import tqdm

from roll.pipeline.agentic.memory.memory_config import MemoryConfig
from roll.pipeline.agentic.memory.memory_structure.base_memory import BaseMemoryStructure
from roll.pipeline.agentic.memory.memory_type.base_type import BaseMemoryType
from roll.utils.logging import get_logger

logger = get_logger()


def _serialize_for_json(obj: Any, tensor_dict: Dict[str, Any], tensor_prefix: str = "") -> Any:
    """
    Convert an object to JSON-serializable format, extracting tensors to tensor_dict.

    Args:
        obj: Object to serialize
        tensor_dict: Dictionary to store extracted tensors
        tensor_prefix: Prefix for tensor keys

    Returns:
        JSON-serializable representation
    """
    if isinstance(obj, torch.Tensor):
        # Store tensor separately and return a reference
        tensor_key = f"{tensor_prefix}_tensor_{len(tensor_dict)}"

        # Handle dtypes not supported by NumPy (e.g., BFloat16)
        original_dtype = str(obj.dtype)
        tensor_cpu = obj.cpu()

        # Convert unsupported dtypes to float32 for storage
        if obj.dtype in [
            torch.bfloat16,
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        ]:
            logger.debug(f"Converting tensor from {obj.dtype} to float32 for storage")
            tensor_cpu = tensor_cpu.to(torch.float32)

        try:
            tensor_dict[tensor_key] = tensor_cpu.numpy()
        except Exception as e:
            # Fallback: convert to float32 if conversion fails
            logger.warning(f"Failed to convert tensor to numpy ({e}), converting to float32")
            tensor_dict[tensor_key] = tensor_cpu.to(torch.float32).numpy()

        return {
            "__tensor__": tensor_key,
            "shape": list(obj.shape),
            "dtype": original_dtype,
            "original_dtype": original_dtype,  # Store original for restoration
        }

    elif isinstance(obj, np.ndarray):
        # Store numpy array separately
        tensor_key = f"{tensor_prefix}_array_{len(tensor_dict)}"
        tensor_dict[tensor_key] = obj
        return {
            "__ndarray__": tensor_key,
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
        }

    elif isinstance(obj, dict):
        return {k: _serialize_for_json(v, tensor_dict, f"{tensor_prefix}_{k}") for k, v in obj.items()}

    elif isinstance(obj, (list, tuple)):
        serialized = [_serialize_for_json(item, tensor_dict, f"{tensor_prefix}_{i}") for i, item in enumerate(obj)]
        return {"__tuple__": serialized} if isinstance(obj, tuple) else serialized

    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj

    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()  # Convert numpy scalars to Python types

    else:
        # For other objects, try to convert to string
        return str(obj)


def _deserialize_from_json(obj: Any, tensor_dict: Dict[str, Any]) -> Any:
    """
    Reconstruct object from JSON representation, restoring tensors from tensor_dict.

    Args:
        obj: JSON-serializable object
        tensor_dict: Dictionary containing extracted tensors

    Returns:
        Reconstructed object
    """
    if isinstance(obj, dict):
        if "__tensor__" in obj:
            # Restore tensor
            tensor_key = obj["__tensor__"]
            if tensor_key in tensor_dict:
                tensor = torch.from_numpy(tensor_dict[tensor_key])

                # Restore original dtype if it was converted during save
                original_dtype_str = obj.get("original_dtype", obj.get("dtype"))
                if original_dtype_str and "bfloat16" in original_dtype_str.lower():
                    tensor = tensor.to(torch.bfloat16)
                elif original_dtype_str and "float8" in original_dtype_str.lower():
                    # Keep as float32 for float8 types since they're not widely supported yet
                    pass

                return tensor
            else:
                logger.warning(f"Tensor key {tensor_key} not found in tensor_dict")
                return None

        elif "__ndarray__" in obj:
            # Restore numpy array
            array_key = obj["__ndarray__"]
            if array_key in tensor_dict:
                return tensor_dict[array_key]
            else:
                logger.warning(f"Array key {array_key} not found in tensor_dict")
                return None

        elif "__tuple__" in obj:
            # Restore tuple
            return tuple(_deserialize_from_json(item, tensor_dict) for item in obj["__tuple__"])

        else:
            # Regular dict
            return {k: _deserialize_from_json(v, tensor_dict) for k, v in obj.items()}

    elif isinstance(obj, list):
        return [_deserialize_from_json(item, tensor_dict) for item in obj]

    else:
        return obj


class TabulerMemory(BaseMemoryStructure):
    """
    Simple tabular memory structure for storing and retrieving entries by UID.

    Search and eviction logic should be handled by separate Searcher and Updater modules.
    """

    def __init__(self, config: MemoryConfig):
        super().__init__(config)

        # Simple dictionary storage indexed by UID, storing memory type instances
        self.entries: Dict[str, BaseMemoryType] = {}

        # Initialize memory
        self.init_memory()

    def init_memory(self):
        """Initialize memory structure and load existing data if available."""
        if self.load_path and os.path.exists(self.load_path):
            self.load_memory()
        else:
            logger.info("Initializing empty tabular memory")

    def load_memory(self):
        """
        Load memory from file. Supports:
        - New JSON+NPZ hybrid format (format_version 2.0)
        - Legacy pickle format (.pkl)
        - Legacy JSON format (old format)
        """
        if not self.load_path or not os.path.exists(self.load_path):
            logger.warning(f"Load path {self.load_path} does not exist")
            return

        try:
            # Determine file format based on extension
            is_pickle = self.load_path.endswith(".pkl") or self.load_path.endswith(".pickle")

            if is_pickle:
                # Load pickle format (supports tensors and complex objects)
                with open(self.load_path, "rb") as f:
                    data = pickle.load(f)

                self.entries.clear()
                entries_data = data.get("entries", {})

                # Check if entries are in portable format (dicts) or old format (class instances)
                for uid, entry_data in entries_data.items():
                    if isinstance(entry_data, dict) and "__dict__" in entry_data:
                        # Portable format: reconstruct from saved __dict__
                        # Use memory_type to rebuild the entry
                        memory_instance = self.memory_type.build_entry(
                            entry_data["__dict__"],
                            uid_key=self.memory_uid_field,
                            uid=uid,
                        )
                        self.entries[uid] = memory_instance
                    elif isinstance(entry_data, dict) and "__value__" in entry_data:
                        # Portable format for objects without __dict__
                        # Reconstruct using build_entry
                        memory_instance = self.memory_type.build_entry(
                            entry_data["__value__"],
                            uid_key=self.memory_uid_field,
                            uid=uid,
                        )
                        self.entries[uid] = memory_instance
                    else:
                        # Old format: direct class instances (requires roll module)
                        self.entries[uid] = entry_data

                self.memory_size = len(self.entries)
                logger.info(f"Loaded {self.memory_size} entries from pickle file {self.load_path}")
            else:
                # Load JSON format
                with open(self.load_path, "r", encoding="utf-8") as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError:
                        logger.warning(
                            f"Memory file {self.load_path} is empty or invalid JSON; initializing empty memory"
                        )
                        self.entries.clear()
                        self.memory_size = 0
                        return

                self.entries.clear()

                # Check format version
                metadata = data.get("metadata", {})
                format_version = metadata.get("format_version", "1.0")

                if format_version == "2.0":
                    # Load tensors from companion NPZ file if it exists
                    tensor_dict = {}
                    tensor_path = self.load_path.replace(".json", "_tensors.npz")
                    if os.path.exists(tensor_path):
                        tensor_data = np.load(tensor_path)
                        tensor_dict = {key: tensor_data[key] for key in tensor_data.files}
                        logger.info(f"Loaded {len(tensor_dict)} tensors from {tensor_path}")

                    # Load entries with tensor restoration
                    entries_data = data.get("entries", {})
                    for uid, entry_data in entries_data.items():
                        if isinstance(entry_data, dict) and "__dict__" in entry_data:
                            # Deserialize entry dict, restoring tensors
                            restored_dict = _deserialize_from_json(entry_data["__dict__"], tensor_dict)
                            memory_instance = self.memory_type.build_entry(
                                restored_dict,
                                uid_key=self.memory_uid_field,
                                uid=uid,
                            )
                            self.entries[uid] = memory_instance
                        elif isinstance(entry_data, dict) and "__value__" in entry_data:
                            # Deserialize entry value, restoring tensors
                            restored_value = _deserialize_from_json(entry_data["__value__"], tensor_dict)
                            memory_instance = self.memory_type.build_entry(
                                restored_value,
                                uid_key=self.memory_uid_field,
                                uid=uid,
                            )
                            self.entries[uid] = memory_instance

                    self.memory_size = len(self.entries)
                    logger.info(f"Loaded {self.memory_size} entries from JSON+NPZ file {self.load_path}")
                else:
                    # Old JSON format (backward compatibility)
                    for entry_data in data.get("entries", []):
                        uid = entry_data.get(self.memory_uid_field, None)
                        memory_instance = self.memory_type.build_entry(
                            entry_data,
                            uid_key=self.memory_uid_field,
                            uid=uid,
                        )
                        if uid is None:
                            uid = getattr(memory_instance, self.memory_uid_field)
                        self.entries[uid] = memory_instance

                    self.memory_size = len(self.entries)
                    logger.info(f"Loaded {self.memory_size} entries from legacy JSON file {self.load_path}")

        except Exception as e:
            logger.error(f"Failed to load memory from {self.load_path}: {e}")

    def save_memory(self, save_path: Optional[str] = None):
        """
        Save memory to file using hybrid JSON+NPZ format.

        This format is completely module-independent and doesn't require 'roll' to inspect:
        - JSON file contains all metadata and simple data structures
        - NPZ file contains all tensors and numpy arrays
        """
        save_path = save_path or self.save_path
        if not self.should_save or not save_path:
            return

        if os.path.isdir(save_path):
            time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(save_path, f"memory_{time_str}.json")

        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            # Dictionary to collect all tensors from all entries
            all_tensors = {}

            # Convert memory entries to JSON-serializable format
            json_entries = {}
            for uid, entry in self.entries.items():
                if hasattr(entry, "__dict__"):
                    # Serialize the instance variables, extracting tensors
                    entry_dict = entry.__dict__.copy()
                    json_entry = _serialize_for_json(
                        entry_dict,
                        all_tensors,
                        tensor_prefix=f"entry_{uid}",
                    )
                    json_entries[uid] = {
                        "__class_name__": type(entry).__name__,
                        "__module__": type(entry).__module__,
                        "__dict__": json_entry,
                    }
                else:
                    # Fallback for objects without __dict__
                    json_value = _serialize_for_json(entry, all_tensors, tensor_prefix=f"entry_{uid}")
                    json_entries[uid] = {
                        "__class_name__": type(entry).__name__,
                        "__module__": type(entry).__module__,
                        "__value__": json_value,
                    }

            # Prepare JSON data
            data = {
                "entries": json_entries,
                "metadata": {
                    "memory_type": str(self.memory_type.__name__),
                    "total_entries": self.memory_size,
                    "format_version": "2.0",  # Mark as new format
                    "has_tensors": len(all_tensors) > 0,
                },
            }

            # Save JSON file
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved {self.memory_size} entries to {save_path}")

            # Save tensors to companion NPZ file if any tensors exist
            if all_tensors:
                tensor_path = save_path.replace(".json", "_tensors.npz")
                np.savez_compressed(tensor_path, **all_tensors)
                logger.info(f"Saved {len(all_tensors)} tensors to {tensor_path}")

        except Exception as e:
            logger.error(f"Failed to save memory to {save_path}: {e}")

    def add_entry(self, item: Any) -> Optional[str]:
        """
        Add a new entry to memory.

        Args:
            item: Raw data to be converted to memory entry using memory_type

        Returns:
            Optional[str]: UID of the added entry, None if failed
        """
        try:
            memory_entry = self.memory_type.build_entry(item)
            uid = getattr(memory_entry, self.memory_uid_field)

            # Add or update entry (store memory type instance directlay)
            self.entries[uid] = memory_entry
            self.memory_size = len(self.entries)

            logger.debug(f"Added entry with uid {uid[:8]}...")
            return uid

        except Exception as e:
            logger.error(f"Failed to add entry: {e}")
            return None

    def delete_entry(self, uid: str) -> bool:
        """
        Delete an entry from memory by UID.

        Args:
            uid: Unique identifier of the entry to delete

        Returns:
            bool: True if successfully deleted, False otherwise
        """
        try:
            if uid not in self.entries:
                logger.warning(f"Entry with uid {uid} not found")
                return False

            # Remove from storage
            del self.entries[uid]
            self.memory_size = len(self.entries)

            logger.debug(f"Deleted entry with uid {uid[:8]}...")
            return True

        except Exception as e:
            logger.error(f"Failed to delete entry: {e}")
            return False

    def search_entry(self, uids: List[str]) -> Tuple[str, List[Any]]:
        """
        Retrieve formatted messages for given UIDs.
        This method is called by external Searcher modules.

        Args:
            uids: List of UIDs to retrieve and format

        Returns:
            Tuple[str, List[Any]]: (formatted_message, triggered_interactions)
        """
        try:
            messages = []
            triggered_interactions = []
            if not uids:
                return "[No Memory Found]", []

            for uid in uids:
                memory_instance = self.get_entry_by_uid(uid)
                if memory_instance:
                    formatted_message = memory_instance.format_message()
                    messages.append(formatted_message)
                    if memory_instance.metadata.get("triggered_interaction", None) is not None:
                        triggered_interactions.append(memory_instance.metadata.get("triggered_interaction"))
                    fetch_count = memory_instance.metadata.get("fetch_count", 0)
                    fetch_count += 1
                    memory_instance.metadata["fetch_count"] = fetch_count
                else:
                    logger.warning(f"Entry with uid {uid} not found during search")

            if not messages:
                return "[No Memory Found]", []

            formatted_list = "\n\n".join(f"{idx + 1}. {entry}" for idx, entry in enumerate(messages))
            message = f"\n\n{formatted_list}"
            return message, triggered_interactions

        except Exception as e:
            logger.error(f"Failed to format entries for search: {e}")
            return "[No Memory Found]", []

    def get_entry_by_uid(self, uid: str) -> Optional[BaseMemoryType]:
        """Get memory type instance by UID."""
        return self.entries.get(uid)

    def get_all_entries(self) -> List[BaseMemoryType]:
        """Get all memory type instances as a list."""
        return list(self.entries.values())

    def get_all_uids(self) -> List[str]:
        """Get all UIDs in the memory."""
        return list(self.entries.keys())

    def clear_memory(self):
        """Clear all entries from memory."""
        self.entries.clear()
        self.memory_size = 0
        logger.info("Cleared all memory entries")

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get basic memory statistics."""
        return {
            "total_entries": self.memory_size,
            "memory_type": self.memory_type.__name__,
        }

    def has_entry(self, uid: str) -> bool:
        """Check if an entry exists by UID."""
        return uid in self.entries

    def get_search_info(self) -> List[Tuple]:
        """
        (uid, search_field) for search in searcher
        """
        return [(uid, getattr(values, self.memory_key_field)) for uid, values in self.entries.items()]


class MultiTaskTabulerMemory(TabulerMemory):
    def __init__(self, config: MemoryConfig):
        self.uid_to_task_id = {}
        self.task_to_uid = defaultdict(list)
        super().__init__(config)

    def load_memory(self):
        super().load_memory()
        self.uid_to_task_id.clear()
        self.task_to_uid.clear()

        for uid, entry in self.entries.items():
            # Try to get task_id from entry attribute or metadata
            task_id = None

            # First try: direct attribute
            if hasattr(entry, "task"):
                task_id = getattr(entry, "task")
            # Second try: metadata
            elif hasattr(entry, "metadata") and entry.metadata:
                task_id = entry.metadata.get("task") or entry.metadata.get("task_id")
            # Third try: check if it's in the entry's __dict__
            elif hasattr(entry, "__dict__"):
                task_id = entry.__dict__.get("task")

            if task_id is not None:
                self.uid_to_task_id[uid] = task_id
                self.task_to_uid[task_id].append(uid)
            else:
                logger.warning(
                    f"Entry {uid[:8]}... loaded without task_id. " "This entry will not be searchable by task."
                )

        logger.info(
            f"Rebuilt task mappings: {len(self.task_to_uid)} tasks, " f"{len(self.uid_to_task_id)} entries mapped"
        )

    def add_entry(self, item: Any) -> Optional[str]:
        """
        Add a new entry to memory.
        """
        assert "task" in item, "Task ID is required for multi-task memory"
        task_id = item["task"]
        uid = super().add_entry(item)
        if uid is not None:
            self.uid_to_task_id[uid] = task_id
            self.task_to_uid[task_id].append(uid)
        return uid

    def delete_entry(self, uid: str) -> bool:
        """
        Delete an entry from memory by UID and clean up task mappings.

        Args:
            uid: Unique identifier of the entry to delete

        Returns:
            bool: True if successfully deleted, False otherwise
        """
        if uid not in self.entries:
            logger.warning(f"Entry with uid {uid} not found")
            return False

        # Get task_id before deletion for cleanup
        task_id = self.uid_to_task_id.get(uid)

        # Delete entry using parent method
        success = super().delete_entry(uid)

        if success and task_id is not None:
            # Clean up task mappings
            if uid in self.uid_to_task_id:
                del self.uid_to_task_id[uid]

            # Remove from task_to_uid list
            if task_id in self.task_to_uid and uid in self.task_to_uid[task_id]:
                self.task_to_uid[task_id].remove(uid)
                if not self.task_to_uid[task_id]:
                    del self.task_to_uid[task_id]

        return success

    def clear_memory(self):
        """Clear all entries from memory and task mappings."""
        super().clear_memory()
        self.uid_to_task_id.clear()
        self.task_to_uid.clear()
        logger.info("Cleared all memory entries and task mappings")

    def init_embedding_for_entry(self, embedding_cluster, micro_batch_size: int = 32):
        """
        Initialize embeddings for all entries across all tasks.
        Args:
            embedding_cluster: Cluster of embedding workers to use for encoding
            micro_batch_size: Size of micro-batches to process on each worker (default: 32)
        """
        # Get search_info for all tasks and combine
        all_search_info = []
        for task_id in self.task_to_uid.keys():
            task_search_info = self.get_search_info(task_id)
            all_search_info.extend(task_search_info)

        if not all_search_info:
            return

        # Filter out entries that already have embeddings (non-string keys)
        text_entries = [(uid, text_key) for uid, text_key in all_search_info if isinstance(text_key, str)]

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
        for uid, embedding in all_embeddings.items():
            if uid in self.entries:
                entry = self.entries[uid]
                # Store original text key in metadata
                if not hasattr(entry, "metadata") or entry.metadata is None:
                    entry.metadata = {}

                # Get original text key from mapping
                if uid in text_key_map:
                    entry.metadata["text_key"] = text_key_map[uid]

                # Update the key field with embedding
                setattr(entry, self.memory_key_field, embedding)
            else:
                # Fallback: try to get entry by uid if method exists
                if hasattr(self, "get_entry_by_uid"):
                    entry = self.get_entry_by_uid(uid)
                    if entry:
                        # Store original text key in metadata
                        if not hasattr(entry, "metadata") or entry.metadata is None:
                            entry.metadata = {}

                        if uid in text_key_map:
                            entry.metadata["text_key"] = text_key_map[uid]

                        # Update the key field with embedding
                        setattr(entry, self.memory_key_field, embedding)

    def get_search_info(self, task_id: str) -> List[Tuple]:
        """
        Get search info for a given task ID.

        Args:
            task_id: Task ID to filter entries by

        Returns:
            List of (uid, search_field) tuples for entries belonging to the task
        """
        if task_id not in self.task_to_uid:
            # Task ID doesn't exist, return empty list
            return []

        result = []
        for uid in self.task_to_uid[task_id]:
            if uid in self.entries:
                result.append((uid, getattr(self.entries[uid], self.memory_key_field)))
        return result
