import time
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Set

import numpy as np
import torch

from roll.pipeline.agentic.memory.memory_type.base_type import (
    BaseMemoryType,
)


@dataclass
class CaseMemoryType(BaseMemoryType):
    """
    Case-based memory type for storing case examples.

    Example usage:
        # Get fields without initialization
        fields = CaseMemoryType.fields

        # Build entry - returns CaseMemoryType instance
        entry = CaseMemoryType.build_entry({
            "question": "What is 2+2?",
            "answer": "4",
            "context": "math problem"
        })

        # Use the entry
        message = entry.format_message()  # For agent
        print(entry)  # Beautiful printing
        dict_data = entry.to_dict()  # Convert back to dict
    """

    # Class-level field definitions (available before initialization)
    fields: ClassVar[Set[str]] = {
        "uid",
        "key",
        "value",
        "timestamp",
        "metadata",
    }
    uid_key: ClassVar[str] = "uid"
    search_key: ClassVar[str] = "key"
    value_key: ClassVar[str] = "value"
    block_fields: ClassVar[Set[str]] = {
        "last_turn_uids",
    }  # the fields blocked from being added to the additional fields

    # Additional instance fields for case memory
    timestamp: float = 0.0
    metadata: Dict = None

    def __post_init__(self):
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = {}

    @classmethod
    def _extract_search_key(cls, data: Any) -> Any:
        """
        Extract search key from input data.

        Args:
            data: Input data

        Returns:
            Any: Search key value
        """
        if isinstance(data, dict):
            if cls.search_key in data:
                return data[cls.search_key]

            for fallback in [
                "question",
                "query",
                "key",
                "input",
                "observation",
            ]:
                if fallback in data:
                    return data[fallback]
            return str(data)
        else:
            raise ValueError(
                f"Invalid data type for building search key: {type(data)}"
            )

    @classmethod
    def _extract_value(cls, data: Any) -> Any:
        """
        Extract value from input data.

        Args:
            data: Input data

        Returns:
            Any: Value to store
        """
        if isinstance(data, dict):
            if cls.value_key in data:
                return data[cls.value_key]
            for fallback in [
                "answer",
                "response",
                "value",
                "output",
                "result",
                "action",
            ]:
                if fallback in data:
                    return data[fallback]
            return data
        else:
            raise ValueError(
                f"Invalid data type for building value: {type(data)}"
            )

    @classmethod
    def _build_additional_fields(cls, data: Any) -> Dict[str, Any]:
        """
        Build case-specific additional fields.

        Args:
            data: Input data

        Returns:
            Dict[str, Any]: Additional fields (timestamp, metadata)
        """
        additional = {"timestamp": time.time(), "metadata": {}}

        if isinstance(data, dict):
            if "timestamp" in data:
                additional["timestamp"] = data["timestamp"]

            # Build metadata from remaining fields
            metadata = {}
            reserved_fields = {
                cls.uid_key,
                cls.search_key,
                cls.value_key,
                "timestamp",
                "metadata",
            }

            for key, value in data.items():
                if (
                    key not in reserved_fields
                    and key not in cls.block_fields
                ):
                    metadata[key] = value

            # Merge with any explicitly provided metadata
            if "metadata" in data and isinstance(data["metadata"], dict):
                metadata.update(data["metadata"])

            additional["metadata"] = metadata
        else:
            raise ValueError(
                f"Invalid data type for building additional fields: {type(data)}"
            )

        return additional

    def format_message(self) -> str:
        """
        Format the case memory entry as a text message for agents.

        Returns:
            str: Formatted message for agent consumption
        """
        message_parts = []

        if self.key and self.value:
            message_parts.append(f"Observation: {self.key}")
            message_parts.append(f"Action: {self.value}")
        elif "text_key" in self.metadata and self.value:
            message_parts.append(
                f"Observation: {self.metadata['text_key']}"
            )
            message_parts.append(f"Action: {self.value}")

        return (
            "|".join(message_parts)
            if len(message_parts) > 0
            else f"Memory Entry (uid: {self.uid[:8]}...)"
        )


@dataclass
class EmbeddingCaseMemoryType(CaseMemoryType):
    """
    Case memory that uses pre-computed embeddings for search.
    """

    fields: ClassVar[Set[str]] = {
        "uid",
        "key",
        "value",
        "embedding",
        "timestamp",
        "metadata",
    }
    search_key: ClassVar[str] = "embedding"

    embedding: Any = None

    def __post_init__(self):
        """Ensure embedding is on CPU if it's a tensor."""
        super().__post_init__()
        if isinstance(self.embedding, torch.Tensor):
            # Always store on CPU to avoid cross-device issues
            self.embedding = self.embedding.cpu()

    @classmethod
    def _extract_search_key(cls, data: Any) -> Any:
        """
        Extract embedding from input data, ensuring it's on CPU.

        Args:
            data: Input data

        Returns:
            Any: Embedding tensor on CPU, or text key if embedding not present
        """
        if isinstance(data, dict):
            if "embedding" in data:
                embedding = data["embedding"]
                if isinstance(embedding, torch.Tensor):
                    # Move to CPU immediately when extracted
                    return embedding.cpu()
                elif isinstance(embedding, list):
                    # Convert from list (loaded from JSON) to tensor on CPU
                    return torch.tensor(embedding, dtype=torch.float32)
                return embedding

            for fallback in [
                "key",
                "question",
                "query",
                "input",
                "observation",
            ]:
                if fallback in data:
                    return data[fallback]
            return str(data)
        else:
            raise ValueError(
                f"Invalid data type for building search key: {type(data)}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert instance to dictionary, handling torch tensors and numpy arrays properly.
        """
        result = {}
        for field_name in self.fields:
            if hasattr(self, field_name):
                value = getattr(self, field_name)

                if isinstance(value, torch.Tensor):
                    result[field_name] = value.cpu().tolist()
                elif isinstance(value, np.ndarray):
                    result[field_name] = value.tolist()
                elif isinstance(value, (np.integer, np.floating)):
                    result[field_name] = value.item()
                else:
                    result[field_name] = value

        return result


@dataclass
class EmbeddingTrajectoryMemoryType(CaseMemoryType):
    block_fields: ClassVar[Set[str]] = {
        "last_turn_uids",
        "messages",
        "task_goal",
        "outcome",
        "reward",
    }  # the fields blocked from being added to the additional fields

    def format_message(self) -> str:
        return self.value
