import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, Set

import numpy as np
import torch

from roll.pipeline.agentic.memory.memory_config import MemoryConfig


@dataclass
class BaseMemoryType(ABC):
    """
    Base class for memory types that define entry format and field mappings.
    
    Usage:
        # Get fields without initialization
        fields = CaseMemoryType.fields
        
        # Build entry - returns MemoryType instance
        entry = CaseMemoryType.build_entry(data)
        
        # Use the entry
        message = entry.format_message()  # For agent
        print(entry)  # Beautiful printing
    """
    
    # Class-level field definitions (available before initialization)
    fields: ClassVar[Set[str]] = {"uid", "key", "value"}
    uid_key: ClassVar[str] = "uid"
    search_key: ClassVar[str] = "key" 
    value_key: ClassVar[str] = "value"
    block_fields: ClassVar[Set[str]] = {"last_turn_uids"} # the fields blocked from being added to the additional fields
    
    # Instance fields (will be set by build_entry)
    uid: str = ""
    key: Any = None
    value: Any = None
    
    @classmethod
    def get_fields(cls) -> Set[str]:
        """
        Get all fields required for this memory type.
        Available without initialization.
        
        Returns:
            Set[str]: All required fields
        """
        return cls.fields.copy()
    
    @classmethod
    def get_field_mapping(cls) -> Dict[str, str]:
        """
        Get the field mapping for this memory type.
        Available without initialization.
        
        Returns:
            Dict[str, str]: Mapping of logical names to actual field names
        """
        return {
            "uid_key": cls.uid_key,
            "search_key": cls.search_key,
            "value_key": cls.value_key
        }
    
    @classmethod
    def validate_entry(cls, entry: Dict[str, Any]) -> None:
        """
        Validate that an entry contains all required fields.
        
        Args:
            entry: Entry to validate
            
        Raises:
            ValueError: If required fields are missing
        """
        if not isinstance(entry, dict):
            raise ValueError(f"Entry must be a dictionary, got {type(entry)}")
        
        missing_fields = cls.fields - set(entry.keys())
        if missing_fields:
            raise ValueError(f"Entry missing required fields: {missing_fields}")
    
    @classmethod
    def build_entry(
        cls, 
        data: Any, 
        uid: str = None,
        uid_key: str = None,
        search_key: str = None,
        value_key: str = None
    ) -> 'BaseMemoryType':
        """
        Build a memory entry with the required format from the provided data.
        Returns an instance of the MemoryType class.
        
        Args:
            data: Input data to build the entry from
            uid: Optional uid, if not provided will generate one
            uid_key: Optional custom uid field name
            search_key: Optional custom search field name  
            value_key: Optional custom value field name
            
        Returns:
            BaseMemoryType: Memory entry instance with format_message() and print() support
        """
        # Generate uid if not provided
        if uid is None:
            uid = str(uuid.uuid4())
        
        # Extract core values
        search_value = cls._extract_search_key(data)
        value_data = cls._extract_value(data)
        
        # Build additional fields
        additional_fields = cls._build_additional_fields(data)
        
        # Create instance with all field values
        entry_data = {
            cls.uid_key: uid,
            cls.search_key: search_value, 
            cls.value_key: value_data,
            **additional_fields
        }
        
        # Create and return instance
        instance = cls(**entry_data)
        return instance
    
    @classmethod
    @abstractmethod
    def _extract_search_key(cls, data: Any) -> Any:
        """
        Extract the search key from input data.
        
        Args:
            data: Input data
            
        Returns:
            Any: Search key value
        """
        pass
    
    @classmethod
    @abstractmethod 
    def _extract_value(cls, data: Any) -> Any:
        """
        Extract the value from input data.
        
        Args:
            data: Input data
            
        Returns:
            Any: Value to store
        """
        pass
    
    @classmethod
    def _build_additional_fields(cls, data: Any) -> Dict[str, Any]:
        """
        Build additional fields for the entry.
        Can be overridden by subclasses to add custom fields.
        
        Args:
            data: Input data
            
        Returns:
            Dict[str, Any]: Additional fields
        """
        return {}
    
    def __str__(self) -> str:
        """Beautiful string representation for printing."""
        return self.format_message()
    
    def __repr__(self) -> str:
        """Developer-friendly representation."""
        class_name = self.__class__.__name__
        uid_short = self.uid[:8] if len(self.uid) > 8 else self.uid
        return f"{class_name}(uid='{uid_short}...', key='{str(self.key)[:30]}...', value='{str(self.value)[:30]}...')"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert instance back to dictionary format, handling torch tensors and numpy arrays."""
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
    
    @abstractmethod
    def format_message(self) -> str:
        """
        Format the memory entry as a text message for agents.
        
        Returns:
            str: Formatted message for agent consumption
        """
        pass