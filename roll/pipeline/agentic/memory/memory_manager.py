from typing import Any, Optional

from roll.pipeline.agentic.memory.memory_config import MemoryConfig
from roll.pipeline.agentic.memory.memory_structure import memory_structure_dict
from roll.pipeline.agentic.memory.searcher import memory_searcher_dict
from roll.pipeline.agentic.memory.updater import memory_updater_dict


class MemoryManager:
    def __init__(self, memory_config: MemoryConfig, mode: str = "train"):
        self.memory_config = memory_config
        self.memory = memory_structure_dict[memory_config.memory_structure](memory_config)
        self.memory_updater = memory_updater_dict[memory_config.memory_update_strategy](memory_config)
        self.memory_searcher = memory_searcher_dict[memory_config.memory_search_strategy](memory_config)
        self.mode = mode
    
    def search(self, query: str):
        message, uids = self.memory_searcher.search(query, self.memory)
        if hasattr(self.memory_updater, "update_access_times") and uids and self.mode == "train":
            self.memory_updater.update_access_times(uids)
        return message
    
    def update(self, data: Any, mode: Optional[str] = None):
        if mode is None:
            mode = self.mode
            
        if mode == "train":
            self.memory_updater.update(data, self.memory)
    
    @property
    def memory_size(self):
        return len(self.memory)
    
    @property
    def memory_key_field(self):
        return self.memory.memory_key_field
    
    @property
    def memory_value_field(self):
        return self.memory.memory_value_field
    
    @property
    def memory_uid_field(self):
        return self.memory.memory_uid_field