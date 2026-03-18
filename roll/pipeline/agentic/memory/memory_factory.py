import time
from typing import Any, Optional, Union

from roll.distributed.scheduler.resource_manager import ResourceManager
from roll.pipeline.agentic.memory.async_memory_manager import \
    AsyncMemoryManager
from roll.pipeline.agentic.memory.memory_config import MemoryConfig
from roll.pipeline.agentic.memory.memory_manager import MemoryManager
from roll.utils.logging import get_logger

logger = get_logger()


def create_memory_manager(
    memory_config: MemoryConfig, 
    mode: str = "train", 
    async_enabled: Optional[bool] = None,
    resource_manager: Optional[ResourceManager] = None,
    memory_model_cluster: Optional[Any] = None,
) -> Union[MemoryManager, 'AsyncMemoryManager']:
    """
    Factory function to create memory managers with optional async capabilities.
    
    Args:
        memory_config: Memory configuration
        mode: Operating mode ("train" or "eval")
        async_enabled: If True, creates async memory manager. If None, auto-detects based on environment
        resource_manager: Resource manager for cluster management
        memory_model_cluster: Optional externally managed memory model cluster (for training)
    
    Returns:
        Either a MemoryManager or AsyncMemoryManager
    """
    if async_enabled is None:
        async_enabled = _should_use_async_memory()
    
    if async_enabled:
        try:
            from roll.pipeline.agentic.memory.async_memory_manager import \
                AsyncMemoryManagerClient
            logger.info("Creating async memory manager")
            return AsyncMemoryManagerClient(memory_config, mode, resource_manager, memory_model_cluster)
        except ImportError as e:
            logger.warning(f"Failed to import async memory manager, falling back to memory manager: {e}")
            async_enabled = False
    
    if not async_enabled:
        logger.info("Creating memory manager")
        return MemoryManager(memory_config, mode)


def _should_use_async_memory() -> bool:
    try:
        import ray
        if not ray.is_initialized():
            logger.debug("Ray not initialized, using memory manager")
            return False
    except ImportError:
        logger.debug("Ray not available, using memory manager")
        return False
    
    return True


class MemoryManagerBuilder:
    """
    Builder pattern for creating memory managers with fluent interface.
    """
    
    def __init__(self):
        self.memory_config = None
        self.mode = "train"
        self.async_enabled = None
        self.performance_monitoring = False
    
    def with_config(self, memory_config: MemoryConfig) -> 'MemoryManagerBuilder':
        """Set memory configuration"""
        self.memory_config = memory_config
        return self
    
    def with_mode(self, mode: str) -> 'MemoryManagerBuilder':
        """Set operating mode"""
        self.mode = mode
        return self
    
    def with_async(self, enabled: bool = True) -> 'MemoryManagerBuilder':
        """Enable or disable async capabilities"""
        self.async_enabled = enabled
        return self
    
    def with_performance_monitoring(self, enabled: bool = True) -> 'MemoryManagerBuilder':
        """Enable performance monitoring"""
        self.performance_monitoring = enabled
        return self
    
    def build(self) -> Union[MemoryManager, 'AsyncMemoryManager']:
        """Build the memory manager"""
        if self.memory_config is None:
            raise ValueError("Memory configuration is required")
        
        memory_manager = create_memory_manager(
            self.memory_config,
            self.mode,
            self.async_enabled,
        )
        
        if self.performance_monitoring:
            memory_manager = _wrap_with_performance_monitoring(memory_manager)
        
        return memory_manager


def _wrap_with_performance_monitoring(memory_manager):
    """Wrap memory manager with performance monitoring"""
    
    class PerformanceMonitoredMemoryManager:
        def __init__(self, wrapped_manager):
            self.wrapped_manager = wrapped_manager
            self.operation_counts = {
                "searches": 0,
                "updates": 0,
                "errors": 0
            }
            self.total_times = {
                "search_time": 0.0,
                "update_time": 0.0
            }
        
        def search(self, query: str) -> str:
            start_time = time.time()
            try:
                result = self.wrapped_manager.search(query)
                self.operation_counts["searches"] += 1
                self.total_times["search_time"] += time.time() - start_time
                return result
            except Exception as e:
                self.operation_counts["errors"] += 1
                raise
        
        def update(self, data, mode=None):
            start_time = time.time()
            try:
                result = self.wrapped_manager.update(data, mode)
                self.operation_counts["updates"] += 1
                self.total_times["update_time"] += time.time() - start_time
                return result
            except Exception as e:
                self.operation_counts["errors"] += 1
                raise
        
        def get_performance_summary(self) -> dict:
            """Get performance summary"""
            summary = {
                "operation_counts": self.operation_counts.copy(),
                "total_times": self.total_times.copy(),
                "average_times": {}
            }
            
            if self.operation_counts["searches"] > 0:
                summary["average_times"]["avg_search_time"] = (
                    self.total_times["search_time"] / self.operation_counts["searches"]
                )
            
            if self.operation_counts["updates"] > 0:
                summary["average_times"]["avg_update_time"] = (
                    self.total_times["update_time"] / self.operation_counts["updates"]
                )
            
            # Include underlying stats if available
            if hasattr(self.wrapped_manager, 'get_performance_stats'):
                summary["underlying_stats"] = self.wrapped_manager.get_performance_stats()
            
            return summary
        
        def __getattr__(self, name):
            # Delegate all other attributes to the wrapped manager
            return getattr(self.wrapped_manager, name)
    
    return PerformanceMonitoredMemoryManager(memory_manager)


def create_high_performance_memory_manager(memory_config: MemoryConfig, mode: str = "train") -> Union[MemoryManager, 'AsyncMemoryManager']:
    """Create a memory manager optimized for high performance scenarios"""
    return (MemoryManagerBuilder()
            .with_config(memory_config)
            .with_mode(mode)
            .with_async(True)
            .with_performance_monitoring(True)
            .build())


def create_compatible_memory_manager(memory_config: MemoryConfig, mode: str = "train") -> MemoryManager:
    """Create a traditional memory manager for maximum compatibility"""
    return (MemoryManagerBuilder()
            .with_config(memory_config)
            .with_mode(mode)
            .with_async(False)
            .build())