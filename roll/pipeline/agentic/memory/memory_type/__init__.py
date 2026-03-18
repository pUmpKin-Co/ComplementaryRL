from roll.pipeline.agentic.memory.memory_config import MemoryType
from roll.pipeline.agentic.memory.memory_type.case_base import (
    CaseMemoryType, EmbeddingCaseMemoryType, EmbeddingTrajectoryMemoryType)

memory_type_dict = {
    MemoryType.case_memory: CaseMemoryType,
    MemoryType.case_embedding_memory: EmbeddingCaseMemoryType,
    MemoryType.trajectory_memory: EmbeddingTrajectoryMemoryType,
}