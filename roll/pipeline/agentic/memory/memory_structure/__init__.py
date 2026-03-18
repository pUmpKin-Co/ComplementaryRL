from roll.pipeline.agentic.memory.memory_config import MemoryStructure
from roll.pipeline.agentic.memory.memory_structure.base_memory import (
    BaseMemoryStructure,
)
from roll.pipeline.agentic.memory.memory_structure.tabuler_memory import (
    MultiTaskTabulerMemory,
    TabulerMemory,
)

memory_structure_dict = {
    MemoryStructure.tabuler: TabulerMemory,
    MemoryStructure.multi_task_tabuler: MultiTaskTabulerMemory,
}
