from roll.pipeline.agentic.memory.memory_config import UpdateStrategy
from roll.pipeline.agentic.memory.updater.base_updater import (FIFOUpdater,
                                                               LRUUpdater,
                                                               RandomUpdater)

memory_updater_dict = {
    UpdateStrategy.random: RandomUpdater,
    UpdateStrategy.fifo: FIFOUpdater,
    UpdateStrategy.lru: LRUUpdater,
}   