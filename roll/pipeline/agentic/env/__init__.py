"""
base agentic codes reference: https://github.com/RAGEN-AI/RAGEN
"""

import gem

from roll.utils.logging import get_logger

logger = get_logger()

gem.register("sokoban", entry_point="roll.pipeline.agentic.env.sokoban:SokobanEnv")
gem.register("frozen_lake", entry_point="roll.pipeline.agentic.env.frozen_lake:FrozenLakeEnv")
gem.register("swe_env", entry_point="roll.pipeline.agentic.env.swe_env.env:SWEEnv")
gem.register("sokoban_mcp", entry_point="roll.pipeline.agentic.env.mcp:SokobanMCPEnv")
gem.register("cli", entry_point="roll.pipeline.agentic.env.cli_env.env:CLIEnv")
gem.register("alfworld", entry_point="roll.pipeline.agentic.env.alfworld.env:AlfworldGemEnv")
gem.register("roll_math", entry_point="roll.pipeline.agentic.env.gem.math_env:MathEnv")
gem.register("minihack", entry_point="roll.pipeline.agentic.env.minihack.env:MiniHackEnv")
gem.register(
    "tb_native_env", entry_point="roll.pipeline.agentic.env.terminal_env.terminal_native_env:TerminalNativeEnv"
)
gem.register(
    "rock_tb_native_env", entry_point="roll.pipeline.agentic.env.terminal_env.rock_tb_native_env:RockTBNativeEnv"
)
gem.register("tb_env", entry_point="roll.pipeline.agentic.env.terminal_env.terminal_env:TerminalBenchEnv")

try:
    import os
    import sys

    current_dir = os.path.dirname(os.path.abspath(__file__))
    relative_path = "../../../../third_party/webshop-minimal"
    module_path = os.path.join(current_dir, relative_path)
    sys.path.append(module_path)
    gem.register("webshop", entry_point="roll.pipeline.agentic.env.webshop.env:WebShopEnv")

except Exception as e:
    logger.info(f"Failed to import webshop: {e}")
