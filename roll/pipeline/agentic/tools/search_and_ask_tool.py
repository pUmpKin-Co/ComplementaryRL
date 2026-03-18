import re
from typing import TYPE_CHECKING, List, Optional

from gem.tools.base_tool import BaseTool

from roll.utils.logging import get_logger

logger = get_logger()

if TYPE_CHECKING:
    from roll.pipeline.agentic.memory.memory_manager import MemoryManager


class KnowledgeSearchAndAsk(BaseTool):
    def __init__(
        self,
        return_interaction: Optional[bool] = False,
        memory_manager: Optional["MemoryManager"] = None,
        tool_instruction: Optional[str] = None,
        patterns: Optional[List[str]] = None,
        max_search_results: Optional[int] = 2,
        env_name: Optional[str] = None,
    ):
        self.return_interaction = return_interaction
        self.memory_manager = memory_manager
        self.env_name = env_name
        self.patterns = [r"<knowledge_search_and_ask>(.*?)</knowledge_search_and_ask>"]

        self.tool_instruction = (
            "You have access to a world knowledge base that can provide critical information when you're uncertain or stuck. "
            "To use this resource effectively, you must formulate detailed, context-rich queries that paint a complete picture of your situation.\n\n"
            "**WHEN TO USE THE KNOWLEDGE BASE:**\n"
            "- When you are uncertain about the rules, mechanics, or goals of the environment.\n"
            "- When you have tried several approaches and are consistently failing.\n"
            "- When you need to verify your understanding of an object, entity, or your own state.\n"
            "- When you suspect there is a hidden mechanic or a specific sequence of actions required.\n\n"
            "**HOW TO STRUCTURE AN EFFECTIVE QUERY:**\n"
            "Your query must be a self-contained summary that allows an external expert to understand your precise dilemma. Structure it around these core questions:\n\n"
            "1.  **What is my current observable state?**\n"
            "    - Describe the environment around you. What do you perceive?\n"
            "    - **What objects, entities, or key elements are present?** Describe their properties, states (e.g., locked, active), and, if relevant, their spatial relationships.\n"
            "    - **What is my own status?** (e.g., active effects, position).\n"
            "    - **What is the apparent immediate goal?**\n\n"
            "2.  **What is the specific problem or barrier?**\n"
            "    - What is preventing you from making progress?\n"
            "    - What are you confused or uncertain about?\n\n"
            "3.  **What have I already tried, and what was the outcome?**\n"
            "    - Summarize your recent actions and the results. This prevents suggesting solutions you've already tested.\n\n"
            "4.  **What specific information do I need?**\n"
            "    - Frame a clear, direct question. What exact piece of knowledge would help you overcome the current barrier?\n\n"
            "**USAGE LIMITATIONS:**\n"
            f"- You have limited access to this tool - you can use it only {max_search_results} times at most. Thereafter, "
            "you will no longer be able to access the knowledge base and must rely solely on your own reasoning.\n"
            "- Use your searches **strategically** - each query should address a critical blocking point in your progress.\n"
            "- Prioritize your most important uncertainties first, as you cannot use this tool indefinitely.\n\n"
            "**FINAL INSTRUCTIONS:**\n"
            "- **Always** wrap your complete, structured query in `<knowledge_search_and_ask>...</knowledge_search_and_ask>` tags.\n"
            "- The quality of the answer you receive depends entirely on the quality of the question you ask. Be specific and detailed.\n"
            "- **Budget your searches wisely** - you only have a limited number of attempts.\n"
        )

        if tool_instruction:
            self.tool_instruction = tool_instruction

        if patterns:
            self.patterns = patterns

    def _parse_action(self, action: str) -> tuple[Optional[str], str, bool]:
        parsed_query = None
        parsed_action = action
        is_valid = False
        prev_end = len(action)
        for pattern in self.patterns:
            matches = re.search(pattern, action, re.DOTALL)
            if matches:
                is_valid = True
                if matches.end() <= prev_end:
                    parsed_query = matches.group(1).strip()
                    parsed_action = action[: matches.end()]
                    prev_end = matches.end()
        return parsed_query, parsed_action, is_valid

    def instruction_string(self) -> str:
        return self.tool_instruction

    def execute_action(self, action: str) -> tuple[bool, bool, str, str]:
        self.triggered_interactions = []
        parsed_query, parsed_action, is_valid = self._parse_action(action)
        if not is_valid:
            observation = ""
            has_error = True
        else:
            search_results, interaction_triggered = self.memory_manager.search_and_ask(
                parsed_query, self.return_interaction, self.env_name
            )
            self.triggered_interactions = interaction_triggered
            if len(search_results) == 0:
                has_error = True
            else:
                has_error = False

            observation = f"Search results: {search_results}\n"

        return is_valid, has_error, observation, parsed_action
