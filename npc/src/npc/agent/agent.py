"""Agent implementation for NPCs"""

import json
import logging
import textwrap
from dataclasses import dataclass

from npc.apis.llm_client import LLMFunction, Model
from npc.prompts import AgentPrompt

from .memory_database import MemoryDatabase


@dataclass
class AgentLLMConfig:
    small_llm: Model
    large_llm: Model


class Agent:
    """An NPC agent that can process observations and choose actions"""
    
    def __init__(
            self, 
            llm_config: AgentLLMConfig,
            initial_working_memory: str = "",
            initial_long_term_memories: list[str] = [],
            personality_traits: list[str] = [],
        ):
        self.working_memory = initial_working_memory
        self.long_term_memory = MemoryDatabase(initial_long_term_memories)
        self.personality_traits = personality_traits
        
        # Current observation state
        self.current_observation = None
        self.current_actions = None

        # LLM functions for memory and decision making
        self.query_generator = LLMFunction(AgentPrompt.MEMORY_QUERIES, llm_config.small_llm)
        self.memory_report_generator = LLMFunction(AgentPrompt.MEMORY_REPORT, llm_config.small_llm)
        self.working_memory_generator = LLMFunction(AgentPrompt.WORKING_MEMORY, llm_config.small_llm)
        self.long_term_memory_generator = LLMFunction(AgentPrompt.LONG_TERM_MEMORY, llm_config.small_llm)
        self.action_decision_generator = LLMFunction(AgentPrompt.ACTION_DECISION, llm_config.small_llm)

    def process_observation(self, observation: str, available_actions: dict[int, str]) -> int:
        """Process an observation and choose an action
        
        Args:
            observation: Natural language description of current state
            available_actions: Dict mapping action indices to descriptions
            
        Returns:
            int: Index of chosen action
        """
        # Update current state
        logging.info(f"Processing observation: {observation}")
        logging.info(f"Available actions: {available_actions}")
        
        self.current_observation = observation
        self.current_actions = available_actions
        
        self.update_working_memory()
        self.update_long_term_memory()
        
        # Choose and return action
        return self.choose_action()

    def update_working_memory(self) -> None:
        """Update working memory based on current observation"""
        logging.info("Updating working memory...")

        # Formulate queries and retrieve from long-term memory
        query_response = self.query_generator.generate(
            working_memory=self.working_memory,
            observation=self.current_observation,
        )
        logging.debug(f"Query response: {query_response['queries']}")
        retrieved_memories = []
        for query in query_response.get("queries", []):
            query_memories = self.long_term_memory.retrieve(query)
            if query_memories:
                retrieved_memories.append(query_memories[0])
        logging.debug(f"Retrieved memories: {retrieved_memories}")

        # Draft memory report based on working memory and retrieved memories
        memory_report_response = self.memory_report_generator.generate(
            working_memory=self.working_memory,
            observation=self.current_observation,
            retrieved_memories=retrieved_memories,
        )
        logging.debug(f"Memory report: {memory_report_response['memory_report']}")

        # Update working memory based on memory report
        working_memory_response = self.working_memory_generator.generate(
            working_memory=self.working_memory,
            memory_report=memory_report_response["memory_report"],
        )
        logging.debug(f"Working memory update response: {working_memory_response['response']}")
        if working_memory_response["updated_working_memory"]:
            self.working_memory = working_memory_response["updated_working_memory"]
            logging.info(f"Updated working memory: {self.working_memory}")

    def update_long_term_memory(self) -> None:
        """Update long-term memory based on current observation"""
        logging.info("Updating long term memory...")

        memory_update_response = self.long_term_memory_generator.generate(
            working_memory=self.working_memory,
            observation=self.current_observation,
        )

        logging.debug(f"Memory update response: {memory_update_response['response']}")

        for memory in memory_update_response["memories"]:
            self.long_term_memory.add_memories([memory])

    def choose_action(self) -> int:
        """Choose next action based on current state
        
        Returns:
            int: Index of chosen action
        """
        # Format available actions for LLM
        actions_str = "\n".join([
            f"- Action {idx}: {desc}"
            for idx, desc in self.current_actions.items()
        ])
        
        # Get action decision
        next_action = self.action_decision_generator.generate(
            working_memory=self.working_memory,
            available_actions=actions_str,
            personality_traits=", ".join(self.personality_traits)
        )
        
        logging.debug(f"Action decision response: {next_action['response']}")
        logging.info(f"Action decision: {next_action['action_decision']}")
        
        # Extract chosen action index from action_decision JSON
        try:
            action_decision = json.loads(next_action["action_decision"])
            return int(action_decision["action_index"])
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            # Default to first action if parsing fails
            logging.error(f"Error parsing action index: {e}")
            return list(self.current_actions.keys())[0]
