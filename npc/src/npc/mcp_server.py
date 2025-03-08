"""MCP server for NPC agent management"""

from dataclasses import dataclass, field
from typing import Dict
import json

from fastmcp import FastMCP, Context

from npc.agent.agent import Agent, AgentLLMConfig
from npc.apis.llm_client import Model
from npc.project_config import OPENROUTER_API_KEY


@dataclass
class AgentConfig:
    """Configuration for an NPC agent"""
    traits: list[str]
    initial_working_memory: str = ""
    initial_long_term_memories: list[str] = field(default_factory=list)


class AgentManager:
    """Manages NPC agent instances"""
    
    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        
        # Default LLM config using latest models
        self.default_llm_config = AgentLLMConfig(
            small_llm=Model.HAIKU,
            large_llm=Model.SONNET,
        )
        
    def create_agent(self, agent_id: str, config: AgentConfig) -> str:
        """Create and store new agent"""
        self.agents[agent_id] = Agent(
            llm_config=self.default_llm_config,
            initial_working_memory=config.initial_working_memory,
            initial_long_term_memories=config.initial_long_term_memories,
            personality_traits=config.traits,
        )
        return agent_id
        
    def get_agent(self, agent_id: str) -> Agent:
        """Get agent by ID"""
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")
        return self.agents[agent_id]
        
    def cleanup_agent(self, agent_id: str) -> None:
        """Remove agent and cleanup resources"""
        if agent_id in self.agents:
            del self.agents[agent_id]


# Create MCP server and agent manager
mcp = FastMCP("NPC Manager")
agent_manager = AgentManager()


@mcp.tool()
async def create_agent(
    agent_id: str,
    config: dict,
    ctx: Context = None,
) -> dict:
    """Create a new NPC agent
    
    Args:
        agent_id: Unique identifier for the agent
        config: Configuration object containing:
            - traits: list[str], Basic personality traits
            - initial_working_memory: str, Initial working memory state
            - initial_long_term_memories: list[str], Initial long-term memories
    """
    try:
        agent_config = AgentConfig(
            traits=config["traits"],
            initial_working_memory=config.get("initial_working_memory", ""),
            initial_long_term_memories=config.get("initial_long_term_memories", []),
        )
        
        agent_id = agent_manager.create_agent(agent_id, agent_config)
        return {
            "status": "created",
            "agent_id": agent_id
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


@mcp.tool()
async def process_observation(
    agent_id: str,
    observation: str,
    available_actions: list[dict],
    ctx: Context = None,
) -> dict:
    """Process observation and return chosen action
    
    Args:
        agent_id: Agent to process observation for
        observation: Current state/situation in natural language
        available_actions: List of possible actions, each containing:
            - name: str, Action identifier
            - description: str, Human readable description
            - parameters: dict, Required parameters and their descriptions
    """
    try:
        agent = agent_manager.get_agent(agent_id)
        
        # Format available actions for agent
        action_descriptions = {
            i: f"{action['name']}: {action['description']}"
            for i, action in enumerate(available_actions)
        }
        
        # Let agent process observation and choose action
        action_index = agent.process_observation(
            observation=observation,
            available_actions=action_descriptions
        )
        
        # Get chosen action details
        chosen_action = available_actions[action_index]
        
        return {
            "action": chosen_action["name"],
            "parameters": {}  # Parameters would be filled in by agent if needed
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


@mcp.tool()
async def cleanup_agent(
    agent_id: str,
    ctx: Context = None,
) -> dict:
    """Gracefully cleanup and remove an agent
    
    Args:
        agent_id: Agent to remove
    """
    try:
        agent_manager.cleanup_agent(agent_id)
        return {
            "status": "removed",
            "agent_id": agent_id
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


@mcp.resource("agent://{agent_id}/info")
async def get_agent_info(agent_id: str) -> str:
    """Get basic information about an agent"""
    try:
        agent = agent_manager.get_agent(agent_id)
        return json.dumps({
            "status": "active",
            "traits": agent.personality_traits,
            "working_memory": agent.working_memory
        })
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": str(e)
        })


if __name__ == "__main__":
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY must be set in project_config.py")
    mcp.run()
