"""MCP server for NPC agent management"""

import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Dict

from fastmcp import FastMCP, Context
from mcp.server import Server as CoreMCPServer  # Renamed to avoid conflict
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.routing import Mount, Route
import uvicorn

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
        
    def create_agent(self, agent_id: str, config: AgentConfig):
        """Create and store new agent"""
        self.agents[agent_id] = Agent(
            llm_config=self.default_llm_config,
            initial_working_memory=config.initial_working_memory,
            initial_long_term_memories=config.initial_long_term_memories,
            personality_traits=config.traits,
        )
        
    def get_agent(self, agent_id: str) -> Agent:
        """Get agent by ID"""
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")
        return self.agents[agent_id]
        
    def cleanup_agent(self, agent_id: str) -> None:
        """Remove agent and cleanup resources"""
        if agent_id in self.agents:
            del self.agents[agent_id]


class MCPServer:
    """NPC MCP Server"""
    
    def __init__(self, name="NPC Manager"):
        """Initialize the MCP server with agent manager"""
        self.agent_manager = AgentManager()
        
        # Create MCP server with default settings
        # FastMCP by default binds to 0.0.0.0 which accepts connections from any IP
        self.mcp = FastMCP(name)
        
        self._register_tools_and_resources()
    
    def _register_tools_and_resources(self):
        """Register all tools and resources with MCP"""
        # Register tools
        @self.mcp.tool()
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
                
                self.agent_manager.create_agent(agent_id, agent_config)
                return {
                    "status": "created",
                    "agent_id": agent_id
                }
            except Exception as e:
                return {
                    "status": "error",
                    "message": str(e)
                }

        @self.mcp.tool()
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
                agent = self.agent_manager.get_agent(agent_id)
                
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

        @self.mcp.tool()
        async def cleanup_agent(
            agent_id: str,
            ctx: Context = None,
        ) -> dict:
            """Gracefully cleanup and remove an agent
            
            Args:
                agent_id: Agent to remove
            """
            try:
                self.agent_manager.cleanup_agent(agent_id)
                return {
                    "status": "removed",
                    "agent_id": agent_id
                }
            except Exception as e:
                return {
                    "status": "error",
                    "message": str(e)
                }

        # Register resources
        @self.mcp.resource("agent://{agent_id}/info")
        async def get_agent_info(agent_id: str) -> str:
            """Get basic information about an agent"""
            try:
                agent = self.agent_manager.get_agent(agent_id)
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


def create_starlette_app(mcp_server: CoreMCPServer, *, debug: bool = False) -> Starlette:
    """Create a Starlette application that can serve the provided mcp server with SSE."""
    # Use the same path for both SSE connection and message handling
    # This matches what the client is expecting
    sse = SseServerTransport("/sse/")

    async def handle_sse(request: Request) -> None:
        async with sse.connect_sse(
            request.scope,
            request.receive,
            request._send,  # noqa: SLF001
        ) as (read_stream, write_stream):
            await mcp_server.run(
                read_stream,
                write_stream,
                mcp_server.create_initialization_options(),
            )

    return Starlette(
        debug=debug,
        routes=[
            Route("/sse", endpoint=handle_sse),
            # Mount message handler at the same base path
            Mount("/sse/", app=sse.handle_post_message),
        ],
    )


def main():
    """Run the MCP server from command line"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="NPC Agent MCP Server")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                      help="Host address to bind the server to (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000,
                      help="Port to run the server on (default: 8000)")
    args = parser.parse_args()
    
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY must be set in project_config.py")
        
    # Create the server instance
    server = MCPServer("NPC Manager")
    
    # Extract the underlying core MCP server (directly accessing FastMCP's _mcp_server)
    mcp_server = server.mcp._mcp_server  # noqa: WPS437
    
    # Create the Starlette app with debug=True
    starlette_app = create_starlette_app(mcp_server, debug=True)
    
    print(f"Starting NPC MCP server on http://{args.host}:{args.port}/sse")
    
    # Run using uvicorn
    uvicorn.run(starlette_app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
