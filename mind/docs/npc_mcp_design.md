# NPC Management System Design

## Overview

This document outlines a design for managing NPCs through the Model Context Protocol (MCP), allowing remote clients (like a Godot game) to interact with NPC agents in a standardized way.

## Core Design Principles

1. Encapsulation
   - Memory systems (working/long-term) are managed internally
   - External interface exposes necessary operations and state
   - LLM integration details handled by agent implementation

2. Standardization
   - Resources provide read-only access to agent information
   - Tools handle all state-changing operations
   - Clear, documented interfaces for client integration

3. Separation of Concerns
   - MCP server handles agent coordination
   - Agents handle memory management and decision making
   - Clients focus on environment simulation

## MCP Server Design

### Tools

```python
@mcp.tool()
async def create_agent(
    agent_id: str,
    config: dict,
) -> dict:
    """Create a new NPC agent
    
    Args:
        agent_id: Unique identifier for the agent
        config: Configuration object containing:
            - traits: list[str], Basic personality traits
            - initial_working_memory: str, Initial working memory state
            - initial_long_term_memories: list[str], Initial long-term memories
    """
    return {
        "status": "created",
        "agent_id": agent_id
    }

@mcp.tool()
async def process_observation(
    agent_id: str,
    observation: str,
    available_actions: list[dict],
) -> dict:
    """Process observation and return chosen action
    
    Args:
        agent_id: Agent to process observation for
        observation: Current state/situation in natural language
        available_actions: List of possible actions, each containing:
            - name: str, Action identifier
            - description: str, Human readable description
            - parameters: dict, Required parameters and their descriptions
    
    Returns:
        dict: {
            "action": str,  # Name of chosen action
            "parameters": dict  # Parameters for the action
        }
    
    Example:
        available_actions = [
            {
                "name": "examine_chest",
                "description": "Look more closely at the chest",
                "parameters": {}
            },
            {
                "name": "open_door", 
                "description": "Open the door and go north",
                "parameters": {}
            }
        ]
    """
    return {
        "action": "chosen_action",
        "parameters": {}
    }

@mcp.tool()
async def cleanup_agent(
    agent_id: str,
) -> dict:
    """Gracefully cleanup and remove an agent
    
    Args:
        agent_id: Agent to remove
    """
    return {
        "status": "removed",
        "agent_id": agent_id
    }
```

### Resources

```python
@mcp.resource("agent://{agent_id}/info")
async def get_agent_info(agent_id: str) -> str:
    """Get basic information about an agent
    
    Returns JSON with:
        - status: Current agent status
        - traits: Basic personality traits
        - working_memory: Current working memory state
    """
    return json.dumps({
        "status": "active",
        "traits": ["curious", "brave"],
        "working_memory": "I am exploring a dimly lit room..."
    })
```

## Implementation Details

### Agent Configuration

```python
@dataclass
class LLMConfig:
    """Configuration for language models used by agent"""
    small_llm: Anthropic  # For routine tasks (memory queries, reports)
    large_llm: Anthropic  # For complex reasoning

@dataclass
class AgentConfig:
    """Configuration for an NPC agent"""
    traits: list[str]
    initial_working_memory: str = ""
    initial_long_term_memories: list[str] = field(default_factory=list)
```

### Memory System

1. Working Memory
   - Maintains current state and recent context
   - Updated through observation processing
   - Used for immediate decision making

2. Long-Term Memory
   - Vector store-based memory database
   - Stores important experiences and knowledge
   - Queried during observation processing
   - Automatically updated based on significant events

3. Memory Processing Flow
   ```python
   def process_observation(observation: str, available_actions: dict) -> int:
       # 1. Generate memory queries
       queries = query_generator.generate(working_memory, observation)
       
       # 2. Retrieve relevant memories
       memories = memory_database.retrieve(queries)
       
       # 3. Generate memory report
       report = memory_report_generator.generate(working_memory, observation, memories)
       
       # 4. Update working memory
       working_memory = working_memory_generator.generate(working_memory, report)
       
       # 5. Update long-term memory
       new_memories = long_term_memory_generator.generate(working_memory, observation)
       memory_database.add_memories(new_memories)
       
       # 6. Choose action
       return action_decision_generator.generate(working_memory, available_actions)
   ```

## Client Integration Examples

### Godot Client (GDScript)

```gdscript
# NPCManager.gd
extends Node

class_name NPCManager

var mcp_client: WebSocketClient
var server_url = "ws://localhost:8080"

func _ready():
    mcp_client = WebSocketClient.new()
    mcp_client.connect("connection_established", self, "_on_connection_established")
    mcp_client.connect("data_received", self, "_on_data_received")
    mcp_client.connect_to_url(server_url)

func create_npc(id: String, config: Dictionary) -> void:
    var tool_request = {
        "type": "tool",
        "name": "create_agent",
        "arguments": {
            "agent_id": id,
            "config": {
                "traits": ["friendly", "helpful"],
                "initial_working_memory": "I am a shopkeeper in this town",
                "initial_long_term_memories": [
                    "I've lived here for many years",
                    "I know most of the townspeople"
                ]
            }
        }
    }
    mcp_client.send_text(JSON.print(tool_request))

func update_npc(id: String, observation: String, actions: Array) -> void:
    var tool_request = {
        "type": "tool",
        "name": "process_observation",
        "arguments": {
            "agent_id": id,
            "observation": observation,
            "available_actions": actions
        }
    }
    mcp_client.send_text(JSON.print(tool_request))

func get_npc_info(id: String) -> void:
    var resource_request = {
        "type": "resource",
        "uri": "agent://" + id + "/info"
    }
    mcp_client.send_text(JSON.print(resource_request))

func cleanup_npc(id: String) -> void:
    var tool_request = {
        "type": "tool",
        "name": "cleanup_agent",
        "arguments": {
            "agent_id": id
        }
    }
    mcp_client.send_text(JSON.print(tool_request))

func _on_connection_established(protocol: String) -> void:
    print("Connected to MCP server")

func _on_data_received() -> void:
    var response = JSON.parse(mcp_client.get_peer(1).get_packet().get_string_from_utf8())
    if response.error:
        print("Error: ", response.error_string)
        return
        
    var data = response.result
    match data.get("type"):
        "tool_response":
            _handle_tool_response(data)
        "resource_response":
            _handle_resource_response(data)

func _handle_tool_response(data: Dictionary) -> void:
    match data.get("tool"):
        "create_agent":
            print("Agent created: ", data.result.agent_id)
        "process_observation":
            var action = data.result.action
            var params = data.result.parameters
            _execute_npc_action(action, params)
        "cleanup_agent":
            print("Agent cleaned up: ", data.result.agent_id)

func _handle_resource_response(data: Dictionary) -> void:
    var info = JSON.parse(data.content).result
    print("Agent info: ", info)

func _execute_npc_action(action: String, params: Dictionary) -> void:
    # Implement game-specific action handling here
    pass
```

### Network Python Client

```python
"""Example network client for NPC MCP server

This example shows how to connect to the MCP server over a network connection
without requiring direct access to the npc library.
"""

import asyncio
import json
import websockets

class NPCClient:
    def __init__(self, server_url="ws://localhost:8080"):
        self.server_url = server_url
        self.websocket = None
        
    async def connect(self):
        """Connect to MCP server"""
        self.websocket = await websockets.connect(self.server_url)
        
    async def close(self):
        """Close connection"""
        if self.websocket:
            await self.websocket.close()
            
    async def call_tool(self, name: str, arguments: dict) -> dict:
        """Call an MCP tool"""
        request = {
            "type": "tool",
            "name": name,
            "arguments": arguments
        }
        await self.websocket.send(json.dumps(request))
        response = await self.websocket.recv()
        return json.loads(response)
        
    async def read_resource(self, uri: str) -> dict:
        """Read an MCP resource"""
        request = {
            "type": "resource",
            "uri": uri
        }
        await self.websocket.send(json.dumps(request))
        response = await self.websocket.recv()
        return json.loads(response)

async def main():
    # Create client and connect
    client = NPCClient()
    await client.connect()
    
    try:
        # Create an agent
        agent_id = "network-test"
        result = await client.call_tool(
            "create_agent",
            {
                "agent_id": agent_id,
                "config": {
                    "traits": ["curious", "brave"],
                    "initial_working_memory": "I am exploring this world",
                    "initial_long_term_memories": [
                        "I enjoy discovering new places",
                        "I am skilled at solving puzzles"
                    ]
                }
            }
        )
        print(f"Created agent: {result}")
        
        # Get agent info
        info = await client.read_resource(f"agent://{agent_id}/info")
        print(f"\nAgent info: {info}")
        
        # Process an observation
        observation = "You are in a dimly lit room. There's a door to the north and a chest in the corner."
        available_actions = [
            {
                "name": "examine_chest",
                "description": "Look more closely at the chest",
                "parameters": {}
            },
            {
                "name": "open_door",
                "description": "Open the door and go north",
                "parameters": {}
            }
        ]
        
        result = await client.call_tool(
            "process_observation",
            {
                "agent_id": agent_id,
                "observation": observation,
                "available_actions": available_actions
            }
        )
        print(f"\nChosen action: {result}")
        
        # Cleanup
        result = await client.call_tool(
            "cleanup_agent",
            {"agent_id": agent_id}
        )
        print(f"\nCleaned up agent: {result}")
        
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())
```

## Benefits

1. Sophisticated Memory Management
   - Vector store-based long-term memory
   - Dynamic working memory updates
   - Automatic memory consolidation

2. Flexible LLM Integration
   - Configurable models for different tasks
   - Separation of memory and decision processes
   - Structured prompt templates

3. Clean External Interface
   - Simple text-based observations
   - Well-structured action format
   - Encapsulated complexity

## Error Handling

1. Tool Errors
   - Invalid actions return error in result
   - Missing agents return standard error response
   - Exceptions wrapped in error responses

2. Memory System
   - Failed queries gracefully degraded
   - Default actions on decision failures
   - Memory update retries

3. Client Integration
   - Network issues handled by client retry logic
   - Clear error messages and status codes
   - Resource cleanup on failures
