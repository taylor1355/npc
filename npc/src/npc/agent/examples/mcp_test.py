"""Example usage of the NPC MCP server"""

import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    # Start MCP server
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "npc.mcp_server"],
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize connection
            await session.initialize()
            
            # Create an agent
            agent_id = "test-agent"
            result = await session.call_tool(
                "create_agent",
                arguments={
                    "agent_id": agent_id,
                    "config": {
                        "traits": ["curious", "brave"],
                        "initial_working_memory": "I am an adventurer exploring this world",
                        "initial_long_term_memories": [
                            "I enjoy discovering new places",
                            "I am skilled at solving puzzles"
                        ]
                    }
                }
            )
            print(f"Created agent: {result}")
            
            # Get agent info
            info = await session.read_resource(f"agent://{agent_id}/info")
            print(f"\nAgent info: {info}")
            
            # Simulate some interactions
            
            # First observation
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
            
            result = await session.call_tool(
                "process_observation",
                arguments={
                    "agent_id": agent_id,
                    "observation": observation,
                    "available_actions": available_actions
                }
            )
            print(f"\nFirst action: {result}")
            
            # Parse the result content as JSON
            result_json = json.loads(result.content[0].text)
            
            # Check if there was an error
            if result_json.get("status") == "error":
                print(f"Error processing observation: {result_json.get('message')}")
                return
                
            # Second observation (based on chosen action)
            if result_json.get("action") == "examine_chest":
                observation = "The chest is made of old wood with iron bindings. It appears to be locked."
                available_actions = [
                    {
                        "name": "try_open",
                        "description": "Attempt to open the locked chest",
                        "parameters": {}
                    },
                    {
                        "name": "leave_chest",
                        "description": "Leave the chest and go to the door",
                        "parameters": {}
                    }
                ]
            else:  # open_door
                observation = "You enter a well-lit hallway. There are paintings on the walls."
                available_actions = [
                    {
                        "name": "examine_paintings",
                        "description": "Study the paintings on the walls",
                        "parameters": {}
                    },
                    {
                        "name": "continue_forward",
                        "description": "Continue down the hallway",
                        "parameters": {}
                    }
                ]
            
            result = await session.call_tool(
                "process_observation",
                arguments={
                    "agent_id": agent_id,
                    "observation": observation,
                    "available_actions": available_actions
                }
            )
            print(f"\nSecond action: {result}")
            
            # Cleanup
            result = await session.call_tool(
                "cleanup_agent",
                arguments={"agent_id": agent_id}
            )
            print(f"\nCleaned up agent: {result}")


if __name__ == "__main__":
    asyncio.run(main())
