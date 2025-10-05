"""Example usage and test script for the NPC MCP server"""

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import time
import traceback
from contextlib import asynccontextmanager, suppress
from typing import Optional, Tuple, AsyncGenerator, Dict, Any, TypeVar, Generic, List

from mcp import ClientSession
from mcp.client.sse import sse_client

# --- Configuration Constants ---
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
SERVER_STARTUP_WAIT_TIME = 3  # Seconds to wait for server to start
AGENT_ID = "test-agent-001"

# --- Configure Logging ---
# Create a logger with a more detailed format
logger = logging.getLogger("mcp_test")
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Define a generic result type for operations that may fail
T = TypeVar('T')

class Result(Generic[T]):
    """Represents the result of an operation that may succeed or fail."""
    
    def __init__(self, value: Optional[T] = None, error: Optional[Exception] = None):
        self.value = value
        self.error = error
        self.success = error is None
    
    @classmethod
    def ok(cls, value: T) -> 'Result[T]':
        """Create a successful result with a value."""
        return cls(value=value)
    
    @classmethod
    def err(cls, error: Exception) -> 'Result[T]':
        """Create a failed result with an error."""
        return cls(error=error)


# --- Helper Functions ---

def get_server_script_path() -> str:
    """Determine the absolute path to the mcp_server.py script."""
    current_dir = os.path.dirname(__file__)
    server_script_rel_path = os.path.join(current_dir, "..", "..", "mcp_server.py")
    return os.path.abspath(server_script_rel_path)


def log_exception_details(exception: Exception, prefix: str = "") -> None:
    """Log detailed information about an exception, including nested exceptions in ExceptionGroups."""
    logger.error(f"{prefix}{type(exception).__name__}: {exception}")
    
    # Handle exception groups (Python 3.11+)
    if hasattr(exception, 'exceptions') and isinstance(exception, BaseExceptionGroup):
        for idx, ex in enumerate(exception.exceptions):
            log_exception_details(ex, prefix=f"  Nested exception #{idx+1}: ")


def collect_process_output(process: subprocess.Popen) -> Tuple[str, str]:
    """Collect stdout and stderr from a subprocess safely."""
    stdout, stderr = "", ""
    try:
        stdout, stderr = process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        logger.warning("Process communication timed out, killing process")
        process.kill()
        # Try again after kill
        with suppress(Exception):
            stdout, stderr = process.communicate(timeout=1)
    except Exception as e:
        logger.error(f"Error collecting process output: {e}")
    
    return stdout, stderr


@asynccontextmanager
async def manage_mcp_server(
    host: str, port: int, start_server_flag: bool
) -> AsyncGenerator[Optional[subprocess.Popen], None]:
    """Context manager to start and stop the MCP server subprocess if requested."""
    server_process: Optional[subprocess.Popen] = None
    
    if not start_server_flag:
        logger.info("Skipping local server start")
        yield None
        return

    server_script = get_server_script_path()
    if not os.path.exists(server_script):
        logger.error(f"Server script not found at {server_script}")
        yield None
        return

    command = [sys.executable, server_script, "--host", host, "--port", str(port)]
    logger.info(f"Starting MCP server with command: {' '.join(command)}")

    try:
        server_process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        logger.info(f"Server process started (PID: {server_process.pid})")
        logger.info(f"Waiting {SERVER_STARTUP_WAIT_TIME}s for server initialization")
        await asyncio.sleep(SERVER_STARTUP_WAIT_TIME)

        # Check if server is still running
        if server_process.poll() is not None:
            logger.error("Server process terminated unexpectedly")
            stdout, stderr = collect_process_output(server_process)
            if stdout:
                logger.info(f"Server stdout:\n{stdout}")
            if stderr:
                logger.error(f"Server stderr:\n{stderr}")
            server_process = None

        yield server_process

    except Exception as e:
        logger.error(f"Error starting server process: {e}")
        log_exception_details(e)
        yield None
        
    finally:
        if server_process and server_process.poll() is None:
            logger.info(f"Stopping MCP server (PID: {server_process.pid})")
            server_process.terminate()
            
            stdout, stderr = collect_process_output(server_process)
            
            if server_process.poll() is None:
                logger.warning("Server did not terminate gracefully, killing")
                server_process.kill()
                server_process.wait()
                logger.info("Server killed")
            else:
                logger.info("Server stopped")
                
            if stderr:
                logger.error(f"Server stderr:\n{stderr}")


@asynccontextmanager
async def connect_mcp_client(server_url: str) -> AsyncGenerator[Optional[ClientSession], None]:
    """Context manager to connect to the MCP server via SSE."""
    logger.info(f"Connecting to MCP server at {server_url}")
    session: Optional[ClientSession] = None
    try:
        # sse_client itself is an async context manager
        async with sse_client(server_url) as (read, write):
            # ClientSession is also an async context manager
            async with ClientSession(read, write) as session:
                logger.info("Initializing MCP session")
                await session.initialize()
                logger.info("MCP session initialized successfully")
                
                # Verify connection by listing tools (optional)
                try:
                    tools_response = await session.list_tools()
                    tool_names = [t.name for t in tools_response.tools]
                    logger.info(f"Available tools: {', '.join(tool_names)}")
                except Exception as e:
                    logger.warning(f"Failed to list tools after connection: {type(e).__name__}: {e}")
                    # Failure to list tools is non-fatal

                yield session
                
    except ConnectionRefusedError:
        logger.error(f"Connection refused. Is the server running at {server_url}?")
        yield None
    except Exception as e:
        logger.error(f"Error connecting to MCP server: {type(e).__name__}: {e}")
        log_exception_details(e)
        yield None


def parse_tool_result(result) -> Result[Dict[str, Any]]:
    """Safely parse JSON content from an MCP tool result.
    
    Returns:
        Result object containing either the parsed data or an error
    """
    if not result or not result.content:
        logger.error("Tool result is empty or has no content")
        return Result.err(ValueError("Empty tool result"))
        
    if not hasattr(result.content[0], 'text'):
        logger.error(f"Unexpected tool result format (missing text): {result.content[0]}")
        return Result.err(ValueError("Missing text in tool result"))

    try:
        data = json.loads(result.content[0].text)
        if not isinstance(data, dict):
            logger.error(f"Tool result JSON is not a dictionary: {data}")
            return Result.err(TypeError("Tool result is not a dictionary"))
        return Result.ok(data)
    except json.JSONDecodeError as e:
        logger.error(f"Could not decode JSON from tool result: {result.content[0].text}")
        return Result.err(e)
    except Exception as e:
        logger.error(f"Error parsing tool result: {e}")
        return Result.err(e)


async def call_tool(session: ClientSession, tool_name: str, arguments: Dict[str, Any]) -> Result[Dict[str, Any]]:
    """Call an MCP tool with error handling.
    
    Args:
        session: Active MCP ClientSession
        tool_name: Name of the tool to call
        arguments: Arguments to pass to the tool
        
    Returns:
        Result object with parsed response data or error
    """
    logger.info(f"Calling tool '{tool_name}'")
    try:
        result_raw = await session.call_tool(tool_name, arguments=arguments)
        result = parse_tool_result(result_raw)
        return result
    except Exception as e:
        logger.error(f"Error calling tool '{tool_name}': {type(e).__name__}: {e}")
        return Result.err(e)


async def run_agent_scenario(session: ClientSession, agent_id: str):
    """Run a predefined test scenario with the agent."""
    logger.info(f"Running agent scenario for ID: {agent_id}")

    # 1. Create Agent
    logger.info("1. Creating agent")
    create_args = {
        "agent_id": agent_id,
        "config": {
            "traits": ["curious", "brave", "observant"],
            "initial_working_memory": "I am an adventurer standing at the entrance of a mysterious structure.",
            "initial_long_term_memories": [
                "Legends speak of treasure within these walls.",
                "I am wary of traps."
            ]
        }
    }
    
    create_result = await call_tool(session, "create_agent", create_args)
    if not create_result.success:
        logger.error("Failed to create agent, aborting scenario")
        return
        
    agent_data = create_result.value
    logger.info(f"Create agent result: {agent_data}")
    
    # Check if creation was successful based on response content
    if agent_data.get("status") != "created":
        error_msg = agent_data.get("message", "Unknown error")
        logger.error(f"Agent creation failed: {error_msg}")
        return

    # 2. Get Agent Info (Resource Read)
    logger.info("2. Getting agent info")
    agent_info_uri = f"agent://{agent_id}/info"
    try:
        info_result_raw = await session.read_resource(agent_info_uri)
        # Resource results might be structured differently, often directly text
        info_text = info_result_raw.contents[0].text if info_result_raw.contents else "{}"
        try:
            info_data = json.loads(info_text)
            logger.info(f"Agent info: {info_data}")
        except json.JSONDecodeError:
            logger.error(f"Could not decode JSON from agent info resource: {info_text}")
    except Exception as e:
        logger.error(f"Error reading resource {agent_info_uri}: {type(e).__name__}: {e}")


    # 3. Process First Observation
    logger.info("3. Processing first observation")
    observation1 = "You are in a dimly lit stone room. Moss covers the damp walls. There's a heavy wooden door to the north and a rusty iron chest in the corner."
    actions1 = [
        {"name": "examine_chest", "description": "Look closely at the rusty chest.", "parameters": {}},
        {"name": "open_door", "description": "Try to open the heavy wooden door.", "parameters": {}},
        {"name": "check_walls", "description": "Examine the mossy walls for hidden switches.", "parameters": {}},
    ]
    process_args1 = {
        "agent_id": agent_id,
        "observation": observation1,
        "available_actions": actions1
    }
    
    action1_result = await call_tool(session, "process_observation", process_args1)
    chosen_action1 = None
    
    if action1_result.success:
        action1_data = action1_result.value
        logger.info(f"Process observation 1 result: {action1_data}")
        
        if action1_data.get("status") != "error":
            chosen_action1 = action1_data.get("action")
            logger.info(f"Agent chose action: {chosen_action1}")
        else:
            error_msg = action1_data.get("message", "Unknown error")
            logger.error(f"Error from process_observation: {error_msg}")


    # 4. Process Second Observation (if first was successful)
    if chosen_action1:
        logger.info("4. Processing second observation")
        
        # Determine observation and actions based on first choice
        observation_map = {
            "examine_chest": (
                "The chest is sealed shut with rust. It looks like it needs a key or brute force.",
                [
                    {"name": "force_chest", "description": "Attempt to pry the chest open.", "parameters": {}},
                    {"name": "leave_chest", "description": "Ignore the chest for now.", "parameters": {}}
                ]
            ),
            "open_door": (
                "The door creaks open into a dark, narrow passage. You hear dripping water.",
                [
                    {"name": "enter_passage", "description": "Proceed carefully into the passage.", "parameters": {}},
                    {"name": "close_door", "description": "Close the door and reconsider.", "parameters": {}}
                ]
            ),
            "check_walls": (
                "You run your hands over the damp moss. You find a loose stone near the floor.",
                [
                    {"name": "pull_stone", "description": "Try to pull the loose stone.", "parameters": {}},
                    {"name": "ignore_stone", "description": "Leave the stone alone.", "parameters": {}}
                ]
            )
        }
        
        # Default fallback if action is unknown
        observation2, actions2 = observation_map.get(
            chosen_action1, 
            ("You stand indecisively.", [{"name": "wait", "description": "Wait and observe.", "parameters": {}}])
        )
        
        if chosen_action1 not in observation_map:
            logger.warning(f"Unknown action '{chosen_action1}' chosen in step 1, using fallback")
        
        process_args2 = {
            "agent_id": agent_id,
            "observation": observation2,
            "available_actions": actions2
        }
        
        action2_result = await call_tool(session, "process_observation", process_args2)
        if action2_result.success:
            logger.info(f"Process observation 2 result: {action2_result.value}")
        # No need to handle failure case specially here as call_tool already logs errors

    # 5. Cleanup Agent
    logger.info("5. Cleaning up agent")
    cleanup_args = {"agent_id": agent_id}
    cleanup_result = await call_tool(session, "cleanup_agent", cleanup_args)
    
    if cleanup_result.success:
        logger.info(f"Cleanup agent result: {cleanup_result.value}")
    
    logger.info(f"Agent scenario finished for ID: {agent_id}")


async def main():
    """Main function to parse args, manage server, connect client, and run test."""
    parser = argparse.ArgumentParser(description="Test the NPC MCP server")
    parser.add_argument("--host", type=str, default=DEFAULT_HOST,
                        help=f"Host address for the MCP server (default: {DEFAULT_HOST})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"Port for the MCP server (default: {DEFAULT_PORT})")
    parser.add_argument("--no-start-server", action="store_true",
                        help="Do not attempt to start the MCP server locally")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")
    args = parser.parse_args()
    
    # Set debug level if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

    start_server_flag = not args.no_start_server
    server_url = f"http://{args.host}:{args.port}/sse"  # SSE endpoint
    logger.info(f"Using server URL: {server_url}")

    # Use context managers for server and client lifecycles
    async with manage_mcp_server(args.host, args.port, start_server_flag) as server_proc:
        # If server failed to start and we were asked to start it, exit
        if start_server_flag and not server_proc:
            logger.error("Exiting due to server startup failure")
            return

        # If server is running (or we didn't need to start it), try connecting client
        async with connect_mcp_client(server_url) as session:
            if session:
                await run_agent_scenario(session, AGENT_ID)
            else:
                logger.error("Exiting due to client connection failure")


if __name__ == "__main__":
    logger.info("--- Starting MCP Test Script ---")
    
    # On Windows, default event loop policy can cause issues with subprocesses
    if sys.platform == "win32":
        logger.debug("Using WindowsProactorEventLoopPolicy")
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Test script interrupted by user")
    except Exception as e:
        logger.error(f"Unhandled exception in main: {type(e).__name__}: {e}")
        traceback.print_exc()
        
    logger.info("--- MCP Test Script Finished ---")
