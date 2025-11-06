"""MCP server CLI entry point"""

import argparse
import asyncio
import logging
import os
import signal
import time

# Disable tqdm progress bars to prevent BrokenPipeError in SSE context
os.environ["TQDM_DISABLE"] = "1"

import uvicorn
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

from mind.project_config import OPENROUTER_API_KEY

from .server import MCPServer

logger = logging.getLogger(__name__)

# Track server start time for uptime calculations
SERVER_START_TIME = time.time()


def create_starlette_app(mcp_server, *, debug: bool = False) -> Starlette:
    """Create a Starlette application for serving the MCP server with SSE"""
    sse = SseServerTransport("/sse/")

    class SSEEndpoint:
        """ASGI app for handling SSE connections"""

        async def __call__(self, scope, receive, send):
            async with sse.connect_sse(scope, receive, send) as (read_stream, write_stream):
                await mcp_server.run(
                    read_stream,
                    write_stream,
                    mcp_server.create_initialization_options(),
                )

    async def health_check(request):
        """Health check endpoint for monitoring server status"""
        uptime = time.time() - SERVER_START_TIME
        return JSONResponse(
            {"status": "healthy", "version": "1.0", "uptime_seconds": round(uptime, 2)},
            status_code=200,
        )

    async def shutdown_server(request):
        """Shut down the MCP server via HTTP request"""
        logger.info("Shutdown request received from client")

        def trigger_shutdown():
            """Send SIGTERM to self for graceful shutdown"""
            os.kill(os.getpid(), signal.SIGTERM)

        # Schedule shutdown after response is sent
        loop = asyncio.get_event_loop()
        loop.call_later(0.5, trigger_shutdown)

        return JSONResponse({"status": "shutting down"}, status_code=200)

    return Starlette(
        debug=debug,
        routes=[
            Route("/sse", endpoint=SSEEndpoint()),
            Mount("/sse/", app=sse.handle_post_message),
            Route("/health", endpoint=health_check, methods=["GET"]),
            Route("/shutdown", endpoint=shutdown_server, methods=["POST"]),
        ],
    )


def main():
    """Run the MCP server from command line"""
    parser = argparse.ArgumentParser(description="NPC Mind MCP Server")
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address to bind the server to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to run the server on (default: 8000)"
    )
    args = parser.parse_args()

    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY must be set in project_config.py")

    # Create the server instance
    server = MCPServer("NPC Mind Server")

    # Extract the underlying core MCP server
    mcp_server = server.mcp._mcp_server

    # Create the Starlette app
    starlette_app = create_starlette_app(mcp_server, debug=True)

    print(f"Starting NPC Mind MCP server on http://{args.host}:{args.port}/sse")

    # Run using uvicorn
    uvicorn.run(starlette_app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
