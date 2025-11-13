"""MCP server CLI entry point"""

import argparse
import asyncio
import logging
import os
import signal
import time
from collections import deque
from typing import Optional

# Disable tqdm progress bars to prevent BrokenPipeError in SSE context
os.environ["TQDM_DISABLE"] = "1"

import uvicorn
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

from mind.logging_config import get_logger
from mind.project_config import OPENROUTER_API_KEY

from .server import MCPServer

logger = get_logger()

# Track server start time for uptime calculations
SERVER_START_TIME = time.time()


class InMemoryLogHandler(logging.Handler):
    """Logging handler that stores recent log entries in memory for /logs endpoint"""

    def __init__(self, max_entries: int = 1000):
        super().__init__()
        self.max_entries = max_entries
        self.logs = deque(maxlen=max_entries)

    def emit(self, record: logging.LogRecord):
        """Store log record in memory buffer"""
        try:
            log_entry = {
                "timestamp": record.created,
                "level": record.levelname,
                "message": self.format(record),
            }
            self.logs.append(log_entry)
        except Exception:
            self.handleError(record)

    def get_logs(self, since: Optional[float] = None, limit: int = 100) -> list[dict]:
        """
        Retrieve stored log entries, newest first.

        Args:
            since: Unix timestamp - only return logs after this time
            limit: Maximum number of entries to return

        Returns:
            List of log entry dicts with timestamp, level, message
        """
        # Convert deque to list (newest last) then reverse for newest-first
        all_logs = list(self.logs)
        all_logs.reverse()

        # Filter by timestamp if provided
        if since is not None:
            all_logs = [log for log in all_logs if log["timestamp"] > since]

        # Apply limit
        return all_logs[:limit]


# Global log handler instance
LOG_HANDLER = InMemoryLogHandler(max_entries=1000)


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

    async def get_logs(request: Request):
        """Retrieve structured log entries for client consumption"""
        # Parse query parameters
        since = request.query_params.get("since")
        limit = request.query_params.get("limit", "100")

        # Convert parameters to appropriate types
        try:
            since_timestamp = float(since) if since else None
            log_limit = int(limit)
        except (ValueError, TypeError):
            return JSONResponse(
                {"error": "Invalid parameters. 'since' must be a number, 'limit' must be an integer."},
                status_code=400,
            )

        # Retrieve logs from handler
        logs = LOG_HANDLER.get_logs(since=since_timestamp, limit=log_limit)

        return JSONResponse({"logs": logs}, status_code=200)

    return Starlette(
        debug=debug,
        routes=[
            Route("/sse", endpoint=SSEEndpoint()),
            Mount("/sse/", app=sse.handle_post_message),
            Route("/health", endpoint=health_check, methods=["GET"]),
            Route("/shutdown", endpoint=shutdown_server, methods=["POST"]),
            Route("/logs", endpoint=get_logs, methods=["GET"]),
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

    # Configure logging to capture logs in memory
    # Set root logger to WARNING to suppress noisy dependency logs (OpenAI, uvicorn, etc.)
    root_logger = logging.getLogger()
    root_logger.addHandler(LOG_HANDLER)
    root_logger.setLevel(logging.WARNING)

    # Set mind application logger to DEBUG for detailed observability
    mind_logger = logging.getLogger('mind')
    mind_logger.setLevel(logging.DEBUG)

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
