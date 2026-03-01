"""Dev-agent memory server CLI entry point"""

import argparse
import asyncio
import logging
import os
import signal
import time
from collections import deque
from typing import Optional

import uvicorn
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

from mind.logging_config import get_logger

from .server import DevAgentMCPServer

logger = get_logger()

SERVER_START_TIME = time.time()


class InMemoryLogHandler(logging.Handler):
    """Logging handler that stores recent log entries in memory for /logs endpoint"""

    def __init__(self, max_entries: int = 1000):
        super().__init__()
        self.max_entries = max_entries
        self.logs = deque(maxlen=max_entries)

    def emit(self, record: logging.LogRecord):
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
        all_logs = list(self.logs)
        all_logs.reverse()

        if since is not None:
            since_ms = int(since * 1000)
            all_logs = [log for log in all_logs if int(log["timestamp"] * 1000) > since_ms]

        return all_logs[:limit]


LOG_HANDLER = InMemoryLogHandler(max_entries=1000)


def create_starlette_app(mcp_server, *, debug: bool = False) -> Starlette:
    """Create a Starlette application for serving the MCP server with SSE"""
    sse = SseServerTransport("/sse/")

    class SSEEndpoint:
        async def __call__(self, scope, receive, send):
            async with sse.connect_sse(scope, receive, send) as (read_stream, write_stream):
                await mcp_server.run(
                    read_stream,
                    write_stream,
                    mcp_server.create_initialization_options(),
                )

    async def health_check(request):
        uptime = time.time() - SERVER_START_TIME
        return JSONResponse(
            {"status": "healthy", "version": "1.0", "uptime_seconds": round(uptime, 2)},
            status_code=200,
        )

    async def shutdown_server(request):
        logger.info("Shutdown request received from client")

        def trigger_shutdown():
            os.kill(os.getpid(), signal.SIGTERM)

        loop = asyncio.get_event_loop()
        loop.call_later(0.5, trigger_shutdown)

        return JSONResponse({"status": "shutting down"}, status_code=200)

    async def get_logs(request: Request):
        since = request.query_params.get("since")
        limit = request.query_params.get("limit", "100")

        try:
            since_timestamp = float(since) if since else None
            log_limit = int(limit)
        except (ValueError, TypeError):
            return JSONResponse(
                {"error": "Invalid parameters. 'since' must be a number, 'limit' must be an integer."},
                status_code=400,
            )

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
    parser = argparse.ArgumentParser(description="Dev Agent Memory Server")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--collection", type=str, default="partner")
    args = parser.parse_args()

    root_logger = logging.getLogger()
    if LOG_HANDLER not in root_logger.handlers:
        root_logger.addHandler(LOG_HANDLER)
    root_logger.setLevel(logging.WARNING)

    mind_logger = logging.getLogger("mind")
    mind_logger.setLevel(logging.DEBUG)

    server = DevAgentMCPServer(collection_name=args.collection)
    mcp_server = server.mcp._mcp_server

    starlette_app = create_starlette_app(mcp_server, debug=True)

    print(f"Starting Dev Agent Memory server on http://{args.host}:{args.port}/sse")
    uvicorn.run(starlette_app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
