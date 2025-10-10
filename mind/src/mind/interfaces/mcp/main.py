"""MCP server CLI entry point"""

import argparse

from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.routing import Mount, Route
import uvicorn

from mind.project_config import OPENROUTER_API_KEY
from .server import MCPServer


def create_starlette_app(mcp_server, *, debug: bool = False) -> Starlette:
    """Create a Starlette application for serving the MCP server with SSE"""
    sse = SseServerTransport("/sse/")

    async def handle_sse(request: Request) -> None:
        async with sse.connect_sse(
            request.scope,
            request.receive,
            request._send,
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
            Mount("/sse/", app=sse.handle_post_message),
        ],
    )


def main():
    """Run the MCP server from command line"""
    parser = argparse.ArgumentParser(description="NPC Mind MCP Server")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                      help="Host address to bind the server to (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000,
                      help="Port to run the server on (default: 8000)")
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
