"""Unit tests for Dev Agent MCP server"""

import json
import uuid

import pytest

from mind.domains.dev_agent.server import DevAgentMCPServer


def parse_dict_response(result):
    """Parse a single-dict MCP tool response"""
    return json.loads(result[0].text)


def parse_list_response(result):
    """Parse a list MCP tool response (one TextContent per item)"""
    return [json.loads(item.text) for item in result]


@pytest.fixture
def server():
    """Create a dev agent server with ephemeral storage for testing.
    Uses a unique collection name per test to prevent cross-test contamination.
    """
    return DevAgentMCPServer(
        collection_name=f"test_{uuid.uuid4().hex[:8]}",
        storage_path=None,  # Ephemeral
    )


class TestStoreMemory:
    @pytest.mark.asyncio
    async def test_store_basic_memory(self, server):
        result = await server.mcp.call_tool("store_memory", {
            "content": "Architecture uses component composition",
            "importance": 7.0,
        })
        response = parse_dict_response(result)
        assert response["status"] == "stored"
        assert response["memory_id"]
        assert response["importance"] == 7.0

    @pytest.mark.asyncio
    async def test_store_memory_with_tags(self, server):
        result = await server.mcp.call_tool("store_memory", {
            "content": "Pathfinding uses 8x amplification",
            "importance": 8.0,
            "tags": ["architecture", "pathfinding"],
        })
        response = parse_dict_response(result)
        assert response["status"] == "stored"
        assert set(response["tags"]) == {"architecture", "pathfinding"}

    @pytest.mark.asyncio
    async def test_store_memory_default_importance(self, server):
        result = await server.mcp.call_tool("store_memory", {
            "content": "Some observation",
        })
        response = parse_dict_response(result)
        assert response["importance"] == 5.0


class TestSearchMemory:
    @pytest.mark.asyncio
    async def test_search_finds_stored_memory(self, server):
        await server.mcp.call_tool("store_memory", {
            "content": "NPCs use bid-based interactions",
            "tags": ["architecture"],
        })
        result = await server.mcp.call_tool("search_memory", {
            "query": "how do NPC interactions work",
        })
        response = parse_list_response(result)
        assert len(response) >= 1
        assert any("bid" in m["content"].lower() for m in response)

    @pytest.mark.asyncio
    async def test_search_with_tag_filter(self, server):
        await server.mcp.call_tool("store_memory", {
            "content": "Architecture pattern A",
            "tags": ["architecture"],
        })
        await server.mcp.call_tool("store_memory", {
            "content": "Debug technique B",
            "tags": ["debugging"],
        })
        result = await server.mcp.call_tool("search_memory", {
            "query": "patterns",
            "tags": ["architecture"],
        })
        response = parse_list_response(result)
        assert len(response) >= 1
        assert all("architecture" in m["tags"] for m in response)

    @pytest.mark.asyncio
    async def test_search_empty_collection(self, server):
        result = await server.mcp.call_tool("search_memory", {
            "query": "anything",
        })
        response = parse_list_response(result)
        assert response == []


class TestConsolidate:
    @pytest.mark.asyncio
    async def test_consolidate_returns_stats(self, server):
        await server.mcp.call_tool("store_memory", {
            "content": "Test memory for stats",
        })
        result = await server.mcp.call_tool("consolidate", {})
        response = parse_dict_response(result)
        assert response["status"] == "stats"
        assert response["total_memories"] == 1

    @pytest.mark.asyncio
    async def test_consolidate_empty_collection(self, server):
        result = await server.mcp.call_tool("consolidate", {})
        response = parse_dict_response(result)
        assert response["total_memories"] == 0
