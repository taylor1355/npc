"""Unit tests for MCP server"""

import json

import pytest

from mind.interfaces.mcp.server import MCPServer


def parse_response(result):
    """Parse MCP response from TextContent list"""
    return json.loads(result[0].text)


class TestMCPServerErrorHandling:
    """Test MCP server error handling - critical for production"""

    @pytest.mark.asyncio
    async def test_decide_action_with_missing_mind(self):
        """Should return error dict when mind doesn't exist"""
        server = MCPServer()

        result = await server.mcp.call_tool(
            "decide_action",
            {
                "mind_id": "nonexistent",
                "observation": {
                    "entity_id": "test",
                    "current_simulation_time": 0,
                },
            },
        )

        response = parse_response(result)

        assert response is not None
        assert response["status"] == "error"
        assert "not found" in response["error_message"]
        assert response["action"] is None
        assert "request_id" in response

    @pytest.mark.asyncio
    async def test_decide_action_with_invalid_observation_missing_required_field(self):
        """Should return error dict for missing required fields"""
        server = MCPServer()

        await server.mcp.call_tool(
            "create_mind",
            {
                "mind_id": "test",
                "config": {
                    "entity_id": "test",
                    "traits": [],
                    "initial_long_term_memories": [],
                },
            },
        )

        result = await server.mcp.call_tool(
            "decide_action",
            {
                "mind_id": "test",
                "observation": {"entity_id": "test"},
            },
        )

        response = parse_response(result)

        assert response is not None
        assert response["status"] == "error"
        assert response["action"] is None
        assert "request_id" in response
        assert "details" in response
        assert (
            "ValidationError" in response["error_message"]
            or "current_simulation_time" in response["error_message"]
        )

    @pytest.mark.asyncio
    async def test_decide_action_with_invalid_observation_wrong_type(self):
        """Should return error dict for wrong field types"""
        server = MCPServer()

        await server.mcp.call_tool(
            "create_mind",
            {
                "mind_id": "test",
                "config": {
                    "entity_id": "test",
                    "traits": [],
                    "initial_long_term_memories": [],
                },
            },
        )

        result = await server.mcp.call_tool(
            "decide_action",
            {
                "mind_id": "test",
                "observation": {
                    "entity_id": "test",
                    "current_simulation_time": "not_an_int",
                },
            },
        )

        response = parse_response(result)

        assert response is not None
        assert response["status"] == "error"
        assert response["action"] is None
        assert "request_id" in response
        assert (
            "ValidationError" in response["error_message"]
            or "int" in response["error_message"].lower()
        )

    @pytest.mark.asyncio
    async def test_decide_action_with_vision_empty_dict(self):
        """Should return error dict for malformed vision observation"""
        server = MCPServer()

        await server.mcp.call_tool(
            "create_mind",
            {
                "mind_id": "test",
                "config": {
                    "entity_id": "test",
                    "traits": [],
                    "initial_long_term_memories": [],
                },
            },
        )

        result = await server.mcp.call_tool(
            "decide_action",
            {
                "mind_id": "test",
                "observation": {
                    "entity_id": "test",
                    "current_simulation_time": 100,
                    "vision": {},
                },
            },
        )

        response = parse_response(result)

        assert response is not None
        assert response["status"] == "error"
        assert response["action"] is None
        assert "request_id" in response
        assert (
            "visible_entities" in response["error_message"]
            or "ValidationError" in response["error_message"]
        )

    @pytest.mark.asyncio
    async def test_decide_action_error_includes_exception_type(self):
        """Should include exception type in error message for debugging"""
        server = MCPServer()

        await server.mcp.call_tool(
            "create_mind",
            {
                "mind_id": "test",
                "config": {
                    "entity_id": "test",
                    "traits": [],
                    "initial_long_term_memories": [],
                },
            },
        )

        result = await server.mcp.call_tool(
            "decide_action",
            {
                "mind_id": "test",
                "observation": {
                    "entity_id": "test",
                    "current_simulation_time": "invalid",
                },
            },
        )

        response = parse_response(result)

        assert response is not None
        assert response["status"] == "error"
        assert ":" in response["error_message"]
        assert "request_id" in response

    @pytest.mark.asyncio
    async def test_decide_action_with_valid_observation(self):
        """Should return success with valid observation"""
        server = MCPServer()

        await server.mcp.call_tool(
            "create_mind",
            {
                "mind_id": "test",
                "config": {
                    "entity_id": "test",
                    "traits": ["curious"],
                    "initial_long_term_memories": [],
                },
            },
        )

        observation = {
            "entity_id": "test",
            "current_simulation_time": 100,
            "status": {
                "position": [5, 5],
                "movement_locked": False,
                "current_interaction": {},
                "controller_state": {},
            },
            "needs": {
                "needs": {"hunger": 75.0, "energy": 50.0},
                "max_value": 100.0,
            },
            "vision": {
                "visible_entities": [
                    {
                        "entity_id": "apple_001",
                        "display_name": "Apple",
                        "position": [6, 5],
                        "interactions": {},
                    }
                ]
            },
            "conversations": [],
        }

        result = await server.mcp.call_tool(
            "decide_action",
            {
                "mind_id": "test",
                "observation": observation,
            },
        )

        response = parse_response(result)

        assert response is not None
        assert response["status"] == "success"
        assert "action" in response
        assert response["error_message"] is None
        assert "request_id" in response

        if response["action"] is not None:
            assert isinstance(response["action"], dict)
            assert "action" in response["action"]
            assert "parameters" in response["action"]
