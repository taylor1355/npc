"""Unit tests for MCP server"""

import json
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage

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
                    "entity_id": "entity_test",
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
                "mind_id": "mind_test",
                "entity_id": "entity_test",
                "config": {
                    "traits": [],
                    "initial_long_term_memories": [],
                },
            },
        )

        result = await server.mcp.call_tool(
            "decide_action",
            {
                "mind_id": "mind_test",
                "observation": {"entity_id": "entity_test"},
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
                "mind_id": "mind_test",
                "entity_id": "entity_test",
                "config": {
                    "traits": [],
                    "initial_long_term_memories": [],
                },
            },
        )

        result = await server.mcp.call_tool(
            "decide_action",
            {
                "mind_id": "mind_test",
                "observation": {
                    "entity_id": "entity_test",
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
                "mind_id": "mind_test",
                "entity_id": "entity_test",
                "config": {
                    "traits": [],
                    "initial_long_term_memories": [],
                },
            },
        )

        result = await server.mcp.call_tool(
            "decide_action",
            {
                "mind_id": "mind_test",
                "observation": {
                    "entity_id": "entity_test",
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
                "mind_id": "mind_test",
                "entity_id": "entity_test",
                "config": {
                    "traits": [],
                    "initial_long_term_memories": [],
                },
            },
        )

        result = await server.mcp.call_tool(
            "decide_action",
            {
                "mind_id": "mind_test",
                "observation": {
                    "entity_id": "entity_test",
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
        """Should return success with valid observation when pipeline succeeds"""
        from mind.cognitive_architecture.actions import Action
        from mind.cognitive_architecture.state import PipelineState

        server = MCPServer()

        await server.mcp.call_tool(
            "create_mind",
            {
                "mind_id": "mind_test",
                "entity_id": "entity_test",
                "config": {
                    "traits": ["curious"],
                    "initial_long_term_memories": [],
                },
            },
        )

        observation = {
            "entity_id": "entity_test",
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

        # Mock the pipeline to return a successful action
        mind = server.minds["mind_test"]
        original_process = mind.pipeline.process

        async def mock_process(state: PipelineState) -> PipelineState:
            state.chosen_action = Action.model_construct(action="wait", parameters={})
            return state

        mind.pipeline.process = mock_process

        result = await server.mcp.call_tool(
            "decide_action",
            {
                "mind_id": "mind_test",
                "observation": observation,
            },
        )

        # Restore original
        mind.pipeline.process = original_process

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

    @pytest.mark.asyncio
    async def test_bid_cleanup_after_response(self, caplog):
        """Should remove bid from pending_incoming_bids after responding.

        Also asserts the attribution decouple: the bid-cleanup log line carries the
        entity FK (entity_test), not the mind PK (mind_test), so the sim /logs
        forwarder routes it to the NPC's Events tab.
        """
        import logging

        from mind.cognitive_architecture.actions import Action, ActionType
        from mind.cognitive_architecture.observations import MindEvent, MindEventType
        from mind.cognitive_architecture.state import PipelineState

        server = MCPServer()

        await server.mcp.call_tool(
            "create_mind",
            {
                "mind_id": "mind_test",
                "entity_id": "entity_test",
                "config": {
                    "traits": ["friendly"],
                    "initial_long_term_memories": [],
                },
            },
        )

        observation = {
            "entity_id": "entity_test",
            "current_simulation_time": 100,
            "status": {
                "position": [5, 5],
                "movement_locked": False,
                "current_interaction": {},
                "controller_state": {},
            },
        }

        # Create bid event
        bid_event = {
            "timestamp": 100,
            "event_type": MindEventType.INTERACTION_BID_RECEIVED,
            "payload": {
                "bid_id": "bid_test_123",
                "bidder_id": "npc_other",
                "bidder_name": "Bob",
                "interaction_name": "conversation",
            },
        }

        # Mock the pipeline to return a bid response action
        mind = server.minds["mind_test"]
        original_process = mind.pipeline.process

        async def mock_process(state: PipelineState) -> PipelineState:
            state.chosen_action = Action.model_construct(
                action=ActionType.RESPOND_TO_INTERACTION_BID,
                parameters={
                    "bid_id": "bid_test_123",
                    "accept": True,
                    "reason": "",
                },
            )
            return state

        mind.pipeline.process = mock_process

        # Verify bid is stored before response
        with caplog.at_level(logging.DEBUG):
            result = await server.mcp.call_tool(
                "decide_action",
                {
                    "mind_id": "mind_test",
                    "observation": observation,
                    "events": [bid_event],
                },
            )

        # Restore original
        mind.pipeline.process = original_process

        response = parse_response(result)

        # Verify response is successful
        assert response["status"] == "success"
        assert response["action"]["action"] == ActionType.RESPOND_TO_INTERACTION_BID

        # Verify bid was removed from pending_incoming_bids
        assert "bid_test_123" not in mind.pending_incoming_bids

        # Attribution decouple: the bid-cleanup line is tagged with the entity FK,
        # never the mind PK, so it lands on the NPC's Events tab.
        cleanup_lines = [
            r.getMessage() for r in caplog.records if "Removed bid bid_test_123" in r.getMessage()
        ]
        assert cleanup_lines, "expected a bid-cleanup log line"
        for line in cleanup_lines:
            assert "[entity_test]" in line
            assert "[mind_test]" not in line

    @pytest.mark.asyncio
    async def test_bid_cleanup_only_for_bid_response_actions(self):
        """Should not affect pending_incoming_bids for non-bid actions"""
        from mind.cognitive_architecture.actions import Action, ActionType
        from mind.cognitive_architecture.observations import MindEventType
        from mind.cognitive_architecture.state import PipelineState

        server = MCPServer()

        await server.mcp.call_tool(
            "create_mind",
            {
                "mind_id": "mind_test",
                "entity_id": "entity_test",
                "config": {
                    "traits": ["friendly"],
                    "initial_long_term_memories": [],
                },
            },
        )

        observation = {
            "entity_id": "entity_test",
            "current_simulation_time": 100,
            "status": {
                "position": [5, 5],
                "movement_locked": False,
                "current_interaction": {},
                "controller_state": {},
            },
        }

        # Create bid event
        bid_event = {
            "timestamp": 100,
            "event_type": MindEventType.INTERACTION_BID_RECEIVED,
            "payload": {
                "bid_id": "bid_test_456",
                "bidder_id": "npc_other",
                "bidder_name": "Alice",
                "interaction_name": "trade",
            },
        }

        # Mock the pipeline to return a non-bid action (e.g., wait)
        mind = server.minds["mind_test"]
        original_process = mind.pipeline.process

        async def mock_process(state: PipelineState) -> PipelineState:
            state.chosen_action = Action.model_construct(
                action=ActionType.WAIT,
                parameters={},
            )
            return state

        mind.pipeline.process = mock_process

        result = await server.mcp.call_tool(
            "decide_action",
            {
                "mind_id": "mind_test",
                "observation": observation,
                "events": [bid_event],
            },
        )

        # Restore original
        mind.pipeline.process = original_process

        response = parse_response(result)

        # Verify response is successful
        assert response["status"] == "success"
        assert response["action"]["action"] == ActionType.WAIT

        # Verify bid was NOT removed (only removed when responding)
        assert "bid_test_456" in mind.pending_incoming_bids


class TestMindConfigValidation:
    """Pydantic range validation on MindConfig.personality_dimensions (NPC-672)"""

    def test_rejects_value_above_one(self):
        from pydantic import ValidationError
        from mind.interfaces.mcp.models import MindConfig
        with pytest.raises(ValidationError):
            MindConfig(
                traits=["curious"],
                personality_dimensions={"extroversion": 1.5},
            )

    def test_rejects_negative_value(self):
        from pydantic import ValidationError
        from mind.interfaces.mcp.models import MindConfig
        with pytest.raises(ValidationError):
            MindConfig(
                traits=["curious"],
                personality_dimensions={"curiosity": -0.1},
            )

    def test_accepts_in_range_values(self):
        from mind.interfaces.mcp.models import MindConfig
        config = MindConfig(
            traits=["curious"],
            personality_dimensions={
                "extroversion": 0.0,
                "curiosity": 0.5,
                "sensitivity": 1.0,
            },
        )
        assert config.personality_dimensions["extroversion"] == 0.0
        assert config.personality_dimensions["curiosity"] == 0.5
        assert config.personality_dimensions["sensitivity"] == 1.0

    def test_config_has_no_entity_id_field(self):
        """entity_id is a create_mind arg (FK), not config: config is pure cognition."""
        from mind.interfaces.mcp.models import MindConfig
        assert "entity_id" not in MindConfig.model_fields


class TestCreateMindDecouplesIds:
    """create_mind treats mind_id (PK) and entity_id (FK) as distinct first-class ids."""

    @pytest.mark.asyncio
    async def test_distinct_ids_flow_independently(self):
        server = MCPServer()

        result = await server.mcp.call_tool(
            "create_mind",
            {
                "mind_id": "mind_abc",
                "entity_id": "entity_xyz",
                "config": {"traits": ["curious"]},
            },
        )
        response = parse_response(result)

        assert response["status"] == "created"
        # Response carries both ids, kept distinct
        assert response["mind_id"] == "mind_abc"
        assert response["entity_id"] == "entity_xyz"
        # Registry keyed by the mind PK, never the entity FK
        assert "mind_abc" in server.minds
        assert "entity_xyz" not in server.minds
        # Stored Mind keeps both fields, distinct
        mind = server.minds["mind_abc"]
        assert mind.mind_id == "mind_abc"
        assert mind.entity_id == "entity_xyz"
        # Memory collection keyed by the mind PK
        assert mind.memory_store.collection.name == "mind_mind_abc"


class TestDecideActionEntityIdMismatch:
    """decide_action warns (does not reject) when the observation entity_id (FK)
    diverges from the routed mind entity_id, so misrouting is diagnosable (NPC-795)."""

    @staticmethod
    async def _run_decide(server, observation):
        """Create a mind (entity_test) with a stubbed pipeline, run decide_action."""
        from mind.cognitive_architecture.actions import Action
        from mind.cognitive_architecture.state import PipelineState

        await server.mcp.call_tool(
            "create_mind",
            {
                "mind_id": "mind_test",
                "entity_id": "entity_test",
                "config": {
                    "traits": [],
                    "initial_long_term_memories": [],
                },
            },
        )

        mind = server.minds["mind_test"]

        async def mock_process(state: PipelineState) -> PipelineState:
            state.chosen_action = Action.model_construct(action="wait", parameters={})
            return state

        mind.pipeline.process = mock_process

        return await server.mcp.call_tool(
            "decide_action",
            {
                "mind_id": "mind_test",
                "observation": observation,
            },
        )

    @pytest.mark.asyncio
    async def test_warns_when_observation_entity_id_differs_from_mind(self, caplog):
        import logging

        server = MCPServer()
        observation = {
            "entity_id": "entity_other",
            "current_simulation_time": 100,
        }

        with caplog.at_level(logging.WARNING):
            result = await self._run_decide(server, observation)

        response = parse_response(result)
        assert response["status"] == "success"

        mismatch_lines = [
            r.getMessage() for r in caplog.records if "entity_id mismatch" in r.getMessage()
        ]
        assert mismatch_lines, "expected an entity_id mismatch warning"
        assert any("entity_other" in line and "entity_test" in line for line in mismatch_lines)

    @pytest.mark.asyncio
    async def test_silent_when_observation_entity_id_matches_mind(self, caplog):
        import logging

        server = MCPServer()
        observation = {
            "entity_id": "entity_test",
            "current_simulation_time": 100,
        }

        with caplog.at_level(logging.WARNING):
            result = await self._run_decide(server, observation)

        response = parse_response(result)
        assert response["status"] == "success"

        mismatch_lines = [
            r.getMessage() for r in caplog.records if "entity_id mismatch" in r.getMessage()
        ]
        assert not mismatch_lines, "no mismatch warning expected when ids agree"
