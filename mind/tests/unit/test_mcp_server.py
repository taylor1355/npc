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
    """decide_action rejects (after logging both ids) when the observation entity_id
    (FK) diverges from the routed mind entity_id - misrouting is a boundary bug (NPC-795)."""

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
    async def test_rejects_when_observation_entity_id_differs_from_mind(self, caplog):
        import logging

        server = MCPServer()
        observation = {
            "entity_id": "entity_other",
            "current_simulation_time": 100,
        }

        with caplog.at_level(logging.WARNING):
            result = await self._run_decide(server, observation)

        response = parse_response(result)
        assert response["status"] == "error"
        assert "entity_id mismatch" in response["error_message"]

        mismatch_lines = [
            r.getMessage() for r in caplog.records if "entity_id mismatch" in r.getMessage()
        ]
        assert mismatch_lines, "expected an entity_id mismatch warning before the reject"
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


class TestMindPersistenceLifecycle:
    """Server-side mind persistence: release retains memory, relink rebinds/rehydrates,
    forget erases (NPC-797).

    Each test runs in an isolated tmp dir so the default MindConfig
    memory_storage_path ("./chroma_db") - the same path create, relink, and forget
    all use - resolves under a hermetic per-test directory. The autouse fixture also
    clears ChromaDB's process-global client cache: PersistentClient caches its
    system per-process regardless of the resolved path, so without the reset a later
    test's "./chroma_db" would alias the first test's on-disk store and counts would
    bleed across tests (a test-isolation artifact, not server behavior - in
    production the server is one long-lived process keyed by distinct mind_ids).
    """

    @pytest.fixture(autouse=True)
    def _isolated_chroma(self, monkeypatch, tmp_path):
        from chromadb.api.client import SharedSystemClient

        SharedSystemClient.clear_system_cache()
        monkeypatch.chdir(tmp_path)
        yield
        SharedSystemClient.clear_system_cache()

    @staticmethod
    async def _create_mind(server, mind_id, entity_id, initial_memories=None):
        config = {"traits": [], "initial_long_term_memories": initial_memories or []}
        return await server.mcp.call_tool(
            "create_mind",
            {"mind_id": mind_id, "entity_id": entity_id, "config": config},
        )

    @pytest.mark.asyncio
    async def test_relink_resident_mind_rebinds_fk_in_place(self):
        """A still-resident mind keeps its collection and just gets a new entity FK."""
        server = MCPServer()

        await self._create_mind(server, "mind_a", "entity_old")
        mind = server.minds["mind_a"]
        mind.memory_store.add_memory(content="a remembered fact", importance=5.0)
        count_before = mind.memory_store.collection.count()
        assert count_before == 1

        result = await server.mcp.call_tool(
            "relink_mind", {"mind_id": "mind_a", "entity_id": "entity_new"}
        )
        response = parse_response(result)

        assert response["status"] == "relinked"
        # Same live instance, FK rebound in place, memory untouched.
        assert server.minds["mind_a"] is mind
        assert mind.entity_id == "entity_new"
        assert mind.memory_store.collection.count() == count_before

    @pytest.mark.asyncio
    async def test_release_retains_collection_then_relink_rehydrates_with_new_fk(self):
        """cleanup_mind retains the collection; relink rehydrates it with a new FK
        and the memory count is unchanged."""
        server = MCPServer()

        await self._create_mind(server, "mind_b", "entity_old")
        server.minds["mind_b"].memory_store.add_memory(content="retained memory", importance=5.0)
        count_before = server.minds["mind_b"].memory_store.collection.count()
        assert count_before == 1

        release = parse_response(
            await server.mcp.call_tool("cleanup_mind", {"mind_id": "mind_b"})
        )
        assert release["status"] == "released"
        assert "mind_b" not in server.minds

        relink = parse_response(
            await server.mcp.call_tool(
                "relink_mind", {"mind_id": "mind_b", "entity_id": "entity_new"}
            )
        )
        assert relink["status"] == "relinked"

        # Rehydrated instance: collection intact, FK is the new one.
        rehydrated = server.minds["mind_b"]
        assert rehydrated.entity_id == "entity_new"
        assert rehydrated.memory_store.collection.count() == count_before

    @pytest.mark.asyncio
    async def test_relink_does_not_reseed_initial_memories(self):
        """reattach skips the seed loop, so relinking does not double the seeds."""
        server = MCPServer()

        await self._create_mind(
            server, "mind_c", "entity_c", initial_memories=["seed one", "seed two"]
        )
        seed_count = server.minds["mind_c"].memory_store.collection.count()
        assert seed_count == 2

        await server.mcp.call_tool("cleanup_mind", {"mind_id": "mind_c"})
        await server.mcp.call_tool(
            "relink_mind", {"mind_id": "mind_c", "entity_id": "entity_c"}
        )

        # If reattach re-seeded, this would be 4. It must stay 2.
        assert server.minds["mind_c"].memory_store.collection.count() == seed_count

    @pytest.mark.asyncio
    async def test_cleanup_retains_but_forget_deletes_collection(self):
        """collection_exists is True after release, False after forget."""
        from mind.cognitive_architecture.memory.vector_db_memory import VectorDBMemory
        from mind.interfaces.mcp.models import MindConfig

        server = MCPServer()
        storage_path = MindConfig(traits=[]).memory_storage_path

        await self._create_mind(server, "mind_d", "entity_d")
        server.minds["mind_d"].memory_store.add_memory(content="x", importance=5.0)

        await server.mcp.call_tool("cleanup_mind", {"mind_id": "mind_d"})
        assert VectorDBMemory.collection_exists(storage_path, "mind_mind_d") is True

        forget = parse_response(
            await server.mcp.call_tool("forget_mind", {"mind_id": "mind_d"})
        )
        assert forget["status"] == "forgotten"
        assert VectorDBMemory.collection_exists(storage_path, "mind_mind_d") is False

        # A relink after forget finds nothing.
        relink = parse_response(
            await server.mcp.call_tool(
                "relink_mind", {"mind_id": "mind_d", "entity_id": "entity_d"}
            )
        )
        assert relink["status"] == "not_found"

    @pytest.mark.asyncio
    async def test_forget_resident_mind_drops_instance_and_collection(self):
        """forget_mind on a resident mind drops it from the registry and deletes its
        collection (no recreated empty shell)."""
        from mind.cognitive_architecture.memory.vector_db_memory import VectorDBMemory
        from mind.interfaces.mcp.models import MindConfig

        server = MCPServer()
        storage_path = MindConfig(traits=[]).memory_storage_path

        await self._create_mind(server, "mind_e", "entity_e")
        server.minds["mind_e"].memory_store.add_memory(content="y", importance=5.0)

        forget = parse_response(
            await server.mcp.call_tool("forget_mind", {"mind_id": "mind_e"})
        )
        assert forget["status"] == "forgotten"
        assert forget["entity_id"] == "entity_e"
        assert "mind_e" not in server.minds
        assert VectorDBMemory.collection_exists(storage_path, "mind_mind_e") is False

    @pytest.mark.asyncio
    async def test_restart_reattach_recovers_memory_across_server_instances(self):
        """Simulate a server restart: create + memory on one instance, drop it, then a
        fresh MCPServer relinks the same mind_id and the memory survives.

        Note: clearing ChromaDB's client cache between server instances (the autouse
        fixture does this at setup/teardown, not mid-test) is unnecessary here -
        within one process the cached client correctly reopens the on-disk store,
        which is exactly the restart-recovery path under test.
        """
        server1 = MCPServer()
        await self._create_mind(server1, "mind_f", "entity_f")
        server1.minds["mind_f"].memory_store.add_memory(
            content="survives restart", importance=5.0
        )
        count_before = server1.minds["mind_f"].memory_store.collection.count()
        assert count_before == 1

        # Drop the whole server instance (collection is persisted on disk).
        del server1

        server2 = MCPServer()
        assert "mind_f" not in server2.minds
        relink = parse_response(
            await server2.mcp.call_tool(
                "relink_mind", {"mind_id": "mind_f", "entity_id": "entity_f"}
            )
        )
        assert relink["status"] == "relinked"
        assert server2.minds["mind_f"].memory_store.collection.count() == count_before
