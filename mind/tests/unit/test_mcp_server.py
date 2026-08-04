"""Unit tests for MCP server"""

import json
import logging
from unittest.mock import patch

import pytest

from mind.constants import DEFAULT_MEMORY_STORAGE_PATH
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

        release = parse_response(await server.mcp.call_tool("cleanup_mind", {"mind_id": "mind_b"}))
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
        await server.mcp.call_tool("relink_mind", {"mind_id": "mind_c", "entity_id": "entity_c"})

        # If reattach re-seeded, this would be 4. It must stay 2.
        assert server.minds["mind_c"].memory_store.collection.count() == seed_count

    @pytest.mark.asyncio
    async def test_cleanup_retains_but_forget_deletes_collection(self):
        """collection_exists is True after release, False after forget."""
        from mind.cognitive_architecture.memory.vector_db_memory import VectorDBMemory

        server = MCPServer()
        storage_path = DEFAULT_MEMORY_STORAGE_PATH

        await self._create_mind(server, "mind_d", "entity_d")
        server.minds["mind_d"].memory_store.add_memory(content="x", importance=5.0)

        await server.mcp.call_tool("cleanup_mind", {"mind_id": "mind_d"})
        assert VectorDBMemory.collection_exists(storage_path, "mind_mind_d") is True

        forget = parse_response(await server.mcp.call_tool("forget_mind", {"mind_id": "mind_d"}))
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

        server = MCPServer()
        storage_path = DEFAULT_MEMORY_STORAGE_PATH

        await self._create_mind(server, "mind_e", "entity_e")
        server.minds["mind_e"].memory_store.add_memory(content="y", importance=5.0)

        forget = parse_response(await server.mcp.call_tool("forget_mind", {"mind_id": "mind_e"}))
        assert forget["status"] == "forgotten"
        assert forget["entity_id"] == "entity_e"
        assert "mind_e" not in server.minds
        assert VectorDBMemory.collection_exists(storage_path, "mind_mind_e") is False

    @pytest.mark.asyncio
    async def test_forget_resident_mind_tolerates_an_already_deleted_collection(self):
        """forget_mind still drops the registry entry when the collection is already gone.

        The load-bearing assertion is that mind_h has left server.minds, not that the
        call avoided raising. An unguarded raise on the delete skipped the registry
        drop on the following line, so the caller was told the forget failed while the
        mind stayed resident and a later relink_mind would still find it. Swallowing
        the exception without restoring that invariant would be the same defect with a
        quieter symptom.
        """
        server = MCPServer()

        await self._create_mind(server, "mind_h", "entity_h")
        mind = server.minds["mind_h"]
        mind.memory_store.add_memory(content="z", importance=5.0)

        # Delete the collection out from under the resident store, so forget_mind's
        # resident branch meets an absent collection.
        mind.memory_store.client.delete_collection("mind_mind_h")

        forget = parse_response(await server.mcp.call_tool("forget_mind", {"mind_id": "mind_h"}))

        assert forget["status"] == "forgotten"
        assert forget["entity_id"] == "entity_h"
        assert "mind_h" not in server.minds

        # Gone for good, not merely dropped from the registry.
        relink = parse_response(
            await server.mcp.call_tool(
                "relink_mind", {"mind_id": "mind_h", "entity_id": "entity_h"}
            )
        )
        assert relink["status"] == "not_found"

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
        server1.minds["mind_f"].memory_store.add_memory(content="survives restart", importance=5.0)
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

    def test_collection_exists_is_false_and_creates_no_dir_for_missing_path(self):
        """collection_exists must be a pure read: a never-persisted path returns False
        and is NOT created on disk as a side effect (PersistentClient otherwise mkdir's
        the path). Regression for the empty-DB-dir leak (PR #17 review)."""
        import os

        from mind.cognitive_architecture.memory.vector_db_memory import VectorDBMemory

        missing_path = os.path.join(os.getcwd(), "never_persisted_db")
        assert not os.path.exists(missing_path)

        assert VectorDBMemory.collection_exists(missing_path, "mind_ghost") is False

        # The probe must not have created the directory.
        assert not os.path.exists(missing_path)

    @pytest.mark.asyncio
    async def test_forget_non_resident_deletes_collection_without_loading_encoder(self):
        """Forgetting a non-resident retained collection must delete it via a bare
        client - no SentenceTransformer load, no get_or_create that would recreate the
        collection before deleting it. Regression for the heavy-construct forget path
        (PR #17 review)."""
        from mind.cognitive_architecture.memory.vector_db_memory import VectorDBMemory

        server = MCPServer()
        storage_path = DEFAULT_MEMORY_STORAGE_PATH

        # Create + release so the collection is retained on disk but no live Mind holds it.
        await self._create_mind(server, "mind_g", "entity_g")
        server.minds["mind_g"].memory_store.add_memory(content="z", importance=5.0)
        await server.mcp.call_tool("cleanup_mind", {"mind_id": "mind_g"})
        assert "mind_g" not in server.minds
        assert VectorDBMemory.collection_exists(storage_path, "mind_mind_g") is True

        # Patch the encoder constructor on the module VectorDBMemory uses; the
        # non-resident forget path must never instantiate it.
        with patch(
            "mind.cognitive_architecture.memory.vector_db_memory.SentenceTransformer"
        ) as encoder_ctor:
            forget = parse_response(
                await server.mcp.call_tool("forget_mind", {"mind_id": "mind_g"})
            )

        assert forget["status"] == "forgotten"
        encoder_ctor.assert_not_called()
        # Collection is gone (not recreated as an empty shell).
        assert VectorDBMemory.collection_exists(storage_path, "mind_mind_g") is False


class TestCustomStoragePathLifecycle:
    """relink_mind / forget_mind must address the path a mind was CREATED with,
    not the default one (NPC-1023).

    memory_storage_path is client-settable per mind at create_mind time, but both
    tools used to build their probe from a throwaway ``MindConfig(traits=[])`` and
    therefore always looked under the default "./chroma_db". A mind on a custom path
    was reachable only while resident: once released, relink reported "not_found"
    and forget reported "forgotten" over a collection that was still on disk.

    Isolation matches TestMindPersistenceLifecycle: a per-test cwd plus a cleared
    ChromaDB process-global client cache, so both the default and the custom path
    resolve under a hermetic directory.
    """

    CUSTOM_PATH = "./tmp/custom_chroma_db"

    @pytest.fixture(autouse=True)
    def _isolated_chroma(self, monkeypatch, tmp_path):
        from chromadb.api.client import SharedSystemClient

        SharedSystemClient.clear_system_cache()
        monkeypatch.chdir(tmp_path)
        yield
        SharedSystemClient.clear_system_cache()

    @classmethod
    async def _create_mind(cls, server, mind_id, entity_id):
        config = {"traits": [], "memory_storage_path": cls.CUSTOM_PATH}
        return await server.mcp.call_tool(
            "create_mind",
            {"mind_id": mind_id, "entity_id": entity_id, "config": config},
        )

    @pytest.mark.asyncio
    async def test_relink_rehydrates_a_released_mind_on_a_custom_storage_path(self):
        """Release then relink must recover the mind, not report "not_found".

        This is the designed retain-on-release contract: cleanup_mind deliberately
        keeps the collection so a later relink can re-attach. Probing the default
        path breaks that contract for every mind not on "./chroma_db".
        """
        from mind.cognitive_architecture.memory.vector_db_memory import VectorDBMemory

        server = MCPServer()
        await self._create_mind(server, "mind_cp", "entity_cp")
        server.minds["mind_cp"].memory_store.add_memory(content="custom", importance=5.0)

        await server.mcp.call_tool("cleanup_mind", {"mind_id": "mind_cp"})
        assert "mind_cp" not in server.minds
        # The collection is retained - on the custom path, where it was created.
        assert VectorDBMemory.collection_exists(self.CUSTOM_PATH, "mind_mind_cp") is True
        # The lifetime rule that makes the relink below possible, asserted directly
        # rather than only through its consequence: release must NOT drop the config,
        # or the retained collection becomes unaddressable.
        assert "mind_cp" in server.mind_configs

        relink = parse_response(
            await server.mcp.call_tool(
                "relink_mind", {"mind_id": "mind_cp", "entity_id": "entity_cp"}
            )
        )

        assert relink["status"] == "relinked"
        # Rehydrated against the retained collection, so the memory survives.
        assert server.minds["mind_cp"].memory_store.collection.count() == 1

    @pytest.mark.asyncio
    async def test_forget_erases_a_released_mind_on_a_custom_storage_path(self):
        """forget_mind must actually delete the custom-path collection.

        The load-bearing assertion is collection_exists, not the status string: the
        defect reported "forgotten" while the collection survived on disk, telling
        the caller the mind was erased when it was not. A status that lies about a
        destructive operation is worse than an honest "not_found".
        """
        from mind.cognitive_architecture.memory.vector_db_memory import VectorDBMemory

        server = MCPServer()
        await self._create_mind(server, "mind_cq", "entity_cq")
        server.minds["mind_cq"].memory_store.add_memory(content="erase me", importance=5.0)

        await server.mcp.call_tool("cleanup_mind", {"mind_id": "mind_cq"})
        assert VectorDBMemory.collection_exists(self.CUSTOM_PATH, "mind_mind_cq") is True

        forget = parse_response(await server.mcp.call_tool("forget_mind", {"mind_id": "mind_cq"}))

        assert forget["status"] == "forgotten"
        assert VectorDBMemory.collection_exists(self.CUSTOM_PATH, "mind_mind_cq") is False
        # The other half of the lifetime rule: forget destroys the target, so it is
        # the one operation that may drop the recorded config.
        assert "mind_cq" not in server.mind_configs

    @pytest.mark.asyncio
    async def test_forget_reports_not_found_when_no_collection_was_erased(self):
        """An unknown mind_id must not be reported as "forgotten".

        Boundary integrity: "forgotten" is a claim that memory was destroyed. When
        nothing was resident and nothing was deleted, the honest answer is
        "not_found" - the same answer relink_mind already gives.
        """
        server = MCPServer()

        forget = parse_response(
            await server.mcp.call_tool("forget_mind", {"mind_id": "never_existed"})
        )

        assert forget["status"] == "not_found"

    @pytest.mark.asyncio
    async def test_second_forget_of_the_same_mind_reports_not_found(self):
        """The repeat case a retrying client actually produces, as opposed to an id
        that never existed.

        forget_mind used to be unconditionally idempotent - it always answered
        "forgotten", including over a mind it had already erased. That made a retry
        indistinguishable from a successful first erase. The honest answer on the
        second call is "not_found": nothing was erased, because there was nothing
        left to erase.
        """
        from mind.cognitive_architecture.memory.vector_db_memory import VectorDBMemory

        server = MCPServer()
        await self._create_mind(server, "mind_cr", "entity_cr")

        first = parse_response(await server.mcp.call_tool("forget_mind", {"mind_id": "mind_cr"}))
        assert first["status"] == "forgotten"
        assert VectorDBMemory.collection_exists(self.CUSTOM_PATH, "mind_mind_cr") is False

        second = parse_response(await server.mcp.call_tool("forget_mind", {"mind_id": "mind_cr"}))
        assert second["status"] == "not_found"


class TestRestartStoragePathParameter:
    """A restarted server can address a custom-path mind when the client supplies the
    path (NPC-1023).

    self.mind_configs is process-local, so a restart empties it. Before the optional
    memory_storage_path parameter existed, that left a custom-path mind permanently
    unreachable: relink_mind probed the default path and reported "not_found", and
    forget_mind could not erase it.

    "Restart" here is a fresh MCPServer over the same on-disk directory - the same
    idiom TestMindPersistenceLifecycle uses - which is exactly what distinguishes this
    from eviction: eviction keeps the map, a restart does not.
    """

    CUSTOM_PATH = "./tmp/restart_chroma_db"

    @pytest.fixture(autouse=True)
    def _isolated_chroma(self, monkeypatch, tmp_path):
        from chromadb.api.client import SharedSystemClient

        SharedSystemClient.clear_system_cache()
        monkeypatch.chdir(tmp_path)
        yield
        SharedSystemClient.clear_system_cache()

    @classmethod
    async def _create_mind(cls, server, mind_id, entity_id):
        config = {"traits": [], "memory_storage_path": cls.CUSTOM_PATH}
        return await server.mcp.call_tool(
            "create_mind",
            {"mind_id": mind_id, "entity_id": entity_id, "config": config},
        )

    @pytest.mark.asyncio
    async def test_relink_after_restart_recovers_a_custom_path_mind_when_path_supplied(self):
        """The restart case: a fresh server has no record, so the client's path is the
        only thing that can locate the collection."""
        server1 = MCPServer()
        await self._create_mind(server1, "mind_rs", "entity_rs")
        server1.minds["mind_rs"].memory_store.add_memory(content="survives", importance=5.0)
        count_before = server1.minds["mind_rs"].memory_store.collection.count()
        assert count_before == 1

        # Restart: the new server shares the on-disk store but not the config map.
        del server1
        server2 = MCPServer()
        assert "mind_rs" not in server2.mind_configs

        relink = parse_response(
            await server2.mcp.call_tool(
                "relink_mind",
                {
                    "mind_id": "mind_rs",
                    "entity_id": "entity_rs",
                    "memory_storage_path": self.CUSTOM_PATH,
                },
            )
        )

        assert relink["status"] == "relinked"
        assert server2.minds["mind_rs"].memory_store.collection.count() == count_before

    @pytest.mark.asyncio
    async def test_relink_after_restart_still_not_found_without_the_path(self):
        """Omitting the parameter preserves the old behavior exactly.

        This is what keeps the change backward-compatible: a client that does not send
        memory_storage_path gets the same default-path probe - and the same honest
        "not_found" - as before.
        """
        server1 = MCPServer()
        await self._create_mind(server1, "mind_rt", "entity_rt")
        del server1

        server2 = MCPServer()
        relink = parse_response(
            await server2.mcp.call_tool(
                "relink_mind", {"mind_id": "mind_rt", "entity_id": "entity_rt"}
            )
        )

        assert relink["status"] == "not_found"

    @pytest.mark.asyncio
    async def test_forget_after_restart_erases_a_custom_path_mind_when_path_supplied(self):
        """The worst half of NPC-1023, at restart scope.

        The load-bearing assertion is collection_exists, not the status string: a
        forget that answers "forgotten" while the collection survives tells the caller
        their data is destroyed when it is not. A status-only test would pass against
        exactly that bug.
        """
        from mind.cognitive_architecture.memory.vector_db_memory import VectorDBMemory

        server1 = MCPServer()
        await self._create_mind(server1, "mind_ru", "entity_ru")
        server1.minds["mind_ru"].memory_store.add_memory(content="erase me", importance=5.0)
        del server1

        server2 = MCPServer()
        assert VectorDBMemory.collection_exists(self.CUSTOM_PATH, "mind_mind_ru") is True

        forget = parse_response(
            await server2.mcp.call_tool(
                "forget_mind",
                {"mind_id": "mind_ru", "memory_storage_path": self.CUSTOM_PATH},
            )
        )

        assert forget["status"] == "forgotten"
        assert VectorDBMemory.collection_exists(self.CUSTOM_PATH, "mind_mind_ru") is False

    @pytest.mark.asyncio
    async def test_forget_after_restart_reports_not_found_without_the_path(self):
        """Without the path the collection is not reachable, so nothing is erased -
        and the status must say so rather than claim a deletion it did not perform."""
        from mind.cognitive_architecture.memory.vector_db_memory import VectorDBMemory

        server1 = MCPServer()
        await self._create_mind(server1, "mind_rv", "entity_rv")
        del server1

        server2 = MCPServer()
        forget = parse_response(await server2.mcp.call_tool("forget_mind", {"mind_id": "mind_rv"}))

        assert forget["status"] == "not_found"
        # The collection is untouched, which is what makes "not_found" the honest answer.
        assert VectorDBMemory.collection_exists(self.CUSTOM_PATH, "mind_mind_rv") is True

    @pytest.mark.asyncio
    async def test_supplied_path_is_ignored_when_a_config_is_recorded(self, caplog):
        """Precedence rule 1: the recorded config beats a caller-supplied path.

        Every other test in this class runs against an empty map, so they all exercise
        case 2 of _config_for. This one pins rule 1 itself - the rule that makes
        memory_storage_path a restart-only fallback rather than a relocation
        instruction. Without it a refactor could invert the precedence and stay green.

        It also pins that the discard is LOGGED. The warning is the entire mitigation
        for the destructive case _config_for documents - a forget_mind that reports
        success over a location the caller never named - so a silent discard is the
        failure this branch exists to prevent, not a cosmetic regression.
        """
        from mind.cognitive_architecture.memory.vector_db_memory import VectorDBMemory

        wrong_path = "./tmp/wrong_path"

        server = MCPServer()
        await self._create_mind(server, "mind_rw", "entity_rw")
        server.minds["mind_rw"].memory_store.add_memory(content="recorded", importance=5.0)
        await server.mcp.call_tool("cleanup_mind", {"mind_id": "mind_rw"})

        with caplog.at_level(logging.WARNING, logger="mind"):
            relink = parse_response(
                await server.mcp.call_tool(
                    "relink_mind",
                    {
                        "mind_id": "mind_rw",
                        "entity_id": "entity_rw",
                        "memory_storage_path": wrong_path,
                    },
                )
            )

        assert relink["status"] == "relinked"
        # The load-bearing assertion: the rehydrated store holds what was written to
        # CUSTOM_PATH, so the recorded path was used - not merely that wrong_path was
        # left alone, which an unrelated failure to relink would also satisfy.
        assert server.minds["mind_rw"].memory_store.collection.count() == 1
        # And the bogus path was never even created on disk.
        assert VectorDBMemory.collection_exists(wrong_path, "mind_mind_rw") is False
        # The discard is diagnosable rather than silent. Both paths are named, so the
        # client can tell which one it sent and which one won.
        assert wrong_path in caplog.text
        assert self.CUSTOM_PATH in caplog.text

    @pytest.mark.asyncio
    async def test_an_equivalent_path_spelling_does_not_warn(self, caplog):
        """The warning must not fire on a caller who AGREES with the record.

        "./x", "x" and "x/" are the same directory. A diagnostic that flags them as a
        mismatch trains its reader to ignore it, which costs exactly the stale-path
        case the warning exists for. Behavior is unaffected either way (the record
        wins), so this pins the diagnostic's precision, not its effect.
        """
        server = MCPServer()
        await self._create_mind(server, "mind_rx", "entity_rx")
        await server.mcp.call_tool("cleanup_mind", {"mind_id": "mind_rx"})

        # Same directory as CUSTOM_PATH ("./tmp/restart_chroma_db"), spelled without
        # the leading "./" and with a trailing separator.
        equivalent = self.CUSTOM_PATH.removeprefix("./") + "/"

        with caplog.at_level(logging.WARNING, logger="mind"):
            relink = parse_response(
                await server.mcp.call_tool(
                    "relink_mind",
                    {
                        "mind_id": "mind_rx",
                        "entity_id": "entity_rx",
                        "memory_storage_path": equivalent,
                    },
                )
            )

        assert relink["status"] == "relinked"
        # Assert on captured records, not on message text. Keying the negative on the
        # warning's wording would go vacuous the moment anyone rewords it - and the
        # positive test above would not notice, because it asserts on the paths rather
        # than the prefix. This form also catches any OTHER spurious warning from
        # project code, which is the property actually under test.
        #
        # Filtered to the "mind" namespace because at_level(logger="mind") sets the
        # LEVEL on that logger without scoping CAPTURE to it - caplog's handler sits on
        # root, so caplog.records also holds anything third-party that propagates there.
        # This block builds a real SentenceTransformer and PersistentClient via
        # Mind.reattach, so an unfiltered assertion would redden on a sentence_transformers
        # or chromadb warning from a version bump, with a failure naming nothing about
        # path spellings.
        assert [
            r for r in caplog.records if r.levelno >= logging.WARNING and r.name.startswith("mind")
        ] == []


class TestRecordedConfigFidelity:
    """relink_mind must rebuild a mind from the config it was CREATED with, not a
    default-constructed one (NPC-1023).

    memory_storage_path is the self-announcing field - get it wrong and relink says
    "not_found". These tests cover the quiet ones. embedding_model is the sharpest:
    Mind.reattach passes it to VectorDBMemory, which constructs its own
    SentenceTransformer and embeds queries with it, so rehydrating under the default
    model means querying a collection with vectors from a different model - a
    dimension error at best and silently meaningless neighbours at worst.
    """

    @pytest.fixture(autouse=True)
    def _isolated_chroma(self, monkeypatch, tmp_path):
        from chromadb.api.client import SharedSystemClient

        SharedSystemClient.clear_system_cache()
        monkeypatch.chdir(tmp_path)
        yield
        SharedSystemClient.clear_system_cache()

    @pytest.mark.asyncio
    async def test_relink_reattaches_with_the_recorded_embedding_model(self):
        """The encoder the rehydrated store builds must be the one that wrote the
        stored vectors.

        SentenceTransformer is patched so the assertion is about which model name is
        requested, not about downloading a real model - and so a fictitious name can
        stand in for "not the default" without touching the network.
        """
        custom_model = "test-only-embedding-model"

        with patch(
            "mind.cognitive_architecture.memory.vector_db_memory.SentenceTransformer"
        ) as encoder_ctor:
            server = MCPServer()
            await server.mcp.call_tool(
                "create_mind",
                {
                    "mind_id": "mind_em",
                    "entity_id": "entity_em",
                    "config": {"traits": [], "embedding_model": custom_model},
                },
            )
            await server.mcp.call_tool("cleanup_mind", {"mind_id": "mind_em"})

            # Only the reattach-time construction is under test.
            encoder_ctor.reset_mock()
            relink = parse_response(
                await server.mcp.call_tool(
                    "relink_mind", {"mind_id": "mind_em", "entity_id": "entity_em"}
                )
            )

        assert relink["status"] == "relinked"
        encoder_ctor.assert_called_once_with(custom_model)

    @pytest.mark.asyncio
    async def test_relink_reattaches_with_the_recorded_llm_model(self):
        """llm_model is recorded for the same reason as the rest, and fails the same way.

        A mind rehydrated on the default LLM is a different reasoner than the one the
        client paid to configure, and nothing in the relink response says so. get_llm is
        patched so the assertion is about which model name is requested, without
        constructing a real client.
        """
        custom_model = "test-only/llm-model"

        with patch("mind.interfaces.mcp.mind.get_llm") as get_llm_mock:
            server = MCPServer()
            await server.mcp.call_tool(
                "create_mind",
                {
                    "mind_id": "mind_lm",
                    "entity_id": "entity_lm",
                    "config": {"traits": [], "llm_model": custom_model},
                },
            )
            await server.mcp.call_tool("cleanup_mind", {"mind_id": "mind_lm"})

            # Only the reattach-time construction is under test.
            get_llm_mock.reset_mock()
            relink = parse_response(
                await server.mcp.call_tool(
                    "relink_mind", {"mind_id": "mind_lm", "entity_id": "entity_lm"}
                )
            )

        assert relink["status"] == "relinked"
        get_llm_mock.assert_called_once_with(custom_model)

    @pytest.mark.asyncio
    async def test_relink_restores_traits_and_personality_from_the_recorded_config(self):
        """A rehydrated mind must be the same character it was before release.

        Rebuilding from MindConfig(traits=[]) returns a mind with no traits and no
        personality dimensions - the NPC comes back a stranger, with nothing in the
        response to indicate it.
        """
        server = MCPServer()
        await server.mcp.call_tool(
            "create_mind",
            {
                "mind_id": "mind_id_fid",
                "entity_id": "entity_fid",
                "config": {
                    "traits": ["gruff", "loyal"],
                    "personality_dimensions": {"extroversion": 0.9},
                },
            },
        )
        await server.mcp.call_tool("cleanup_mind", {"mind_id": "mind_id_fid"})

        relink = parse_response(
            await server.mcp.call_tool(
                "relink_mind", {"mind_id": "mind_id_fid", "entity_id": "entity_fid_new"}
            )
        )
        assert relink["status"] == "relinked"

        rehydrated = server.minds["mind_id_fid"]
        assert rehydrated.traits == ["gruff", "loyal"]
        assert rehydrated.personality_dimensions == {"extroversion": 0.9}

    @pytest.mark.asyncio
    async def test_recorded_config_drops_the_seed_payload(self):
        """The map records how to rebuild a mind, not what it was seeded with.

        The two dropped fields are dropped for different reasons, so this test pins two
        different claims:

        initial_long_term_memories is ignored by Mind.reattach by design (re-seeding on
        every relink would duplicate the originals), so retaining it would pin every
        NPC's seed text for the process lifetime to no purpose. Dropping it is what
        keeps each entry bounded, and makes the never-re-seed contract structural.

        initial_working_memory is NOT ignored - Mind.reattach reads it directly, and
        there is no live mind in the reattach branch to take it from - so dropping it is
        a deliberate choice that a relinked mind starts blank. The creating snapshot is
        stale by relink time, so replaying it would resurrect old state rather than
        restore fidelity. See _config_to_record.
        """
        server = MCPServer()
        await server.mcp.call_tool(
            "create_mind",
            {
                "mind_id": "mind_seed",
                "entity_id": "entity_seed",
                "config": {
                    "traits": ["curious"],
                    "initial_long_term_memories": ["a seeded memory"],
                },
            },
        )

        recorded = server.mind_configs["mind_seed"]
        assert recorded.initial_long_term_memories == []
        assert recorded.initial_working_memory is None
        # The fields that DO drive a rebuild are kept.
        assert recorded.traits == ["curious"]
