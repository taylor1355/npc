"""Unit tests for MemoryQueryNode"""

from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage

from mind.cognitive_architecture.models import Observation, StatusObservation
from mind.cognitive_architecture.nodes.cognitive_update.models import WorkingMemory
from mind.cognitive_architecture.nodes.memory_query.node import MemoryQueryNode
from mind.cognitive_architecture.state import PipelineState


@pytest.mark.asyncio
class TestMemoryQueryNode:
    """Test MemoryQueryNode in isolation"""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM that returns memory queries"""
        mock = AsyncMock()
        # Default response with 3 diverse queries
        mock.ainvoke.return_value = AIMessage(
            content="""{
                "queries": [
                    "blacksmithing techniques I learned",
                    "customer orders for ceremonial items",
                    "recent forge work and accomplishments"
                ]
            }""",
            usage_metadata={"input_tokens": 150, "output_tokens": 40, "total_tokens": 190},
        )
        return mock

    @pytest.fixture
    def node(self, mock_llm):
        """Create a MemoryQueryNode with mocked LLM"""
        return MemoryQueryNode(mock_llm)

    @pytest.fixture
    def basic_state(self):
        """Create a basic pipeline state"""
        return PipelineState(
            observation=Observation(
                entity_id="test_npc",
                current_simulation_time=100,
                status=StatusObservation(position=(5, 10), movement_locked=False),
            ),
            working_memory=WorkingMemory(
                situation_assessment="At the forge",
                active_goals=["Complete sword order"],
                emotional_state="Focused",
            ),
        )

    async def test_populates_memory_queries_field(self, node, mock_llm, basic_state):
        """Should populate memory_queries in state"""
        assert basic_state.memory_queries == []

        result = await node.process(basic_state)

        assert len(result.memory_queries) > 0
        assert isinstance(result.memory_queries, list)
        assert all(isinstance(q, str) for q in result.memory_queries)

    async def test_returns_queries_from_llm_output(self, node, mock_llm, basic_state):
        """Should use queries specified by LLM"""
        result = await node.process(basic_state)

        assert len(result.memory_queries) == 3
        assert "blacksmithing techniques I learned" in result.memory_queries
        assert "customer orders for ceremonial items" in result.memory_queries
        assert "recent forge work and accomplishments" in result.memory_queries

    async def test_tracks_token_usage(self, node, mock_llm, basic_state):
        """Should track token usage in state"""
        result = await node.process(basic_state)

        assert "memory_query" in result.tokens_used
        assert result.tokens_used["memory_query"] == 190

    async def test_tracks_timing(self, node, mock_llm, basic_state):
        """Should track execution time in state"""
        result = await node.process(basic_state)

        assert "memory_query" in result.time_ms
        assert result.time_ms["memory_query"] >= 0

    async def test_handles_single_query(self, node, mock_llm, basic_state):
        """Should handle LLM returning single query"""
        mock_llm.ainvoke.return_value = AIMessage(
            content="""{
                "queries": ["recent blacksmithing work"]
            }""",
            usage_metadata={"input_tokens": 150, "output_tokens": 20, "total_tokens": 170},
        )

        result = await node.process(basic_state)

        assert len(result.memory_queries) == 1
        assert result.memory_queries[0] == "recent blacksmithing work"

    async def test_handles_maximum_queries(self, node, mock_llm, basic_state):
        """Should handle LLM returning maximum allowed queries (5)"""
        mock_llm.ainvoke.return_value = AIMessage(
            content="""{
                "queries": [
                    "query 1",
                    "query 2",
                    "query 3",
                    "query 4",
                    "query 5"
                ]
            }""",
            usage_metadata={"input_tokens": 150, "output_tokens": 50, "total_tokens": 200},
        )

        result = await node.process(basic_state)

        assert len(result.memory_queries) == 5

    async def test_preserves_other_state_fields(self, node, mock_llm, basic_state):
        """Should not modify unrelated state fields"""
        original_working_memory = basic_state.working_memory
        original_observation = basic_state.observation

        result = await node.process(basic_state)

        assert result.working_memory == original_working_memory
        assert result.observation == original_observation

    async def test_llm_called_once_per_process(self, node, mock_llm, basic_state):
        """Should call LLM exactly once per process call"""
        await node.process(basic_state)

        assert mock_llm.ainvoke.call_count == 1

    async def test_replaces_existing_queries(self, node, mock_llm, basic_state):
        """Should replace existing queries rather than appending"""
        # Pre-populate with old queries
        basic_state.memory_queries = ["old query 1", "old query 2"]

        result = await node.process(basic_state)

        # Should have new queries only, not old ones
        assert len(result.memory_queries) == 3
        assert "old query 1" not in result.memory_queries
        assert "blacksmithing techniques I learned" in result.memory_queries
