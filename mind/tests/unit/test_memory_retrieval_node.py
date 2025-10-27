"""Unit tests for MemoryRetrievalNode"""

from unittest.mock import AsyncMock

import pytest

from mind.cognitive_architecture.models import Memory, Observation, StatusObservation
from mind.cognitive_architecture.nodes.memory_retrieval.node import MemoryRetrievalNode
from mind.cognitive_architecture.state import PipelineState


@pytest.mark.asyncio
class TestMemoryRetrievalNode:
    """Test MemoryRetrievalNode in isolation"""

    @pytest.fixture
    def mock_memory_store(self):
        """Create a mock memory store"""
        mock = AsyncMock()
        mock.search.return_value = []  # Default empty return
        return mock

    @pytest.fixture
    def node(self, mock_memory_store):
        """Create a MemoryRetrievalNode with mocked store"""
        return MemoryRetrievalNode(mock_memory_store)

    @pytest.fixture
    def basic_state(self):
        """Create a basic pipeline state"""
        return PipelineState(
            observation=Observation(
                entity_id="test_npc",
                current_simulation_time=100,
                status=StatusObservation(position=(5, 10), movement_locked=False),
            ),
            memory_queries=["recent blacksmithing work", "sword commission"],
        )

    async def test_retrieves_memories_for_queries(self, node, mock_memory_store, basic_state):
        """Should fetch memories for each query"""
        # Setup mock to return memories
        mock_memory_store.search.return_value = [
            Memory(id="mem_1", content="Yesterday I worked on a sword", importance=7.0),
            Memory(id="mem_2", content="Customer ordered ceremonial blade", importance=8.0),
        ]

        # Execute
        result = await node.process(basic_state)

        # Verify search was called for each query
        assert mock_memory_store.search.call_count == 2

        # Verify memories were added to state
        assert len(result.retrieved_memories) > 0

    async def test_deduplicates_by_memory_id(self, node, mock_memory_store, basic_state):
        """Should remove duplicate memories from multiple queries"""
        # Setup mock to return overlapping memories
        memory_1 = Memory(id="mem_1", content="Sword work", importance=7.0)
        memory_2 = Memory(id="mem_2", content="Blade order", importance=8.0)

        # Both queries return same memory_1 plus unique memories
        mock_memory_store.search.side_effect = [
            [memory_1, memory_2],  # First query
            [memory_1, Memory(id="mem_3", content="Forge hot", importance=5.0)],  # Second query
        ]

        # Execute
        result = await node.process(basic_state)

        # Should have 3 unique memories (mem_1 only once)
        memory_ids = [m.id for m in result.retrieved_memories]
        assert len(memory_ids) == 3
        assert len(set(memory_ids)) == 3  # All unique

    async def test_handles_empty_queries(self, node, mock_memory_store):
        """Should handle state with no memory queries"""
        state = PipelineState(
            observation=Observation(
                entity_id="test_npc",
                current_simulation_time=100,
                status=StatusObservation(position=(5, 10), movement_locked=False),
            ),
            memory_queries=[],  # No queries
        )

        # Execute
        result = await node.process(state)

        # Should not call search
        mock_memory_store.search.assert_not_called()

        # Should have empty memories list
        assert result.retrieved_memories == []

    async def test_handles_no_results(self, node, mock_memory_store, basic_state):
        """Should handle memory store returning no results"""
        # Setup mock to return empty list
        mock_memory_store.search.return_value = []

        # Execute
        result = await node.process(basic_state)

        # Should have empty memories list
        assert result.retrieved_memories == []

    async def test_tracks_timing(self, node, mock_memory_store, basic_state):
        """Should track execution time in state"""
        mock_memory_store.search.return_value = []

        # Execute
        result = await node.process(basic_state)

        # Should have timing info
        assert "memory_retrieval" in result.time_ms
        assert result.time_ms["memory_retrieval"] >= 0

    async def test_preserves_other_state(self, node, mock_memory_store):
        """Should not modify unrelated state fields"""
        state = PipelineState(
            observation=Observation(
                entity_id="test_npc",
                current_simulation_time=100,
                status=StatusObservation(position=(5, 10), movement_locked=False),
            ),
            memory_queries=["test query"],
            personality_traits=["brave", "honest"],
            cognitive_context={"test": "data"},
        )

        mock_memory_store.search.return_value = []

        # Execute
        result = await node.process(state)

        # Should preserve other fields
        assert result.personality_traits == ["brave", "honest"]
        assert result.cognitive_context == {"test": "data"}
        assert result.observation == state.observation
