"""Unit tests for MemoryConsolidationNode"""

from unittest.mock import MagicMock

import pytest

from mind.cognitive_architecture.observations import Observation, StatusObservation
from mind.cognitive_architecture.nodes.cognitive_update.models import NewMemory
from mind.cognitive_architecture.nodes.memory_consolidation.node import MemoryConsolidationNode
from mind.cognitive_architecture.state import PipelineState


@pytest.mark.asyncio
class TestMemoryConsolidationNode:
    """Test MemoryConsolidationNode in isolation"""

    @pytest.fixture
    def mock_memory_store(self):
        """Create a mock memory store"""
        mock = MagicMock()
        return mock

    @pytest.fixture
    def node(self, mock_memory_store):
        """Create a MemoryConsolidationNode with mocked store"""
        return MemoryConsolidationNode(mock_memory_store)

    @pytest.fixture
    def basic_state(self):
        """Create a basic pipeline state with daily memories"""
        return PipelineState(
            observation=Observation(
                entity_id="test_npc",
                current_simulation_time=1500,
                status=StatusObservation(position=(10, 15), movement_locked=False),
            ),
            daily_memories=[
                NewMemory(content="Forged a ceremonial blade", importance=8.0),
                NewMemory(content="Customer was very pleased", importance=7.5),
                NewMemory(content="Learned new tempering technique", importance=9.0),
            ],
        )

    async def test_adds_memories_to_store(self, node, mock_memory_store, basic_state):
        """Should add all daily memories to memory store"""
        await node.process(basic_state)

        # Should have called add_memory for each daily memory
        assert mock_memory_store.add_memory.call_count == 3

    async def test_clears_daily_memories(self, node, mock_memory_store, basic_state):
        """Should clear daily_memories list after consolidation"""
        assert len(basic_state.daily_memories) == 3

        result = await node.process(basic_state)

        assert len(result.daily_memories) == 0

    async def test_passes_correct_content_and_importance(
        self, node, mock_memory_store, basic_state
    ):
        """Should pass memory content and importance to store"""
        await node.process(basic_state)

        # Check first call arguments
        first_call = mock_memory_store.add_memory.call_args_list[0]
        assert first_call.kwargs["content"] == "Forged a ceremonial blade"
        assert first_call.kwargs["importance"] == 8.0

        # Check second call
        second_call = mock_memory_store.add_memory.call_args_list[1]
        assert second_call.kwargs["content"] == "Customer was very pleased"
        assert second_call.kwargs["importance"] == 7.5

    async def test_includes_timestamp_from_observation(self, node, mock_memory_store, basic_state):
        """Should include current simulation time as timestamp"""
        await node.process(basic_state)

        # All calls should have the simulation time
        for call in mock_memory_store.add_memory.call_args_list:
            assert call.kwargs["timestamp"] == 1500

    async def test_includes_location_from_observation(self, node, mock_memory_store, basic_state):
        """Should include location from observation status"""
        await node.process(basic_state)

        # All calls should have the location
        for call in mock_memory_store.add_memory.call_args_list:
            assert call.kwargs["location"] == (10, 15)

    async def test_handles_missing_location(self, node, mock_memory_store):
        """Should handle observations without status/position"""
        state = PipelineState(
            observation=Observation(
                entity_id="test_npc",
                current_simulation_time=1500,
                status=None,  # No status
            ),
            daily_memories=[NewMemory(content="Test memory", importance=5.0)],
        )

        await node.process(state)

        # Should call with None location
        call = mock_memory_store.add_memory.call_args
        assert call.kwargs["location"] is None

    async def test_handles_empty_daily_memories(self, node, mock_memory_store):
        """Should handle state with no daily memories"""
        state = PipelineState(
            observation=Observation(
                entity_id="test_npc",
                current_simulation_time=1500,
                status=StatusObservation(position=(5, 5), movement_locked=False),
            ),
            daily_memories=[],  # Empty
        )

        result = await node.process(state)

        # Should not call memory store
        mock_memory_store.add_memory.assert_not_called()
        # Should still return valid state
        assert result.daily_memories == []

    async def test_tracks_timing(self, node, mock_memory_store, basic_state):
        """Should track execution time in state"""
        result = await node.process(basic_state)

        # Should have timing tracked (via Node base class)
        assert "memory_consolidation" in result.time_ms
        assert result.time_ms["memory_consolidation"] >= 0

    async def test_preserves_other_state_fields(self, node, mock_memory_store, basic_state):
        """Should not modify unrelated state fields"""
        original_observation = basic_state.observation
        original_working_memory = basic_state.working_memory

        result = await node.process(basic_state)

        assert result.observation == original_observation
        assert result.working_memory == original_working_memory

    async def test_processes_memories_in_order(self, node, mock_memory_store, basic_state):
        """Should process memories in the order they appear in list"""
        await node.process(basic_state)

        # Check call order matches list order
        call_contents = [
            call.kwargs["content"] for call in mock_memory_store.add_memory.call_args_list
        ]
        assert call_contents[0] == "Forged a ceremonial blade"
        assert call_contents[1] == "Customer was very pleased"
        assert call_contents[2] == "Learned new tempering technique"

    async def test_handles_various_importance_scores(self, node, mock_memory_store):
        """Should handle memories with different importance scores"""
        state = PipelineState(
            observation=Observation(
                entity_id="test_npc",
                current_simulation_time=1500,
                status=StatusObservation(position=(5, 5), movement_locked=False),
            ),
            daily_memories=[
                NewMemory(content="Very important", importance=10.0),
                NewMemory(content="Somewhat important", importance=5.0),
                NewMemory(content="Least important", importance=1.0),
            ],
        )

        await node.process(state)

        # Check importance values preserved
        calls = mock_memory_store.add_memory.call_args_list
        assert calls[0].kwargs["importance"] == 10.0
        assert calls[1].kwargs["importance"] == 5.0
        assert calls[2].kwargs["importance"] == 1.0
