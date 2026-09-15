"""Unit tests for MemoryConsolidationNode"""

from unittest.mock import MagicMock

import pytest

from mind.cognitive_architecture.nodes.memory_consolidation.node import MemoryConsolidationNode
from mind.cognitive_architecture.observations import Observation, StatusObservation
from mind.cognitive_architecture.state import PipelineState
from mind.cognitive_architecture.working_memory import FormedMemory


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
        return MemoryConsolidationNode(mock_memory_store, write_timestamp=1500)

    @pytest.fixture
    def basic_state(self):
        """A pipeline state whose per-memory stamps CONTRADICT its observation.

        The observation says the NPC is at (10, 15) in zone_forge. Each memory
        says it was formed somewhere else. That disagreement is deliberate and is
        what makes the location assertions below falsifiable: a node that read
        circumstances off the observation - the pre-NPC-1476 behaviour - would
        write (10, 15) three times and fail, rather than passing by coincidence
        because both sources happened to agree.
        """
        return PipelineState(
            observation=Observation(
                entity_id="test_npc",
                current_simulation_time=1500,
                status=StatusObservation(
                    position=(10, 15),
                    movement_locked=False,
                    current_zone_id="zone_forge",
                    current_zone_name="the Forge",
                ),
            ),
            daily_memories=[
                FormedMemory(
                    content="Forged a ceremonial blade",
                    importance=8.0,
                    formed_at_position=(1, 1),
                    formed_in_zone_id="zone_smithy",
                ),
                FormedMemory(
                    content="Customer was very pleased",
                    importance=7.5,
                    formed_at_position=(2, 2),
                    formed_in_zone_id="zone_market",
                ),
                FormedMemory(
                    content="Learned new tempering technique",
                    importance=9.0,
                    formed_at_position=(3, 3),
                    formed_in_zone_id="zone_library",
                ),
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

    async def test_writes_the_explicit_timestamp(self, node, mock_memory_store, basic_state):
        """Should stamp memories with the timestamp the caller supplied"""
        await node.process(basic_state)

        # All calls should have the supplied game time
        for call in mock_memory_store.add_memory.call_args_list:
            assert call.kwargs["timestamp"] == 1500

    async def test_an_unknown_timestamp_is_written_as_none_not_as_zero(
        self, mock_memory_store, basic_state
    ):
        """Regression for the consolidation write path.

        The only production caller had no observation of its own and fabricated
        one with current_simulation_time=0, which the node read - so every lived
        memory was stamped at the epoch and decayed from it, while untimestamped
        config seeds scored perfect recency and permanently outranked them.

        0 is a *valid* game-minute reading (the clock starts there), so it cannot
        double as "unknown". None must travel through untouched, letting the
        recency term abstain rather than claim a false measurement.
        """
        node = MemoryConsolidationNode(mock_memory_store, write_timestamp=None)

        await node.process(basic_state)

        assert mock_memory_store.add_memory.call_args_list
        for call in mock_memory_store.add_memory.call_args_list:
            assert call.kwargs["timestamp"] is None

    async def test_the_observation_time_is_not_used_as_the_write_timestamp(
        self, node, mock_memory_store, basic_state
    ):
        """The stamp comes from the constructor, never from the carrier state.

        basic_state's observation carries current_simulation_time=1500 as well,
        so this pins the source by making them disagree: if the node fell back to
        reading the observation, it would write 1500 rather than 9999.
        """
        node = MemoryConsolidationNode(mock_memory_store, write_timestamp=9999)
        assert basic_state.observation.current_simulation_time == 1500

        await node.process(basic_state)

        for call in mock_memory_store.add_memory.call_args_list:
            assert call.kwargs["timestamp"] == 9999

    async def test_consolidation_uses_each_memorys_own_stamp_not_the_batchs_observation(
        self, node, mock_memory_store, basic_state
    ):
        """THE FORMATION-VS-CONSOLIDATION FALSIFIER [NPC-1476].

        Consolidation runs once a day over a whole batch. Reading circumstances
        off `state.observation` here stamped every memory of the day with
        whichever cell the NPC happened to occupy when the batch was written - so
        a morning at the market and an afternoon at the forge both recorded the
        same place, and the record was wrong for all but one of them.

        The fixture's observation deliberately disagrees with all three memories
        (see `basic_state`), so the pre-change behaviour writes (10, 15) /
        "zone_forge" three times and this goes red.
        """
        await node.process(basic_state)

        calls = mock_memory_store.add_memory.call_args_list
        assert [c.kwargs["location"] for c in calls] == [(1, 1), (2, 2), (3, 3)]
        assert [c.kwargs["zone_id"] for c in calls] == [
            "zone_smithy",
            "zone_market",
            "zone_library",
        ]

    async def test_the_observations_own_place_never_reaches_the_store(
        self, node, mock_memory_store, basic_state
    ):
        """The negative half of the test above, stated separately.

        Asserting only that the right values arrive leaves open that the wrong
        ones arrive too, under some other keyword. Nothing from the observation
        may appear at all.
        """
        await node.process(basic_state)

        for call in mock_memory_store.add_memory.call_args_list:
            assert call.kwargs["location"] != (10, 15)
            assert call.kwargs["zone_id"] != "zone_forge"

    async def test_an_unstamped_memory_is_written_unplaced(self, node, mock_memory_store):
        """A memory formed with no status, or outside any known place.

        Both reach the store as None rather than as a fabricated origin or an
        empty-string zone, so the retrieval terms abstain on them. The
        observation here still carries a full status, which is the point: even
        with a place available to borrow, the node must not borrow it.
        """
        state = PipelineState(
            observation=Observation(
                entity_id="test_npc",
                current_simulation_time=1500,
                status=StatusObservation(
                    position=(10, 15),
                    movement_locked=False,
                    current_zone_id="zone_forge",
                    current_zone_name="the Forge",
                ),
            ),
            daily_memories=[FormedMemory(content="Test memory", importance=5.0)],
        )

        await node.process(state)

        call = mock_memory_store.add_memory.call_args
        assert call.kwargs["location"] is None
        assert call.kwargs["zone_id"] is None

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
                FormedMemory(content="Very important", importance=10.0),
                FormedMemory(content="Somewhat important", importance=5.0),
                FormedMemory(content="Least important", importance=1.0),
            ],
        )

        await node.process(state)

        # Check importance values preserved
        calls = mock_memory_store.add_memory.call_args_list
        assert calls[0].kwargs["importance"] == 10.0
        assert calls[1].kwargs["importance"] == 5.0
        assert calls[2].kwargs["importance"] == 1.0
