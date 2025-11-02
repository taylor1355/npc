"""Unit tests for CognitiveUpdateNode"""

from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage

from mind.cognitive_architecture.models import Memory, Observation, StatusObservation
from mind.cognitive_architecture.nodes.cognitive_update.models import (
    NewMemory,
    WorkingMemory,
)
from mind.cognitive_architecture.nodes.cognitive_update.node import CognitiveUpdateNode
from mind.cognitive_architecture.state import PipelineState


@pytest.mark.asyncio
class TestCognitiveUpdateNode:
    """Test CognitiveUpdateNode in isolation"""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM that returns structured output"""
        mock = AsyncMock()
        # Default response with valid CognitiveUpdateOutput structure
        mock.ainvoke.return_value = AIMessage(
            content="""{
                "situation_assessment": "Currently at the forge",
                "current_goals": ["Complete sword order"],
                "emotional_state": "Focused",
                "updated_working_memory": {
                    "situation_assessment": "Working on sword commission",
                    "active_goals": ["Finish blade", "Heat treatment"],
                    "emotional_state": "Determined"
                },
                "new_memories": [
                    {"content": "Started sword commission", "importance": 7.0}
                ]
            }""",
            usage_metadata={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
        )
        return mock

    @pytest.fixture
    def node(self, mock_llm):
        """Create a CognitiveUpdateNode with mocked LLM"""
        return CognitiveUpdateNode(mock_llm)

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
                active_goals=["Work on sword"],
                emotional_state="Focused",
            ),
            retrieved_memories=[
                Memory(id="mem_1", content="Learned blacksmithing from master", importance=8.0),
                Memory(id="mem_2", content="Customer wants ceremonial blade", importance=7.0),
            ],
        )

    async def test_updates_working_memory(self, node, mock_llm, basic_state):
        """Should update working memory based on LLM output"""
        result = await node.process(basic_state)

        # Should have updated working memory
        assert result.working_memory is not None
        assert isinstance(result.working_memory, WorkingMemory)
        assert result.working_memory.situation_assessment == "Working on sword commission"

    async def test_updates_cognitive_context(self, node, mock_llm, basic_state):
        """Should populate cognitive_context dict with situation, goals, emotional state"""
        result = await node.process(basic_state)

        # Should have cognitive context populated
        assert "situation_assessment" in result.cognitive_context
        assert "current_goals" in result.cognitive_context
        assert "emotional_state" in result.cognitive_context

        assert result.cognitive_context["situation_assessment"] == "Currently at the forge"
        assert result.cognitive_context["current_goals"] == ["Complete sword order"]
        assert result.cognitive_context["emotional_state"] == "Focused"

    async def test_adds_new_memories_to_daily_buffer(self, node, mock_llm, basic_state):
        """Should extend daily_memories with new memories from LLM"""
        # Start with empty daily memories
        assert len(basic_state.daily_memories) == 0

        result = await node.process(basic_state)

        # Should have added new memories
        assert len(result.daily_memories) == 1
        assert result.daily_memories[0].content == "Started sword commission"
        assert result.daily_memories[0].importance == 7.0

    async def test_handles_no_memories_in_context(self, node, mock_llm):
        """Should handle state with no retrieved memories"""
        state = PipelineState(
            observation=Observation(
                entity_id="test_npc",
                current_simulation_time=100,
                status=StatusObservation(position=(5, 10), movement_locked=False),
            ),
            working_memory=WorkingMemory(situation_assessment="Exploring"),
            retrieved_memories=[],  # No memories
        )

        result = await node.process(state)

        # Should still process successfully
        assert result.working_memory is not None
        # LLM should have been called with "No relevant memories found."
        call_args = mock_llm.ainvoke.call_args
        assert "No relevant memories found" in str(call_args)

    async def test_tracks_token_usage(self, node, mock_llm, basic_state):
        """Should track token usage in state"""
        result = await node.process(basic_state)

        # Should have token usage tracked
        assert "cognitive_update" in result.tokens_used
        assert result.tokens_used["cognitive_update"] == 150

    async def test_tracks_timing(self, node, mock_llm, basic_state):
        """Should track execution time in state"""
        result = await node.process(basic_state)

        # Should have timing tracked (via Node base class)
        assert "cognitive_update" in result.time_ms
        assert result.time_ms["cognitive_update"] >= 0

    async def test_formats_memories_for_llm(self, node, mock_llm, basic_state):
        """Should format retrieved memories as newline-separated strings for LLM"""
        await node.process(basic_state)

        # Check the LLM was called with formatted memories
        call_args = mock_llm.ainvoke.call_args
        messages = call_args[0][0]
        prompt_content = messages[0].content

        # Should contain formatted memories
        assert "Learned blacksmithing from master" in prompt_content
        assert "Customer wants ceremonial blade" in prompt_content

    async def test_handles_empty_new_memories_list(self, node, mock_llm, basic_state):
        """Should handle LLM returning no new memories"""
        # Mock LLM to return empty new_memories list
        mock_llm.ainvoke.return_value = AIMessage(
            content="""{
                "situation_assessment": "Nothing significant",
                "current_goals": [],
                "emotional_state": "Calm",
                "updated_working_memory": {
                    "situation_assessment": "Routine work",
                    "emotional_state": "Neutral"
                },
                "new_memories": []
            }""",
            usage_metadata={"input_tokens": 80, "output_tokens": 20, "total_tokens": 100},
        )

        result = await node.process(basic_state)

        # Should handle empty memories gracefully
        assert len(result.daily_memories) == 0

    async def test_preserves_existing_daily_memories(self, node, mock_llm, basic_state):
        """Should extend daily_memories without replacing existing ones"""
        # Add existing memory
        existing_memory = NewMemory(content="Previous event", importance=5.0)
        basic_state.daily_memories.append(existing_memory)

        result = await node.process(basic_state)

        # Should have both old and new memories
        assert len(result.daily_memories) == 2
        assert result.daily_memories[0] == existing_memory
        assert result.daily_memories[1].content == "Started sword commission"
