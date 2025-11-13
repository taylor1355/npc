"""Unit tests for ActionSelectionNode"""

from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage

from mind.cognitive_architecture.actions import Action, ActionType, AvailableAction
from mind.cognitive_architecture.observations import Observation, StatusObservation
from mind.cognitive_architecture.nodes.action_selection.node import ActionSelectionNode
from mind.cognitive_architecture.nodes.cognitive_update.models import WorkingMemory
from mind.cognitive_architecture.state import PipelineState


@pytest.mark.asyncio
class TestActionSelectionNode:
    """Test ActionSelectionNode in isolation"""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM that returns action selection output"""
        mock = AsyncMock()
        # Default response selecting a wait action
        mock.ainvoke.return_value = AIMessage(
            content="""{
                "chosen_action": {
                    "action": "wait",
                    "parameters": {}
                }
            }""",
            usage_metadata={"input_tokens": 200, "output_tokens": 30, "total_tokens": 230},
        )
        return mock

    @pytest.fixture
    def node(self, mock_llm):
        """Create an ActionSelectionNode with mocked LLM"""
        return ActionSelectionNode(mock_llm)

    @pytest.fixture
    def basic_state(self):
        """Create a basic pipeline state with available actions"""
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
            cognitive_context={
                "situation_assessment": "Currently forging a sword",
                "current_goals": ["Finish blade", "Heat treatment"],
                "emotional_state": "Determined",
            },
            personality_traits=["diligent", "perfectionist"],
            available_actions=[
                AvailableAction(
                    name="interact_with",
                    description="Interact with an entity",
                    parameters={"entity_id": "ID of entity to interact with"},
                ),
                AvailableAction(name="wait", description="Wait and observe"),
                AvailableAction(
                    name="move_to",
                    description="Move to a position",
                    parameters={"x": "X coordinate", "y": "Y coordinate"},
                ),
            ],
        )

    async def test_populates_chosen_action_field(self, node, mock_llm, basic_state):
        """Should populate chosen_action in state"""
        assert basic_state.chosen_action is None

        result = await node.process(basic_state)

        assert result.chosen_action is not None
        assert isinstance(result.chosen_action, Action)

    async def test_returns_action_from_llm_output(self, node, mock_llm, basic_state):
        """Should use action specified by LLM"""
        result = await node.process(basic_state)

        assert result.chosen_action.action == ActionType.WAIT
        assert result.chosen_action.parameters == {}

    async def test_tracks_token_usage(self, node, mock_llm, basic_state):
        """Should track token usage in state"""
        result = await node.process(basic_state)

        assert "action_selection" in result.tokens_used
        assert result.tokens_used["action_selection"] == 230

    async def test_tracks_timing(self, node, mock_llm, basic_state):
        """Should track execution time in state"""
        result = await node.process(basic_state)

        assert "action_selection" in result.time_ms
        assert result.time_ms["action_selection"] >= 0

    async def test_handles_action_with_parameters(self, node, mock_llm, basic_state):
        """Should correctly parse action with parameters"""
        mock_llm.ainvoke.return_value = AIMessage(
            content="""{
                "chosen_action": {
                    "action": "move_to",
                    "parameters": {"destination": [15, 20]}
                }
            }""",
            usage_metadata={"input_tokens": 200, "output_tokens": 40, "total_tokens": 240},
        )

        result = await node.process(basic_state)

        assert result.chosen_action.action == ActionType.MOVE_TO
        assert result.chosen_action.parameters == {"destination": [15, 20]}

    async def test_handles_interaction_action(self, node, mock_llm, basic_state):
        """Should handle interaction actions"""
        mock_llm.ainvoke.return_value = AIMessage(
            content="""{
                "chosen_action": {
                    "action": "interact_with",
                    "parameters": {"entity_id": "anvil_001", "interaction_name": "use"}
                }
            }""",
            usage_metadata={"input_tokens": 200, "output_tokens": 45, "total_tokens": 245},
        )

        result = await node.process(basic_state)

        assert result.chosen_action.action == ActionType.INTERACT_WITH
        assert result.chosen_action.parameters == {"entity_id": "anvil_001", "interaction_name": "use"}

    async def test_handles_empty_personality_traits(self, node, mock_llm):
        """Should handle state with no personality traits"""
        state = PipelineState(
            observation=Observation(
                entity_id="test_npc",
                current_simulation_time=100,
                status=StatusObservation(position=(5, 10), movement_locked=False),
            ),
            working_memory=WorkingMemory(),
            personality_traits=[],
            available_actions=[AvailableAction(name="wait", description="Wait")],
        )

        result = await node.process(state)

        assert result.chosen_action is not None
        assert isinstance(result.chosen_action, Action)

    async def test_handles_empty_cognitive_context(self, node, mock_llm):
        """Should handle state with no cognitive context"""
        state = PipelineState(
            observation=Observation(
                entity_id="test_npc",
                current_simulation_time=100,
                status=StatusObservation(position=(5, 10), movement_locked=False),
            ),
            working_memory=WorkingMemory(),
            cognitive_context={},
            personality_traits=["test"],
            available_actions=[AvailableAction(name="wait", description="Wait")],
        )

        result = await node.process(state)

        assert result.chosen_action is not None

    async def test_preserves_other_state_fields(self, node, mock_llm, basic_state):
        """Should not modify unrelated state fields"""
        original_memories = basic_state.retrieved_memories.copy()
        original_working_memory = basic_state.working_memory

        result = await node.process(basic_state)

        assert result.retrieved_memories == original_memories
        assert result.working_memory == original_working_memory
        assert result.personality_traits == basic_state.personality_traits

    async def test_llm_called_once_per_process(self, node, mock_llm, basic_state):
        """Should call LLM exactly once per process call"""
        await node.process(basic_state)

        assert mock_llm.ainvoke.call_count == 1

    async def test_handles_complex_action_parameters(self, node, mock_llm, basic_state):
        """Should handle actions with multiple complex parameters"""
        mock_llm.ainvoke.return_value = AIMessage(
            content="""{
                "chosen_action": {
                    "action": "act_in_interaction",
                    "parameters": {
                        "interaction_id": "conversation_123",
                        "response": "I agree to help",
                        "intensity": 0.8
                    }
                }
            }""",
            usage_metadata={"input_tokens": 200, "output_tokens": 60, "total_tokens": 260},
        )

        result = await node.process(basic_state)

        assert result.chosen_action.action == ActionType.ACT_IN_INTERACTION
        assert result.chosen_action.parameters["interaction_id"] == "conversation_123"
        assert result.chosen_action.parameters["response"] == "I agree to help"
        assert result.chosen_action.parameters["intensity"] == 0.8
