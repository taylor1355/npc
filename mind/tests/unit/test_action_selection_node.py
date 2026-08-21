"""Unit tests for ActionSelectionNode"""

import logging
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage

from mind.cognitive_architecture.actions import Action, ActionType, AvailableAction
from mind.cognitive_architecture.nodes.action_selection.node import ActionSelectionNode
from mind.cognitive_architecture.nodes.formatting import format_substrate_goal
from mind.cognitive_architecture.observations import (
    GoalDetail,
    GoalObservation,
    Observation,
    StatusObservation,
)
from mind.cognitive_architecture.state import PipelineState
from mind.cognitive_architecture.working_memory import WorkingMemory
from tests.fixtures.observations import wire_current_interaction, wire_property_spec


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
        assert result.tokens_used["action_selection"].total_tokens == 230

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
        assert result.chosen_action.parameters == {
            "entity_id": "anvil_001",
            "interaction_name": "use",
        }

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

    async def test_renders_personality_dimensions_in_prompt(self, node, mock_llm):
        """Personality dimensions should be rendered into the LLM prompt with sorted keys"""
        state = PipelineState(
            observation=Observation(
                entity_id="test_npc",
                current_simulation_time=100,
                status=StatusObservation(position=(5, 10), movement_locked=False),
            ),
            working_memory=WorkingMemory(),
            personality_traits=["curious"],
            personality_dimensions={"extroversion": 0.85, "curiosity": 0.2},
            available_actions=[AvailableAction(name="wait", description="Wait")],
        )

        await node.process(state)

        # The rendered prompt is passed as a HumanMessage to ainvoke
        call_args = mock_llm.ainvoke.call_args
        rendered = call_args[0][0][0].content
        # Sorted alphabetically: curiosity before extroversion
        assert "curiosity: 0.20" in rendered
        assert "extroversion: 0.85" in rendered
        assert rendered.index("curiosity: 0.20") < rendered.index("extroversion: 0.85")
        # Dimensions must render on separate lines (multi-line convention matches
        # other prompt sections like personality_traits / available_actions)
        assert "curiosity: 0.20\nextroversion: 0.85" in rendered
        assert "curiosity: 0.20, extroversion: 0.85" not in rendered

    async def test_handles_empty_personality_dimensions(self, node, mock_llm):
        """Empty personality_dimensions should render a sentinel string, not crash"""
        state = PipelineState(
            observation=Observation(
                entity_id="test_npc",
                current_simulation_time=100,
                status=StatusObservation(position=(5, 10), movement_locked=False),
            ),
            working_memory=WorkingMemory(),
            personality_traits=["curious"],
            personality_dimensions={},
            available_actions=[AvailableAction(name="wait", description="Wait")],
        )

        result = await node.process(state)

        assert result.chosen_action is not None
        call_args = mock_llm.ainvoke.call_args
        rendered = call_args[0][0][0].content
        assert "No personality dimensions provided" in rendered

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
        # Set up an active interaction so act_in_interaction is valid.
        # NPC-688: validity is grounded in BOTH current_interaction AND
        # activity_state == interacting, so set both authoritative signals.
        # NPC-1278: the payload must be the real wire shape, and the act must
        # carry one of the parameters that shape advertises — an act naming only
        # invented parameters is now rejected.
        basic_state.observation.status.current_interaction = wire_current_interaction(
            "negotiation",
            act_parameters={
                "response": wire_property_spec("string", "", "What you say in reply"),
                "intensity": wire_property_spec("float", 0.5, "How forcefully you press"),
            },
        )
        basic_state.observation.status.activity_state = {"state_name": "interacting"}

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

    async def test_all_log_records_carry_entity_id(self, node, mock_llm, basic_state, caplog):
        """Every record from process() must carry the entity id so the simulation's
        log forwarder can attribute it to the NPC's Events tab (NPC-789)"""
        with caplog.at_level(logging.DEBUG, logger="mind"):
            await node.process(basic_state)

        assert caplog.records, "process() should emit log records"
        for record in caplog.records:
            assert "test_npc" in record.getMessage(), (
                f"Unattributed log record: {record.getMessage()!r}"
            )

    async def test_fallback_log_records_carry_entity_id(self, node, mock_llm, basic_state, caplog):
        """Retry and fallback-warning records must also carry the entity id (NPC-789)"""
        mock_llm.ainvoke.return_value = AIMessage(
            content="not valid json",
            usage_metadata={"input_tokens": 10, "output_tokens": 3, "total_tokens": 13},
        )

        with caplog.at_level(logging.DEBUG, logger="mind"):
            result = await node.process(basic_state)

        assert result.chosen_action.action == ActionType.WAIT
        assert caplog.records, "fallback path should emit log records"
        for record in caplog.records:
            assert "test_npc" in record.getMessage(), (
                f"Unattributed log record: {record.getMessage()!r}"
            )


class TestSubstrateGoalPromptVariable:
    """`{substrate_goal}` must be formattable in BOTH arms.

    LangChain's PromptTemplate raises at format time on any declared variable
    that is missing, and this node catches the resulting error and falls back to
    WAIT. A helper that returned None for "no active goal" would therefore turn
    every goal-less cycle into a permanently waiting NPC — a silent failure that
    reads as an NPC that stopped doing things.
    """

    def test_absent_goal_formats_to_a_sentinel_string(self):
        rendered = format_substrate_goal(None)

        assert isinstance(rendered, str)
        assert rendered.strip()

    def test_goal_without_active_goal_formats_to_the_same_sentinel(self):
        assert format_substrate_goal(GoalObservation(candidate_count=4)) == format_substrate_goal(
            None
        )

    def test_active_goal_formats_as_an_advisory_pull(self):
        rendered = format_substrate_goal(
            GoalObservation(
                active_goal=GoalDetail(
                    label="Find something to eat", urgency=1.21, drive_source="hunger"
                )
            )
        )

        assert "Find something to eat" in rendered
        assert "1.21" in rendered
        assert "hunger" in rendered


@pytest.mark.asyncio
class TestActionSelectionSubstrateGoalArms:
    """The node must render successfully with and without a substrate goal"""

    @pytest.fixture
    def mock_llm(self):
        mock = AsyncMock()
        mock.ainvoke.return_value = AIMessage(
            content='{"chosen_action": {"action": "wait", "parameters": {}}}',
            usage_metadata={"input_tokens": 200, "output_tokens": 30, "total_tokens": 230},
        )
        return mock

    @pytest.fixture
    def node(self, mock_llm):
        return ActionSelectionNode(mock_llm)

    def _state(self, goal):
        return PipelineState(
            observation=Observation(
                entity_id="test_npc",
                current_simulation_time=100,
                status=StatusObservation(position=(5, 10), movement_locked=False),
                goal=goal,
            ),
            working_memory=WorkingMemory(
                situation_assessment="At the forge",
                active_goals=[],
                emotional_state="Focused",
            ),
            available_actions=[AvailableAction(name="wait", description="Wait and observe")],
        )

    async def test_formats_without_a_goal(self, node):
        result = await node.process(self._state(None))

        assert result.chosen_action.action == ActionType.WAIT

    async def test_formats_with_a_goal(self, node):
        state = self._state(
            GoalObservation(
                active_goal=GoalDetail(label="Rest", urgency=0.8, drive_source="energy"),
                candidate_count=3,
            )
        )

        result = await node.process(state)

        assert result.chosen_action.action == ActionType.WAIT

    async def test_prompt_receives_the_rendered_pull(self, node, mock_llm):
        state = self._state(
            GoalObservation(active_goal=GoalDetail(label="Seek company", urgency=0.7))
        )

        await node.process(state)

        prompt_text = str(mock_llm.ainvoke.call_args[0][0])
        assert "Seek company" in prompt_text
