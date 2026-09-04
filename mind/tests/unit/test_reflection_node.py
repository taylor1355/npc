"""Unit tests for ReflectionNode.

Ports every prompt-rendering and state-effect assertion from the retired
CognitiveUpdateNode and ActionSelectionNode test suites: the merged node must
render the same inputs and produce the same state effects the two-node
sequence did for equivalent responses. All rendering assertions see the plain
string content path because caching is structurally off under AsyncMock.
"""

import logging
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage

from mind.cognitive_architecture.actions import Action, ActionType, AvailableAction
from mind.cognitive_architecture.memory import Memory
from mind.cognitive_architecture.nodes.formatting import format_goal_options, format_substrate_goal
from mind.cognitive_architecture.nodes.reflection.node import ReflectionNode
from mind.cognitive_architecture.observations import (
    ConversationMessage,
    GoalDetail,
    GoalObservation,
    MindEvent,
    MindEventType,
    Observation,
    StatusObservation,
)
from mind.cognitive_architecture.state import PipelineState
from mind.cognitive_architecture.working_memory import NewMemory, WorkingMemory
from tests.fixtures.observations import wire_current_interaction, wire_property_spec

VALID_RESPONSE = """{
    "updated_working_memory": {
        "situation_assessment": "Working on sword commission",
        "active_goals": ["Finish blade", "Heat treatment"],
        "emotional_state": "Determined"
    },
    "new_memories": [
        {"content": "Started sword commission", "importance": 7.0}
    ],
    "chosen_action": {"action": "wait", "parameters": {}}
}"""


def make_mock_llm(content: str = VALID_RESPONSE, total_tokens: int = 150) -> AsyncMock:
    mock = AsyncMock()
    mock.ainvoke.return_value = AIMessage(
        content=content,
        usage_metadata={
            "input_tokens": total_tokens - 50,
            "output_tokens": 50,
            "total_tokens": total_tokens,
        },
    )
    return mock


def rendered_prompt(mock_llm: AsyncMock) -> str:
    """The full prompt text the node sent (static prefix + dynamic suffix)"""
    return mock_llm.ainvoke.call_args[0][0][0].content


@pytest.mark.asyncio
class TestReflectionNode:
    """Test ReflectionNode in isolation"""

    @pytest.fixture
    def mock_llm(self):
        return make_mock_llm()

    @pytest.fixture
    def node(self, mock_llm):
        return ReflectionNode(mock_llm)

    @pytest.fixture
    def basic_state(self):
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

    async def test_updates_working_memory(self, node, mock_llm, basic_state):
        """Should update working memory with situation, goals, and emotional state"""
        result = await node.process(basic_state)

        assert result.working_memory is not None
        assert isinstance(result.working_memory, WorkingMemory)
        assert result.working_memory.situation_assessment == "Working on sword commission"
        assert result.working_memory.active_goals == ["Finish blade", "Heat treatment"]
        assert result.working_memory.emotional_state == "Determined"

    async def test_adds_new_memories_to_daily_buffer(self, node, mock_llm, basic_state):
        """Should extend daily_memories with new memories from LLM"""
        assert len(basic_state.daily_memories) == 0

        result = await node.process(basic_state)

        assert len(result.daily_memories) == 1
        assert result.daily_memories[0].content == "Started sword commission"
        assert result.daily_memories[0].importance == 7.0

    async def test_populates_chosen_action_field(self, node, mock_llm, basic_state):
        """Should populate chosen_action in state"""
        assert basic_state.chosen_action is None

        result = await node.process(basic_state)

        assert result.chosen_action is not None
        assert isinstance(result.chosen_action, Action)
        assert result.chosen_action.action == ActionType.WAIT
        assert result.chosen_action.parameters == {}

    async def test_appends_action_chosen_event(self, node, mock_llm, basic_state):
        """Should append an ACTION_CHOSEN event carrying the chosen action"""
        result = await node.process(basic_state)

        assert len(result.recent_events) == 1
        event = result.recent_events[0]
        assert event.event_type == MindEventType.ACTION_CHOSEN
        assert event.timestamp == 100
        assert event.payload["action"] == ActionType.WAIT
        assert event.payload["parameters"] == {}

    async def test_state_effects_match_the_two_node_sequence(self, node, mock_llm, basic_state):
        """One merged response produces the union of both old nodes' state writes"""
        existing_memory = NewMemory(content="Previous event", importance=5.0)
        basic_state.daily_memories.append(existing_memory)

        result = await node.process(basic_state)

        # cognitive_update's writes: working memory replaced, daily extended
        assert result.working_memory.situation_assessment == "Working on sword commission"
        assert result.daily_memories == [
            existing_memory,
            NewMemory(content="Started sword commission", importance=7.0),
        ]
        # action_selection's writes: chosen action + ACTION_CHOSEN event
        assert result.chosen_action.action == ActionType.WAIT
        assert [e.event_type for e in result.recent_events] == [MindEventType.ACTION_CHOSEN]

    async def test_llm_called_once_per_process(self, node, mock_llm, basic_state):
        """The whole point of the merge: one round-trip per decision"""
        await node.process(basic_state)

        assert mock_llm.ainvoke.call_count == 1

    async def test_tracks_token_usage(self, node, mock_llm, basic_state):
        result = await node.process(basic_state)

        assert "reflection" in result.tokens_used
        assert result.tokens_used["reflection"].total_tokens == 150

    async def test_tracks_timing(self, node, mock_llm, basic_state):
        result = await node.process(basic_state)

        assert "reflection" in result.time_ms
        assert result.time_ms["reflection"] >= 0

    async def test_formats_memories_for_llm(self, node, mock_llm, basic_state):
        """Should format retrieved memories as newline-separated strings for LLM"""
        await node.process(basic_state)

        prompt = rendered_prompt(mock_llm)
        assert "Learned blacksmithing from master" in prompt
        assert "Customer wants ceremonial blade" in prompt

    async def test_handles_no_memories_in_context(self, node, mock_llm, basic_state):
        """Should render the sentinel when no memories were retrieved"""
        basic_state.retrieved_memories = []

        result = await node.process(basic_state)

        assert result.working_memory is not None
        assert "No relevant memories found" in rendered_prompt(mock_llm)

    @staticmethod
    def _recent_events_section(prompt: str) -> str:
        """The rendered Recent Events section, sliced from its own headings.

        Slicing makes these contract tests rather than substring tests: a repr
        leaking in anywhere else in the prompt cannot make them pass, and a
        repr leaking in *here* cannot be masked by prose elsewhere.
        """
        return prompt.split("### Recent Events")[1].split("### Current Observation")[0]

    async def test_recent_events_render_as_prose_not_repr(self, node, mock_llm, basic_state):
        """The event buffer reaches the model as sentences, not Python reprs (NPC-1335)"""
        basic_state.recent_events = [
            MindEvent(
                timestamp=101,
                event_type=MindEventType.INTERACTION_BID_RECEIVED,
                payload={
                    "interaction_name": "conversation",
                    "bidder_id": "npc_carol",
                    "bid_id": "bid_e5f6a7b8",
                    "bid_type": 0,
                },
            ),
            MindEvent(
                timestamp=105,
                event_type=MindEventType.MOVEMENT_COMPLETED,
                payload={
                    "status": "ARRIVED",
                    "intended_destination": [10, 20],
                    "actual_destination": [10, 20],
                },
            ),
        ]

        await node.process(basic_state)

        section = self._recent_events_section(rendered_prompt(mock_llm))
        assert "MindEvent(" not in section
        assert "event_type=<MindEventType" not in section
        assert "Interaction bid received: conversation" in section
        assert "Arrived at (10, 20)" in section
        # One line per event, so exactly one separator between the two.
        assert section.strip().count("\n") == 1

    async def test_empty_recent_events_render_a_sentinel(self, node, mock_llm, basic_state):
        """An empty buffer must say so rather than render "[]" or nothing"""
        basic_state.recent_events = []

        await node.process(basic_state)

        section = self._recent_events_section(rendered_prompt(mock_llm)).strip()
        assert section
        assert section != "[]"

    async def test_handles_empty_new_memories_list(self, node, mock_llm, basic_state):
        """Should handle LLM returning no new memories"""
        mock_llm.ainvoke.return_value = AIMessage(
            content="""{
                "updated_working_memory": {
                    "situation_assessment": "Routine work",
                    "emotional_state": "Neutral"
                },
                "new_memories": [],
                "chosen_action": {"action": "wait", "parameters": {}}
            }""",
            usage_metadata={"input_tokens": 80, "output_tokens": 20, "total_tokens": 100},
        )

        result = await node.process(basic_state)

        assert len(result.daily_memories) == 0

    async def test_handles_action_with_parameters(self, node, mock_llm, basic_state):
        """Should correctly parse action with parameters"""
        mock_llm.ainvoke.return_value = AIMessage(
            content="""{
                "updated_working_memory": {"situation_assessment": "Heading out"},
                "new_memories": [],
                "chosen_action": {"action": "move_to", "parameters": {"destination": [15, 20]}}
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
                "updated_working_memory": {"situation_assessment": "Using the anvil"},
                "new_memories": [],
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
                "updated_working_memory": {"situation_assessment": "Negotiating"},
                "new_memories": [],
                "chosen_action": {
                    "action": "act_in_interaction",
                    "parameters": {
                        "response": "I agree to help",
                        "intensity": 0.8
                    }
                }
            }""",
            usage_metadata={"input_tokens": 200, "output_tokens": 60, "total_tokens": 260},
        )

        result = await node.process(basic_state)

        assert result.chosen_action.action == ActionType.ACT_IN_INTERACTION
        assert result.chosen_action.parameters["response"] == "I agree to help"
        assert result.chosen_action.parameters["intensity"] == 0.8

    async def test_retries_when_interaction_action_contains_an_unadvertised_parameter(
        self, node, mock_llm, basic_state
    ):
        basic_state.observation.status.current_interaction = wire_current_interaction(
            "conversation",
            act_parameters={"message": wire_property_spec("string", "", "What you say")},
        )
        basic_state.observation.status.activity_state = {"state_name": "interacting"}
        invalid = AIMessage(
            content="""{
                "updated_working_memory": {"situation_assessment": "Talking"},
                "new_memories": [],
                "chosen_action": {
                    "action": "act_in_interaction",
                    "parameters": {"message": "Hello", "interaction_name": "conversation"}
                }
            }"""
        )
        valid = AIMessage(
            content="""{
                "updated_working_memory": {"situation_assessment": "Talking"},
                "new_memories": [],
                "chosen_action": {
                    "action": "act_in_interaction",
                    "parameters": {"message": "Hello"}
                }
            }"""
        )
        mock_llm.ainvoke.side_effect = [invalid, valid]

        result = await node.process(basic_state)

        assert mock_llm.ainvoke.call_count == 2
        assert result.chosen_action.parameters == {"message": "Hello"}

    async def test_renders_complete_conversation_once(self, node, mock_llm, basic_state):
        message = ConversationMessage(
            speaker_id="npc_bob",
            speaker_name="Bob",
            message="Have you seen the smith today?",
            id="message_1",
            timestamp=99,
            declarations=[{"kind": "farewell"}],
        )
        basic_state.conversation_histories = {"conversation_1": [message]}
        basic_state.recent_events = [
            MindEvent(
                timestamp=100,
                event_type=MindEventType.INTERACTION_OBSERVATION,
                payload={
                    "interaction_name": "conversation",
                    "interaction_id": "conversation_1",
                    "conversation_history": [message.model_dump()],
                    "total_message_count": 1,
                },
            )
        ]

        await node.process(basic_state)

        prompt = rendered_prompt(mock_llm)
        assert "### Active Conversation Transcript" in prompt
        assert "Bob: Have you seen the smith today? [farewell]" in prompt
        assert prompt.count("Have you seen the smith today?") == 1

    async def test_preserves_other_state_fields(self, node, mock_llm, basic_state):
        """Should not modify unrelated state fields"""
        original_memories = basic_state.retrieved_memories.copy()

        result = await node.process(basic_state)

        assert result.retrieved_memories == original_memories
        assert result.personality_traits == basic_state.personality_traits

    async def test_renders_personality_traits_in_prompt(self, node, mock_llm, basic_state):
        """Personality traits should be rendered into the reflection prompt"""
        basic_state.personality_traits = ["idiotic", "pedantic", "aquarium-enthusiast"]

        await node.process(basic_state)

        assert "idiotic, pedantic, aquarium-enthusiast" in rendered_prompt(mock_llm)

    async def test_renders_personality_dimensions_in_prompt(self, node, mock_llm, basic_state):
        """Personality dimensions should be rendered with sorted keys, multi-line"""
        basic_state.personality_dimensions = {"extroversion": 0.85, "curiosity": 0.2}

        await node.process(basic_state)

        rendered = rendered_prompt(mock_llm)
        # Sorted alphabetically: curiosity before extroversion
        assert "curiosity: 0.20" in rendered
        assert "extroversion: 0.85" in rendered
        assert rendered.index("curiosity: 0.20") < rendered.index("extroversion: 0.85")
        # Dimensions render on separate lines (multi-line convention matches
        # other prompt sections like personality_traits / available_actions)
        assert "curiosity: 0.20\nextroversion: 0.85" in rendered
        assert "curiosity: 0.20, extroversion: 0.85" not in rendered

    async def test_handles_empty_personality(self, node, mock_llm, basic_state):
        """Empty personality should render sentinel strings, not crash.

        LangChain PromptTemplate requires every declared variable, so the node
        must always pass personality_traits and personality_dimensions even when
        the NPC has none.
        """
        basic_state.personality_traits = []
        basic_state.personality_dimensions = {}

        result = await node.process(basic_state)

        assert result.working_memory is not None
        assert result.chosen_action is not None
        rendered = rendered_prompt(mock_llm)
        assert "No specific traits" in rendered
        assert "No personality dimensions provided" in rendered

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

    async def test_working_memory_logged_as_single_record(
        self, node, mock_llm, basic_state, caplog
    ):
        """Working-memory fields land in one record: one Events-tab entry per thought"""
        with caplog.at_level(logging.DEBUG, logger="mind"):
            await node.process(basic_state)

        wm_records = [r for r in caplog.records if "Updated working memory" in r.getMessage()]
        assert len(wm_records) == 1
        message = wm_records[0].getMessage()
        assert "Situation:" in message
        assert "Active goals:" in message
        assert "Emotional state:" in message

    async def test_new_memories_logged_as_single_record(self, node, mock_llm, basic_state, caplog):
        """Memory-storage lines land in one record, not one per memory"""
        with caplog.at_level(logging.DEBUG, logger="mind"):
            await node.process(basic_state)

        storing_records = [r for r in caplog.records if "Storing" in r.getMessage()]
        assert len(storing_records) == 1
        assert "Started sword commission" in storing_records[0].getMessage()


@pytest.mark.asyncio
class TestObservationEnrichmentArms:
    """reflection must format in both arms of the enrichment A/B"""

    @pytest.fixture
    def mock_llm(self):
        return make_mock_llm()

    @pytest.fixture
    def node(self, mock_llm):
        return ReflectionNode(mock_llm)

    async def test_formats_an_unenriched_observation(self, node):
        state = PipelineState(
            observation=Observation(
                entity_id="test_npc",
                current_simulation_time=100,
                status=StatusObservation(position=(0, 0), movement_locked=False),
            ),
            working_memory=WorkingMemory(
                situation_assessment="idle", active_goals=[], emotional_state="steady"
            ),
        )

        result = await node.process(state)

        assert result.working_memory is not None

    async def test_enriched_observation_reaches_the_prompt(self, node, mock_llm):
        from tests.fixtures.observations import create_enriched_observation

        state = PipelineState(
            observation=create_enriched_observation(),
            working_memory=WorkingMemory(
                situation_assessment="idle", active_goals=[], emotional_state="calm"
            ),
        )

        await node.process(state)

        prompt_text = rendered_prompt(mock_llm)
        assert "Mood: stressed" in prompt_text
        assert "Subconscious pull: Find something to eat" in prompt_text
        assert "familiarity 0.62" in prompt_text


class TestSubstrateGoalPromptVariable:
    """`{substrate_goal}` must be formattable in BOTH arms.

    LangChain's PromptTemplate raises at format time on any declared variable
    that is missing, and in the reflection node that error would burn every
    retry before the salvage fallback collapsed the cycle to WAIT. A helper
    that returned None for "no active goal" would therefore turn every
    goal-less cycle into a permanently waiting NPC — a silent failure that
    reads as an NPC that stopped doing things.
    """

    def test_absent_goal_formats_to_a_sentinel_string(self):
        rendered = format_substrate_goal(None)

        assert isinstance(rendered, str)
        assert rendered.strip()

    def test_goal_without_active_goal_formats_to_the_same_sentinel(self):
        assert format_substrate_goal(GoalObservation()) == format_substrate_goal(None)

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
class TestReflectionSubstrateGoalArms:
    """The node must render successfully with and without a substrate goal"""

    @pytest.fixture
    def mock_llm(self):
        return make_mock_llm()

    @pytest.fixture
    def node(self, mock_llm):
        return ReflectionNode(mock_llm)

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
            )
        )

        result = await node.process(state)

        assert result.chosen_action.action == ActionType.WAIT

    async def test_prompt_receives_the_rendered_pull(self, node, mock_llm):
        state = self._state(
            GoalObservation(active_goal=GoalDetail(label="Seek company", urgency=0.7))
        )

        await node.process(state)

        assert "Seek company" in rendered_prompt(mock_llm)


def _goal_with_options() -> GoalObservation:
    """A small wire-shaped goal block with a two-entry option menu."""
    return GoalObservation.model_validate(
        {
            "contract_version": 1,
            "urgency_max": 1.3,
            "active_goal": {
                "template_id": "satisfy_hunger",
                "label": "Find food",
                "urgency": 0.87,
                "drive_source": "hunger",
                "preference_alignment": 0.12,
                "age_minutes": 14,
            },
            "goals": [
                {
                    "template_id": "satisfy_hunger",
                    "label": "Find food",
                    "urgency": 0.87,
                    "drive_source": "hunger",
                    "preference_alignment": 0.12,
                    "is_active": True,
                }
            ],
            "options": [
                {
                    "option_id": "satisfy_hunger:0",
                    "description": "Apple (consume, 0 away)",
                    "score": 0.6756,
                    "segments": [
                        {
                            "goal_template_id": "satisfy_hunger",
                            "goal_label": "Find food",
                            "steps": [
                                {
                                    "action": {
                                        "name": "INTERACT_WITH",
                                        "parameters": {
                                            "entity_id": "apple_01",
                                            "interaction_name": "consume",
                                        },
                                    },
                                    "target": {
                                        "interaction_name": "consume",
                                        "entity_id": "apple_01",
                                    },
                                    "factors": {
                                        "urgency": 0.87,
                                        "utility": 0.91,
                                        "responsiveness": 0.85,
                                        "policy_modifier": 1.0,
                                    },
                                    "step_score": 0.6756,
                                }
                            ],
                        }
                    ],
                },
                {
                    "option_id": "explore_area:1",
                    "description": "Wander and explore",
                    "score": 0.005,
                    "segments": [
                        {
                            "goal_template_id": "explore_area",
                            "goal_label": "Explore the area",
                            "steps": [
                                {
                                    "action": {"name": "WANDER", "parameters": {}},
                                    "target": None,
                                    "factors": {
                                        "urgency": 0.05,
                                        "utility": 0.1,
                                        "responsiveness": 1.0,
                                        "policy_modifier": 1.0,
                                    },
                                    "step_score": 0.005,
                                }
                            ],
                        }
                    ],
                },
            ],
            "option_total": 14,
        }
    )


class TestGoalOptionsPromptVariable:
    """`{goal_options}` must be formattable in every arm, like `{substrate_goal}`"""

    def test_absent_goal_formats_to_a_sentinel_string(self):
        rendered = format_goal_options(None)

        assert isinstance(rendered, str)
        assert rendered.strip()

    def test_goal_without_options_formats_to_the_same_sentinel(self):
        assert format_goal_options(GoalObservation()) == format_goal_options(None)

    def test_options_render_headline_steps_and_truncation(self):
        rendered = format_goal_options(_goal_with_options())

        assert "2 of 14 evaluated options" in rendered
        assert "Option satisfy_hunger:0: Apple (consume, 0 away) (score: 0.68)" in rendered
        assert "serves 'Find food'" in rendered
        assert "utility 0.91" in rendered
        assert "habituation 0.85" in rendered
        assert "step score 0.68" in rendered
        assert "12 lower-scoring options exist" in rendered

    def test_full_menu_renders_no_truncation_note(self):
        goal = _goal_with_options()
        goal = goal.model_copy(update={"option_total": 2})

        assert "not shown" not in format_goal_options(goal)


@pytest.mark.asyncio
class TestReflectionGoalOptions:
    """The option menu must reach the prompt, and the pick must survive parsing"""

    @pytest.fixture
    def mock_llm(self):
        return make_mock_llm()

    @pytest.fixture
    def node(self, mock_llm):
        return ReflectionNode(mock_llm)

    def _state(self, goal):
        return PipelineState(
            observation=Observation(
                entity_id="test_npc",
                current_simulation_time=100,
                status=StatusObservation(position=(5, 10), movement_locked=False),
                goal=goal,
            ),
            working_memory=WorkingMemory(
                situation_assessment="Hungry near the orchard",
                active_goals=[],
                emotional_state="Peckish",
            ),
            available_actions=[AvailableAction(name="wait", description="Wait and observe")],
        )

    async def test_prompt_receives_the_rendered_options(self, node, mock_llm):
        await node.process(self._state(_goal_with_options()))

        prompt = rendered_prompt(mock_llm)
        assert "### Goal Options" in prompt
        assert "Option satisfy_hunger:0" in prompt
        assert "score: 0.68" in prompt

    async def test_goalless_state_renders_the_options_sentinel(self, node, mock_llm):
        await node.process(self._state(None))

        assert "No evaluated options this cycle." in rendered_prompt(mock_llm)

    async def test_selected_option_id_round_trips_from_the_response(self, mock_llm):
        response = """{
            "updated_working_memory": {
                "situation_assessment": "Hungry, apple at hand",
                "active_goals": ["Eat"],
                "emotional_state": "Focused"
            },
            "new_memories": [],
            "chosen_action": {
                "action": "interact_with",
                "parameters": {
                    "entity_id": "apple_01",
                    "interaction_name": "consume"
                },
                "selected_option_id": "satisfy_hunger:0",
                "selection_rationale": "Hunger dominates and the apple is adjacent."
            }
        }"""
        mock_llm.ainvoke.return_value = AIMessage(
            content=response,
            usage_metadata={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
        )
        node = ReflectionNode(mock_llm)

        result = await node.process(self._state(_goal_with_options()))

        assert result.chosen_action.selected_option_id == "satisfy_hunger:0"
        assert (
            result.chosen_action.selection_rationale
            == "Hunger dominates and the apple is adjacent."
        )

    async def test_absent_selection_fields_stay_none(self, node):
        """An off-menu answer (no selection fields at all) is fully legal"""
        result = await node.process(self._state(_goal_with_options()))

        assert result.chosen_action.action == ActionType.WAIT
        assert result.chosen_action.selected_option_id is None
        assert result.chosen_action.selection_rationale is None
