"""Regression tests for NPC-688: observation-grounded interaction gating.

The MCP mind must ground "am I interacting?" in the current OBSERVATION
(current_interaction AND activity_state == interacting), never in a stale
free-text working-memory belief. These tests pin that behavior so the
act_in_interaction desync loop (NPC-687) cannot structurally recur.
"""

from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from mind.cognitive_architecture.actions import Action, ActionType
from mind.cognitive_architecture.nodes.action_selection.node import ActionSelectionNode
from mind.cognitive_architecture.nodes.cognitive_update.models import WorkingMemory
from mind.cognitive_architecture.nodes.formatting import format_interaction_status
from mind.cognitive_architecture.observations import (
    Observation,
    StatusObservation,
)
from mind.cognitive_architecture.state import PipelineState


def _interacting_status() -> StatusObservation:
    return StatusObservation(
        position=(10, 10),
        movement_locked=True,
        current_interaction={"interaction_id": "conv_1", "interaction_name": "conversation"},
        activity_state={"state_name": "interacting"},
    )


def _torn_down_status() -> StatusObservation:
    """current_interaction still set but controller already left interacting state.

    Mirrors the cross-field teardown race documented in the Godot
    entity_controller._on_interaction_ended handler.
    """
    return StatusObservation(
        position=(10, 10),
        movement_locked=False,
        current_interaction={"interaction_id": "conv_1", "interaction_name": "conversation"},
        activity_state={"state_name": "idle"},
    )


def _not_interacting_status() -> StatusObservation:
    return StatusObservation(
        position=(10, 10),
        movement_locked=False,
        current_interaction={},
        activity_state={"state_name": "idle"},
    )


class TestIsInteractingGrounding:
    """is_interacting() requires BOTH authoritative signals to agree."""

    def test_true_only_when_both_signals_agree(self):
        assert _interacting_status().is_interacting() is True

    def test_false_when_interaction_set_but_state_not_interacting(self):
        # The teardown race: belief would say "interacting" but state disagrees.
        assert _torn_down_status().is_interacting() is False

    def test_false_when_no_current_interaction(self):
        assert _not_interacting_status().is_interacting() is False

    def test_false_when_state_interacting_but_no_interaction(self):
        status = StatusObservation(
            position=(0, 0),
            current_interaction={},
            activity_state={"state_name": "interacting"},
        )
        assert status.is_interacting() is False

    def test_backward_compatible_missing_activity_state_defaults_false(self):
        # Older sim payload: no activity_state at all. Must not crash and must
        # NOT report interacting (avoids spurious act_in_interaction).
        status = StatusObservation(
            position=(0, 0),
            current_interaction={"interaction_id": "x", "interaction_name": "conversation"},
        )
        assert status.activity_state == {}
        assert status.is_interacting() is False

    def test_observation_accessor_defaults_false_without_status(self):
        obs = Observation(entity_id="npc", current_simulation_time=1, status=None)
        assert obs.is_interacting() is False


class TestAvailableActionsGrounding:
    """get_available_actions must gate interaction actions on is_interacting()."""

    def _action_names(self, status: StatusObservation) -> set[str]:
        obs = Observation(entity_id="npc", current_simulation_time=1, status=status)
        return {a.name for a in obs.get_available_actions()}

    def test_offers_act_in_interaction_when_grounded_interacting(self):
        names = self._action_names(_interacting_status())
        assert ActionType.ACT_IN_INTERACTION in names
        assert ActionType.CANCEL_INTERACTION in names
        assert ActionType.WAIT not in names

    def test_no_act_in_interaction_during_teardown_race(self):
        # current_interaction set, but activity_state already idle.
        names = self._action_names(_torn_down_status())
        assert ActionType.ACT_IN_INTERACTION not in names
        assert ActionType.CANCEL_INTERACTION not in names
        # wait becomes available again so the NPC has a safe option.
        assert ActionType.WAIT in names

    def test_no_act_in_interaction_when_not_interacting(self):
        names = self._action_names(_not_interacting_status())
        assert ActionType.ACT_IN_INTERACTION not in names
        assert ActionType.WAIT in names


class TestValidatorGrounding:
    """The Action validator hard-rejects act_in_interaction when not grounded."""

    def _state(self, status: StatusObservation) -> PipelineState:
        return PipelineState(
            observation=Observation(entity_id="npc", current_simulation_time=1, status=status),
        )

    def test_act_in_interaction_valid_when_grounded(self):
        state = self._state(_interacting_status())
        action = Action.model_validate(
            {"action": "act_in_interaction", "parameters": {"message": "hi"}},
            context={"state": state},
        )
        assert action.action == ActionType.ACT_IN_INTERACTION

    def test_act_in_interaction_rejected_during_teardown_race(self):
        state = self._state(_torn_down_status())
        with pytest.raises(ValidationError):
            Action.model_validate(
                {"action": "act_in_interaction", "parameters": {"message": "hi"}},
                context={"state": state},
            )

    def test_act_in_interaction_rejected_when_not_interacting(self):
        state = self._state(_not_interacting_status())
        with pytest.raises(ValidationError):
            Action.model_validate(
                {"action": "act_in_interaction", "parameters": {"message": "hi"}},
                context={"state": state},
            )


class TestInteractionStatusRendering:
    """format_interaction_status produces a grounded, corrective string."""

    def test_interacting_string_confirms_interaction(self):
        obs = Observation(entity_id="npc", current_simulation_time=1, status=_interacting_status())
        text = format_interaction_status(obs)
        assert "ARE currently in an interaction" in text
        assert "conversation" in text

    def test_not_interacting_string_marks_belief_stale(self):
        obs = Observation(entity_id="npc", current_simulation_time=1, status=_torn_down_status())
        text = format_interaction_status(obs)
        assert "NOT currently in any interaction" in text
        assert "stale" in text

    def test_missing_status_defaults_not_interacting(self):
        obs = Observation(entity_id="npc", current_simulation_time=1, status=None)
        assert "NOT currently in any interaction" in format_interaction_status(obs)


@pytest.mark.asyncio
class TestActionSelectionRegroundsBelief:
    """End-to-end node test: stale working-memory belief is overridden.

    Feeds an observation that says NOT interacting plus a working memory that
    claims "in conversation". The action_selection prompt must surface the
    grounded status, and the chosen action must not be act_in_interaction.
    """

    async def test_does_not_emit_act_in_interaction_when_observation_says_not_interacting(self):
        # LLM mock that would *try* to act in interaction if it could.
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = AIMessage(
            content='{"chosen_action": {"action": "wander", "parameters": {}}}',
            usage_metadata={"input_tokens": 100, "output_tokens": 10, "total_tokens": 110},
        )
        node = ActionSelectionNode(mock_llm)

        obs = Observation(
            entity_id="npc",
            current_simulation_time=1,
            status=_torn_down_status(),  # interaction torn down
        )
        state = PipelineState(
            observation=obs,
            # available_actions built from the grounded observation: no act_in_interaction
            available_actions=obs.get_available_actions(),
            working_memory=WorkingMemory(
                situation_assessment="I am in a conversation with Alice",
                active_goals=["Continue the conversation"],
                current_plan=["Reply to Alice"],
                emotional_state="Engaged",
            ),
        )

        # The grounded action set must not include act_in_interaction.
        offered = {a.name for a in state.available_actions}
        assert ActionType.ACT_IN_INTERACTION not in offered

        result = await node.process(state)

        # Prompt rendered to the LLM must surface the corrective status.
        # call_llm invokes self.llm.ainvoke([HumanMessage(content=prompt_text)]).
        sent_messages = mock_llm.ainvoke.call_args[0][0]
        rendered_text = "\n".join(getattr(m, "content", str(m)) for m in sent_messages)
        assert "NOT currently in any interaction" in rendered_text

        # And the emitted action is not an interaction-participation action.
        assert result.chosen_action.action != ActionType.ACT_IN_INTERACTION
