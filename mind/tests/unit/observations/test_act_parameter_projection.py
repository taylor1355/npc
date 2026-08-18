"""Regression tests for NPC-1278: act_in_interaction parameter projection.

Four sites read ``current_interaction["interaction_name"]`` — a key the wire has
never carried — and one of them then hardcoded the parameter hints for a single
named interaction. The compound effect was that NO interaction parameter ever
reached the LLM, the offered ``act_in_interaction`` action was described with the
literal word "interaction", and the validator's parameter check could never fire.

The invariant these tests pin: **the simulation's schema is the only source of
what an act may carry.** A new interaction, or a new parameter on an existing
one, must reach the LLM's action menu with zero changes to this repository.
"""

import ast
import logging
import re
from pathlib import Path

import pytest
from pydantic import ValidationError

import mind.cognitive_architecture.actions.models as action_models
import mind.cognitive_architecture.nodes.formatting as formatting_module
import mind.cognitive_architecture.observations.models as observation_models
from mind.cognitive_architecture.actions import Action, ActionType
from mind.cognitive_architecture.nodes.formatting import format_interaction_status
from mind.cognitive_architecture.observations import (
    UNNAMED_INTERACTION,
    WIRE_KEY_ACT_PARAMETERS,
    Observation,
    StatusObservation,
)
from mind.cognitive_architecture.state import PipelineState
from tests.fixtures.observations import (
    wire_conversation_interaction,
    wire_current_interaction,
    wire_property_spec,
)


def _status(current_interaction: dict) -> StatusObservation:
    return StatusObservation(
        position=(10, 10),
        movement_locked=True,
        current_interaction=current_interaction,
        activity_state={"state_name": "interacting"},
    )


def _observation(current_interaction: dict) -> Observation:
    return Observation(
        entity_id="npc", current_simulation_time=1, status=_status(current_interaction)
    )


def _mind_warnings(caplog) -> list[str]:
    """Only this package's warnings — an unrelated library's chatter is not a signal."""
    return [r.getMessage() for r in caplog.records if r.name.startswith("mind")]


def _act_action(observation: Observation):
    for action in observation.get_available_actions():
        if action.name == ActionType.ACT_IN_INTERACTION:
            return action
    return None


class TestWireShapeHintProjection:
    """Test 1 — the parameters the simulation advertises reach the action menu.

    Red-first: against the un-fixed code every assertion below fails, because
    the hints were hardcoded behind a comparison against a wire key that does
    not exist.
    """

    def test_advertised_parameters_become_action_hints(self):
        action = _act_action(_observation(wire_conversation_interaction()))

        assert action is not None
        assert set(action.parameters) == {"message", "is_farewell"}

    def test_hint_prose_carries_the_simulation_s_own_description(self):
        action = _act_action(_observation(wire_conversation_interaction()))

        assert "Message text to send in conversation" in action.parameters["message"]
        assert "intends this message to end the conversation" in action.parameters["is_farewell"]

    def test_hint_prose_carries_type_and_default_so_the_act_can_validate(self):
        # The simulation rejects an act whose parameters fail PropertySpec
        # validation, so the LLM must be able to see the contract it has to meet.
        hint = _act_action(_observation(wire_conversation_interaction())).parameters["is_farewell"]

        assert "type: bool" in hint
        assert "default: False" in hint

    def test_falsy_default_is_still_advertised(self):
        # Membership, not truthiness: `false` and `0` are legitimate defaults and
        # must not read as "no default supplied".
        interaction = wire_current_interaction(
            "measure", act_parameters={"depth": wire_property_spec("int", 0, "How deep")}
        )

        assert "default: 0" in _act_action(_observation(interaction)).parameters["depth"]

    def test_interaction_name_reaches_the_offered_description(self):
        action = _act_action(_observation(wire_conversation_interaction()))

        assert "conversation" in action.description

    def test_continue_branch_names_the_real_interaction(self):
        observation = _observation(wire_conversation_interaction())

        continues = [
            a for a in observation.get_available_actions() if a.name == ActionType.CONTINUE
        ]

        assert len(continues) == 1
        assert "conversation" in continues[0].description

    def test_interaction_status_prompt_names_the_real_interaction(self):
        text = format_interaction_status(_observation(wire_conversation_interaction()))

        assert "conversation" in text


class TestDegradationMatrix:
    """act_parameter_hints() runs every decision cycle — it must never raise.

    A raised exception on the prompt path collapses the cycle into the WAIT
    fallback, which presents as an NPC that silently stopped acting rather than
    as an error.
    """

    def test_no_current_interaction_is_silent(self, caplog):
        status = StatusObservation(position=(0, 0), current_interaction={})

        with caplog.at_level(logging.WARNING, logger="mind"):
            assert status.act_parameter_hints() == {}

        assert _mind_warnings(caplog) == []

    def test_parameterless_interaction_is_legitimate_and_silent(self, caplog):
        # An interaction that advertises nothing (a chair being sat on) is a
        # normal state, not a contract break.
        status = _status(wire_current_interaction("sit"))

        with caplog.at_level(logging.WARNING, logger="mind"):
            assert status.act_parameter_hints() == {}

        assert _mind_warnings(caplog) == []

    def test_missing_key_on_a_populated_interaction_warns(self, caplog):
        payload = wire_current_interaction("sit")
        del payload[WIRE_KEY_ACT_PARAMETERS]
        status = _status(payload)

        with caplog.at_level(logging.WARNING, logger="mind"):
            assert status.act_parameter_hints() == {}

        assert any(WIRE_KEY_ACT_PARAMETERS in m for m in _mind_warnings(caplog))

    def test_non_dict_payload_warns_and_yields_nothing(self, caplog):
        payload = wire_current_interaction("sit")
        payload[WIRE_KEY_ACT_PARAMETERS] = ["message"]
        status = _status(payload)

        with caplog.at_level(logging.WARNING, logger="mind"):
            assert status.act_parameter_hints() == {}

        assert _mind_warnings(caplog)

    def test_one_malformed_entry_is_skipped_and_the_good_ones_survive(self, caplog):
        payload = wire_current_interaction(
            "conversation",
            act_parameters={
                "message": wire_property_spec("string", "", "Message text"),
                "broken": "not a spec",
            },
        )
        status = _status(payload)

        with caplog.at_level(logging.WARNING, logger="mind"):
            hints = status.act_parameter_hints()

        assert set(hints) == {"message"}
        assert any("broken" in m for m in _mind_warnings(caplog))

    def test_missing_name_falls_back_to_a_generic_label(self, caplog):
        payload = wire_current_interaction("conversation")
        del payload["name"]

        with caplog.at_level(logging.WARNING, logger="mind"):
            assert _status(payload).interaction_display_name() == UNNAMED_INTERACTION

        assert _mind_warnings(caplog)

    def test_offering_actions_survives_a_wholly_malformed_interaction(self):
        # The end-to-end degradation guarantee: whatever the payload, the NPC
        # still gets an action menu.
        status = StatusObservation(
            position=(0, 0),
            current_interaction={"garbage": object()},
            activity_state={"state_name": "interacting"},
        )
        observation = Observation(entity_id="npc", current_simulation_time=1, status=status)

        names = {a.name for a in observation.get_available_actions()}

        assert ActionType.ACT_IN_INTERACTION in names


class TestValidationReachability:
    """Tests 6a/6b — the parameter check must be able to fire at all."""

    def _state(self, current_interaction: dict) -> PipelineState:
        return PipelineState(observation=_observation(current_interaction))

    def test_the_wire_never_carried_the_key_the_old_check_compared(self):
        """6a — passes against un-fixed code; it is the proof the check was dead.

        The pre-fix validator compared ``current_interaction["interaction_name"]``
        against a hardcoded interaction name. The wire contract has no such key,
        so the comparison's left side was always the fallback and the branch was
        unreachable for every interaction that has ever existed.
        """
        assert "interaction_name" not in wire_conversation_interaction()

    def test_bare_act_is_rejected_when_the_interaction_advertises_parameters(self):
        """6b — the regression: fails against un-fixed code, which accepted this."""
        with pytest.raises(ValidationError):
            Action.model_validate(
                {"action": "act_in_interaction", "parameters": {}},
                context={"state": self._state(wire_conversation_interaction())},
            )

    def test_any_single_advertised_parameter_satisfies_the_check(self):
        # PropertySpec supplies defaults for anything omitted, so a partial act
        # is legitimate — only the bare {} is a wasted turn.
        action = Action.model_validate(
            {"action": "act_in_interaction", "parameters": {"is_farewell": True}},
            context={"state": self._state(wire_conversation_interaction())},
        )

        assert action.action == ActionType.ACT_IN_INTERACTION

    def test_unadvertised_parameters_alone_do_not_satisfy_the_check(self):
        with pytest.raises(ValidationError):
            Action.model_validate(
                {"action": "act_in_interaction", "parameters": {"volume": "loud"}},
                context={"state": self._state(wire_conversation_interaction())},
            )

    def test_parameterless_interaction_accepts_a_bare_act(self):
        action = Action.model_validate(
            {"action": "act_in_interaction", "parameters": {}},
            context={"state": self._state(wire_current_interaction("sit"))},
        )

        assert action.action == ActionType.ACT_IN_INTERACTION

    def test_rejection_names_the_advertised_parameters(self):
        with pytest.raises(ValidationError) as exc_info:
            Action.model_validate(
                {"action": "act_in_interaction", "parameters": {}},
                context={"state": self._state(wire_conversation_interaction())},
            )

        message = str(exc_info.value)
        assert "message" in message
        assert "is_farewell" in message


# The functions that offer or validate an act. None of them may name a specific
# interaction or a specific act parameter — that is the acceptance criterion, and
# a structural scan is what keeps a convenience special-case from creeping back
# in the next time one interaction "just needs" a hint.
ACTION_OFFERING_FUNCTIONS = (
    (observation_models, "interaction_display_name"),
    (observation_models, "act_parameter_hints"),
    (observation_models, "_format_parameter_hint"),
    (observation_models, "get_available_actions"),
    (action_models, "_validate_act_in_interaction"),
    (formatting_module, "format_interaction_status"),
)

# Simulation vocabulary: interaction names and act-parameter names drawn from the
# live schema, plus the fictional pair this file invents. ``conversation`` and
# ``message`` are the two the original bug hardcoded.
FORBIDDEN_VOCABULARY = (
    "conversation",
    "message",
    "farewell",
    "debate",
    "stance",
    "conceded",
)


def _function_literals(tree: ast.AST, function_name: str) -> list[str]:
    """String literals inside one named function, its docstring excluded.

    Docstrings are excluded deliberately: prose explaining *why* an interaction
    must not be named is not itself a hardcoded interaction name.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        if node.name != function_name:
            continue

        body = node.body
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
            body = body[1:]  # drop the docstring statement

        return [
            child.value
            for statement in body
            for child in ast.walk(statement)
            if isinstance(child, ast.Constant) and isinstance(child.value, str)
        ]

    raise AssertionError(f"function '{function_name}' not found — the scan is pointed at nothing")


def _offenders(literals: list[str]) -> list[tuple[str, str]]:
    """Whole-word hits only: 'position' must not read as the interaction 'sit'."""
    return [
        (literal, word)
        for literal in literals
        for word in FORBIDDEN_VOCABULARY
        if re.search(rf"\b{word}\b", literal.lower())
    ]


class TestNoInteractionVocabularyInPython:
    """The acceptance criterion, asserted structurally rather than by inspection."""

    @pytest.mark.parametrize(
        ("module", "function_name"),
        ACTION_OFFERING_FUNCTIONS,
        ids=[f"{m.__name__.rsplit('.', 1)[-1]}.{n}" for m, n in ACTION_OFFERING_FUNCTIONS],
    )
    def test_action_offering_code_names_no_interaction_or_parameter(self, module, function_name):
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))

        offenders = _offenders(_function_literals(tree, function_name))

        assert offenders == [], (
            f"{module.__name__}.{function_name} hardcodes simulation vocabulary: {offenders}. "
            "Action offering and validation must be derived from the wire schema."
        )

    def test_the_scan_reports_a_hardcoded_interaction_when_one_is_present(self):
        """Known-positive: a clean report above is only evidence if this goes red.

        Reconstructs the shape of the bug — a branch on a hardcoded interaction
        name supplying a hardcoded parameter hint — and confirms the scan catches
        it.
        """
        sabotaged = ast.parse(
            "def act_parameter_hints(self):\n"
            '    """Docstring naming conversation, which must be ignored."""\n'
            '    if self.name == "conversation":\n'
            '        return {"message": "The message to send in the conversation"}\n'
            "    return {}\n"
        )

        offenders = _offenders(_function_literals(sabotaged, "act_parameter_hints"))

        assert {word for _, word in offenders} == {"conversation", "message"}

    def test_the_scan_ignores_docstring_prose(self):
        documented = ast.parse(
            "def act_parameter_hints(self):\n"
            '    """Never name an interaction here, not even conversation."""\n'
            "    return {}\n"
        )

        assert _offenders(_function_literals(documented, "act_parameter_hints")) == []

    def test_the_scan_fails_loudly_when_pointed_at_nothing(self):
        with pytest.raises(AssertionError):
            _function_literals(ast.parse("x = 1\n"), "act_parameter_hints")


class TestNewInteractionKindNeedsZeroPythonChanges:
    """Test 3 — the acceptance proof.

    A wholly fictional interaction, with a wholly fictional parameter. Neither
    name appears anywhere in this repository's source (enforced by
    ``TestNoInteractionVocabularyInPython`` above), yet both must reach the
    LLM's action menu, its prompt text, and its validator.
    """

    DEBATE = wire_current_interaction(
        name="debate",
        description="A structured argument between two parties",
        act_parameters={
            "stance": wire_property_spec("string", "neutral", "The position you are arguing for"),
            "conceded": wire_property_spec("bool", False, "Whether you yield the point"),
        },
    )

    def test_a_fictional_interaction_reaches_the_action_menu(self):
        action = _act_action(_observation(self.DEBATE))

        assert set(action.parameters) == {"stance", "conceded"}
        assert "The position you are arguing for" in action.parameters["stance"]
        assert "debate" in action.description

    def test_a_fictional_interaction_reaches_the_prompt_text(self):
        assert "debate" in format_interaction_status(_observation(self.DEBATE))

    def test_a_fictional_interaction_s_parameters_are_validated(self):
        state = PipelineState(observation=_observation(self.DEBATE))

        accepted = Action.model_validate(
            {"action": "act_in_interaction", "parameters": {"stance": "for"}},
            context={"state": state},
        )
        assert accepted.action == ActionType.ACT_IN_INTERACTION

        with pytest.raises(ValidationError):
            Action.model_validate(
                {"action": "act_in_interaction", "parameters": {}},
                context={"state": state},
            )
