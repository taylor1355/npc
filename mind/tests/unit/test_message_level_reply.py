"""NPC-1909: an answer is ``is_answer`` plus the message-level ``in_reply_to``.

The simulation removed ``answer_in_reply_to``; its ``DeclarationKind.message_parameters()``
lists ``in_reply_to`` in the answer kind's ``declare_with``. The mind sends whatever the
conversation interaction advertises, so the contract is the advertised parameters and the
catalogue fixture, both pinned here to the simulation's copy.
"""

from mind.cognitive_architecture.actions import ActionType
from tests.fixtures.remark_catalogue import (
    catalogue,
    remark_conversation_observation,
    wire_catalogue,
)


def _act_parameters() -> dict:
    observation = remark_conversation_observation()
    action = next(
        a for a in observation.get_available_actions() if a.name == ActionType.ACT_IN_INTERACTION
    )
    return action.parameters


def test_answer_parameters_are_the_message_level_pair():
    params = _act_parameters()
    assert "is_answer" in params
    assert "in_reply_to" in params
    assert "answer_in_reply_to" not in params


def test_every_answer_remark_declares_with_is_answer_and_in_reply_to():
    answers = [r for r in wire_catalogue()["remarks"] if r.get("declares") == "answer"]
    assert answers
    for remark in answers:
        assert remark["declare_with"] == ["is_answer", "in_reply_to"]
    assert catalogue().declaring_parameters()  # still parses end to end
