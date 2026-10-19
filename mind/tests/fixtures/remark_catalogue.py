"""The simulation's remark catalogue and a remark-speaking conversation, as data.

``remark_catalogue_4e38abc0f.json`` is the catalogue ``remark_catalogue.gd::build``
is specified to emit (NPC-1916 plan section 2.1) over the shipped grammar at game
commit ``4e38abc0f``: the 20 wire remarks of ``resources/grammar/grammar.tres``
(provisional ``response.*`` children excluded), sorted by key, each gloss its
primary gloss with ``{slot}`` placeholders, ``declare_with`` the declaring kind's
``parameter_specs()`` keys in declaration order. It was transcribed by hand from
that commit because the builder lands in the game repository after this one
(Decision 6); re-derive it from the builder's output once it exists.

Remark keys appear here and nowhere in ``src/`` -- this is data, the role the
simulation plays in production.
"""

from __future__ import annotations

import json
from pathlib import Path

from mind.cognitive_architecture.observations import Observation, StatusObservation
from mind.cognitive_architecture.remark_catalogue import RemarkCatalogue
from mind.cognitive_architecture.state import PipelineState

from .observations import wire_current_interaction, wire_property_spec

# The game commit the catalogue and the advertised parameters were read from.
GAME_SHA = "4e38abc0f"

CATALOGUE_PATH = Path(__file__).with_name(f"remark_catalogue_{GAME_SHA}.json")


def wire_catalogue() -> dict:
    """The catalogue exactly as it crosses the wire in ``create_mind``."""
    return json.loads(CATALOGUE_PATH.read_text(encoding="utf-8"))


def catalogue() -> RemarkCatalogue:
    return RemarkCatalogue.model_validate(wire_catalogue())


def wire_remark_conversation_interaction() -> dict:
    """The conversation interaction at ``GAME_SHA``, every act parameter it advertises.

    ``conversation_interaction.gd::_init`` declares ``message``,
    ``utterance_element`` and ``addressee_ids`` inline, then
    ``DeclarationRegistry.contribute_parameter_specs`` adds each kind's specs in
    registry order (farewell, share_belief, ask_who_told, answer).
    """
    return wire_current_interaction(
        name="conversation",
        description="Multi-party conversation",
        act_parameters={
            "message": wire_property_spec("string", "", "Message text to send in conversation"),
            "utterance_element": wire_property_spec(
                "string", "", "The key of the grammar remark this message says."
            ),
            "addressee_ids": wire_property_spec(
                "array", [], "Optional: ids of the participants this message is spoken TO."
            ),
            "is_farewell": wire_property_spec(
                "bool", False, "Whether the speaker intends this message to end the conversation"
            ),
            "share_belief_kind": wire_property_spec(
                "string", "", "Typed belief-record kind to share"
            ),
            "share_belief_subject_id": wire_property_spec(
                "string", "", "Stable id of the belief subject"
            ),
            "ask_who_told_kind": wire_property_spec(
                "string",
                "",
                "Belief-record kind asked about (place_knowledge); empty = not asking",
            ),
            "ask_who_told_subject_id": wire_property_spec(
                "string", "", "Id of the place to ask 'who told you about it?'"
            ),
            "answer_in_reply_to": wire_property_spec(
                "string",
                "",
                "Message id of the open question addressed to you that this message answers. "
                "The answer's content comes from your own memory, not from parameters.",
            ),
        },
        needs_filled=["social"],
    )


def remark_conversation_observation(entity_id: str = "npc_a") -> Observation:
    return Observation(
        entity_id=entity_id,
        current_simulation_time=100,
        status=StatusObservation(
            position=(5, 5),
            movement_locked=True,
            current_interaction=wire_remark_conversation_interaction(),
            activity_state={"state_name": "interacting"},
        ),
    )


def remark_conversation_state(with_catalogue: bool = True) -> PipelineState:
    return PipelineState(
        observation=remark_conversation_observation(),
        remark_catalogue=catalogue() if with_catalogue else None,
    )
