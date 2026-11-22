"""The simulation's dimension catalogue and a graded conversation, as data.

``dimension_catalogue_contract.json`` is the ``dimensions`` block of the
``create_mind`` payload as NPC-2312's design section 8 specifies it, carrying the
cordiality band table of section 2.5. The simulation's contract test pins its
own builder against this file (the role ``remark_catalogue_4e38abc0f.json``
plays for remarks), so a change to either side's shape fails in one place.

Dimension names appear here and nowhere in ``src/`` -- this is data, the role the
simulation plays in production.
"""

from __future__ import annotations

import json
from pathlib import Path

from mind.cognitive_architecture.dimension_catalogue import DimensionCatalogue
from mind.cognitive_architecture.observations import (
    ConversationMessage,
    Observation,
    StatusObservation,
)
from mind.cognitive_architecture.state import PipelineState

from .observations import wire_property_spec
from .remark_catalogue import wire_remark_conversation_interaction

CONTRACT_PATH = Path(__file__).with_name("dimension_catalogue_contract.json")


def wire_dimensions() -> dict:
    """The ``dimensions`` block exactly as it crosses the wire in ``create_mind``."""
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def dimensions() -> DimensionCatalogue:
    return DimensionCatalogue.model_validate(wire_dimensions())


def wire_graded_conversation_interaction() -> dict:
    """The conversation interaction, advertising one required float per dimension."""
    interaction = wire_remark_conversation_interaction()
    for dimension in wire_dimensions()["dimensions"]:
        interaction["act_in_interaction_parameters"][dimension["parameter"]] = wire_property_spec(
            "float",
            None,
            f"How you treat the listener, {dimension['low']:g} to {dimension['high']:g}.",
            required=True,
        )
    return interaction


def graded_conversation_state(with_dimensions: bool = True, declarations=None) -> PipelineState:
    """A conversation where Bo's last line carries ``declarations`` (heard lines)."""
    observation = Observation(
        entity_id="npc_a",
        current_simulation_time=100,
        status=StatusObservation(
            position=(5, 5),
            movement_locked=True,
            current_interaction=wire_graded_conversation_interaction(),
            activity_state={"state_name": "interacting"},
        ),
    )
    return PipelineState(
        observation=observation,
        dimensions=dimensions() if with_dimensions else None,
        conversation_histories={
            "conv_fixture": [
                ConversationMessage(
                    id="msg_fixture",
                    speaker_id="npc_b",
                    speaker_name="Bo",
                    message="You again.",
                    declarations=list(declarations or []),
                )
            ]
        },
    )
