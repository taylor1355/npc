"""FS10b lane 3 (M1/M3): the DEED_PERCEIVED mind event and its shared constants."""

import logging

import pytest

from mind.cognitive_architecture.actions import Action, ActionType
from mind.cognitive_architecture.observations import MindEvent, MindEventType, RecalledDeed
from mind.cognitive_architecture.observations import models as observation_models
from mind.cognitive_architecture.observations.models import _age_in_words
from mind.cognitive_architecture.state import PipelineState
from mind.constants import DEFAULT_RECENCY_DECAY_PER_GAME_HOUR
from mind.interfaces.mcp.server import MCPServer
from tests.unit.test_mcp_server import parse_response

# Shape of DeedPerceivedObservation.get_data in the simulation.
WIRE_EVENT = {
    "timestamp": 480,
    "event_type": "DEED_PERCEIVED",
    "payload": {
        "act": "gift",
        "patients": ["Bob"],
        "place": "Market",
        "role": "patient",
        "actor_id": "npc_alice",
        "actor": "Alice",
        "footprint": 0.62,
    },
}


def test_recency_decay_matches_the_simulation_twin():
    # Twin: the simulation's deed_memory.gd::MIND_RECENCY_DECAY_PER_GAME_HOUR. The
    # substrate keeps the deeds this mind would rank highest only while both agree.
    assert DEFAULT_RECENCY_DECAY_PER_GAME_HOUR == 0.995


def test_event_round_trips_through_model_validate():
    event = MindEvent.model_validate(WIRE_EVENT)
    assert event.event_type is MindEventType.DEED_PERCEIVED
    assert event.payload["act"] == "gift"
    assert MindEvent.model_validate(event.model_dump()) == event


def test_str_renders_words_not_numbers():
    text = str(MindEvent.model_validate(WIRE_EVENT))
    assert text == "Alice did a deed: gift to Bob at Market (done to you)"
    assert "0.62" not in text


def test_str_is_total_over_malformed_payloads():
    for payload in ({}, {"patients": "Bob", "act": 5}, {"patients": [None, 3], "place": None}):
        event = MindEvent(timestamp=1, event_type=MindEventType.DEED_PERCEIVED, payload=payload)
        assert isinstance(str(event), str)
    assert str(MindEvent(timestamp=1, event_type="DEED_PERCEIVED", payload={})).startswith(
        "Someone did a deed"
    )


async def test_decide_action_accepts_deed_perceived_event():
    server = MCPServer()
    await server.mcp.call_tool(
        "create_mind",
        {
            "mind_id": "mind_test",
            "entity_id": "entity_test",
            "config": {"traits": [], "initial_long_term_memories": []},
        },
    )
    mind = server.minds["mind_test"]

    async def mock_process(state: PipelineState) -> PipelineState:
        state.chosen_action = Action.model_construct(action=ActionType.WAIT, parameters={})
        return state

    mind.pipeline.process = mock_process
    result = await server.mcp.call_tool(
        "decide_action",
        {
            "mind_id": "mind_test",
            "observation": {
                "entity_id": "entity_test",
                "current_simulation_time": 480,
                "status": {
                    "position": [5, 5],
                    "movement_locked": False,
                    "current_interaction": {},
                    "controller_state": {},
                },
            },
            "events": [WIRE_EVENT],
        },
    )
    response = parse_response(result)
    assert response["status"] == "success", response


def test_non_string_patients_never_render():
    event = MindEvent(
        timestamp=1,
        event_type=MindEventType.DEED_PERCEIVED,
        payload={"act": "gift", "patients": [True, 0, "Bob"]},
    )
    assert str(event) == "Someone did a deed: gift to Bob"


def test_event_and_conduct_render_the_same_clause():
    fields = {"act": "gift", "patients": ["Bob", "Cy"], "place": "Market", "role": "patient"}
    clause = RecalledDeed.model_validate(fields).render()
    event = MindEvent(timestamp=1, event_type=MindEventType.DEED_PERCEIVED, payload=fields)
    assert str(event) == f"Someone did a deed: {clause}"
    empty = RecalledDeed().render()
    assert str(MindEvent(timestamp=1, event_type="DEED_PERCEIVED", payload={})) == (
        f"Someone did a deed: {empty}"
    )


@pytest.fixture(autouse=True)
def _reset_warned_roles():
    observation_models._WARNED_DEED_ROLES.clear()
    yield
    observation_models._WARNED_DEED_ROLES.clear()


def _warnings(caplog):
    return [r for r in caplog.records if "Unknown deed role" in r.getMessage()]


def test_unknown_conduct_role_warns_and_renders_as_a_perceiver(caplog):
    with caplog.at_level(logging.WARNING):
        deed = RecalledDeed(act="gift", role="bystander")
    assert deed.role == "bystander"
    assert len(_warnings(caplog)) == 1
    assert "done to you" not in deed.render()


def test_unknown_event_role_warns_and_renders_as_a_perceiver(caplog):
    event = MindEvent(
        timestamp=1,
        event_type=MindEventType.DEED_PERCEIVED,
        payload={"act": "gift", "role": "bystander"},
    )
    with caplog.at_level(logging.WARNING):
        text = str(event)
    assert len(_warnings(caplog)) == 1
    assert text == "Someone did a deed: gift"


def test_known_roles_do_not_warn(caplog):
    with caplog.at_level(logging.WARNING):
        RecalledDeed(role="patient")
        RecalledDeed(role="perceiver")
        str(MindEvent.model_validate(WIRE_EVENT))
    assert not _warnings(caplog)


@pytest.mark.parametrize(
    ("minutes", "band"),
    [
        (0, "just now"),
        (29.9, "just now"),
        (30, "a little while ago"),
        (239.9, "a little while ago"),
        (240, "earlier today"),
        (1439.9, "earlier today"),
        (1440, "a day or two ago"),
        (4319.9, "a day or two ago"),
        (4320, "a while ago"),
        (-1, ""),
        (float("nan"), ""),
        (float("inf"), ""),
        (float("-inf"), ""),
        (None, ""),
        (True, ""),
    ],
)
def test_age_bands_at_their_boundaries(minutes, band):
    assert _age_in_words(minutes) == band


def test_an_unknown_role_warns_once_per_distinct_value(caplog):
    with caplog.at_level(logging.WARNING):
        RecalledDeed(role="bystander")
        RecalledDeed(role="bystander")
        str(MindEvent(timestamp=1, event_type="DEED_PERCEIVED", payload={"role": "bystander"}))
        assert len(_warnings(caplog)) == 1
        RecalledDeed(role="witness")
        RecalledDeed(role="witness")
    assert len(_warnings(caplog)) == 2
