"""NPC-1903: ``format_conversation_histories`` renders what a mind reads of its partners.

The partner line is prompt content like the terseness instruction: present only when
something is notable, and byte-identical to the pre-channel output otherwise.
"""

from mind.cognitive_architecture.nodes.formatting import format_conversation_histories
from mind.cognitive_architecture.observations import (
    ConversationMessage,
    ConversationObservation,
    ConversationPartner,
    PublicCue,
)
from mind.cognitive_architecture.working_memory import WorkingMemory
from mind.interfaces.mcp.mind import Mind
from tests.fixtures import create_blacksmith_observation

INTERACTION_ID = "interaction_c1"


def _histories() -> dict[str, list[ConversationMessage]]:
    message = ConversationMessage(
        speaker_id="npc_bo",
        speaker_name="Bo",
        message="Morning.",
        timestamp=3,
        is_system=False,
        id="m1",
    )
    return {INTERACTION_ID: [message]}


def _hungry_bo() -> ConversationPartner:
    return ConversationPartner(
        entity_id="npc_bo",
        display_name="Bo",
        appearance=[PublicCue(channel="hunger", band="hungry")],
    )


def test_partner_line_sits_under_the_conversation_header():
    text = format_conversation_histories(
        _histories(), "npc_alice", {INTERACTION_ID: [_hungry_bo()]}
    )

    lines = text.splitlines()
    assert lines[0] == f"Conversation {INTERACTION_ID}:"
    assert lines[1] == "  Bo appears hungry."


def test_output_is_unchanged_when_partners_is_none_empty_or_unnotable():
    baseline = format_conversation_histories(_histories(), "npc_alice")
    unnotable = ConversationPartner(entity_id="npc_bo", display_name="Bo", appearance=[])
    unreadable = ConversationPartner(entity_id="npc_bo", display_name="Bo")

    assert format_conversation_histories(_histories(), "npc_alice", None) == baseline
    assert format_conversation_histories(_histories(), "npc_alice", {}) == baseline
    assert (
        format_conversation_histories(_histories(), "npc_alice", {INTERACTION_ID: [unnotable]})
        == baseline
    )
    assert (
        format_conversation_histories(_histories(), "npc_alice", {INTERACTION_ID: [unreadable]})
        == baseline
    )


def _mind() -> Mind:
    return Mind(
        mind_id="mind_test",
        entity_id="npc_alice",
        traits=[],
        pipeline=None,
        memory_store=None,
        working_memory=WorkingMemory(),
        llm_model="test/model",
    )


def _observation(partner: ConversationPartner | None = None) -> ConversationObservation:
    return ConversationObservation(
        interaction_id=INTERACTION_ID,
        interaction_name="conversation",
        participants=["npc_alice", "npc_bo"],
        initiator_id="npc_alice",
        conversation_history=[],
        partners=[] if partner is None else [partner],
    )


def test_mind_keeps_the_latest_partners_per_conversation():
    mind = _mind()
    mind.update_conversations([_observation(_hungry_bo())])
    fed = ConversationPartner(
        entity_id="npc_bo",
        display_name="Bo",
        appearance=[PublicCue(channel="hunger", band="peckish")],
    )

    mind.update_conversations([_observation(fed)])

    assert mind.conversation_partners[INTERACTION_ID] == [fed], (
        "appearance is a reading of now: the latest observation replaces the last"
    )


def test_snapshot_carries_partners_and_cleanup_drops_them():
    mind = _mind()
    mind.update_conversations([_observation(_hungry_bo())])
    mind._finished_conversation_ids.add(INTERACTION_ID)

    state = mind.build_pipeline_state(create_blacksmith_observation())

    assert state.conversation_partners[INTERACTION_ID][0].appearance is not None
    assert INTERACTION_ID not in mind.conversation_partners
    assert INTERACTION_ID not in mind.conversation_histories


def test_an_observation_after_the_finish_marker_is_cleaned_up_on_the_next_build():
    mind = _mind()
    mind._finished_conversation_ids.add(INTERACTION_ID)
    mind.update_conversations([_observation(_hungry_bo())])
    assert INTERACTION_ID in mind.conversation_partners

    mind.build_pipeline_state(create_blacksmith_observation())

    assert INTERACTION_ID not in mind.conversation_partners
    assert INTERACTION_ID not in mind.conversation_histories
    assert INTERACTION_ID not in mind._finished_conversation_ids


def test_an_observation_that_omits_partners_replaces_the_stored_list_with_empty():
    mind = _mind()
    mind.update_conversations([_observation(_hungry_bo())])

    mind.update_conversations([_observation()])

    assert mind.conversation_partners[INTERACTION_ID] == [], (
        "partners is authoritative per observation: an omitted key means []"
    )
