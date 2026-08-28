"""Regression tests for NPC-1297: conversation message deduplication.

``Mind.update_conversations`` aggregates overlapping rolling windows of
conversation history into one stored transcript. It carried **three** distinct
defects, and only the first is the one the issue title names:

1. The dedup key was ``timestamp`` **alone**, shared across all speakers. The
   simulation truncates game time to whole minutes and deterministically appends
   two messages in one tick (a participant's message, then the system's
   "message limit reached" notice) — so the second was silently dropped.
2. The ``existing_timestamps`` set was computed **once before** the message loop
   and never added to, so a single batch containing the same message twice
   stored it twice.
3. ``timestamp is None`` bypassed dedup **unconditionally and forever**, so an
   unstamped message was re-appended on every cycle that re-sent it.

The fix is a two-branch key: by ``id`` when the producer sent one, by the
composite ``(timestamp, speaker_id, message)`` when it did not. The composite
branch is **permanent, not transitional** — the simulation and this server are
deployed independently, so an id-less producer is a shape that must keep working.

Tests target ``Mind.conversation_histories`` directly: no pipeline node reads
``PipelineState.conversation_histories``, so there is no end-to-end prompt
assertion available to make instead.
"""

import logging

import pytest

from mind.cognitive_architecture.observations import ConversationMessage
from mind.cognitive_architecture.observations.models import ConversationObservation
from mind.cognitive_architecture.working_memory import WorkingMemory
from mind.interfaces.mcp.mind import Mind

INTERACTION_ID = "interaction_conv_1"


def make_mind() -> Mind:
    """A Mind with only the state ``update_conversations`` touches.

    ``update_conversations`` reads ``self.conversation_histories`` and
    ``self.entity_id`` and nothing else, so the pipeline and memory store stay
    ``None`` rather than dragging a live LLM client into a pure-logic test.
    """
    return Mind(
        mind_id="mind_test",
        entity_id="entity_test",
        traits=[],
        pipeline=None,
        memory_store=None,
        working_memory=WorkingMemory(),
        llm_model="test/model",
    )


def make_message(
    speaker_id: str = "npc_alice",
    message: str = "Hello!",
    timestamp: int | None = 10,
    *,
    msg_id: str | None = None,
    is_system: bool = False,
) -> ConversationMessage:
    return ConversationMessage(
        speaker_id=speaker_id,
        speaker_name=speaker_id.replace("npc_", "").title(),
        message=message,
        timestamp=timestamp,
        is_system=is_system,
        id=msg_id,
    )


def make_observation(messages: list[ConversationMessage]) -> ConversationObservation:
    return ConversationObservation(
        interaction_id=INTERACTION_ID,
        interaction_name="conversation",
        participants=["npc_alice", "npc_bob"],
        initiator_id="npc_alice",
        conversation_history=messages,
    )


def stored(mind: Mind) -> list[ConversationMessage]:
    return mind.conversation_histories.get(INTERACTION_ID, [])


class TestSameTimestampDistinctSpeakers:
    """Defect 1: the key was ``timestamp`` alone, so one of the two was dropped.

    These are deliberately **two-cycle** tests. In a single batch defect 2 masks
    defect 1 — the old code never added to its timestamp set, so both messages
    appended and the test would pass against the un-fixed code. Two cycles is
    also the faithful reproduction: the simulation calls ``send_observations()``
    once after the participant's message and again after the system notice.
    """

    def test_two_same_timestamp_messages_from_different_speakers_both_survive(self):
        """The exact shape the simulation emits at the message cap.

        ``_handle_act`` appends the participant's message and sends observations,
        then ``_check_message_limits_and_end`` appends a system message and sends
        again — with no game time between them, so both carry the same truncated
        integer timestamp. Under the old key the system notice was silently lost.
        """
        mind = make_mind()
        alice = make_message("npc_alice", "Hello!", 10, msg_id="message_a")
        system = make_message(
            "system", "Message limit reached (15).", 10, msg_id="message_b", is_system=True
        )

        mind.update_conversations([make_observation([alice])])
        mind.update_conversations([make_observation([alice, system])])

        assert len(stored(mind)) == 2
        assert [m.message for m in stored(mind)] == ["Hello!", "Message limit reached (15)."]

    def test_same_timestamp_distinct_speakers_survive_without_ids(self):
        """The composite branch must fix defect 1 too — it is the id-less path's key."""
        mind = make_mind()
        alice = make_message("npc_alice", "Hello!", 10)
        system = make_message("system", "Message limit reached (15).", 10, is_system=True)

        mind.update_conversations([make_observation([alice])])
        mind.update_conversations([make_observation([alice, system])])

        assert len(stored(mind)) == 2


class TestDedupStillHolds:
    """Anti-cheat: deleting the dedup mechanism must NOT make the suite pass.

    Without this, the tests above could be "fixed" by appending unconditionally.
    """

    def test_same_message_across_three_overlapping_windows_stored_once(self):
        """Rolling windows re-send the same message every cycle."""
        mind = make_mind()
        first = make_message("npc_alice", "Hello!", 10, msg_id="message_a")
        second = make_message("npc_bob", "Hi there!", 11, msg_id="message_b")
        third = make_message("npc_alice", "How are you?", 12, msg_id="message_c")

        mind.update_conversations([make_observation([first])])
        mind.update_conversations([make_observation([first, second])])
        mind.update_conversations([make_observation([first, second, third])])

        assert len(stored(mind)) == 3
        assert [m.id for m in stored(mind)] == ["message_a", "message_b", "message_c"]

    def test_same_message_across_overlapping_windows_stored_once_without_ids(self):
        mind = make_mind()
        first = make_message("npc_alice", "Hello!", 10)
        second = make_message("npc_bob", "Hi there!", 11)

        mind.update_conversations([make_observation([first])])
        mind.update_conversations([make_observation([first, second])])
        mind.update_conversations([make_observation([first, second])])

        assert len(stored(mind)) == 2


class TestIntraBatchDuplicates:
    """Defect 2: the index was built once before the loop and never added to."""

    def test_duplicate_within_a_single_batch_stored_once_by_id(self):
        mind = make_mind()
        msg = make_message("npc_alice", "Hello!", 10, msg_id="message_a")

        mind.update_conversations([make_observation([msg, msg])])

        assert len(stored(mind)) == 1

    def test_duplicate_within_a_single_batch_stored_once_by_composite(self):
        mind = make_mind()

        mind.update_conversations(
            [
                make_observation(
                    [
                        make_message("npc_alice", "Hello!", 10),
                        make_message("npc_alice", "Hello!", 10),
                    ]
                )
            ]
        )

        assert len(stored(mind)) == 1


class TestNullTimestamp:
    """Defect 3: ``timestamp is None`` bypassed dedup unconditionally."""

    def test_null_timestamp_message_is_deduplicated(self):
        mind = make_mind()
        msg = make_message("npc_alice", "Hello!", None)

        mind.update_conversations([make_observation([msg])])
        mind.update_conversations([make_observation([msg])])
        mind.update_conversations([make_observation([msg])])

        assert len(stored(mind)) == 1

    def test_distinct_null_timestamp_messages_are_all_kept(self):
        """Paired positive: dedup must not over-collapse unstamped messages.

        Without this, defect 3 could be "fixed" by dropping every ``None``-stamped
        message after the first, which is a different bug with the same test count.
        """
        mind = make_mind()

        mind.update_conversations(
            [
                make_observation(
                    [
                        make_message("npc_alice", "Hello!", None),
                        make_message("npc_bob", "Hi there!", None),
                        make_message("npc_alice", "How are you?", None),
                    ]
                )
            ]
        )

        assert len(stored(mind)) == 3


class TestMixedAndVersionSkew:
    """Both branches active at once, and the sim-downgrade / sim-upgrade paths."""

    def test_mixed_id_and_idless_batch_uses_both_branches(self):
        mind = make_mind()

        mind.update_conversations(
            [
                make_observation(
                    [
                        make_message("npc_alice", "Hello!", 10, msg_id="message_a"),
                        make_message("npc_bob", "Hi there!", 10),
                    ]
                )
            ]
        )
        # Re-sent identically: neither branch may append a second copy.
        mind.update_conversations(
            [
                make_observation(
                    [
                        make_message("npc_alice", "Hello!", 10, msg_id="message_a"),
                        make_message("npc_bob", "Hi there!", 10),
                    ]
                )
            ]
        )

        assert len(stored(mind)) == 2

    def test_message_stored_with_id_is_recognised_when_it_rearrives_without_one(self):
        """Simulation downgrade: the composite index must cover id-bearing messages."""
        mind = make_mind()

        mind.update_conversations(
            [make_observation([make_message("npc_alice", "Hello!", 10, msg_id="message_a")])]
        )
        mind.update_conversations([make_observation([make_message("npc_alice", "Hello!", 10)])])

        assert len(stored(mind)) == 1
        assert stored(mind)[0].id == "message_a"

    def test_message_stored_without_id_is_upgraded_in_place_when_the_id_arrives(self):
        """Simulation upgrade: adopt the id onto the held copy, do not append a second."""
        mind = make_mind()

        mind.update_conversations([make_observation([make_message("npc_alice", "Hello!", 10)])])
        assert stored(mind)[0].id is None

        mind.update_conversations(
            [make_observation([make_message("npc_alice", "Hello!", 10, msg_id="message_a")])]
        )

        assert len(stored(mind)) == 1
        # Backfilled, not merely deduplicated — a later id-keyed re-send must hit
        # the id branch rather than falling through to the composite one.
        assert stored(mind)[0].id == "message_a"


class TestBoundaryIntegrity:
    """An empty id is a producer bug, not an absent id."""

    def test_empty_id_falls_back_to_composite_and_warns(self, caplog):
        mind = make_mind()

        with caplog.at_level(logging.WARNING):
            mind.update_conversations(
                [
                    make_observation(
                        [
                            make_message("npc_alice", "Hello!", 10, msg_id=""),
                            make_message("npc_alice", "Hello!", 10, msg_id=""),
                        ]
                    )
                ]
            )

        assert len(stored(mind)) == 1, "empty ids must dedup via the composite key"
        assert any("EMPTY id" in record.message for record in caplog.records)

    def test_distinct_empty_id_messages_are_not_collapsed_together(self):
        """ "" must never act as an identity — every blank would collide with every other."""
        mind = make_mind()

        mind.update_conversations(
            [
                make_observation(
                    [
                        make_message("npc_alice", "Hello!", 10, msg_id=""),
                        make_message("npc_bob", "Hi there!", 10, msg_id=""),
                    ]
                )
            ]
        )

        assert len(stored(mind)) == 2

    def test_composite_collision_under_distinct_ids_keeps_both_and_warns(self, caplog):
        mind = make_mind()

        with caplog.at_level(logging.WARNING):
            mind.update_conversations(
                [
                    make_observation(
                        [
                            make_message("npc_alice", "Hello!", 10, msg_id="message_a"),
                            make_message("npc_alice", "Hello!", 10, msg_id="message_b"),
                        ]
                    )
                ]
            )

        assert len(stored(mind)) == 2
        assert [m.id for m in stored(mind)] == ["message_a", "message_b"]
        assert any("distinct ids" in record.message for record in caplog.records)


class TestIdFieldIsOptional:
    """Version skew: the field is permanently optional, not transitional."""

    def test_conversation_message_parses_without_an_id(self):
        msg = ConversationMessage.model_validate(
            {
                "speaker_id": "npc_alice",
                "speaker_name": "Alice",
                "message": "Hello!",
                "timestamp": 10,
                "is_system": False,
            }
        )
        assert msg.id is None

    def test_conversation_message_parses_with_an_id(self):
        msg = ConversationMessage.model_validate(
            {
                "speaker_id": "npc_alice",
                "speaker_name": "Alice",
                "message": "Hello!",
                "timestamp": 10,
                "is_system": False,
                "id": "message_a",
            }
        )
        assert msg.id == "message_a"


@pytest.mark.parametrize(
    "timestamp",
    [0, 10, None],
    ids=["zero", "positive", "absent"],
)
def test_dedup_holds_across_timestamp_shapes(timestamp):
    """Zero is a valid game-minute reading, not a "not set" sentinel."""
    mind = make_mind()
    msg = make_message("npc_alice", "Hello!", timestamp)

    mind.update_conversations([make_observation([msg])])
    mind.update_conversations([make_observation([msg])])

    assert len(stored(mind)) == 1
