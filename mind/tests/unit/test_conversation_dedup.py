"""Regression tests for NPC-1297: conversation message deduplication.

``Mind.update_conversations`` aggregates overlapping rolling windows of
conversation history into one stored transcript. It carried **three** defects:

1. The dedup key was ``timestamp`` **alone**, shared across all speakers. The
   simulation truncates game time to whole minutes and deterministically appends
   two messages in one tick (a participant's message, then the system's
   "message limit reached" notice) — so the second was silently dropped.
2. The index was computed **once before** the message loop and never added to,
   so a single batch containing the same message twice stored it twice.
3. ``timestamp is None`` bypassed dedup **unconditionally and forever**.

The fix is to key on the message's ``id``, which the simulation mints as a class
invariant. Defect 1 dissolves (ids distinguish same-minute messages), defect 3
dissolves (timestamps take no part in identity at all), and defect 2 is fixed
separately — by mutating the index *inside* the loop, which an id-only key still
requires.

There is deliberately **no fallback keying strategy**. ``id`` is required, an
id-less payload is refused loudly at the parse boundary, and an empty id is
refused loudly here. Keeping a composite-key path alive for a producer that
cannot exist would be the "parallel legacy code path" the project's anti-pattern
table forbids.

Tests target ``Mind.conversation_histories`` directly: no pipeline node reads
``PipelineState.conversation_histories``, so there is no end-to-end prompt
assertion available to make instead.
"""

import logging

import pytest
from pydantic import ValidationError

from mind.cognitive_architecture.observations import ConversationMessage, MindEvent, MindEventType
from mind.cognitive_architecture.observations.models import ConversationObservation
from mind.cognitive_architecture.working_memory import WorkingMemory
from mind.interfaces.mcp.mind import Mind
from mind.interfaces.mcp.server import (
    CONVERSATION_MARKER_FIELD,
    _extract_conversation_observations,
)

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
    msg_id: str = "message_a",
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

    Deliberately a **two-cycle** test. In a single batch defect 2 masks defect 1
    — the old code never added to its timestamp set, so both messages appended
    and the test would pass against the un-fixed code. Two cycles is also the
    faithful reproduction: the simulation calls ``send_observations()`` once
    after the participant's message and again after the system notice.
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


class TestDedupStillHolds:
    """Anti-cheat: deleting the dedup mechanism must NOT make the suite pass.

    Without this, the test above could be "fixed" by appending unconditionally.
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


class TestIntraBatchDuplicates:
    """Defect 2: the index was built once before the loop and never added to.

    Independent of the keying strategy — an id-only key gets this wrong too if
    the set is not mutated inside the loop.
    """

    def test_duplicate_within_a_single_batch_stored_once(self):
        mind = make_mind()
        msg = make_message("npc_alice", "Hello!", 10, msg_id="message_a")

        mind.update_conversations([make_observation([msg, msg])])

        assert len(stored(mind)) == 1

    def test_three_copies_within_a_single_batch_stored_once(self):
        mind = make_mind()
        msg = make_message("npc_alice", "Hello!", 10, msg_id="message_a")

        mind.update_conversations([make_observation([msg, msg, msg])])

        assert len(stored(mind)) == 1


class TestIdentityIsTheIdAlone:
    """Defect 3 dissolved: no field other than ``id`` takes part in identity.

    These are the guard against quietly reintroducing a composite key. Each
    fails if ``timestamp``, ``speaker_id``, or ``message`` is folded back in.
    """

    def test_same_id_dedups_even_when_the_timestamp_differs(self):
        """A timestamp is not part of identity, so a re-send may correct it freely."""
        mind = make_mind()

        mind.update_conversations(
            [make_observation([make_message("npc_alice", "Hello!", 10, msg_id="message_a")])]
        )
        mind.update_conversations(
            [make_observation([make_message("npc_alice", "Hello!", 11, msg_id="message_a")])]
        )

        assert len(stored(mind)) == 1

    def test_same_id_dedups_when_the_timestamp_is_absent(self):
        """The old code let a ``None`` timestamp bypass dedup forever."""
        mind = make_mind()
        msg = make_message("npc_alice", "Hello!", None, msg_id="message_a")

        mind.update_conversations([make_observation([msg])])
        mind.update_conversations([make_observation([msg])])
        mind.update_conversations([make_observation([msg])])

        assert len(stored(mind)) == 1

    def test_distinct_ids_are_kept_when_every_other_field_is_identical(self):
        """Two messages identical but for their ids are two messages."""
        mind = make_mind()

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


class TestEmptyIdIsRefusedLoudly:
    """An empty id is a producer bug, and there is nowhere to fall through to."""

    def test_empty_id_message_is_refused_and_logged_at_error(self, caplog):
        mind = make_mind()

        with caplog.at_level(logging.ERROR):
            mind.update_conversations([make_observation([make_message(msg_id="")])])

        assert stored(mind) == [], "A message with no usable identity must not be stored"
        assert any(
            "EMPTY id" in record.message and record.levelno >= logging.ERROR
            for record in caplog.records
        )

    def test_whitespace_only_id_is_refused(self):
        """Normalisation and the emptiness check must agree on what blank means."""
        mind = make_mind()

        mind.update_conversations([make_observation([make_message(msg_id="   ")])])

        assert stored(mind) == []

    def test_a_valid_message_in_the_same_batch_still_lands(self):
        """Refusal is per-message; one bad message must not cost the others."""
        mind = make_mind()

        mind.update_conversations(
            [
                make_observation(
                    [
                        make_message("npc_alice", "Hello!", 10, msg_id=""),
                        make_message("npc_bob", "Hi there!", 10, msg_id="message_b"),
                    ]
                )
            ]
        )

        assert [m.id for m in stored(mind)] == ["message_b"]

    def test_surrounding_whitespace_is_normalised_before_keying(self):
        """The value validated must be the value stored and keyed on."""
        mind = make_mind()

        mind.update_conversations([make_observation([make_message(msg_id="  message_a  ")])])
        mind.update_conversations([make_observation([make_message(msg_id="message_a")])])

        assert len(stored(mind)) == 1
        assert stored(mind)[0].id == "message_a"


class TestIdIsRequired:
    """The model rejects an id-less message rather than accommodating it."""

    def test_conversation_message_requires_an_id(self):
        with pytest.raises(ValidationError):
            ConversationMessage.model_validate(
                {
                    "speaker_id": "npc_alice",
                    "speaker_name": "Alice",
                    "message": "Hello!",
                    "timestamp": 10,
                    "is_system": False,
                }
            )

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


class TestTheLoudBranchCannotRotUnnoticed:
    """The malformed-conversation branch keys on a payload FIELD NAME.

    Renaming that field on the wire would leave the discriminator matching
    nothing: every malformed conversation would take the routine debug branch,
    and the loud path would be dead while still looking alive. Nothing else
    would notice, because a check that stops firing reports exactly what a
    clean tree reports.
    """

    def test_the_conversation_marker_field_still_exists_on_the_model(self):
        assert CONVERSATION_MARKER_FIELD in ConversationObservation.model_fields


class TestMalformedConversationIsLoudNotSilent:
    """A stricter model must not turn into a silent whole-conversation drop.

    ``_extract_conversation_observations`` validates inside ``except
    ValidationError: continue``. Most payloads reaching it are other interaction
    kinds and are skipped routinely — but a payload that IS a conversation and
    fails validation would otherwise vanish with nothing said, which is strictly
    worse than the dedup bug this issue is about.
    """

    def _event(self, payload: dict) -> MindEvent:
        return MindEvent(
            timestamp=1, event_type=MindEventType.INTERACTION_OBSERVATION, payload=payload
        )

    def test_id_less_conversation_payload_is_refused_loudly(self, caplog):
        payload = {
            "interaction_id": INTERACTION_ID,
            "interaction_name": "conversation",
            "participants": ["npc_alice"],
            "initiator_id": "npc_alice",
            "conversation_history": [
                {
                    "speaker_id": "npc_alice",
                    "speaker_name": "Alice",
                    "message": "Hello!",
                    "timestamp": 10,
                    "is_system": False,
                }
            ],
        }

        with caplog.at_level(logging.DEBUG):
            result = _extract_conversation_observations([self._event(payload)], "entity_test")

        assert result == [], "An unparseable conversation must not be handed on as valid"
        assert any(
            "MALFORMED conversation observation" in record.message
            and record.levelno >= logging.ERROR
            for record in caplog.records
        ), "A dropped conversation must be reported at ERROR, not swallowed at debug"

    def test_non_conversation_observation_is_still_skipped_quietly(self, caplog):
        """The routine case must not become noise — sitting and eating land here too."""
        payload = {"interaction_id": "interaction_sit_1", "interaction_name": "sit"}

        with caplog.at_level(logging.DEBUG):
            result = _extract_conversation_observations([self._event(payload)], "entity_test")

        assert result == []
        assert not any(record.levelno >= logging.ERROR for record in caplog.records), (
            "A non-conversation interaction observation is routine, not an error"
        )

    def test_a_well_formed_conversation_still_parses(self, caplog):
        """Paired positive: the loud branch must not swallow the good case."""
        payload = {
            "interaction_id": INTERACTION_ID,
            "interaction_name": "conversation",
            "participants": ["npc_alice"],
            "initiator_id": "npc_alice",
            "conversation_history": [
                {
                    "speaker_id": "npc_alice",
                    "speaker_name": "Alice",
                    "message": "Hello!",
                    "timestamp": 10,
                    "is_system": False,
                    "id": "message_a",
                }
            ],
        }

        with caplog.at_level(logging.DEBUG):
            result = _extract_conversation_observations([self._event(payload)], "entity_test")

        assert len(result) == 1
        assert result[0].conversation_history[0].id == "message_a"
        assert not any(record.levelno >= logging.ERROR for record in caplog.records)
