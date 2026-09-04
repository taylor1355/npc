"""Unit tests for format_recent_events (NPC-1335).

The event buffer used to reach the reflection prompt as ``pformat`` output —
Python ``repr`` of Pydantic models, enum members and all. These tests pin the
prose rendering that replaced it, and the guard that keeps a repr from creeping
back in.
"""

from mind.cognitive_architecture.nodes.formatting import (
    format_conversation_histories,
    format_recent_events,
)
from mind.cognitive_architecture.observations import (
    ConversationMessage,
    MindEvent,
    MindEventType,
)
from tests.fixtures.observations import create_conversation_events, create_social_events


def _event(timestamp: int, event_type: MindEventType, /, **payload) -> MindEvent:
    """Build one buffer entry.

    The leading parameters are positional-only because bid payloads carry their
    own ``timestamp`` wire key, which would otherwise collide with this
    helper's.
    """
    return MindEvent(timestamp=timestamp, event_type=event_type, payload=payload)


class TestFormatRecentEvents:
    """Rendering of the recent-events buffer for the reflection prompt"""

    def test_empty_buffer_renders_a_sentinel(self):
        """A blank section reads to the model as a formatting error.

        ``pformat([])`` at least rendered "[]"; a bare ``"\\n".join([])`` would
        leave the "### Recent Events" heading followed by nothing.
        """
        rendered = format_recent_events([])

        assert rendered.strip()
        assert "MindEvent" not in rendered

    def test_each_event_is_one_line_prefixed_with_its_timestamp(self):
        events = [
            _event(101, MindEventType.INTERACTION_STARTED, interaction_name="conversation"),
            _event(104, MindEventType.INTERACTION_FINISHED, interaction_name="conversation"),
        ]

        rendered = format_recent_events(events)

        lines = rendered.split("\n")
        assert len(lines) == 2
        assert lines[0] == "[t=101] Interaction started: conversation"
        assert lines[1] == "[t=104] Interaction finished: conversation"

    def test_output_contains_no_model_repr(self):
        """The regression guard for the whole issue.

        Every one of these substrings appears in the ``pformat`` rendering this
        replaced, and none of them carries information the model can use.
        """
        events = [
            _event(
                103,
                MindEventType.INTERACTION_BID_RECEIVED,
                interaction_name="conversation",
                bid_type=0,
                bid_id="bid_e5f6a7b8",
                bidder_id="npc_carol",
                provider_id="npc_alice",
                timestamp=103.0,
                force=False,
            ),
            _event(
                105,
                MindEventType.MOVEMENT_COMPLETED,
                status="ARRIVED",
                intended_destination=[10, 20],
                actual_destination=[10, 20],
            ),
        ]

        rendered = format_recent_events(events)

        assert "MindEvent(" not in rendered
        assert "event_type=<" not in rendered
        assert "MindEventType." not in rendered
        assert "payload={" not in rendered
        assert "Interaction bid received: conversation" in rendered
        assert "Arrived at (10, 20)" in rendered

    def test_rendering_is_substantially_smaller_than_the_repr(self):
        """The point of the change, asserted as a property rather than a count.

        A ratio rather than a token figure: the absolute number depends on the
        buffer's composition, but no plausible composition of non-conversation
        events should render longer than half its repr.
        """
        from pprint import pformat

        events = [
            _event(100 + i, MindEventType.INTERACTION_STARTED, interaction_name="conversation")
            for i in range(5)
        ]

        assert len(format_recent_events(events)) < len(pformat(events)) / 2


class TestCommittedEventFixtures:
    """The buffers ``tools/measure_recent_events_rendering.py`` measures.

    Kept under test so the numbers that tool prints cannot drift from a fixture
    nothing exercises — the failure mode that made the NPC-1318 baseline
    unreproducible.
    """

    def test_social_buffer_renders_every_bid_identifier(self):
        rendered = format_recent_events(create_social_events())

        assert "bid_a1b2c3d4" in rendered
        assert "npc_carol" in rendered
        assert "counter-offer: join with npc_bob, npc_dave" in rendered
        assert "item_apple_3" in rendered

    def test_conversation_buffer_points_to_the_typed_transcript_without_repeating_it(self):
        rendered = format_recent_events(create_conversation_events())

        assert "full transcript below" in rendered
        assert "Have you seen the smith today?" not in rendered

    def test_fixture_buffers_are_in_arrival_order(self):
        """Timestamps ascend, so the rendered prefix reads as a sequence."""
        for events in (create_social_events(), create_conversation_events()):
            timestamps = [event.timestamp for event in events]
            assert timestamps == sorted(timestamps)


class TestFormatConversationHistories:
    def test_empty_histories_render_a_sentinel(self):
        assert format_conversation_histories({}, "npc_alice") == "No active conversation."

    def test_complete_transcript_renders_roles_system_messages_and_declarations(self):
        messages = [
            ConversationMessage(
                speaker_id="npc_alice",
                speaker_name="Alice",
                message=f"Alice turn {index}",
                id=f"alice_{index}",
                timestamp=index,
            )
            for index in range(1, 14)
        ]
        messages.extend(
            [
                ConversationMessage(
                    speaker_id="npc_bob",
                    speaker_name="Bob",
                    message="I should go now.",
                    id="bob_14",
                    timestamp=14,
                    declarations=[{"kind": "farewell"}],
                ),
                ConversationMessage(
                    speaker_id="system",
                    speaker_name="System",
                    message="Message limit reached.",
                    id="system_15",
                    timestamp=15,
                    is_system=True,
                ),
            ]
        )

        rendered = format_conversation_histories({"conversation_1": messages}, "npc_alice")

        assert "Conversation conversation_1:" in rendered
        assert "[YOU] Alice: Alice turn 1" in rendered
        assert "[YOU] Alice: Alice turn 13" in rendered
        assert "Bob: I should go now. [farewell]" in rendered
        assert "System: Message limit reached. [system]" in rendered
        assert len(rendered.splitlines()) == 16
