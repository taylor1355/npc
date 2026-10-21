"""Message handles: how a mind names a transcript row without ever seeing its id (NPC-2021).

The prompt prefixes each transcript row with a per-cycle handle (``[m3]``) and
never prints the message id. The map is a pure function of the conversation
histories, so the prompt that shows a handle, every validation retry and the wire
translation agree for one state snapshot.
"""

import re

from mind.cognitive_architecture.nodes.formatting import format_conversation_histories
from mind.cognitive_architecture.observations import ConversationMessage
from mind.cognitive_architecture.references import (
    HandleTable,
    message_handles,
    normalize_handle,
)

UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def _message(index: int, speaker: str = "npc_wren", text: str = "") -> ConversationMessage:
    return ConversationMessage(
        id=f"msg_{index:08x}-0000-4000-8000-{index:012x}",
        speaker_id=speaker,
        speaker_name=speaker.removeprefix("npc_").title(),
        message=text or f"line {index}",
        timestamp=100 + index,
    )


def _histories() -> dict[str, list[ConversationMessage]]:
    return {
        "conv_b": [_message(3), _message(4)],
        "conv_a": [_message(1), _message(2, "npc_alice")],
    }


class TestMinting:
    def test_handles_number_rows_in_transcript_order(self):
        """Interaction ids sorted, messages in stored order, one global ordinal."""
        histories = _histories()

        handles = message_handles(histories)

        assert list(handles) == ["m1", "m2", "m3", "m4"]
        assert [m.message for m in handles.values()] == ["line 1", "line 2", "line 3", "line 4"]

    def test_the_map_is_a_pure_function_of_the_histories(self):
        histories = _histories()
        assert message_handles(histories) == message_handles(histories)

    def test_a_growing_history_keeps_earlier_handles(self):
        histories = _histories()
        before = {h: m.id for h, m in message_handles(histories).items()}

        histories["conv_b"].append(_message(5))
        after = {h: m.id for h, m in message_handles(histories).items()}

        assert all(after[h] == mid for h, mid in before.items())
        assert len(after) == len(before) + 1

    def test_no_history_has_no_handles(self):
        assert message_handles({}) == {}

    def test_a_handle_never_collides_with_a_place_handle_or_an_option_id(self):
        assert all(h.startswith("m") for h in message_handles(_histories()))


class TestRendering:
    def test_every_row_carries_its_handle_and_no_row_carries_an_id(self):
        histories = _histories()

        text = format_conversation_histories(histories, "npc_alice", annotate_handles=True)

        for handle, message in message_handles(histories).items():
            assert f"[{handle}] [t={message.timestamp}]" in text
            assert message.id not in text
        assert UUID_RE.search(text) is None

    def test_the_handle_beside_a_row_resolves_to_that_row(self):
        """What the model READS beside a row is what the validator resolves."""
        histories = _histories()
        handles = message_handles(histories)
        text = format_conversation_histories(histories, "npc_alice", annotate_handles=True)

        for line in text.splitlines():
            match = re.match(r"\[(m\d+)\] \[t=\d+\] \S+: (.*)$", line)
            if match:
                assert handles[match.group(1)].message == match.group(2)

    def test_a_message_object_shared_by_two_conversations_gets_a_handle_per_row(self):
        """Rows are matched by position, not object identity (review of #68)."""
        shared = next(iter(_histories().values()))[0]
        histories = {"c1": [shared], "c2": [shared]}

        text = format_conversation_histories(histories, "npc_alice", annotate_handles=True)

        assert "[m1]" in text and "[m2]" in text
        assert list(message_handles(histories)) == ["m1", "m2"]

    def test_without_handles_the_rows_are_unannotated(self):
        text = format_conversation_histories(_histories(), "npc_alice")
        assert "[m1]" not in text


class TestResolution:
    def test_spellings_resolve_to_the_message_id(self):
        histories = _histories()
        table = HandleTable(None, histories)
        wanted = message_handles(histories)["m2"].id

        for spelling in ("m2", "[m2]", "M2", " [M2] "):
            assert table.resolve("message", spelling) == wanted

    def test_a_raw_id_is_refused_even_when_it_belongs_to_a_message(self):
        histories = _histories()
        table = HandleTable(None, histories)

        assert table.resolve("message", message_handles(histories)["m1"].id) is None

    def test_an_out_of_range_handle_is_refused(self):
        assert HandleTable(None, _histories()).resolve("message", "m9") is None

    def test_remembering_a_message_names_the_speaker_and_never_the_handle(self):
        table = HandleTable(None, _histories())
        remembered = table.remembered("message", "m2")
        assert "Alice" in remembered and "line 2" in remembered
        assert "m2" not in remembered

    def test_a_long_message_is_truncated_in_memory(self):
        histories = {"c": [_message(1, text="x" * 200)]}
        assert len(HandleTable(None, histories).remembered("message", "m1")) < 100

    def test_normalize_handle(self):
        assert normalize_handle(" [P2] ") == "p2"
