"""Wits terseness instruction (conscious tier, NPC-1090).

Prompt CONTENT on the conversation payload, not routing: the simulation sends a
cognitive-energy number with each conversation observation, and this side owns
the band edges and wording. The instruction lives in the reflection prompt's
dynamic suffix only, and a fresh or unreported mind renders byte-identical to
a prompt that never knew about wits.
"""

import math
from pathlib import Path

import pytest

from mind.cognitive_architecture.nodes.formatting import (
    WITS_FRESH_THRESHOLD,
    format_terseness_instruction,
)
from mind.cognitive_architecture.nodes.reflection.node import (
    CACHE_BREAKPOINT_MARKER,
    ReflectionNode,
)
from mind.cognitive_architecture.observations import ConversationMessage, Observation
from mind.cognitive_architecture.observations.models import ConversationObservation
from mind.cognitive_architecture.state import PipelineState
from mind.cognitive_architecture.working_memory import WorkingMemory
from mind.interfaces.mcp.mind import Mind
from tests.fixtures import create_blacksmith_observation
from tests.unit.test_reflection_node import VALID_RESPONSE, make_mock_llm, rendered_prompt

TIRED = "You're getting mentally tired. Keep responses to 1-2 sentences."
EXHAUSTED = "You're mentally exhausted. Respond in short phrases only."
DEPLETED = "You can barely think. One-word or very brief responses only."
ALL_BANDS = (TIRED, EXHAUSTED, DEPLETED)

PROMPT_PATH = (
    Path(__file__).parents[2] / "src/mind/cognitive_architecture/nodes/reflection/prompt.md"
)


def _message() -> ConversationMessage:
    return ConversationMessage(
        speaker_id="npc_alice",
        speaker_name="Alice",
        message="Hello!",
        timestamp=10,
        id="message_a",
    )


def _conversation(wits: float | None = None) -> ConversationObservation:
    kwargs = {} if wits is None else {"wits": wits}
    return ConversationObservation(
        interaction_id="conv_1",
        interaction_name="conversation",
        participants=["npc_alice", "npc_bob"],
        initiator_id="npc_alice",
        conversation_history=[_message()],
        **kwargs,
    )


def _mind() -> Mind:
    return Mind(
        mind_id="mind_test",
        entity_id="test_npc",
        traits=[],
        pipeline=None,
        memory_store=None,
        working_memory=WorkingMemory(),
        llm_model="test/model",
    )


class TestConversationObservationWits:
    PAYLOAD = {
        "interaction_id": "conv_1",
        "interaction_name": "conversation",
        "participants": ["npc_alice", "npc_bob"],
        "initiator_id": "npc_alice",
        "conversation_history": [],
    }

    def test_payload_carrying_wits_parses(self):
        obs = ConversationObservation.model_validate({**self.PAYLOAD, "wits": 42.5})
        assert obs.wits == 42.5

    def test_payload_omitting_wits_parses_as_unknown(self):
        obs = ConversationObservation.model_validate(self.PAYLOAD)
        assert obs.wits is None


class TestMindCarriesWits:
    def test_latest_wits_reaches_pipeline_state(self):
        mind = _mind()
        mind.update_conversations([_conversation(wits=70.0)])
        mind.update_conversations([_conversation(wits=35.0)])

        state = mind.build_pipeline_state(create_blacksmith_observation(simulation_time=10))

        assert state.wits == 35.0

    def test_omitted_wits_overwrites_a_stale_strained_reading(self):
        mind = _mind()
        mind.update_conversations([_conversation(wits=10.0)])
        mind.update_conversations([_conversation()])

        state = mind.build_pipeline_state(create_blacksmith_observation(simulation_time=10))

        assert state.wits is None


class TestFormatTerseness:
    def test_threshold_is_pinned_at_sixty(self):
        # The simulation mirrors this number (its brevity threshold).
        assert WITS_FRESH_THRESHOLD == 60.0

    @pytest.mark.parametrize("wits", [None, 60.0, 75, 100, 250, math.nan])
    def test_fresh_or_unknown_is_empty(self, wits):
        assert format_terseness_instruction(wits) == ""

    @pytest.mark.parametrize(
        ("wits", "expected"),
        [
            (59.9, TIRED),
            (40, TIRED),
            (39.9, EXHAUSTED),
            (20, EXHAUSTED),
            (19.9, DEPLETED),
            (0, DEPLETED),
            (-5, DEPLETED),
        ],
    )
    def test_bands_and_clamping(self, wits, expected):
        text = format_terseness_instruction(wits)
        assert [band for band in ALL_BANDS if band in text] == [expected]


@pytest.mark.asyncio
class TestReflectionRendersTerseness:
    def _state(self, wits: float | None, *, conversation: bool = True) -> PipelineState:
        kwargs = {} if wits is None else {"wits": wits}
        return PipelineState(
            observation=Observation(entity_id="test_npc", current_simulation_time=100),
            working_memory=WorkingMemory(situation_assessment="x"),
            conversation_histories={"conv_1": [_message()]} if conversation else {},
            **kwargs,
        )

    async def _prompt(self, wits: float | None, *, conversation: bool = True) -> str:
        llm = make_mock_llm(VALID_RESPONSE)
        await ReflectionNode(llm).process(self._state(wits, conversation=conversation))
        return rendered_prompt(llm)

    @pytest.mark.parametrize(
        ("wits", "band"), [(59, TIRED), (45, TIRED), (30, EXHAUSTED), (10, DEPLETED)]
    )
    async def test_strained_wits_adds_the_band_text_only(self, wits, band):
        prompt = await self._prompt(wits)
        assert [b for b in ALL_BANDS if b in prompt] == [band]

    @pytest.mark.parametrize("wits", [75, 60, 100])
    async def test_fresh_wits_is_byte_identical_to_no_wits(self, wits):
        baseline = await self._prompt(None)
        prompt = await self._prompt(wits)
        assert not any(b in prompt for b in ALL_BANDS)
        assert prompt == baseline

    async def test_instruction_sits_in_the_dynamic_suffix_only(self):
        llm = make_mock_llm(VALID_RESPONSE)
        node = ReflectionNode(llm)
        await node.process(self._state(10))
        assert DEPLETED not in node.static_prefix
        assert DEPLETED in rendered_prompt(llm)
        static_text, _ = PROMPT_PATH.read_text().split(CACHE_BREAKPOINT_MARKER)
        assert "terseness_instruction" not in static_text

    async def test_no_active_conversation_means_no_instruction(self):
        prompt = await self._prompt(10, conversation=False)
        assert not any(b in prompt for b in ALL_BANDS)
        assert prompt == await self._prompt(None, conversation=False)
