"""Prompt-caching mechanics (NPC-1319). No network involved.

The contract under test: the static prefix is formatted once and byte-identical
across calls; a cache breakpoint is requested only for allowlisted models with
a long-enough prefix; and whichever branch is taken, the model sees the same
bytes. AsyncMock has no real string model_name, so caching is structurally off
under test doubles — load-bearing for every prompt-rendering assertion in the
node suites.
"""

from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage
from langchain_core.prompts import PromptTemplate

from mind import constants
from mind.apis.langchain_llm import supports_cache_control
from mind.cognitive_architecture.nodes.base import LLMNode
from mind.cognitive_architecture.nodes.reflection.node import ReflectionNode
from mind.cognitive_architecture.observations import Observation, StatusObservation
from mind.cognitive_architecture.state import PipelineState
from mind.cognitive_architecture.working_memory import WorkingMemory

VALID_RESPONSE = (
    '{"updated_working_memory": {"situation_assessment": "ok"}, '
    '"new_memories": [], '
    '"chosen_action": {"action": "wait", "parameters": {}}}'
)


def make_llm(model_name: str | None = None, content: str = VALID_RESPONSE) -> AsyncMock:
    mock = AsyncMock()
    if model_name is not None:
        mock.model_name = model_name
    mock.ainvoke.return_value = AIMessage(
        content=content,
        usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
    )
    return mock


def make_state(situation: str = "At the forge") -> PipelineState:
    return PipelineState(
        observation=Observation(
            entity_id="test_npc",
            current_simulation_time=100,
            status=StatusObservation(position=(5, 10), movement_locked=False),
        ),
        working_memory=WorkingMemory(situation_assessment=situation),
    )


class TestSupportsCacheControl:
    def test_allowlisted_models(self):
        assert supports_cache_control(constants.SONNET)
        assert supports_cache_control(constants.GEMINI_FLASH)
        assert supports_cache_control(constants.GEMINI_FLASH_LITE)

    def test_unknown_slug_defaults_off(self):
        """A provider that 400s on an unrecognised key must not take the
        pipeline down, so unknown slugs get no breakpoint."""
        assert not supports_cache_control("someone/some-new-model")


class TestBuildStaticPrefix:
    def test_formats_static_variables(self):
        prefix = LLMNode.build_static_prefix("Rules: {world_knowledge}!", world_knowledge="be kind")

        assert prefix == "Rules: be kind!"

    def test_raises_on_dynamic_variable_in_static_template(self):
        """A dynamic variable above the breakpoint would silently fragment the
        cache per call; construction must refuse it loudly."""
        with pytest.raises(ValueError, match="working_memory"):
            LLMNode.build_static_prefix(
                "Rules: {world_knowledge}\nState: {working_memory}",
                world_knowledge="be kind",
            )


class TestLLMNodeContentConstruction:
    def _node(self, llm, static_prefix, cache_control):
        node = LLMNode(
            llm=llm,
            prompt=PromptTemplate.from_template("dynamic: {input}"),
            static_prefix=static_prefix,
            cache_control=cache_control,
        )
        node.step_name = "test_step"
        return node

    def test_cache_control_requires_static_prefix(self):
        with pytest.raises(ValueError, match="static_prefix"):
            self._node(AsyncMock(), static_prefix=None, cache_control=True)

    @pytest.mark.asyncio
    async def test_no_static_prefix_sends_plain_string(self):
        llm = make_llm(constants.GEMINI_FLASH_LITE, content="ok")
        node = LLMNode(llm=llm, prompt=PromptTemplate.from_template("dynamic: {input}"))
        node.step_name = "test_step"

        await node.call_llm(make_state(), input="hello")

        assert llm.ainvoke.call_args[0][0][0].content == "dynamic: hello"

    @pytest.mark.asyncio
    async def test_breakpoint_enabled_sends_two_blocks_with_marker_on_first_only(self):
        prefix = "STATIC " * 1024  # comfortably above MIN_CACHEABLE_PREFIX_CHARS
        llm = make_llm(constants.GEMINI_FLASH_LITE, content="ok")
        node = self._node(llm, static_prefix=prefix, cache_control=True)

        await node.call_llm(make_state(), input="hello")

        content = llm.ainvoke.call_args[0][0][0].content
        assert isinstance(content, list) and len(content) == 2
        assert content[0]["cache_control"] == {"type": "ephemeral"}
        assert "cache_control" not in content[1]
        # Same bytes as the single-block branch would send
        assert content[0]["text"] + content[1]["text"] == prefix + "dynamic: hello"

    @pytest.mark.asyncio
    async def test_cache_control_false_sends_one_concatenated_string(self):
        prefix = "STATIC " * 1024
        llm = make_llm(constants.GEMINI_FLASH_LITE, content="ok")
        node = self._node(llm, static_prefix=prefix, cache_control=False)

        await node.call_llm(make_state(), input="hello")

        assert llm.ainvoke.call_args[0][0][0].content == prefix + "dynamic: hello"

    @pytest.mark.asyncio
    async def test_async_mock_without_model_name_disables_breakpoint(self):
        """AsyncMock's auto-attribute is not a string, so caching is off under
        test doubles and prompt assertions see plain strings."""
        prefix = "STATIC " * 1024
        llm = make_llm(model_name=None, content="ok")
        node = self._node(llm, static_prefix=prefix, cache_control=True)

        assert node._cache_breakpoint_enabled() is False

        await node.call_llm(make_state(), input="hello")

        assert llm.ainvoke.call_args[0][0][0].content == prefix + "dynamic: hello"

    @pytest.mark.asyncio
    async def test_unknown_model_disables_breakpoint(self):
        prefix = "STATIC " * 1024
        llm = make_llm("someone/some-new-model", content="ok")
        node = self._node(llm, static_prefix=prefix, cache_control=True)

        assert node._cache_breakpoint_enabled() is False

    @pytest.mark.asyncio
    async def test_short_prefix_disables_breakpoint(self):
        """Below every provider minimum a breakpoint buys nothing; send the
        same bytes as one plain block instead."""
        prefix = "short prefix "
        llm = make_llm(constants.GEMINI_FLASH_LITE, content="ok")
        node = self._node(llm, static_prefix=prefix, cache_control=True)

        assert node._cache_breakpoint_enabled() is False

        await node.call_llm(make_state(), input="hello")

        assert llm.ainvoke.call_args[0][0][0].content == prefix + "dynamic: hello"


class TestReflectionNodeCaching:
    @pytest.mark.asyncio
    async def test_static_prefix_is_byte_identical_across_calls(self):
        llm = make_llm()
        node = ReflectionNode(llm)
        prefix = node.static_prefix

        await node.process(make_state("First situation"))
        first = llm.ainvoke.call_args[0][0][0].content
        await node.process(make_state("Second situation, quite different"))
        second = llm.ainvoke.call_args[0][0][0].content

        assert first[: len(prefix)] == prefix
        assert second[: len(prefix)] == prefix
        assert first != second, "the dynamic suffix must actually vary"

    def test_reflection_prefix_clears_the_length_gate(self):
        node = ReflectionNode(make_llm())

        assert node.cache_control is True
        assert len(node.static_prefix) >= constants.MIN_CACHEABLE_PREFIX_CHARS

    @pytest.mark.asyncio
    async def test_reflection_requests_breakpoint_for_allowlisted_model(self):
        llm = make_llm(constants.GEMINI_FLASH_LITE)
        node = ReflectionNode(llm)

        await node.process(make_state())

        content = llm.ainvoke.call_args[0][0][0].content
        assert isinstance(content, list) and len(content) == 2
        assert content[0]["text"] == node.static_prefix
        assert content[0]["cache_control"] == {"type": "ephemeral"}
        assert "cache_control" not in content[1]

    @pytest.mark.asyncio
    async def test_blocked_and_plain_branches_send_identical_bytes(self):
        cached_llm = make_llm(constants.GEMINI_FLASH_LITE)
        plain_llm = make_llm()
        state_args = "Identical situation"

        await ReflectionNode(cached_llm).process(make_state(state_args))
        blocks = cached_llm.ainvoke.call_args[0][0][0].content
        await ReflectionNode(plain_llm).process(make_state(state_args))
        plain = plain_llm.ainvoke.call_args[0][0][0].content

        assert blocks[0]["text"] + blocks[1]["text"] == plain

    @pytest.mark.asyncio
    async def test_retry_preserves_prefix_and_breakpoint(self):
        llm = make_llm(constants.GEMINI_FLASH_LITE)
        llm.ainvoke.side_effect = [
            AIMessage(
                content="not valid json",
                usage_metadata={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
            ),
            AIMessage(
                content=VALID_RESPONSE,
                usage_metadata={"input_tokens": 12, "output_tokens": 30, "total_tokens": 42},
            ),
        ]
        node = ReflectionNode(llm)

        await node.process(make_state())

        assert llm.ainvoke.call_count == 2
        retry_messages = llm.ainvoke.call_args[0][0]
        assert len(retry_messages) == 3, "original + AI response + correction request"
        first_content = retry_messages[0].content
        assert isinstance(first_content, list)
        assert first_content[0]["cache_control"] == {"type": "ephemeral"}, (
            "a retried decision must still read the cache"
        )
