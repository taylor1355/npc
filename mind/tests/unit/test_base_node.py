"""Unit tests for base node classes"""

import json
import logging
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, ValidationError

from mind.cognitive_architecture.nodes.base import LLMNode, Node, entity_tag
from mind.cognitive_architecture.observations import Observation, StatusObservation
from mind.cognitive_architecture.state import PipelineState


class TestNodeTimingDecorator:
    """Test Node base class timing functionality"""

    @pytest.mark.asyncio
    async def test_node_tracks_timing_automatically(self):
        """Node subclasses should automatically track timing"""

        class TestNode(Node):
            step_name = "test_step"

            async def process(self, state: PipelineState) -> PipelineState:
                return state

        node = TestNode()
        state = PipelineState(
            observation=Observation(
                entity_id="test",
                current_simulation_time=0,
                status=StatusObservation(position=(0, 0), movement_locked=False),
            )
        )

        result = await node.process(state)

        assert "test_step" in result.time_ms
        assert result.time_ms["test_step"] >= 0


class TestLLMNodeInitialization:
    """Test LLMNode initialization and configuration"""

    def test_init_with_structured_output(self):
        """Should initialize with Pydantic output model"""

        class TestOutput(BaseModel):
            value: str

        mock_llm = AsyncMock()
        prompt = PromptTemplate.from_template("Test {input}")

        node = LLMNode(llm=mock_llm, prompt=prompt, output_model=TestOutput)

        assert node.llm == mock_llm
        assert node.prompt == prompt
        assert node.output_model == TestOutput
        assert node.parser is not None
        assert node.max_retries == 0

    def test_init_with_raw_string_output(self):
        """Should initialize for raw string output"""
        mock_llm = AsyncMock()
        prompt = PromptTemplate.from_template("Test {input}")

        node = LLMNode(llm=mock_llm, prompt=prompt, output_model=None)

        assert node.output_model is None
        assert node.parser is None

    def test_init_with_max_retries(self):
        """Should accept max_retries parameter"""

        class TestOutput(BaseModel):
            value: str

        mock_llm = AsyncMock()
        prompt = PromptTemplate.from_template("Test {input}")

        node = LLMNode(llm=mock_llm, prompt=prompt, output_model=TestOutput, max_retries=3)

        assert node.max_retries == 3

    def test_init_rejects_retries_without_output_model(self):
        """Should raise ValueError if max_retries > 0 without output_model"""
        mock_llm = AsyncMock()
        prompt = PromptTemplate.from_template("Test {input}")

        with pytest.raises(ValueError, match="max_retries > 0 requires output_model"):
            LLMNode(llm=mock_llm, prompt=prompt, output_model=None, max_retries=2)


class TestLLMNodeRawStringOutput:
    """Test LLMNode with raw string output"""

    @pytest.mark.asyncio
    async def test_call_llm_returns_raw_string(self):
        """Should return raw string when output_model is None"""
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = AIMessage(
            content="This is a raw response",
            usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        )

        prompt = PromptTemplate.from_template("Test {input}")
        node = LLMNode(llm=mock_llm, prompt=prompt, output_model=None)
        node.step_name = "test_step"

        state = PipelineState(
            observation=Observation(
                entity_id="test",
                current_simulation_time=0,
                status=StatusObservation(position=(0, 0), movement_locked=False),
            )
        )

        result = await node.call_llm(state, input="hello")

        assert result == "This is a raw response"
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_raw_string_tracks_tokens(self):
        """Should track tokens for raw string output"""
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = AIMessage(
            content="Response",
            usage_metadata={"input_tokens": 20, "output_tokens": 10, "total_tokens": 30},
        )

        prompt = PromptTemplate.from_template("{input}")
        node = LLMNode(llm=mock_llm, prompt=prompt, output_model=None)
        node.step_name = "test_step"

        state = PipelineState(
            observation=Observation(
                entity_id="test",
                current_simulation_time=0,
                status=StatusObservation(position=(0, 0), movement_locked=False),
            )
        )

        await node.call_llm(state, input="test")

        assert state.tokens_used["test_step"].total_tokens == 30


class TestLLMNodeStructuredOutput:
    """Test LLMNode with Pydantic structured output"""

    @pytest.mark.asyncio
    async def test_call_llm_returns_parsed_model(self):
        """Should parse and return Pydantic model"""

        class TestOutput(BaseModel):
            message: str
            count: int

        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = AIMessage(
            content='{"message": "hello", "count": 42}',
            usage_metadata={"input_tokens": 10, "output_tokens": 8, "total_tokens": 18},
        )

        prompt = PromptTemplate.from_template("{input}")
        node = LLMNode(llm=mock_llm, prompt=prompt, output_model=TestOutput)
        node.step_name = "test_step"

        state = PipelineState(
            observation=Observation(
                entity_id="test",
                current_simulation_time=0,
                status=StatusObservation(position=(0, 0), movement_locked=False),
            )
        )

        result = await node.call_llm(state, input="test")

        assert isinstance(result, TestOutput)
        assert result.message == "hello"
        assert result.count == 42

    @pytest.mark.asyncio
    async def test_structured_output_tracks_tokens(self):
        """Should track tokens for structured output"""

        class TestOutput(BaseModel):
            value: str

        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = AIMessage(
            content='{"value": "test"}',
            usage_metadata={"input_tokens": 15, "output_tokens": 5, "total_tokens": 20},
        )

        prompt = PromptTemplate.from_template("{input}")
        node = LLMNode(llm=mock_llm, prompt=prompt, output_model=TestOutput)
        node.step_name = "test_step"

        state = PipelineState(
            observation=Observation(
                entity_id="test",
                current_simulation_time=0,
                status=StatusObservation(position=(0, 0), movement_locked=False),
            )
        )

        await node.call_llm(state, input="test")

        assert state.tokens_used["test_step"].total_tokens == 20


class TestLLMNodeRetryLogic:
    """Test LLMNode retry logic with validation"""

    @pytest.mark.asyncio
    async def test_retry_on_json_decode_error(self):
        """Should retry when LLM returns invalid JSON"""

        class TestOutput(BaseModel):
            value: str

        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = [
            AIMessage(
                content="not valid json",
                usage_metadata={"input_tokens": 10, "output_tokens": 3, "total_tokens": 13},
            ),
            AIMessage(
                content='{"value": "success"}',
                usage_metadata={"input_tokens": 12, "output_tokens": 4, "total_tokens": 16},
            ),
        ]

        prompt = PromptTemplate.from_template("{input}")
        node = LLMNode(llm=mock_llm, prompt=prompt, output_model=TestOutput, max_retries=1)
        node.step_name = "test_step"

        state = PipelineState(
            observation=Observation(
                entity_id="test",
                current_simulation_time=0,
                status=StatusObservation(position=(0, 0), movement_locked=False),
            )
        )

        result = await node.call_llm(state, input="test")

        assert result.value == "success"
        assert mock_llm.ainvoke.call_count == 2

    @pytest.mark.asyncio
    async def test_retry_on_validation_error(self):
        """Should retry when Pydantic validation fails"""

        class TestOutput(BaseModel):
            required_field: str

        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = [
            AIMessage(
                content='{"wrong_field": "oops"}',
                usage_metadata={"input_tokens": 10, "output_tokens": 4, "total_tokens": 14},
            ),
            AIMessage(
                content='{"required_field": "correct"}',
                usage_metadata={"input_tokens": 15, "output_tokens": 5, "total_tokens": 20},
            ),
        ]

        prompt = PromptTemplate.from_template("{input}")
        node = LLMNode(llm=mock_llm, prompt=prompt, output_model=TestOutput, max_retries=1)
        node.step_name = "test_step"

        state = PipelineState(
            observation=Observation(
                entity_id="test",
                current_simulation_time=0,
                status=StatusObservation(position=(0, 0), movement_locked=False),
            )
        )

        result = await node.call_llm(state, input="test")

        assert result.required_field == "correct"
        assert mock_llm.ainvoke.call_count == 2

    @pytest.mark.asyncio
    async def test_retry_exhaustion_raises_error(self):
        """Should raise error after all retries exhausted"""

        class TestOutput(BaseModel):
            value: str

        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = AIMessage(
            content="invalid json every time",
            usage_metadata={"input_tokens": 10, "output_tokens": 4, "total_tokens": 14},
        )

        prompt = PromptTemplate.from_template("{input}")
        node = LLMNode(llm=mock_llm, prompt=prompt, output_model=TestOutput, max_retries=2)

        state = PipelineState(
            observation=Observation(
                entity_id="test",
                current_simulation_time=0,
                status=StatusObservation(position=(0, 0), movement_locked=False),
            )
        )

        with pytest.raises((json.JSONDecodeError, ValidationError)):
            await node.call_llm(state, input="test")

        assert mock_llm.ainvoke.call_count == 3  # Initial + 2 retries

    @pytest.mark.asyncio
    async def test_retry_tracks_all_tokens(self):
        """Should track tokens from all retry attempts"""

        class TestOutput(BaseModel):
            value: str

        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = [
            AIMessage(
                content="bad",
                usage_metadata={"input_tokens": 10, "output_tokens": 1, "total_tokens": 11},
            ),
            AIMessage(
                content="also bad",
                usage_metadata={"input_tokens": 12, "output_tokens": 2, "total_tokens": 14},
            ),
            AIMessage(
                content='{"value": "good"}',
                usage_metadata={"input_tokens": 14, "output_tokens": 4, "total_tokens": 18},
            ),
        ]

        prompt = PromptTemplate.from_template("{input}")
        node = LLMNode(llm=mock_llm, prompt=prompt, output_model=TestOutput, max_retries=2)
        node.step_name = "test_step"

        state = PipelineState(
            observation=Observation(
                entity_id="test",
                current_simulation_time=0,
                status=StatusObservation(position=(0, 0), movement_locked=False),
            )
        )

        await node.call_llm(state, input="test")

        # Should sum all attempts: 11 + 14 + 18 = 43
        assert state.tokens_used["test_step"].total_tokens == 43

    @pytest.mark.asyncio
    async def test_retry_tracks_tokens_even_on_failure(self):
        """Should track tokens even when all retries fail"""

        class TestOutput(BaseModel):
            value: str

        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = [
            AIMessage(
                content="bad1",
                usage_metadata={"input_tokens": 10, "output_tokens": 1, "total_tokens": 11},
            ),
            AIMessage(
                content="bad2",
                usage_metadata={"input_tokens": 11, "output_tokens": 1, "total_tokens": 12},
            ),
        ]

        prompt = PromptTemplate.from_template("{input}")
        node = LLMNode(llm=mock_llm, prompt=prompt, output_model=TestOutput, max_retries=1)
        node.step_name = "test_step"

        state = PipelineState(
            observation=Observation(
                entity_id="test",
                current_simulation_time=0,
                status=StatusObservation(position=(0, 0), movement_locked=False),
            )
        )

        with pytest.raises((json.JSONDecodeError, ValidationError)):
            await node.call_llm(state, input="test")

        # Should still track tokens: 11 + 12 = 23
        assert state.tokens_used["test_step"].total_tokens == 23


class TestUsageExtraction:
    """_extract_usage edge cases.

    The through-line: a zero this code reports must always be a zero somebody
    measured. "The provider said nothing" is a different answer and gets a
    different representation (None here, unreported_calls downstream).
    """

    def _node(self):
        return LLMNode(llm=AsyncMock(), prompt=PromptTemplate.from_template("{input}"))

    def test_extract_usage_preserves_prompt_and_completion_split(self):
        """The split arrives in usage_metadata and must survive extraction"""
        response = AIMessage(
            content="test",
            usage_metadata={"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
        )

        usage = self._node()._extract_usage(response)

        assert usage.prompt_tokens == 5
        assert usage.completion_tokens == 3
        assert usage.total_tokens == 8
        assert usage.model_calls == 1
        assert usage.unreported_calls == 0

    def test_extract_usage_returns_none_when_provider_reported_nothing(self):
        """None, not a zeroed record - the two mean different things"""
        assert self._node()._extract_usage(AIMessage(content="test")) is None

    def test_extract_usage_reports_a_measured_zero_as_a_record(self):
        """A provider that reports zero is measured data, not missing data"""
        response = AIMessage(
            content="test",
            usage_metadata={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        )

        usage = self._node()._extract_usage(response)

        assert usage is not None, "a reported zero must not be collapsed into None"
        assert usage.total_tokens == 0
        assert usage.model_calls == 1

    def test_cache_read_tokens_are_captured(self):
        """cache_read rides input_token_details and is free to read through"""
        response = AIMessage(
            content="test",
            usage_metadata={
                "input_tokens": 100,
                "output_tokens": 10,
                "total_tokens": 110,
                "input_token_details": {"cache_read": 80},
            },
        )

        usage = self._node()._extract_usage(response)

        assert usage.cached_prompt_tokens == 80
        assert usage.cache_reporting is True

    def test_absent_cache_details_are_not_reported_as_zero_cache_hits(self):
        """input_token_details is NotRequired; absent != zero hits"""
        response = AIMessage(
            content="test",
            usage_metadata={"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
        )

        usage = self._node()._extract_usage(response)

        assert usage.cache_reporting is False, (
            "no cache accounting reported must stay distinguishable from zero hits"
        )


class TestUsageRecordingSentinels:
    """call_llm must never let an absent key or a zero impersonate the other."""

    def _state(self):
        return PipelineState(
            observation=Observation(
                entity_id="test",
                current_simulation_time=0,
                status=StatusObservation(position=(0, 0), movement_locked=False),
            )
        )

    @pytest.mark.asyncio
    async def test_step_that_cost_zero_tokens_is_recorded_not_omitted(self):
        """'ran and was free' must not collapse into 'never ran'"""
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = AIMessage(
            content="Response",
            usage_metadata={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        )

        node = LLMNode(
            llm=mock_llm, prompt=PromptTemplate.from_template("{input}"), output_model=None
        )
        node.step_name = "test_step"
        state = self._state()

        await node.call_llm(state, input="test")

        assert "test_step" in state.tokens_used, (
            "a step that legitimately cost zero must still get a key"
        )
        assert state.tokens_used["test_step"].total_tokens == 0
        assert state.tokens_used["test_step"].model_calls == 1

    @pytest.mark.asyncio
    async def test_missing_usage_metadata_records_unreported_not_zero(self):
        """A silent provider is recorded as an unreported call, not a free one"""
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = AIMessage(content="Response")

        node = LLMNode(
            llm=mock_llm, prompt=PromptTemplate.from_template("{input}"), output_model=None
        )
        node.step_name = "test_step"
        state = self._state()

        await node.call_llm(state, input="test")

        usage = state.tokens_used["test_step"]
        assert usage.model_calls == 1
        assert usage.unreported_calls == 1
        assert usage.is_fully_unreported()

    @pytest.mark.asyncio
    async def test_every_provider_call_including_retries_is_counted(self):
        """Three round-trips for one decision must be countable as three"""

        class TestOutput(BaseModel):
            value: str

        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = [
            AIMessage(
                content="bad",
                usage_metadata={"input_tokens": 10, "output_tokens": 1, "total_tokens": 11},
            ),
            AIMessage(
                content="also bad",
                usage_metadata={"input_tokens": 12, "output_tokens": 2, "total_tokens": 14},
            ),
            AIMessage(
                content='{"value": "good"}',
                usage_metadata={"input_tokens": 14, "output_tokens": 4, "total_tokens": 18},
            ),
        ]

        node = LLMNode(
            llm=mock_llm,
            prompt=PromptTemplate.from_template("{input}"),
            output_model=TestOutput,
            max_retries=2,
        )
        node.step_name = "test_step"
        state = self._state()

        await node.call_llm(state, input="test")

        usage = state.tokens_used["test_step"]
        assert usage.model_calls == 3, "retries are real spend and must stay countable"
        assert usage.total_tokens == 43
        assert usage.prompt_tokens == 36
        assert usage.completion_tokens == 7

    @pytest.mark.asyncio
    async def test_retry_exhaustion_still_records_the_calls_it_made(self):
        """The raise escapes past any handler that can reach PipelineState"""

        class TestOutput(BaseModel):
            value: str

        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = [
            AIMessage(
                content="bad1",
                usage_metadata={"input_tokens": 10, "output_tokens": 1, "total_tokens": 11},
            ),
            AIMessage(
                content="bad2",
                usage_metadata={"input_tokens": 11, "output_tokens": 1, "total_tokens": 12},
            ),
        ]

        node = LLMNode(
            llm=mock_llm,
            prompt=PromptTemplate.from_template("{input}"),
            output_model=TestOutput,
            max_retries=1,
        )
        node.step_name = "test_step"
        state = self._state()

        with pytest.raises((json.JSONDecodeError, ValidationError)):
            await node.call_llm(state, input="test")

        assert state.tokens_used["test_step"].model_calls == 2
        assert state.tokens_used["test_step"].total_tokens == 23


class TestEntityTagAttribution:
    """NPC-789: log records must carry the entity id for Events-tab attribution"""

    def _make_state(self, entity_id="entity_attribution_test"):
        return PipelineState(
            observation=Observation(
                entity_id=entity_id,
                current_simulation_time=0,
                status=StatusObservation(position=(0, 0), movement_locked=False),
            )
        )

    def test_entity_tag_brackets_entity_id(self):
        """Should wrap the entity id in brackets, matching per-entity log convention"""
        state = self._make_state("npc_alice")
        assert entity_tag(state) == "[npc_alice]"

    def test_entity_tag_falls_back_when_entity_id_empty(self):
        """Should produce a recognizable fallback instead of an empty tag"""
        state = self._make_state("")
        assert entity_tag(state) == "[unknown]"

    @pytest.mark.asyncio
    async def test_raw_string_log_records_carry_entity_id(self, caplog):
        """call_llm raw-string path must emit only attributed records"""
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = AIMessage(
            content="Response",
            usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        )
        node = LLMNode(
            llm=mock_llm,
            prompt=PromptTemplate.from_template("{input}"),
            output_model=None,
        )
        node.step_name = "raw_step"
        state = self._make_state()

        with caplog.at_level(logging.DEBUG, logger="mind"):
            await node.call_llm(state, input="hi")

        assert caplog.records, "call_llm should emit log records"
        for record in caplog.records:
            assert "entity_attribution_test" in record.getMessage(), (
                f"Unattributed log record: {record.getMessage()!r}"
            )

    @pytest.mark.asyncio
    async def test_retry_log_records_carry_entity_id(self, caplog):
        """call_llm retry path must emit only attributed records"""

        class TestOutput(BaseModel):
            value: str

        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = [
            AIMessage(
                content="not valid json",
                usage_metadata={"input_tokens": 10, "output_tokens": 3, "total_tokens": 13},
            ),
            AIMessage(
                content='{"value": "success"}',
                usage_metadata={"input_tokens": 12, "output_tokens": 4, "total_tokens": 16},
            ),
        ]
        node = LLMNode(
            llm=mock_llm,
            prompt=PromptTemplate.from_template("{input}"),
            output_model=TestOutput,
            max_retries=1,
        )
        node.step_name = "retry_step"
        state = self._make_state()

        with caplog.at_level(logging.DEBUG, logger="mind"):
            result = await node.call_llm(state, input="hi")

        assert result.value == "success"
        assert caplog.records, "call_llm should emit log records"
        for record in caplog.records:
            assert "entity_attribution_test" in record.getMessage(), (
                f"Unattributed log record: {record.getMessage()!r}"
            )
