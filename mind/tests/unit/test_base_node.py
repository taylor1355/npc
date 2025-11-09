"""Unit tests for base node classes"""

from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, ValidationError

from mind.cognitive_architecture.nodes.base import LLMNode, Node
from mind.cognitive_architecture.state import PipelineState
from mind.cognitive_architecture.observations import Observation, StatusObservation


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
                status=StatusObservation(position=(0, 0), movement_locked=False)
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
            usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
        )

        prompt = PromptTemplate.from_template("Test {input}")
        node = LLMNode(llm=mock_llm, prompt=prompt, output_model=None)
        node.step_name = "test_step"

        state = PipelineState(
            observation=Observation(
                entity_id="test",
                current_simulation_time=0,
                status=StatusObservation(position=(0, 0), movement_locked=False)
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
            usage_metadata={"input_tokens": 20, "output_tokens": 10, "total_tokens": 30}
        )

        prompt = PromptTemplate.from_template("{input}")
        node = LLMNode(llm=mock_llm, prompt=prompt, output_model=None)
        node.step_name = "test_step"

        state = PipelineState(
            observation=Observation(
                entity_id="test",
                current_simulation_time=0,
                status=StatusObservation(position=(0, 0), movement_locked=False)
            )
        )

        await node.call_llm(state, input="test")

        assert state.tokens_used["test_step"] == 30


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
            usage_metadata={"input_tokens": 10, "output_tokens": 8, "total_tokens": 18}
        )

        prompt = PromptTemplate.from_template("{input}")
        node = LLMNode(llm=mock_llm, prompt=prompt, output_model=TestOutput)
        node.step_name = "test_step"

        state = PipelineState(
            observation=Observation(
                entity_id="test",
                current_simulation_time=0,
                status=StatusObservation(position=(0, 0), movement_locked=False)
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
            usage_metadata={"input_tokens": 15, "output_tokens": 5, "total_tokens": 20}
        )

        prompt = PromptTemplate.from_template("{input}")
        node = LLMNode(llm=mock_llm, prompt=prompt, output_model=TestOutput)
        node.step_name = "test_step"

        state = PipelineState(
            observation=Observation(
                entity_id="test",
                current_simulation_time=0,
                status=StatusObservation(position=(0, 0), movement_locked=False)
            )
        )

        await node.call_llm(state, input="test")

        assert state.tokens_used["test_step"] == 20


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
                usage_metadata={"input_tokens": 10, "output_tokens": 3, "total_tokens": 13}
            ),
            AIMessage(
                content='{"value": "success"}',
                usage_metadata={"input_tokens": 12, "output_tokens": 4, "total_tokens": 16}
            ),
        ]

        prompt = PromptTemplate.from_template("{input}")
        node = LLMNode(llm=mock_llm, prompt=prompt, output_model=TestOutput, max_retries=1)
        node.step_name = "test_step"

        state = PipelineState(
            observation=Observation(
                entity_id="test",
                current_simulation_time=0,
                status=StatusObservation(position=(0, 0), movement_locked=False)
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
                usage_metadata={"input_tokens": 10, "output_tokens": 4, "total_tokens": 14}
            ),
            AIMessage(
                content='{"required_field": "correct"}',
                usage_metadata={"input_tokens": 15, "output_tokens": 5, "total_tokens": 20}
            ),
        ]

        prompt = PromptTemplate.from_template("{input}")
        node = LLMNode(llm=mock_llm, prompt=prompt, output_model=TestOutput, max_retries=1)
        node.step_name = "test_step"

        state = PipelineState(
            observation=Observation(
                entity_id="test",
                current_simulation_time=0,
                status=StatusObservation(position=(0, 0), movement_locked=False)
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
            usage_metadata={"input_tokens": 10, "output_tokens": 4, "total_tokens": 14}
        )

        prompt = PromptTemplate.from_template("{input}")
        node = LLMNode(llm=mock_llm, prompt=prompt, output_model=TestOutput, max_retries=2)

        state = PipelineState(
            observation=Observation(
                entity_id="test",
                current_simulation_time=0,
                status=StatusObservation(position=(0, 0), movement_locked=False)
            )
        )

        with pytest.raises(Exception):  # JSONDecodeError or ValidationError
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
                usage_metadata={"input_tokens": 10, "output_tokens": 1, "total_tokens": 11}
            ),
            AIMessage(
                content="also bad",
                usage_metadata={"input_tokens": 12, "output_tokens": 2, "total_tokens": 14}
            ),
            AIMessage(
                content='{"value": "good"}',
                usage_metadata={"input_tokens": 14, "output_tokens": 4, "total_tokens": 18}
            ),
        ]

        prompt = PromptTemplate.from_template("{input}")
        node = LLMNode(llm=mock_llm, prompt=prompt, output_model=TestOutput, max_retries=2)
        node.step_name = "test_step"

        state = PipelineState(
            observation=Observation(
                entity_id="test",
                current_simulation_time=0,
                status=StatusObservation(position=(0, 0), movement_locked=False)
            )
        )

        await node.call_llm(state, input="test")

        # Should sum all attempts: 11 + 14 + 18 = 43
        assert state.tokens_used["test_step"] == 43

    @pytest.mark.asyncio
    async def test_retry_tracks_tokens_even_on_failure(self):
        """Should track tokens even when all retries fail"""
        class TestOutput(BaseModel):
            value: str

        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = [
            AIMessage(
                content="bad1",
                usage_metadata={"input_tokens": 10, "output_tokens": 1, "total_tokens": 11}
            ),
            AIMessage(
                content="bad2",
                usage_metadata={"input_tokens": 11, "output_tokens": 1, "total_tokens": 12}
            ),
        ]

        prompt = PromptTemplate.from_template("{input}")
        node = LLMNode(llm=mock_llm, prompt=prompt, output_model=TestOutput, max_retries=1)
        node.step_name = "test_step"

        state = PipelineState(
            observation=Observation(
                entity_id="test",
                current_simulation_time=0,
                status=StatusObservation(position=(0, 0), movement_locked=False)
            )
        )

        with pytest.raises(Exception):
            await node.call_llm(state, input="test")

        # Should still track tokens: 11 + 12 = 23
        assert state.tokens_used["test_step"] == 23


class TestTokenExtraction:
    """Test _extract_tokens edge cases"""

    def test_extract_tokens_with_usage_metadata(self):
        """Should extract tokens from usage_metadata"""
        mock_llm = AsyncMock()
        prompt = PromptTemplate.from_template("{input}")
        node = LLMNode(llm=mock_llm, prompt=prompt)

        response = AIMessage(
            content="test",
            usage_metadata={"input_tokens": 5, "output_tokens": 3, "total_tokens": 8}
        )

        tokens = node._extract_tokens(response)
        assert tokens == 8

    def test_extract_tokens_without_usage_metadata(self):
        """Should return 0 when no usage_metadata"""
        mock_llm = AsyncMock()
        prompt = PromptTemplate.from_template("{input}")
        node = LLMNode(llm=mock_llm, prompt=prompt)

        response = AIMessage(content="test")

        tokens = node._extract_tokens(response)
        assert tokens == 0

    def test_extract_tokens_with_missing_total_tokens(self):
        """Should return 0 when total_tokens is missing from usage_metadata"""
        mock_llm = AsyncMock()
        prompt = PromptTemplate.from_template("{input}")
        node = LLMNode(llm=mock_llm, prompt=prompt)

        response = AIMessage(
            content="test",
            usage_metadata={"input_tokens": 5, "output_tokens": 3, "total_tokens": 0}
        )

        tokens = node._extract_tokens(response)
        assert tokens == 0
