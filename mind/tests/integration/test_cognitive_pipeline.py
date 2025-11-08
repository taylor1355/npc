"""Integration tests for cognitive pipeline with real LLM calls

Tests end-to-end pipeline behavior with minimal LLM calls.
Uses engineered scenarios with clear expected outcomes.
"""

import pytest

from mind.apis.langchain_llm import get_llm
from mind.cognitive_architecture.memory.vector_db_memory import VectorDBMemory
from mind.cognitive_architecture.actions import ActionType
from mind.cognitive_architecture.pipeline import CognitivePipeline
from mind.cognitive_architecture.state import PipelineState
from mind.constants import DEFAULT_SMALL_MODEL
from tests.fixtures import (
    create_blacksmith_observation,
    create_risk_scenario_observation,
)


@pytest.mark.asyncio
class TestCognitivePipelineIntegration:
    """Integration tests with real LLM - minimal calls, maximum coverage"""

    @pytest.fixture
    def llm(self):
        """Real LLM (Gemini Flash via OpenRouter)"""
        return get_llm(DEFAULT_SMALL_MODEL, temperature=0.7)

    @pytest.fixture
    def memory_store(self):
        """In-memory vector database"""
        store = VectorDBMemory(collection_name="test_pipeline_integration")
        yield store
        store.clear()

    @pytest.fixture
    def pipeline(self, llm, memory_store):
        """Full cognitive pipeline"""
        return CognitivePipeline(llm=llm, memory_store=memory_store)

    async def test_full_pipeline_execution(self, pipeline):
        """Comprehensive test: pipeline completes with valid outputs"""
        # Arrange: Blacksmith with low energy, bed nearby
        observation = create_blacksmith_observation(simulation_time=1000)
        state = PipelineState(
            observation=observation,
            personality_traits=["diligent", "perfectionist"],
            available_actions=[],
        )

        # Act
        result = await pipeline.process(state)

        # Assert: Pipeline completed all nodes
        assert "memory_query" in result.time_ms
        assert "memory_retrieval" in result.time_ms
        assert "cognitive_update" in result.time_ms
        assert "action_selection" in result.time_ms
        assert all(t >= 0 for t in result.time_ms.values())

        # Assert: Memory queries generated
        assert len(result.memory_queries) > 0
        assert all(isinstance(q, str) and len(q) > 0 for q in result.memory_queries)

        # Assert: Working memory updated with valid structure
        assert result.working_memory is not None
        assert len(result.working_memory.situation_assessment) > 0
        assert len(result.working_memory.emotional_state) > 0

        # Assert: Valid action selected
        assert result.chosen_action is not None
        assert isinstance(result.chosen_action.action, str)
        assert result.chosen_action.action in [e.value for e in ActionType]
        assert isinstance(result.chosen_action.parameters, dict)

        # Assert: Token tracking works
        assert all(
            node in result.tokens_used
            for node in ["memory_query", "cognitive_update", "action_selection"]
        )
        assert all(t > 0 for t in result.tokens_used.values())

        # Assert: Daily memories have valid structure
        for mem in result.daily_memories:
            assert hasattr(mem, "content") and hasattr(mem, "importance")
            assert 1.0 <= mem.importance <= 10.0

    async def test_memory_retrieval_integration(self, pipeline, memory_store):
        """Test that populated memories are retrieved"""
        # Arrange: Pre-populate relevant memories
        memory_store.add_memory("I crafted a ceremonial blade", importance=8.0, timestamp=900)
        memory_store.add_memory("The forge needs more coal", importance=7.0, timestamp=950)

        observation = create_blacksmith_observation(simulation_time=1000)
        state = PipelineState(
            observation=observation, personality_traits=["diligent"], available_actions=[]
        )

        # Act
        result = await pipeline.process(state)

        # Assert: Memories were retrieved and have valid structure
        assert len(result.retrieved_memories) > 0
        assert all(hasattr(m, "id") and hasattr(m, "content") for m in result.retrieved_memories)

    async def test_personality_drives_decision_making(self, pipeline):
        """Test that personality traits influence decision-making process

        Engineered scenario with safe vs risky options.
        Tests that:
        - Personality traits are reflected in working memory assessment
        - Different personalities produce different cognitive assessments
        - Both personalities make valid, contextually appropriate decisions
        """
        observation = create_risk_scenario_observation()

        # Cautious personality
        state_cautious = PipelineState(
            observation=observation,
            personality_traits=["cautious", "fearful", "risk-averse"],
            available_actions=[],
        )
        result_cautious = await pipeline.process(state_cautious)

        # Reckless personality
        state_reckless = PipelineState(
            observation=observation,
            personality_traits=["reckless", "adventurous", "thrill-seeking"],
            available_actions=[],
        )
        result_reckless = await pipeline.process(state_reckless)

        # Assert: Both completed successfully
        assert result_cautious.chosen_action is not None
        assert result_reckless.chosen_action is not None
        assert result_cautious.working_memory is not None
        assert result_reckless.working_memory is not None

        # Assert: Both made valid action choices
        assert result_cautious.chosen_action.action in [e.value for e in ActionType]
        assert result_reckless.chosen_action.action in [e.value for e in ActionType]

        # Assert: Working memory shows different assessments (personality influenced thinking)
        # Personalities should lead to different situation assessments or emotional states
        cautious_assessment = result_cautious.working_memory.situation_assessment.lower()
        reckless_assessment = result_reckless.working_memory.situation_assessment.lower()

        # Verify assessments are different and non-trivial
        assert len(cautious_assessment) > 20
        assert len(reckless_assessment) > 20
        assert cautious_assessment != reckless_assessment

    async def test_sequential_decisions_maintain_state(self, pipeline):
        """Test that state evolves correctly across multiple decisions"""
        # Arrange
        observation = create_blacksmith_observation(simulation_time=1000)
        state = PipelineState(
            observation=observation, personality_traits=["diligent"], available_actions=[]
        )

        # Act: First decision
        result1 = await pipeline.process(state)

        # Update state with previous working memory, advance time
        state.working_memory = result1.working_memory
        state.observation = create_blacksmith_observation(simulation_time=1100)

        # Act: Second decision
        result2 = await pipeline.process(state)

        # Assert: State maintained and evolved
        assert result2.working_memory is not None
        assert result1.working_memory is not result2.working_memory  # New object created
