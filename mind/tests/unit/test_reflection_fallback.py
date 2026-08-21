"""Fallback matrix for ReflectionNode's partial salvage (NPC-1319).

The retired two-node pipeline had asymmetric failure modes: an action_selection
failure degraded to WAIT with memory committed, but a cognitive_update failure
lost the ENTIRE decision (no action, no memory write, telemetry discarded at
the server boundary — NPC-1195). The merged node must be strictly better on
both axes: every salvage case ends with a non-null action, a never-empty
working memory, and delivered telemetry, and each output field is rescued
independently so a malformed working memory does not discard a valid action or
vice versa.
"""

import logging
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage

from mind.cognitive_architecture.actions import ActionType, AvailableAction
from mind.cognitive_architecture.nodes.reflection.node import ReflectionNode
from mind.cognitive_architecture.observations import Observation, StatusObservation
from mind.cognitive_architecture.pipeline import CognitivePipeline
from mind.cognitive_architecture.state import PipelineState
from mind.cognitive_architecture.working_memory import WorkingMemory

VALID_WM = '{"situation_assessment": "Fixing the fence", "emotional_state": "calm"}'


def make_failing_llm(content: str) -> AsyncMock:
    """An LLM that returns the same (invalid) response on every attempt"""
    mock = AsyncMock()
    mock.ainvoke.return_value = AIMessage(
        content=content,
        usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
    )
    return mock


def make_state(**observation_overrides) -> PipelineState:
    status = StatusObservation(
        position=(5, 10),
        movement_locked=observation_overrides.pop("movement_locked", False),
    )
    return PipelineState(
        observation=Observation(
            entity_id="test_npc",
            current_simulation_time=100,
            status=status,
            **observation_overrides,
        ),
        working_memory=WorkingMemory(
            situation_assessment="At the forge",
            active_goals=["Work on sword"],
            emotional_state="Focused",
        ),
        available_actions=[AvailableAction(name="wait", description="Wait and observe")],
    )


def assert_salvage_floors(result: PipelineState):
    """The invariants every salvage case must land on"""
    assert result.chosen_action is not None
    assert str(result.working_memory) != "No working memory", (
        "a parse failure must never wipe the NPC's mind"
    )
    assert "reflection" in result.tokens_used, "telemetry must be delivered, not discarded"
    assert result.tokens_used["reflection"].model_calls == 3, (
        "all three attempts are real spend and must stay countable"
    )


@pytest.mark.asyncio
class TestReflectionSalvageMatrix:
    async def test_not_json_falls_back_to_wait_and_keeps_working_memory(self):
        node = ReflectionNode(make_failing_llm("this is not json at all, every time"))
        state = make_state()

        result = await node.process(state)

        assert_salvage_floors(result)
        assert result.chosen_action.action == ActionType.WAIT
        assert result.working_memory.situation_assessment == "At the forge"
        assert result.daily_memories == []

    async def test_missing_chosen_action_salvages_working_memory_and_memories(self):
        content = (
            "{"
            f'"updated_working_memory": {VALID_WM}, '
            '"new_memories": [{"content": "Fence post snapped", "importance": 4.0}]'
            "}"
        )
        node = ReflectionNode(make_failing_llm(content))
        state = make_state()

        result = await node.process(state)

        assert_salvage_floors(result)
        assert result.chosen_action.action == ActionType.WAIT
        assert result.working_memory.situation_assessment == "Fixing the fence"
        assert [m.content for m in result.daily_memories] == ["Fence post snapped"]

    async def test_unknown_action_salvages_working_memory(self):
        content = (
            "{"
            f'"updated_working_memory": {VALID_WM}, '
            '"new_memories": [], '
            '"chosen_action": {"action": "fly_away", "parameters": {}}'
            "}"
        )
        node = ReflectionNode(make_failing_llm(content))
        state = make_state()

        result = await node.process(state)

        assert_salvage_floors(result)
        assert result.chosen_action.action == ActionType.WAIT
        assert result.working_memory.situation_assessment == "Fixing the fence"

    async def test_movement_locked_move_to_falls_back_to_wait(self):
        content = (
            "{"
            f'"updated_working_memory": {VALID_WM}, '
            '"new_memories": [], '
            '"chosen_action": {"action": "move_to", "parameters": {"destination": [1, 2]}}'
            "}"
        )
        node = ReflectionNode(make_failing_llm(content))
        state = make_state(movement_locked=True)

        result = await node.process(state)

        assert_salvage_floors(result)
        assert result.chosen_action.action == ActionType.WAIT
        assert result.working_memory.situation_assessment == "Fixing the fence"

    async def test_act_in_interaction_while_not_interacting_falls_back_to_wait(self):
        content = (
            "{"
            f'"updated_working_memory": {VALID_WM}, '
            '"new_memories": [], '
            '"chosen_action": {"action": "act_in_interaction", "parameters": {"message": "hi"}}'
            "}"
        )
        node = ReflectionNode(make_failing_llm(content))
        state = make_state()

        result = await node.process(state)

        assert_salvage_floors(result)
        assert result.chosen_action.action == ActionType.WAIT

    async def test_interact_with_entity_not_in_vision_falls_back_to_wait(self):
        content = (
            "{"
            f'"updated_working_memory": {VALID_WM}, '
            '"new_memories": [], '
            '"chosen_action": {"action": "interact_with", '
            '"parameters": {"entity_id": "ghost_1", "interaction_name": "use"}}'
            "}"
        )
        node = ReflectionNode(make_failing_llm(content))
        state = make_state(vision={"visible_entities": []})

        result = await node.process(state)

        assert_salvage_floors(result)
        assert result.chosen_action.action == ActionType.WAIT

    async def test_valid_action_survives_a_malformed_working_memory(self):
        """A working memory that arrives as a string fails validation; the
        action is rescued independently rather than degraded to WAIT."""
        content = (
            "{"
            '"updated_working_memory": "feeling pretty good about the fence", '
            '"new_memories": [], '
            '"chosen_action": {"action": "wander", "parameters": {}}'
            "}"
        )
        node = ReflectionNode(make_failing_llm(content))
        state = make_state()

        result = await node.process(state)

        assert_salvage_floors(result)
        assert result.chosen_action.action == ActionType.WANDER
        assert result.working_memory.situation_assessment == "At the forge"

    async def test_valid_action_rescued_while_empty_working_memory_is_refused(self):
        """The wipe guard: an empty WorkingMemory validates trivially, so
        salvage must refuse it and keep the current one."""
        content = (
            "{"
            '"updated_working_memory": {}, '
            '"new_memories": "definitely not a list", '
            '"chosen_action": {"action": "wander", "parameters": {}}'
            "}"
        )
        node = ReflectionNode(make_failing_llm(content))
        state = make_state()

        result = await node.process(state)

        assert_salvage_floors(result)
        assert result.chosen_action.action == ActionType.WANDER
        assert result.working_memory.situation_assessment == "At the forge"
        assert result.daily_memories == []

    async def test_one_bad_memory_entry_does_not_discard_the_others(self):
        content = (
            "{"
            f'"updated_working_memory": {VALID_WM}, '
            '"new_memories": ['
            '{"content": "Good one", "importance": 5.0}, '
            '{"content": "Out of range", "importance": 99.0}, '
            '{"content": "Also good", "importance": 3.0}'
            "], "
            '"chosen_action": {"action": "wander", "parameters": {}}'
            "}"
        )
        node = ReflectionNode(make_failing_llm(content))
        state = make_state()

        result = await node.process(state)

        assert_salvage_floors(result)
        assert result.chosen_action.action == ActionType.WANDER
        assert [m.content for m in result.daily_memories] == ["Good one", "Also good"]

    async def test_salvage_log_records_carry_entity_id(self, caplog):
        """Retry and salvage-warning records must carry the entity id (NPC-789)"""
        node = ReflectionNode(make_failing_llm("not valid json"))
        state = make_state()

        with caplog.at_level(logging.DEBUG, logger="mind"):
            result = await node.process(state)

        assert result.chosen_action.action == ActionType.WAIT
        assert caplog.records, "salvage path should emit log records"
        for record in caplog.records:
            assert "test_npc" in record.getMessage(), (
                f"Unattributed log record: {record.getMessage()!r}"
            )
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1, "salvage announces itself in exactly one warning record"


@pytest.mark.asyncio
class TestReflectionRetry:
    async def test_successful_retry_recovers_without_salvage(self):
        mock = AsyncMock()
        mock.ainvoke.side_effect = [
            AIMessage(
                content="not valid json",
                usage_metadata={"input_tokens": 100, "output_tokens": 5, "total_tokens": 105},
            ),
            AIMessage(
                content=(
                    "{"
                    f'"updated_working_memory": {VALID_WM}, '
                    '"new_memories": [], '
                    '"chosen_action": {"action": "wait", "parameters": {}}'
                    "}"
                ),
                usage_metadata={"input_tokens": 120, "output_tokens": 40, "total_tokens": 160},
            ),
        ]
        node = ReflectionNode(mock)
        state = make_state()

        result = await node.process(state)

        assert result.tokens_used["reflection"].model_calls == 2
        assert result.chosen_action.action == ActionType.WAIT
        assert result.working_memory.situation_assessment == "Fixing the fence"


@pytest.mark.asyncio
class TestFullPipelineGarbageRun:
    async def test_wait_fallback_survives_pipeline_revalidation(self):
        """End to end: a reflection that fails every attempt still yields a
        decision the pipeline's downstream re-validation accepts."""
        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = [
            # memory_query succeeds
            AIMessage(
                content='{"queries": ["the forge"]}',
                usage_metadata={"input_tokens": 50, "output_tokens": 10, "total_tokens": 60},
            ),
            # reflection burns all three attempts
            AIMessage(content="garbage 1"),
            AIMessage(content="garbage 2"),
            AIMessage(content="garbage 3"),
        ]
        memory_store = AsyncMock()
        memory_store.search.return_value = []

        pipeline = CognitivePipeline(mock_llm, memory_store)
        state = make_state()

        result = await pipeline.process(state)

        assert result.chosen_action is not None
        assert result.chosen_action.action == ActionType.WAIT
        assert str(result.working_memory) != "No working memory"
        assert "reflection" in result.tokens_used
        assert result.tokens_used["reflection"].model_calls == 3
