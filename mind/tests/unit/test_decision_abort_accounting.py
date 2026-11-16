"""A decision that fails, times out or is abandoned still reports what it cost.

[NPC-1689] / [NPC-1195]. Before this, every failure outside ReflectionNode's salvage
loop lost all spend: LangGraph hands each node a copy of PipelineState, so a node that
raised took its tokens_used write with it, and the error response carried no
telemetry. The round-trip ledger (nodes/base.py::recording_round_trips) is written as
each provider round-trip settles, outside graph state, and the error branches fold it.

These tests drive the real pipeline (real nodes, real ChromaDB store) with a scripted
LLM, through the real MCP tool, so the claim that matters -- spend reaches the wire --
is asserted at the wire.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field

import pytest
from langchain_core.messages import AIMessage
from mcp import types
from mcp.shared.exceptions import McpError
from mcp.shared.memory import create_connected_server_and_client_session

from mind.cognitive_architecture.nodes.base import recording_round_trips
from mind.cognitive_architecture.observations import Observation
from mind.cognitive_architecture.state import PipelineState
from mind.interfaces.mcp import server as server_module
from mind.interfaces.mcp.server import MCPServer

MIND_ID = "mind_abort"
ENTITY_ID = "entity_abort"

OBSERVATION = {
    "entity_id": ENTITY_ID,
    "current_simulation_time": 480,
    "status": {
        "position": [5, 5],
        "movement_locked": False,
        "current_interaction": {},
        "controller_state": {},
    },
}

QUERY_JSON = json.dumps({"queries": ["what is nearby"]})
REFLECTION_JSON = json.dumps(
    {
        "updated_working_memory": {
            "situation_assessment": "calm",
            "active_goals": [],
            "emotional_state": "ok",
        },
        "new_memories": [],
        "chosen_action": {"action": "wait", "parameters": {}},
    }
)


def _reply(content: str, total: int = 100) -> AIMessage:
    return AIMessage(
        content=content,
        usage_metadata={
            "input_tokens": total - 20,
            "output_tokens": 20,
            "total_tokens": total,
        },
    )


@dataclass
class Hang:
    """A provider call that never answers."""


@dataclass
class ScriptedLLM:
    """Plays a script, one step per ainvoke: an AIMessage, an exception, or Hang.

    Hang sets ``in_flight``, then awaits forever; ``cancelled`` records that the
    provider call was actually torn down by a CancelledError, which is the property
    the server's deadline and cancel handling exist to deliver.
    """

    script: list
    calls: int = 0
    cancelled: bool = False
    in_flight: asyncio.Event = field(default_factory=asyncio.Event)
    model_name: str = "scripted/model"

    async def ainvoke(self, messages):
        step = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        if isinstance(step, Hang):
            self.in_flight.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise
        if isinstance(step, BaseException):
            raise step
        return step


async def _make_server(monkeypatch, llm: ScriptedLLM) -> MCPServer:
    monkeypatch.setattr("mind.interfaces.mcp.mind.get_llm", lambda *a, **k: llm)
    server = MCPServer()
    await server.mcp.call_tool(
        "create_mind",
        {
            "mind_id": MIND_ID,
            "entity_id": ENTITY_ID,
            "config": {"traits": [], "initial_long_term_memories": []},
        },
    )
    return server


async def _decide(server: MCPServer, observation: dict | None = None) -> dict:
    result = await asyncio.wait_for(
        server.mcp.call_tool(
            "decide_action",
            {"mind_id": MIND_ID, "observation": observation or OBSERVATION},
        ),
        5,
    )
    return json.loads(result[0].text)


class TestErrorResponsesCarryTelemetry:
    async def test_first_node_transport_failure_reports_one_unreported_call(self, monkeypatch):
        llm = ScriptedLLM([TimeoutError("provider stalled")])
        server = await _make_server(monkeypatch, llm)

        response = await _decide(server)

        assert response["status"] == "error"
        telemetry = response["telemetry"]
        assert telemetry["provenance"] == "unreported"
        assert telemetry["model_calls"] == 1
        assert telemetry["unreported_calls"] == 1
        assert telemetry["per_step_ms"] == {}
        assert telemetry["server_ms"] >= 0

    async def test_later_failure_keeps_the_earlier_steps_spend(self, monkeypatch):
        """The case the old design lost: memory_query succeeded (metered), then
        reflection's provider call raised. Without the ledger the first step's
        tokens died with the failed node's state copy."""
        llm = ScriptedLLM([_reply(QUERY_JSON, total=120), RuntimeError("provider 5xx")])
        server = await _make_server(monkeypatch, llm)

        response = await _decide(server)

        assert response["status"] == "error"
        telemetry = response["telemetry"]
        assert telemetry["provenance"] == "metered"
        assert telemetry["model_calls"] == 2
        assert telemetry["unreported_calls"] == 1
        assert telemetry["total_tokens"] == 120
        assert telemetry["per_step"]["memory_query"]["total_tokens"] == 120
        assert telemetry["per_step"]["reflection"]["unreported_calls"] == 1

    async def test_retry_exhaustion_counts_every_attempt(self, monkeypatch):
        """memory_query has max_retries=2 and no salvage: three invalid answers
        raise out of the pipeline. All three attempts were billed."""
        llm = ScriptedLLM([_reply("not json at all {{{", total=50)])
        server = await _make_server(monkeypatch, llm)

        response = await _decide(server)

        assert response["status"] == "error"
        telemetry = response["telemetry"]
        assert telemetry["provenance"] == "metered"
        assert telemetry["model_calls"] == 3
        assert telemetry["unreported_calls"] == 0
        assert telemetry["total_tokens"] == 150

    async def test_pre_pipeline_rejection_stays_telemetry_free(self, monkeypatch):
        """Absence means 'no model ran'; it must remain distinguishable."""
        llm = ScriptedLLM([_reply(QUERY_JSON)])
        server = await _make_server(monkeypatch, llm)

        mismatched = {**OBSERVATION, "entity_id": "someone_else"}
        response = await _decide(server, mismatched)

        assert response["status"] == "error"
        assert "telemetry" not in response
        assert llm.calls == 0

        unknown = json.loads(
            (
                await server.mcp.call_tool(
                    "decide_action", {"mind_id": "nobody", "observation": OBSERVATION}
                )
            )[0].text
        )
        assert unknown["status"] == "error"
        assert "telemetry" not in unknown

    async def test_no_action_selected_carries_the_completed_pipelines_telemetry(self, monkeypatch):
        llm = ScriptedLLM([_reply(QUERY_JSON, total=90), _reply(REFLECTION_JSON, total=110)])
        server = await _make_server(monkeypatch, llm)

        original = server.minds[MIND_ID].pipeline.process

        async def process_then_drop_action(state: PipelineState) -> PipelineState:
            result = await original(state)
            result.chosen_action = None
            return result

        server.minds[MIND_ID].pipeline.process = process_then_drop_action

        response = await _decide(server)

        assert response["status"] == "error"
        assert response["error_message"] == "Pipeline did not select an action"
        telemetry = response["telemetry"]
        assert telemetry["provenance"] == "metered"
        assert telemetry["total_tokens"] == 200
        assert telemetry["model_calls"] == 2


class TestDecisionDeadline:
    async def test_a_stalled_provider_is_abandoned_with_its_cost(self, monkeypatch, caplog):
        # Long enough that memory_retrieval (synchronous embedding work, no await to
        # cancel at) finishes first, so the deadline lands while reflection hangs.
        monkeypatch.setattr(server_module, "DECISION_DEADLINE_SECONDS", 2.0)
        llm = ScriptedLLM([_reply(QUERY_JSON, total=120), Hang()])
        server = await _make_server(monkeypatch, llm)

        with caplog.at_level(logging.WARNING):
            response = await _decide(server)

        assert response["status"] == "error"
        assert "deadline" in response["error_message"]
        telemetry = response["telemetry"]
        assert telemetry["provenance"] == "metered"
        assert telemetry["model_calls"] == 2
        assert telemetry["unreported_calls"] == 1
        assert telemetry["total_tokens"] == 120
        assert llm.cancelled, "the provider call must be torn down, not left running"
        assert any(
            f"[{ENTITY_ID}]" in r.getMessage() and "deadline" in r.getMessage()
            for r in caplog.records
            if r.levelno == logging.WARNING
        )

    async def test_the_mind_is_untouched_by_an_abandoned_decision(self, monkeypatch):
        monkeypatch.setattr(server_module, "DECISION_DEADLINE_SECONDS", 0.2)
        llm = ScriptedLLM([Hang()])
        server = await _make_server(monkeypatch, llm)
        before = server.minds[MIND_ID].working_memory.model_copy(deep=True)

        await _decide(server)

        assert server.minds[MIND_ID].working_memory == before
        assert server.minds[MIND_ID].daily_memories == []

    async def test_a_foreign_timeout_error_takes_the_generic_branch(self, monkeypatch):
        """Only the deadline's own expiry is 'the deadline': a provider's
        TimeoutError (here, instantly) must not be reported as one."""
        llm = ScriptedLLM([TimeoutError("provider read timeout")])
        server = await _make_server(monkeypatch, llm)

        response = await _decide(server)

        assert response["error_message"] == "Unexpected server error"
        assert "telemetry" in response


class TestCancellation:
    async def test_task_level_cancel_stops_the_provider_call_and_logs_spend(
        self, monkeypatch, caplog
    ):
        llm = ScriptedLLM([_reply(QUERY_JSON, total=120), Hang()])
        server = await _make_server(monkeypatch, llm)
        mind = server.minds[MIND_ID]
        before = mind.working_memory.model_copy(deep=True)

        with caplog.at_level(logging.WARNING):
            task = asyncio.ensure_future(
                server.mcp.call_tool(
                    "decide_action", {"mind_id": MIND_ID, "observation": OBSERVATION}
                )
            )
            await asyncio.wait_for(llm.in_flight.wait(), 5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 5)

        assert llm.cancelled
        assert mind.working_memory == before
        messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        line = next(m for m in messages if "cancelled" in m)
        assert f"[{ENTITY_ID}]" in line
        assert "2 model calls" in line
        assert "1 unreported" in line
        assert "120 tokens" in line

    async def test_protocol_level_cancel_reaches_the_provider_call(self, monkeypatch, caplog):
        """The falsifier for 'the game abandoning a decision stops the spend': a real
        notifications/cancelled over a real client session, through fastmcp,
        LangGraph, LLMNode, down to the awaited provider call."""
        llm = ScriptedLLM([_reply(QUERY_JSON, total=120), Hang()])
        server = await _make_server(monkeypatch, llm)

        with caplog.at_level(logging.WARNING):
            async with create_connected_server_and_client_session(server.mcp._mcp_server) as client:
                call = asyncio.ensure_future(
                    client.call_tool(
                        "decide_action", {"mind_id": MIND_ID, "observation": OBSERVATION}
                    )
                )
                await asyncio.wait_for(llm.in_flight.wait(), 5)
                # The SDK numbers client requests from 0: initialize is 0 and
                # call_tool is the next request on the session.
                request_id = _in_flight_request_id(client)
                await client.send_notification(
                    types.ClientNotification(
                        types.CancelledNotification(
                            method="notifications/cancelled",
                            params=types.CancelledNotificationParams(
                                requestId=request_id, reason="game abandoned the decision"
                            ),
                        )
                    )
                )
                with pytest.raises(McpError):
                    await asyncio.wait_for(call, 5)
                await asyncio.wait_for(_until(lambda: llm.cancelled), 1)

        assert llm.cancelled
        assert any(
            "cancelled" in r.getMessage() and f"[{ENTITY_ID}]" in r.getMessage()
            for r in caplog.records
            if r.levelno == logging.WARNING
        )


def _in_flight_request_id(client) -> int:
    """The id of the one request the client has sent and not seen answered."""
    pending = list(client._response_streams)
    assert len(pending) == 1, pending
    return pending[0]


async def _until(predicate) -> None:
    while not predicate():
        await asyncio.sleep(0.01)


class TestLedgerAgreesWithState:
    async def test_full_pipeline_ledger_equals_result_tokens_used(self, monkeypatch):
        """Two records of one spend must not drift: this also pins that the
        ContextVar write inside LangGraph's per-node task reaches the opener."""
        llm = ScriptedLLM([_reply(QUERY_JSON, total=90), _reply(REFLECTION_JSON, total=110)])
        server = await _make_server(monkeypatch, llm)
        mind = server.minds[MIND_ID]
        state = mind.build_pipeline_state(Observation.model_validate(OBSERVATION))

        with recording_round_trips() as ledger:
            result = await mind.pipeline.process(state)

        assert ledger == result.tokens_used
        assert set(ledger) == {"memory_query", "reflection"}
