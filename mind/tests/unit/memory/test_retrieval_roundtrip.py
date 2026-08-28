"""End-to-end memory round-trip: decide -> consolidate -> decide.

Every retrieval-quality measurement taken before this lane ran against a store
that had structurally never been written to: `decide_action` only appends to the
mind's daily buffer, and nothing drains that buffer into ChromaDB except the
`consolidate_memories` tool, which the simulation calls on wake. Short scenarios
and harnesses never slept, so retrieval was inert in every observation of it.

Testing the formula against a fixture-populated store would dodge that entirely.
These tests fill the store **through production code** - the real MCP tools, the
real pipeline, the real consolidation node - with only the LLM faked, so the
write path's own defects are in scope rather than fixtured away.
"""

import json
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage

from mind.cognitive_architecture.memory.vector_db_memory import VectorDBQuery
from mind.cognitive_architecture.nodes.memory_retrieval.node import MemoryRetrievalNode
from mind.interfaces.mcp.server import MCPServer

MEMORY_CONTENT = "The mountain bandits burned the north bridge and I barely escaped"
MEMORY_IMPORTANCE = 9.0

CYCLE_ONE_TIME = 5_000
CYCLE_TWO_TIME = 5_060


class FakeLLM:
    """Minimal stand-in for a chat model.

    The node layer only ever calls `ainvoke(messages) -> AIMessage` and parses
    `.content` as JSON, so this is the whole surface. Dispatch is on the
    reflection schema's distinctive key rather than on call order, which would
    silently mis-answer if a node ever retried.
    """

    def __init__(self):
        self.calls = 0

    async def ainvoke(self, messages, **kwargs):
        self.calls += 1
        prompt = str(messages[0].content)

        if "chosen_action" in prompt:
            payload = {
                "updated_working_memory": {
                    "situation_assessment": "Shaken but safe",
                    "active_goals": ["reach the village"],
                    "recent_events": ["escaped the bandits"],
                    "current_plan": ["keep moving"],
                    "emotional_state": "afraid",
                },
                "new_memories": [{"content": MEMORY_CONTENT, "importance": MEMORY_IMPORTANCE}],
                "chosen_action": {"action": "wander", "parameters": {}},
            }
        else:
            payload = {"queries": ["bandits on the road", "the north bridge"]}

        return AIMessage(content=json.dumps(payload))


@pytest.fixture
def isolated_chroma(monkeypatch, tmp_path):
    """Hermetic per-test ChromaDB, matching the pattern the server tests use."""
    from chromadb.api.client import SharedSystemClient

    SharedSystemClient.clear_system_cache()
    monkeypatch.chdir(tmp_path)
    yield
    SharedSystemClient.clear_system_cache()


@pytest.fixture
def retrieval_spy(monkeypatch):
    """Record what each cycle's retrieval node put into state.

    Patched at the class before any pipeline is constructed: the graph binds
    `self.memory_retrieval_node.process` at compile time, so a later instance
    patch would never be called.
    """
    captured: list[list] = []
    original = MemoryRetrievalNode.process

    async def spy(self, state):
        result = await original(self, state)
        captured.append(list(result.retrieved_memories))
        return result

    monkeypatch.setattr(MemoryRetrievalNode, "process", spy)
    return captured


async def _decide(server, mind_id, simulation_time):
    return await server.mcp.call_tool(
        "decide_action",
        {
            "mind_id": mind_id,
            "observation": {
                "entity_id": "entity_roundtrip",
                "current_simulation_time": simulation_time,
            },
        },
    )


@pytest.mark.usefixtures("isolated_chroma")
class TestMemoryRoundTrip:
    async def _server_with_mind(self, mind_id="mind_roundtrip"):
        server = MCPServer()
        await server.mcp.call_tool(
            "create_mind",
            {
                "mind_id": mind_id,
                "entity_id": "entity_roundtrip",
                "config": {"traits": ["wary"], "initial_long_term_memories": []},
            },
        )
        return server

    async def test_consolidated_memory_carries_the_observed_simulation_time(self):
        """Regression for the consolidation write path.

        `consolidate_memories` runs outside the decision cycle and used to
        fabricate an observation with current_simulation_time=0, which the
        consolidation node read as the memory's timestamp. **Every lived memory
        in the system was therefore stamped at the epoch**, so the recency term
        was fed a constant and decayed everything from the beginning of time.

        The stamp must be the time the mind last actually observed.
        """
        with patch("mind.interfaces.mcp.mind.get_llm", return_value=FakeLLM()):
            server = await self._server_with_mind()
            await _decide(server, "mind_roundtrip", CYCLE_ONE_TIME)

            mind = server.minds["mind_roundtrip"]
            assert mind.daily_memories, "the fake reflection should have produced a memory"

            await server.mcp.call_tool("consolidate_memories", {"mind_id": "mind_roundtrip"})

            stored = await mind.memory_store.search(
                VectorDBQuery(query="bandits", top_k=5, current_simulation_time=CYCLE_TWO_TIME)
            )

        assert len(stored) == 1
        assert stored[0].timestamp == CYCLE_ONE_TIME, (
            "consolidated memories must carry the last observed game time, not 0"
        )
        assert stored[0].importance == MEMORY_IMPORTANCE, (
            "reflection's LLM importance rating must survive consolidation"
        )

    async def test_second_cycle_retrieves_what_the_first_cycle_remembered(self, retrieval_spy):
        """The store starts empty and retrieval fills it through production code.

        Cycle 1 retrieves nothing because nothing has been written. Consolidation
        drains the daily buffer. Cycle 2 must then actually retrieve the memory -
        which is the first end-to-end evidence that the formula operates on a
        non-empty store at all.
        """
        with patch("mind.interfaces.mcp.mind.get_llm", return_value=FakeLLM()):
            server = await self._server_with_mind()

            await _decide(server, "mind_roundtrip", CYCLE_ONE_TIME)
            assert retrieval_spy[0] == [], "nothing has been consolidated yet"

            await server.mcp.call_tool("consolidate_memories", {"mind_id": "mind_roundtrip"})

            await _decide(server, "mind_roundtrip", CYCLE_TWO_TIME)

        assert len(retrieval_spy) == 2
        second_cycle = retrieval_spy[1]
        assert second_cycle, "the consolidated memory must be retrievable on the next cycle"
        assert any(m.content == MEMORY_CONTENT for m in second_cycle)
        assert all(m.timestamp == CYCLE_ONE_TIME for m in second_cycle)

    async def test_consolidating_before_any_decision_writes_no_false_timestamp(self):
        """A mind can be consolidated before it has ever decided.

        There is genuinely no observed time in that case. Writing 0 would be a
        fabricated measurement that scores as the oldest possible memory forever;
        None lets the recency term abstain instead.
        """
        from mind.cognitive_architecture.working_memory import NewMemory

        with patch("mind.interfaces.mcp.mind.get_llm", return_value=FakeLLM()):
            server = await self._server_with_mind()
            mind = server.minds["mind_roundtrip"]
            assert mind.last_simulation_time is None

            mind.daily_memories.append(NewMemory(content="A memory from nowhen", importance=4.0))
            await server.mcp.call_tool("consolidate_memories", {"mind_id": "mind_roundtrip"})

            stored = await mind.memory_store.search(
                VectorDBQuery(query="nowhen", top_k=5, current_simulation_time=100)
            )

        assert len(stored) == 1
        assert stored[0].timestamp is None
