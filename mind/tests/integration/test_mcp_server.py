"""Integration tests for MCP server with structured observations"""

import json

import pytest

from mind.cognitive_architecture.models import (
    EntityData,
    NeedsObservation,
    Observation,
    StatusObservation,
    VisionObservation,
)
from mind.cognitive_architecture.nodes.cognitive_update.models import WorkingMemory
from mind.interfaces.mcp.models import MindConfig
from mind.interfaces.mcp.server import MCPServer


@pytest.fixture
def mcp_server():
    """Create an MCP server instance for testing"""
    return MCPServer("Test Mind Server")


@pytest.fixture
def test_mind_config():
    """Create a test mind configuration"""
    return MindConfig(
        entity_id="test_npc_001",
        traits=["curious", "brave"],
        llm_model="google/gemini-2.0-flash-lite-001",
        embedding_model="all-MiniLM-L6-v2",
        memory_storage_path="./tmp/test_chroma_db",
        initial_working_memory=WorkingMemory(
            situation_assessment="I am exploring a new area",
            active_goals=["Learn about my surroundings"],
            emotional_state="Curious and alert",
        ),
        initial_long_term_memories=[
            "I am an adventurer exploring unknown lands",
            "I have basic survival skills",
        ],
    )


@pytest.fixture
def test_observation():
    """Create a test observation"""
    return Observation(
        entity_id="test_npc_001",
        current_simulation_time=100,
        status=StatusObservation(position=(10, 10), movement_locked=False),
        needs=NeedsObservation(
            needs={"hunger": 50.0, "energy": 75.0, "fun": 60.0, "hygiene": 80.0}
        ),
        vision=VisionObservation(
            visible_entities=[
                EntityData(
                    entity_id="tree_001",
                    display_name="Oak Tree",
                    position=(11, 10),
                    interactions={
                        "examine": {
                            "name": "examine",
                            "description": "Look closely at the tree",
                            "needs_filled": ["fun"],
                            "needs_drained": [],
                        }
                    },
                ),
                EntityData(
                    entity_id="campfire_001",
                    display_name="Campfire",
                    position=(10, 11),
                    interactions={
                        "rest": {
                            "name": "rest",
                            "description": "Rest by the fire",
                            "needs_filled": ["energy"],
                            "needs_drained": [],
                        }
                    },
                ),
            ]
        ),
    )


@pytest.mark.asyncio
async def test_create_mind(mcp_server, test_mind_config):
    """Test creating a new mind"""
    result = await mcp_server.mcp.call_tool(
        "create_mind", {"mind_id": "test_mind_001", "config": test_mind_config.model_dump()}
    )

    # Parse response from TextContent list
    response = json.loads(result[0].text)

    assert response["status"] == "created"
    assert response["mind_id"] == "test_mind_001"
    assert "test_mind_001" in mcp_server.minds


@pytest.mark.asyncio
async def test_decide_action(mcp_server, test_mind_config, test_observation):
    """Test deciding an action based on observation"""
    # Create mind
    await mcp_server.mcp.call_tool(
        "create_mind", {"mind_id": "test_mind_001", "config": test_mind_config.model_dump()}
    )

    # Decide action using new structured format
    result = await mcp_server.mcp.call_tool(
        "decide_action", {"mind_id": "test_mind_001", "observation": test_observation.model_dump()}
    )

    # Parse response from TextContent list
    response = json.loads(result[0].text)

    assert response["status"] == "success"
    assert response["action"] is not None
    assert response.get("error_message") is None


@pytest.mark.asyncio
async def test_consolidate_memories(mcp_server, test_mind_config):
    """Test memory consolidation"""
    # Create mind
    await mcp_server.mcp.call_tool(
        "create_mind", {"mind_id": "test_mind_001", "config": test_mind_config.model_dump()}
    )

    # Manually add some daily memories for testing
    mind = mcp_server.minds["test_mind_001"]
    from mind.cognitive_architecture.nodes.cognitive_update.models import NewMemory

    mind.daily_memories.append(NewMemory(content="Test memory", importance=5.0))

    # Consolidate
    result = await mcp_server.mcp.call_tool("consolidate_memories", {"mind_id": "test_mind_001"})

    # Parse response from TextContent list
    response = json.loads(result[0].text)

    assert response["status"] == "success"
    assert response["consolidated_count"] == 1
    assert len(mind.daily_memories) == 0  # Should be cleared


@pytest.mark.asyncio
async def test_cleanup_mind(mcp_server, test_mind_config):
    """Test mind cleanup"""
    # Create mind
    await mcp_server.mcp.call_tool(
        "create_mind", {"mind_id": "test_mind_001", "config": test_mind_config.model_dump()}
    )

    assert "test_mind_001" in mcp_server.minds

    # Cleanup
    result = await mcp_server.mcp.call_tool("cleanup_mind", {"mind_id": "test_mind_001"})

    # Parse response from TextContent list
    response = json.loads(result[0].text)

    assert response["status"] == "removed"
    assert "test_mind_001" not in mcp_server.minds


@pytest.mark.asyncio
async def test_full_workflow(mcp_server, test_mind_config, test_observation):
    """Test complete workflow: create, decide, consolidate, cleanup"""
    # Step 1: Create mind
    result = await mcp_server.mcp.call_tool(
        "create_mind", {"mind_id": "test_mind_001", "config": test_mind_config.model_dump()}
    )
    create_response = json.loads(result[0].text)
    assert create_response["status"] == "created"

    # Step 2: Make a decision
    result = await mcp_server.mcp.call_tool(
        "decide_action", {"mind_id": "test_mind_001", "observation": test_observation.model_dump()}
    )
    action_response = json.loads(result[0].text)
    assert action_response["status"] == "success"
    assert action_response["action"] is not None

    # Step 3: Check if memories were formed
    mind = mcp_server.minds["test_mind_001"]
    memory_count = len(mind.daily_memories)

    # Step 4: Consolidate if there are memories
    if memory_count > 0:
        result = await mcp_server.mcp.call_tool(
            "consolidate_memories", {"mind_id": "test_mind_001"}
        )
        consolidate_response = json.loads(result[0].text)
        assert consolidate_response["status"] == "success"
        assert consolidate_response["consolidated_count"] == memory_count

    # Step 5: Cleanup
    result = await mcp_server.mcp.call_tool("cleanup_mind", {"mind_id": "test_mind_001"})
    cleanup_response = json.loads(result[0].text)
    assert cleanup_response["status"] == "removed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
