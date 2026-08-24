"""Test memory deduplication in the retrieval node"""

import pytest

from mind.cognitive_architecture.memory.vector_db_memory import VectorDBMemory, VectorDBQuery
from mind.cognitive_architecture.observations import Observation, StatusObservation
from mind.cognitive_architecture.nodes.memory_retrieval.node import MemoryRetrievalNode
from mind.cognitive_architecture.state import PipelineState


@pytest.mark.asyncio
async def test_memory_deduplication():
    """Test that duplicate memories from multiple queries are deduplicated"""

    # Create in-memory vector store
    memory_store = VectorDBMemory(collection_name="test_dedup")
    memory_store.clear()

    # Add test memories
    memory_store.add_memory("The apprentice is learning quickly")
    memory_store.add_memory("Working on a sword commission")
    memory_store.add_memory("The forge needs more coal")

    # Create retrieval node
    retrieval_node = MemoryRetrievalNode(memory_store, memories_per_query=2)

    # Create test state with queries that will return overlapping results
    state = PipelineState(
        observation=Observation(
            entity_id="test_npc",
            current_simulation_time=100,
            status=StatusObservation(position=(0, 0), movement_locked=False),
        ),
        memory_queries=[
            "apprentice learning skills",  # Should match memory 1
            "apprentice training progress",  # Should also match memory 1
            "sword crafting work",  # Should match memory 2
        ],
    )

    # Process retrieval
    result_state = await retrieval_node.process(state)

    # Check that memories are deduplicated
    memory_ids = [m.id for m in result_state.retrieved_memories]
    unique_ids = set(memory_ids)

    # Should have no duplicates
    assert len(memory_ids) == len(unique_ids), f"Found duplicate memory IDs: {memory_ids}"

    # Each memory should have a unique ID
    for memory in result_state.retrieved_memories:
        assert memory.id.startswith(
            "memory_"
        ), f"Memory ID should start with 'memory_' but got: {memory.id}"

    print(f"✓ Retrieved {len(result_state.retrieved_memories)} unique memories")
    print(f"✓ All memories have unique IDs: {memory_ids}")

    # Cleanup
    memory_store.clear()


@pytest.mark.asyncio
async def test_memory_ids_are_stable():
    """Test that memory IDs persist across retrievals"""

    memory_store = VectorDBMemory(collection_name="test_stable_ids")
    memory_store.clear()

    # Add a memory
    added_memory = memory_store.add_memory("Test memory content")
    original_id = added_memory.id

    # Retrieve it
    query = VectorDBQuery(query="Test memory", top_k=1)
    results = await memory_store.search(query)

    assert len(results) == 1
    assert results[0].id == original_id, f"Memory ID changed from {original_id} to {results[0].id}"

    print(f"✓ Memory ID is stable: {original_id}")

    # Cleanup
    memory_store.clear()


if __name__ == "__main__":
    import asyncio

    print("Running memory deduplication tests...\n")

    asyncio.run(test_memory_deduplication())
    print()
    asyncio.run(test_memory_ids_are_stable())

    print("\n✓ All tests passed!")
