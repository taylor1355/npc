"""Unit tests for VectorDBMemory"""

import pytest

from mind.cognitive_architecture.memory.vector_db_memory import (
    VectorDBMemory,
    VectorDBQuery,
)


@pytest.mark.asyncio
class TestVectorDBMemory:
    """Test VectorDBMemory in isolation"""

    @pytest.fixture
    def memory_store(self):
        """Create a VectorDBMemory instance with test collection"""
        store = VectorDBMemory(collection_name="test_collection")
        yield store
        # Cleanup
        store.clear()

    async def test_add_and_search_memory(self, memory_store):
        """Should add memory and retrieve it via search"""
        # Add a memory
        memory_store.add_memory(content="Worked on sword at forge", importance=7.0, timestamp=100)

        # Search for it
        query = VectorDBQuery(query="forge work", top_k=5)
        results = await memory_store.search(query)

        # Should find the memory
        assert len(results) > 0
        assert any("sword" in m.content.lower() for m in results)

    async def test_search_returns_top_k_results(self, memory_store):
        """Should limit results to top_k"""
        # Add multiple memories
        for i in range(10):
            memory_store.add_memory(content=f"Memory number {i}", importance=5.0)

        # Search with top_k=3
        query = VectorDBQuery(query="memory", top_k=3)
        results = await memory_store.search(query)

        # Should only return 3 results
        assert len(results) <= 3

    async def test_importance_affects_retrieval(self, memory_store):
        """Should prioritize high-importance memories"""
        # Add low importance memory
        memory_store.add_memory(
            content="Routine blacksmith work",
            importance=2.0,
        )

        # Add high importance memory with similar content
        memory_store.add_memory(
            content="Important blacksmith discovery",
            importance=9.0,
        )

        # Search
        query = VectorDBQuery(query="blacksmith", top_k=2, importance_weight=0.5)
        results = await memory_store.search(query)

        # High importance should be ranked higher
        assert len(results) > 0
        # The high importance one should be first
        assert results[0].importance > results[1].importance

    async def test_recency_decay(self, memory_store):
        """Should apply recency decay to old memories"""
        # Add old memory
        memory_store.add_memory(
            content="Long ago blacksmith event",
            importance=8.0,
            timestamp=100,
        )

        # Add recent memory
        memory_store.add_memory(
            content="Recent blacksmith event",
            importance=8.0,
            timestamp=1000,
        )

        # Search with current time and high recency weight
        query = VectorDBQuery(
            query="blacksmith event",
            top_k=2,
            current_simulation_time=1000,
            recency_weight=0.5,
        )
        results = await memory_store.search(query)

        # Should retrieve both
        assert len(results) == 2
        # Recent one should be first due to recency weighting
        assert results[0].timestamp == 1000

    async def test_search_empty_store(self, memory_store):
        """Should return empty list when no memories exist"""
        query = VectorDBQuery(query="anything", top_k=5)
        results = await memory_store.search(query)

        assert results == []

    async def test_clear_removes_all_memories(self, memory_store):
        """Should remove all memories from the store"""
        # Add memories
        for i in range(5):
            memory_store.add_memory(content=f"Content {i}", importance=5.0)

        # Verify they exist
        query = VectorDBQuery(query="content", top_k=10)
        results = await memory_store.search(query)
        assert len(results) > 0

        # Clear
        memory_store.clear()

        # Verify empty
        results_after = await memory_store.search(query)
        assert results_after == []

    async def test_metadata_preserved(self, memory_store):
        """Should preserve metadata in stored memories"""
        # Add memory with metadata
        memory_store.add_memory(
            content="Event at forge",
            importance=7.0,
            timestamp=500,
            location=(10, 20),
        )

        # Retrieve
        query = VectorDBQuery(query="forge", top_k=1)
        results = await memory_store.search(query)

        # Check metadata preserved
        assert len(results) > 0
        retrieved = results[0]
        assert retrieved.timestamp == 500
        assert retrieved.location == (10, 20)
        assert retrieved.importance == 7.0

    async def test_add_memory_returns_memory_object(self, memory_store):
        """Should return Memory object with generated ID and embedding"""
        # Add a memory
        memory = memory_store.add_memory(
            content="Test memory content",
            importance=5.0,
            timestamp=100,
            location=(5, 10),
        )

        # Should return Memory with all fields populated
        assert memory.id is not None
        assert memory.content == "Test memory content"
        assert memory.importance == 5.0
        assert memory.timestamp == 100
        assert memory.location == (5, 10)
        assert memory.embedding is not None
        assert len(memory.embedding) > 0

    async def test_memory_ids_are_stable(self, memory_store):
        """Should preserve memory IDs across retrievals"""
        # Add a memory
        added_memory = memory_store.add_memory(content="Test memory content")
        original_id = added_memory.id

        # Retrieve it
        query = VectorDBQuery(query="Test memory", top_k=1)
        results = await memory_store.search(query)

        assert len(results) == 1
        assert results[0].id == original_id
