"""Unit tests for VectorDBMemory"""

import pytest
from pydantic import ValidationError

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

    async def test_add_memory_with_tags(self, memory_store):
        """Should store and retrieve tags via round-trip"""
        memory = memory_store.add_memory(
            content="Architecture insight about pathfinding",
            importance=7.0,
            tags=["architecture", "pathfinding"],
        )
        assert memory.tags == ["architecture", "pathfinding"]

        query = VectorDBQuery(query="pathfinding", top_k=1)
        results = await memory_store.search(query)
        assert set(results[0].tags) == {"architecture", "pathfinding"}

    async def test_tag_filtered_search(self, memory_store):
        """Should filter results by tags when provided"""
        memory_store.add_memory(content="Social interaction at market", tags=["social"])
        memory_store.add_memory(content="Architecture of the castle", tags=["architecture"])
        memory_store.add_memory(content="Routine patrol route", tags=["routine"])

        query = VectorDBQuery(query="activity", top_k=10, tags=["social"])
        results = await memory_store.search(query)

        assert len(results) >= 1
        assert all("social" in m.tags for m in results)

    async def test_search_without_tags_returns_all(self, memory_store):
        """No tag filter = all results returned"""
        memory_store.add_memory(content="Tagged memory", tags=["test"])
        memory_store.add_memory(content="Untagged memory")

        query = VectorDBQuery(query="memory", top_k=10)
        results = await memory_store.search(query)
        assert len(results) == 2

    async def test_add_memory_without_tags(self, memory_store):
        """Backward compatible - no tags = empty list"""
        memory = memory_store.add_memory(content="No tags here")
        assert memory.tags == []

    async def test_multi_tag_or_filter(self, memory_store):
        """Should match memories with ANY of the requested tags, by exact membership.

        `antisocial` overlaps `social` as a substring and `architecture_review`
        overlaps `architecture`; both must be excluded. Without them the OR branch
        passes under either reading of $contains (membership or substring), so the
        overlapping tags are what make this test constrain the semantic rather than
        merely observe it.
        """
        memory_store.add_memory(content="Social debugging session", tags=["social", "debugging"])
        memory_store.add_memory(content="Architecture review", tags=["architecture"])
        memory_store.add_memory(content="Routine task", tags=["routine"])
        memory_store.add_memory(content="Avoided the crowd", tags=["antisocial"])
        memory_store.add_memory(content="Reviewed the architecture", tags=["architecture_review"])

        query = VectorDBQuery(query="work", top_k=10, tags=["social", "architecture"])
        results = await memory_store.search(query)

        result_tag_sets = [set(m.tags) for m in results]
        assert any("social" in ts for ts in result_tag_sets)
        assert any("architecture" in ts for ts in result_tag_sets)
        assert not any(ts == {"routine"} for ts in result_tag_sets)
        # Substring-overlapping tags must not be swept in by the OR filter.
        assert not any("antisocial" in ts for ts in result_tag_sets)
        assert not any("architecture_review" in ts for ts in result_tag_sets)

    async def test_untagged_memories_excluded_by_tag_filter(self, memory_store):
        """Tag filter should exclude memories with no tags"""
        memory_store.add_memory(content="Has tags", tags=["important"])
        memory_store.add_memory(content="No tags at all")

        query = VectorDBQuery(query="memory", top_k=10, tags=["important"])
        results = await memory_store.search(query)

        assert len(results) == 1
        assert results[0].tags == ["important"]

    async def test_tag_filter_is_exact_not_substring(self, memory_store):
        """`social` must not match `antisocial` - $contains on an array is membership, not substring."""
        memory_store.add_memory(content="Avoided the crowd", tags=["antisocial"])
        memory_store.add_memory(content="Chatted at the market", tags=["social"])

        results = await memory_store.search(
            VectorDBQuery(query="people", top_k=10, tags=["social"])
        )

        assert [m.tags for m in results] == [["social"]]

    async def test_unknown_query_field_is_rejected(self):
        """A misspelled/unsupported filter must raise, not silently run unfiltered.

        Regression for the silent-drop bug: with pydantic's default
        extra='ignore', VectorDBQuery(query=..., tags=[...]) built against a
        model without a `tags` field discarded the filter and returned an
        unfiltered result set indistinguishable from a filtered one. Constructing
        with a field the model does not declare must fail loudly.
        """
        with pytest.raises(ValidationError):
            VectorDBQuery(query="anything", top_k=5, tagz=["typo"])

    async def test_similarity_influences_ranking(self, memory_store):
        """High-similarity/low-importance should outrank low-similarity/high-importance.

        Regression for the bug where similarity_score was hardcoded to 1.0,
        making the combined score depend only on importance + recency. With a
        modest importance_weight, real semantic similarity must be able to flip
        the ranking in favor of the more relevant (but less important) memory.
        """
        # On-topic but low importance.
        memory_store.add_memory(
            content="The blacksmith forged a gleaming steel sword at the forge",
            importance=1.0,
        )
        # Off-topic but high importance.
        memory_store.add_memory(
            content="A gentle rain fell over the quiet meadow at dawn",
            importance=10.0,
        )

        # Modest importance weight: similarity should dominate.
        query = VectorDBQuery(
            query="forging a sword at the blacksmith forge",
            top_k=2,
            importance_weight=0.2,
            recency_weight=0.0,
        )
        results = await memory_store.search(query)

        assert len(results) == 2
        # The semantically relevant (low-importance) memory must rank first,
        # which is only possible once real similarity feeds the combined score.
        assert "sword" in results[0].content.lower()
        assert results[0].importance < results[1].importance


@pytest.fixture
def isolated_chroma(monkeypatch, tmp_path):
    """Isolate ChromaDB's process-global client cache and CWD per test, so a
    PersistentClient opened here can't alias another test's on-disk store."""
    from chromadb.api.client import SharedSystemClient

    SharedSystemClient.clear_system_cache()
    monkeypatch.chdir(tmp_path)
    yield
    SharedSystemClient.clear_system_cache()


@pytest.mark.usefixtures("isolated_chroma")
class TestDeleteCollection:
    """delete_collection is idempotent (delete-if-exists), with no encoder load."""

    def test_delete_absent_collection_on_existing_path_does_not_raise(self):
        """Deleting a missing collection from an existing storage path is a silent
        no-op, not a raise (ChromaDB's delete_collection raises on absent). The path
        must already exist (so the missing-path early-return doesn't mask the case),
        but the named collection must not - exercising the delete-if-exists guard."""
        import os

        storage_path = os.path.join(os.getcwd(), "chroma_persist")

        # Persist a DIFFERENT collection so the storage path exists on disk while the
        # target collection does not - isolating the absent-collection branch.
        VectorDBMemory(collection_name="present_one", storage_path=storage_path)
        assert os.path.exists(storage_path)
        assert VectorDBMemory.collection_exists(storage_path, "never_made") is False

        # Must not raise.
        VectorDBMemory.delete_collection(storage_path, "never_made")

        # Still absent (delete-if-exists left the world unchanged).
        assert VectorDBMemory.collection_exists(storage_path, "never_made") is False


@pytest.mark.usefixtures("isolated_chroma")
class TestClearGuards:
    """clear() promises an empty collection and a valid self.collection afterward.

    It must keep that promise when the collection it is clearing is already gone, and
    when a concurrent clear recreates the name in the window between its own delete
    and create. Both states are reachable: forget_mind deletes a collection out from
    under a still-resident store, and two clears can interleave.
    """

    def test_clear_tolerates_an_already_deleted_collection(self):
        """A store whose collection was deleted underneath it can still clear().

        The load-bearing assertions are the ones after the call: not raising is only
        half the contract, and absorbing the delete without recreating would trade a
        raise for a store whose self.collection points at nothing.
        """
        store = VectorDBMemory(collection_name="orphaned")
        store.add_memory(content="doomed", importance=5.0)

        # Model forget_mind deleting the collection out from under a resident store.
        store.client.delete_collection("orphaned")

        store.clear()

        assert store.collection.count() == 0
        store.add_memory(content="written after clear", importance=5.0)
        assert store.collection.count() == 1

    def test_clear_survives_a_concurrent_recreate_between_delete_and_create(self, monkeypatch):
        """The delete guard alone does not make clear() race-safe: a concurrent clear
        can finish its own delete+create in the window between ours, and then our
        recreate meets a name that already exists.

        The interleaving is forced deterministically by wrapping delete_collection -
        the wrapper performs the real delete and then plays the other thread's
        recreate before returning, which is exactly the state clear() resumes into.
        """
        store = VectorDBMemory(collection_name="raced")
        real_delete = store.client.delete_collection
        interleaved = []

        def delete_then_concurrent_recreate(name):
            real_delete(name)
            if interleaved:
                return
            interleaved.append(name)
            store.client.create_collection(name=name, metadata={"hnsw:space": "cosine"})

        monkeypatch.setattr(store.client, "delete_collection", delete_then_concurrent_recreate)

        store.clear()

        assert interleaved == ["raced"], "the race window was never exercised"
        assert store.collection.count() == 0
        store.add_memory(content="written after the race", importance=5.0)
        assert store.collection.count() == 1
