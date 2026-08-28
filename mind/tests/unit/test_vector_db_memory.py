"""Unit tests for VectorDBMemory"""

import uuid

import pytest
from pydantic import ValidationError

from mind.cognitive_architecture.memory.retrieval import RetrievalWeights
from mind.cognitive_architecture.memory.vector_db_memory import (
    VectorDBMemory,
    VectorDBQuery,
)


@pytest.mark.asyncio
class TestVectorDBMemory:
    """Test VectorDBMemory in isolation"""

    @pytest.fixture
    def memory_store(self):
        """A VectorDBMemory whose state cannot reach any other test.

        Isolation used to be a shared collection name plus a teardown `clear()`,
        and that is not sufficient in either half.

        On chromadb 1.5.9, **deleting a collection poisons collections created
        afterwards**: a row added later reads back with a deleted row's metadata
        merged into its own, and it happens across differently-named collections
        in the same client (a control without any delete stays clean). So one
        test's `clear()` - whether from this teardown or from
        `test_clear_removes_all_memories` - could make a later test see tags no
        memory in it ever had.

        Two changes, because either alone is insufficient: a per-test collection
        name removes the shared-state half, and resetting chromadb's
        process-global system cache around every test discards the poisoned
        state that a delete leaves behind. Teardown no longer deletes anything;
        the cache reset is what reclaims it.
        """
        from chromadb.api.client import SharedSystemClient

        SharedSystemClient.clear_system_cache()
        store = VectorDBMemory(collection_name=f"test_collection_{uuid.uuid4().hex[:12]}")
        yield store
        SharedSystemClient.clear_system_cache()

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

        # Search. Importance weighted above relevance, as the removed
        # importance_weight=0.5 did (it implied a relevance coefficient of 0.5
        # against an importance coefficient of 0.5, with recency taking 0.2 -
        # the ratio, not the absolute numbers, is what this test ever depended on).
        query = VectorDBQuery(
            query="blacksmith",
            top_k=2,
            weights=RetrievalWeights(relevance=1.0, importance=2.0, recency=1.0),
        )
        results = await memory_store.search(query)

        # High importance should be ranked higher
        assert len(results) > 0
        # The high importance one should be first
        assert results[0].importance > results[1].importance

    async def test_recency_decay(self, memory_store):
        """Should apply recency decay to old memories.

        Ages are stated in game minutes and the gap is deliberately large: the
        curve is now Park's exponential (0.995 per game HOUR) rather than the
        hyperbolic 1/(1 + delta/1000) it replaced, so the previous 900-minute gap
        was only 15 game hours - about a 7% difference, which is correct
        behaviour for two memories from the same day but too thin to constrain
        anything. 1500 game hours apart is a real forgetting-curve difference.
        """
        # Roughly 62 game days old.
        memory_store.add_memory(
            content="Long ago blacksmith event",
            importance=8.0,
            timestamp=10_000,
        )

        # Formed this instant.
        memory_store.add_memory(
            content="Recent blacksmith event",
            importance=8.0,
            timestamp=100_000,
        )

        query = VectorDBQuery(
            query="blacksmith event",
            top_k=2,
            current_simulation_time=100_000,
            weights=RetrievalWeights(relevance=1.0, importance=1.0, recency=2.0),
        )
        results = await memory_store.search(query)

        # Should retrieve both
        assert len(results) == 2
        # Recent one should be first due to recency weighting
        assert results[0].timestamp == 100_000

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

        # Modest importance weight: similarity should dominate. The removed
        # importance_weight=0.2 / recency_weight=0.0 pair implied a 0.8 relevance
        # coefficient against 0.2 importance - the same 4:1 ratio expressed here
        # explicitly, which is the point of the new model: relevance is a weight
        # you set, not a leftover.
        query = VectorDBQuery(
            query="forging a sword at the blacksmith forge",
            top_k=2,
            weights=RetrievalWeights(relevance=4.0, importance=1.0, recency=0.0),
        )
        results = await memory_store.search(query)

        assert len(results) == 2
        # The semantically relevant (low-importance) memory must rank first,
        # which is only possible once real similarity feeds the combined score.
        assert "sword" in results[0].content.lower()
        assert results[0].importance < results[1].importance

    async def test_a_high_importance_memory_cosine_missed_can_still_be_retrieved(
        self, memory_store
    ):
        """The candidate pool must be wider than top_k, or the score cannot promote.

        This is the test that proves the formula was structurally inert.
        `search()` asked ChromaDB for `n_results=min(top_k, count)`, so the
        candidate set WAS the cosine top-k and the weighted score could only
        reorder it. At the production top_k of 2 that meant sorting two items,
        and a memory cosine had not already surfaced was unreachable **at any
        weight** - which is why the importance and recency weights were close to
        decorative.

        Here a semantically distant memory of maximum importance sits behind six
        closer trivia. It is not in the cosine top 2, so it cannot be returned
        unless the pool is widened before scoring. Weighted heavily toward
        importance, it must come back first.
        """
        # Six on-topic trivia: the cosine top-k for this query.
        for i in range(6):
            memory_store.add_memory(
                content=f"I hammered another horseshoe at the forge, number {i}",
                importance=1.0,
            )

        # Off-topic, but the most significant thing that ever happened to her.
        memory_store.add_memory(
            content="My mother died in the winter fever and I held her hand at the end",
            importance=10.0,
        )

        query = VectorDBQuery(
            query="working at the forge on horseshoes",
            top_k=2,
            weights=RetrievalWeights(relevance=1.0, importance=3.0, recency=1.0),
        )
        results = await memory_store.search(query)

        assert len(results) == 2
        assert "mother" in results[0].content.lower(), (
            "a high-importance memory outside the cosine top-k must still be "
            f"retrievable; got {[m.content for m in results]}"
        )

    async def test_a_lived_memory_outranks_untimestamped_backstory(self, memory_store):
        """Regression for the recency inversion.

        Untimestamped memories used to score *perfect* recency (1.0). Config
        seeds are exactly the untimestamped case and lived memories exactly the
        timestamped one, so hardcoded backstory permanently outranked
        experience, by a margin that widened with playtime.

        Content is near-identical so relevance is effectively matched and
        recency is the only thing left to decide the order.
        """
        # The initial_long_term_memories path: no timestamp, never rated.
        memory_store.add_memory(content="Alice worked at the blacksmith forge", importance=5.0)

        # A lived memory, formed just now.
        memory_store.add_memory(
            content="Alice worked at the blacksmith forge today",
            importance=5.0,
            timestamp=100_000,
        )

        query = VectorDBQuery(
            query="Alice at the blacksmith forge",
            top_k=2,
            current_simulation_time=100_000,
        )
        results = await memory_store.search(query)

        assert len(results) == 2
        assert results[0].timestamp == 100_000, (
            "a memory that actually happened must outrank untimestamped backstory"
        )
        assert results[1].timestamp is None

    async def test_an_unrated_memory_reads_back_as_unrated(self, memory_store):
        """None round-trips as None, not as a fabricated default.

        ChromaDB refuses an empty metadata dict, so a memory with no importance,
        no timestamp and no tags is stored with no metadata at all. It must read
        back as all-unset rather than raising or acquiring defaults.
        """
        added = memory_store.add_memory(content="Something nobody ever rated")

        results = await memory_store.search(
            VectorDBQuery(query="Something nobody ever rated", top_k=10)
        )
        assert results

        # Located by id rather than by rank: this class shares one collection
        # across its tests, so asserting on results[0] would couple the check to
        # what the neighbouring tests left behind.
        match = next((m for m in results if m.id == added.id), None)
        assert match is not None
        assert match.importance is None
        assert match.timestamp is None
        assert match.tags == []

    @pytest.mark.xfail(
        reason=(
            "Upstream chromadb 1.5.9 defect, not our arithmetic: after "
            "delete_collection + get_or_create_collection under the same name, a "
            "newly added row reads back with a DELETED row's metadata merged into "
            "its own. Reproduced with the chromadb API alone, so nothing in this "
            "repo can fix it here. VectorDBMemory.clear() is the only path that "
            "reaches the state and has no production call site, so exposure today "
            "is test-isolation only - which the per-test collection name in the "
            "memory_store fixture now removes. Left as a documented xfail rather "
            "than deleted so the day chromadb fixes it is visible."
        ),
        strict=False,
    )
    async def test_an_unrated_memory_does_not_inherit_a_deleted_memorys_metadata(
        self, memory_store
    ):
        """Pins the upstream metadata-staleness defect described in the xfail marker.

        The sequence matters: write tagged memories, clear, then write an
        untagged unrated one. Without the clear it passes, which is why this
        surfaced as cross-test contamination rather than as a direct failure.
        """
        memory_store.add_memory(content="Avoided the crowd", tags=["antisocial"])
        memory_store.add_memory(content="Chatted at the market", tags=["social"])
        memory_store.clear()

        added = memory_store.add_memory(content="Something nobody ever rated")

        results = await memory_store.search(
            VectorDBQuery(query="Something nobody ever rated", top_k=10)
        )

        match = next((m for m in results if m.id == added.id), None)
        assert match is not None
        assert match.tags == [], f"inherited a deleted row's metadata: {match.tags}"
        assert match.importance is None


@pytest.mark.asyncio
class TestRecencyReinforcementPersistence:
    """Retrieval writes the reinforced anchor back to the store.

    The arithmetic itself is covered in
    tests/unit/memory/test_retrieval_scoring.py. What is at stake here is the
    plumbing: that the write happens, that it does not destroy neighbouring
    metadata, and that a failed write is not silent.
    """

    def _store(self, alpha):
        from chromadb.api.client import SharedSystemClient

        SharedSystemClient.clear_system_cache()
        return VectorDBMemory(
            collection_name=f"test_reinforce_{uuid.uuid4().hex[:12]}",
            recency_reinforcement_alpha=alpha,
        )

    def _stored_metadata(self, store, memory_id):
        return store.collection.get(ids=[memory_id], include=["metadatas"])["metadatas"][0]

    async def test_retrieval_moves_the_anchor_and_preserves_other_metadata(self):
        """ChromaDB's update MERGES rather than replaces - verified against 1.5.9.

        This is the load-bearing half: passing only the two changed keys must not
        drop importance, tags or the creation timestamp. Under replace semantics
        they would be blanked, and every later retrieval would abstain on
        importance while looking entirely normal.
        """
        store = self._store(alpha=0.3)
        added = store.add_memory(
            content="The bandit raid on the north bridge",
            importance=8.0,
            timestamp=0,
            tags=["danger"],
        )

        await store.search(
            VectorDBQuery(query="bandit raid north bridge", top_k=1, current_simulation_time=6000)
        )

        stored = self._stored_metadata(store, added.id)
        assert stored["effective_time"] == pytest.approx(1800.0, abs=1e-6)
        assert stored["last_accessed"] == 6000
        # Untouched by the partial update.
        assert stored["importance"] == 8.0
        assert stored["timestamp"] == 0
        assert stored["tags"] == ["danger"]

    async def test_a_repeatedly_recalled_memory_ends_more_recent_than_a_once_recalled_one(self):
        """Equal creation time; only retrieval count differs.

        Compared on the stored anchors rather than through one combined query, so
        the assertion is about the reinforcement and not about which phrasing the
        embedding model happened to prefer.
        """
        store = self._store(alpha=0.3)
        often = store.add_memory(content="The bandit raid on the north bridge", timestamp=0)
        once = store.add_memory(content="The harvest festival in the village square", timestamp=0)

        for _ in range(5):
            await store.search(
                VectorDBQuery(
                    query="bandit raid north bridge", top_k=1, current_simulation_time=6000
                )
            )
        await store.search(
            VectorDBQuery(
                query="harvest festival village square", top_k=1, current_simulation_time=6000
            )
        )

        often_anchor = self._stored_metadata(store, often.id)["effective_time"]
        once_anchor = self._stored_metadata(store, once.id)["effective_time"]

        assert often_anchor > once_anchor
        assert once_anchor < 6000, "one recall must not erase the memory's whole age"

    async def test_alpha_one_reproduces_last_access_decay_end_to_end(self):
        store = self._store(alpha=1.0)
        added = store.add_memory(content="The bandit raid on the north bridge", timestamp=0)

        await store.search(
            VectorDBQuery(query="bandit raid north bridge", top_k=1, current_simulation_time=6000)
        )

        assert self._stored_metadata(store, added.id)["effective_time"] == pytest.approx(6000.0)

    async def test_alpha_zero_reproduces_creation_time_decay_end_to_end(self):
        store = self._store(alpha=0.0)
        added = store.add_memory(content="The bandit raid on the north bridge", timestamp=0)

        await store.search(
            VectorDBQuery(query="bandit raid north bridge", top_k=1, current_simulation_time=6000)
        )

        assert self._stored_metadata(store, added.id)["effective_time"] == pytest.approx(0.0)

    async def test_a_reset_clock_does_not_drag_the_anchor_backwards(self):
        """Scenario restart under a retained collection - open question D-5.

        Writing an EMA update against a reset clock would persist a pulled-back
        anchor, and repeated retrieval would drag it below `now` until the
        clamp's warning stopped firing. Refusing the write keeps the damage
        read-time only.
        """
        store = self._store(alpha=0.3)
        added = store.add_memory(content="The bandit raid on the north bridge", timestamp=100_000)

        await store.search(
            VectorDBQuery(query="bandit raid north bridge", top_k=1, current_simulation_time=5)
        )

        assert self._stored_metadata(store, added.id)["effective_time"] == pytest.approx(100_000.0)

    async def test_a_failed_reinforcement_write_is_reported_not_swallowed(
        self, monkeypatch, caplog
    ):
        """A silently-failed write means the memory stops aging correctly forever.

        Retrieval itself must still succeed - the caller asked for memories and
        we have them - but the failure has to reach the log naming the memory, or
        the store degrades invisibly.
        """
        import logging

        store = self._store(alpha=0.3)
        added = store.add_memory(content="The bandit raid on the north bridge", timestamp=0)

        def explode(*args, **kwargs):
            raise RuntimeError("simulated ChromaDB write failure")

        monkeypatch.setattr(store.collection, "update", explode)

        with caplog.at_level(logging.ERROR, logger="mind"):
            results = await store.search(
                VectorDBQuery(
                    query="bandit raid north bridge", top_k=1, current_simulation_time=6000
                )
            )

        assert results, "retrieval must still return what it found"
        assert any(added.id in record.getMessage() for record in caplog.records), (
            "the failed write must be reported and name the memory it affected"
        )

    def test_an_out_of_range_alpha_is_rejected_at_construction(self):
        """Outside [0, 1] the update pushes the anchor past the present, or back
        beyond the memory's own creation. Reject where it is configured."""
        for bad_alpha in (-0.1, 1.1):
            with pytest.raises(ValueError):
                VectorDBMemory(
                    collection_name=f"test_bad_alpha_{uuid.uuid4().hex[:8]}",
                    recency_reinforcement_alpha=bad_alpha,
                )


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
