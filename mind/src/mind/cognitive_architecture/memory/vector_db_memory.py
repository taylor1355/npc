"""Simple memory store using ChromaDB for vector storage"""

import os

import chromadb
from chromadb.errors import NotFoundError
from pydantic import BaseModel, ConfigDict
from sentence_transformers import SentenceTransformer

from mind.constants import DEFAULT_EMBEDDING_MODEL, DEFAULT_RECENCY_REINFORCEMENT_ALPHA
from mind.logging_config import get_logger

from ..id_generator import IdGenerator
from .models import ImportanceScore, Memory, VectorDBMetadata
from .retrieval import (
    RetrievalContext,
    RetrievalWeights,
    ScoredCandidate,
    candidate_pool_size,
    default_terms,
    rank,
    reinforced_time,
    should_reinforce,
)

logger = get_logger()


def _delete_collection_if_exists(client: chromadb.ClientAPI, collection_name: str) -> None:
    """Delete a collection, treating "already gone" as success.

    ChromaDB's delete_collection raises on a missing collection rather than
    no-op'ing, so every caller wanting delete-if-exists has to absorb that raise.
    Absorbing it directly is safe because delete_collection signals this condition
    with a precise NotFoundError. Prefer this shape over a get_collection probe
    followed by an unguarded delete - the probe leaves a window in which a concurrent
    deleter can win the race, so the delete raises anyway and the idempotence the
    probe was meant to buy does not hold.
    """
    try:
        client.delete_collection(collection_name)
    except NotFoundError:
        pass


class VectorDBQuery(BaseModel):
    """Query parameters for vector database search

    Rejects unknown fields. Pydantic's default is extra='ignore', which makes a
    misspelled or not-yet-supported filter silently vanish: the search then runs
    unfiltered and returns a full, plausible-looking result set that the caller
    reads as filtered. A dropped filter is a wrong answer, not a missing one, so
    it must fail at construction rather than degrade into unfiltered data.
    """

    model_config = ConfigDict(extra="forbid")

    query: str
    top_k: int = 5

    # Per-query weight override. None uses the module defaults (Park's 1/1/1).
    # This replaces the previous importance_weight/recency_weight pair, which
    # expressed relevance only implicitly and could drive it negative; those
    # fields are removed rather than deprecated, and extra="forbid" above means
    # any straggler fails at construction instead of silently reverting.
    weights: RetrievalWeights | None = None

    current_simulation_time: int | None = None
    # Filter to memories with ANY of these tags. Storage layer only: nothing in the
    # cognitive pipeline sets this yet, and no node passes tags to add_memory, so
    # every stored memory is currently untagged. Producer/consumer wiring is NPC-1013.
    tags: list[str] | None = None


class ChromaQueryResult(BaseModel):
    """Wrapper for ChromaDB query results with cleaner access"""

    ids: list[list[str]]
    documents: list[list[str]]

    # Entries are None for rows stored with no metadata at all. ChromaDB refuses
    # to write an empty metadata dict, so a memory whose every field is unset
    # (never rated, never timestamped, untagged) is written without metadata and
    # reads back as None here - which is a faithful round-trip of "nothing was
    # recorded", not a loss.
    metadatas: list[list[dict | None]]

    distances: list[list[float]] | None = None

    @property
    def first_query_ids(self) -> list[str]:
        """Get IDs from first query result"""
        return self.ids[0] if self.ids else []

    @property
    def first_query_documents(self) -> list[str]:
        """Get documents from first query result"""
        return self.documents[0] if self.documents else []

    @property
    def first_query_metadatas(self) -> list[dict | None]:
        """Get metadatas from first query result"""
        return self.metadatas[0] if self.metadatas else []

    @property
    def first_query_distances(self) -> list[float | None]:
        """Get cosine distances from first query result, or Nones when absent"""
        if self.distances and self.distances[0]:
            return list(self.distances[0])
        return [None] * len(self.first_query_ids)

    def iter_first_query(self):
        """Iterate over (id, document, metadata, distance) tuples for first query"""
        return zip(
            self.first_query_ids,
            self.first_query_documents,
            self.first_query_metadatas,
            self.first_query_distances,
        )


class VectorDBMemory:
    """Vector-based memory storage using ChromaDB - a configurable component for memory systems

    This is a modular building block that can be used by task-specific memory systems
    (e.g., episodic memory, semantic memory). It handles vector embeddings, similarity
    search, and metadata storage.
    """

    def __init__(
        self,
        collection_name: str = "memories",
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        storage_path: str | None = None,
        recency_reinforcement_alpha: float = DEFAULT_RECENCY_REINFORCEMENT_ALPHA,
    ):
        """Initialize vector database memory component

        Args:
            collection_name: Name of the ChromaDB collection
            embedding_model: SentenceTransformer model name for embeddings
            storage_path: Directory path for persistent storage (None = in-memory only)
            recency_reinforcement_alpha: How strongly one retrieval pulls a
                memory's recency anchor toward the present. 1.0 reproduces
                Park's decay-from-last-retrieval exactly; 0.0 reproduces
                decay-from-creation exactly. See
                DEFAULT_RECENCY_REINFORCEMENT_ALPHA.
        """
        if not 0.0 <= recency_reinforcement_alpha <= 1.0:
            raise ValueError(
                "recency_reinforcement_alpha must lie in [0, 1] "
                f"(got {recency_reinforcement_alpha}); outside it, retrieval would push a "
                "memory's anchor past the present or backwards past its own creation"
            )
        self.recency_reinforcement_alpha = recency_reinforcement_alpha

        # Initialize embedding model
        self.encoder = SentenceTransformer(embedding_model)

        # Initialize ChromaDB with telemetry disabled
        settings = chromadb.Settings(anonymized_telemetry=False, allow_reset=True)

        if storage_path:
            self.client = chromadb.PersistentClient(path=storage_path, settings=settings)
        else:
            self.client = chromadb.EphemeralClient(settings=settings)

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    @staticmethod
    def collection_exists(storage_path: str, collection_name: str) -> bool:
        """Check whether a persisted collection exists without creating it.

        Opens a PersistentClient at storage_path and probes for the collection via
        get_collection, which raises NotFoundError when absent. Unlike the
        constructor (which uses get_or_create_collection), this is strictly
        read-only: it never creates the collection as a side effect, so it's safe
        for relink to decide whether retained memory survived a release.

        Args:
            storage_path: Directory path for persistent storage
            collection_name: Name of the ChromaDB collection to probe

        Returns:
            True if the collection exists, False otherwise
        """
        # PersistentClient(path=...) creates the directory if it's absent, so probing
        # a never-persisted path would leave an empty DB dir on disk. Short-circuit
        # before constructing the client to keep this probe truly read-only.
        if not storage_path or not os.path.exists(storage_path):
            return False

        settings = chromadb.Settings(anonymized_telemetry=False, allow_reset=True)
        client = chromadb.PersistentClient(path=storage_path, settings=settings)
        try:
            client.get_collection(name=collection_name)
            return True
        except NotFoundError:
            return False

    @staticmethod
    def delete_collection(storage_path: str, collection_name: str) -> None:
        """Delete a persisted collection without instantiating a full store.

        Mirrors collection_exists: opens a bare PersistentClient and deletes the
        named collection directly. Unlike constructing VectorDBMemory(...) to call
        .clear(), this loads no SentenceTransformer encoder and never calls
        get_or_create_collection (which would recreate the very collection we're
        about to delete). Use this for the non-resident forget path, where no live
        store exists.

        Idempotent (delete-if-exists): safe to call whether or not the path or the
        collection exists. A never-persisted path is a no-op, and an absent
        collection is a no-op too - see _delete_collection_if_exists for why the
        raise is absorbed rather than probed for.

        Args:
            storage_path: Directory path for persistent storage
            collection_name: Name of the ChromaDB collection to delete
        """
        if not storage_path or not os.path.exists(storage_path):
            return

        settings = chromadb.Settings(anonymized_telemetry=False, allow_reset=True)
        client = chromadb.PersistentClient(path=storage_path, settings=settings)
        _delete_collection_if_exists(client, collection_name)

    def add_memory(
        self,
        content: str,
        importance: ImportanceScore | None = None,
        timestamp: int | None = None,
        location: tuple[int, int] | None = None,
        tags: list[str] | None = None,
        subject_ids: list[str] | None = None,
    ) -> Memory:
        """Add a memory to the store

        Args:
            content: Memory content text
            importance: Poignancy on the 1-10 rubric, from the LLM that formed
                the memory. None means never rated - a caller with no rating must
                pass None rather than inventing one, because the retrieval
                scorer abstains on None and cannot tell an invented midpoint
                from a real rating.
            timestamp: Elapsed game minutes
                (SimulationTime.get_elapsed_game_minutes). None means unknown;
                do not substitute 0, which is a valid reading.
            location: Grid coordinates (x, y)
            tags: Categorical tags for filtering
            subject_ids: Entities this memory is about. Reserved for NPC-401 /
                NPC-411; no production caller sets it yet.
        """
        tag_list = tags or []
        subject_id_list = subject_ids or []

        # Generate memory ID
        memory_id = IdGenerator.generate_memory_id()

        # Create memory object
        memory = Memory(
            id=memory_id,
            content=content,
            timestamp=timestamp,
            importance=importance,
            location=location,
            tags=tag_list,
        )

        # Generate embedding
        embedding = self.encoder.encode(content, show_progress_bar=False).tolist()
        memory.embedding = embedding

        metadata = VectorDBMetadata(
            importance=importance,
            timestamp=timestamp,
            location_x=location[0] if location else None,
            location_y=location[1] if location else None,
            tags=tag_list,
            subject_ids=subject_id_list,
            # The EMA is seeded at creation; retrieval pulls it toward the
            # present from there.
            effective_time=timestamp,
            last_accessed=timestamp,
        )

        # Store in ChromaDB (empty arrays not allowed in metadata, so exclude them)
        metadata_dict = metadata.model_dump(exclude_none=True)
        if not metadata_dict.get("tags"):
            metadata_dict.pop("tags", None)
        if not metadata_dict.get("subject_ids"):
            metadata_dict.pop("subject_ids", None)

        # VectorDBMetadata.schema_version is non-optional, so exclude_none always
        # leaves at least one key. That is what keeps this dict non-empty, and it
        # must stay that way: ChromaDB rejects an empty metadata dict, and
        # passing metadatas=None instead makes the row read back carrying a
        # deleted row's metadata. See VectorDBMetadata.schema_version.
        if not metadata_dict:
            raise ValueError(
                f"refusing to write memory {memory_id} with empty metadata - "
                "ChromaDB would return another row's metadata for it"
            )

        self.collection.add(
            ids=[memory_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[metadata_dict],
        )

        return memory

    async def search(self, query: VectorDBQuery) -> list[Memory]:
        """Search for memories using semantic similarity"""

        # Generate query embedding
        query_embedding = self.encoder.encode(query.query, show_progress_bar=False).tolist()

        # Search in ChromaDB
        collection_count = self.collection.count()
        if collection_count == 0:
            return []

        # Build tag filter using ChromaDB's native $contains operator
        where_clause = None
        if query.tags:
            if len(query.tags) == 1:
                where_clause = {"tags": {"$contains": query.tags[0]}}
            else:
                where_clause = {"$or": [{"tags": {"$contains": t}} for t in query.tags]}

        # Over-fetch by cosine, then score the pool. Fetching exactly top_k would
        # make the weighted score a reranker over the cosine top-k - see
        # retrieval.candidate_pool_size for why that is not a retrieval formula.
        raw_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=candidate_pool_size(query.top_k, collection_count),
            where=where_clause,
            include=["documents", "metadatas", "distances"],
        )

        # Parse into typed model
        results = ChromaQueryResult(**raw_results)

        if not results.first_query_ids:
            return []

        candidates = [
            ScoredCandidate(
                memory_id=memory_id,
                content=content,
                # None means the row was stored with no metadata at all; that is
                # an all-unset record, not a parse failure.
                metadata=VectorDBMetadata.model_validate(metadata_dict or {}),
                distance=distance,
            )
            for memory_id, content, metadata_dict, distance in results.iter_first_query()
        ]

        # All scoring arithmetic lives in retrieval.py, which imports no storage
        # backend - that is what makes the formula testable as a pure function.
        context = RetrievalContext(
            query=query.query, current_simulation_time=query.current_simulation_time
        )
        ranked = rank(
            candidates,
            context,
            query.weights or RetrievalWeights(),
            default_terms(),
            query.top_k,
        )

        self._reinforce_retrieved(
            [candidate for _, candidate in ranked], query.current_simulation_time
        )

        return [
            Memory(
                id=candidate.memory_id,
                content=candidate.content,
                timestamp=candidate.metadata.timestamp,
                importance=candidate.metadata.importance,
                location=candidate.metadata.get_location(),
                tags=candidate.metadata.tags,
            )
            for _, candidate in ranked
        ]

    def _reinforce_retrieved(self, candidates: list[ScoredCandidate], now: int | None) -> None:
        """Pull each returned memory's recency anchor toward the present.

        This is what makes recency measure how persistently a memory has
        mattered rather than only when it was formed. One metadata update per
        returned memory per query - no LLM call, no re-embedding. ChromaDB's
        `update` MERGES the keys given with those already stored (verified
        against 1.5.9), so passing only the two changed fields cannot drop
        importance, tags or the creation timestamp.

        Failure is logged, never swallowed: a silently-failed write means the
        memory stops aging correctly for the rest of its life, and every later
        retrieval would look entirely normal. Retrieval itself still succeeds -
        the caller asked for memories and we have them - so this reports and
        continues rather than raising.
        """
        if now is None:
            return

        ids: list[str] = []
        metadatas: list[dict] = []
        for candidate in candidates:
            anchor = candidate.metadata.recency_anchor
            if not should_reinforce(anchor, now):
                continue
            ids.append(candidate.memory_id)
            metadatas.append(
                {
                    "effective_time": reinforced_time(
                        anchor, now, self.recency_reinforcement_alpha
                    ),
                    "last_accessed": now,
                }
            )

        if not ids:
            return

        try:
            self.collection.update(ids=ids, metadatas=metadatas)
        except Exception:
            logger.exception(
                f"Failed to reinforce recency for {len(ids)} retrieved memories "
                f"({', '.join(ids)}). Their recency anchors are now stale, so they will "
                "age as if this retrieval never happened."
            )

    def drop_collection(self) -> None:
        """Delete this store's collection, treating "already gone" as success.

        The destructive counterpart to clear(): nothing is recreated, so self.collection
        is left pointing at a deleted collection. Callers that keep using the store must
        reassign it - clear() does; forget_mind, the external caller, is discarding the
        store anyway.
        Exposed so callers outside this module (forget_mind) can drop a live store's
        collection without reaching through .client and importing chromadb's exception
        types to guard it.
        """
        _delete_collection_if_exists(self.client, self.collection.name)

    def clear(self):
        """Clear all memories, leaving an empty collection behind.

        Both halves are guarded, because both can lose the same race. A live store can
        outlive its collection (forget_mind deletes it out from under a resident store),
        so the delete tolerates an absent collection; and a concurrent clear can complete
        its own delete+create in the window between ours, so the recreate has to tolerate
        the name already existing - create_collection raises InternalError on a duplicate
        name, which is far too broad to swallow around a mutating call. get_or_create is
        the right shape here for the same reason it is the wrong shape in
        delete_collection: there recreating would defeat the purpose, here recreating is
        the purpose.

        Under that race the collection handed back is the concurrent clear's rather than
        ours. That is still an empty collection under the name, which is what this method
        promises; what it must not do is leave self.collection pointing at a deleted
        collection, which is what letting the raise escape would do.
        """
        # Read the name before the drop: afterward self.collection refers to a deleted
        # collection, and this method's job is to stop depending on it.
        collection_name = self.collection.name
        self.drop_collection()
        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )
