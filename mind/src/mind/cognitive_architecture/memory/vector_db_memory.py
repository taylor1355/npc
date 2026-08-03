"""Simple memory store using ChromaDB for vector storage"""

import os

import chromadb
from chromadb.errors import NotFoundError
from pydantic import BaseModel, ConfigDict, Field
from sentence_transformers import SentenceTransformer

from ..id_generator import IdGenerator
from .models import Memory


class VectorDBMetadata(BaseModel):
    """Metadata stored with each memory in ChromaDB"""

    importance: float
    timestamp: int | None = None
    location_x: int | None = None
    location_y: int | None = None
    tags: list[str] = Field(default_factory=list)

    def get_location(self) -> tuple[int, int] | None:
        """Extract location tuple if both coordinates present"""
        if self.location_x is not None and self.location_y is not None:
            return (self.location_x, self.location_y)
        return None


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
    importance_weight: float = 0.3
    recency_weight: float = 0.2
    current_simulation_time: int | None = None
    tags: list[str] | None = None  # Filter to memories with ANY of these tags


class ChromaQueryResult(BaseModel):
    """Wrapper for ChromaDB query results with cleaner access"""

    ids: list[list[str]]
    documents: list[list[str]]
    metadatas: list[list[dict]]
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
    def first_query_metadatas(self) -> list[dict]:
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
        embedding_model: str = "all-MiniLM-L6-v2",
        storage_path: str | None = None,
    ):
        """Initialize vector database memory component

        Args:
            collection_name: Name of the ChromaDB collection
            embedding_model: SentenceTransformer model name for embeddings
            storage_path: Directory path for persistent storage (None = in-memory only)
        """
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
        collection is a no-op too - ChromaDB's delete_collection raises on a missing
        collection rather than no-op'ing, so the raise is absorbed here.

        Absorbing it directly is only safe because delete_collection raises a
        precise NotFoundError. It is not catchable in the general case: some
        ChromaDB releases signal the same condition with a bare ValueError, which
        is far too broad to swallow around a mutating call. Prefer this shape over
        a get_collection probe followed by an unguarded delete - the probe leaves a
        window in which a concurrent deleter can win the race, so the delete raises
        anyway and the idempotence this function promises does not hold.

        Args:
            storage_path: Directory path for persistent storage
            collection_name: Name of the ChromaDB collection to delete
        """
        if not storage_path or not os.path.exists(storage_path):
            return

        settings = chromadb.Settings(anonymized_telemetry=False, allow_reset=True)
        client = chromadb.PersistentClient(path=storage_path, settings=settings)
        try:
            client.delete_collection(collection_name)
        except NotFoundError:
            return

    def add_memory(
        self,
        content: str,
        importance: float = 1.0,
        timestamp: int | None = None,
        location: tuple[int, int] | None = None,
        tags: list[str] | None = None,
    ) -> Memory:
        """Add a memory to the store

        Args:
            content: Memory content text
            importance: Importance score (0.0-10.0)
            timestamp: Simulation timestamp (game ticks/frames)
            location: Grid coordinates (x, y)
            tags: Categorical tags for filtering
        """
        tag_list = tags or []

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
        )

        # Store in ChromaDB (empty arrays not allowed in metadata, so exclude them)
        metadata_dict = metadata.model_dump(exclude_none=True)
        if not metadata_dict.get("tags"):
            metadata_dict.pop("tags", None)

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

        raw_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(query.top_k, collection_count),
            where=where_clause,
            include=["documents", "metadatas", "distances"],
        )

        # Parse into typed model
        results = ChromaQueryResult(**raw_results)

        if not results.first_query_ids:
            return []

        # Convert results to Memory objects with combined scoring
        memories = []

        for memory_id, content, metadata_dict, distance in results.iter_first_query():
            # Parse metadata with type safety
            metadata = VectorDBMetadata.model_validate(metadata_dict)

            # Calculate combined score. The collection uses cosine space, so the
            # ChromaDB distance is (1 - cosine_similarity). Convert back to a
            # similarity in [0, 1] (clamped); fall back to 1.0 when a backend
            # omits distances so older/partial results still rank deterministically.
            if distance is None:
                similarity_score = 1.0
            else:
                similarity_score = max(0.0, min(1.0, 1.0 - distance))
            importance_score = metadata.importance / 10.0

            # Calculate recency score if we have timestamps
            if query.current_simulation_time is not None and metadata.timestamp is not None:
                time_delta = query.current_simulation_time - metadata.timestamp
                # Decay over simulation time (adjust decay rate as needed)
                recency_score = 1.0 / (1.0 + time_delta / 1000.0)
            else:
                recency_score = 1.0

            combined_score = (
                (1 - query.importance_weight - query.recency_weight) * similarity_score
                + query.importance_weight * importance_score
                + query.recency_weight * recency_score
            )

            memory = Memory(
                id=memory_id,
                content=content,
                timestamp=metadata.timestamp,
                importance=metadata.importance,
                location=metadata.get_location(),
                tags=metadata.tags,
            )
            memories.append((combined_score, memory))

        # Sort by combined score and return
        memories.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in memories[: query.top_k]]

    def clear(self):
        """Clear all memories"""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.create_collection(
            name=self.collection.name, metadata={"hnsw:space": "cosine"}
        )
        self._id_counter = 0
