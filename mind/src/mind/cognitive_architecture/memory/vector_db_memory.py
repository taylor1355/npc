"""Simple memory store using ChromaDB for vector storage"""

import chromadb
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from ..id_generator import IdGenerator
from ..models import Memory


class VectorDBMetadata(BaseModel):
    """Metadata stored with each memory in ChromaDB"""

    importance: float
    timestamp: int | None = None
    location_x: int | None = None
    location_y: int | None = None

    def get_location(self) -> tuple[int, int] | None:
        """Extract location tuple if both coordinates present"""
        if self.location_x is not None and self.location_y is not None:
            return (self.location_x, self.location_y)
        return None


class VectorDBQuery(BaseModel):
    """Query parameters for vector database search"""

    query: str
    top_k: int = 5
    importance_weight: float = 0.3
    recency_weight: float = 0.2
    current_simulation_time: int | None = None


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

    def iter_first_query(self):
        """Iterate over (id, document, metadata) tuples for first query"""
        return zip(self.first_query_ids, self.first_query_documents, self.first_query_metadatas)


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

    def add_memory(
        self,
        content: str,
        importance: float = 1.0,
        timestamp: int | None = None,
        location: tuple[int, int] | None = None,
    ) -> Memory:
        """Add a memory to the store

        Args:
            content: Memory content text
            importance: Importance score (0.0-10.0)
            timestamp: Simulation timestamp (game ticks/frames)
            location: Grid coordinates (x, y)
        """

        # Generate memory ID
        memory_id = IdGenerator.generate_memory_id()

        # Create memory object
        memory = Memory(
            id=memory_id,
            content=content,
            timestamp=timestamp,
            importance=importance,
            location=location,
        )

        # Generate embedding
        embedding = self.encoder.encode(content, show_progress_bar=False).tolist()
        memory.embedding = embedding

        metadata = VectorDBMetadata(
            importance=importance,
            timestamp=timestamp,
            location_x=location[0] if location else None,
            location_y=location[1] if location else None,
        )

        # Store in ChromaDB
        self.collection.add(
            ids=[memory_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[metadata.model_dump(exclude_none=True)],
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

        raw_results = self.collection.query(
            query_embeddings=[query_embedding], n_results=min(query.top_k, collection_count)
        )

        # Parse into typed model
        results = ChromaQueryResult(**raw_results)

        if not results.first_query_ids:
            return []

        # Convert results to Memory objects with combined scoring
        memories = []

        for memory_id, content, metadata_dict in results.iter_first_query():
            # Parse metadata with type safety
            metadata = VectorDBMetadata.model_validate(metadata_dict)

            # Calculate combined score
            similarity_score = 1.0  # ChromaDB returns sorted by similarity
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
