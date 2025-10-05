"""Simple memory store using ChromaDB for vector storage"""

import time
from typing import Optional
import chromadb
from sentence_transformers import SentenceTransformer

from ..models import Memory


class MemoryStore:
    """Vector-based memory storage using ChromaDB"""

    def __init__(
        self,
        collection_name: str = "memories",
        model_name: str = "all-MiniLM-L6-v2",
        persist_directory: Optional[str] = None
    ):
        # Initialize embedding model
        self.encoder = SentenceTransformer(model_name)

        # Initialize ChromaDB with telemetry disabled
        settings = chromadb.Settings(
            anonymized_telemetry=False,
            allow_reset=True
        )

        if persist_directory:
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=settings
            )
        else:
            self.client = chromadb.EphemeralClient(settings=settings)

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        self._id_counter = 0

    def add_memory(self, content: str, importance: float = 1.0) -> Memory:
        """Add a memory to the store"""

        # Create memory object
        memory = Memory(
            content=content,
            timestamp=time.time(),
            importance=importance
        )

        # Generate embedding
        embedding = self.encoder.encode(content).tolist()
        memory.embedding = embedding

        # Store in ChromaDB
        self._id_counter += 1
        self.collection.add(
            ids=[str(self._id_counter)],
            embeddings=[embedding],
            documents=[content],
            metadatas=[{
                "timestamp": memory.timestamp,
                "importance": memory.importance
            }]
        )

        return memory

    async def search(
        self,
        query: str,
        top_k: int = 5,
        importance_weight: float = 0.3,
        recency_weight: float = 0.2
    ) -> list[Memory]:
        """Search for memories using semantic similarity"""

        # Generate query embedding
        query_embedding = self.encoder.encode(query).tolist()

        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count())
        )

        if not results["ids"] or not results["ids"][0]:
            return []

        # Convert results to Memory objects
        memories = []
        current_time = time.time()

        for i, doc in enumerate(results["documents"][0]):
            metadata = results["metadatas"][0][i]

            # Calculate combined score
            similarity_score = 1.0  # ChromaDB returns sorted by similarity
            importance_score = metadata.get("importance", 1.0) / 10.0
            age_hours = (current_time - metadata.get("timestamp", current_time)) / 3600
            recency_score = 1.0 / (1.0 + age_hours / 24)  # Decay over days

            combined_score = (
                (1 - importance_weight - recency_weight) * similarity_score +
                importance_weight * importance_score +
                recency_weight * recency_score
            )

            memory = Memory(
                content=doc,
                timestamp=metadata.get("timestamp"),
                importance=metadata.get("importance", 1.0)
            )
            memories.append((combined_score, memory))

        # Sort by combined score and return
        memories.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in memories[:top_k]]

    def clear(self):
        """Clear all memories"""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"}
        )
        self._id_counter = 0