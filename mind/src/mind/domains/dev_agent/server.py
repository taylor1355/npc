"""Dev-agent MCP server — lightweight memory server for Claude Code integration"""

import json
from pathlib import Path

from fastmcp import FastMCP

from mind.cognitive_architecture.memory.vector_db_memory import VectorDBMemory, VectorDBQuery
from mind.constants import DEFAULT_MEMORY_STORAGE_PATH


class DevAgentMCPServer:
    """MCP server providing memory storage and retrieval for dev agents.

    Wraps VectorDBMemory with MCP tools for storing, searching,
    and inspecting memories. No LLM dependency — only embeddings.
    """

    def __init__(
        self,
        name: str = "Dev Agent Memory Server",
        collection_name: str = "partner",
        storage_path: str | None = None,
    ):
        if storage_path is None:
            storage_path = str(Path(DEFAULT_MEMORY_STORAGE_PATH))

        self.memory = VectorDBMemory(
            collection_name=collection_name,
            storage_path=storage_path,
        )
        self.mcp = FastMCP(name)
        self._register_tools_and_resources()

    def _register_tools_and_resources(self):
        """Register all tools and resources with MCP"""

        @self.mcp.tool()
        async def store_memory(
            content: str,
            importance: float = 5.0,
            tags: list[str] | None = None,
        ) -> dict:
            """Store a memory with optional tags and importance score.

            Args:
                content: The memory content to store (will be embedded for semantic search)
                importance: Importance score from 0.0 to 10.0 (default 5.0)
                tags: Optional tags for categorization and filtered retrieval

            Returns:
                dict with memory id and status
            """
            memory = self.memory.add_memory(
                content=content,
                importance=importance,
                tags=tags,
            )
            return {
                "status": "stored",
                "memory_id": memory.id,
                "tags": memory.tags,
                "importance": memory.importance,
            }

        @self.mcp.tool()
        async def search_memory(
            query: str,
            top_k: int = 5,
            tags: list[str] | None = None,
            importance_weight: float = 0.3,
            recency_weight: float = 0.2,
        ) -> list[dict]:
            """Semantic search across memories, optionally filtered by tags.

            Tags use OR semantics: memories matching ANY of the provided tags are returned.
            Results are ranked by combined similarity, importance, and recency scores.

            Args:
                query: Natural language search query
                top_k: Maximum number of results (default 5)
                tags: Optional tag filter (OR semantics)
                importance_weight: Weight for importance in scoring (default 0.3)
                recency_weight: Weight for recency in scoring (default 0.2)

            Returns:
                List of matching memories with content, tags, and importance
            """
            db_query = VectorDBQuery(
                query=query,
                top_k=top_k,
                tags=tags,
                importance_weight=importance_weight,
                recency_weight=recency_weight,
            )
            results = await self.memory.search(db_query)
            return [
                {
                    "memory_id": m.id,
                    "content": m.content,
                    "importance": m.importance,
                    "tags": m.tags,
                }
                for m in results
            ]

        @self.mcp.tool()
        async def consolidate() -> dict:
            """Return memory collection statistics. Full consolidation (clustering,
            reflection) is a future feature — this tool reports stats only.

            Returns:
                dict with collection count and status
            """
            count = self.memory.collection.count()
            return {
                "status": "stats",
                "total_memories": count,
                "note": "Cluster-based consolidation not yet implemented. See backlog.",
            }

        @self.mcp.resource("memory://stats")
        async def get_memory_stats() -> str:
            """Memory collection statistics: count, collection name."""
            count = self.memory.collection.count()
            return json.dumps(
                {
                    "total_memories": count,
                    "collection_name": self.memory.collection.name,
                },
                indent=2,
            )
