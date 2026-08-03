"""Memory models for the cognitive architecture"""

from pydantic import BaseModel, Field


class Memory(BaseModel):
    """A single memory with metadata"""

    id: str
    content: str
    timestamp: int | None = None  # Simulation timestamp (game ticks/frames)
    importance: float = Field(default=1.0, ge=0.0, le=10.0)
    embedding: list[float] | None = None
    location: tuple[int, int] | None = None  # Grid coordinates (x, y)
    tags: list[str] = Field(default_factory=list)

    def __str__(self) -> str:
        """Format memory for LLM consumption.

        This is a prompt surface, and it has exactly one render site: the
        cognitive-update node joins retrieved memories through it into its prompt
        (nodes/cognitive_update/node.py). No other node renders a Memory - the rest
        take working-memory text - so tags reach them only indirectly, via whatever
        cognitive-update writes back. The tags segment below is inert today because
        nothing populates Memory.tags, so the first commit that wires a producer
        changes that prompt's content without touching this file. Wiring is NPC-1013.
        """
        parts = [f"[{self.id}"]

        if self.timestamp is not None:
            parts.append(f"T:{self.timestamp}")

        if self.location is not None:
            parts.append(f"L:{self.location}")

        if self.tags:
            parts.append(f"tags:{','.join(self.tags)}")

        header = " | ".join(parts) + "]"
        return f"{header} {self.content}"
