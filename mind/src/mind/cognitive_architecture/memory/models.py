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

        This is a prompt surface: retrieved memories are rendered through it into the
        memory-query, cognitive-update, and action-selection prompts. The tags segment
        below is inert today because nothing populates Memory.tags, so the first commit
        that wires a producer silently changes prompt content for all three LLM nodes.
        Wiring is NPC-1013.
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
