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

    def __str__(self) -> str:
        """Format memory for LLM consumption"""
        parts = [f"[{self.id}"]

        if self.timestamp is not None:
            parts.append(f"T:{self.timestamp}")

        if self.location is not None:
            parts.append(f"L:{self.location}")

        header = " | ".join(parts) + "]"
        return f"{header} {self.content}"
