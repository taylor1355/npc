"""Input/output models for memory query generation node"""

from pydantic import BaseModel, Field


class MemoryQueryInput(BaseModel):
    """Input for memory query generation"""

    working_memory: str = Field(description="Current working memory content")
    observation: str = Field(description="Current observation from environment")


class MemoryQueryOutput(BaseModel):
    """Output from memory query generation"""

    queries: list[str] = Field(
        description="List of diverse queries to search memory with", min_length=1, max_length=5
    )
