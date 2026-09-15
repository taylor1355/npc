"""Working-memory models shared across the cognitive architecture.

WorkingMemory is the mind's persistent between-cycles state and NewMemory the
unit of memory formation. They are consumed well beyond the node that writes
them (pipeline state, the MCP wire models, the Mind runtime), so they live at
the architecture level rather than inside any single node package.

FormedMemory is NewMemory plus the circumstances it was formed under. The split
is deliberate and is explained on the class: NewMemory is the LLM's output
schema, so anything added there is something the model is asked to produce.
"""

from pydantic import BaseModel, Field

from mind.cognitive_architecture.observations.models import StatusObservation


class NewMemory(BaseModel):
    """A new memory to be stored from this decision cycle"""

    content: str = Field(description="The memory content")
    importance: float = Field(description="Importance score (1-10)", ge=1.0, le=10.0)


class FormedMemory(NewMemory):
    """A NewMemory with the circumstances of its FORMATION attached.

    Deliberately NOT fields on ``NewMemory``: that model is the LLM's structured
    output schema (``ReflectionOutput.new_memories`` feeds
    ``PydanticOutputParser``, whose ``get_format_instructions()`` derives the
    JSON schema shown in the prompt). A field added there is a field the model is
    ASKED TO FILL, and a fabricated zone id is strictly worse than none - it is a
    default impersonating data, arriving with an LLM's confidence and no way
    downstream to tell it from a real reading. These are stamped by the
    substrate-facing code from the live observation, never by the model.

    **Formation, not consolidation** (NPC-1476). Consolidation runs once per day
    over a whole batch, so reading the location there stamped every memory in the
    day with the single cell the NPC happened to occupy when the batch was
    written. Stamping at formation is the only point at which "where the NPC was
    when this happened" is still true.

    Both fields are None-by-absence, all the way down: no status on the
    observation, or no place known underfoot, yields None, which reaches
    ``VectorDBMetadata`` as None, which ``model_dump(exclude_none=True)`` drops,
    which ``SpatialTerm`` abstains on. Never ``""``, never a sentinel zone.
    """

    formed_at_position: tuple[int, int] | None = None
    formed_in_zone_id: str | None = None

    @classmethod
    def stamp(cls, memory: NewMemory, status: StatusObservation | None) -> "FormedMemory":
        """Attach the circumstances in ``status`` to ``memory``.

        Takes the status block rather than the whole ``Observation`` so this
        module keeps its single, leaf-ward import: ``observations.models``
        imports nothing from here, and narrowing the argument keeps it that way
        as both models grow.
        """
        return cls(
            content=memory.content,
            importance=memory.importance,
            formed_at_position=status.position if status else None,
            formed_in_zone_id=status.current_zone_id if status else None,
        )


class WorkingMemory(BaseModel):
    """Structured working memory (flexible, extensible)"""

    model_config = {"extra": "allow"}

    situation_assessment: str = ""
    active_goals: list[str] = Field(default_factory=list)
    recent_events: list[str] = Field(default_factory=list)
    current_plan: list[str] = Field(default_factory=list)
    emotional_state: str = ""

    def __str__(self) -> str:
        """Format working memory for LLM consumption"""
        parts = []
        if self.situation_assessment:
            parts.append(f"Situation: {self.situation_assessment}")
        if self.active_goals:
            parts.append(f"Active Goals: {', '.join(self.active_goals)}")
        if self.recent_events:
            parts.append(f"Recent Events: {', '.join(self.recent_events)}")
        if self.current_plan:
            parts.append(f"Current Plan: {', '.join(self.current_plan)}")
        if self.emotional_state:
            parts.append(f"Emotional State: {self.emotional_state}")
        return "\n".join(parts) if parts else "No working memory"
