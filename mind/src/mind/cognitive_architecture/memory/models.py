"""Memory models for the cognitive architecture.

Deliberately free of storage-backend imports. VectorDBMetadata lives here rather
than beside the ChromaDB client because the retrieval scorer
(memory/retrieval.py) reads it, and that module must stay importable without
pulling in chromadb - which is what lets its tests exercise the formula as a
pure function instead of measuring embedding noise.
"""

from typing import Annotated

from pydantic import BaseModel, Field

# Importance is authored on a 1-10 scale; 0.0 is reachable only as a stored
# value, never from reflection (NewMemory constrains it to ge=1.0).
ImportanceScore = Annotated[float, Field(ge=0.0, le=10.0)]


class Memory(BaseModel):
    """A single memory with metadata"""

    id: str
    content: str

    # Elapsed game MINUTES, as reported by the simulation's
    # SimulationTime.get_elapsed_game_minutes() and carried across the wire on
    # Observation.current_simulation_time. Not ticks, not frames, not wall time:
    # the sim exposes a time scale and a pause toggle, so wall time and game time
    # diverge arbitrarily and only game minutes mean what the NPC would mean by
    # "hours ago".
    #
    # None means "unknown", and 0 does NOT: the game clock legitimately starts at
    # 0, so a zero sentinel could not be told apart from a memory formed in the
    # first game minute. Consumers must treat None as an abstention rather than
    # substituting a value - see memory/retrieval.py::RecencyTerm.
    timestamp: int | None = None

    # None means "never scored", which is a different fact from "scored low".
    # Reflection-written memories carry a real LLM poignancy rating; memories
    # written by any other path (config seeds, for instance) have never been
    # rated, and ImportanceTerm abstains on them rather than voting a fabricated
    # midpoint.
    importance: ImportanceScore | None = None

    embedding: list[float] | None = None
    location: tuple[int, int] | None = None  # Grid coordinates (x, y)
    tags: list[str] = Field(default_factory=list)

    def __str__(self) -> str:
        """Format memory for LLM consumption.

        This is a prompt surface, and it has exactly one render site: the
        reflection node joins retrieved memories through it into its prompt
        (nodes/reflection/node.py). No other node renders a Memory - the rest
        take working-memory text - so tags reach them only indirectly, via whatever
        reflection writes back. The tags segment below is inert today because
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


class VectorDBMetadata(BaseModel):
    """Metadata stored alongside each memory in the vector store.

    Every field is optional-by-absence on purpose: ChromaDB drops keys we exclude
    on write, so a field that was never set reads back as None rather than as a
    plausible default. Scoring terms turn that None into an abstention.
    """

    # Always written, and load-bearing for that reason.
    #
    # Every other field here is optional, so a memory that was never rated, never
    # timestamped, unplaced and untagged has nothing to write - and ChromaDB
    # rejects an empty metadata dict outright. Passing `metadatas=None` instead
    # is NOT a safe alternative: a row added that way comes back from `query()`
    # carrying **a stale metadata dict left by a previously deleted row**, so a
    # brand-new unrated memory silently inherits another memory's tags. Verified
    # against chromadb 1.5.2; see the regression test in
    # tests/unit/test_vector_db_memory.py.
    #
    # Keeping one always-present key makes that state unreachable. It doubles as
    # a real schema marker for any later metadata migration.
    schema_version: int = 1

    # See Memory.importance: None means never scored, not scored zero.
    importance: ImportanceScore | None = None

    # See Memory.timestamp: elapsed game minutes, None means unknown.
    timestamp: int | None = None

    location_x: int | None = None
    location_y: int | None = None
    tags: list[str] = Field(default_factory=list)

    # Entities this memory is ABOUT. Reserved: nothing writes it yet. It exists
    # now because the relationship term [NPC-401] and the per-target stance term
    # [NPC-411] both key off it, and adding a field to an already-persisted
    # collection later means either a migration or a permanent two-schema read
    # path. Producing it is NPC-1013 / NPC-401.
    subject_ids: list[str] = Field(default_factory=list)

    # Per-memory override of the recency decay base, in the same per-game-hour
    # units as DEFAULT_RECENCY_DECAY_PER_GAME_HOUR. Reserved: nothing writes it
    # yet. NPC-406 has the mind emitting a per-memory decay_rate, and a term that
    # reads its base from a module constant cannot accept that without being
    # rewritten - so RecencyTerm reads this field from day one.
    #
    # Bounded at construction rather than clamped at use: a base above 1.0 would
    # make recency GROW with age and push the term outside [0, 1], and that must
    # fail loudly where it is written rather than be silently corrected where it
    # is read.
    decay_base: Annotated[float, Field(gt=0.0, le=1.0)] | None = None

    # The reinforced anchor the recency curve decays FROM, in elapsed game
    # minutes. Initialized to `timestamp` at write and pulled toward "now" by an
    # exponential moving average on every retrieval - see
    # retrieval.py::reinforced_time for the update and the reasoning.
    #
    # Not an integer: the EMA is a weighted blend of two game-minute readings and
    # is generally fractional. Rounding would quantize away small reinforcements
    # entirely for memories retrieved close to their own creation.
    effective_time: float | None = None

    # When this memory was genuinely last retrieved, in elapsed game minutes.
    #
    # Deliberately NOT what the recency curve reads, and deliberately not merged
    # into `effective_time`: a field named `last_accessed` holding a decayed
    # average would be a name that lies about its contents - the same class of
    # defect as a default impersonating data. This one means exactly what it
    # says, so anything later wanting a true last-access reading (an eviction
    # policy, a "you have not thought about this in weeks" cue) has an honest
    # source rather than an average it would have to un-blend.
    last_accessed: int | None = None

    def get_location(self) -> tuple[int, int] | None:
        """Extract location tuple if both coordinates present"""
        if self.location_x is not None and self.location_y is not None:
            return (self.location_x, self.location_y)
        return None

    @property
    def recency_anchor(self) -> float | None:
        """The game-minute reading the recency curve decays from.

        `effective_time` when present, else `timestamp`. That fallback covers
        rows PERSISTED before this field existed - a data-compatibility path for
        an on-disk collection, not a code-level shim for in-repo callers - and it
        resolves itself the first time such a memory is retrieved and written
        back.
        """
        if self.effective_time is not None:
            return self.effective_time
        return self.timestamp
