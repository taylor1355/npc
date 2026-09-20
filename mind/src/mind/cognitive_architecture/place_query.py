"""Typed attention request for the simulation's own place knowledge (NPC-1481)."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Canonical vocabulary currently backed by the simulation's
# ZoneSeeking.NEED_INTERACTIONS table. Adding a new seeking consumer requires a
# paired expansion here; accepting an invented display string would make every
# query look valid while deterministically returning nothing.
PlaceAfford = Literal["hunger", "consume", "cook", "harvest", "harvest_plant"]


class PlaceQuery(BaseModel):
    """Optional attention work returned beside, and independent of, an action."""

    model_config = ConfigDict(extra="forbid")

    afford: PlaceAfford = Field(
        description=("Canonical need or interaction to look for in this NPC's own place knowledge")
    )
    max_distance: int | None = Field(default=None, ge=0)
    min_expected_providers: float | None = Field(default=None, ge=0.0)
    limit: int | None = Field(default=None, ge=1, le=3)
