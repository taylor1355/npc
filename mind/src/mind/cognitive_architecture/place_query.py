"""Place-domain payload and the current arm of the typed query request union."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Canonical vocabulary currently backed by the simulation's
# ZoneSeeking.NEED_INTERACTIONS table. Adding a new seeking consumer requires a
# paired expansion here; accepting an invented display string would make every
# query look valid while deterministically returning nothing.
PlaceAfford = Literal["hunger", "consume", "cook", "harvest", "harvest_plant"]


class PlaceQuery(BaseModel):
    """Handler-owned criteria for searching this NPC's own place knowledge."""

    model_config = ConfigDict(extra="forbid")

    afford: PlaceAfford = Field(
        description=("Canonical need or interaction to look for in this NPC's own place knowledge")
    )
    max_distance: int | None = Field(default=None, ge=0)
    min_expected_providers: float | None = Field(default=None, ge=0.0)
    limit: int | None = Field(default=None, ge=1, le=3)


class PlaceQueryRequest(BaseModel):
    """Discriminated place arm of the domain-query request contract.

    The kind is closed and the payload is validated by the place handler model.
    New domain kinds add an explicit model arm; unknown kinds remain validation
    errors rather than silently becoming place requests.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["place"]
    payload: PlaceQuery
