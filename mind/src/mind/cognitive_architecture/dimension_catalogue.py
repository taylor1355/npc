"""The simulation's dimension catalogue: where a line's words sit on a graded axis.

A *dimension* is a signed score a speaker declares on every line they say (how
the speaker treats the listener, say). SimpleMind reads the score off the phrase
it drew; an MCP mind writes its own words, so it must be told what each part of
the scale means and must state where its words sit. The simulation builds the
catalogue once from its band tables and sends it in the ``create_mind`` payload
as ``dimensions``, because it is stable world vocabulary, not per-cycle state.

Nothing in this module -- or anywhere else in ``src/`` -- names a dimension, a
band or the parameter that carries one. The catalogue is the only source of all
of them, so a dimension the simulation adds reaches the prompt, the heard-line
rendering and the validator with no Python change
(``tests/unit/test_dimension_catalogue.py`` pins that rule).

Three jobs, each derived from the rows:

* **Prompt** -- ``prompt_block`` renders each dimension's range, meaning, how it
  differs from its neighbours, and every band with its effect and landmark lines.
* **Heard lines** -- ``marker_for`` turns a stored declaration into
  ``[<kind>: <Band> <value>]``.
* **Validation** -- ``check_turn`` requires every advertised dimension parameter
  to be a real number inside the range. A missing value is refused, never read as
  0, and an out-of-range value is refused, never clamped.

A band owns ``low <= value < high``; the last band also owns the top edge, so
every value in the range falls in exactly one band.
"""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from mind.cognitive_architecture.actions.exceptions import DimensionValueError
from mind.logging_config import get_logger

logger = get_logger()

# The catalogue shape this server understands. An unknown version degrades to
# "no dimensions" with a warning, never to a refused create_mind: a newer
# simulation must not stop its MCP NPCs from existing.
DIMENSION_CATALOGUE_CONTRACT_VERSION = 1

# Band edges are authored decimals; contiguity is judged with this slack so a
# float spelling difference never refuses a table that has no gap.
_EDGE_TOLERANCE = 1e-9


def _is_real_number(value: Any) -> bool:
    """A JSON number: not a bool, not a numeric string, not NaN or infinite."""
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _format_number(value: float) -> str:
    """``-0.75``, ``0.4``, ``1``: shortest faithful spelling."""
    return f"{value:g}"


def _range_text(low: float, high: float) -> str:
    return f"{_format_number(low)} to {_format_number(high)}"


class Band(BaseModel):
    """One named stretch of a dimension, with what it means and what it does."""

    model_config = ConfigDict(extra="ignore")

    name: str = Field(min_length=1)
    low: float
    high: float
    # What the speaker is doing at this stretch of the scale.
    meaning: str = ""
    # What the world does about it; "" when nothing.
    effect: str = ""
    # Example lines that sit in this band, as the simulation's bank words them.
    landmarks: list[str] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _strip(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode="after")
    def _is_a_stretch(self) -> Band:
        if not self.name:
            raise ValueError("a band name must not be blank")
        if not (math.isfinite(self.low) and math.isfinite(self.high)):
            raise ValueError(f"band '{self.name}' has a non-finite edge")
        if self.low >= self.high:
            raise ValueError(f"band '{self.name}' has low {self.low} not below high {self.high}")
        return self


class Dimension(BaseModel):
    """One graded axis: its range, its wire names, and its band table."""

    model_config = ConfigDict(extra="ignore")

    key: str = Field(min_length=1)
    # The act parameter a turn sets this dimension through. Also the name of the
    # field the mind's structured output carries: they map one to one.
    parameter: str = Field(min_length=1)
    # How a heard line carries the stored value: the declaration's ``kind``, and
    # the key inside that declaration that holds the number.
    declaration_kind: str = Field(min_length=1)
    declaration_field: str = Field(min_length=1)
    # One line on what the axis is about.
    about: str = ""
    # How this axis differs from the neighbouring one a mind also sets.
    contrast: str = ""
    low: float
    high: float
    bands: list[Band] = Field(min_length=1)

    @field_validator("key", "parameter", "declaration_kind", "declaration_field")
    @classmethod
    def _strip(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode="after")
    def _bands_tile_the_range(self) -> Dimension:
        for name in ("key", "parameter", "declaration_kind", "declaration_field"):
            if not getattr(self, name):
                raise ValueError(f"dimension {name} must not be blank")
        if not (math.isfinite(self.low) and math.isfinite(self.high)) or self.low >= self.high:
            raise ValueError(
                f"dimension '{self.key}' range [{self.low}, {self.high}] is not a stretch"
            )
        names = [band.name for band in self.bands]
        if len(set(names)) != len(names):
            raise ValueError(f"dimension '{self.key}' repeats band names: {names}")
        if abs(self.bands[0].low - self.low) > _EDGE_TOLERANCE:
            raise ValueError(
                f"dimension '{self.key}' first band starts at {self.bands[0].low}, "
                f"not at the range's low {self.low}"
            )
        if abs(self.bands[-1].high - self.high) > _EDGE_TOLERANCE:
            raise ValueError(
                f"dimension '{self.key}' last band ends at {self.bands[-1].high}, "
                f"not at the range's high {self.high}"
            )
        for below, above in zip(self.bands, self.bands[1:], strict=False):
            if abs(below.high - above.low) > _EDGE_TOLERANCE:
                raise ValueError(
                    f"dimension '{self.key}' bands '{below.name}' and '{above.name}' leave a gap "
                    f"or overlap: {below.high} then {above.low}"
                )
        return self

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high

    def band_for(self, value: float) -> Band:
        """The band owning ``value``. Raises ``ValueError`` outside the range."""
        if not self.contains(value):
            raise ValueError(f"{value} is outside {self.key}'s range")
        for band in self.bands:
            if value < band.high:
                return band
        return self.bands[-1]

    def range_text(self) -> str:
        return _range_text(self.low, self.high)

    def check_value(self, raw: Any) -> float:
        """The value as a float, or ``DimensionValueError``. Never clamps, never defaults."""
        guidance = f"Set '{self.parameter}' to a number from {self.range_text()}"
        if raw is None:
            raise DimensionValueError(self.parameter, "is missing from the turn", guidance)
        if not _is_real_number(raw):
            raise DimensionValueError(self.parameter, f"must be a number, not {raw!r}", guidance)
        value = float(raw)
        if not self.contains(value):
            raise DimensionValueError(
                self.parameter, f"is {_format_number(value)}, outside its range", guidance
            )
        return value

    def prompt_lines(self) -> list[str]:
        title = f"**{self.key}**" + (f": {self.about}" if self.about else "")
        lines = [
            title,
            f"Set `{self.parameter}` to a number from {self.range_text()} on every line you "
            "say, from where your words sit on this scale. A turn without it, or with a "
            "value outside the range, is refused.",
        ]
        if self.contrast:
            lines.append(self.contrast)
        lines.append("Bands, in order:")
        for band in self.bands:
            row = f"- {band.name} ({_range_text(band.low, band.high)})"
            if band.meaning:
                row += f": {band.meaning}."
            if band.effect:
                row += f" Effect: {band.effect}."
            if band.landmarks:
                row += " e.g. " + "; ".join(f'"{line}"' for line in band.landmarks)
            lines.append(row)
        reading = (
            f"A heard line marked [{self.declaration_kind}: Band value] was said at that value."
        )
        if self.contains(0.0):
            reading += f" A line with no such marker was said at 0 ({self.band_for(0.0).name})."
        lines.append(reading)
        return lines


class DimensionCatalogue(BaseModel):
    """Every graded dimension a turn may carry, as the simulation sends them."""

    contract_version: int
    dimensions: list[Dimension] = Field(min_length=1)

    @model_validator(mode="after")
    def _names_are_unique(self) -> DimensionCatalogue:
        for attribute in ("key", "parameter", "declaration_kind"):
            seen = [getattr(d, attribute) for d in self.dimensions]
            duplicates = sorted({s for s in seen if seen.count(s) > 1})
            if duplicates:
                raise ValueError(f"dimension catalogue repeats {attribute}: {duplicates}")
        return self

    @classmethod
    def parse_or_degrade(cls, raw: Any) -> DimensionCatalogue | None:
        """The ``MindConfig`` boundary: a catalogue, or None with a warning.

        Absent (an older simulation) is None and silent. Degrades, with a warning,
        for an unknown ``contract_version`` or an empty dimension list. Any other
        malformation is a contract break and raises, so ``create_mind`` refuses it
        loudly instead of silently running with a calibration nobody can trust.
        """
        if raw is None or isinstance(raw, DimensionCatalogue):
            return raw
        if isinstance(raw, dict):
            version = raw.get("contract_version")
            if version != DIMENSION_CATALOGUE_CONTRACT_VERSION:
                logger.warning(
                    "dimensions contract_version %r is not %d; running without the dimension "
                    "catalogue. This server is likely older than the simulation and needs a "
                    "pull and a restart.",
                    version,
                    DIMENSION_CATALOGUE_CONTRACT_VERSION,
                )
                return None
            if not raw.get("dimensions"):
                logger.warning("dimensions lists no dimension; running without the catalogue.")
                return None
        return cls.model_validate(raw)

    def applicable(self, advertised_parameters: dict) -> list[Dimension]:
        """The dimensions whose parameter the interaction advertises.

        Whether an interaction grades its lines is read off its own schema, so no
        interaction is named here.
        """
        return [d for d in self.dimensions if d.parameter in advertised_parameters]

    def prompt_block(self, advertised_parameters: dict) -> str:
        """The calibration for the dynamic prompt section; "" when nothing applies."""
        return "\n\n".join(
            "\n".join(d.prompt_lines()) for d in self.applicable(advertised_parameters)
        )

    def marker_for(self, declaration: Any) -> str | None:
        """``[<kind>: <Band> <value>]`` for a heard declaration of a dimension, else None.

        Never raises: it runs while the prompt is built, so a stored value the
        catalogue cannot place falls back to the bare ``[<kind>]`` marker (None
        here) with a warning rather than costing the cycle.
        """
        if not isinstance(declaration, dict):
            return None
        kind = declaration.get("kind")
        if not isinstance(kind, str):
            return None
        dimension = next((d for d in self.dimensions if d.declaration_kind == kind.strip()), None)
        if dimension is None:
            return None
        raw = declaration.get(dimension.declaration_field)
        if not _is_real_number(raw) or not dimension.contains(float(raw)):
            logger.warning(
                "heard %s declaration carries %r in '%s'; showing it without a band.",
                dimension.declaration_kind,
                raw,
                dimension.declaration_field,
            )
            return None
        value = float(raw)
        return (
            f"[{dimension.declaration_kind}: {dimension.band_for(value).name} "
            f"{_format_number(value)}]"
        )

    def check_turn(self, parameters: dict, advertised_parameters: dict) -> None:
        """Raise ``DimensionValueError`` unless every applicable dimension is validly set."""
        for dimension in self.applicable(advertised_parameters):
            dimension.check_value(parameters.get(dimension.parameter))
