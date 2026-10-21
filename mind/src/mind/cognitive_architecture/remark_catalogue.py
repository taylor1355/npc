"""The simulation's remark catalogue: what a conversational turn may say it says.

A *remark* is one thing an actor can say, keyed permanently by the simulation's
phrase grammar (``GrammarSymbol`` there). Each wire remark either declares
nothing (phatic: a greeting, a plain reply) or declares exactly one
``DeclarationKind`` through that kind's own act parameters (``declare_with``).
The simulation builds the catalogue once from its grammar and sends it in the
``create_mind`` payload, because it is stable world vocabulary, not per-cycle
state.

Nothing in this module -- or anywhere in this repository -- names a remark key,
a declaration kind or a parameter that declares one. The catalogue is the only
source of all three, so a remark the simulation adds reaches the prompt and the
validator with no Python change. ``tests/unit/test_remark_vocabulary.py`` pins
that rule the way ``ConversationMessage.declarations`` pins it for kinds.

The agreement rule is the contract's: a phatic remark declares nothing; a typed
remark declares exactly its own kind. This side checks *presence* only. The
simulation may judge a declaration more strictly, and it stays the trust
boundary: it drops a disagreeing element and keeps the words.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from mind.cognitive_architecture.actions.exceptions import RemarkElementError
from mind.cognitive_architecture.references import is_parameter_set
from mind.logging_config import get_logger

logger = get_logger()

# The catalogue shape this server understands. An unknown version degrades to
# "no catalogue" (keys optional) with a warning, never to a refused create_mind:
# a newer simulation must not stop its MCP NPCs from existing.
REMARK_CATALOGUE_CONTRACT_VERSION = 1

# The act parameter a conversational turn names its remark in. A wire contract
# name (``ActInInteractionAction.PARAM_UTTERANCE_ELEMENT`` in the simulation),
# like ``ConversationMessage.utterance_element``; never a remark or kind.
WIRE_KEY_UTTERANCE_ELEMENT = "utterance_element"

# The act parameter that says who a message is spoken TO. A wire contract name
# (``ActInInteractionAction.PARAM_ADDRESSEE_IDS`` in the simulation); the reply
# parameter is deliberately not named here -- see ``message_level_parameters``.
WIRE_KEY_ADDRESSEE_IDS = "addressee_ids"

# How many valid keys an error lists before eliding. The full set is in the
# prompt's catalogue block; the error only has to point the retry at it.
_MAX_KEYS_IN_ERROR = 24


def _same_value(actual: Any, expected: str) -> bool:
    return isinstance(actual, str) and actual.strip() == expected


class RemarkEntry(BaseModel):
    """One wire remark in the catalogue contract.

    ``extra="ignore"`` so the row can grow columns additively (as ``fields``
    did) without a contract-version bump.
    """

    model_config = ConfigDict(extra="ignore")

    key: str = Field(min_length=1)
    # The primary gloss, with ``{slot}`` placeholders where the wording reads a
    # slot. Prompt content only.
    gloss: str = ""
    # The DeclarationKind key this remark says; "" is phatic.
    declares: str = ""
    # The act parameters that declare ``declares`` (the kind's parameter specs).
    declare_with: list[str] = Field(default_factory=list)
    # Slot kinds the remark's wording may read. Prompt content only.
    slots: list[str] = Field(default_factory=list)
    # Act parameters this remark fixes to one exact value (a stance remark sets
    # its stance parameter). Every key is one of ``declare_with``.
    fields: dict[str, str] = Field(default_factory=dict)

    @field_validator("key", "declares")
    @classmethod
    def _strip(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode="after")
    def _declaration_is_consistent(self) -> RemarkEntry:
        if not self.key:
            raise ValueError("a remark key must not be blank")
        if self.declares and not self.declare_with:
            raise ValueError(
                f"remark '{self.key}' declares '{self.declares}' but names no declare_with "
                "parameter, so no act could ever agree with it"
            )
        if not self.declares and self.declare_with:
            raise ValueError(
                f"remark '{self.key}' is phatic but names declare_with parameters "
                f"{self.declare_with}"
            )
        stray = [name for name in self.fields if name not in self.declare_with]
        if stray:
            raise ValueError(
                f"remark '{self.key}' sets fields {stray} that are not in its "
                f"declare_with {self.declare_with}"
            )
        return self

    @property
    def is_phatic(self) -> bool:
        return not self.declares

    def prompt_line(self) -> str:
        """``key: "gloss" [declares K via p1, p2] [slots a, b]``."""
        line = f'{self.key}: "{self.gloss}"' if self.gloss else self.key
        if self.declares:
            line += f" [declares {self.declares} via {', '.join(self.declare_with)}]"
        if self.fields:
            line += f" [sets {', '.join(f'{k}={v}' for k, v in self.fields.items())}]"
        if self.slots:
            line += f" [slots {', '.join(self.slots)}]"
        return line


class RemarkCatalogue(BaseModel):
    """Every wire remark a turn may name, sorted by key as the simulation sends it."""

    contract_version: int
    remarks: list[RemarkEntry] = Field(min_length=1)

    @model_validator(mode="after")
    def _keys_are_unique(self) -> RemarkCatalogue:
        seen: set[str] = set()
        duplicates = sorted({r.key for r in self.remarks if r.key in seen or seen.add(r.key)})
        if duplicates:
            raise ValueError(f"remark catalogue repeats keys: {duplicates}")
        return self

    @classmethod
    def parse_or_degrade(cls, raw: Any) -> RemarkCatalogue | None:
        """The ``MindConfig`` boundary: a catalogue, or None with a warning.

        Degrades (keys become optional) for an unknown ``contract_version`` or an
        empty remark list. Any other malformation is a contract break and raises,
        so ``create_mind`` refuses it loudly instead of silently running
        without the grammar.
        """
        if raw is None or isinstance(raw, RemarkCatalogue):
            return raw
        if isinstance(raw, dict):
            version = raw.get("contract_version")
            if version != REMARK_CATALOGUE_CONTRACT_VERSION:
                logger.warning(
                    "remark_catalogue contract_version %r is not %d; running without the "
                    "catalogue (utterance_element becomes optional). This server is likely "
                    "older than the simulation and needs a pull and a restart.",
                    version,
                    REMARK_CATALOGUE_CONTRACT_VERSION,
                )
                return None
            if not raw.get("remarks"):
                logger.warning(
                    "remark_catalogue lists no remarks; running without the catalogue "
                    "(utterance_element becomes optional)."
                )
                return None
        return cls.model_validate(raw)

    def get(self, key: str) -> RemarkEntry | None:
        return next((remark for remark in self.remarks if remark.key == key), None)

    def keys(self) -> list[str]:
        return [remark.key for remark in self.remarks]

    def message_level_parameters(self) -> set[str]:
        """``declare_with`` parameters shared by more than one kind.

        A parameter two kinds both list belongs to the message, not to either
        kind (the reply target is one): its presence says nothing about which
        kind a turn declares. Derived from the rows, so no parameter is named.
        """
        owners: dict[str, set[str]] = {}
        for remark in self.remarks:
            for name in remark.declare_with:
                owners.setdefault(name, set()).add(remark.declares)
        return {name for name, kinds in owners.items() if len(kinds) > 1}

    def reply_parameter(self, remark: RemarkEntry) -> str | None:
        """The message-level parameter naming the message a fixed-field remark replies to.

        A remark that fixes a field and lists a message-level parameter is spoken
        in reply to the message that parameter names. None for any other remark.
        """
        if not remark.fields:
            return None
        shared = self.message_level_parameters()
        return next((name for name in remark.declare_with if name in shared), None)

    def declaring_parameters(self) -> dict[str, list[str]]:
        """Each declared kind -> the union of its remarks' ``declare_with``, in order."""
        out: dict[str, list[str]] = {}
        for remark in self.remarks:
            if remark.is_phatic:
                continue
            params = out.setdefault(remark.declares, [])
            for name in remark.declare_with:
                if name not in params:
                    params.append(name)
        return out

    def keys_for(self, declared: str) -> list[str]:
        """Remark keys that agree with declaring ``declared`` ("" = nothing)."""
        return [remark.key for remark in self.remarks if remark.declares == declared]

    def declared_kinds(self, parameters: dict) -> list[str]:
        """The kinds an act's parameters declare, by presence of their parameters."""
        shared = self.message_level_parameters()
        return [
            kind
            for kind, names in self.declaring_parameters().items()
            if any(is_parameter_set(parameters.get(n)) for n in names if n not in shared)
        ]

    def prompt_block(self) -> str:
        """One line per remark, for the dynamic prompt section."""
        return "\n".join(remark.prompt_line() for remark in self.remarks)

    @staticmethod
    def applies_to(advertised_parameters: dict) -> bool:
        """Whether an interaction's advertised act parameters make it a remark turn.

        An interaction that advertises ``utterance_element`` speaks in remarks;
        nothing else is consulted, so no interaction is named here.
        """
        return WIRE_KEY_UTTERANCE_ELEMENT in advertised_parameters

    def check_turn(self, parameters: dict) -> None:
        """Raise ``RemarkElementError`` unless the turn names an agreeing remark.

        Required: a key that is in the catalogue and agrees with what the turn
        declares -- nothing for a phatic remark, exactly its own kind (with all
        of that kind's ``declare_with`` parameters set) for a typed one. The
        error names the problem and the keys valid for what was declared, so the
        reflection node's existing retry loop can re-ask with that guidance.
        """
        declared = self.declared_kinds(parameters)
        raw = parameters.get(WIRE_KEY_UTTERANCE_ELEMENT)
        key = raw.strip() if isinstance(raw, str) else ""

        if len(declared) > 1:
            raise RemarkElementError(
                f"the turn declares {len(declared)} kinds at once ({', '.join(declared)}); "
                "one message declares at most one. Keep the parameters of one kind only",
                self._keys_text(None),
            )
        declared_kind = declared[0] if declared else ""

        if not key:
            raise RemarkElementError(
                f"'{WIRE_KEY_UTTERANCE_ELEMENT}' is required on every conversational turn: "
                "name the remark your words say",
                self._keys_text(declared_kind),
            )

        remark = self.get(key)
        if remark is None:
            raise RemarkElementError(
                f"'{key}' is not a remark in the catalogue", self._keys_text(declared_kind)
            )

        if remark.is_phatic and declared_kind:
            raise RemarkElementError(
                f"'{key}' declares nothing, but the turn declares '{declared_kind}'. Either name "
                f"a remark of '{declared_kind}' or drop its parameters",
                self._keys_text(declared_kind),
            )
        if not remark.is_phatic:
            # "Declares its own kind" reduces to "every declare_with is set": a
            # turn declaring another kind (alone) leaves these unset, and one
            # declaring both was refused above.
            missing = [n for n in remark.declare_with if not is_parameter_set(parameters.get(n))]
            if missing:
                what = f"declares '{declared_kind}'" if declared_kind else "declares nothing"
                raise RemarkElementError(
                    f"'{key}' says '{remark.declares}' and must set "
                    f"{', '.join(remark.declare_with)}, but the turn {what}"
                    + (f" (unset: {', '.join(missing)})" if missing else ""),
                    self._keys_text(declared_kind),
                )
            wrong = {
                name: value
                for name, value in remark.fields.items()
                if not _same_value(parameters.get(name), value)
            }
            if wrong:
                needed = ", ".join(f"{k}={v}" for k, v in wrong.items())
                raise RemarkElementError(
                    f"'{key}' fixes {needed}; the turn must set exactly that",
                    self._keys_text(declared_kind),
                )

    def _keys_text(self, declared_kind: str | None) -> str:
        if declared_kind is None:
            keys = self.keys()
            label = "all remarks"
        elif declared_kind:
            keys = self.keys_for(declared_kind)
            label = f"remarks of '{declared_kind}'"
        else:
            keys = self.keys_for("")
            label = "remarks that declare nothing"
        shown = keys[:_MAX_KEYS_IN_ERROR]
        more = f", ... (+{len(keys) - len(shown)} more)" if len(keys) > len(shown) else ""
        return f"Valid keys ({label}): {', '.join(shown) or 'none'}{more}"
