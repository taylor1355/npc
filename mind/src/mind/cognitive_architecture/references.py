"""References: how an action parameter names something it was never shown an id for.

The simulation keys things by opaque ids: a message by its id, a place by a zone
UUID. The prompt shows neither. It shows a short per-cycle HANDLE beside the thing
-- ``[m3]`` beside a transcript row, ``[p2]`` beside a place -- the way a menu
option carries its ``option_id`` (NPC-1643 for places, NPC-2021 for messages). The
model writes the handle into the action, validation refuses a handle that names
nothing (the reflection node's retry loop re-asks with the valid ones in hand), and
the handle is swapped for the real id exactly once, as the action leaves the mind
(``Action.wire_payload``).

**Which parameters are references is read off the wire, not named here.** The
simulation marks a parameter's referent with an optional ``refers_to`` key on its
act-parameter spec::

    {"type": "string", "default": "", "description": "...", "refers_to": "message"}

or, when the referent depends on a sibling parameter::

    "refers_to": {"by": "<sibling parameter>", "domains": {"<its value>": "<domain>"}}

So a new reference parameter in the simulation reaches the prompt, the validator
and the wire translation with no Python change here. The only closed vocabulary in
this module is the DOMAINS -- ``message`` and ``place`` -- which are the contract's
vocabulary, not the grammar's.

A handle is a pure function of one cycle's state (``HandleTable``), so the prompt
that showed it, every validation retry, and the wire translation derive the
identical map from the identical snapshot; nothing is stored that could outlive the
cycle. Memory never holds a handle: ``HandleTable.remembered`` renders the referent
as prose.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from mind.logging_config import get_logger

if TYPE_CHECKING:
    from mind.cognitive_architecture.observations import ConversationMessage, Observation

logger = get_logger()

#: The wire key on an act-parameter spec naming what the parameter refers to.
WIRE_KEY_REFERS_TO = "refers_to"
#: Conditional form: the sibling parameter whose value selects the domain.
WIRE_KEY_REFERS_BY = "by"
#: Conditional form: sibling value -> domain. A domain of "" means "plain".
WIRE_KEY_REFERS_DOMAINS = "domains"

DOMAIN_MESSAGE = "message"
DOMAIN_PLACE = "place"

#: Domains this mind turns into handles. A domain the simulation names that is
#: not in this tuple degrades to a plain parameter (logged at DEBUG: a newer
#: simulation is expected to be ahead of this table), never to a refusal.
HANDLED_DOMAINS = (DOMAIN_MESSAGE, DOMAIN_PLACE)

#: The prompt's wording for what a handle of each domain names.
_DOMAIN_PHRASES: Mapping[str, str] = {
    DOMAIN_MESSAGE: "this message answers an open question: then the [mN] handle of that question's row, never an id",
    DOMAIN_PLACE: "you are naming a place you know: then its [pN] handle from the place list, never a name or an id",
}

MESSAGE_HANDLE_PREFIX = "m"

#: How much of a message a memory of replying to it keeps.
_REMEMBERED_MESSAGE_CHARS = 60


def normalize_handle(reference: str) -> str:
    """``" [M2] "`` -> ``"m2"``. Leniency on the token only: the simulation never
    sees a handle, so accepting its bracketed or upper-case spelling cannot make the
    two tiers disagree about what was supplied."""
    return reference.strip().strip("[]").strip().lower()


def message_handles(
    histories: Mapping[str, list[ConversationMessage]],
) -> dict[str, ConversationMessage]:
    """This cycle's message handles, ``{"m1": message, ...}``.

    Numbered in the order the transcript prints: interaction ids sorted, messages in
    stored order, one global 1-based ordinal. A PURE FUNCTION of the histories, so
    the prompt, every validation retry and the wire translation agree for a state
    snapshot. Across cycles the map is stable while histories only grow and no
    earlier-sorting conversation starts or ends; nothing depends on that, because a
    handle is only ever meaningful within the cycle that showed it.
    """
    handles: dict[str, ConversationMessage] = {}
    for interaction_id in sorted(histories):
        for message in histories[interaction_id]:
            handles[f"{MESSAGE_HANDLE_PREFIX}{len(handles) + 1}"] = message
    return handles


@dataclass(frozen=True)
class ParameterReference:
    """What one act parameter refers to, as the simulation declared it.

    Either ``domain`` is set (unconditional), or ``by`` names the sibling parameter
    whose value selects the domain through ``domains``.
    """

    domain: str = ""
    by: str = ""
    domains: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, value: Any, *, parameter: str = "") -> ParameterReference | None:
        """Parse a spec's ``refers_to`` value; None when absent or unusable.

        Never raises: this runs on the prompt path every cycle, so a malformed
        value degrades that one parameter to a plain one, with a log.
        """
        if value is None:
            return None
        if isinstance(value, str):
            return cls(domain=value.strip()) if value.strip() else None
        if isinstance(value, dict):
            by = value.get(WIRE_KEY_REFERS_BY)
            domains = value.get(WIRE_KEY_REFERS_DOMAINS)
            if isinstance(by, str) and by and isinstance(domains, dict):
                usable = {
                    str(key): domain.strip()
                    for key, domain in domains.items()
                    if isinstance(domain, str) and domain.strip()
                }
                return cls(by=by, domains=usable) if usable else None
        logger.warning(
            "Parameter %r has an unusable %r (%r); treating it as a plain parameter.",
            parameter,
            WIRE_KEY_REFERS_TO,
            value,
        )
        return None

    def domain_for(self, parameters: Mapping[str, Any]) -> str:
        """The domain this parameter refers to given the act's other parameters.

        "" when the referent is undetermined: a conditional whose selector is
        unset or unlisted, or a domain this mind does not handle. Undetermined
        means plain -- the simulation owns the verdict on such a value.
        """
        domain = self.domain
        if self.by:
            selector = parameters.get(self.by)
            domain = self.domains.get(str(selector), "") if selector is not None else ""
        if domain not in HANDLED_DOMAINS:
            if domain:
                logger.debug(
                    "Reference domain %r is not handled; passing the value through.", domain
                )
            return ""
        return domain

    def hint(self) -> str:
        """The prompt qualifier telling the model to give a handle, or "" when
        there is nothing it can be told."""
        if self.by:
            clauses = [
                f"{self.by}={key}: {_DOMAIN_PHRASES[domain]}"
                for key, domain in self.domains.items()
                if domain in _DOMAIN_PHRASES
            ]
            if not clauses:
                return ""
            return "leave empty unless declaring; then " + "; ".join(clauses)
        phrase = _DOMAIN_PHRASES.get(self.domain)
        return f"leave empty unless {phrase}" if phrase else ""


def is_parameter_set(value: Any) -> bool:
    """Whether an act parameter value counts as *declared* rather than omitted.

    The simulation's kinds pick sentinels outside their own domain (``false``
    for an affirmative-only flag, ``""`` for an id), so "not set" here is the
    union of those sentinels: absent, null, false, a blank string, the string
    ``"false"``, or an empty collection. It is a presence mirror; the kind's own
    ``is_declared`` in the simulation remains authoritative.
    """
    if value is None or value is False:
        return False
    if isinstance(value, str):
        stripped = value.strip()
        return bool(stripped) and stripped.lower() != "false"
    if isinstance(value, list | tuple | dict):
        return len(value) > 0
    return True


class HandleTable:
    """One cycle's handles over a state snapshot: resolve, enumerate, remember.

    Built from the observation (places) and the conversation histories (messages)
    of the state a decision runs on. Cheap, and rebuilt by each consumer from the
    same snapshot -- there is no stored table to go stale.
    """

    def __init__(
        self,
        observation: Observation | None,
        histories: Mapping[str, list[ConversationMessage]] | None,
    ):
        self._observation = observation
        self._messages = message_handles(histories or {})

    def handles(self, domain: str) -> list[str]:
        """The handles valid for ``domain`` this cycle, in prompt order."""
        if domain == DOMAIN_MESSAGE:
            return list(self._messages)
        if domain == DOMAIN_PLACE:
            return list(self._observation.place_handles()) if self._observation else []
        return []

    def resolve(self, domain: str, reference: str) -> str | None:
        """The real id a handle names, or None when it names nothing this cycle.

        Accepts a handle only. A raw id is REFUSED even when it belongs to
        something in view: the prompt never shows one, so a mind holding one got
        it through a channel that should not exist, and refusing surfaces the leak
        where accepting would hide it.
        """
        token = normalize_handle(reference)
        if domain == DOMAIN_MESSAGE:
            message = self._messages.get(token)
            return message.id if message is not None else None
        if domain == DOMAIN_PLACE and self._observation is not None:
            descriptor = self._observation.resolve_place_handle(token)
            return descriptor.zone_id if descriptor is not None else None
        return None

    def remembered(self, domain: str, reference: str) -> str:
        """The referent as prose for the ACTION_CHOSEN memory -- never a handle
        (it would name a different thing next cycle) and never an id."""
        token = normalize_handle(reference)
        if domain == DOMAIN_MESSAGE:
            message = self._messages.get(token)
            if message is None:
                return "an unknown message"
            speaker = message.speaker_name.strip() or "someone"
            text = message.message.strip()
            if len(text) > _REMEMBERED_MESSAGE_CHARS:
                text = text[: _REMEMBERED_MESSAGE_CHARS - 3].rstrip() + "..."
            return f'{speaker}\'s message "{text}"'
        if domain == DOMAIN_PLACE and self._observation is not None:
            descriptor = self._observation.resolve_place_handle(token)
            if descriptor is not None:
                return descriptor.label()
        return "an unknown place"
