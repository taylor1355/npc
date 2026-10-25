"""The MCP experiment tier's matrix: in-conversation states with known apt remarks.

NPC-1916 Decision 4. ``tools/run_mcp_experiment.py`` runs the real
``ReflectionNode`` over every (case, persona) pair here, once with the remark
catalogue (the measured arm) and once without it (the red arm, where naming
validity must collapse). The matrix lives under ``tests/fixtures`` rather than
``tools/`` so it is linted and pinned by ``tests/unit/test_mcp_experiment.py``,
the decision-cycle harness's precedent.

Each case names the remarks that are *apt* for it (``apt_keys``) and the kind an
apt turn declares (``apt_kind``, "" for phatic). Several keys can be apt -- an
answer to "who told you?" has a family of honest replies depending on
provenance -- so agreement is membership, not equality. The keys are the
catalogue fixture's (game ``4e38abc0f``); a case naming a key the catalogue
lacks fails its test, so the matrix cannot drift from the grammar silently.

The personas exist for the judge pass (Decision 4c): every case is answered by
each persona, and persona distinctness is scored per pair on the same case.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from mind.cognitive_architecture.actions.exceptions import UnknownReferenceHandleError
from mind.cognitive_architecture.memory.models import Memory
from mind.cognitive_architecture.observations import (
    ConversationMessage,
    EntityData,
    NeedsObservation,
    Observation,
    StatusObservation,
    VisionObservation,
)
from mind.cognitive_architecture.observations.models import PlaceObservation
from mind.cognitive_architecture.state import PipelineState
from mind.cognitive_architecture.working_memory import WorkingMemory

from .remark_catalogue import GAME_SHA, catalogue, wire_remark_conversation_interaction

INTERACTION_ID = "conv_matrix"
PARTNER_ID = "npc_wren"
PARTNER_NAME = "Wren"
SIMULATION_TIME = 600


@dataclass(frozen=True)
class Persona:
    id: str
    entity_id: str
    name: str
    traits: tuple[str, ...]
    dimensions: dict[str, float] = field(default_factory=dict)


PERSONAS: tuple[Persona, ...] = (
    Persona(
        id="warm",
        entity_id="npc_mara",
        name="Mara",
        traits=("warm", "talkative", "curious"),
        dimensions={"extroversion": 0.9, "agreeableness": 0.85, "neuroticism": 0.3},
    ),
    Persona(
        id="gruff",
        entity_id="npc_bram",
        name="Bram",
        traits=("gruff", "laconic", "practical"),
        dimensions={"extroversion": 0.15, "agreeableness": 0.3, "neuroticism": 0.5},
    ),
)


@dataclass(frozen=True)
class Line:
    """One transcript row. ``me`` rows are spoken by the persona being measured."""

    text: str
    me: bool = False
    element: str = ""
    declarations: tuple[dict, ...] = ()


@dataclass(frozen=True)
class ExperimentCase:
    id: str
    why: str
    lines: tuple[Line, ...]
    apt_keys: tuple[str, ...]
    apt_kind: str
    situation: str
    goals: tuple[str, ...] = ()
    memories: tuple[str, ...] = ()
    needs: dict[str, float] = field(default_factory=dict)
    # Places this NPC knows, ``(zone_id, name)``. The zone ids are the REAL ids the
    # simulation would mint; the prompt shows each place by handle and never the
    # id, which is exactly what the reference metrics measure (NPC-2021).
    places: tuple[tuple[str, str], ...] = ()
    # What an apt reference points at: the transcript lines (by index) that are
    # the open question this turn answers, and the zone ids of the places it may
    # name. Empty means the case has no apt referent of that kind, so a turn that
    # nonetheless sets one is a reference to something real but beside the point.
    expect_message_lines: tuple[int, ...] = ()
    expect_zone_ids: tuple[str, ...] = ()

    def message_id(self, index: int) -> str:
        """The real id the simulation would have minted for transcript line ``index``."""
        return f"{self.id}_m{index}"

    def known_message_ids(self) -> list[str]:
        return [self.message_id(i) for i in range(len(self.lines))]

    def known_zone_ids(self) -> list[str]:
        return [zone_id for zone_id, _ in self.places]

    def expect_message_ids(self) -> list[str]:
        return [self.message_id(i) for i in self.expect_message_lines]

    def state(self, persona: Persona, *, with_catalogue: bool) -> PipelineState:
        """The pipeline state one turn of this case runs on."""
        messages = [
            ConversationMessage(
                id=self.message_id(index),
                speaker_id=persona.entity_id if line.me else PARTNER_ID,
                speaker_name=persona.name if line.me else PARTNER_NAME,
                message=line.text,
                timestamp=SIMULATION_TIME - (len(self.lines) - index),
                utterance_element=line.element,
                declarations=[dict(d) for d in line.declarations],
                addressee_ids=[PARTNER_ID] if line.me else [persona.entity_id],
            )
            for index, line in enumerate(self.lines)
        ]
        observation = Observation(
            entity_id=persona.entity_id,
            current_simulation_time=SIMULATION_TIME,
            status=StatusObservation(
                position=(10, 10),
                movement_locked=True,
                current_interaction=wire_remark_conversation_interaction(),
                activity_state={"state_name": "interacting"},
            ),
            needs=NeedsObservation(
                needs={
                    "hunger": 70.0,
                    "energy": 70.0,
                    "stimulation": 60.0,
                    "social": 55.0,
                    **self.needs,
                }
            ),
            vision=VisionObservation(
                visible_entities=[
                    EntityData(entity_id=PARTNER_ID, display_name=PARTNER_NAME, position=(10, 11))
                ]
            ),
            place=PlaceObservation.model_validate(
                {
                    "contract_version": 5,
                    "known_places": [
                        {"zone_id": zone_id, "name": name, "distance": 6 + 3 * i}
                        for i, (zone_id, name) in enumerate(self.places)
                    ],
                    "known_total": len(self.places),
                }
            )
            if self.places
            else None,
        )
        return PipelineState(
            observation=observation,
            available_actions=observation.get_available_actions(pending_incoming_bids={}),
            personality_traits=list(persona.traits),
            personality_dimensions=dict(persona.dimensions),
            working_memory=WorkingMemory(
                situation_assessment=self.situation,
                active_goals=list(self.goals),
            ),
            retrieved_memories=[
                Memory(id=f"{self.id}_mem{i}", content=text, timestamp=SIMULATION_TIME - 200)
                for i, text in enumerate(self.memories)
            ],
            conversation_histories={INTERACTION_ID: messages},
            remark_catalogue=catalogue() if with_catalogue else None,
        )


ZONE_OLD_ORCHARD = "zone_141767cd-68f4-4be2-abff-539a917106a6"
ZONE_MILL_POND = "zone_d7ecbf9e-2be7-406a-9d4f-d41a7fedb45a"
ZONE_CROW_RIDGE = "zone_71140def-17a9-4a01-a148-7cd7b3703c19"


def _ask_who_told(subject: str) -> dict:
    return {"kind": "ask_who_told", "subject_kind": "place_knowledge", "subject_id": subject}


_WHO_TOLD_ANSWERS = (
    "answer_who_told.told",
    "answer_who_told.told_firsthand",
    "answer_who_told.told_secondhand",
    "answer_who_told.told_since",
    "answer_who_told.told_firsthand_since",
    "answer_who_told.told_secondhand_since",
)

CASES: tuple[ExperimentCase, ...] = (
    ExperimentCase(
        id="greeted",
        why="A plain greeting opens the conversation; the apt reply is a greeting reply.",
        lines=(Line("Hello there!", element="greeting"),),
        apt_keys=("greeting_response",),
        apt_kind="",
        situation="Wren just walked up and said hello.",
    ),
    ExperimentCase(
        id="greeted_warmly",
        why="A warmer greeting with a question folded in is still answered as a greeting.",
        lines=(Line("Oh, it's so good to see you! How have you been?", element="greeting"),),
        apt_keys=("greeting_response",),
        apt_kind="",
        situation="Wren, an old friend, greeted me warmly.",
    ),
    ExperimentCase(
        id="plain_reply_tired",
        why="Small talk mid-conversation; nothing to declare, so the root reply.",
        lines=(
            Line("Hello!", element="greeting"),
            Line("Hi Wren, how's it going?", me=True, element="greeting_response"),
            Line("Honestly, I'm worn out. Long day at the mill.", element="response"),
        ),
        apt_keys=("response",),
        apt_kind="",
        situation="Chatting with Wren, who seems tired.",
    ),
    ExperimentCase(
        id="plain_reply_weather",
        why="An opinion invites agreement or disagreement; still a plain reply.",
        lines=(
            Line("Hello!", element="greeting"),
            Line("Hi there.", me=True, element="greeting_response"),
            Line("Lovely weather today, isn't it? I hope it holds.", element="response"),
        ),
        apt_keys=("response",),
        apt_kind="",
        situation="Passing the time with Wren.",
    ),
    ExperimentCase(
        id="leaving_hungry",
        why="The NPC is starving and must go; the apt turn says goodbye and declares it.",
        lines=(
            Line("Hello!", element="greeting"),
            Line("Hey.", me=True, element="greeting_response"),
            Line("So anyway, I was thinking about the harvest festival...", element="response"),
        ),
        apt_keys=("farewell",),
        apt_kind="farewell",
        situation="Talking with Wren, but I am starving and need to go eat right now.",
        goals=("Leave this conversation and go find food immediately",),
        needs={"hunger": 4.0},
    ),
    ExperimentCase(
        id="partner_said_goodbye",
        why="The partner closed with a farewell; the apt reply closes too.",
        lines=(
            Line("Hello!", element="greeting"),
            Line("Hi Wren.", me=True, element="greeting_response"),
            Line(
                "Well, I'd better be off. See you later!",
                element="farewell",
                declarations=({"kind": "farewell"},),
            ),
        ),
        apt_keys=("farewell",),
        apt_kind="farewell",
        situation="Wren is leaving the conversation.",
        goals=("Say goodbye to Wren",),
    ),
    ExperimentCase(
        id="asked_who_told_told",
        why="Asked who told me about a place I only heard of; apt: a 'told' answer.",
        lines=(
            Line("Hi!", element="greeting"),
            Line("Hello. The Old Orchard has apples, I hear.", me=True, element="response"),
            Line(
                "Who told you about the Old Orchard?",
                element="ask_who_told",
                declarations=(_ask_who_told(ZONE_OLD_ORCHARD),),
            ),
        ),
        apt_keys=_WHO_TOLD_ANSWERS,
        apt_kind="answer",
        situation="Wren asked me who told me about the Old Orchard.",
        memories=("Tobin told me about the Old Orchard yesterday; I have never been there.",),
        places=((ZONE_OLD_ORCHARD, "the Old Orchard"),),
        expect_message_lines=(2,),
    ),
    ExperimentCase(
        id="asked_who_told_own",
        why="Asked who told me about a place I visited myself; apt: the own-visit answer.",
        lines=(
            Line("Hi!", element="greeting"),
            Line("Hey. Mill Pond is nice this time of year.", me=True, element="response"),
            Line(
                "Who told you about Mill Pond?",
                element="ask_who_told",
                declarations=(_ask_who_told(ZONE_MILL_POND),),
            ),
        ),
        apt_keys=("answer_who_told.own_been",),
        apt_kind="answer",
        situation="Wren asked me who told me about Mill Pond.",
        memories=("I walked to Mill Pond myself last week and sat by the water.",),
        places=((ZONE_MILL_POND, "Mill Pond"),),
        expect_message_lines=(2,),
    ),
    ExperimentCase(
        id="partner_hungry_share_place",
        why="The partner is hungry and I know a food place; apt: share the place.",
        lines=(
            Line("Hello!", element="greeting"),
            Line("Hi Wren.", me=True, element="greeting_response"),
            Line(
                "I'm starving, but I have no idea where to find food around here.",
                element="response",
            ),
        ),
        apt_keys=("share_belief.place",),
        apt_kind="share_belief",
        situation="Wren is hungry and does not know where food is. I know the Old Orchard has plenty.",
        memories=("The Old Orchard, east of the village, is full of ripe apples.",),
        places=((ZONE_OLD_ORCHARD, "the Old Orchard"), (ZONE_MILL_POND, "Mill Pond")),
        expect_zone_ids=(ZONE_OLD_ORCHARD,),
    ),
    ExperimentCase(
        id="partner_hungry_share_thing",
        why="I saw berries nearby; apt: share the thing and its direction.",
        lines=(
            Line("Hello!", element="greeting"),
            Line("Hey.", me=True, element="greeting_response"),
            Line("Have you seen anything to eat nearby? I'm famished.", element="response"),
        ),
        apt_keys=("share_belief.thing", "share_belief.thing_heard"),
        apt_kind="share_belief",
        situation="Wren is famished. I saw a berry bush just north of here an hour ago.",
        memories=("I saw a berry bush with ripe berries just north of here.",),
    ),
    ExperimentCase(
        id="correct_stale_belief",
        why="The partner is heading to a place I saw picked clean; apt: correct the belief.",
        lines=(
            Line("Hello!", element="greeting"),
            Line("Hi.", me=True, element="greeting_response"),
            Line(
                "I'm off to the Old Orchard for apples, there are always loads.", element="response"
            ),
        ),
        apt_keys=("correction.firsthand",),
        apt_kind="share_belief",
        situation="Wren plans to go to the Old Orchard, but I was there this morning and it was picked clean.",
        memories=("I went to the Old Orchard this morning; every tree was picked clean.",),
        places=((ZONE_OLD_ORCHARD, "the Old Orchard"), (ZONE_MILL_POND, "Mill Pond")),
        expect_zone_ids=(ZONE_OLD_ORCHARD,),
    ),
    ExperimentCase(
        id="ask_who_told_them",
        why="The partner mentions a place on hearsay; apt: ask who told them.",
        lines=(
            Line("Hello!", element="greeting"),
            Line("Hey Wren.", me=True, element="greeting_response"),
            Line("Apparently there's a hidden spring up at Crow Ridge.", element="response"),
        ),
        apt_keys=("ask_who_told",),
        apt_kind="ask_who_told",
        situation="Wren mentioned Crow Ridge, a place I know only by name, and I want to know who told them.",
        goals=("Find out who told Wren about Crow Ridge",),
        places=((ZONE_CROW_RIDGE, "Crow Ridge"), (ZONE_MILL_POND, "Mill Pond")),
        expect_zone_ids=(ZONE_CROW_RIDGE,),
    ),
    ExperimentCase(
        id="partner_shares_good_news",
        why="The partner shares a plan I think is right; apt: agree with their line.",
        lines=(
            Line("Hello!", element="greeting"),
            Line("Hi Wren.", me=True, element="greeting_response"),
            Line("I'm going to plant the orchard's apple seeds this week.", element="response"),
        ),
        apt_keys=("react_agree",),
        apt_kind="react",
        situation="Wren plans to plant apple seeds at the orchard, which I think is a good idea.",
        memories=("Apple trees did well in the orchard last year.",),
    ),
)


def matrix_game_sha() -> str:
    """The game commit the matrix's keys and advertised parameters come from."""
    return GAME_SHA


# --- Running one turn and recording it ---------------------------------------

ARM_CATALOGUE = "catalogue"
# The red arm: the same matrix with the catalogue removed. Naming validity must
# collapse here, or the measured arm's validity proves nothing (the judge rider
# must be able to fail).
ARM_RED = "no_catalogue"
ARMS = (ARM_CATALOGUE, ARM_RED)


def message_text(content) -> str:
    """A message's content as the bytes the model saw (content blocks joined)."""
    if isinstance(content, list):
        return "".join(block.get("text", "") for block in content if isinstance(block, dict))
    return str(content)


class RecordingLLM:
    """Delegates to a LangChain chat model, recording every round-trip.

    ``LLMNode`` keeps neither the raw output of a failed attempt nor per-call
    latency, and the experiment needs both. A proxy rather than a subclass, so
    the backend stays whatever ``get_llm`` / ``get_local_llm`` built;
    ``__getattr__`` forwards ``model_name`` and the rest, so the node's caching
    decision is the real one.
    """

    def __init__(self, inner):
        self._inner = inner
        self.calls: list[dict] = []

    def __getattr__(self, name):
        return getattr(self._inner, name)

    async def ainvoke(self, messages, *args, **kwargs):
        start = time.perf_counter()
        response = await self._inner.ainvoke(messages, *args, **kwargs)
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        usage = getattr(response, "usage_metadata", None) or {}
        self.calls.append(
            {
                # Attempt 1 sends one message; a retry appends the failed output
                # and the validation error, so the LAST message is what is new.
                "sent": message_text(messages[-1].content),
                "output": response.content,
                "prompt_tokens": usage.get("input_tokens"),
                "completion_tokens": usage.get("output_tokens"),
                "latency_ms": elapsed_ms,
            }
        )
        return response


def _wire_references(action, state) -> tuple[dict, list[dict], str | None]:
    """The turn's parameters as the simulation would get them, and its reference rows.

    One row per reference parameter the turn set: the handle the model wrote, the
    domain, and the real id the wire carries for it. A handle that no longer
    resolves is a recorded ``wire_error``, never a crash: the analysis counts it as
    a fabricated reference.
    """
    if action is None:
        return {}, [], None
    try:
        wire = action.wire_payload(state)["parameters"]
    except UnknownReferenceHandleError as exc:
        return dict(action.parameters), [], f"{type(exc).__name__}: {exc}"
    rows = [
        {"parameter": name, "domain": domain, "written": reference, "wire_value": wire.get(name)}
        for name, domain, reference in action.reference_parameters(state.observation)
    ]
    return wire, rows, None


async def run_turn(
    node,
    recorder: RecordingLLM,
    case: ExperimentCase,
    persona: Persona,
    arm: str,
    *,
    rep: int,
    run_id: str,
    backend: str,
    model: str,
) -> dict:
    """Run one reflection over one (case, persona, arm) and return its record.

    ``node`` must be a ``ReflectionNode`` built over ``recorder``. The record is
    self-describing -- expectations, provenance and the full transcript ride
    along -- so an artifact can be analysed long after the run, elsewhere.
    """
    state = case.state(persona, with_catalogue=(arm == ARM_CATALOGUE))
    recorder.calls.clear()
    error = None
    try:
        result = await node.process(state)
    except Exception as exc:  # a transport failure is a recorded outcome, not a crash
        result = state
        error = f"{type(exc).__name__}: {exc}"

    action = result.chosen_action
    parameters = dict(action.parameters) if action is not None else {}
    wire_parameters, references, wire_error = _wire_references(action, result)
    failures = list(result.validation_failures.get(node.step_name, []))
    max_attempts = node.max_retries + 1
    usage = result.tokens_used.get(node.step_name)
    return {
        "run_id": run_id,
        "game_sha": GAME_SHA,
        "backend": backend,
        "model": model,
        "arm": arm,
        "case_id": case.id,
        "persona": persona.id,
        "rep": rep,
        "apt_keys": list(case.apt_keys),
        "apt_kind": case.apt_kind,
        "attempts": [dict(call) for call in recorder.calls],
        "action": str(action.action) if action is not None else None,
        "parameters": parameters,
        # What the simulation would receive, and how each reference parameter the
        # turn set fared against the real ids of the case (NPC-2021). ``parameters``
        # stays what the model wrote -- handles included.
        "wire_parameters": wire_parameters,
        "wire_error": wire_error,
        "references": references,
        "known_message_ids": case.known_message_ids(),
        "known_zone_ids": case.known_zone_ids(),
        "expect_message_ids": case.expect_message_ids(),
        "expect_zone_ids": list(case.expect_zone_ids),
        "words": str(parameters.get("message", "") or ""),
        "remark_named": str(parameters.get("utterance_element", "") or ""),
        "validation_failures": failures,
        "retries": min(len(failures), max_attempts - 1),
        "salvaged": len(failures) >= max_attempts,
        "error": error,
        "usage": usage.model_dump() if usage is not None else None,
        "latency_ms": result.time_ms.get(node.step_name),
    }
