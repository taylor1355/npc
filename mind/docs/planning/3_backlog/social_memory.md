# Relationship-Weighted Memory Retrieval

## Scope

Relationship *state* is not the mind's to hold. The Godot substrate owns per-pair
relationship state in `RelationshipRegistry` and ships it inside every observation.
This document covers only what the mind does with it.

An earlier version of this file proposed a mind-side relationship record —
interaction count, recency, emotional valence, topics discussed, shared experiences,
trust/friendship levels — on the premise that NPCs "treat every interaction as if
meeting for the first time". That premise held when it was written and does not hold
now, so the proposal is retracted. Interaction count, recency, emotional valence and
friendship level already ship in the substrate; topics discussed and shared
experiences are assigned to a substrate store; trust was rejected there on the record.
The governing split is NPC-399: the simulation defines the canonical interface; the
backend coordinates separately.

## What the substrate owns

`RelationshipData` (npc-simulation, `relationship_data.gd`) is per-pair and holds four
scalars — `familiarity`, `sentiment`, `last_interaction_time`, `interaction_count`.
`RelationshipRegistry` is the single writer, decay and save/load included.

Three of the four cross the wire. `entity_data.gd::to_dict` omits
`last_interaction_time` deliberately: a raw game-minute stamp with a not-set sentinel,
meaningless to a reader that has no frame of reference for the simulation clock. So
this repo's mirror, `observations/models.py::RelationshipState`, *matches* the wire
contract rather than falling short of it. The whole relationship key is omitted for
strangers, which makes its presence the signal that shared history exists at all.

**Topics discussed and shared experiences** belong to the substrate's topic library —
per-pair records stored alongside `RelationshipRegistry`, familiarity-gated
(npc-simulation, `conversation-semantic-layer.md`, "The topic library"). Designed; no
code yet. The mind must not grow a second one.

**Trust is not a relationship field.** The same document rejects it twice under "Trust
is per-record, not per-pair" — as a sentiment proxy (you can dislike someone and still
believe them) and as a third relationship float. The accepted alternative derives
credibility per record from the teller's track record. Out of scope here.

## What the mind adds on top

- **Relationship-weighted retrieval** (NPC-401). Scores retrieved memories using the
  relationship fields the observation already carries; stores nothing. Blocked by
  NPC-400.
- **Participant tagging** (NPC-1013). Filtering retrieval by who was present.
  `Memory.tags` and `VectorDBQuery.tags` exist as a storage column with no producer
  and no consumer, so the field is inert today.
- **Rendering shared history.** Partly shipped: `Observation.__str__` already emits
  familiarity, sentiment, and shared-interaction count per visible entity. Missing is
  the retrieved *episodes* with that partner, which the two items above supply.

## Open questions

- **Cross-referencing two NPCs' memories of the same event.** Whether that shares a
  mechanism with "where I learned this, and from whom" is NPC-1302's decision, not
  this document's. Deliberately left open in both directions.
- Whether participant tags are chosen by the query LLM or derived from context —
  NPC-1013.

## Priority rationale

**Obviousness**: Moderate. The store this used to propose already exists; what remains
is one weighting term.

**Development velocity**: Neutral. It adds a term to a scoring function that does not
exist yet, and no new state anywhere.

**Concreteness**: High. An NPC recalling the specific thing you did together is
visible immediately; the relationship numbers behind it already reach the prompt.
