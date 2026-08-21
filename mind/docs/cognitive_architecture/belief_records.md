# Belief Records

An NPC's beliefs are not the world. The simulation keeps ground truth — every zone
that exists, every entity, every claim — and separately each NPC keeps a much
smaller, often stale, sometimes wrong set of records *about* it. This document
defines what one of those records **is**.

It is a **contract, not a schema**, and it names no field list deliberately. The
field list belongs to the simulation; a copy transcribed here would create a
synchronization obligation nothing can enforce, since no CI in either repo can
check a Python field list against a GDScript class. What it names instead are
**invariants** a mind may rely on, **obligations** a mind takes on when it holds a
record, and a table handing six open questions to their owners.

```
npc-simulation (owns the definition)                    mind (this repo)

  registry plane ── ground truth ────────┐
    every zone, entity, claim that exists│  enforcement reads truth:
                                         │  a refusal is valid whether
                                         │  or not the refused party knew
                                         v
  knowledge plane ── per NPC ────────────┐
    BeliefStore, one per NPC             │   observation      cognitive
      └─ place records (first kind)      ├─── boundary ─────> pipeline
      └─ future kinds join by key        │   (NPC-1299,       (this repo)
                                         │    unbuilt)
    decisions read ONLY this plane ──────┘
```

**Decisions read only the knowledge plane.** A hungry NPC's candidate places are
the ones *it knows*, which may be out of date — acting on old news, being corrected
on arrival, and updating the record is legible behavior rather than a bug
(npc-simulation: `docs/design_research/explorations/world/zone-layer-design.md`,
"Ground truth and knowledge: the two-plane model").

Nothing in this repo holds a belief record today. Falsifier:
`grep -rnE "PlaceKnowledge|BeliefStore|learned_from|told_by|learned_at" mind/src/`
returns nothing, while the control `grep -rl "VectorDBMemory" mind/src/` returns
several files — so the zero is an absence rather than a broken search. **Scope it
to `mind/src/`, not `mind/`:** this document names those very tokens, so an
unscoped sweep matches itself and reports a false positive forever after. This
document is written ahead of the wiring so that the four tenants below do not each
invent their own answer.

## The envelope: what every belief record carries

Five properties. They are stated as rules about the *shape* of knowledge, not as
names of fields, so that a rename in the simulation cannot silently falsify them.

**1. A belief record is held by exactly one NPC.** It is always *someone's*, and
the holder is never optional and never inferred. Two structural forms satisfy this:
the holder is the **container** the record was found in — the per-NPC store the
simulation ships, `BeliefStore` owned by `SubstrateComponent` — or an explicit
**key** in a centralized pool. Forbidden is a nullable holder field, or a record
whose holder must be reconstructed from context. A record read out of its container
carries its holder with it.

**2. The subject is an id, not prose.** A belief record is about a zone, an entity,
an agent, a proposition — named by a stable identifier in a namespace the
simulation owns. Not a description, not a coordinate, not free text. This is the
sharpest break from how durable knowledge works in this repo today, where the unit
is `Memory.content: str` and everything about the subject is recoverable only by
reading English. A subject id makes "everything this NPC believes about zone X" a
lookup rather than a search.

**3. How it was acquired is a closed vocabulary, and it is load-bearing.** Every
record carries how its holder came to know the thing. Three sub-properties
generalize from the first implementation:

- The vocabulary is **closed**, and its member spellings are a save contract from
  birth — they land in every save row and cannot be tidied later without a version
  bump (`place_knowledge.gd::_SOURCE_NAMES`).
- An unrecognized value is **refused, not defaulted**. The parser returns an
  out-of-enum sentinel that is deliberately kept out of the enum so that iterations
  over the members never see it (`place_knowledge.gd::SOURCE_INVALID`,
  `::source_from_string`). A parser that answered with a real member on a typo would
  report an NPC as a place's founder because a save row was malformed.
- Exactly one member — the socially-acquired one — carries the id of who supplied
  the knowledge (`place_knowledge.gd::told_by_id`). That is the provenance chain
  word of mouth needs.

Acquisition is load-bearing rather than decorative because it is what makes "how
many of these knowers learned this socially" answerable at all, forever after.

**4. Acquisition is immutable; belief is not.** First write wins
(`belief_store.gd::learn_place` returns false and changes nothing when the subject
is already known). An NPC who found a place by walking it and was later told about
it did not acquire it socially, and overwriting would inflate the social count in
the flattering direction. Later updates change **what** the holder believes about
the subject; they never rewrite **how** it came to know of it.

**5. First-acquisition time is history, not a cache.** The simulation-time reading
at acquisition is never overwritten (`place_knowledge.gd::learned_at_minutes`), and
it is distinct from any last-observed or last-confirmed timestamp a tenant adds
alongside it. Collapsing the two loses the ability to ask how long something has
been known.

**Everything else is per-tenant payload** — what this NPC believes *about* the
subject, plus whatever confidence, staleness or decay that tenant needs. **The
payload may be empty.** The first implementation ships with none at all, and that
is the reference case rather than a degenerate one: a record that says only
*"I know this place, and here is how I learned it"* is already useful.

## What a belief record does not carry

The exclusion is stated as a **test**, not a list, so it stays applicable to
tenants nobody has designed yet:

> Is there a **producer** that writes this value, distinct from a **reader** that
> could compute it on demand? If it would be recomputed on read anyway, storing it
> forks a second learning store over the same experience stream.

The simulation's design argues the worked example: an NPC's *appraisal* of a place
— its fit to that NPC's drives, its distance, its expected contention — is derived,
never learned. It composes on read from stores that already exist, so persisting it
would create exactly the redundant accumulator the design forbids (npc-simulation:
`zone-layer-design.md`, "Appraisal is derived, never learned"). That conclusion is
the design's, cited rather than re-derived here.

The corollary matters as much: **an always-empty field in every save row is a
completeness claim in structural form.** A field no producer fills advertises a
capability the system does not have. It joins when something writes it.

## The four tenants

Four systems want a per-holder, provenance-carrying record. They are listed as
*(tenant, subject namespace, where defined, owning issue)* — with **no field
lists**, because three of the four do not exist yet and their field lists are their
owners' to write.

| Tenant | Subject is | Defined in | Issue |
|---|---|---|---|
| Place knowledge | a zone id | npc-simulation | NPC-1212 |
| Spatial memory of entities | an entity id | npc-simulation | NPC-201 |
| Structured / semantic-opinion memory | the id of whatever the opinion is about | this repo | NPC-411 |
| Theory of mind | another agent's id, plus a proposition about them | both repos | NPC-394 (sim) / NPC-446 (here) |

Only the first has an implementation: `PlaceKnowledge` records held in a
`BeliefStore`, landed with NPC-1212 part 2 (npc-simulation:
`src/field/components/substrate/knowledge/`). The other three rows are claims about
*intent*, drawn from the issues that own them; the subject column says what kind of
id each will need, not what its owner has decided.

## Authority: the simulation defines, the mind holds a view

The simulation **owns the canonical record**: field names, the acquisition
vocabulary and its save-format spellings, refusal semantics, and the store's API.
This repo neither restates those nor proposes changes to them here.

This repo owns two things, and only two: what it may **assume** about any belief
record handed to it (the five invariants above), and what it is **obliged to
preserve** when one enters its own systems. That is why the contract reads as
invariants rather than a field table — *"every belief record carries an acquisition
drawn from a closed vocabulary; a mind may branch on it and must never invent a
member"*, never a transcribed enum.

**This is also the staleness argument, and it is not hypothetical.** The
simulation's own design document and its own shipped code already disagree about
the record's field list, in the same repo, with nothing flagging it: the design
section still describes fields the implementation deliberately omits. A document of
invariants survives that. A transcribed field list in a second repo and a second
language would simply be wrong, and nobody would know.

Minds do author records for their own tenants — semantic opinion, theory of mind —
and those go in the same envelope. The contract is the shared type; the simulation
owns the first instance and its vocabulary, not the type itself.

**Two conventions worth matching rather than reinventing.** Malformed rows are
refused *loudly at row level* — skip, log which field refused it, continue — because
one bad row should not cost the good rows (`belief_store.gd::from_save_data`). And
records serialize into a **kind-keyed envelope**, so a second record kind adds a key
rather than a discriminator column on every row (`belief_store.gd::SAVE_KEY_PLACE`).
A mind-authored tenant should match both.

**No base class, no registry, no Python mirror — yet.** The simulation deliberately
declined to build a record base class or kind registry until the second tenant's
shape is known, and the same restraint applies here: a documented type is not a
mandate for a Python type. Do not read this document as authorizing code it
deliberately does not order. A future mirror has **preconditions**, all of them
currently unmet:

- **A producer and a consumer.** The wire shape that would carry a belief record
  across the boundary is NPC-1299 under NPC-33's version bump, and exists in neither
  repo's source. Falsifier: `grep -rn "PlaceObservation" mind/src/` and the same
  search against the simulation's `src/` both return nothing, with
  `StatusObservation` as a control that hits at both roots; the simulation's
  `tools/mcp_parity_manifest.yaml` independently records the gap and assigns it to
  NPC-1299. A model with zero call sites is false completeness, not progress.
- **An explicit extras policy, or a version handshake.** Pydantic's default is
  `extra="ignore"`, and this repo has exactly one model that rejects unknown keys
  (`vector_db_memory.py::VectorDBQuery`, whose docstring makes the argument: a
  dropped filter is a wrong answer, not a missing one). Everything a record would
  pass through on the way in and back out inherits the permissive default, so an
  undeclared key vanishes with no log line and no test failure. Choosing between a
  rejecting policy, declared fields, and a version handshake is NPC-1116 / NPC-33.
- **At least one rendering site.** A model that parses correctly and never reaches
  a prompt satisfies nothing.

## What this contract does not decide

Six questions this document is deliberately silent on. Each row names the owner and
the property at stake, so the owner can decide freely while knowing what their
answer has to preserve.

| Question | Owner | Property at stake |
|---|---|---|
| Is a belief record stored as a memory, as a knowledge-graph node kind, or in a store beside memory? | NPC-1303 | Whichever answer wins must keep subject-id keying. A record whose subject survives only as prose cannot be looked up by subject. |
| Does typed content survive the write into memory, or flatten to prose? | NPC-1306 | Whether provenance is queryable at all. A record flattened at write time cannot answer "what do I believe only because someone told me" — the acquisition became a substring. |
| Is "who else knows this" the same mechanism seen from the other side? | NPC-1302 | If it is the same mechanism, it is an acquisition record with holder and subject swapped, and inherits invariant 4. If separate, it needs its own envelope. |
| Does this answer the episodic / semantic split? | NPC-1304 | A record is semantic; the event that created it is episodic. The envelope makes the distinction *expressible*. It does not assert the mapping. |
| What does a belief record look like on the wire? | NPC-1299, NPC-33 | Simulation-owned. This contract states only that the envelope must survive the crossing intact. |
| Retrieval weighting, credibility, relationship-weighted belief | NPC-1301 | **Out of scope entirely.** Decided on the simulation side; not restated and not re-derived here. |

One thing this contract *does* constrain beyond its own scope, called out rather
than buried: naming acquisition-immutability as an invariant means that if NPC-1302
concludes "same mechanism", that mechanism inherits first-write-wins. This is
judged legitimate — immutability is the property that makes social-learning counts
computable — but it is a real constraint on another issue's solution space, and
NPC-1302 should decide against it knowingly.

## Sources

Cited for the claim each actually supports. Cross-repo paths carry their repo,
because a bare relative path resolves to nothing from here. These commands are the
anti-staleness mechanism: run them, do not assume them.

**The two-plane split, and appraisal-is-derived** — npc-simulation:
`docs/design_research/explorations/world/zone-layer-design.md`, sections "Ground
truth and knowledge: the two-plane model" and its "Appraisal is derived, never
learned" paragraph. Note that this section's own field list is **older than the
implementation** and lists fields the code deliberately omits; it is cited here for
the two-plane argument and the appraisal argument, not for field names.

**The acquisition vocabulary, refusal semantics, and immutability** — npc-simulation:
`src/field/components/substrate/knowledge/`, symbols `PlaceKnowledge::Source`,
`::SOURCE_INVALID`, `::source_from_string`, `::told_by_id`, `::learned_at_minutes`,
`BeliefStore::learn_place`, `::SAVE_KEY_PLACE`, `::from_save_data`. Falsifier:
`git ls-tree origin/main -r --name-only | grep "substrate/knowledge"` in that repo.
Symbol cites are chosen over line numbers so a rename returns an empty grep — a
loud break — rather than a plausible wrong line.

**Where durable knowledge lives in this repo today** —
`cognitive_architecture/memory/models.py::Memory` (the unit is `content: str`;
`location` is a bare coordinate pair with no namespace) and
`cognitive_architecture/memory/vector_db_memory.py::VectorDBQuery` (semantic
similarity plus a tag filter, and nothing else).

**The `tags` field is an inert slot, not a destination for provenance.** Nothing on
any production path populates it. The independent probe is the call sites, not the
docstring that says so — a docstring beside the code is not independent evidence
about that code. `git grep -n "add_memory(" origin/main -- mind/src` returns the
definition plus exactly two call sites (`nodes/memory_consolidation/node.py` and
`interfaces/mcp/mind.py`), neither of which passes `tags=`; every `tags=` producer
the tree contains lives under `mind/tests`. Wiring a producer is NPC-1013.

**Exactly one model in `mind/src` rejects unknown keys.** Falsifier:
`git grep -n "model_config\|ConfigDict\|class Config" origin/main -- mind/src`
enumerates every policy declaration, so a fourth one or a changed policy shows up
immediately. At the time of writing there are three declarations —
`actions/models.py` (`use_enum_values`), `vector_db_memory.py::VectorDBQuery`
(`extra="forbid"`), `working_memory.py` (`extra="allow"`) — and every
model not in that list inherits Pydantic's permissive default, including
`Observation` and `VectorDBMetadata`, the latter re-validated on every read in
`VectorDBMemory.search`.

**This is the first mind-repo document to reference the simulation's knowledge-plane
design.** Falsifier, evaluated against `origin/main` before this commit:
`git grep -rn "zone-layer" origin/main -- mind/` returned nothing, while the control
`git grep -l -i "godot" origin/main -- mind/` returns hits throughout the tree. Run
against the working tree the same search now returns **this file**, which is the
expected outcome and not a refutation — scope the query to `origin/main` as of the
parent commit if you want to re-check the original claim.
