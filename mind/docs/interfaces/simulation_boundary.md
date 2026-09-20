# Simulation Boundary

What the Godot simulation actually sends this server, where each side's truth
lives, and how to check that simulation state is reaching the mind. The protocol
this server *exposes* is [interfaces/mcp.md](mcp.md); this is the other side.
Symbols written `file.gd::symbol` live in the npc-simulation repository.

## Shape

The simulation sends **structured data, not prose**.
`composite_observation.gd::get_data` produces a dict keyed by each
sub-observation's `get_type()`; `server.py` validates that dict into the
`Observation` model and renders it for the LLM in `Observation.__str__`. Nothing
is flattened to text before it crosses the wire.

Requests route by `mind_id`, the mind's own primary key. The simulation entity a
mind drives is named by a separate `entity_id`, fixed at `create_mind` and
repeated *inside* every observation. `decide_action` rejects the request when the
observation's `entity_id` does not match the routed mind's own — that mismatch
means the observation reached the wrong mind.

Two things do **not** arrive in the observation dict:

- **Conversations.** They reach the server as `INTERACTION_OBSERVATION` events and
  are lifted out by `server.py::_extract_conversation_observations` before
  validation. There is no `conversations` key on the wire.
- **Create-time configuration.** Traits, seed long-term memories and personality
  dimensions ride the `create_mind` payload assembled by
  `mcp_mind_client.gd::build_create_mind_config`. Stable facts are established
  once; per-cycle observations carry what changed.

## Boundary map

Assembled for NPCs by `npc_controller.gd::get_current_state_observation`, which
extends the base `entity_controller.gd::get_current_state_observation`.

| Wire key | Simulation producer | `Observation` field |
|---|---|---|
| `status` | `npc_controller.gd::get_current_state_observation` | `status` |
| `needs` | `drives_component.gd::create_needs_observation` | `needs` |
| `vision` | `vision_component.gd::create_vision_observation` | `vision` |
| `goal` | `substrate_component.gd::create_goal_observation` | `goal` |
| `place` | `substrate_component.gd::create_place_observation` | `place` |
| `mood` | `substrate_component.gd::create_mood_observation` | `mood` |
| `inventory` | `inventory_component.gd::create_observation` | `inventory` |

The observation types themselves are the simulation's `src/minds/observations/`
directory — read them there rather than from a list here.

### The status block's place keys are optional-by-absence

`status.current_zone_id` and `status.current_zone_name` name the place the NPC is
standing in ([NPC-1476]). Three properties, all of them contract:

- **Both keys travel or NEITHER does.** `status_observation.gd::get_data` omits
  the pair when there is no place; it never sends `""` and never sends null.
  Absence means *"the NPC stands on ground it knows no place for"*, which is a
  different fact from an empty id, and `SpatialTerm` abstains on it rather than
  scoring it. Read them with a presence check, never by comparing against `""`.
- **The value is belief, not ground truth.** The simulation filters the zones
  covering the NPC's cell to those in that NPC's own belief store *before*
  taking the innermost (`substrate_component.gd::get_current_place`). An NPC
  spawned or loaded into a zone without prior knowledge of it correctly reports
  no place; a restored belief can already know the zone.
- **Never re-resolve the name.** It arrives with the id precisely so this server
  does not look one up. There is no zone registry here, and inventing a lookup
  would be the omniscient path the sim-side filtering exists to close.

**Transport compatibility.** These are fields inside an *existing* block, not a
new root key, so the "declare here first, then deploy" rule below does not bind
them: `StatusObservation` keeps pydantic's default `ignore`, so a simulation that
ships them ahead of this server is silently tolerated, and this server reading an
older simulation gets `None`. Both orderings are safe for *breakage*. What is not
safe is assuming the mind-side consumers do anything before the simulation ships:
without a supplied known-zone stamp, the spatial term abstains.

### Formation provenance and memory retrieval

`ReflectionNode` stamps each LLM-produced `NewMemory` into a `FormedMemory`
using the current status cell and known zone. The LLM output schema has no
formation fields. `MemoryConsolidationNode` passes each memory's own stamps to
`VectorDBMemory.add_memory`; it does not substitute the location of the later
consolidation call. Optional metadata omits an unknown zone, and retrieval
reconstructs that absence as `None`. The explicit write timestamp remains a
separate consolidation argument; these fields add spatial provenance.

`MemoryRetrievalNode` passes the current status zone through `VectorDBQuery` to
`RetrievalContext`. `SpatialTerm` compares that zone with each memory's formation
zone: same place is 1, another place is 0, and either side unknown abstains. This
is a scoring term, not a filter on the candidate pool. Its weight of 0.5 is a
reasoned policy; realized-spread measurements belong with their recorded run
evidence, not an assertion that this value has been behaviorally calibrated.

The weighted, pool-normalized retrieval scorer remains distinct from the
simulation's proposed ADR-40 (belief as a posterior). Place selection models
offering uncertainty and absolute travel utility; memory retrieval combines
terms whose realized ranges differ. The shared contract here is the supplied
spatial provenance, not identical scoring arithmetic.

### Required delivery sequence for place-stamped memories

NPC-1476 (place-stamped memories) retains the approved sequence: simulation PR,
then simulation merge and server deployment, then the mind PR. Nested-key
compatibility does not waive that sequence. The delivery record must identify
the merged simulation revision, deployed server revision, and an observed
status payload carrying the paired place keys before reporting the spatial
consumer as exercised. Retain formation/storage/retrieval evidence from that
state; source presence and older test results do not establish deployment.
This document specifies the contract and does not report those steps complete.

### Places cross as ids; the model sees handles

`MOVE_TO` may name a place instead of a cell: exactly one of `destination` or
`zone_id` (NPC-1643). The simulation receives and resolves a **real zone id**;
the model never sees one. The prompt labels each place in the `place` block with
a per-cycle handle — `[p1]`, `[p2]`, … — from `PlaceObservation.place_handles`, a
pure function of that cycle's observation, so rendering, validation and
translation agree without storing a map. The action validator refuses any
`zone_id` that is not one of this cycle's handles (a stale handle, a name, a raw
id); `Action.wire_payload` swaps the handle for the id as the action leaves
`decide_action`; and the `ACTION_CHOSEN` event records the place's *name*, so a
later prompt carries neither a UUID nor a handle from another cycle.

**Delivery order:** this server must deploy strictly after the simulation that
understands `zone_id`. An older simulation refuses `{zone_id}` as a malformed
`MOVE_TO`, at ERROR, every cycle a mind names a place.

### Place queries are deferred attention, not actions

With response protocol v2, reflection may return a top-level `place_query` beside
the chosen `action`. Its shape is `{afford, max_distance?,
min_expected_providers?, limit?}` and its vocabulary is the canonical need and
interaction identifiers the simulation's existing `ZoneSeeking` table can score.
The query survives independently when action salvage falls back to `WAIT`; the
simulation queues it before dispatching the action.

The simulation validates and clamps the request, searches only the querying NPC's
own uncapped place-memory keys, ranks matches through the existing place-seeking
utility, and exposes at most three descriptors in the next cycle's
`place.query_result`. This server models that field as
`list[PlaceDescriptor] | None`: absence/`None` means no query was answered, while
an explicit `[]` means it ran and found no matching remembered place. Non-empty
answers share the ordinary per-cycle handles and may enter the ordinary goal menu;
the query result itself never commands movement.

This is a paired response/observation contract. Deploy the protocol-v2 mind only
with a simulation client that recognizes `place_query`, and deploy the v6 place
observation model before a simulation begins emitting `query_result`.

## Vocabulary this server must not hardcode

- **Drive names** — `needs.gd::Need`, spelled by `needs.gd::get_display_name`.
  They arrive as dictionary *keys*, so a rename there fails silently, not loudly.
- **Mood bands** — mirror `substrate_state.gd::valence_band` and `::arousal_band`.
  Branch on the band, never on the free-text `label`.
- **Event names** — `MindEventType` mirrors `mind_event.gd::Type` with two
  deliberate differences: `OBSERVATION` is absent here (it is the observation
  argument, not an event), and `ACTION_CHOSEN` exists only here.
- **Action names** — the accepted set is the match statement in
  `mcp_mind_client.gd::_create_action_from_mcp_response`, not the simulation's
  `src/contracts/actions/` directory, which also holds actions never routed
  through MCP. An unmatched name degrades to a wait and logs a warning.

## Checking whether simulation state reaches this server

The simulation carries `tools/mcp_parity_manifest.yaml`, checked by
`./tools/audit_mcp_parity.sh` and explained in `docs/process/mcp_parity.md`. It
exists because substrate features have repeatedly shipped without the
serialization layer being updated. Run it when you need the question answered; it
is a `/pr` gate there, not a GitHub Actions check.

Read a `pending` disposition as **intent, not fact** — it names a Linear issue and
asserts nothing about what is on the wire today. The falsifiable check is a
field's presence in `Observation` together with the simulation-side `get_data()`
or `to_dict()` that emits it.

## Known asymmetries

- `needs.max_need_value` is the wire spelling of `NeedsObservation.max_value`.
  The field accepts both; the wire name is the one the simulation sends.
- `entity_data.gd::to_dict` deliberately omits `last_interaction_time` — a raw
  game-minute stamp carrying a not-set sentinel, meaningless to a reader with no
  frame of reference for the simulation clock.
- `ACTION_CHOSEN` is a `MindEventType` member with no counterpart in
  `mind_event.gd::Type`, so the simulation never emits it.

## Parsing posture

`Observation` is `extra="forbid"` (NPC-1116). A root key this server does not
declare raises a `ValidationError`, `server.py` returns an error response, and
`mcp_mind_client.gd::_on_decide_action_response` logs it at ERROR and falls back
to a wait. That is loud but **total**: every MCP NPC stops acting, every cycle,
until a code change ships.

The consequence is a cross-repo ordering rule that runs opposite to the old one:

- **A new observation type must be declared here first, and deployed**, before
  the simulation-side `add_observation` merges. The server is a long-lived
  process launched from `.mind_launch.json` against a sibling clone, so pulling
  simulation `main` without restarting it is enough to cause the outage.
- Nested blocks are **not** uniformly forbidding. The `Goal*` family and
  `InventoryObservation` forbid; `StatusObservation`, `NeedsObservation`,
  `VisionObservation`, `MoodObservation`, `EntityData` and `VisibleInteraction`
  keep pydantic's default `ignore`. **Root forbid catches a new BLOCK, not a new
  FIELD inside an existing block.**
- `ConversationObservation` must never gain forbid: `server.py` validates it
  inside `try/except ValidationError: continue`, so forbidding there would
  silently skip every `INTERACTION_OBSERVATION` — the same bug one layer over.

**The parity audit cannot see this class of defect.** `audit_mcp_parity.py`
scans `audit_scope.substrate_files` for substrate-component getters and exports;
inventory is not substrate state, and the audit stays green through both this
bug and any regression of it. Forbid is the mechanism precisely because auditing
is not.

## See also

- [interfaces/mcp.md](mcp.md) — the protocol this server exposes
- In [npc-simulation](https://github.com/taylor1355/npc-simulation), under
  `docs/reference/minds/`: `mcp_mind.md`, `observations.md`, `simple_mind.md`
