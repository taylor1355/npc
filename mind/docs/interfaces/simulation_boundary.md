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
| `mood` | `substrate_component.gd::create_mood_observation` | `mood` |
| `inventory` | `inventory_component.gd::create_observation` | `inventory` |

The observation types themselves are the simulation's `src/minds/observations/`
directory — read them there rather than from a list here.

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
