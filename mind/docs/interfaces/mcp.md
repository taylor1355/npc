# MCP Server

## Overview

The MCP server exposes the cognitive architecture over HTTP, enabling the Godot simulation to create NPC minds and request decisions based on structured observations. Communication uses the Model Context Protocol (MCP) with Server-Sent Events (SSE) transport.

## Architecture

```
MCP Server (server.py)
├── Mind Management
│   └── minds: Dict[str, Mind] - Active mind instances
├── Tools (RPC methods)
│   ├── create_mind - Initialize cognitive pipeline
│   ├── decide_action - Process observation → action
│   ├── consolidate_memories - Daily → long-term
│   ├── cleanup_mind - Release instance, RETAIN memory
│   ├── relink_mind - Re-bind to an entity, rehydrating if needed
│   └── forget_mind - Erase memory permanently
└── Resources (State endpoints)
    ├── mind://{id}/state - Complete mental state
    ├── mind://{id}/working_memory - Current context
    └── mind://{id}/daily_memories - Unconsolidated memories
```

## Core Components

### MCPServer (`server.py`)

Manages active mind instances and exposes MCP tools for mind lifecycle and decision-making. Built on FastMCP with Starlette ASGI framework for SSE transport.

### Mind (`mind.py`)

Encapsulates a single mind's state and cognitive pipeline.

**Key Responsibilities:**
- Runs 4-node cognitive pipeline: memory query → retrieval → cognitive update → action selection
- Maintains working memory (current situational awareness)
- Buffers daily memories before consolidation to long-term storage
- Tracks conversation histories per interaction
- Manages event buffer with retention policy (60 game minutes OR max 15 events)

**Configuration:** Initialized via `Mind.from_config(mind_id, entity_id, config)`. `mind_id` (PK) keys the mind and its memory collection; `entity_id` (FK) names the simulation entity the mind drives; `MindConfig` carries the cognition-only settings (personality traits, LLM model, memory storage path). `entity_id` is no longer a `MindConfig` field.

## MCP Tools

The server exposes six RPC methods for mind lifecycle and decision-making.

The three lifecycle-ending tools are deliberately distinct, and the difference is
what happens to the persisted ChromaDB collection:

| Tool | In-memory instance | Persisted collection | Status values |
|---|---|---|---|
| `cleanup_mind` | dropped | **retained** | `released` |
| `relink_mind` | restored | read | `relinked` \| `not_found` |
| `forget_mind` | dropped | **deleted** | `forgotten` \| `not_found` |

### create_mind

Creates a new mind instance. Takes `mind_id` (PK), `entity_id` (FK), and `MindConfig` as separate arguments. `mind_id` keys the mind and its memory collection; `entity_id` names the simulation entity the mind drives; `MindConfig` holds cognition-only settings (traits, LLM model, memory storage path, optional seed memories). `entity_id` is no longer a `MindConfig` field.

#### `remark_catalogue` (optional)

`MindConfig.remark_catalogue` carries the simulation's wire remarks (NPC-1916): what a conversational turn may say it says. It rides `create_mind` once, not every observation, because it is stable world vocabulary. The simulation builds it with `remark_catalogue.gd::build` and omits it when no grammar is installed.

```json
{"contract_version": 1,
 "remarks": [{"key": "...", "gloss": "... {slot} ...", "declares": "<kind or empty>",
              "declare_with": ["<param>", "..."], "slots": ["<slot kind>"]}]}
```

- Rows are sorted by key. `declares` is `""` for a phatic remark, which declares nothing; otherwise it is the `DeclarationKind` key, and `declare_with` names the act parameters that declare it.
- An unknown `contract_version`, or an empty `remarks` list, **degrades** to no catalogue with a warning. Any other malformation refuses `create_mind`. Rows use `extra="ignore"`, so new columns such as FS10b's stance `fields` are additive.
- The catalogue is kept in the recorded config (`_config_to_record`), so a relinked mind validates against the grammar it was created with.

#### `refers_to` on act-parameter specs (optional)

An act parameter that names something by id (a message, a place) says so on its own spec in `act_in_interaction_parameters`, with an optional key beside `type` / `default` / `description` (NPC-2021):

```json
{"type": "string", "default": "", "description": "...", "refers_to": "message"}
{"type": "string", "default": "", "description": "...",
 "refers_to": {"by": "<sibling parameter>", "domains": {"<its value>": "<domain>", "<other>": ""}}}
```

- `refers_to` rides every observation with the spec (the catalogue cannot carry it: it is per parameter, and a parameter's referent can depend on a sibling's value). The mind reads it with `StatusObservation.act_parameter_references()`; no parameter is named in `mind/src`.
- The domains the mind turns into handles are `message` and `place`. Any other domain, an unusable value, or an absent key leaves the parameter a plain string, exactly as before, so an older simulation or a newer one is safe in both directions.
- For a set reference parameter the mind validates that the value is one of this cycle's handles (`[mN]` beside a transcript row, `[pN]` beside a place), re-asks through the reflection node's existing retry loop when it is not, and `Action.wire_payload` swaps the handle for the real id as the action leaves `decide_action`. The simulation therefore receives real ids and the prompt never carries one.
- A conditional form's `domains` value of `""`, or a selector that is unset or unlisted, means undetermined: the value passes through to the simulation, which remains the authority.

**What it changes.** While a catalogue is present and the current interaction advertises `utterance_element`, every `act_in_interaction` turn **must** set `utterance_element` to a catalogue key that agrees with what the turn declares:

- a phatic remark sets no `declare_with` parameter of any kind;
- a typed remark sets all of its kind's `declare_with` parameters;
- one turn declares at most one kind.

A missing, unknown or disagreeing key fails `Action` validation with a message naming the problem and the valid keys for the declared kind, and the reflection node's existing retry budget re-asks the model. Each failed attempt's error text is recorded in `PipelineState.validation_failures`. With no catalogue the key stays optional. The check is a presence mirror of `ConversationInteraction._agreed_element`, which stays authoritative and drops a disagreeing element while keeping the words. The catalogue is rendered into the prompt's dynamic section ("Remarks You Can Name") only while such a turn is possible. The prompt's static section carries the authored paragraph on what conversations can accomplish, and the naming rule, so the provider cache is unaffected. Measuring how well a model meets the rule is the job of the on-demand experiment tier, described in [cognitive_architecture/mcp_experiment.md](../cognitive_architecture/mcp_experiment.md).

### decide_action

Processes a structured observation and recent events through the cognitive pipeline and returns an action. Takes `mind_id`, `observation` dict containing entity status, needs, vision, and conversations, and optional `events` list of temporal occurrences since last decision.

**Parameters:**
- `mind_id`: Unique identifier for the mind
- `observation`: Structured observation dict with entity status, needs, vision, and conversations
- `events` (optional): List of MindEvent dicts with `timestamp`, `event_type`, and `payload` fields

**Processing Flow:**
1. Validates observation structure
2. Deserializes and validates events (if provided)
3. Updates conversation histories and event buffer
4. Runs cognitive pipeline: query memories → retrieve → update working memory → select action
5. Returns action dict or error

**Response fields:** every response carries `status`, `request_id` and
`protocol_version` (an int; currently `4`, bumped when the wire contract changes
in a way a client must know about: a response shape, or an accepted event type). A **success** response also carries `action`,
an optional top-level `query`, and a `telemetry` object. A query is a sibling of
`action`, not an action parameter: asking where to attend does not replace or
delay execution of the chosen action. The strict request union currently has one
production arm: `{"kind":"place","payload":{...}}`. Its closed payload shape
is `{afford: str, max_distance?: int, min_expected_providers?: float, limit?: int}`;
`afford` is one of the canonical need or interaction identifiers supported by the
simulation's existing place-seeking scorer, and `limit` is at most 3. The
simulation searches only that NPC's own remembered places and returns a typed
answer in the next observation's top-level `query_result` block. Unknown kinds
and malformed envelopes are explicitly refused.

`telemetry` carries `provenance` (`metered` | `unreported`), the
`prompt_tokens` / `completion_tokens` / `total_tokens` split,
`cached_prompt_tokens` with a `cache_reporting` flag distinguishing "no cache
hits" from "the provider does not report them", `model_calls` (provider
round-trips, retries included), `unreported_calls`, `model`, `server_ms`, and
per-step breakdowns.

An **error** response carries no `telemetry`, and a client must read that
absence as *unknown cost*, never as zero — the two are different facts, and on
the retry-exhaustion path the attempt genuinely spent tokens that no longer
reach this boundary. `provenance: "unreported"` on a *success* response means
the pipeline ran but no step reported usage, which distinguishes a silent
provider from a cheap decision.

**Event Types:** `INTERACTION_BID_REJECTED`, `INTERACTION_BID_RECEIVED`, `INTERACTION_STARTED`, `INTERACTION_FINISHED`, `INTERACTION_CANCELED`, `INTERACTION_OBSERVATION`, `ERROR`

#### The NPC action vocabulary is not an MCP tool surface

The six tools on this page are the *server's* API. The verbs an NPC chooses
between are a different, closed vocabulary carried inside `decide_action`'s
response, and declaring one means three edits rather than a `@mcp.tool()`:

| Layer | Where | Role |
|---|---|---|
| Name | `actions/models.py::ActionType` | Closed `str, Enum`. A name outside it fails validation. |
| Menu | `observations/models.py::Observation.get_available_actions` | The per-cycle prose menu rendered into `{available_actions}`. |
| Gate | `Action.validate_executable` | Per-action `_validate_*`; raises, and the error text is fed back for a corrective retry. |

The simulation matches these names upper-cased in
`McpMindClient._create_action_from_mcp_response`. **A name it does not match
falls through to `WaitAction` with only a warning** — no exception and no failed
cycle — so the two vocabularies are pinned against each other, in both
directions, by `tests/unit/actions/test_mark_zone_contract.py`.

#### mark_zone

Declares a stretch of visible ground to be a named place (NPC-1224 / NPC-1309).
Offered whenever the observation carries vision.

| Parameter | Meaning |
|---|---|
| `cells` | The ground, as `[[x, y], …]`. |
| `radius` | Alternative to `cells`: a disc of this radius around the marker. |
| `name` | The coined name. **Required here**, though the simulation would accept a blank one and derive a name itself. |
| `kind` | Serialized `Zone.Kind`; `gathering_ground` is the one current value. |

**Exactly one of `cells`/`radius` per act** — never both, never neither, because
a precedence rule between them would silently discard half of what the caller
asked for. "Supplied" mirrors the simulation's own predicates rather than key
presence: an empty `cells` list is *not* a supplied extent, and `radius: -1` is
the not-supplied sentinel (`-1` rather than `0`, because a radius of zero is a
legitimate one-cell mark). So `{"cells": [], "radius": 3}` is accepted while
`{"cells": [], "radius": -1}` is refused, even though both name the same keys.

Two things are deliberately **not** validated here, each because the simulation
is the better authority: the radius **bound** (it refuses anything past the
marker's sight and names both numbers; sight radius does not cross the wire) and
the `kind` **vocabulary** (an unrecognised kind is refused there against
`Zone.kind_names()`, so an allowlist here would make a new kind need a second
edit in this repository).

`purpose_tags` is **not** a parameter. Nothing in the simulation reads
`ZoneAttributes.purpose_tags`, so declaring it would advertise configuration
nothing honours.

#### The `place` observation block

Optional. Carries the substrate's passive place knowledge: `current_place` (the
innermost known zone covering the NPC's cell), a **capped** `known_places` list
with `known_total` beside it, `target_place` for the active journey, and the
`mark_budget`. `known_total >
len(known_places)` means the passive list is a truncation, which is the only way to tell "I know
three places" from "I know thirty and was shown three".

#### Generic domain-query result

The optional top-level `query_result` is a strict discriminated result envelope:
`{"contract_version":1,"kind":"place","payload":[PlaceDescriptor,...]}`.
No block means no query was answered this cycle; a present empty payload means
the handler ran and found no match. Place results share the same per-cycle place
handles as the passive list and can become ordinary simulation-side candidates;
they never directly author an action. The envelope's generic layer does not
contain place-specific rules: the registered handler owns payload validation,
epistemic authorization, result caps, and serialization. The place result cap
remains three.

`mark_budget.next_slot_in_minutes` is `-1.0` when a slot is already free —
deliberately not `0.0`, because a genuinely-zero wait is a real answer. Read
occupancy from `active`/`cap` and treat a negative value as "no wait", never as a
duration.

### consolidate_memories

Moves daily memories from buffer to long-term ChromaDB storage. Typically called at natural break points like sleep or scene transitions.

### cleanup_mind

Releases a mind from memory while **retaining** its persisted collection. Drops the in-memory `Mind` (freeing its pipeline and working state) but deliberately does *not* delete the ChromaDB collection, so a later `relink_mind` can re-attach and recover the mind's long-term memory. Use `forget_mind` to actually erase memory.

Returns status `released` — not `removed`. The name reflects the retain-on-release contract: the mind is released, its memory persists.

### relink_mind

Re-binds a mind to a (possibly new) driven entity, rehydrating it from its retained collection if it is no longer resident.

**Parameters:**
- `mind_id`: The mind's own identifier (PK) — keys the registry and the retained collection
- `entity_id`: The simulation entity this mind should now drive (FK)
- `memory_storage_path` (optional): Where to look for the collection when the server has no record of this mind — i.e. after a restart. See *Addressing a non-resident mind* below.

**Status:** `relinked` when the mind was resident (FK rebound in place) or rehydrated from a retained collection; `not_found` when neither exists.

Re-attaching never re-seeds `initial_long_term_memories`, so relinking repeatedly does not duplicate the original seeds.

### forget_mind

The destructive counterpart to `cleanup_mind`: deletes the persisted collection outright and drops the in-memory instance.

**Parameters:**
- `mind_id`: Mind to forget
- `memory_storage_path` (optional): as for `relink_mind`

**Status:** `forgotten` only when something was actually erased (a resident mind, a retained collection, or both); `not_found` otherwise.

`forgotten` is a claim that memory was destroyed, so it is never returned over a collection that survived — telling a caller the mind is gone when it is not is worse than an honest `not_found`. **This means `forget_mind` is not unconditionally idempotent:** forgetting the same `mind_id` twice returns `forgotten` then `not_found`. A client that treats any non-`forgotten` status as a failure will misread a retry.

### Addressing a non-resident mind

`memory_storage_path` (and `embedding_model`, `llm_model`, `traits`, `personality_dimensions`) are client-settable per mind at `create_mind` time, so the server cannot assume a default-constructed `MindConfig` describes any particular mind. It records each mind's creating config and resolves `relink_mind` / `forget_mind` through it, which is what keeps a custom-path mind addressable after release, and what makes a rehydrated mind come back with the same embedding model, LLM, traits, and personality it was created with.

That record is **process-local**. Across an eviction (`cleanup_mind` then `relink_mind`) it is intact. Across a **server restart** it is empty, and the client becomes the only remaining witness to where the collection lives — hence the optional `memory_storage_path` parameter on both tools.

Three limits worth knowing:

- Supplying the path after a restart restores **addressability, not fidelity**. Only the path crosses the process boundary, so the mind is rehydrated with default `embedding_model`, `llm_model`, `traits`, and `personality_dimensions`. A client that sets a non-default `embedding_model` should not rely on restart-relink: the rehydrated store would embed queries with a different model than wrote the stored vectors, which fails loudly only when the vector widths differ.
- Omitting the parameter preserves the previous behavior exactly (probe the default path), so a client that does not send it is unaffected.
- On `forget_mind` the parameter widens the **destructive** reach. When no record exists, `mind_id` and `memory_storage_path` are both client-supplied and combine into a delete of collection `mind_<mind_id>` under that directory — so a restarted server can erase a `mind_*` collection in **any** ChromaDB directory the process can write, not just its own. This is not a new capability class (`create_mind` already accepts an arbitrary `memory_storage_path`, so the write reach was already client-controlled) and the trust model is a local Godot client, but it does convert a server-chosen delete target into a client-chosen one, and the naming is unguessable only by convention. The `relink_mind` side is read-only by comparison: `Mind.reattach` is reached only after the collection is confirmed to exist, so a bogus path cannot even create a directory.

A caller-supplied path that disagrees with a recorded config is **ignored, and logged as a warning** — the record wins, because this server has no move operation. The warning exists so a client acting on a stale path can diagnose it before a destructive call reports success over a location the client never named.

## MCP Resources

Read-only endpoints for inspecting mind state:

- `mind://{mind_id}/state` - Complete mental state (traits, working memory, memory counts, active conversations)
- `mind://{mind_id}/working_memory` - Current situational awareness
- `mind://{mind_id}/daily_memories` - Unconsolidated memories pending storage

## Integration with Godot

The Godot simulation connects through a layered client system:

**Godot → Python Flow:**
1. `McpMindClient` (GDScript) - Collects observations from simulation
2. `McpSdkClient` (C#) - Marshals data and manages WebSocket
3. `McpServiceProxy` (C#) - MCP protocol communication
4. **MCP Server** (this component) - Validates and processes
5. **Cognitive Pipeline** - Generates decision
6. Response flows back through layers to simulation

**Key Data Transformations:**
- `CompositeObservation.get_data()` → observation dict
- Observation dict → validated `Observation` Pydantic model
- `Action.model_dump()` → action dict for Godot

For what the simulation puts in that dict, which vocabulary is owned on its side,
and how to check a piece of simulation state is reaching this server, see
[simulation_boundary.md](simulation_boundary.md).

## HTTP Endpoints

Beyond MCP protocol endpoints, the server provides standard HTTP endpoints for lifecycle management and monitoring.

### /health (GET)

Returns server status and uptime. Used by clients to detect when the server is ready after startup, avoiding failed connection attempts during initialization.

**Integration:** Godot client polls this endpoint until receiving 200 OK before attempting MCP connection.

### /shutdown (POST)

Triggers graceful server shutdown via HTTP request instead of OS process signals. This enables cross-platform server management without handling Windows/WSL process boundaries, where killing wrapper processes can leave Python processes orphaned.

### /logs (GET)

Returns structured log entries for client-side display and debugging. Stores the most recent 1000 log entries in memory with timestamp, level, and message.

**Query Parameters:**
- `since` (optional): Unix timestamp - returns only logs after this time for incremental fetching
- `limit` (optional): Maximum entries to return (default: 100)

**Response Format:**
```json
{
  "logs": [
    {"timestamp": 1736096823.456, "level": "INFO", "message": "Server started"},
    {"timestamp": 1736096824.123, "level": "DEBUG", "message": "SSE client connected"}
  ]
}
```

**Integration:** Game clients poll this endpoint every 1-2 seconds to fetch new logs for developer console display alongside simulation logs.

## Running the Server

### Basic Usage

```bash
cd /home/hearn/projects/npc/mind
poetry run python -m mind.interfaces.mcp.main
```

Server starts on `localhost:8000` by default.

### Command-Line Arguments

```bash
# Custom host and port
python -m mind.interfaces.mcp.main --host 127.0.0.1 --port 8000

# View all options
python -m mind.interfaces.mcp.main --help
```

### Configuration

Set `OPENROUTER_API_KEY` environment variable for LLM access.

### Server Output

```
Starting NPC Mind MCP server on http://127.0.0.1:8000/sse
```

## Error Handling

Tools return errors with `status: "error"` and descriptive `error_message`. Common failures include invalid mind IDs, malformed observations, LLM API errors, and ChromaDB storage issues.

## Development

**Adding Tools:**

Define async methods with `@self.mcp.tool()` decorator. FastMCP handles serialization via type hints. Access active minds through `self.minds` dictionary.

**Testing:**

Integration tests in `tests/integration/test_mcp_server.py` verify the complete workflow: create mind → decide action → consolidate memories → cleanup. Run via `poetry run pytest tests/integration/test_mcp_server.py -v`.

**Debugging:**

Common issues include observation validation failures (check FastMCP errors for field mismatches), LLM API errors (verify OPENROUTER_API_KEY and rate limits), and memory storage problems (monitor ChromaDB logs).
