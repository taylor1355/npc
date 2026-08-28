# NPC Mind - Claude Code Context

## Project Overview

This repository provides the **Mind tier** cognitive architecture for the NPC simulation system. It runs as an MCP (Model Context Protocol) server that provides LLM-based decision-making and memory management as a service to the Godot simulation.

**Status:** Production-ready core architecture with LangGraph pipeline, structured memory system, and full MCP integration.

## Sister Project

The Godot-based simulation is located at `/mnt/c/Users/taylor/Dev/gamedev/npc-simulation` on WSL.

The sister project contains the visual simulation environment (Controller and Client tiers) where these NPC cognitive architectures (Mind tier) operate. Refer to its documentation for:
- Three-tier architecture overview (Controller → Client → Mind)
- Scene architecture and entity systems
- MCP integration patterns and protocol details
- Visual components and UI

## Architecture

```
src/mind/
├── cognitive_architecture/          # LangGraph-based pipeline
│   ├── pipeline.py                  # StateGraph orchestration
│   ├── state.py                     # PipelineState model
│   ├── working_memory.py            # WorkingMemory, NewMemory (shared models)
│   ├── nodes/                       # Processing nodes
│   │   ├── base.py                  # Node & LLMNode base classes (caching, retry, salvage hook)
│   │   ├── memory_query/            # Generate semantic search queries
│   │   │   ├── node.py
│   │   │   ├── models.py
│   │   │   └── prompt.md
│   │   ├── memory_retrieval/        # Vector search via ChromaDB
│   │   │   └── node.py
│   │   ├── reflection/              # Update working memory, form memories, choose action
│   │   │   ├── node.py
│   │   │   ├── models.py            # ReflectionOutput
│   │   │   └── prompt.md            # Split at a cache-breakpoint marker
│   │   └── memory_consolidation/    # Daily → long-term storage
│   │       └── node.py
│   └── memory/
│       └── vector_db_memory.py      # ChromaDB backend
├── interfaces/mcp/                  # MCP server interface
│   ├── server.py                    # FastMCP server
│   ├── mind.py                      # Mind runtime state
│   ├── models.py                    # MindConfig, validation
│   └── main.py                      # Server entry point
├── apis/                            # LLM integration
│   ├── langchain_llm.py             # Primary LLM client (OpenRouter)
│   ├── llm_client.py                # Direct OpenAI client
│   └── messages.py                  # Message utilities
└── prompts/                         # Legacy prompt templates (archived)
```

### Cognitive Pipeline (3-step LangGraph)

1. **Memory Query** - Generate semantic search queries from observations
2. **Memory Retrieval** (no LLM) - ChromaDB vector search with deduplication
3. **Reflection** - One LLM call that updates WorkingMemory, forms new memories, AND
   chooses the action (merged from the former cognitive_update + action_selection
   pair, NPC-1319). Its prompt's static prefix (scaffold + world knowledge + format
   instructions) is served from the provider's prompt cache for allowlisted models.

**Performance:** the pre-merge baseline, measured live (NPC-1318, 2026-08-20), was
**5,367 tokens per decision cycle** (mean over 10 cycles, cold memory store,
`google/gemini-2.5-flash-lite`; per-node means 551.2 / 2,802.5 / 2,012.9 for
memory_query / cognitive_update / action_selection). Reproduce with the harness in
NPC-1318's baseline comment: `PYTHONPATH=$PWD/src:$PWD python measure_baseline.py`
(**that harness is not in this repo** — it exists only as an attachment on the Linear
issue, so reproducing the number means retrieving it from there first)
from a mind checkout with credentials linked. The reflection merge removes the
duplicated prompt content between the two retired nodes; re-measure with the same
harness rather than extrapolating.

### Memory System

- **Vector storage** via ChromaDB with semantic search
- **Metadata:** timestamp, location, unique ID
- **Importance scoring:** 1-10 scale via LLM evaluation
- **Deduplication:** ID-based filtering in retrieval
- **Daily buffer:** Consolidates to long-term on demand
- **Recency decay:** Applied in retrieval scoring

### Working Memory (Structured State)

```python
class WorkingMemory(BaseModel):
    situation_assessment: str = ""
    active_goals: list[str] = []
    recent_events: list[str] = []
    current_plan: list[str] = []
    emotional_state: str = ""

    class Config:
        extra = "allow"  # Extensible for future fields
```

## MCP Server

### Tools

- **`create_mind(mind_id, entity_id, config)`** - Initialize a mind. `mind_id` (PK) keys the mind/memory collection, `entity_id` (FK) names the driven simulation entity, `config` (MindConfig) carries cognition-only settings (traits, seed memories, LLM/personality). `entity_id` is no longer a MindConfig field
- **`decide_action(mind_id, observation, events)`** - Process structured observation + events → action dict
- **`consolidate_memories(mind_id)`** - Transfer daily memories → long-term storage
- **`cleanup_mind(mind_id)`** - Release the in-memory instance but **retain** its ChromaDB collection → `released`. Not a delete; `forget_mind` is
- **`relink_mind(mind_id, entity_id, memory_storage_path=None)`** - Re-bind a mind to a (possibly new) entity, rehydrating from the retained collection when it is no longer resident → `relinked` | `not_found`
- **`forget_mind(mind_id, memory_storage_path=None)`** - Permanently delete the collection and drop the instance → `forgotten` | `not_found`. Not idempotent: a second forget returns `not_found`, since `forgotten` must never be claimed over memory that survived

The server records each mind's creating `MindConfig` so relink/forget address the mind's own storage path and rehydrate it with its own `embedding_model`, traits, and personality rather than defaults. That record is process-local, so the optional `memory_storage_path` exists to locate a mind after a **server restart**; it restores addressability, not the rest of the config (NPC-1023).

### Resources

- **`mind://{id}/state`** - Complete mental state snapshot
- **`mind://{id}/working_memory`** - Current working memory (JSON)
- **`mind://{id}/daily_memories`** - Unconsolidated memories list

### Integration Flow

```
Godot (GDScript)
  → McpMindClient (serializes CompositeObservation)
    → McpSdkClient (C#, manages WebSocket)
      → MCP Server (Python, FastMCP)
        → Mind.decide_action()
          → CognitivePipeline.process()
            → Returns Action dict
              → Flows back to Godot for execution
```

## Key Commands

### Server
```bash
# Start MCP server (default: localhost:3000)
uv run python -m mind.interfaces.mcp.main

# With environment variable for API key
export OPENROUTER_API_KEY="your-key"
uv run python -m mind.interfaces.mcp.main
```

### Testing
```bash
# Run all tests
uv run pytest

# Integration tests only
uv run pytest tests/integration -v

# The scope CI and the pre-commit hook actually gate (the two OpenRouter-live
# integration files are excluded from both)
uv run pytest tests/unit tests/integration/test_http_endpoints.py     tests/integration/test_memory_retrieval.py -q

# Interactive development
jupyter notebook notebooks/test_cognitive_pipeline.ipynb
```

### Development
```bash
# Build/refresh the environment from uv.lock (never re-resolves)
uv sync --frozen

# Re-resolve after editing pyproject.toml, then commit uv.lock
uv lock

# Opt in to the CUDA torch build (the default is CPU - see pyproject.toml)
uv sync --no-group cpu --group cuda

# Check code structure
tree src/mind/cognitive_architecture/nodes/
```

The environment lives wherever `UV_PROJECT_ENVIRONMENT` points, and `mind/.venv`
only when it points nowhere. The simulation repo's `tools/setup_mind.sh` sets it to
a shared path outside this tree, so one environment serves every worktree.

### Commit hooks and notebook stripping

`.pre-commit-config.yaml` lives at the **repo root**, one level above this file.
Install and verify from the root:

```bash
cd ..                       # repo root, not mind/
pre-commit install
ls "$(git rev-parse --git-common-dir)/hooks/pre-commit"
```

That must be a real hook; only `*.sample` files means nothing is installed. Use
`--git-common-dir` rather than a literal `.git/hooks`: in a **linked worktree** `.git`
is a file, so `ls .git/hooks/pre-commit` fails with `Not a directory`, which reads
like a missing hook rather than a wrong path. Hooks live in the common dir, so one
install covers every worktree.

**Nothing runs locally until you do this.** Hooks are machine-local: a fresh clone
installs none. CI (`.github/workflows/ci.yml`, NPC-1029) runs the same gates — ruff
check, ruff format, and the pytest scope the pytest-unit hook mirrors — on every PR,
so skipped hooks surface at PR time rather than never. Locally an uninstalled hook
set is still silent, so confirm the file exists rather than assuming (NPC-1024).

**mypy is configured but held at `stages: [manual]`**, so it does not run on commit:

```bash
pre-commit run --hook-stage manual mypy --all-files
```

It currently reports 41 errors across 12 files, so gating commits on it would block work
it did not cause. Cleanup is tracked as NPC-1034; promote the hook out of `stages` once
it is clean.

**Treat that count as a floor, not a measure of the debt.** It is whatever the current
resolution settings surface, and it has already moved once — 20 errors in 10 files
before `mypy_path` made the internal imports resolve. Re-measure after any change to
`[tool.mypy]` rather than quoting the number.

Two things make the hook work, and both are easy to break:

- It is a `local` hook that `cd`s into `mind/`, because mypy resolves its config from
  the **current directory only** — unlike ruff, which walks up from each file — and
  pre-commit runs hooks from the repo root, which has no `pyproject.toml`. Run from the
  root, mypy loads no config at all and reports `Config File: Default`.
- Every setting lives in `mind/pyproject.toml`'s `[tool.mypy]`, and the hook entry
  passes **no flags**. This is deliberate: `uv run mypy src/mind` — the obvious
  thing to run by hand — must behave identically to the hook.
  `mypy_path = "$MYPY_CONFIG_FILE_DIR/src"` is the load-bearing one; without it the
  absolute `from mind.…` imports resolve to nothing and `ignore_missing_imports`
  quietly degrades them to `Any`, so the check passes by not looking. Keep the
  `$MYPY_CONFIG_FILE_DIR` prefix: mypy resolves a relative `mypy_path` against the
  **working directory**, not against the config file, and an entry pointing at a
  directory that does not exist is dropped **silently** — so simplifying it back to
  `"src"` restores the shallow check with no signal that anything changed.

Notebook outputs are stripped by the `nbstripout` **pre-commit hook**, and only by
that hook. `mind/.gitattributes` no longer declares `filter=nbstripout`, because a
clean filter runs on `git add` and checkout independently of pre-commit and its
configuration is uncommittable: the interpreter path and `required = true` live in
machine-local `.git/config`. Linked worktrees share that config but have different
roots, so the relative path it held resolved per worktree, and any worktree without
`mind/.venv` aborted routine git commands outright:

```
mind/.venv/bin/python -m nbstripout: 1: mind/.venv/bin/python: not found
fatal: <notebook>: clean filter 'nbstripout' failed
```

**If you still have that filter configured locally, drop it** — the attribute is
gone, so it is inert for `.ipynb`, but the stale config is a trap for anyone who
re-adds an attribute later:

```bash
git config --unset-all filter.nbstripout.clean
git config --unset-all filter.nbstripout.smudge
git config --unset-all filter.nbstripout.required
```

The hook covers strictly more than the filter did — `.gitattributes` scoped it to
`mind/**`, so the notebooks under the repo-root `archived/` tree were never stripped.
What the hook cannot cover is `git commit --no-verify`, or an IDE committing with
hooks disabled; the Claude review workflow flags committed notebook outputs, so that
case is caught at review time rather than commit time.

## Current Development Focus

See [docs/planning/roadmap.md](docs/planning/roadmap.md) for detailed planning.

**Near-term priorities:**
1. End-to-end integration testing with npc-simulation repo
2. Prompt refinement (remove meta-cognitive framing)
3. Conversation history formalization (optional)

**Future features:**
- Planning system (multi-timescale behavioral coherence)
- Enhanced memory retrieval (contrastive queries, reflections)
- Emotional & social intelligence (mood-congruent recall, theory of mind)

## Documentation

- **[docs/README.md](docs/README.md)** - Overview and architecture
- **[docs/cognitive_architecture/overview.md](docs/cognitive_architecture/overview.md)** - Vision and design philosophy
- **[docs/interfaces/mcp.md](docs/interfaces/mcp.md)** - MCP protocol details
- **[docs/planning/roadmap.md](docs/planning/roadmap.md)** - Development roadmap and priorities
- **[docs/planning/backlog/](docs/planning/backlog/)** - Feature specifications

## Design Principles

- **Flexibility over rigidity** - Let structure emerge from use
- **Simplicity first** - Add complexity only when needed
- **Type safety** - Strong typing with Pydantic models throughout
- **Observability** - Automatic timing/token tracking via base classes
- **Test-driven** - Comprehensive test coverage (integration + notebook)

## Technology Stack

- **LangGraph** - Async pipeline orchestration
- **LangChain** - LLM abstraction (via OpenRouter)
- **ChromaDB** - Vector database for semantic memory
- **Pydantic** - Type-safe data models
- **FastMCP** - MCP protocol implementation
- **uv** - Dependency management and environment resolution
