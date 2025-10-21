# Testing Plan - Mind Cognitive Architecture

**Created:** October 19, 2025

## Overview

This document outlines the testing strategy for the mind cognitive architecture, covering unit tests, integration tests, and end-to-end validation with the npc-simulation Godot project.

## Testing Layers

```
┌─────────────────────────────────────┐
│   End-to-End Tests                  │  ← Full stack with Godot
│   (Manual + Automated)              │
├─────────────────────────────────────┤
│   Integration Tests                 │  ← Full pipeline with mocks
│   (pytest + cheap LLM)              │
├─────────────────────────────────────┤
│   Unit Tests                        │  ← Individual components
│   (pytest + mocks)                  │
└─────────────────────────────────────┘
```

## 1. Unit Tests

### Goal
Verify individual components work in isolation with deterministic inputs.

### Scope

#### 1.1 Pipeline Nodes
Test each node independently with mocked dependencies.

**Memory Query Node:**
- ✅ Validates PROMPT_VARS correctly
- ✅ Generates semantic queries from observations
- ✅ Handles empty/minimal observations
- ✅ Token tracking works
- Mock: LLM responses

**Memory Retrieval Node:**
- ✅ Fetches memories from ChromaDB
- ✅ Deduplication by memory ID works
- ✅ Returns empty list when no memories found
- ✅ Handles invalid query gracefully
- Mock: VectorDBMemory

**Cognitive Update Node:**
- ✅ Updates working memory correctly
- ✅ Forms new memories with importance scores
- ✅ Handles empty retrieved memories
- ✅ Cognitive context has required keys
- Mock: LLM responses

**Action Selection Node:**
- ✅ Chooses valid action from available actions
- ✅ Includes required parameters for action type
- ✅ Uses cognitive context in decision
- ✅ Handles no available actions gracefully
- Mock: LLM responses

#### 1.2 Memory System
Test memory operations without LLM calls.

**VectorDBMemory:**
- ✅ Add memory with metadata
- ✅ Search returns relevant results
- ✅ Recency decay affects scoring
- ✅ Importance weighting works
- ✅ Clear removes all memories
- ✅ Handles empty collection

**Memory Models:**
- ✅ Memory validation (id, content, importance range)
- ✅ Metadata structure (timestamp, location)
- ✅ String formatting for LLM consumption

#### 1.3 State Management

**PipelineState:**
- ✅ Observation validation
- ✅ WorkingMemory default values
- ✅ Token/time tracking accumulation

**WorkingMemory:**
- ✅ Pydantic validation
- ✅ Extensible fields (extra="allow")
- ✅ String formatting

**Observation/Action Models:**
- ✅ Structured observation validation
- ✅ Action type and parameter validation
- ✅ Nested model validation (StatusObservation, NeedsObservation, etc.)

### Implementation

**Location:** `tests/unit/`

**Structure:**
```
tests/unit/
├── test_nodes.py           # Individual node tests
├── test_memory.py          # Memory system tests
├── test_models.py          # Pydantic model tests
└── fixtures.py             # Shared test fixtures
```

**Mocking Strategy:**
- Use `unittest.mock` for LLM calls
- Use in-memory ChromaDB for memory tests
- Use pytest fixtures for common test data

## 2. Integration Tests

### Goal
Verify full pipeline works with realistic data and cheap LLM.

### Scope

#### 2.1 Full Pipeline Tests
Test complete observation → action flow.

**Scenarios:**
- ✅ Basic decision with simple observation
- ✅ Decision with retrieved memories
- ✅ Memory formation during cognitive update
- ✅ Multiple decision cycles with state persistence
- ✅ Conversation history tracking
- ✅ Error handling (LLM failures, invalid responses)

**Test Data:**
Based on actual Godot observation structure from npc-simulation:
- Realistic NeedsObservation (hunger, energy, fun, hygiene)
- VisionObservation with visible entities and interactions
- StatusObservation with position and state
- ConversationObservation with interaction context

#### 2.2 MCP Server Integration
Test MCP tools with real server instance.

**Tools Testing:**
- ✅ `create_mind` - Initialize with config
- ✅ `decide_action` - Process observation dict
- ✅ `consolidate_memories` - Daily → long-term transfer
- ✅ `cleanup_mind` - Resource cleanup

**Resource Testing:**
- ✅ `mind://{id}/state` - State snapshot
- ✅ `mind://{id}/working_memory` - Working memory JSON
- ✅ `mind://{id}/daily_memories` - Unconsolidated memories

**Existing Test:**
- `tests/test_mcp_integration.py` already exists
- Verify it covers above scenarios
- Extend with realistic Godot-like observations

#### 2.3 Memory Consolidation
Test daily memory → long-term flow.

**Scenarios:**
- ✅ Consolidation with various importance scores
- ✅ Daily buffer clearing after consolidation
- ✅ Retrieval of consolidated memories
- ✅ Multiple consolidation cycles

### Implementation

**Location:** `tests/integration/`

**Structure:**
```
tests/integration/
├── test_pipeline_scenarios.py    # Full pipeline scenarios
├── test_mcp_tools.py             # MCP server tool tests
├── test_memory_lifecycle.py      # Memory formation → consolidation
└── fixtures/
    ├── godot_observations.py     # Realistic observation mocks
    └── test_configs.py           # Mind configurations
```

**LLM Configuration:**
- Use cheap model: `google/gemini-2.0-flash-lite-001` (current default)
- Set low temperature (0) for deterministic testing
- Mock LLM for fast tests, real LLM for integration validation

**Observation Fixtures:**
Create realistic observations based on Godot structure:
```python
def create_blacksmith_observation():
    """Blacksmith NPC at forge with low hunger"""
    return Observation(
        entity_id="blacksmith_001",
        current_simulation_time=1000,
        status=StatusObservation(
            position=(5, 10),
            movement_locked=False,
            current_interaction={},
            controller_state={"state": "idle"}
        ),
        needs=NeedsObservation(
            needs={
                "hunger": 25.0,  # Low!
                "energy": 70.0,
                "fun": 50.0,
                "hygiene": 80.0
            }
        ),
        vision=VisionObservation(
            visible_entities=[
                EntityData(
                    entity_id="bread_001",
                    display_name="Fresh Bread",
                    position=(6, 10),
                    interactions={
                        "eat": {
                            "name": "eat",
                            "description": "Eat the bread",
                            "needs_filled": ["hunger"],
                            "needs_drained": []
                        }
                    }
                )
            ]
        )
    )
```

## 3. End-to-End Tests

### Goal
Verify mind + npc-simulation integration works in real gameplay.

### Scope

#### 3.1 Manual Testing
Human-verified gameplay scenarios.

**Setup:**
1. Start MCP server: `poetry run python src/mind/interfaces/mcp/main.py`
2. Launch Godot project
3. Create NPC with MCP mind

**Test Scenarios:**
- ✅ NPC responds to low hunger (seeks food)
- ✅ NPC responds to low energy (seeks rest)
- ✅ NPC engages in social interactions
- ✅ NPC references past memories in decisions
- ✅ NPC forms new memories during play
- ✅ Server handles multiple NPCs simultaneously
- ✅ Performance: <5s per decision cycle
- ✅ No crashes or communication errors

**Validation:**
- Check Godot console for MCP communication logs
- Use `mind://{id}/working_memory` resource to inspect state
- Verify memory formation via `mind://{id}/daily_memories`

#### 3.2 Automated E2E (Future)
Scripted Godot scenarios with assertions.

**Approach:**
- Use Godot's GDScript test framework
- Create deterministic scenarios with SimpleMind baseline
- Compare MCP mind behavior against expectations
- Automate via CI/CD pipeline

**Not Priority Now** - Focus on manual validation first.

### Implementation

**Location:** `docs/testing/`

**Documents:**
```
docs/testing/
├── e2e-test-scenarios.md      # Manual test scenarios
├── e2e-results.md             # Test run results log
└── integration-issues.md      # Known issues and fixes
```

## Test Execution Plan

### Phase 1: Unit Tests (Current Priority)
**Effort:** ~1 day

1. Create test fixtures for observations/actions
2. Write node unit tests with mocked LLM
3. Write memory system tests
4. Write model validation tests
5. Achieve >80% code coverage for core components

**Success Criteria:**
- All unit tests pass
- Tests run in <10 seconds
- No external dependencies (LLM, ChromaDB persistence)

### Phase 2: Integration Tests
**Effort:** ~1 day

1. Enhance existing `test_mcp_integration.py`
2. Add realistic Godot observation fixtures
3. Test full pipeline with cheap LLM
4. Test memory lifecycle (formation → consolidation → retrieval)
5. Test error handling and edge cases

**Success Criteria:**
- Integration tests pass with real LLM
- Tests complete in <2 minutes
- Realistic observations from Godot structure

### Phase 3: End-to-End Validation
**Effort:** ~0.5 day

1. Document manual test scenarios
2. Start MCP server and Godot simulation
3. Execute test scenarios with human verification
4. Document any integration issues found
5. Fix critical issues, defer minor improvements

**Success Criteria:**
- NPCs make decisions via cognitive pipeline
- No crashes or communication errors
- Performance <5s per decision
- At least 3 different behavioral scenarios validated

## Test Data Strategy

### Mock Observations
Create a library of realistic observations based on common gameplay scenarios:

**Scenarios:**
- `hungry_npc_near_food` - Low hunger, food visible
- `tired_npc_near_bed` - Low energy, bed visible
- `social_npc_near_other` - High fun need, another NPC visible
- `working_npc_at_forge` - NPC engaged in crafting interaction
- `conversing_npcs` - Active conversation with history

**Storage:**
- `tests/fixtures/observations.py` - Observation factory functions
- `tests/fixtures/configs.py` - Mind configuration templates

### Test Assertions
Define clear assertions for different test types:

**Unit Tests:**
- Output types are correct
- Required fields are present
- State updates as expected

**Integration Tests:**
- Actions are valid for current state
- Memories are formed with correct importance
- Working memory maintains coherence
- Token usage is reasonable

**E2E Tests:**
- NPCs behave plausibly
- No error messages in logs
- Performance meets targets

## Continuous Testing

### Pre-commit Hooks (Future)
- Run unit tests before commit
- Check code formatting (black, isort)
- Type checking (mypy)

### CI/CD Pipeline (Future)
- Run all unit tests on push
- Run integration tests on PR
- E2E smoke tests before release

### Test Documentation
- Keep this plan updated as tests evolve
- Document new test scenarios as they're added
- Track known issues and workarounds

## Success Metrics

**Unit Tests:**
- ✅ >80% code coverage
- ✅ All tests pass
- ✅ Fast execution (<10s)

**Integration Tests:**
- ✅ Realistic scenarios pass
- ✅ Memory lifecycle works end-to-end
- ✅ Error handling prevents crashes

**E2E Tests:**
- ✅ NPCs make plausible decisions
- ✅ Performance acceptable (<5s)
- ✅ No communication failures

## Next Steps

1. ✅ Create unit test structure (`tests/unit/`)
2. ✅ Write node unit tests with mocked LLM
3. ✅ Write memory system tests
4. ✅ Enhance integration tests with Godot-like observations
5. ✅ Manual E2E testing with Godot simulation
6. Document findings and iterate

---

**See Also:**
- [Roadmap](roadmap.md) - Development priorities
- [MCP Integration Docs](../interfaces/mcp.md) - MCP protocol details
- [npc-simulation Three-Tier Architecture](https://github.com/taylor1355/npc-simulation/docs/three-tier-architecture.md)
