# Godot ↔ Python MCP Integration Plan

## Current State Analysis

### What We Have Built (Python Side)

**Location**: `/home/hearn/projects/npc/mind/src/mind/interfaces/mcp/`

**MCP Server** ([server.py](../../src/mind/interfaces/mcp/server.py)):
- FastMCP-based server with tools and resources
- `Mind` class with cognitive pipeline integration
- Structured observations (StatusObservation, NeedsObservation, VisionObservation, ConversationObservation)

**Tools**:
1. `create_mind(mind_id, config)` - Creates new Mind with CognitivePipeline
2. `decide_action(request)` - Processes SimulationRequest → returns MindResponse
3. `consolidate_memories(mind_id)` - Consolidates daily memories to long-term storage
4. `cleanup_mind(mind_id)` - Removes mind and frees resources

**Resources**:
- `mind://{mind_id}/state` - Complete mental state (working memory, daily memories, conversation histories)
- `mind://{mind_id}/working_memory` - Current working memory
- `mind://{mind_id}/daily_memories` - Unconsolidated daily memories

**Data Models** ([models.py](../../src/mind/interfaces/mcp/models.py)):
```python
# Input
class SimulationRequest(BaseModel):
    mind_id: str
    observation: Observation  # Structured observations

# Output
class MindResponse(BaseModel):
    status: str  # "success" or "error"
    action: Action | None
    error_message: str | None

# Configuration
class MindConfig(BaseModel):
    entity_id: str
    traits: list[str]
    llm_model: str = "google/gemini-2.0-flash-lite-001"
    embedding_model: str = "all-MiniLM-L6-v2"
    memory_storage_path: str = "./chroma_db"
    initial_working_memory: WorkingMemory | None = None
    initial_long_term_memories: list[str] = []
```

### What Godot Expects (Godot Side)

**Location**: `/mnt/c/Users/hearn/Dev/gamedev/npc-simulation/src/clients/mcp/`

**MCP Client** (mcp_mind_client.gd):
- C# SDK bridge for WebSocket communication
- Event formatter that converts game state to natural language
- Async request/response pattern with retry logic

**Expected MCP Tools**:
1. `create_agent(agent_id, config)` - ✅ Maps to our `create_mind`
2. `decide_action(agent_id, observation, available_actions)` - ⚠️ Different format
3. `cleanup_agent(agent_id)` - ✅ Maps to our `cleanup_mind`

**Request Format** (what Godot sends):
```python
{
    "agent_id": "npc_001",
    "observation": "Position: (10, 5)\nHunger: 45%\nEnergy: 80%\n...",  # Natural language!
    "available_actions": ["move_to", "interact_with", "wait", "wander", "continue"]
}
```

**Response Format** (what Godot expects):
```python
{
    "action": "interact_with",
    "parameters": {
        "entity_id": "item_food_1",
        "interaction_name": "eating"
    }
}
```

**Godot Action Types** (BaseAction subclasses):
- `WaitAction` - Idle for duration
- `WanderAction` - Random movement
- `MoveToAction` - Move to position (x, y)
- `InteractWithAction` - Request interaction (entity_id, interaction_name)
- `ContinueAction` - Continue current behavior
- `RespondToInteractionBidAction` - Accept/reject interaction (bid_id, accept: bool)
- `ActInInteractionAction` - Action within interaction (interaction_id, message: str)
- `CancelInteractionAction` - Cancel interaction (interaction_id)

## The Gap

### Format Mismatch

| Aspect | Python MCP Server | Godot Client |
|--------|------------------|--------------|
| **Tool Name** | `decide_action` | `decide_action` ✅ |
| **Input Format** | `SimulationRequest` with structured `Observation` | Flat text `observation` string |
| **Available Actions** | Extracted from Observation.vision.visible_entities | Explicit list parameter |
| **Output Format** | `MindResponse` with `Action` model | Dictionary with `action` and `parameters` |
| **Mind ID** | `mind_id` | `agent_id` |

### Key Differences

**1. Observation Format**
- **Godot sends**: Natural language string formatted by `EventFormatter`
  ```
  "Position: (10, 5)\nHunger: 45%\nEnergy: 80%\nVisible: Oak Tree (can examine), Campfire (can rest)\n"
  ```
- **Python expects**: Structured `Observation` with typed sub-observations
  ```python
  Observation(
      entity_id="npc_001",
      current_simulation_time=1234,
      status=StatusObservation(position=(10, 5), ...),
      needs=NeedsObservation(needs={"hunger": 45.0, ...}),
      vision=VisionObservation(visible_entities=[...])
  )
  ```

**2. Available Actions**
- **Godot sends**: List of action names `["move_to", "interact_with", ...]`
- **Python expects**: Extracted from `Observation.vision.visible_entities[].interactions`

**3. Action Response**
- **Python returns**: `Action(action="interact_with", parameters={"entity_id": "...", ...})`
- **Godot expects**: Plain dict `{"action": "interact_with", "parameters": {...}}`

## Integration Strategy

### Option 1: Adapter Layer (Recommended)

Create a thin adapter between Godot's format and our cognitive architecture format.

**Pros**:
- Preserves our clean structured observation architecture
- Godot client unchanged (battle-tested)
- Clear separation of concerns

**Cons**:
- Extra conversion step
- Potential information loss from natural language → structured

**Implementation**:
```python
# New file: src/mind/interfaces/mcp/godot_adapter.py

class GodotMcpAdapter:
    """Adapts Godot's MCP format to our cognitive architecture format"""

    def parse_observation_text(
        self,
        observation_text: str,
        entity_id: str
    ) -> Observation:
        """Parse Godot's natural language observation into structured format"""
        # Extract position: "Position: (10, 5)"
        # Extract needs: "Hunger: 45%"
        # Extract vision: "Visible: Oak Tree (can examine)"
        # Return Observation(...)

    def action_to_dict(self, action: Action) -> dict:
        """Convert Action model to Godot dict format"""
        return {
            "action": action.action,
            "parameters": action.parameters
        }
```

**Modified MCP Tools**:
```python
@self.mcp.tool()
async def decide_action(
    agent_id: str,  # Match Godot's naming
    observation: str,  # Accept text
    available_actions: list[str],  # Explicit list
    ctx: Context = None
) -> dict:  # Return plain dict
    """Godot-compatible decide_action endpoint"""
    adapter = GodotMcpAdapter()

    # Convert to our format
    structured_obs = adapter.parse_observation_text(observation, agent_id)
    request = SimulationRequest(
        mind_id=agent_id,
        observation=structured_obs
    )

    # Use existing logic
    response = await self._internal_decide_action(request)

    # Convert back to Godot format
    return adapter.action_to_dict(response.action)
```

### Option 2: Dual Interface

Maintain both interfaces - one for Godot (text-based), one for future structured clients.

**Tools**:
- `decide_action` - Godot-compatible (text observation)
- `decide_action_structured` - Future clients (structured Observation)

**Pros**:
- Both interfaces available
- No breaking changes

**Cons**:
- Code duplication
- Maintenance burden

### Option 3: Update Godot Client

Modify Godot to send structured observations instead of text.

**Pros**:
- Clean interface
- Type safety end-to-end

**Cons**:
- Major Godot refactor
- Breaking change to battle-tested code
- Loses natural language observations for debugging

## Recommended Approach

**Phase 1: Adapter Layer (This PR)**
1. Create `GodotMcpAdapter` class
2. Add Godot-compatible tool wrappers:
   - `create_agent` → wraps `create_mind`
   - `decide_action` → parses text, calls internal pipeline, returns dict
   - `cleanup_agent` → wraps `cleanup_mind`
3. Keep existing structured tools for future use
4. Add integration tests that mimic Godot requests

**Phase 2: Observation Parser (Next PR)**
1. Robust text → structured observation parser
2. Handle edge cases (missing data, malformed input)
3. Add parser unit tests

**Phase 3: Enhanced Integration (Future)**
1. Add `consolidate_memories` trigger from Godot (e.g., during sleep)
2. Mental state resources for debugging UI
3. Bidirectional MCP (Godot queries mind state)

## Implementation Details

### Observation Text Parser

The parser needs to extract structured data from Godot's formatted text:

**Input Example**:
```
Position: (10, 5)
Movement Locked: false
Hunger: 45%
Energy: 80%
Fun: 60%
Hygiene: 90%
Visible Entities:
  - Oak Tree (examine)
  - Campfire (rest)
  - Tom at (11, 5) (conversation)
```

**Parsing Strategy**:
```python
import re
from typing import Optional

def parse_observation_text(text: str, entity_id: str) -> Observation:
    # Extract position
    position_match = re.search(r"Position: \((\d+), (\d+)\)", text)
    position = tuple(map(int, position_match.groups())) if position_match else (0, 0)

    # Extract needs (percentages)
    needs = {}
    for need_name in ["Hunger", "Energy", "Fun", "Hygiene"]:
        match = re.search(rf"{need_name}: (\d+)%", text)
        if match:
            needs[need_name.lower()] = float(match.group(1))

    # Extract visible entities
    entities = []
    visible_section = re.search(r"Visible Entities:(.*?)(?:\n\n|$)", text, re.DOTALL)
    if visible_section:
        for line in visible_section.group(1).split("\n"):
            # Parse "  - Oak Tree (examine)" or "  - Tom at (11, 5) (conversation)"
            entity_match = re.match(r"\s*-\s+(.+?)\s+\((.+)\)", line.strip())
            if entity_match:
                display_name, interactions_str = entity_match.groups()
                # Create EntityData...

    return Observation(
        entity_id=entity_id,
        current_simulation_time=0,  # TODO: Extract from observation
        status=StatusObservation(position=position, ...),
        needs=NeedsObservation(needs=needs),
        vision=VisionObservation(visible_entities=entities)
    )
```

### Action Converter

Convert our Action model to Godot's dict format:

```python
def action_to_godot_dict(action: Action) -> dict:
    """Convert Action to Godot-compatible dictionary"""
    if action is None:
        # Fallback to CONTINUE
        return {"action": "continue", "parameters": {}}

    result = {
        "action": action.action,
        "parameters": action.parameters.copy()
    }

    return result
```

### Testing Strategy

**Unit Tests** ([test_godot_adapter.py](../../tests/test_godot_adapter.py)):
```python
def test_parse_simple_observation():
    text = "Position: (10, 5)\nHunger: 45%\nEnergy: 80%"
    obs = adapter.parse_observation_text(text, "npc_001")
    assert obs.status.position == (10, 5)
    assert obs.needs.needs["hunger"] == 45.0

def test_action_to_dict():
    action = Action(
        action="interact_with",
        parameters={"entity_id": "tree_001", "interaction_name": "examine"}
    )
    result = adapter.action_to_dict(action)
    assert result == {
        "action": "interact_with",
        "parameters": {"entity_id": "tree_001", "interaction_name": "examine"}
    }
```

**Integration Tests** (extend test_mcp_integration.py):
```python
async def test_godot_decide_action_format():
    """Test that Godot's text observation format works"""
    # Create mind
    await server.mcp.call_tool("create_agent", {...})

    # Send Godot-style request
    result = await server.mcp.call_tool("decide_action", {
        "agent_id": "npc_001",
        "observation": "Position: (10, 5)\nHunger: 45%\nVisible: Oak Tree (examine)",
        "available_actions": ["wait", "interact_with", "wander"]
    })

    response = json.loads(result[0].text)
    assert "action" in response
    assert "parameters" in response
```

## File Changes Required

### New Files
1. `src/mind/interfaces/mcp/godot_adapter.py` - Adapter class
2. `tests/test_godot_adapter.py` - Unit tests for adapter

### Modified Files
1. `src/mind/interfaces/mcp/server.py` - Add Godot-compatible tool wrappers
2. `tests/test_mcp_integration.py` - Add Godot format tests
3. `docs/integration/godot_mcp_integration_plan.md` - This document

## Open Questions

1. **Timestamp**: Where does Godot include simulation time in the observation text?
2. **Conversations**: How are conversation histories formatted in observation text?
3. **Available Actions**: Should we validate against the provided list, or extract from vision?
4. **Error Handling**: What should Godot receive if parsing fails?
5. **Fallback**: Should we cache the last valid observation for recovery?

## Next Steps

1. ✅ Document current state (this file)
2. ⬜ Implement `GodotMcpAdapter` class
3. ⬜ Add Godot-compatible tool wrappers to MCP server
4. ⬜ Write unit tests for adapter
5. ⬜ Add integration tests with Godot format
6. ⬜ Test with actual Godot simulation
7. ⬜ Document any discovered gaps or issues
