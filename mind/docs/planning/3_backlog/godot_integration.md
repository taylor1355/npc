# Godot ↔ Mind Integration Contract

Current contract between the Godot simulation and Python mind server.

## Overview

The Godot simulation uses a **SimpleMind** for local rule-based decision-making and an **MCP Mind Server** (Python) for LLM-powered cognitive architecture. Both implement the same interface contract.

## Data Flow

```
Godot Simulation
    ↓ MindRequest (events, timestamp)
Mind Server (Python) OR SimpleMind (GDScript)
    ↓ process_observation()
Mind Response (chosen action)
    ↓
Godot Simulation executes action
```

## Core Contract

### 1. create_agent

Create a new NPC agent.

**Parameters:**
- `agent_id: String` - Unique identifier
- `config: Dictionary` with:
  - `traits: Array[String]` - Personality traits
  - `initial_long_term_memories: Array[String]` - Starting memories (optional)

**Returns:**
```gdscript
{
    "status": "created",  # or "error"
    "agent_id": String
}
```

### 2. process_observation

Process events and choose action.

**Input:** `MindRequest`
- `npc_id: String` - Agent identifier
- `events: Array[MindEvent]` - Event stream
- `timestamp: int` - Current simulation time (game minutes)

**Returns:** `MindResponse`
- `status: Status` - SUCCESS or ERROR
- `action: BaseAction` - Chosen action

### 3. cleanup_agent

Remove agent and free resources.

**Parameters:**
- `agent_id: String`

**Returns:**
```gdscript
{
    "status": "removed",  # or "error"
    "agent_id": String
}
```

### 4. get_agent_info

Get agent state.

**Parameters:**
- `agent_id: String`

**Returns:**
```gdscript
{
    "status": "active",  # or "error"
    "traits": Array[String]
}
```

## Event Types (MindEvent)

Godot sends these event types:

- **OBSERVATION** - Status updates (position, needs, vision)
- **ERROR** - Error messages
- **INTERACTION_BID_PENDING** - Outgoing bid waiting
- **INTERACTION_BID_RECEIVED** - Incoming bid from another NPC
- **INTERACTION_BID_REJECTED** - Bid was rejected
- **INTERACTION_STARTED** - Interaction began
- **INTERACTION_OBSERVATION** - Interaction updates (e.g., conversation messages)
- **INTERACTION_CANCELED** - Interaction canceled
- **INTERACTION_FINISHED** - Interaction completed

## Observation Types

Wrapped in **CompositeObservation**:

### StatusObservation
- `position: Vector2i` - Grid position
- `movement_locked: bool` - Can move?
- `controller_state: String` - Current controller state

### NeedsObservation
- `needs: Dictionary` - Need values (Hunger, Energy, Fun, Hygiene)

### VisionObservation
- `visible_entities: Array[EntityData]` - What the NPC can see

### ConversationObservation
- `participants: Array` - Who's in the conversation
- `conversation_history: Array` - Recent messages

## Action Types (BaseAction)

The mind must return one of:

- **WaitAction** - Idle for duration
- **WanderAction** - Random movement
- **MoveToAction** - Move to specific position
- **InteractWithAction** - Request interaction with entity
- **ContinueAction** - Continue current behavior
- **RespondToInteractionBidAction** - Accept/reject interaction bid
- **ActInInteractionAction** - Action within current interaction

## Current SimpleMind Behavior

The SimpleMind uses a state machine:

**States:**
- AgentIdleState - Default, handles incoming bids, chooses actions based on needs
- AgentWanderingState - Random movement
- AgentMovingToTargetState - Targeted movement
- AgentBiddingState - Waiting for bid response
- AgentInteractingState - In an interaction

**Decision Logic:**
1. Sync state with controller
2. Process event stream
3. Update observations and needs
4. Let current state choose action
5. Return chosen action

## MCP Mind Server (Python) - Current Implementation

**Location:** `/home/hearn/projects/npc/mind/src/mind/mcp_server.py`

**Status:** 🔴 Uses old `Agent` class, needs update to `CognitivePipeline`

**Tools:**
- `create_agent` - ✅ Implemented
- `decide_action` - ⚠️ Uses old architecture
- `cleanup_agent` - ✅ Implemented

**Resources:**
- `agent://{id}/mental_state` - Returns personality, working memory, long-term memories

**Issues:**
1. Uses old `mind.agent.agent.Agent` instead of new `CognitivePipeline`
2. Doesn't handle MindRequest/MindResponse format
3. `decide_action` expects old format (observation string + action list)
4. Doesn't process event stream
5. Returns simple action name, not BaseAction objects

## Required Changes for Integration

### Python Mind Server

**File:** `mcp_server.py`

1. **Replace Agent with CognitivePipeline**
   ```python
   from mind.cognitive_architecture.pipeline import CognitivePipeline
   from mind.cognitive_architecture.state import PipelineState
   from mind.cognitive_architecture.models import ObservationContext, AvailableAction
   from mind.cognitive_architecture.memory.vector_db_memory import VectorDBMemory
   ```

2. **Update create_agent**
   - Initialize CognitivePipeline + VectorDBMemory
   - Store per-agent pipeline instances
   - Seed long-term memories if provided

3. **Implement process_observation**
   - Accept MindRequest format
   - Convert events → observation text
   - Extract available actions from current state
   - Call pipeline.process()
   - Convert Action → GDScript action format
   - Return MindResponse

4. **Update get_agent_info / mental_state resource**
   - Access pipeline state
   - Return working_memory, daily_memories, cognitive_context
   - Format for Godot consumption

### Event → Observation Conversion

Need to convert MindEvent stream to natural language observation:

**Example:**
```
Events:
- StatusObservation(position=(5,10), needs={Hunger: 75%, Energy: 60%})
- VisionObservation(entities=[Tom at (6,10), Forge at (4,10)])
- ConversationObservation(history=[Tom: "Good morning!"])

→ Observation Text:
"You are at position (5,10). Hunger: 75%, Energy: 60%. You see Tom at (6,10) and the Forge at (4,10). Tom said: 'Good morning!'"
```

### Action Format Conversion

Need to map CognitivePipeline Action → Godot BaseAction:

**Python Action:**
```python
Action(
    action="interact_with",
    parameters={"entity_id": "tom_001", "interaction_name": "conversation"}
)
```

**Godot Action (JSON):**
```json
{
    "type": "interact_with",
    "entity_id": "tom_001",
    "interaction_name": "conversation"
}
```

## Proposed MCP Server Interface

### Tool: process_observation

**Input:**
```json
{
    "agent_id": "npc_001",
    "request": {
        "npc_id": "npc_001",
        "events": [
            {
                "type": "OBSERVATION",
                "timestamp": 1234,
                "payload": {...}
            }
        ],
        "timestamp": 1234
    }
}
```

**Output:**
```json
{
    "status": "SUCCESS",
    "action": {
        "type": "interact_with",
        "entity_id": "tom_001",
        "interaction_name": "conversation"
    }
}
```

## Memory System Integration

**Simulation Time:**
- Godot sends `timestamp` (game minutes)
- Python stores as `Memory.timestamp` for recency calculations
- Enables temporal reasoning

**Simulation Location:**
- Extract from StatusObservation
- Store as `Memory.location` (x, y)
- Enables spatial reasoning

**Memory Consolidation:**
- Daily memories accumulate during active play
- Call `consolidate_daily_memories()` during NPC sleep/downtime
- Moves memories to long-term storage

## Testing Strategy

1. **Unit Tests** - Test event→observation conversion
2. **Integration Tests** - Mock Godot events, verify actions
3. **End-to-End** - Run Godot + Python server, observe behavior
4. **Performance** - Measure latency with multiple NPCs

## Next Steps

1. Update `mcp_server.py` to use CognitivePipeline
2. Implement event→observation formatter
3. Implement action format converter
4. Add consolidation trigger (sleep event)
5. Test with Godot simulation
6. Document new fields/capabilities in Godot

## Open Questions

1. **Action Parameters** - How to handle complex parameters (e.g., conversation messages)?
2. **Error Handling** - What if pipeline fails? Fallback to SimpleMind?
3. **Performance** - Acceptable latency for NPC decisions?
4. **State Sync** - Should Python track controller state like SimpleMind does?
5. **Memory Seeding** - How to initialize working_memory from Godot?
