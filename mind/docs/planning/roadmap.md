# Cognitive Architecture Development Roadmap

## Overview

This roadmap outlines the phased evolution of the NPC cognitive architecture, building on the existing sophisticated prompt engineering while streamlining execution for efficiency.

## Current State Analysis

### Existing Pipeline (5 LLM calls per decision)
1. **Generate memory queries** (~100-200 tokens) - ✅ KEEP: Critical for RAG performance
2. **Create memory report** (~500+ tokens) - ❌ REDUNDANT with working memory update
3. **Update working memory** (~500+ tokens) - ❌ MERGE with memory report
4. **Generate long-term memories** (~300+ tokens) - ⚠️ Make selective/conditional
5. **Decide on action** (~300+ tokens) - ✅ KEEP: Core functionality

**Total**: ~1800+ tokens per decision cycle

### Identified Issues
- Memory report → working memory update is redundant consolidation
- Every observation triggers memory formation (should be selective)
- Only first retrieved memory used per query (wastes retrieval potential)
- Working memory is unstructured string (loses rich psychological framework)
- LlamaIndex is showing its age - less ergonomic than modern alternatives
- MCP protocol underutilized - treating it like REST instead of leveraging its features
- **Memories lack contextual metadata** - No simulation time, location, or ID for citation during reasoning
- **Prompts use meta-cognitive framing** - LLM plays "cognitive scientist analyzing NPC" instead of directly inhabiting character (simulating a Turing machine with another Turing machine)

## Phase 0: Test Infrastructure & MCP Usage (HIGH PRIORITY)

### Goal
Create minimal test setup and properly use MCP protocol features.

### Deliverables

#### 0.0 Prompt Architecture Redesign
- **Problem**: Current prompts ask LLM to play a "cognitive scientist" analyzing the NPC rather than directly inhabiting the character
  - Creates unnecessary indirection (simulating a Turing machine with another Turing machine)
  - Wastes tokens on meta-cognitive framing instead of character expression
  - Reduces authenticity of NPC behavior
- **Solution**: Rewrite prompts to directly embody the character while still leveraging cognitive science principles
  - Generate character-specific system prompt that establishes the NPC's identity, personality, and perspective
  - Reframe cognitive operations (memory formation, decision-making, etc.) as first-person mental processes
  - Preserve psychological sophistication without the "person driving a person" dynamic
  - Example transformation:
    ```
    # Current (meta-cognitive)
    "As a cognitive scientist, analyze how this NPC would react..."

    # Improved (direct embodiment)
    "You are [Character Name]. Consider how this situation affects you..."
    ```
- **Benefits**:
  - More natural, authentic NPC behavior
  - Reduced token overhead from meta-framing
  - Better alignment between prompt structure and cognitive architecture goals

#### 0.1 Test Notebook
- Simple Jupyter notebook to test single decision cycles
- Spin up MCP server
- Set initial mind state (working memory, some memories, personality)
- Test scenarios (single decisions):
  - Choose between available actions
  - Retrieve relevant memories
  - Update cognitive state
- Measure tokens and latency

#### 0.2 Better MCP Usage
- Use MCP's structured types instead of JSON strings:
  ```python
  # Current
  def decide_action(observation: str, actions: str) -> str:
      return json.dumps({"action_index": 0})

  # Better
  @mcp.tool()
  def decide_action(
      observation: Observation,
      available_actions: List[Action]
  ) -> ActionDecision:
      return ActionDecision(action_index=0, confidence=0.8)
  ```
- Let MCP handle serialization/deserialization
- Use proper request/response types

### Success Metrics
- Notebook runs end-to-end
- Baseline token counts recorded
- MCP handling structured data properly

## Phase 1: Smart Streamlining (Weeks 1-2)

### Goal
Eliminate redundancy while preserving psychological sophistication. Reduce to ~1100 tokens typical.

### Deliverables

#### 1.1 Consolidate Memory Processing
- Merge memory_report and working_memory prompts into single "cognitive_update" prompt
- Preserve psychological frameworks from both prompts
- Output structured but flexible cognitive state

#### 1.2 Enhance Memory Retrieval & Metadata
- Keep query generation (critical for RAG)
- Generate 3-5 diverse queries instead of 1-2
- Use top-k memories per query, not just first match
- Add relevance scoring to retrieved memories
- **Add rich metadata to memories**:
  - `simulation_time`: In-game time when memory was formed
  - `location`: Grid coordinates where event occurred
  - `memory_id`: Unique ID for concise citation (e.g., "M1234")
  - Format for LLM: `[M1234 | T:Day2-08:00 | L:(5,10)] Memory content here`
  - Enables NPCs to reason about temporal/spatial context and cite specific memories

#### 1.3 Selective Memory Formation
- Make long-term memory formation conditional
- Trigger on: significant events, goal relevance, emotional intensity
- Add importance scoring during formation

#### 1.4 Migrate to LangGraph
- Replace LlamaIndex with LangGraph for better ergonomics
- Use LangGraph's built-in state management
- Leverage graph structure for cognitive pipeline
- Better streaming and async support

### New Pipeline
1. Generate diverse queries from observation (~150 tokens)
2. Retrieve top-k memories for each query (no LLM)
3. Update cognitive state with observation + memories (~500 tokens)
4. Decide action from cognitive state (~300 tokens)
5. IF significant: form long-term memory (~300 tokens conditional)

### Success Metrics
- Token usage reduced by 40%
- Decision quality maintained (test with scenarios)
- Memory retrieval relevance improved
- LangGraph migration complete with tests passing

## Phase 2: Flexible Structure & Tool-Based Updates (Weeks 3-4)

### Goal
Implement flexible structured state with tool-based selective updates.

### Deliverables

#### 2.1 Flexible Cognitive State
Simple, extensible structure (not prescriptive):
```json
{
  "current_situation": "Working at the forge, feeling focused",
  "active_goal": "Complete the sword commission",
  "recent_events": ["Started work", "Greeted apprentice"],
  "current_plan": ["Finish blade", "Polish", "Deliver"],
  "emotional_state": "content but slightly tired",
  // Additional fields as needed by the NPC
  // Not all fields required - let it be organic
}
```

The LLM can add/modify fields as appropriate for the character and situation.

#### 2.2 Tool-Based Selective Updates
Define tools for cognitive updates:
```python
def update_situation(new_situation: str):
    """Update current understanding of situation"""

def add_recent_event(event: str):
    """Add to recent events list"""

def update_plan(plan_modification: dict):
    """Modify current plan (add/remove/reorder steps)"""

def set_emotional_state(emotion: str):
    """Update emotional state"""

def set_field(field: str, value: Any):
    """Generic field update for flexibility"""
```

LLM can call just the tools needed rather than regenerating entire state.

#### 2.3 LangGraph Cognitive Graph
Structure the cognitive pipeline as a graph:
- Nodes: observation, query_generation, retrieval, cognitive_update, action_decision
- Edges: conditional paths based on significance, plan status
- State: maintained between nodes
- Checkpointing: for debugging and recovery

#### 2.4 Query Enhancement
- Use current cognitive state to inform queries
- Add contrastive queries (what's different/unexpected)
- Query for goal-relevant memories
- Temporal queries based on similar situations

### Success Metrics
- Tool calls reduce token usage by additional 20%
- State remains coherent across updates
- Graph structure improves debuggability

## Phase 3: Planning & Adaptation (Weeks 5-6)

### Goal
Add planning capabilities while maintaining flexibility.

### Deliverables

#### 3.1 Flexible Planning
- Plans as loose intentions, not rigid schedules
- Multiple granularities without fixed hierarchy
- Natural language plans with optional structure
- Plans stored in cognitive state but not prescriptively

#### 3.2 Memory Scoring & Retrieval
- Importance scoring (1-10) during formation
- Recency decay in retrieval scoring
- Combined scoring: relevance * importance * recency_factor
- Different strategies for different query types

#### 3.3 Plan Adaptation
- Detect plan deviation through state changes
- Local adjustments via tool calls
- Full replan when necessary
- Learn from patterns over time

### Success Metrics
- Plans guide behavior without constraining it
- Natural deviation and recovery from plans
- Memory retrieval shows appropriate biases

## Phase 4: Emotional & Social Dimensions (Weeks 7-8)

### Goal
Add emotional and social awareness naturally.

### Deliverables

#### 4.1 Emotional Integration
- Emotional state affects memory retrieval
- Mood-congruent recall patterns
- Emotional memories tagged during formation
- Natural emotional evolution through state updates

#### 4.2 Social Awareness
- Track relationships in cognitive state (flexibly)
- Social context affects memory queries
- Shared experiences have higher salience
- Tool calls for relationship updates

#### 4.3 Personality Expression
- Personality influences cognitive updates naturally
- Not forced traits but emergent patterns
- Individual differences in how state is maintained
- Gradual evolution through experience

### Success Metrics
- Emotional continuity across interactions
- Social relationships affect behavior naturally
- Personality differences emerge without forcing

## Implementation Guidelines

### Flexibility Principles
- Avoid rigid schemas that constrain NPC expression
- Let structure emerge from use rather than prescribing it
- Tools enable selective updates without full regeneration
- Keep the human-readable quality of the state

### LangGraph Migration
- Start with simple graph for existing pipeline
- Gradually add nodes and conditional edges
- Use built-in persistence for state management
- Leverage streaming for real-time updates

### Tool Design
- Tools should be atomic and composable
- Generic tools (set_field) for flexibility
- Specific tools for common operations
- Let LLM decide which tools to use

### Testing Strategy
- Test flexibility with diverse NPC personalities
- Ensure state coherence across tool updates
- Verify graph execution paths
- Monitor token usage reduction

## Appendix: Technology Choices

### Why LangGraph over LlamaIndex
- **Better State Management**: First-class state with persistence
- **Graph Structure**: Natural fit for cognitive pipeline
- **Tool Integration**: Native tool calling support
- **Modern Ergonomics**: Cleaner API, better typing
- **Streaming**: Built-in streaming and async
- **Debugging**: Visual graph exploration, checkpointing

### Tool-Based Updates Benefits
- **Efficiency**: Update only what changed (~50-100 tokens vs 500)
- **Coherence**: Preserve unchanged state naturally
- **Flexibility**: Add new tools without changing prompts
- **Debugging**: See exactly what's being modified
- **Concurrency**: Multiple tools can execute in parallel

## Next Steps

1. Set up LangGraph environment and migrate memory database
2. Create flexible cognitive state schema
3. Implement tool definitions for state updates
4. Build graph structure for cognitive pipeline
5. Test streamlined pipeline with tool-based updates