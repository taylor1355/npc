# Simulation Metadata Tool

## Problem

Currently, the LLM must guess how the simulation's systems work based on minimal context in prompts. This leads to:
- Misinterpretation of values (e.g., thinking Hunger 75% = hungry when it means well-fed)
- Incorrect parameter formatting (not knowing entity IDs, interaction names, coordinate format)
- Inability to reason about game mechanics (vision range, state transitions, interaction requirements)

**Short-term workaround**: Hardcode key mechanics in prompts (need system, interaction effects, parameter formats)

**Long-term solution**: Expose simulation metadata as a tool

---

## Proposed Solution

### Add `get_simulation_info` Tool

Allow the LLM to query simulation mechanics during reasoning:

```python
@tool
def get_simulation_info(query: str) -> str:
    """Get information about simulation systems and mechanics.

    Args:
        query: What to learn about. Options:
            - "need_system" - How needs work, ranges, urgency thresholds
            - "interactions" - How to format interaction actions
            - "action_parameters" - Expected parameter formats for each action
            - "state_machine" - Valid state transitions
            - "vision_system" - How NPCs perceive environment

    Returns:
        Markdown documentation for the queried system
    """
```

### Implementation Phases

#### Phase 1: Static Documentation Tool
- Parse Godot docs from `/mnt/c/Users/hearn/Dev/gamedev/npc-simulation/docs`
- Expose as tool callable during cognitive processing
- Cache docs in memory for fast access

#### Phase 2: Dynamic Metadata via MCP
- Godot exposes `get_game_metadata` MCP tool
- Returns current game state info:
  - Need decay rates for this NPC
  - Vision range configuration
  - Available interaction types at current location
  - State machine context

#### Phase 3: Learning System
- Track which metadata queries improve decision quality
- Proactively provide relevant info based on observation type
- Build NPC-specific understanding of game mechanics over time

---

## Benefits

**Immediate**:
- Correct interpretation of game values
- Proper parameter formatting
- Informed reasoning about mechanics

**Long-term**:
- Adapts to game mechanic changes automatically
- Works across different game projects (RPG vs Sim vs Strategy)
- NPCs can "learn" game mechanics through experience
- Reduces prompt engineering burden

---

## Implementation Complexity

**Phase 1: Medium** (2-3 hours)
- Parse existing Godot docs
- Create tool definition
- Wire into cognitive pipeline
- Test with realistic scenarios

**Phase 2: High** (4-6 hours)
- Requires Godot-side MCP tool implementation
- Coordinate serialization formats
- Handle async tool calls in pipeline
- Integration testing

**Phase 3: Very High** (1-2 weeks)
- Requires tracking query effectiveness
- Proactive information retrieval logic
- Per-NPC metadata caching
- Learning algorithm design

---

## Priority

**Medium** - Not urgent but valuable

Current workaround (hardcoded prompts) is sufficient for initial development. Implement Phase 1 when:
1. Mechanics become more complex
2. Working across multiple game projects
3. Frequent mechanic changes during development

---

## Related Work

- Phase 2 tool-based updates (roadmap.md)
- Memory metadata enhancement (simulation_time, location)
- MCP protocol better usage (Phase 0)

---

## Success Metrics

- Zero need value misinterpretations in test scenarios
- 100% correct parameter formatting
- NPCs reason explicitly about game mechanics in traces
- Faster onboarding to new game projects (measure by time to first correct decision)
