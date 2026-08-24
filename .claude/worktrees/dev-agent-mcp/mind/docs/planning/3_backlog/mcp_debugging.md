# MCP Mental State Debugging

## Problem Statement

Understanding why NPCs make certain decisions is currently opaque. Developers and curious players need visibility into the cognitive processes - what memories were retrieved, what plans are active, why certain actions were chosen. Without this, debugging is guesswork and players can't appreciate the depth of NPC cognition.

## Design Goals

- Expose internal mental state via MCP resources
- Make decision processes inspectable and traceable
- Support both development debugging and player curiosity
- Don't impact performance when not being observed

## Technical Approach

### Mental State Resources

Expose via MCP resources:

```
cognition://agent/{id}/working_memory    # Current context
cognition://agent/{id}/recent_memories   # Last N memories
cognition://agent/{id}/current_plan      # Active plan if any
cognition://agent/{id}/last_decision     # Trace of last action choice
cognition://agent/{id}/personality       # Traits and preferences
cognition://agent/{id}/relationships     # Social connections
```

### Decision Traces

For each decision, capture:
- What observations were received
- Which memories were retrieved and why
- How needs influenced the choice
- What alternatives were considered
- Why the final action was selected

### Integration Points

1. **MCP Playground**: Inspect any agent's mental state
2. **Game UI**: Optional "mind reading" debug panel
3. **Development Tools**: VS Code extension for real-time monitoring
4. **Analytics**: Track patterns across many agents

### Performance Considerations

- Only generate detailed traces when resource is queried
- Cache recent state for quick access
- Use lazy evaluation for expensive computations

## Benefits

- **Development Velocity**: Dramatically faster debugging
- **Player Engagement**: "Mind reading" reveals NPC depth
- **Quality Assurance**: Can verify NPCs behave as intended
- **Research Tool**: Understand emergent behaviors

## Dependencies

- Basic MCP resource implementation
- Mental state to expose (memory, plans, etc.)
- Decision tracking in agent logic

## Priority Rationale

**Obviousness**: Moderate - Clear need for debugging, multiple ways to implement.

**Development Velocity**: Extremely positive
- Short-term: Implementing resource exposure takes modest effort
- Long-term: Dramatically accelerates all debugging and development
- Tech tree: Doesn't unlock new features but makes everything else easier
- Net: Massive multiplier for development speed

**Concreteness**: Zero - Pure developer tool with no player-visible impact. However, indirectly enables better NPC behaviors by making them easier to debug and refine.

This is a pure development accelerator - high velocity, no concreteness.