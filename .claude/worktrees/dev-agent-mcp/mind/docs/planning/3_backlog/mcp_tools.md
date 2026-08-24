# MCP Tools and Active Perception

## Problem Statement

NPCs currently receive pushed observations during decision cycles. They can't actively query for information they need or pull specific details on demand. Active perception through MCP tools would allow NPCs to query simulation state when needed, reducing unnecessary data transfer and enabling more sophisticated reasoning.

## Design Goals

- Enable NPCs to actively query simulation state
- Reduce token usage by pulling only needed information
- Support incremental decision refinement
- Maintain clean separation between mental and physical state

## Technical Approach

### Bidirectional MCP

As detailed in the simulation's bidirectional MCP plans:
- Godot becomes an MCP server exposing simulation resources
- Mind service can query physical state on demand
- Clean pull-based data flow

### Available Resources (Read-Only)

```
simulation://agent/{id}/physical_state
simulation://agent/{id}/perception
simulation://agent/{id}/needs
simulation://world/entities
simulation://world/time
```

These are queries for information, not commands to change state.

### Important: No Special Powers

NPCs do NOT get special tools to:
- Spawn items (must craft/find like anyone else)
- Teleport (must walk)
- Modify world state directly (must use interactions)

All world changes still go through:
1. NPC decides on action
2. Action creates bid
3. Bid leads to interaction
4. Interaction modifies world

Special cases are not special enough to break this flow.

## Benefits

- **Token Efficiency**: Pull only needed data
- **Decision Quality**: Can query for specific details
- **Debugging**: Clear view of what NPCs are checking
- **Clean Architecture**: Respects data ownership

## Dependencies

- Bidirectional MCP implementation
- Thread-safe state access in Godot
- Resource registration system

## Priority Rationale

**Obviousness**: Moderate - Clear need for active perception, implementation detailed elsewhere.

**Development Velocity**: Mixed
- Short-term: Significant work on bidirectional MCP, threading, state snapshots
- Long-term: Cleaner architecture, easier debugging
- Tech tree: Enables sophisticated perception strategies
- Net: Neutral - architectural improvement more than feature enabler

**Concreteness**: Low - Players don't see the difference between push and pull perception, only potentially better NPC decisions.

Architectural improvement that cleans up data flow.