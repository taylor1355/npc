# Compositional Plan Language

## Problem Statement

Plans are currently free-form text that LLMs must parse and interpret. A compositional language would provide structure while maintaining expressivity, improve plan interpretability, and enable speculative features like plan sharing and analysis. This is part of the "behavioral programmer" vision.

## Design Goals

- Structure plans without limiting expressivity
- Individual elements can still be free-form text
- Enable plan inspection and debugging
- Support future features like sharing and pattern analysis

## Technical Approach

### Hybrid Structure

Compositional wrapper around free text intentions:
```
morning_routine = sequence(
  "check the weather",
  move_to("bathroom"),
  "get ready for the day",
  if_available("breakfast",
    "eat breakfast"
  ),
  "head to workshop"
)
```

The language provides structure (sequence, conditionals) while preserving free-form expressivity within actions. These are intentions, not scripts - the actual behavior emerges from context.

### Plan Deviation

Plans are aspirational guides. NPCs naturally deviate when:
- Something more important comes up (emergent from needs/context)
- Opportunities arise (friend walks by, interesting event)
- Obstacles appear (path blocked, item unavailable)
- Mood or needs change

This is NOT programmed randomness - it emerges from the NPC evaluating their current context against their plan.

### Language Elements

Structural primitives:
- Sequences: ordering of intentions
- Conditionals: anticipated branches
- Loops: repeated activities
- Parallel: concurrent goals

Action primitives remain free-form text describing intentions, not specific behaviors.

### Execution Engine

- Parse structural elements
- Pass intentions to LLM for interpretation in context
- Track execution position with minimal tokens
- Allow natural deviation based on circumstances

## Benefits

- **UI Integration**: Can display plan structure to players
- **Token Efficiency**: Track plan position without re-parsing
- **Debugging**: Clear intention flow and execution state
- **Interpretability**: Programmatic access to plan structure
- **Speculative Features**: Enables plan sharing, pattern analysis, chunking

## Dependencies

- Basic planning system
- Action execution framework
- Parser for structural elements

## Priority Rationale

**Obviousness**: Low - Many design choices, benefits becoming clearer through discussion.

**Development Velocity**: Slightly negative
- Short-term: Parser, validator, execution engine add complexity
- Benefits: Easier debugging, plan inspection, state tracking
- Long-term: Programmatic access simplifies many plan-related features
- Net: Slightly negative - complexity outweighs benefits initially

**Concreteness**: Low to moderate - Players can see NPCs' plan structure in UI, understand their intentions better. Not directly visible in behavior but improves transparency.

Infrastructure investment with clearer benefits than initially apparent.