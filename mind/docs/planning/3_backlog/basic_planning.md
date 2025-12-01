# Basic Daily Planning

## Problem Statement

NPCs currently make moment-to-moment decisions without any broader structure. They can't work toward goals, maintain routines, or exhibit the kind of purposeful behavior that makes characters feel alive. The vision document describes NPCs as "behavioral programmers" - this is the simplest form of that concept.

## Design Goals

- Give NPCs daily structure they can follow
- Allow for deviation and adaptation when needed
- Create observable patterns players can learn
- Foundation for hierarchical planning system

## Technical Approach

### Simple Daily Plans

Start with the most concrete planning level:
- Morning routine (hygiene, breakfast)
- Work periods (crafting, gathering, socializing)
- Evening activities
- Sleep schedule

Plans should be:
- **Flexible**: "Sometime this morning" not "8:47 AM exactly"
- **Interruptible**: High-priority needs or events can override
- **Adaptive**: Failed actions trigger replanning
- **Observable**: Players can see NPCs following routines

### Plan Generation

1. **Initial Generation**: Based on personality, role, and needs
2. **Execution**: Follow plan while monitoring for deviations
3. **Local Repair**: Small adjustments when things go wrong
4. **Replanning**: Generate new plan when repair isn't enough

### Plan Representation

Keep it simple initially:
- Ordered list of activities
- Each with rough time window
- Priority/importance level
- Conditions for skipping

This is much simpler than the full compositional language described in the vision, but provides a foundation to build on.

## Benefits

- **Immediate Behavior Structure**: NPCs have purposeful daily patterns
- **Player Predictability**: Can learn NPC schedules
- **Foundation for Hierarchy**: Daily plans slot into weekly/monthly/yearly
- **Emergent Stories**: Plans + execution variance = interesting events

## Dependencies

- Working memory to track current plan
- Basic need system to influence plan generation
- Action execution system

## Priority Rationale

**Obviousness**: Moderate - Clear that NPCs need structure, but many ways to implement it.

**Development Velocity**: Positive long-term despite short-term cost
- Short-term: New systems for plan generation, execution, monitoring, and repair
- Tech tree: Foundation for hierarchical planning, behavioral chunking, routines
- Complexity: Adds significant state management and edge cases
- Net: Positive due to enabling future planning features

**Concreteness**: Very high - Players immediately see NPCs with purposeful daily routines instead of random wandering. One of the most visible improvements to NPC behavior.

This transforms NPCs from reactive to proactive agents.