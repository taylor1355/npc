# Hierarchical Planning System

## Problem Statement

Daily plans alone create repetitive behavior. NPCs need longer-term goals and themes that guide their daily activities while maintaining coherence across different timescales. The vision document's core insight is NPCs as "behavioral programmers" writing plans at multiple temporal scales.

## Design Goals

- Create coherent behavior across days, weeks, months, and years
- Balance structure with flexibility
- Enable long-term character development
- Reduce token usage through plan reuse

## Technical Approach

### Temporal Hierarchy

Four levels of planning, each serving different purposes:

1. **Yearly Themes** (2-3 abstract goals):
   - "Become more social"
   - "Master woodworking"
   - "Explore the world"
   - Compressed into global context

2. **Monthly Projects** (concrete objectives):
   - "Build friendship with the blacksmith"
   - "Complete masterwork chair"
   - "Map the northern forest"
   - Visible as current focus

3. **Weekly Routines** (behavioral templates):
   - "Monday market days"
   - "Workshop mornings"
   - "Social evenings"
   - Default patterns to follow

4. **Daily Plans** (specific sequences):
   - Detailed action schedules
   - Reactive adjustments
   - Currently executing

### Plan Relationships

- Higher levels constrain lower levels
- Daily plans implement weekly routines
- Weekly routines work toward monthly projects
- Monthly projects align with yearly themes

### Token Efficiency

Key insight: Generate plans rarely, execute frequently
- Yearly themes: Generated once per year (or on major life events)
- Monthly projects: Revised monthly or when completed
- Weekly routines: Adjusted as needed
- Daily plans: Generated each morning or when disrupted

This could reduce token usage by 90% compared to making every decision from scratch.

### Plan Revision

Plans aren't rigid:
- Track execution success
- Detect when plans aren't working
- Revise at appropriate level
- Learn from failures

## Benefits

- **Behavioral Coherence**: Actions connect to long-term goals
- **Character Development**: NPCs grow and change over time
- **Token Efficiency**: Reuse plans instead of regenerating
- **Emergent Narratives**: Long-term goals create story arcs

## Dependencies

- Basic daily planning (build up from there)
- Working memory to track plan hierarchy
- Plan execution monitoring

## Priority Rationale

**Obviousness**: Low - Many design choices and complex interactions between levels.

**Development Velocity**: Mixed with strong long-term benefits
- Short-term: Significant complexity managing multiple planning levels and their interactions
- Tech tree: Enables yearly character arcs, monthly projects, emergent stories
- System stretch: Could easily become overengineered with too many levels
- Complexity: Edge cases around plan conflicts, revision triggers, coherence maintenance
- Net: Positive long-term but requires careful implementation to avoid complexity explosion

**Concreteness**: High once implemented - Players see NPCs with long-term goals, character development, and coherent life stories instead of daily repetition.

This is the vision document's core innovation but requires substantial groundwork.