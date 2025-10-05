# Working Memory Management

## Problem Statement

NPCs need to maintain context across decision cycles. Currently, each decision starts fresh, losing important context about ongoing activities, recent events, and current goals. This creates disjointed behavior where NPCs "forget" what they were just doing.

## Design Goals

- Maintain continuity across decision cycles
- Manage limited context window efficiently
- Preserve important information while pruning irrelevant details
- Support both automatic and explicit memory updates

## Technical Approach

### Working Memory as Context Bridge

Working memory serves as the bridge between observations and decisions:
- Contains current goals and active plans
- Tracks recent events and observations
- Maintains conversation context
- Records current emotional/need state

### Update Mechanisms

1. **Automatic Updates**:
   - Append new observations
   - Decay old information
   - Summarize when approaching context limits

2. **Explicit Updates**:
   - Plan changes
   - Goal completion
   - Important event flags

3. **Memory Integration**:
   - Pull relevant long-term memories based on current context
   - Update working memory with retrieved information
   - Mark which memories have been recently accessed

### Context Window Management

As context grows:
1. Identify least relevant information
2. Summarize or remove old details
3. Preserve critical information (goals, plans, recent events)
4. Use importance scoring to guide pruning

## Benefits

- **Behavioral Continuity**: NPCs remember what they're doing
- **Contextual Decisions**: Actions make sense given recent history
- **Conversation Coherence**: Can maintain multi-turn dialogues
- **Observable Progress**: Players can see NPCs working toward goals

## Dependencies

- Basic memory system for retrieval
- Observation formatting from simulation
- Clear notion of "importance" for pruning

## Priority Rationale

**Obviousness**: High - Clearly needed for continuity between decisions. Without it, NPCs "forget" what they were just doing.

**Development Velocity**: Mixed
- Short-term: Adds state management complexity and context window management
- Long-term: Enables multi-step behaviors, conversations, goal pursuit
- Net: Slightly positive - foundational for coherent behavior

**Concreteness**: High - Players immediately notice when NPCs remember context vs acting randomly. Visible in conversations and multi-step tasks.

Essential for any behavior more complex than single actions.