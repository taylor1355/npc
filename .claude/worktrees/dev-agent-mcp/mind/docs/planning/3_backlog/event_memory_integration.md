# Event Memory Integration

## Problem Statement

Events are currently accumulated in a short-term buffer (60 minute retention, max 15 events) and formatted into prompts for immediate decision-making. However, significant events should be consolidated into long-term memory so NPCs can recall important occurrences from the past beyond the retention window.

## Status: Blocked on Event System Stabilization

This feature depends on:
- Event system working reliably in production
- Understanding which events are significant enough to remember long-term
- Observing patterns in how events influence NPC behavior
- Ensuring event buffer retention policy is working as expected

## Current Implementation

**Event Buffer** (implemented):
- Events accumulate in Mind.event_buffer with retention policy
- Retention: Events newer than 60 game minutes OR up to 15 most recent events
- Events formatted separately from observations in cognitive node prompts
- Events are distinct from observations (temporal occurrences vs state snapshots)

**Memory Formation** (existing):
- CognitiveUpdateNode creates NewMemory objects for significant experiences
- Memories stored in daily buffer, consolidated to long-term during sleep
- Memory importance scoring (1-10) guides what gets stored

## Proposed Integration

**Event-to-Memory Consolidation**:
- During memory consolidation, evaluate recent events for long-term significance
- Events that meet significance criteria converted to Memory objects
- Consider event type, entities involved, and recency in importance scoring

**Significance Criteria** (to be refined):
- Interaction rejections (current bug fix focus) - helps NPCs learn from failed attempts
- Goal completions or failures
- Emotional events (high emotion_intensity)
- Novel encounters or situations
- Events involving important relationships

**Memory Queries**:
- MemoryQueryNode could generate queries based on event types
- Example: "previous interactions with [entity]" when processing rejection event
- Helps NPCs recall relevant context when similar events occur

## Dependencies

- Event system must be stable and validated in production
- Understanding of event frequency and types in typical gameplay
- Observation of which events actually influence NPC decision-making
- Memory consolidation node must be working reliably

## Priority Rationale

**Obviousness**: Medium - Clear value for behavioral consistency, but event system needs validation first.

**Development Velocity**: Medium
- Builds on existing memory consolidation infrastructure
- Requires careful tuning of significance criteria
- May need iteration based on gameplay testing

**Concreteness**: Low - Internal cognitive architecture, behavioral changes emerge indirectly.

This is a refinement to consider once the event system is stable and we observe which events meaningfully impact NPC behavior.
