# Generative Agents Memory System

**Status:** Backlog - Research needed
**Priority:** Medium
**Related Paper:** [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442) (Park et al., 2023)

## Problem Statement

Current memory consolidation is a simple placeholder that adds all daily memories to long-term storage without any filtering, reflection, or abstraction. The Generative Agents paper describes a sophisticated memory system that could significantly improve NPC believability and coherence.

## Research Goals

Study the Generative Agents paper and identify applicable techniques for our architecture:

### 1. Memory Stream Management
- How do they handle the memory stream structure?
- What's their approach to time-based organization?
- How do they balance retrieval vs. storage?

### 2. Reflection Mechanism
- When and how do they trigger reflections?
- How are reflections generated from memories?
- What role do reflections play in behavior?

### 3. Importance Scoring
- How do they calculate memory importance?
- Do they use LLM-based or heuristic scoring?
- How does importance affect retrieval?

### 4. Memory Retrieval
- What's their retrieval algorithm (recency + relevance + importance)?
- How do they balance the three factors?
- Do they use any filtering or ranking mechanisms?

### 5. Consolidation Process
- When does consolidation happen (sleep/downtime)?
- How do they merge/filter similar memories?
- Do they implement forgetting curves?

## Implementation Considerations

### What We Have
- ✅ Daily memory buffer (`state.daily_memories`)
- ✅ Importance scoring during memory formation (via cognitive_update)
- ✅ Placeholder consolidation node (`MemoryConsolidationNode`)
- ✅ Combined scoring (relevance + importance + recency)

### What We Need
- Reflection generation mechanism
- Sophisticated consolidation algorithm
- Memory abstraction/generalization
- Forgetting curve implementation
- Better retrieval balancing

## Success Criteria

- [ ] Comprehensive understanding of their memory architecture
- [ ] Clear mapping of their techniques to our system
- [ ] Prototype implementation of key features (reflections, consolidation)
- [ ] Measurable improvement in NPC believability
- [ ] Performance remains acceptable (token usage, latency)

## Dependencies

- Current memory system (VectorDBMemory, daily buffer)
- Cognitive update node generating new memories
- LLM access for reflection generation

## Priority Rationale

**Medium priority** - The current placeholder system is functional but not sophisticated. This research would unlock:
- More believable long-term character development
- Better memory management (less noise in long-term storage)
- Richer NPC reasoning through reflections

However, it's not blocking immediate progress. We can improve other systems first (structured cognitive state, planning) then return to enhance memory consolidation.

## Next Steps

1. Read the Generative Agents paper in detail
2. Document key techniques applicable to our architecture
3. Design enhancements to `MemoryConsolidationNode`
4. Implement reflection generation
5. Test with realistic scenarios
