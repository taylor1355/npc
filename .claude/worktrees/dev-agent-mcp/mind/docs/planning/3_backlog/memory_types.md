# Memory Type Distinctions

## Problem Statement

Currently all memories are treated the same - specific events and general knowledge are stored and retrieved identically. Cognitive science distinguishes between different types of memory (episodic, semantic, procedural, etc.) that serve different functions. The memory system should eventually reflect these distinctions.

## Status: Depends on Basic Memory Implementation

The specific requirements and design for this feature will be shaped by:
- How the basic memory system is implemented
- What retrieval patterns emerge as useful
- Performance characteristics discovered in practice
- How memories are actually used by the decision system

## General Concepts

Psychology distinguishes several memory types:
- **Episodic**: Specific events and experiences
- **Semantic**: General knowledge and facts
- **Procedural**: How to do things (though this might be behavioral chunks)
- Others depending on the model

How these map to the NPC system will depend on implementation experience.

## Dependencies

- Basic memory system must be implemented first
- Memory reflection/consolidation patterns
- Understanding of how NPCs actually use memories

## Priority Rationale

**Obviousness**: Low - Clear need for distinction, unclear how to implement.

**Development Velocity**: Unknown
- Depends entirely on basic memory architecture
- Could simplify or complicate retrieval
- May emerge naturally from usage patterns

**Concreteness**: Low - Internal architecture change, not directly visible.

This is a refinement to consider once basic memory is working and we understand its usage patterns.