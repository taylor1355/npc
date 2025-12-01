# Knowledge Graph Memory Structure

## Problem Statement

Memories are currently independent items retrieved by similarity. Real knowledge is interconnected - people, places, events, and concepts form a web of relationships. A graph structure could better represent these connections and enable more sophisticated retrieval.

## Design Goals

- Represent relationships between memories
- Enable graph-based retrieval patterns
- Support reasoning about connections
- Maintain compatibility with vector similarity

## Technical Approach

### Established Hybrid Approach

There's significant literature on combining vector embeddings with knowledge graphs:
- Nodes: Memories with embeddings
- Edges: Typed relationships between memories
- Retrieval: Both similarity search and graph traversal
- Scoring: Combined metrics from both approaches

This is well-researched territory with proven benefits.

### Relationship Types

Could include:
- Temporal (before/after, during)
- Causal (caused by, leads to)
- Social (involves person X, witnessed by Y)
- Spatial (near, at location)
- Conceptual (related to, type of)

### Retrieval Patterns

- Start with vector similarity for initial candidates
- Traverse graph to find related context
- Use relationship types to guide traversal
- Combine scores from similarity and connectivity

## Benefits

- **Richer Retrieval**: Find causally and temporally related memories
- **Better Context**: Pull connected memories together naturally
- **Reasoning**: Follow chains of relationships
- **Memory Coherence**: Related memories stay connected

## Dependencies

- Basic memory system with embeddings
- Reflection system to identify relationships
- Graph database or equivalent structure

## Priority Rationale

**Obviousness**: Low - While literature exists, specific implementation has many choices.

**Development Velocity**: Negative
- Short-term: Significant complexity over pure vector store
- Long-term: Better memory system but doesn't unlock new features
- Complexity: Graph maintenance, traversal, scoring combinations
- Performance: Graph operations at scale
- Net: Negative - adds complexity despite established patterns

**Concreteness**: Moderate - Players notice more coherent memory retrieval, NPCs remembering related events together, better contextual understanding.

Established technology that significantly improves memory quality but adds system complexity.