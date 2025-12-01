# Potential MCP/Cognitive Architecture Features

This document collects potential features and improvements for the NPC cognitive architecture and MCP server implementation. These items were discovered by mining the simulation repository's planning documents and will be refined into individual backlog items.

## Core MCP Architecture

### 1. MCP Sampling Implementation
**Description**: Use MCP Sampling to let clients control LLM calls, enabling "bring your own compute"
**Motivation**: Fundamentally changes token economy - players can bring their own API keys or buy tokens directly
**Implementation Notes**:
- MCP Sampling allows clients to control LLM calls without servers needing API keys
- Enables sustainable business model for distributed compute
- Discovered through MCP protocol analysis

### 2. Hierarchical Planning Architecture
**Description**: Reduce token usage by 90% through plans that are generated rarely but executed frequently
**Motivation**: Current implementation uses ~2,300 tokens per NPC decision (5 separate LLM calls)
**Implementation Notes**:
- NPCs as behavioral programmers writing/revising plans at multiple timescales
- Plans executed many times before regeneration
- Token-efficient architecture for scaling to 100+ NPCs

### 3. Behavioral Composition
**Description**: Discovery and reuse of behavioral chunks from repeated patterns
**Motivation**: NPCs can learn and reuse common action sequences
**Implementation Notes**:
- Identify repeated patterns in NPC behavior
- Create reusable behavioral primitives
- Reduce decision complexity through composition

### 4. Resource-Based Architecture
**Description**: Use MCP Resources and Prompts for cleaner separation of concerns
**Motivation**: Resources provide read-only access to agent state, Prompts enable reusable decision logic
**Implementation Notes**:
- Leverage MCP Resources for state exposure and debugging
- Use MCP Prompts for structured, reusable decision templates
- Cleaner architecture with better separation between state and logic

## Bidirectional MCP

### 5. Python as MCP Client
**Description**: Enable Mind Service to pull simulation state via MCP resources
**Motivation**: Currently all data is pushed from simulation, even if not needed
**Implementation Notes**:
- Mind Service queries simulation resources on demand
- Reduces unnecessary data transfer
- Enables more flexible decision architectures

### 6. Lazy Evaluation
**Description**: Pull data only when needed for decisions instead of receiving everything
**Motivation**: Reduce bandwidth and processing for unused data
**Implementation Notes**:
- Query specific resources as needed
- Different AI strategies can query different resources
- Supports incremental decision refinement

### 7. Caching Strategy
**Description**: Cache simulation resources with TTL to reduce redundant queries
**Motivation**: Avoid repeated queries for slowly-changing data
**Implementation Notes**:
- Implement TTL-based caching for resources
- Invalidation strategies for dynamic data
- Balance freshness vs query overhead

### 8. Mental State Resources
**Description**: Expose agent mental state as queryable MCP resources
**Motivation**: Enable debugging and introspection of agent cognition
**Implementation Notes**:
- Resources for working memory, long-term memory, current plans
- Useful for debugging and monitoring
- Enables external tools to inspect mental state

## Memory & Conversation

### 9. Conversation Memory Patterns
**Description**: Handle the sliding window (10 msg history, send last 5)
**Motivation**: Current system assumes NPCs maintain their own memory of conversations
**Implementation Notes**:
- Implement conversation context management
- Handle memory degradation over time
- Maintain coherence with limited context

### 10. Per-Agent Mind Persistence
**Description**: Save/restore agent state across mind switches
**Motivation**: Preserve NPC personality and memories when switching minds
**Implementation Notes**:
- Serialize agent state (memories, traits, plans)
- Restore state when switching between mind implementations
- Enable A/B testing without losing agent identity

### 11. Mind Component Data Format
**Description**: Standardize how agent data is preserved
**Motivation**: Ensure consistency across different mind implementations
**Implementation Notes**:
- Define standard format for agent state
- Version compatibility for format evolution
- Support for partial state updates

## Extended Intelligence

### 12. Graduated Intelligence
**Description**: Support minds for items/buildings/GameMaster, not just NPCs
**Motivation**: Any entity can have decision-making capabilities
**Implementation Notes**:
- Items with minds (enchanted objects, AI systems)
- Buildings with minds (smart homes, defensive structures)
- GameMaster with sophisticated world manipulation logic

### 13. Multi-Agent Coordination
**Description**: Theory of mind, reputation tracking, awareness of other agents
**Motivation**: Enable complex social behaviors and emergent group dynamics
**Implementation Notes**:
- Track what agents know about other agents
- Reputation/relationship systems
- Collaborative planning capabilities
- Social graph management

### 14. Need-Influenced Decision Making
**Description**: Incorporate need levels into action selection
**Motivation**: NPCs should prioritize actions based on their current needs
**Implementation Notes**:
- Needs as soft constraints on behavior
- Urgency weighting for critical needs
- Balance between need satisfaction and other goals

## Observation Optimization

### 15. Observation Format Optimization
**Description**: Streamline what gets sent to reduce tokens
**Motivation**: Current observations may include redundant or unnecessary data
**Implementation Notes**:
- Analyze current observation format for redundancy
- Compress observation representation
- Send only deltas when appropriate
- Structure observations for better LLM comprehension

## Cognitive Processing

### 16. Hierarchical Thinking System
**Description**: Implement System 1 (fast) and System 2 (slow) thinking with different compute/context levels
**Motivation**: Reduce latency and token usage by escalating only when needed
**Implementation Notes**:
- Subconscious level uses existing context with weaker LLMs
- Conscious level can retrieve memories and edit working memory
- Based on Kahneman's dual-process theory
- Lower levels produce fewer output tokens

### 17. Mind Wandering
**Description**: Occasionally sample random memories weighted by salience when needs are satisfied
**Motivation**: Creates more realistic, unpredictable NPC behavior
**Implementation Notes**:
- More likely when NPC needs are lower
- Sample from memory based on salience weights
- Can lead to spontaneous actions or conversations

### 18. Multimodal Sensory Processing
**Description**: Use vision-language models to process visual input and answer questions about scenes
**Motivation**: Enable NPCs to understand and react to visual environment
**Implementation Notes**:
- Vision-language model as QA service
- Part of RAG system for scene understanding
- Store significant sensory inputs as memories with multimodal embeddings

## Memory Architecture

### 19. Hybrid Memory System
**Description**: Combine vector embeddings with knowledge graph for semantic search and graph traversal
**Motivation**: More sophisticated memory retrieval than pure vector search
**Implementation Notes**:
- Initial semantic search over embeddings
- Lightweight LLM traverses knowledge graph from retrieved nodes
- Create/prune edges between memory nodes dynamically
- Consider libraries like MemOS or Neo4j

### 20. Memory Stream with Scoring
**Description**: Score memories based on recency, importance, and relevance for retrieval
**Motivation**: Cognitively-inspired memory system proven in Generative Agents paper
**Implementation Notes**:
- Recency: exponential decay based on last accessed time
- Importance: 1-10 score assigned by LLM on encoding
- Relevance: semantic similarity to query
- All subscores normalized to [0,1] range

### 21. Reflection Trees
**Description**: Represent agent reflections as knowledge graph for sophisticated reasoning
**Motivation**: Enable higher-order thinking and self-awareness
**Implementation Notes**:
- Build reflection hierarchy as graph structure
- Use graph techniques for manipulation and access
- Critical for making memory stream work effectively

### 22. Structured Memory Components
**Description**: Provide structured memory (calendars, scratchpads, world state) alongside unstructured
**Motivation**: Agents need both free-form and structured information
**Implementation Notes**:
- Calendar for temporal planning
- Working memory scratchpad for current context
- Track last known state of relevant world objects
- Free modification of structured components

### 23. SQL-like Query System
**Description**: Allow agents to formulate SQL queries for memory retrieval
**Motivation**: More precise memory queries than pure semantic search
**Implementation Notes**:
- Agents generate SQL or SQL-like queries
- Dynamic tag system for narrowing searches
- Agents can add/remove tags up to limit

### 24. Tag Daemon
**Description**: Background process that continuously tags memories based on importance
**Motivation**: Improve memory searchability over time
**Implementation Notes**:
- Priority queue based on importance * (1 - tags_checked/total_tags)
- Check high-importance memories first
- Continuously improve memory organization

## Infrastructure & Compute

### 25. Local/Cloud Compute Fusion
**Description**: Bundle local and cloud compute as unified LLM API endpoints
**Motivation**: Enable sustainable business model with user choice
**Implementation Notes**:
- Local vLLM servers for GPU/CPU utilization
- Thread-safe priority queue for requests
- Higher priority for latency-sensitive tasks
- Prioritize local compute to save costs
- Token allowance system with subscriptions

### 26. MCP Server Orchestration
**Description**: Single MCP server managing multiple entities efficiently
**Motivation**: Reduce overhead of multiple server instances
**Implementation Notes**:
- One server handles multiple NPCs/entities
- Efficient resource sharing
- Simplified deployment for users

### 27. Society of Mind Architecture
**Description**: Multiple parallel cognitive subsystems working together
**Motivation**: Modular, scalable cognitive architecture inspired by Minsky
**Implementation Notes**:
- Main decision-making loop
- Memory maintenance process
- Planning subsystem
- Each subsystem optimized independently
- Hierarchically gated compute across subsystems

### 28. DSPy Integration
**Description**: Use DSPy framework for optimizing cognitive architecture
**Motivation**: Systematic optimization of prompts and cognitive processes
**Implementation Notes**:
- Explore DSPy for prompt optimization
- Potential for automated architecture improvement
- Stanford NLP framework for LLM programs

## Social & Learning

### 29. Relationship-Based Memory Boost
**Description**: Weight memory importance by relationships with involved NPCs
**Motivation**: Social connections should influence what NPCs remember
**Implementation Notes**:
- Boost/deboost memories based on relationship strength
- Memories involving friends more likely to be recalled
- Dynamic adjustment as relationships change

### 30. Semantic Opinion Memory
**Description**: Store opinions that influence related concepts through semantic similarity
**Motivation**: Realistic opinion formation and bias
**Implementation Notes**:
- Opinions stored as semantic memories
- Dense vector retrieval benefits imprecision
- Opinion on one thing influences similar things
- Creates realistic biases and preferences


## Notes

These features range from quick wins (terminology cleanup) to major architectural changes (hierarchical planning). Priority should be based on:

1. **Token cost reduction** - Features that reduce API costs
2. **Architecture alignment** - Features that unify the codebase
3. **Business enablement** - Features that enable the distributed compute model
4. **Developer experience** - Features that make the system easier to work with

Next steps:
- Refine descriptions based on current implementation status
- Estimate effort for each feature
- Identify dependencies between features
- Create individual backlog items for high-priority features