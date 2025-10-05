# NPC Cognitive Architecture Design

*Note: This is a vision and exploration document, not a rigid specification. It presents ideas, patterns, and possibilities to explore during development. Specific implementations should emerge from experimentation and practical needs.*

## Executive Summary

This document presents a comprehensive cognitive architecture for NPCs that creates believable agents with rich inner lives. The architecture treats NPCs as **behavioral programmers** - entities that write, execute, and revise behavioral programs for themselves, with natural gaps between intention and execution that create human-like variance.

The system is built on three foundational principles:
1. **Hierarchical Planning**: Multi-timescale behavioral organization from yearly themes to moment-by-moment actions
2. **Society of Mind**: Parallel cognitive subsystems that collaborate and compete for control
3. **Distributed Cognition**: Flexible compute allocation across local and cloud resources via MCP

## Architecture Overview

### Cognitive Stack

The architecture could implement three layers of cognitive processing, each with different latency and capability tradeoffs:

```
┌─────────────────────────────────────────┐
│     METACOGNITIVE (Reflective)         │ ← Slow, strategic
│   • Reflection & learning               │   Larger models
│   • Plan revision                       │   Rare activation
│   • Strategy formation                  │
├─────────────────────────────────────────┤
│       CONSCIOUS (Deliberate)           │ ← Moderate speed
│   • Active memory retrieval            │   Standard models
│   • Plan generation                    │   On-demand
│   • Complex reasoning                  │
├─────────────────────────────────────────┤
│      SUBCONSCIOUS (Reactive)          │ ← Fast, contextual
│   • Contextual understanding           │   Cheap LLMs (GPT-5 nano,
│   • Common situations                  │   Gemini Flash, Haiku)
│   • Eventually: cached patterns        │   High frequency
└─────────────────────────────────────────┘
```

Note that even the "subconscious" layer will initially use LLMs for contextual understanding - just cheaper, faster models. Heuristics and cached responses can be added later as an optimization once patterns become clear.

### Society of Mind Architecture

Multiple specialized subsystems operate in parallel, each optimized for its domain:

- **Memory Subsystem**: Maintains working and long-term memory, handles retrieval and consolidation
- **Planning Subsystem**: Generates and monitors hierarchical behavioral plans
- **Social Subsystem**: Tracks relationships, manages theory of mind, coordinates with other agents
- **Emotion Subsystem**: Modulates decisions based on affective state
- **Attention Subsystem**: Allocates cognitive resources based on salience and relevance

### Integration via MCP

The cognitive architecture connects to the Godot simulation through bidirectional Model Context Protocol:

```
Godot Simulation ←→ Cognitive Architecture
    (MCP Server)        (MCP Server)
    Physical State      Mental State
    Perceptions        Plans & Goals
    Available Actions  Memories
```

## Core Subsystems

### Memory Architecture

The memory system combines multiple approaches for flexible and efficient retrieval:

#### Hybrid Storage Backend

**Vector Embeddings + Knowledge Graph**
- Semantic similarity search finds relevant memories quickly
- Graph traversal explores relationships between memories
- Dynamic edge creation/pruning based on co-activation
- Libraries to evaluate: Custom implementation → MemOS → Neo4j

#### Memory Stream with Cognitive Scoring

Based on cognitive science research, memories are scored on three dimensions:

- **Recency**: Exponential decay from last access time
- **Importance**: LLM-assigned significance score (e.g., 1-10 scale)
- **Relevance**: Cosine similarity to current query

The retrieval score combines these factors: `score = α·recency + β·importance + γ·relevance`

The specific weights (α, β, γ) will be tuned through experimentation.

#### Reflection Trees

Higher-order memories that synthesize and abstract from primary experiences:
- Hierarchical structure enables reasoning about patterns
- Self-reflection creates metacognitive awareness
- Critical for long-term coherence and character development

#### Structured Memory Components

Alongside unstructured episodic memories:
- **Calendar**: Temporal planning and appointment tracking
- **Scratchpad**: Working memory for current context
- **World State Cache**: Last known state of important objects
- **Social Graph**: Relationship network and reputation

#### Continuous Organization

**Tag Daemon** background process:
- Prioritizes untagged high-importance memories
- Continuously improves searchability
- Runs during idle cognitive cycles

#### Query Interfaces

Multiple query modalities:
- Natural language semantic search
- SQL-like structured queries
- Graph traversal patterns
- Temporal range queries

### Planning & Behavior System

NPCs generate and execute hierarchical plans that provide behavioral coherence while allowing flexibility:

#### Temporal Hierarchy

Plans exist at multiple timescales, each serving different cognitive functions:

**Yearly Themes** (2-3 abstract goals)
```
"Become more social"
"Master woodworking craft"
"Explore the northern territories"
```

**Monthly Projects** (concrete objectives)
```
"Build friendship with the blacksmith"
"Complete masterwork chair"
"Map the mountain passes"
```

**Weekly Routines** (behavioral templates)
```
Monday: Market day, restock supplies
Tuesday-Thursday: Workshop focus
Friday: Social evening at tavern
Weekend: Flexible exploration
```

**Daily Plans** (specific action sequences)
```
Morning: Hygiene → Breakfast → Check mail
Workday: Craft furniture with breaks
Evening: Dinner → Socialize OR Read
```

#### Behavioral Chunking

NPCs discover and reuse behavioral patterns:

1. **Pattern Detection**: Identify repeated action sequences
2. **Chunk Creation**: Abstract into named behavioral unit
3. **Effect Learning**: Track which needs/goals the chunk satisfies
4. **Social Sharing**: Chunks can spread through observation

Example chunks:
- `morning_routine`: Standard wake-up sequence
- `craft_chair`: Complete furniture creation process
- `tavern_social`: Evening socializing pattern

#### Plan-Reality Feedback Loop

Plans adapt based on execution through a continuous monitoring process:

1. **Deviation Detection**: Measure how far execution has drifted from the plan
2. **Local Repair**: Small adjustments to get back on track
3. **Global Replanning**: Major revisions when local fixes aren't enough
4. **Pattern Recognition**: Repeated sequences become reusable chunks
5. **Learning**: Update expectations based on outcomes

#### Plan Representation

Plans might use a compositional language that could include:

- **Sequential behaviors**: "Do A, then B, then C"
- **Alternative behaviors**: "Do A or B depending on context"
- **Conditional behaviors**: "If condition X, do Y"
- **Interruptible behaviors**: "Can be paused for high-priority events"
- **Temporal anchors**: "At 8am" or "After sunset"
- **Flexible timing**: "Sometime this morning"

For example, a daily plan might specify morning routines, work periods with alternatives (craft OR gather materials), and conditional social interactions based on current needs. The exact representation will emerge from what proves most useful in practice.

### Decision Engine

The decision system balances multiple influences to select contextually appropriate actions:

#### Hierarchical Decision Making

**System 1 (Fast/Subconscious)**
- Pattern matching against cached decisions
- Simple heuristics and habits
- Immediate responses to routine situations
- Low latency, minimal tokens

**System 2 (Slow/Conscious)**
- Deliberate reasoning about options
- Memory retrieval and planning
- Handles novel or complex situations
- Higher latency, more tokens

**Escalation Triggers**
- Novelty detection
- High-stakes decisions
- Plan deviation beyond threshold
- Emotional intensity

#### Need-Influenced Decisions

Needs influence behavior through multiple interconnected pathways:

- **LLM Context**: Need levels included in prompts naturally influence decisions through the model's understanding
- **Action Biasing**: Some rule-based weighting can nudge toward need-satisfying actions without dominating choice
- **Memory Retrieval**: Hungry NPCs might recall food-related memories more readily
- **Emotional State**: Low needs affect mood, which colors all decision-making
- **Attention Focus**: Urgent needs draw cognitive resources and influence what gets noticed
- **Simulation Outcomes**: Need states exposed to Godot affect success rates, action costs, and interaction availability

This multi-pathway influence creates nuanced behavior where needs matter without reducing NPCs to simple need-satisfaction machines.

#### Mind Wandering

When needs are satisfied, spontaneous thoughts emerge:
- Random memory sampling weighted by salience
- Can trigger unexpected actions or conversations
- Creates behavioral variety and surprises
- More likely during low-activity periods

### Social Intelligence

NPCs maintain sophisticated models of their social world:

#### Theory of Mind

NPCs maintain models of what other agents know, believe, and feel. This might include:

- **Knowledge Tracking**: What facts has this agent observed or been told?
- **Belief Modeling**: What might they believe that differs from reality?
- **Emotional Estimation**: What is their current emotional state?
- **Relationship Status**: How do we relate to each other?
- **Interaction History**: When and how have we interacted?

The representation could range from simple tags to complex probabilistic models, depending on what proves useful. The key is that NPCs can reason about others' mental states, enabling deception, empathy, and coordination.

#### Relationship-Based Memory

Social connections influence memory:
- Memories involving friends are boosted
- Negative relationships can suppress memories
- Shared experiences strengthen bonds
- Social events get higher importance scores

#### Semantic Opinion Formation

Opinions naturally integrate with the knowledge graph memory structure, following cognitive science principles:

- **Emotional Intensity Drives Retrieval**: Strong emotions (positive OR negative) make memories more accessible - the absolute value matters for retrieval probability
- **Valence Affects Behavior**: Once retrieved, positive/negative valence influences approach/avoid decisions
- **Graph-Based Spreading**: Emotional weights propagate through conceptual connections in the knowledge graph
- **Clustering Effect**: Strongly emotional experiences create highly interconnected memory clusters that are easily activated

For example, a terrifying wolf encounter creates a high-intensity negative node that:
1. Is *more* likely to be retrieved (high absolute emotional value)
2. Connects strongly to related concepts (forest, night, howling)
3. Triggers avoidance behaviors when retrieved
4. Can create rumination patterns (can't stop thinking about feared things)

This models realistic psychological phenomena where we often can't forget things we'd prefer to forget.

#### Multi-Agent Coordination

Collaborative behavior emerges through:
- Shared plan negotiation
- Reputation tracking
- Social norm adherence
- Group decision protocols

## Communication Layer

### Bidirectional MCP Architecture

Both simulation and cognition expose queryable resources:

#### Simulation Resources (Godot as MCP Server)

```
simulation://agent/{id}/physical_state
    position, movement_state, controller_state

simulation://agent/{id}/perception
    visible_entities, available_interactions

simulation://agent/{id}/needs
    current_values, decay_rates, thresholds

simulation://world/entities
    global_registry, positions, states

simulation://world/time
    elapsed_minutes, time_scale, day_cycle
```

#### Cognitive Resources (Mind as MCP Server)

```
cognition://agent/{id}/mental_state
    working_memory, recent_memories, personality

cognition://agent/{id}/plans
    current_plans, execution_state, deviations

cognition://agent/{id}/social
    relationships, beliefs_about_others
```

### MCP Sampling

Enable sustainable token economics:
- Clients control LLM calls directly
- Players can bring their own API keys
- No server-side API key management
- Pay-as-you-go or subscription models

### Lazy Evaluation & Caching

Resource queries can be optimized through intelligent caching:

- **Variable TTLs**: Rapidly changing data (position) expires quickly while stable data (personality) persists
- **Predictive Prefetching**: Anticipate likely next queries based on patterns
- **Invalidation Strategies**: Event-based (on change) vs time-based (TTL) vs hybrid
- **Hierarchical Caching**: Different cache levels for different access patterns

The specific caching strategy will depend on observed access patterns and performance requirements. The key principle is to avoid redundant queries while maintaining consistency.

## Compute Management

### Infrastructure Layer

#### Local/Cloud Compute Fusion

A unified compute layer could abstract over different processing resources:

- **Local Models**: User's GPUs running vLLM or similar inference servers
- **Cloud APIs**: Commercial endpoints (OpenAI, Anthropic, Google)
- **Hybrid Routing**: Decisions about where to send each request based on:
  - Latency requirements
  - Task complexity
  - Cost considerations
  - Resource availability
  - User preferences

The routing logic might consider factors like preferring local compute for privacy, using cloud for complex reasoning, and caching frequent requests. Implementation details would emerge from actual usage patterns and requirements.

#### Cognitive Load Distribution

Map cognitive functions to compute resources:

| Function | Model | Latency | Frequency |
|----------|-------|---------|-----------|
| Subconscious | Small/Cached | <100ms | Continuous |
| Conscious | Medium | <1s | On-demand |
| Metacognitive | Large | <5s | Rare |
| Reflection | Large | <10s | Daily |

#### MCP Server Orchestration

Single server manages multiple entities:
- Shared resources (behavioral primitives, cultural knowledge)
- Entity-specific state (memories, plans, relationships)
- Efficient batch processing
- Connection pooling

## Implementation Roadmap

Following the prioritization principles of **obviousness**, **development velocity**, and **concreteness**:

### Phase 1: Obvious Foundations (Weeks 1-3)
*Do the obvious things first to constrain less obvious design decisions*

1. **Basic Memory System**
   - Vector store for embeddings (constrains retrieval patterns)
   - Simple working memory string (constrains context management)
   - Clear memory lifecycle (add, retrieve, forget)

2. **Simple Decision Making**
   - Need-based action selection (obvious utility function)
   - Single-context decisions (no planning yet)
   - Direct observation → action mapping

3. **MCP Integration**
   - Basic tool interface (create_agent, decide_action)
   - Simple resource exposure (mental_state)
   - Established communication protocol

### Phase 2: Development Accelerators (Weeks 4-5)
*Build tools and patterns that speed up everything else*

1. **Debugging Infrastructure**
   - MCP resource inspection for all state
   - Decision explanation traces
   - Memory query visualization

2. **Testing Frameworks**
   - Scenario replay system
   - Behavioral assertions
   - Memory operation mocks

3. **Rapid Iteration Tools**
   - Hot-reload for prompts
   - A/B testing harness
   - Performance profiling

### Phase 3: Concrete Behaviors (Weeks 6-8)
*Create tangible features that can be refined through use*

1. **Working Memory Management**
   - Context window optimization
   - Relevance filtering
   - State summarization

2. **Daily Planning**
   - Simple sequential plans
   - Basic need satisfaction
   - Observable behavior patterns

3. **Social Interactions**
   - Conversation memory
   - Simple relationship tracking
   - Opinion formation

### Phase 4: Architectural Elaboration (Weeks 9-12)
*With foundations solid, build sophisticated features*

1. **Hierarchical Planning**
   - Multiple time scales
   - Plan revision based on outcomes
   - Behavioral chunking

2. **System 1/2 Thinking**
   - Escalation triggers
   - Cached responses
   - Deliberation when needed

3. **Hybrid Memory**
   - Graph relationships
   - Reflection generation
   - Semantic search improvements

### Phase 5: Intelligence Amplification (Weeks 13-16)
*Enhance intelligence now that core systems work*

1. **Theory of Mind**
   - Model other agents' knowledge
   - Belief tracking
   - Perspective taking

2. **Emotional System**
   - Affect-driven decisions
   - Mood persistence
   - Emotional contagion

3. **Advanced Coordination**
   - Multi-agent planning
   - Reputation systems
   - Cultural norm emergence

### Phase 6: Scale & Polish (Weeks 17-20)
*Optimize for production use*

1. **Bidirectional MCP**
   - Simulation as MCP server
   - Resource-based architecture
   - Lazy evaluation

2. **MCP Sampling**
   - Client-controlled LLM calls
   - Token economy implementation
   - API key management

3. **Performance Optimization**
   - Aggressive caching
   - Batch processing
   - Token usage reduction

## Design Decisions & Options

### Memory Backend Choice

**Option A: Custom Implementation**
- Pros: Full control, optimized for use case, no dependencies
- Cons: More development time, need to solve solved problems
- When: Choose if unique requirements emerge

**Option B: MemOS Integration**
- Pros: Faster development, proven patterns, active development
- Cons: External dependency, less flexibility, potential overhead
- When: Choose if it closely matches needs

**Option C: Neo4j Graph Database**
- Pros: Mature, powerful graph operations, good tooling
- Cons: Heavy dependency, operational complexity
- When: Choose if graph operations become central

**Recommendation**: Start with Option A for learning, evaluate B/C if custom becomes bottleneck

### LLM Routing Strategy

**Option A: Fixed Model per Task**
```python
TASK_MODELS = {
    "memory_query": "haiku",
    "decision": "sonnet",
    "reflection": "opus"
}
```

**Option B: Dynamic Complexity-Based**
```python
def select_model(request):
    complexity = estimate_complexity(request)
    if complexity < 0.3: return "haiku"
    elif complexity < 0.7: return "sonnet"
    else: return "opus"
```

**Option C: User Preferences**
```python
def select_model(request, user_prefs):
    if user_prefs.force_model:
        return user_prefs.force_model
    else:
        return dynamic_select(request)
```

**Recommendation**: Implement B with C as override for player control

### State Persistence Format

**Option A: JSON**
- Pros: Human readable, easy debugging, standard tooling
- Cons: Verbose, slower parsing, no schema validation
- When: Development phase, debugging priority

**Option B: Protocol Buffers**
- Pros: Efficient, strongly typed, versioning support
- Cons: Build complexity, less readable
- When: Production, performance matters

**Option C: Custom Binary**
- Pros: Maximum efficiency, exact control
- Cons: High development cost, hard to debug
- When: Extreme performance requirements

**Recommendation**: Use A for MVP, plan migration to B for production

### Caching Strategy

**Option A: TTL-Based**
```python
cache.set(key, value, ttl=60)  # Expires in 60 seconds
```

**Option B: Event-Based Invalidation**
```python
on_event("entity_moved"):
    cache.invalidate(f"position:{entity_id}")
```

**Option C: Hybrid with Priorities**
```python
cache.set(key, value,
    ttl=60,
    invalidate_on=["position_change"],
    priority="high"  # Keep in cache under pressure
)
```

**Recommendation**: Implement C for maximum flexibility

## Gap Areas & Future Exploration

These areas require additional research and design:

### Emotion System
*How do emotions emerge, persist, and influence behavior?*

- **Emotion Generation**: Mapping events to affective responses
- **Mood Dynamics**: How emotions decay and combine over time
- **Decision Influence**: Emotional biasing of action selection
- **Emotional Contagion**: Spread of emotions through social networks
- **Memory Formation**: Emotional enhancement of memory encoding

### Attention System
*How do agents allocate limited cognitive resources?*

- **Salience Computation**: What draws attention?
- **Focus Management**: Sustained vs divided attention
- **Interruption Handling**: When to break focus
- **Selective Processing**: Ignoring irrelevant information
- **Attention Restoration**: Recovery from cognitive fatigue

### Consciousness Escalation
*When and how to promote processing to higher levels?*

- **Novelty Detection**: Recognizing situations requiring deliberation
- **Surprise Metrics**: Deviation from predictions
- **Stakes Assessment**: Importance of current decision
- **Resource Availability**: Cognitive budget management
- **Escalation Hysteresis**: Avoiding thrashing between levels

### Time Perception
*How do agents experience and reason about time?*

- **Subjective Duration**: Fast/slow time experience
- **Planning Horizons**: How far ahead to plan
- **Temporal Abstraction**: Learning appropriate time granularities
- **Schedule Flexibility**: Rigid vs fluid time commitments
- **Temporal Reasoning**: Understanding cause-effect delays

### Dream & Consolidation
*What happens during downtime?*

- **Memory Consolidation**: Strengthening important memories
- **Pattern Extraction**: Finding regularities in experience
- **Garbage Collection**: Forgetting irrelevant details
- **Creative Recombination**: Generating novel solutions
- **Personality Drift**: Slow changes over time

## Integration Points

### Simulation Integration

The cognitive architecture integrates with the Godot simulation through:

```python
# Godot calls mind for decisions
response = await mcp_client.call_tool(
    "decide_action",
    {
        "agent_id": "npc_001",
        "observation": current_state,
        "available_actions": valid_actions
    }
)

# Mind queries simulation for state
physical_state = await mcp_client.get_resource(
    f"simulation://agent/npc_001/physical_state"
)
```

### External Tools

Debugging and inspection via MCP:

```python
# MCP Playground can inspect both sides
mental_state = await get_resource("cognition://agent/npc_001/mental_state")
physical_state = await get_resource("simulation://agent/npc_001/physical_state")

# Visualize decision process
trace = await get_resource("cognition://agent/npc_001/last_decision_trace")
```

### Analytics & Monitoring

Event streams for analysis:

```python
events.emit("decision_made", {
    "agent_id": agent_id,
    "action": chosen_action,
    "reasoning_time": elapsed_ms,
    "tokens_used": token_count,
    "escalation_level": "conscious"
})
```

## Success Metrics

Success should be measured across different dimensions, recognizing that player experience and developer experience often have different requirements:

### Player Experience (Primary)
- **Believability**: Do NPCs feel like they have inner lives and coherent personalities?
- **Behavioral Richness**: Variety and depth of observed NPC behaviors
- **Emergent Narratives**: Interesting stories arising from NPC interactions
- **Emotional Investment**: Do players care about NPCs and their stories?
- **Discovery Satisfaction**: Joy in understanding NPC behaviors and motivations
- **Predictable Unpredictability**: NPCs are consistent enough to understand but varied enough to surprise

### Developer Experience (Enabling)
- **Debug Transparency**: Can developers understand why NPCs made specific decisions?
- **Iteration Speed**: How quickly can behaviors be tested and refined?
- **System Composability**: Can new features be added without breaking existing ones?
- **Maintainability**: Is the codebase understandable and modifiable?
- **Testing Confidence**: Can we verify NPCs behave as intended?

### Architectural Quality (Foundation)
- **Separation of Concerns**: Clean boundaries between cognition and simulation
- **Open/Closed Frontier**: Systems naturally evolve from exploratory to stable
  - **Frontier systems**: Still discovering requirements through implementation. Expect frequent modification as we learn what works. We won't know which systems these are until we build them and see what keeps changing
  - **Mature systems**: Requirements well-understood, API stabilized. Open for extension through new components/events/plugins, but closed for modification of core logic. Examples: event systems, component architectures, protocol definitions
  - **The transition**: As patterns emerge and requirements clarify, actively refactor frontier systems toward open/closed. A system that requires constant core modifications is a velocity killer
  - **Architectural impact**: Mature open/closed systems enable parallel development, reduce regression risk, and accelerate feature velocity dramatically. Much of "technical debt" is really systems stuck on the frontier that should have matured
- **Flexibility**: Can the system accommodate different types of minds and behaviors?
- **Extensibility**: How easily can new cognitive capabilities be added?
- **Robustness**: Graceful handling of edge cases and failures

### Performance (Scaling)
Note: These are considerations, not premature optimization targets
- **Cognitive Depth**: Rich behavior is more important than number of NPCs
- **Response Quality**: Better decisions matter more than faster decisions
- **Resource Efficiency**: Use resources wisely but don't sacrifice quality
- **Scaling Pathway**: Architecture should allow future optimization without redesign

## Conclusion

This cognitive architecture creates NPCs that are more than scripted automatons - they are entities with plans, memories, relationships, and the capacity for surprise. By treating NPCs as behavioral programmers who imperfectly execute their own plans, we achieve the emergence of lifelike behavior from comprehensible components.

The architecture's strength lies not in any single feature but in how the pieces work together:
- Plans provide structure, but execution noise creates variety
- Memory gives continuity, but forgetting prevents stagnation
- Social awareness enables coordination, but individual goals create conflict
- Multiple cognitive levels balance efficiency with capability

Most importantly, the system is designed to be **discoverable** - both by developers who extend it and players who interact with it. The use of MCP for integration ensures that the boundaries between simulation and cognition remain clean while enabling rich bidirectional communication.

This is not just a technical architecture but a framework for creating memorable characters and emergent narratives that will surprise even their creators.