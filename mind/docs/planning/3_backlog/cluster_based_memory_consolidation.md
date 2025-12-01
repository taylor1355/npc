# Cluster-Based Memory Consolidation

**Status:** Backlog - Design ready for implementation
**Priority:** High
**Related Docs:** [generative_agents_memory.md](generative_agents_memory.md), [memory_reflection.md](memory_reflection.md)

## Problem Statement

Current memory consolidation ([memory_consolidation/node.py](../../src/mind/cognitive_architecture/nodes/memory_consolidation/node.py)) is a simple placeholder that adds all daily memories to long-term storage without filtering, reflection, or abstraction. This leads to memory bloat, no learning from patterns, and missed opportunities for NPCs to develop coherent understanding of their experiences.

## Design Goals

- Organize daily memories by thematic coherence using embedding-based clustering
- Allocate cognitive resources (memory budget, reflection generation) based on cluster importance
- Implement intelligent memory processing: prune low-value memories, chunk similar ones, preserve important unique memories
- Generate higher-order reflections from coherent experience clusters
- Enable emergent personality development through pattern recognition

## Technical Approach

### 1. Clustering Daily Memories

At consolidation time (during sleep/downtime), cluster the `state.daily_memories` by embedding similarity:

**Baseline:** All memories in a single cluster (sanity check, no clustering)

**Clustering algorithm options:**
- K-means on embeddings (requires pre-specifying K)
- HDBSCAN (automatic cluster count, density-based)
- Hierarchical clustering with dynamic threshold

**Cluster count determination:**
- Elbow method on within-cluster sum of squares
- Silhouette score maximization
- Domain heuristic: 3-7 clusters for a typical day (adjustable based on memory count)

**Open question:** Should clustering consider temporal proximity in addition to semantic similarity? Memories from the same "narrative arc" might cluster together even if embeddings differ.

### 2. Cohesiveness Scoring

For each cluster, estimate cohesiveness to determine "theme strength":

**Mathematical representation:**
- Model cluster as Gaussian distribution in embedding space
- Extract mean embedding and covariance matrix

**Cohesiveness metric options:**
- **Inverse variance:** Tight clusters (low variance) = high cohesiveness
- **Silhouette score:** How well-separated from other clusters
- **Average pairwise cosine similarity:** Mean similarity of all memory pairs in cluster
- **Weighted by importance:** Factor in constituent memories' importance scores
- **Temporal coherence bonus:** Memories from similar time windows boost cohesiveness

**High-dimensional challenge:** Current embeddings are 384-dimensional (MiniLM). Full covariance matrix = 384x384. Solutions:
- **Diagonal covariance:** Assume dimension independence
- **Dimensionality reduction:** PCA/UMAP to reduce to 10-50 dimensions before Gaussian estimation
- **Matryoshka embeddings:** Use variable-size embeddings (models trained to be meaningful at multiple truncation sizes: 64, 128, 256, 384 dims). Could use smaller dimensions for clustering/cohesiveness computation, full dimensions for retrieval
- **Simplify to cosine similarity:** Skip Gaussian entirely, use average pairwise cosine similarity instead

**Open question:** Should emotionally intense memories (once emotional system exists) get special weighting in cohesiveness calculation?

### 3. Memory Budget Allocation

Allocate consolidation resources based on cluster cohesiveness:

**Budget dimensions:**
- **Storage budget:** Max memories to preserve from this cluster
- **Token budget:** LLM calls allowed for chunking/reflection
- **Reflection slots:** Number of reflections to generate

**Allocation strategy:**
```
cohesive_clusters = sorted(clusters, key=cohesiveness, reverse=True)
budgets = allocate_proportional_to_cohesiveness(cohesive_clusters)
```

**High-cohesion clusters get:**
- Larger storage budget (preserve more detail)
- More tokens for sophisticated chunking
- Multiple reflection generations

**Low-cohesion clusters get:**
- Aggressive pruning
- Simple rule-based chunking (or no chunking)
- No reflections

**Open question:** Should budget allocation also consider cross-day patterns? E.g., if today's cluster matches a long-term memory theme, allocate extra budget?

### 4. Three-Way Memory Processing

For each memory in each cluster, decide: prune, chunk, or store directly.

**Decision tree:**
```
for memory in cluster:
    if should_prune(memory, cluster):
        # Drop low-importance memories from low-cohesion clusters
        discard(memory)
    elif should_chunk(memory, cluster):
        # Merge with similar memories into synthesis
        add_to_chunk_group(memory)
    else:
        # Store verbatim (high importance or unique)
        store_directly(memory)

# After processing all memories
for chunk_group in cluster.chunk_groups:
    synthesized = await llm_synthesize(chunk_group)
    store_directly(synthesized)
```

**Pruning criteria:**
- Memory importance < threshold (e.g., < 3/10)
- Cluster cohesiveness < threshold (random/incoherent experiences)
- Redundant with existing long-term memories

**Chunking criteria:**
- Multiple similar memories (high pairwise similarity)
- Cluster size exceeds budget (must compress)
- Routine/repetitive experiences (e.g., "ate breakfast" x 5 → "had normal meals")

**Direct storage criteria:**
- High importance score (≥ 7/10)
- Unique/novel memory (low similarity to others)
- From high-cohesion clusters (preserve detail)

**Open question:** How to balance individual memory importance vs. cluster cohesiveness when they conflict? (E.g., high-importance memory in low-cohesion cluster)

### 5. Reflection Generation

Generate higher-order memories for the most cohesive clusters:

**Reflection allocation:**
- Top N clusters (e.g., N=3) get reflection generation
- Higher cohesiveness → more reflections per cluster
- Reflection budget constrained by available compute (sleep time)

**Reflection types:**
- **Pattern synthesis:** "I've been spending a lot of time with the blacksmith lately"
- **Causal insight:** "When I visit the market early, I find better deals"
- **Social learning:** "The baker is friendlier after I helped him last week"
- **Self-reflection:** "I feel more energized when I get enough sleep"
- **Inspired connections:** Creative links between clusters (cross-cluster reflections)

**Implementation:**
```python
for cluster, cohesiveness in top_cohesive_clusters[:3]:
    reflection_prompt = f"""
    Given these related memories from today:
    {format_cluster_memories(cluster)}

    Generate 1-2 higher-level insights or patterns you notice.
    """

    reflections = await llm_generate_reflections(reflection_prompt)

    # Reflections become high-importance memories
    for reflection in reflections:
        state.daily_memories.append(Memory(
            content=reflection,
            importance=8.0,  # High importance
            timestamp=current_time,
            is_reflection=True
        ))
```

**Open question:** Should reflections reference their source memories (graph edges)? This would enable "why do I believe this?" reasoning chains.

## Benefits

**Memory Efficiency:**
- Reduces memory bloat through intelligent pruning
- Compresses redundant memories via chunking
- Focuses storage on coherent, meaningful experiences

**Emergent Intelligence:**
- NPCs develop themes/interests based on experience clustering
- Reflections create wisdom/learned patterns over time
- Higher-order thinking emerges from pattern recognition

**Personality Development:**
- Consistently cohesive clusters → strong personality traits
- NPCs with varied experiences → complex, multifaceted personalities
- Individual differences in what experiences cluster (personal relevance)

**Player-Facing:**
- NPCs reference past patterns coherently ("I've learned that...")
- Conversations reveal accumulated wisdom from reflections
- Character development visible over time through changed reflections

## Implementation Phases

### Phase 1: Basic Clustering (Week 1)
- Implement embedding-based clustering (start with K-means, K=5)
- Compute simple cohesiveness metric (average pairwise cosine similarity)
- Visualize/log clusters for debugging
- Success: Daily memories organized into meaningful groups

### Phase 2: Budget Allocation & Processing (Week 2)
- Implement cohesiveness-based budget allocation
- Build prune/chunk/store decision logic
- LLM-based memory chunking for high-redundancy groups
- Success: Memory count reduced, important memories preserved

### Phase 3: Reflection Generation (Week 3)
- Implement reflection prompts for top-N clusters
- Store reflections as high-importance memories
- Test reflection quality and relevance
- Success: NPCs generate insightful higher-order memories

### Phase 4: Tuning & Optimization (Week 4)
- Experiment with clustering algorithms (HDBSCAN, hierarchical)
- Tune cohesiveness metrics and budget allocation
- Optimize token usage for consolidation
- A/B test different strategies
- Success: Consolidation is efficient and produces believable results

## Dependencies

- ✅ Current memory system (VectorDBMemory, daily buffer) - [vector_db_memory.py](../../src/mind/cognitive_architecture/memory/vector_db_memory.py)
- ✅ Importance scoring during memory formation - [cognitive_update/node.py](../../src/mind/cognitive_architecture/nodes/cognitive_update/node.py)
- ✅ Placeholder consolidation node to replace - [memory_consolidation/node.py](../../src/mind/cognitive_architecture/nodes/memory_consolidation/node.py)
- ✅ LLM access for chunking and reflection generation
- ⚠️ Sleep/downtime periods when consolidation runs (see [sleep_as_cognitive_resource.md](sleep_as_cognitive_resource.md))

## Priority Rationale

**Obviousness:** High - The TODO in memory_consolidation/node.py explicitly calls for this. Clustering is a clear, well-understood approach to organizing memories.

**Development Velocity:** Positive
- **Short-term:** Isolated to memory consolidation node, doesn't entangle other systems
- **Long-term:** Enables sleep-time cognitive modules (plan refinement, social model updates, skill consolidation)
- **Tech tree:** Reflection generation unlocks metacognition and learning behaviors
- **Token efficiency:** Reduces long-term memory bloat, lowers retrieval costs
- **Foundation:** Establishes pattern for other "batch processing during downtime" features

**Concreteness:** High - Players will notice:
- NPCs reference learned patterns ("I've learned that...")
- Characters develop coherent interests/themes over time
- Memory efficiency improves performance (faster retrieval)
- NPCs with more sleep have richer reflections (see sleep_as_cognitive_resource.md)

## Open Questions & Design Challenges

### Clustering Algorithm Selection
- Which algorithm works best for variable-size daily memory buffers?
- How to handle days with very few memories (< 5)? Skip clustering?
- Should we have a minimum cluster size to avoid singleton clusters?

### Cohesiveness Metrics
- Is Gaussian estimation overkill? Would simpler metrics suffice?
- How to combine multiple cohesiveness signals (separation, density, importance)?
- Should temporal coherence be part of cohesiveness or a separate factor?
- **Dimensionality:** Full 384-dim embeddings, reduced (PCA/UMAP), or Matryoshka truncation?

### Budget Allocation Fairness
- Should low-cohesion clusters get *zero* budget or just *less* budget?
- Risk: Completely ignoring low-cohesion clusters might lose important random events
- Alternative: Minimum budget for all clusters, bonus for high cohesiveness

### Chunking vs. Pruning Trade-offs
- Chunking preserves information but costs LLM tokens
- Pruning is free but loses information
- When is each strategy preferred?
- Could use rule-based chunking for low-budget clusters (no LLM needed)

### Reflection Quality
- How to ensure reflections are insightful, not trivial?
- Should reflections be validated before storing (quality check)?
- Can reflections be "wrong" and need correction later?
- How to avoid reflection loops (reflection about reflections about reflections...)?

### Cross-Day Patterns
- Should consolidation compare today's clusters with historical clusters?
- Could detect recurring themes: "This is the third time this week I've argued with the guard"
- Requires maintaining cluster centroids or summaries over time
- Complexity vs. benefit trade-off

### Emotional/Social Integration
- Should emotionally intense memories get special clustering treatment?
- Should social interactions cluster separately from solo experiences?
- How to integrate with future emotional_memory.md and social_memory.md features?

### Computational Cost
- Clustering + multiple LLM calls could be expensive
- How to limit token usage while maintaining quality?
- Should consolidation be interruptible if NPC wakes up early?
- Batch processing multiple NPCs' consolidations for efficiency?

### Evaluation & Validation
- How to measure if consolidation is "working"?
- Qualitative: Do NPCs feel smarter/more coherent?
- Quantitative: Memory compression ratio, retrieval performance, token costs
- Player surveys: Do players notice NPC personality development?

## Success Criteria

- [ ] Daily memories successfully clustered into 3-7 coherent groups
- [ ] Cohesiveness scores correlate with human judgment of "theme strength"
- [ ] Memory count reduced by 40-60% through pruning and chunking
- [ ] High-importance and unique memories preserved (no critical memory loss)
- [ ] Reflections generated for top cohesive clusters are insightful and accurate
- [ ] NPCs reference reflections in conversations and decision-making
- [ ] Token costs for consolidation are acceptable (< 5K tokens per day per NPC)
- [ ] Player playtest: NPCs feel like they "learn" and "develop personality" over time
- [ ] Performance: Consolidation completes within simulated sleep duration
- [ ] Integration: Works seamlessly with sleep_as_cognitive_resource.md budgeting

## Future Extensions

- **Knowledge graph integration:** Reflections create edges between related memory clusters
- **Multi-day clustering:** Detect recurring themes across weeks/months
- **Social cluster sharing:** NPCs can share reflections with each other (cultural learning)
- **Emotional processing:** Clusters with intense emotions get therapeutic processing
- **Skill abstraction:** Repeated action patterns become learned behavioral chunks
- **Forgetting curves:** Old, low-cohesion memories gradually decay even after storage
- **Periodic long-term memory maintenance:** During sleep, cluster and chunk older low-importance memories that have accumulated in long-term storage. Similar to daily consolidation but operating on week/month-old memories. Gradually compress "distant past" into increasingly abstract summaries while preserving important/unique events. This prevents long-term memory from growing unbounded while maintaining coherent life history.
