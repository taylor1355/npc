# Sleep as Cognitive Resource

**Status:** Backlog - Design ready for POC
**Priority:** High
**Related Docs:** [cluster_based_memory_consolidation.md](cluster_based_memory_consolidation.md), [basic_planning.md](basic_planning.md)

## Problem Statement

Sleep is currently a cosmetic need in the simulation - NPCs sleep because they "should," but it serves no functional cognitive purpose. This misses an opportunity for emergent intelligence differences and creates shallow gameplay where well-rested vs. exhausted NPCs behave identically.

More fundamentally, **cognitive maintenance activities need computational resources**. Memory consolidation requires LLM calls during downtime. Without a resource allocation mechanism tied to sleep, consolidation either runs unconditionally (expensive) or not at all (no emergence).

## Design Insight

**Sleep provides the time and cycles needed for memory consolidation.**

This creates emergent properties:
- Well-rested NPCs → more consolidation cycles → better memory → smarter decisions → prioritize sleep
- Sleep-deprived NPCs → fewer cycles → aggressive pruning → lost memories → poorer reasoning → less sleep

**Sleep gives cognitive architecture a functional role, not just a cosmetic one.**

## Design Goals (POC Scope)

- Implement biologically-inspired sleep cycles (light → deep → light phases)
- Run memory consolidation during deep sleep phases
- Reset working memory during first deep sleep
- Create observable differences based on sleep patterns (emotional state, memory quality)
- Enable incremental memory processing across multiple sleep cycles
- Simple sleep inertia for waking from deep vs. light sleep

## Technical Approach

### 1. Sleep Cycle Architecture

**90-minute cycles with three phases:**
```
Light (40min) → Deep (30min) → Light (20min)
```

**Biological inspiration:**
- Real sleep has ~90min cycles
- Deep sleep is when memory consolidation happens (hippocampus → cortex transfer)
- Deep sleep is harder to wake from (grogginess)
- First deep sleep period is when working memory gets processed

**Implementation:**
```python
class SleepCycle:
    """Sleep cycle with light and deep phases"""

    CYCLE_DURATION = 5400  # 90 minutes simulation time
    DEEP_SLEEP_BUDGET = 1000  # tokens per deep sleep period

    phases = [
        ("light", 2400),  # 40min - safe wake zone
        ("deep", 1800),   # 30min - consolidation happens here
        ("light", 1200),  # 20min - safe wake zone
    ]

    def __init__(self):
        self.current_phase = "light"
        self.cycle_number = 0
        self.working_memory_reset_done = False

    async def sleep_loop(self, npc):
        """Run sleep cycles until NPC wakes"""
        self.working_memory_reset_done = False

        while npc.is_sleeping():
            for phase_name, duration in self.phases:
                self.current_phase = phase_name

                if phase_name == "deep":
                    await self.deep_sleep_processing(npc)

                # Wait for phase to complete (or wake)
                await sleep_sim_time(duration, interruptible=True)

                if not npc.is_sleeping():
                    return  # Woken up mid-cycle

            self.cycle_number += 1
```

### 2. Deep Sleep Processing

**Two things happen during deep sleep:**

**A. Working memory reset (first deep sleep only):**
```python
async def deep_sleep_processing(self, npc):
    # First deep sleep: consolidate working memory
    if not self.working_memory_reset_done:
        working_memory_summary = str(npc.working_memory)

        if working_memory_summary.strip():
            # Working memory becomes a daily memory
            npc.daily_memories.append(Memory(
                content=f"Recent thoughts: {working_memory_summary}",
                importance=5.0,  # Medium importance
                timestamp=now()
            ))

        # Clear working memory for fresh start
        npc.working_memory.reset()
        self.working_memory_reset_done = True
```

**B. Memory consolidation (every deep sleep):**
```python
    # Process highest-urgency daily memories
    await consolidate_memories(npc, budget=self.DEEP_SLEEP_BUDGET)

async def consolidate_memories(npc, budget):
    """
    Process highest-urgency daily memories during deep sleep.
    Integrates with cluster_based_memory_consolidation.md
    """
    # Compute urgency = age * importance
    for mem in npc.daily_memories:
        hours_old = (now() - mem.timestamp) / 3600
        mem.urgency = hours_old * mem.importance

    # Select top-urgency memories within budget
    sorted_mems = sorted(npc.daily_memories,
                        key=lambda m: m.urgency,
                        reverse=True)

    to_process = select_within_budget(sorted_mems, budget)

    # Cluster and consolidate (see cluster_based_memory_consolidation.md)
    clusters = cluster_memories(to_process)
    for cluster in clusters:
        cohesiveness = compute_cohesiveness(cluster)
        prune_chunk_or_store(cluster, cohesiveness)
        if should_reflect(cluster, cohesiveness):
            generate_reflection(cluster)

    # Remove processed memories from daily buffer
    npc.daily_memories = [m for m in npc.daily_memories
                          if m not in to_process]
```

**Open question:** Should budget be fixed (1000 tokens) or scale with cycle number (later cycles have more context)?

### 3. Sleep Patterns & Memory Quality

**Sleep duration determines consolidation cycles:**

**Well-rested (8 hours = ~5 cycles):**
- 5 deep sleep periods × 1000 tokens = 5000 tokens of consolidation
- Most daily memories processed
- Reflections generated for cohesive clusters
- Working memory reset → fresh cognitive state

**Moderate sleep (4.5 hours = 3 cycles):**
- 3 deep sleep periods × 1000 tokens = 3000 tokens
- Higher-urgency memories processed
- Lower-urgency memories remain in daily buffer
- Partial backlog accumulation

**Short sleep (90min = 1 cycle):**
- 1 deep sleep period = 1000 tokens
- Only most urgent memories processed
- Significant backlog accumulation
- Working memory still gets reset

**No sleep (all-nighter):**
- No consolidation
- Daily buffer fills up with unprocessed memories
- Working memory not reset (cognitive clutter)
- Next sleep session has VERY high urgency memories

### 4. Wake State & Sleep Inertia

**Simple emotional state update on wake:**

```python
def on_wake(npc):
    """Update emotional/cognitive state based on wake circumstances"""

    if npc.sleep_cycle.current_phase == "deep":
        # Woken from deep sleep - grogginess
        npc.emotional_state.clarity = 0.3
        npc.emotional_state.mood -= 0.2
        npc.emotional_state.energy = 0.4

    else:  # light phase
        # Natural wake - feeling good
        hours_slept = (now() - npc.sleep_start_time) / 3600
        quality = compute_sleep_quality(hours_slept)

        npc.emotional_state.clarity = 0.7 + quality * 0.3
        npc.emotional_state.mood += 0.1 + quality * 0.2
        npc.emotional_state.energy = 0.5 + quality * 0.5

def compute_sleep_quality(hours):
    """Simple quality curve: 0-4h = poor, 4-6h = ok, 6-8h = good, 8+ = great"""
    if hours < 4:
        return 0.2
    elif hours < 6:
        return 0.5
    elif hours < 8:
        return 0.8
    else:
        return 1.0
```

**Observable behaviors:**

**Woken from deep sleep (e.g., alarm at 3am):**
- "Ugh... what... give me a minute..."
- Low clarity, cranky mood, low energy
- Makes worse decisions until clarity recovers

**Natural wake from light sleep (e.g., 7am after 8 hours):**
- "Good morning! Ready to start the day."
- High clarity, good mood, energized
- Makes better decisions

**Open question:** Should sleep inertia wear off over time (e.g., 15min) or is instant state change sufficient for POC?

### 5. Interruption Handling & Autonomy

**NPCs can wake/sleep freely:**
- Sleep is not a forced "end of day" event
- NPCs decide when to sleep (planning system's job)
- Can be woken by external events (alarm, conversation, danger)

**Interruption during light sleep:**
- No penalty - just stops consolidation
- Already-processed memories stay processed
- Unprocessed memories remain in daily buffer

**Interruption during deep sleep:**
- Grogginess penalty (low clarity)
- That cycle's consolidation completes (already async running)
- Lost opportunity for subsequent cycles

**Multiple sleep sessions:**
- Naps during day + night sleep both work
- Each sleep session continues processing daily buffer
- Memory urgency increases while awake
- Incremental progress across all sleep sessions

**Open question:** Should there be a cooldown between sleep sessions (can't immediately go back to sleep)? Or let NPCs nap freely?

## Benefits

**Emergent Complexity:**
- Intelligence differences arise from sleep patterns, not hand-tuned stats
- NPCs develop unique cognitive profiles based on life circumstances
- Observable consequences of lifestyle choices

**Strategic Gameplay:**
- Players notice NPCs are "sharper" when well-rested
- Sleep-deprived NPCs make mistakes, forget commitments
- NPC schedules affect their capabilities

**Cognitive Realism:**
- Mirrors human sleep's role in memory consolidation
- Working memory reset matches sleep's "fresh start" feeling
- Grogginess from deep wake is relatable

**Development Velocity:**
- Establishes pattern for future sleep-time maintenance modules
- Clear resource allocation mechanism (cycles × budget)
- Modular: New maintenance activities just add to deep sleep processing

## Implementation Phases

### Phase 1: Sleep Cycle Infrastructure (Week 1)
- Implement SleepCycle class with phase tracking
- Sleep loop with interruptible phases
- Track cycle number and current phase
- Log phase transitions for debugging
- Success: NPCs cycle through light/deep/light phases

### Phase 2: Deep Sleep Processing (Week 2)
- Implement working memory reset (first deep sleep)
- Integrate memory consolidation with cluster_based_memory_consolidation.md
- Process memories with urgency-based selection
- Success: Daily buffer drains during sleep, working memory resets

### Phase 3: Wake State & Inertia (Week 3)
- Implement emotional state updates on wake
- Differentiate deep wake (groggy) vs light wake (refreshed)
- Compute sleep quality from duration
- Success: Observable mood/clarity differences

### Phase 4: Testing & Tuning (Week 4)
- Test various sleep patterns (naps, all-nighters, normal sleep)
- Tune budget per deep sleep (1000 tokens optimal?)
- Verify interruption handling works correctly
- A/B test different quality curves
- Success: System feels balanced and natural

## Dependencies

- ✅ Sleep/wake tracking in simulation (NPCs can initiate/interrupt sleep)
- ⚠️ cluster_based_memory_consolidation.md (primary work done during deep sleep)
- ⚠️ Working memory model (needs reset() method)
- ⚠️ Emotional state model (clarity, mood, energy fields)
- ⚠️ Planning system (NPCs decide when to sleep, but not required for POC)

## Priority Rationale

**Obviousness:** High - Sleep obviously affects cognitive function. Modeling sleep cycles is a clear mechanism. The daily_memories buffer needs a trigger for consolidation, and sleep is the obvious answer.

**Development Velocity:** Very Positive
- **Unlocks features:** Memory consolidation, and later plan refinement, social updates, etc.
- **Cost control:** Bounds token usage to deep sleep cycles
- **Foundation:** Establishes "sleep-time processing" as a pattern
- **Modularity:** Future maintenance activities just add to deep sleep
- **Parallel development:** Once sleep cycles exist, multiple features can build on them

**Concreteness:** Very High - Players notice immediately:
- "The blacksmith forgot our conversation because he pulled an all-nighter"
- "The scholar is insightful because she sleeps 8 hours"
- Sleep-deprived NPCs are cranky and forgetful
- Emergent personality differences without invisible stats

## Open Questions & Design Challenges

### Sleep Cycle Parameters
- Is 90min the right cycle length for gameplay feel?
- Should cycle length vary by individual (genetic variation)?
- Is 40/30/20 split optimal, or should deep sleep be longer/shorter?

### Budget Allocation
- Fixed 1000 tokens per deep sleep, or dynamic?
- Should budget scale with cycle number (more context in later cycles)?
- Should budget scale with NPC "experience" (older NPCs consolidate faster)?
- Hard cap on total consolidation per sleep session?

### Working Memory Reset
- Should working memory always reset on first deep sleep, or only if sleep > threshold?
- What if NPC takes multiple short naps - working memory resets each time?
- Should working memory summary importance be fixed (5.0) or computed?

### Sleep Inertia
- Instant state change vs. gradual recovery (15-30min)?
- How severe should deep wake grogginess be (clarity = 0.3 seems harsh)?
- Should clarity affect decision-making, or just dialog/appearance?
- Can stimulants (coffee) reduce sleep inertia?

### Interruption Mechanics
- Can NPCs be woken by all events, or only high-priority ones?
- Should deep sleep be "harder" to interrupt (requires louder stimuli)?
- What about sleep alarms - waking from deep at specific time?

### Memory Urgency Formula
- `urgency = age * importance` - is this the right formula?
- Should urgency be non-linear (exponential growth with age)?
- Should low-importance memories cap urgency (never urgent)?
- Should very old memories decay urgency (ancient = irrelevant)?

### Multiple Sleep Sessions
- Should there be cooldown between sleep sessions?
- How to prevent gaming the system (sleep 2h, wake, sleep 2h × 4 = same as 8h)?
- Should sleep efficiency decrease with fragmentation?
- Does biology suggest polyphasic sleep is less effective for consolidation?

### Integration with Planning
- Should NPCs plan when to sleep based on memory backlog?
- Can NPCs recognize "I need sleep, I'm forgetting things"?
- Should sleep priority increase with daily_memories buffer size?
- Emergency sleep when buffer overflows?

### Observability
- How do players know an NPC is sleep-deprived? Visual indicators? Dialog?
- Should NPCs complain about being tired?
- Debug view: show cycle number, phase, memories processed
- Metrics: track sleep quality, consolidation stats, memory buffer size

## Success Criteria

- [ ] Sleep cycles progress through light/deep/light phases correctly
- [ ] Working memory resets during first deep sleep
- [ ] Daily memories processed incrementally across cycles
- [ ] Memory consolidation quality scales with sleep duration
- [ ] Sleep-deprived NPCs have fuller daily_memories buffers
- [ ] Well-rested NPCs have emptier buffers and more reflections in long-term memory
- [ ] Waking from deep sleep produces groggy state
- [ ] Waking from light sleep produces refreshed state
- [ ] Emotional state (clarity, mood, energy) varies observably with sleep quality
- [ ] Interruption handling works (can wake mid-cycle without bugs)
- [ ] Token costs remain predictable (budget × cycles)
- [ ] Playtest: Sleep feels mechanically important, not just cosmetic
- [ ] Playtest: Players notice NPC behavior differences based on sleep

## Future Extensions (NOT in POC)

These are bio-inspired ideas to add later:

### Sleep Debt & Deprivation
- Track cumulative sleep debt over days
- Chronic deprivation → cognitive degradation beyond memory
- Sleep debt → crash into extended deep sleep (45min instead of 30min)
- 72+ hours without deep sleep → forced collapse

### REM Sleep Processing
- Third phase with distinct processing (currently REM = light in POC)
- Emotional processing during REM
- Creative recombination ("dreams")
- Cross-cluster reflections and inspired insights

### Sleep Inertia Dynamics
- Grogginess wears off over 15-30 minutes
- Different severity based on sleep depth
- Can be reduced by stimulants (coffee)
- Affects decision quality temporarily

### Dynamic Budget Allocation
- Later cycles have more budget (more context to work with)
- Budget varies by NPC (individual differences in consolidation efficiency)
- Budget increases with sleep quality (environment, stress level)
- Different maintenance activities (plan refinement, social updates) share budget

### Advanced Wake Mechanics
- Track which stage NPC naturally wakes in (usually light)
- Sleep alarms optimize for light phase wake (smart alarm)
- Dream recall if woken during REM
- Microsleeps if severely sleep-deprived

### Plan Refinement During Sleep
- Analyze plan execution: what worked, what didn't?
- Update behavioral chunks with actual outcomes
- Revise longer-term plans based on daily progress

### Social Model Updates
- Process social interactions during sleep
- Update theory-of-mind models
- Consolidate relationship changes

### Emotional Processing
- Process emotionally intense memories during REM
- PTSD-like effects if trauma can't be processed (insufficient sleep)
- Mood regulation through emotional memory organization

### Skill Consolidation
- Mental practice/rehearsal of learned behaviors
- Transfer learning across similar situations
- Strengthen successful patterns, weaken failed ones

### Sleep Quality Factors
- Environment affects consolidation efficiency (noise, comfort)
- Stress/anxiety reduces deep sleep quality
- Interrupted sleep less efficient than continuous
- Age/individual differences in sleep needs

### Long-term Memory Maintenance
- Periodic clustering of old long-term memories (see cluster_based_memory_consolidation.md Future Extensions)
- Compress distant past into abstract summaries
- Strengthen frequently-accessed memories
- Prune old low-importance memories

### Sleep as Social Activity
- Co-sleeping affects sleep quality
- Household dynamics (baby wakes parent)
- Cultural sleep patterns (siestas, polyphasic schedules)
- Sleep schedules coordinated with partners

### Chronotype & Circadian Rhythms
- Morning people vs night owls
- Optimal sleep time varies by individual
- Consolidation efficiency varies with circadian alignment
- Seasonal effects (winter = more sleep needed)
