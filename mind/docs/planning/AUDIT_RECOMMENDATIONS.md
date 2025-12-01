# Planning System Audit & Recommendations

**Based on:** Lessons learned from reorganizing the npc-simulation planning system
**Prepared:** November 2025

---

## Executive Summary

The mind repo's planning system has good foundations (prioritization heuristics, design docs) but would benefit from the structural improvements applied to npc-simulation. Key issues: no INDEX files for navigation, stale roadmap dates, inconsistent structure between tiers.

**Recommended effort:** 2-3 hours for structural changes, plus ongoing maintenance.

---

## Current State Assessment

### What's Good

1. **Prioritization heuristics** in README.md - Excellent framework (obviousness, velocity, concreteness)
2. **Rich backlog** - 25 well-structured design documents
3. **Clear immediate tasks** - Detailed implementation plans with phases
4. **Completed milestones** - Good historical reference in roadmap

### Issues Identified

| Issue | Severity | Description |
|-------|----------|-------------|
| No INDEX files | HIGH | 25 backlog docs with no navigation or summary |
| Stale roadmap | HIGH | "Last Updated: October 10, 2025" - nearly 2 months old |
| Inconsistent structure | MEDIUM | roadmap.md is a file; backlog/ and immediate/ are directories |
| No numerical prefixes | LOW | Directories don't sort intuitively |
| Unclear tier separation | MEDIUM | immediate/ has docs but unclear relationship to roadmap |
| Git audit needed | HIGH | "Near-term Priorities (Next 1-2 Weeks)" from October - likely completed |

---

## Recommended Changes

### 1. Add Numerical Prefixes to Directories

**Current:**
```
docs/planning/
├── backlog/
├── immediate/
├── README.md
├── roadmap.md
└── testing-plan.md
```

**Recommended:**
```
docs/planning/
├── 1_immediate/
│   └── INDEX.md        # Main file + individual task docs if large
├── 2_roadmap/
│   └── INDEX.md        # Convert roadmap.md to this
├── 3_backlog/
│   ├── INDEX.md        # NEW - summary of all 25 docs
│   └── [existing design docs]
├── README.md           # Keep as system overview
└── testing-plan.md     # Move to appropriate tier or archive
```

**Why:** Numerical prefixes provide intuitive sorting in file browsers and make the tier hierarchy obvious.

### 2. Create Backlog INDEX.md

The backlog has 25 design documents with no navigation. Create an index that categorizes them:

```markdown
# Feature Backlog

## Memory Systems
- [Basic Memory System](basic_memory_system.md) - **Status**: Complete
- [Memory Reflection](memory_reflection.md) - **Status**: Design Phase
- [Generative Agents Memory](generative_agents_memory.md) - **Status**: Design Phase
...

## Planning & Behavior
- [Basic Planning](basic_planning.md) - **Status**: Design Phase | **Priority**: High
- [Hierarchical Planning](hierarchical_planning.md) - **Status**: Design Phase
- [Behavioral Chunking](behavioral_chunking.md) - **Status**: Design Phase
...

## Emotional & Social
- [Emotional Memory](emotional_memory.md) - ...
- [Social Memory](social_memory.md) - ...
- [Theory of Mind](theory_of_mind.md) - ...
...
```

**Benefits:**
- Quick overview of what's available
- Status tracking without opening each file
- Categorization aids discovery

### 3. Convert roadmap.md to 2_roadmap/INDEX.md

The roadmap is currently a single file. Consider:

1. **Move to directory:** `docs/planning/2_roadmap/INDEX.md`
2. **Update dates:** "Last Updated: October 10, 2025" is stale
3. **Audit against git:** Check if "Near-term Priorities" items were completed
4. **Simplify phases:** If the roadmap feels overwhelming, defer ambitious sections

The npc-simulation roadmap went from 5 phases to 2 focused ones. Ask: what's the smallest roadmap that captures your actual priorities?

### 4. Audit Completed Work

Cross-reference the roadmap's "Near-term Priorities (Next 1-2 Weeks)" against reality:

| Roadmap Item | Status Check |
|--------------|--------------|
| End-to-End Integration Testing | Was this done? Check git log |
| Prompt Refinement | Was this done? |
| Conversation History Formalization | Was this done or deferred? |

Also check immediate/ tasks:
- `fix-entity-hallucination.md` - Phase 1 marked as "✅ DONE", Phase 2 as "IN PROGRESS" - current?
- `code_organization_refactor.md` - Status?
- `simulation-mechanics-understanding.md` - Status?

### 5. Establish the "No Duplication" Rule

**Critical principle:** Items should exist in exactly ONE tier.

- If a backlog item is promoted to roadmap, **move the design doc** to the roadmap folder
- If a roadmap item becomes immediate, **move or reference** from immediate tier
- Never have the same item listed in multiple places with different statuses

### 6. Consider What to Defer

Looking at the backlog, there are 25 design documents. Some questions:

- **potential_features.md** (12KB) - Is this a grab-bag? Consider splitting into actionable items or archiving
- **sleep_as_cognitive_resource.md** (18KB) - Large design doc - is this near-term priority or long-term vision?
- **cluster_based_memory_consolidation.md** (15KB) - Same question

For items that are "interesting but not now", consider adding a "Deferred" status with explicit reasoning.

---

## Implementation Checklist

### Phase 1: Structural Changes (~1 hour)

```bash
# Rename directories with numerical prefixes
git mv docs/planning/immediate docs/planning/1_immediate
git mv docs/planning/backlog docs/planning/3_backlog

# Create roadmap directory and move roadmap.md
mkdir docs/planning/2_roadmap
git mv docs/planning/roadmap.md docs/planning/2_roadmap/INDEX.md
```

- [ ] Rename directories with numerical prefixes
- [ ] Create `2_roadmap/` directory, move `roadmap.md` → `INDEX.md`
- [ ] Create `1_immediate/INDEX.md` that lists the 3 task files
- [ ] Create `3_backlog/INDEX.md` with categorized summary

### Phase 2: Content Audit (~1-2 hours)

- [ ] Update roadmap dates and check completed items against git log
- [ ] Update immediate task statuses (are they still in progress?)
- [ ] Add Status/Priority to backlog INDEX entries
- [ ] Identify items to defer with explicit reasoning

### Phase 3: Ongoing Maintenance

- [ ] Update planning docs when committing related changes
- [ ] Mark items complete as they're finished
- [ ] Promote items from backlog → roadmap → immediate as they mature
- [ ] Periodic grooming (monthly?) to clean up stale items

---

## Quick Reference: Entry Formats

### Immediate Task Entry
```markdown
### Fix Entity Hallucination
**Status**: In Progress (Phase 2) | **Effort**: 2-3 days

When LLM returns invalid entity IDs, validate and retry with error feedback.

**Files**: [fix-entity-hallucination.md](fix-entity-hallucination.md)
```

### Roadmap Entry
```markdown
### 1.2 Prompt Refinement
**Status**: Not Started | **Effort**: 1-2 days
**Priority**: MEDIUM

Remove meta-cognitive framing from prompts for better character immersion.

**Tasks:**
- Review 3 prompt files
- Rewrite for direct embodiment
- Test behavior changes
```

### Backlog Entry
```markdown
#### M.3. [Generative Agents Memory](generative_agents_memory.md)
**Status**: Design Phase
**Priority**: Medium
**Complexity**: Medium

Enhanced memory retrieval with contrastive queries and reflections.
```

---

## Questions to Discuss

1. **Roadmap scope:** Is the current roadmap achievable? Would deferring "Conversation History Formalization" and other lower-priority items help focus?

2. **Backlog organization:** Should the 25 backlog docs be categorized (Memory, Planning, Emotional, Infrastructure)? Would a simpler grouping work better?

3. **Immediate task status:** Are the 3 immediate tasks current? Should any be archived or promoted to "completed"?

4. **Cross-repo alignment:** Both repos have planning systems now. Should there be any coordination or shared conventions?

---

## Related Documents

- `README.md` - Prioritization heuristics (keep as-is, it's good)
- `testing-plan.md` - Consider moving to `2_roadmap/` or archiving
- npc-simulation's `docs/meta/planning/` - Reference implementation of these patterns
