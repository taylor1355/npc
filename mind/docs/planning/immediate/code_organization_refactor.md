# Code Organization Refactor

**Priority:** Medium
**Effort:** ~1-2 days
**Impact:** Improved maintainability, better discoverability

## Problem

Current code organization violates **locality of behavior** principles:

1. **Models dumped together** - `cognitive_architecture/models.py` contains unrelated models:
   - Memory, Observation (with all sub-observations), Action
   - Should be co-located with the code that uses them

2. **Poor discoverability** - Hard to find where models are defined when you need them:
   - Looking for `StatusObservation`? It's buried in `models.py`
   - Want to update `Action`? Hunt through a monolithic file

3. **Cognitive overhead** - Developers must remember which models live where:
   - Not self-documenting
   - Requires mental map of codebase structure

## Principle: Locality of Behavior

**Code that changes together should live together.**

### Good Examples in Our Codebase
```
cognitive_architecture/nodes/cognitive_update/
├── node.py           # CognitiveUpdateNode
├── models.py         # CognitiveUpdateOutput, WorkingMemory, NewMemory
└── prompt.md         # Prompt used by this node
```

✅ Everything for cognitive_update is in one place
✅ Models are co-located with their usage
✅ Self-documenting structure

### Bad Examples in Our Codebase
```
cognitive_architecture/
├── models.py         # EVERYTHING: Memory, Observation, Action, EntityData...
└── nodes/
    └── memory_retrieval/
        └── node.py   # Uses Memory, VectorDBQuery, but they're elsewhere
```

❌ Models scattered far from usage
❌ Need to import from parent directory
❌ Not obvious where models are defined

## Proposed Refactor

### 1. Move Memory Models to memory/

**Current:**
```
cognitive_architecture/models.py  # Memory, VectorDBMetadata, VectorDBQuery
```

**Proposed:**
```
cognitive_architecture/memory/
├── models.py         # Memory, VectorDBMetadata, VectorDBQuery
└── vector_db_memory.py  # VectorDBMemory (already here)
```

**Reasoning:** Memory models are only used by memory system

### 2. Create observations/ Module

**Current:**
```
cognitive_architecture/models.py  # Observation, StatusObservation, NeedsObservation, VisionObservation, ConversationObservation, ConversationMessage, EntityData
```

**Proposed:**
```
cognitive_architecture/observations/
├── __init__.py       # Re-exports for convenience
├── base.py           # Observation (base class)
├── status.py         # StatusObservation
├── needs.py          # NeedsObservation
├── vision.py         # VisionObservation, EntityData
└── conversation.py   # ConversationObservation, ConversationMessage
```

**Reasoning:**
- Observations are a cohesive concept
- Sub-observations are only used together
- Module allows grouping related models
- Still easy to import: `from mind.cognitive_architecture.observations import Observation`

### 3. Move Action to actions/

**Current:**
```
cognitive_architecture/models.py  # Action
```

**Proposed:**
```
cognitive_architecture/actions/
├── models.py         # Action
└── __init__.py       # Re-export Action
```

**Reasoning:**
- Prepares for future action-related code
- May add action validators, action history, etc.
- Keeps action concept isolated

### 4. Keep Node-Specific Models in Nodes

**Already good:**
```
nodes/cognitive_update/models.py   # CognitiveUpdateOutput, WorkingMemory
nodes/memory_query/models.py       # MemoryQueryOutput
nodes/action_selection/models.py   # ActionSelectionOutput
```

✅ Don't change these - they follow locality of behavior

## Migration Strategy

### Phase 1: Create New Locations (No Breaking Changes)
1. Create new directories/files
2. Copy models to new locations
3. Keep old imports working via `__init__.py` re-exports

### Phase 2: Update Imports Incrementally
1. Update one module at a time to use new imports
2. Run tests after each change
3. Commit small, focused changes

### Phase 3: Remove Old Locations
1. Delete deprecated `models.py`
2. Remove compatibility imports
3. Update documentation

## Benefits

1. **Self-documenting structure** - File location tells you what it's for
2. **Faster development** - Know exactly where to find/modify code
3. **Better testing** - Can test related models together
4. **Easier onboarding** - New developers understand structure immediately
5. **Future-proof** - Easy to add related functionality near existing code

## Example: Finding ConversationObservation

**Current workflow:**
1. "Where's ConversationObservation defined?"
2. Search codebase or ask someone
3. Find it in `models.py` among 200+ lines

**After refactor:**
1. "Where's ConversationObservation?"
2. Check `observations/conversation.py`
3. Found immediately - clear and obvious

## Non-Goals

- **Don't** create deep nesting (max 2-3 levels)
- **Don't** split models that are always used together
- **Don't** create single-file modules (unless they'll grow)

## Success Metrics

- ✅ No model lives in a generic `models.py` file
- ✅ Models are within 1 directory of their primary usage
- ✅ Developers can find model definitions in <10 seconds
- ✅ New models have an obvious home

## Related Principles

- **Locality of Behavior** (Primary)
- **Screaming Architecture** - Structure should reveal intent
- **Low Coupling, High Cohesion** - Related things together
- **YAGNI** - Don't over-engineer, but set up for growth

---

**See Also:**
- [Planning README](../README.md) - Prioritization principles
- [Roadmap](../roadmap.md) - Development priorities
