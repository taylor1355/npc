# Planning Documentation

This directory organizes planning documents for the NPC cognitive architecture project.

## Prioritization Heuristics

When deciding what to work on next, consider these principles:

### 1. Obviousness
Do obvious things first to constrain less obvious things.
- Clear, unambiguous tasks reduce the solution space
- Obvious foundations make complex features easier to reason about
- Example: Implement basic memory system before hierarchical planning
- Note: "Obvious" doesn't mean easy - it means the requirement is clear even if implementation is complex

### 2. Development Velocity
How does this affect the speed and ease of future development?
- **Positive factors**: Tools, debugging infrastructure, reusable systems, unlocking dependencies
- **Negative factors**: Added complexity, technical debt, system stretching beyond design
- **Tech tree effects**: Features that partially implement or enable future features
- Example: Basic planning has upfront cost but enables hierarchical planning, behavioral chunking
- Note: Player-facing features often have negative short-term velocity (complexity) but positive long-term effects (foundation for more features)

### 3. Concreteness
Does this directly improve the player experience?
- Visible behaviors, gameplay improvements, things players notice and enjoy
- The earlier you have player-facing features, the longer you have to refine them
- Example: NPCs with daily routines vs random behavior
- Note: This is distinct from developer experience - debugging tools have zero concreteness but high development velocity

## Applying These Principles

These heuristics work together to guide development:

- **Obvious** features provide solid foundations
- **Accelerating** features multiply development speed
- **Concrete** features enable rapid iteration

The best features to prioritize often satisfy multiple criteria. A debugging tool might be obvious (clearly needed), accelerating (speeds up everything else), and concrete (immediately usable).

## Feature Evaluation Questions

When evaluating potential features, ask:

1. **Is it obvious?** Will implementing this clarify other design decisions?
2. **Does it accelerate development?** Will this make other features easier to build?
3. **Is it concrete?** Can we build a working version quickly and iterate?
4. **What does it unlock?** What becomes possible after this is built?
5. **What does it depend on?** What must exist first?

## Strategic Sequencing

Consider the dependency graph between features. Sometimes a seemingly minor feature unlocks many others, making it high priority despite appearing unimportant in isolation.

Similarly, some features become trivial once the right infrastructure exists, while others remain complex regardless. Prioritize infrastructure that simplifies many downstream features.

## Balancing Planning and Implementation

These heuristics favor action over analysis paralysis. Perfect plans are less valuable than imperfect implementations that can be refined through use. However, some upfront thinking about sequencing and dependencies pays dividends.

The goal is to maintain momentum while building in the right order - not to create a perfect plan, but to avoid costly architectural mistakes and rework.