# Improve NPC Understanding of Simulation Mechanics

## Problem Statement

NPCs are fixating on controller state indicators like "waiting" and misinterpreting them as problems that need to be solved, rather than normal state transitions in the simulation.

**Example from logs**:
```
Situation: I am currently stalled at position (1, 0). The movement controller is reporting a 'waiting' state...
Active goals: ["Resolve the 'waiting' controller state immediately to enable movement."]
```

The NPC repeatedly chooses `wait` action trying to "resolve" the waiting state, getting stuck in a loop of waiting to resolve being in a waiting state.

## Root Cause

The LLM lacks context about how the Godot simulation works:
- What controller states mean
- What's a normal state vs an error state
- How movement and actions flow through the system
- When to trust the available actions vs overthink the state

## Solution: Add Simulation Mechanics to Action Selection Prompt

### Key Information to Add

**Controller States (Normal Behavior)**:
- `idle` - Standing still, ready for new action
- `waiting` - Awaiting response from mind server (THIS IS NORMAL - don't try to "fix" it)
- `moving` - Currently executing movement
- `bidding` - Interaction bid in progress
- `interacting` - Currently performing interaction

**Movement System**:
- Movement is NOT locked by default
- `movement_locked` flag specifically indicates when movement is blocked
- Controller states are informational, not problems to solve

**Action Selection**:
- Choose from `available_actions` list
- Trust that if an action is available, it's executable
- Don't second-guess or try to "resolve" normal controller states

### Proposed Prompt Addition

Add a new "## Simulation Mechanics" section to `action_selection/prompt.md`:

```markdown
## Simulation Mechanics

**Controller States** (informational only - NOT problems to solve):
- `idle`: Standing still, ready for action
- `waiting`: Awaiting mind server response (NORMAL during decision-making)
- `moving`: Executing movement command
- `bidding`: Interaction bid in progress
- `interacting`: Performing interaction

**Important**:
- Controller states are STATUS INDICATORS, not problems requiring action
- If you're seeing "waiting" state, that's because you're currently making a decision - it's normal!
- Choose actions based on goals and available actions, NOT on "fixing" controller states
- Movement is only blocked when `movement_locked: True` is explicitly shown

**Action Selection**:
- Choose from the available actions list
- Trust that available actions are executable
- Focus on NPC's goals and needs, not on controller state indicators
```

### Alternative: Add to Working Memory Guidelines

Instead of in prompt, could add to `cognitive_update/prompt.md` guidance on working memory's "situation assessment":

```markdown
When updating situation assessment:
- Focus on NPC position, visible entities, needs, and goals
- Controller state (idle/waiting/moving) is informational only
- Don't describe controller states as problems unless movement_locked is true
```

## Files to Change

**Option 1: Action Selection Prompt** (Recommended)
- `/home/hearn/projects/npc/mind/src/mind/cognitive_architecture/nodes/action_selection/prompt.md`
- Add "Simulation Mechanics" section before "## Task"

**Option 2: Cognitive Update Prompt**
- `/home/hearn/projects/npc/mind/src/mind/cognitive_architecture/nodes/cognitive_update/prompt.md`
- Add guidance in working memory update instructions

**Option 3: Both** (Belt and suspenders)
- Add mechanics overview in action selection
- Add "don't fixate on states" guidance in cognitive update

## Expected Outcome

- NPCs stop trying to "resolve" or "investigate" normal controller states
- Working memory situation assessments focus on position, entities, needs
- Action selection based on goals rather than state indicators
- No more wait loops trying to fix being in a waiting state

## Success Criteria

- No more logs showing "Active goals: ['Resolve the waiting controller state']"
- NPCs choose goal-directed actions (move_to, interact_with) instead of wait
- Situation assessments describe position and environment, not controller debugging

## Priority

**High** - This is causing NPCs to get stuck in unproductive loops, blocking gameplay testing.
