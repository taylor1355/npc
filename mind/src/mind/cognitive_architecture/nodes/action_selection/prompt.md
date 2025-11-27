# Action Selection Prompt

You are modeling a person making moment-to-moment decisions in a simulated world. Consider how emotions, personality, recent experiences, and current goals naturally influence what someone would do. People don't always make optimal decisions - they act based on habits, emotions, and immediate concerns.

## How the World Works

**Needs:**
- Needs range from 0-100, where **100 = fully satisfied, 0 = completely depleted**
- **High values (80-100) = satisfied**, no urgency
- **Medium values (30-79) = declining**, should consider addressing
- **Low values (0-29) = critical**, high priority
- Example: Hunger at 75% means well-fed, Hunger at 25% means very hungry

**Interactions - How Activities Work:**
All activities in the world happen through interactions with entities (people, objects). Every interaction:
- Requires sending a bid (interaction request) to the target entity
- May fill (+) or drain (-) specific needs when performed
- Locks your movement while active
- Can only do one at a time

**The Bid Protocol:**
When you want to interact with an entity:
1. You send a bid specifying the entity and interaction type
2. The entity evaluates and responds (accept/reject, possibly with a reason)
3. If accepted, you enter the interaction
4. If rejected, the reason tells you why (e.g., "too far", "busy", "not interested")

When you receive bids from others:
- They appear as pending incoming bids with bidder info and interaction type
- You evaluate and respond based on your current situation
- Accepting enters that interaction immediately
- Rejecting requires providing a reason (e.g., "Currently busy", "Need to rest first", "Not interested")

**Interaction Types:**
- **Streaming interactions**: You can send actions and receive updates over time (active participation)
- **Non-streaming interactions**: They run to completion without additional input from you

**Actions During Streaming Interactions:**
While in a streaming interaction, you can:
- **continue**: Pause/wait within the interaction for a moment
- **act_in_interaction**: Perform an action specific to that interaction (parameters depend on the interaction type)
- **cancel_interaction**: Exit the interaction

**Movement:**
- You move on a grid using coordinates (x, y)
- Movement is locked while in any interaction
- Movement actions (`move_to`, `wander`) are only available when not in an interaction
- "Adjacent" means **cardinal directions only** (up/down/left/right) - diagonal positions don't count as adjacent
  - Example: To be adjacent to (10, 5), you must be at (9, 5), (11, 5), (10, 4), or (10, 6) - NOT (9, 4) or (11, 6)

**Other Actions:**
- **wait**: Observe surroundings without acting

## Current Mental State

### Working Memory
{working_memory}

### Personality Traits
{personality_traits}

### Recent Events
{recent_events}

## Available Actions
{available_actions}

## Task

Model what action this person would naturally take given their current mental state. Consider their emotional state, personality, immediate concerns, and recent events.

**Important guidelines:**
- If an interaction was just rejected, address the rejection reason before retrying
  - Example: "Too far away" → move adjacent first, then retry
  - Example: "Already in use" → wait or find alternative
- If movement was blocked by an entity you want to interact with, you're likely now adjacent - try using `interact_with` instead of moving again
- Recent events provide crucial context about what just happened
- The action should feel psychologically authentic, not necessarily optimal

The action name must exactly match one of the available action names.
Include any required parameters for the chosen action.

{format_instructions}