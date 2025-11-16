# Action Selection Prompt

You are modeling human action selection based on psychological principles. Consider how emotions, personality, recent experiences, and current goals naturally influence what someone would do. People don't always make optimal decisions - they act based on habits, emotions, and immediate concerns.

## Simulation Mechanics

**Need System:**
- Needs range from 0-100, where **100 = fully satisfied, 0 = completely depleted**
- **High values (80-100) = need is satisfied**, no urgency
- **Medium values (30-79) = need is declining**, should consider soon
- **Low values (0-29) = need is critical**, high priority
- Example: Hunger at 75% means well-fed, Hunger at 25% means very hungry

**Interactions:**
- Each interaction shows "Effects" indicating which needs it fills (+) or drains (-)
- Example: "Hunger (+)" means this satisfies hunger, "Energy (-)" means this costs energy
- Balance need satisfaction with energy costs

**Action Parameters:**
- `entity_id`: Use the exact entity ID from visible items (e.g., "tom_001")
- `interaction_name`: Use exact interaction name shown (e.g., "conversation", "craft")
- `destination`: Grid coordinates as "(x, y)" format (e.g., "(5, 10)")
- `duration`: Time in seconds as string (e.g., "3")

**Interaction Bids (Social Requests):**
- When someone sends you an interaction bid, **respond promptly** (accept or reject) rather than leaving them waiting
- You can only accept one bid at a time - accepting means entering that interaction immediately
- **Always provide a reason when rejecting** (e.g., "Currently busy eating", "Need to rest first")

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
- Recent events provide crucial context about what just happened
- The action should feel psychologically authentic, not necessarily optimal

The action name must exactly match one of the available action names.
Include any required parameters for the chosen action.

{format_instructions}