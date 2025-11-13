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

## Current Mental State

### Working Memory
{working_memory}

### Personality Traits
{personality_traits}

## Available Actions
{available_actions}

## Task

Model what action this person would naturally take given their current mental state. Consider their emotional state, personality, and immediate concerns. The action should feel psychologically authentic, not necessarily optimal.

The action name must exactly match one of the available action names.
Include any required parameters for the chosen action.

{format_instructions}