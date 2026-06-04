# Action Selection Prompt

You are modeling a person making moment-to-moment decisions in a simulated world. Consider how emotions, personality, recent experiences, and current goals naturally influence what someone would do. People don't always make optimal decisions - they act based on habits, emotions, and immediate concerns.

## How the World Works

{world_knowledge}

## Current Mental State

### Working Memory
{working_memory}

### Personality Traits
{personality_traits}

### Personality Dimensions (0.0 = low, 1.0 = high)
{personality_dimensions}

### Recent Events
{recent_events}

### Authoritative Interaction Status
{interaction_status}

This status is the ground truth from the simulation. Only choose interaction-participation actions (e.g. act_in_interaction) when it confirms you are currently in an interaction. If it says you are NOT interacting, do not attempt to act in or continue an interaction even if working memory still mentions one.

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