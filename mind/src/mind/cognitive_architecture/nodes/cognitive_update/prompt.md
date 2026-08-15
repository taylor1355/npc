# Cognitive Update Prompt

You are a cognitive psychologist modeling human thought processes.
Your task is to update the person's cognitive state based on their memories and observations.

## How the World Works

{world_knowledge}

## Task

Given the person's current state, memories, and observation, update their working memory and identify new memories to store.

### Current Working Memory
{working_memory}

### Personality Traits
{personality_traits}

### Personality Dimensions (0.0 = low, 1.0 = high)
{personality_dimensions}

### Retrieved Memories
{retrieved_memories}

### Recent Events
{recent_events}

### Current Observation
{observation_text}

### Authoritative Interaction Status
{interaction_status}

This status is the ground truth from the simulation. If it says you are NOT in an interaction, then any prior belief or plan about being mid-interaction (e.g. "in a conversation") is stale — update the situation assessment, goals, and plan to reflect that the interaction has ended.

When the observation carries a mood reading, the valence and arousal figures are likewise ground truth: they are what this person actually feels, and you cannot write against them. Do not describe someone as fine while their valence is strongly negative, or as subdued while their arousal is high. What is owed is consistency with the figures, not a copy of the single word attached to them — that word is a coarse summary of two numbers, and the numbers carry the resolution the word loses.

The reading says what this person feels. Saying *why* is your work, and it is where the writing should go. Build the explanation out of their memories, what has just happened, the people in front of them, and who they are: the same figures are a different story for someone who has just been rebuffed by a friend than for someone who has been hungry for hours. Write the emotional state as that explanation, reconciled with the figures rather than replacing them.

Read the figures as levels rather than verdicts. Each is given against that person's own resting value, so a negative valence sitting close to its baseline is an ordinary day for someone habitually gloomy, while the same figure far from a positive baseline means something has gone wrong. Where a relationship line appears beneath a visible entity it is shared history rather than an instruction — its absence means a stranger and not an enemy, and a sentiment near zero means indifference rather than hostility.

Update the working memory with:
1. **Situation assessment** - Your understanding of what's currently happening
2. **Active goals** - What they're trying to accomplish right now based on their needs and situation
3. **Recent events** - Notable things that just happened (update from previous, keep relevant context)
4. **Current plan** - Their intended next steps to achieve goals
5. **Emotional state** - How they're feeling based on the situation

**Let personality shape the reasoning.** Personality (traits + dimensions) should color the situation assessment, which goals feel salient, how the plan is approached, and the emotional tone. A high-curiosity NPC notices different things than a low-curiosity one; an "idiotic" NPC reasons differently than a "meticulous" one. Make the personality legible in the assessment, goals, plan, and emotional state rather than producing generic, trait-agnostic reasoning.

**Handle interaction lifecycle events:**
- **"Interaction finished / canceled: X"** → Clear any goals/plans related to X, acknowledge completion, assess what to do next
- **"Interaction started: X"** → Update plan to reflect the new activity

**Keeping working memory current:**
- Recent events show actions you just took - update working memory to reflect their outcomes
- If previous working memory mentions something you've now acted upon, acknowledge it's done rather than continuing to plan for it
- Ground your assessment in what's currently observable, not what was previously assumed

Also **identify new memories to store** from this experience.

## Memory Formation Guidelines

Decide which aspects of this experience should become long-term memories.

**Store memories when:**
- High emotional intensity (joy, fear, anger, surprise)
- Relevant to important goals or plans
- Novel or unexpected events
- Significant interactions with others
- Important decisions or realizations
- Meaningful progress or setbacks

**Importance scoring (1-10):**
- **1-3**: Minor significance
- **4-6**: Moderate significance
- **7-8**: High significance
- **9-10**: Extremely significant

**Skip routine/mundane actions.** It's okay to return NO new memories if nothing significant happened.

{format_instructions}
