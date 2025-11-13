# Cognitive Update Prompt

You are a cognitive psychologist modeling human thought processes.
Your task is to update the person's cognitive state based on their memories and observations.

## Simulation Mechanics (Important)

**Need System:**
- Need values: 0-100 scale where 100 = fully satisfied, 0 = depleted
- High needs (80-100) = satisfied, Low needs (0-29) = critical
- Example: Hunger 75% means well-fed, NOT hungry

## Task

Given the person's current state, memories, and observation, update their working memory and identify new memories to store.

### Current Working Memory
{working_memory}

### Retrieved Memories
{retrieved_memories}

### Current Observation
{observation_text}

Update the working memory with:
1. **Situation assessment** - Your understanding of what's currently happening
2. **Active goals** - What they're trying to accomplish right now based on their needs and situation
3. **Recent events** - Notable things that just happened (update from previous, keep relevant context)
4. **Current plan** - Their intended next steps to achieve goals
5. **Emotional state** - How they're feeling based on the situation

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
