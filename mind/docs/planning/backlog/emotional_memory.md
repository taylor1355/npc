# Emotional Memory

## Problem Statement

Memories are stored with flat importance scores, but emotions affect memory formation and retrieval in humans. Strong emotional experiences are remembered more vividly and retrieved more readily. NPCs need this emotional coloring to have realistic memory patterns.

## Design Goals

- Attach emotional valence and intensity to memories
- Use emotion in retrieval scoring
- Enable mood-congruent recall
- Model how emotions fade over time

## Technical Approach

### Emotional Tagging

When memories form, LLM assigns:
- Valence (-1 to 1: negative to positive)
- Intensity (0 to 1: strength of emotion)
- Emotion type (joy, fear, anger, sadness, etc.)

This supplements the importance score with emotional dimensions.

### Retrieval Weighting

Memory retrieval considers:
- Emotional intensity increases retrieval probability
- Current mood makes mood-congruent memories more accessible
- Strong emotions persist longer than weak ones
- Traumatic memories (high negative intensity) have special retrieval patterns

### Emotional Decay

- Intensity naturally decreases over time
- Positive emotions fade faster than negative (negativity bias)
- Repeated retrieval can reinforce or modify emotional associations

## Benefits

- **Realistic Memory**: Matches human memory patterns
- **Emotional Continuity**: Past experiences influence current mood
- **Narrative Depth**: Traumatic or joyful events have lasting impact
- **Behavioral Influence**: Emotions guide decision-making

## Dependencies

- Basic memory system with scoring
- Emotion/mood system (if separate)
- Memory reflection for emotional processing

## Priority Rationale

**Obviousness**: Moderate - Clear that emotions affect memory, implementation has choices.

**Development Velocity**: Slightly negative
- Short-term: Adds another scoring dimension to manage
- Long-term: Doesn't unlock new features, adds depth to existing memory
- Complexity: Balancing emotional and importance scores
- Net: Slightly negative - complexity without unlocking new capabilities

**Concreteness**: Moderate - Players notice when NPCs remember emotional events more clearly and when mood affects what they talk about.

Adds realism to memory but increases scoring complexity.