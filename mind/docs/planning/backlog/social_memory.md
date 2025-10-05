# Social Memory and Relationships

## Problem Statement

NPCs treat every interaction as if meeting for the first time. They don't remember past conversations, shared experiences, or relationship history. This makes social interactions feel hollow and prevents meaningful relationship development.

## Design Goals

- Track interaction history with each NPC
- Remember conversation topics and emotional context
- Build relationship models over time
- Enable reference to shared experiences

## Technical Approach

### Relationship Records

Per-NPC tracking:
- Interaction count and recency
- Emotional valence of interactions
- Topics discussed
- Shared experiences
- Trust/friendship levels

### Memory Integration

Social memories get special handling:
- Higher retrieval weight for memories involving current conversation partner
- Automatic tagging with participant IDs
- Cross-referencing between NPCs' memories of same event

### Conversation Context

When starting conversations:
- Retrieve recent interactions with this NPC
- Include relationship status in context
- Reference shared history naturally

## Benefits

- **Relationship Depth**: NPCs build meaningful connections
- **Conversation Continuity**: Can reference past discussions
- **Social Dynamics**: Friendships and rivalries emerge
- **Player Investment**: Relationships feel real and worth cultivating

## Dependencies

- Basic memory system
- Conversation system
- NPC identification/registry

## Priority Rationale

**Obviousness**: High - Clearly needed for any social depth.

**Development Velocity**: Slightly positive
- Short-term: Adds relationship tracking and memory tagging
- Tech tree: Enables group dynamics, social influence, reputation systems
- Complexity: Must handle relationship state across multiple NPCs
- Net: Slightly positive - foundational for other social features

**Concreteness**: Very high - Players immediately notice when NPCs remember them, reference past conversations, and treat friends differently than strangers.

This transforms NPCs from strangers to potential friends.