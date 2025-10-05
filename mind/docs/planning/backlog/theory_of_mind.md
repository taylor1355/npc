# Theory of Mind

## Problem Statement

LLMs with memory already exhibit some theory of mind emergently - they can reason about what others know and feel. However, this emergent behavior is inconsistent, token-expensive, and not grounded in cognitive science. Structured tracking would make this capability reliable and efficient.

## Design Goals

- Structure the theory of mind that already emerges from LLMs
- Reduce token usage by maintaining explicit state
- Ensure consistent perspective-taking across decisions
- Ground the implementation in cognitive science principles

## Technical Approach

### Mental Models

Instead of relying on LLM emergence, explicitly track:
- What information has been shared with each NPC
- Observed goals and patterns
- Emotional state indicators
- Knowledge boundaries

This structured state gets injected into context, reducing the tokens needed for the LLM to reconstruct this understanding.

### Information Tracking

- Mark memories with "known_by" tags
- Track information flow through conversations
- Distinguish public knowledge from private knowledge
- Provide this as structured context rather than hoping LLM infers it

### Perspective Taking

Rather than asking the LLM to figure out what others know:
- Inject structured knowledge state into prompts
- Track information disclosure explicitly
- Reduce ambiguity in social reasoning

## Benefits

- **Token Efficiency**: Don't re-derive mental models each decision
- **Consistency**: Structured state ensures reliable behavior
- **Cognitive Grounding**: Based on how humans actually model others
- **Debuggability**: Can inspect what NPCs "think" others know

## Dependencies

- Social memory system
- Conversation tracking
- Structured state management

## Priority Rationale

**Obviousness**: Low - Many ways to structure what LLMs do emergently.

**Development Velocity**: Mixed
- Short-term: Adding structured state management is complex
- Long-term: Reduces token usage for all social interactions
- Tech tree: Makes social features more reliable and debuggable
- Net: Neutral - trades implementation complexity for operational efficiency

**Concreteness**: Moderate - Players see more consistent social behavior and NPCs that reliably remember who knows what.

This structures emergent capabilities for efficiency and reliability.