# Documentation Style Guide

## Core Principles

1. Every word must be useful
   Documentation should be concise and purposeful, focusing on what developers need to know to effectively use and maintain the system. Remove any fluff or redundant explanations that don't add value.

2. Use diagrams for structure
   Visual representations help developers quickly understand system organization and relationships. Use ASCII diagrams to show hierarchies and connections.
   ```
   System
   ├── Component A (file.py) - purpose
   ├── Component B (file.py) - purpose
   └── Component C (file.py) - purpose
   ```

3. Emphasize Purpose and Responsibilities
   Documentation should clearly explain *what* a component or system does, its primary responsibilities, and *how* it integrates with other parts of the application. Focus on the 'why' and 'what' over minute implementation details. While key configurations, parameters, or method signatures critical for usage should be included, avoid exhaustive lists of all internal properties or private methods.

4. Include critical details
   Document important constants, configurations, and public interfaces (key methods, class attributes, parameters) that affect system behavior or are necessary for integration. Reference source files to help developers locate implementations.

5. Prioritize Clarity and Readability
   Strive for descriptive explanations that help a developer understand a system's role and behavior quickly. Use narrative descriptions, often supplemented by bullet points for key responsibilities or features, rather than just listing technical specifications. The goal is to provide enough detail for comprehension without overwhelming the reader. Aim for strategically concise language: be descriptive enough to convey necessary information and context, but avoid unnecessary verbosity that could increase cognitive load. If a shorter phrasing conveys the same meaning effectively, prefer it. However, do not sacrifice critical context or clarity for the sake of extreme brevity. The key is to find a balance that makes the documentation both informative and easy to digest.
   
   **Balance Readability with Skimmability**: Write complete, understandable sentences while maintaining a structure that supports quick scanning:
   - **Use bullet points** for lists of features, responsibilities, or options
   - **Keep paragraphs short** (2-3 sentences) to maintain visual breaks
   - **Bold key terms** to help readers quickly locate information
   - **Avoid walls of text** by breaking up content with headers and lists
   
   Examples of the balance:
   - ✅ Good bullet: "Manages both single-party and multi-party interactions through a type property"
   - ❌ Poor bullet: "Manages single/multi-party via type property" (too telegraphic)
   - ✅ Good paragraph: "The interaction system uses a bid-based pattern. NPCs request interactions through bids, and targets evaluate whether to accept based on their current state."
   - ❌ Poor paragraph: One long paragraph explaining the entire bid system without breaks

## Document Structure

1. Overview
   Start with a clear explanation of the system's purpose and how it fits into the larger application. List core components and their roles to give readers immediate context.

2. Components
   For each major component, explain its purpose, responsibilities, and how it interacts with other parts of the system. Focus on the public interface rather than internal details.

3. Setup Requirements
   Provide clear instructions for adding the system to a scene. Include required nodes, configuration values, and common usage patterns to help developers get started quickly.

## Common Mistakes

1. Code Dumps or Overly Technical Listings
   - ❌ Copying large blocks of implementation code, or exhaustively listing all internal members (private methods, internal variables) without context.
   - ✅ Describing the component's overall purpose, its key public interfaces (APIs, signals, critical exported properties), and how they are intended to be used. Focus on what a consumer of the component needs to know.

2. Abstraction Level
   - ❌ Too high: "Manages state changes" (Too vague).
   - ✅ Specific: "Tracks entity position based on grid coordinates and handles collisions with static obstacles defined in the tilemap's physics layer." (Clear and informative).

3. Missing Context
   - ❌ Listing properties or methods without explaining their purpose or typical usage (e.g., "Property: `max_speed: float`").
   - ✅ Explaining why a property exists and how it influences behavior (e.g., "`max_speed: float` - Defines the maximum velocity the entity can reach. Used by the movement system to cap acceleration.").

4. Redundant Information
   - ❌ Repeating obvious details that can be inferred or are standard (e.g., "This function is a function.").
   - ✅ Focusing on non-obvious requirements, important interactions, or critical configurations.

5. Dry Specification Lists
   - ❌ Presenting information as a dry list of technical specifications (e.g., just method signatures or property names) without descriptive text explaining their role or usage.
   - ✅ Providing narrative explanations for how a component works and how its key features are used, supplemented by technical details where necessary, to aid understanding.

## Examples

### Good Component Description

**MemoryDatabase (`memory_database.py`)**

The `MemoryDatabase` provides semantic memory storage and retrieval for NPCs using ChromaDB as a vector store. It enables NPCs to store experiences and retrieve relevant memories based on semantic similarity.

**Key Responsibilities:**
*   **Memory Storage:**
    *   Provides an `add_memory(memory_text: str, metadata: dict = None)` method to store new memories with optional metadata.
    *   Automatically generates embeddings for semantic search.
*   **Memory Retrieval:**
    *   `query_memories(query_text: str, n_results: int = 5)` - Retrieves semantically similar memories.
    *   Returns memories with relevance scores and metadata.
*   **Configuration:**
    *   `collection_name: str` - ChromaDB collection for the agent's memories
    *   `embedding_model: str = "all-MiniLM-L6-v2"` - Model used for generating embeddings
*   **Integration:**
    *   Used by the Agent class to maintain long-term memory
    *   Provides context for decision-making based on past experiences

**Usage Example:**
```python
# Initialize database for an agent
memory_db = MemoryDatabase(agent_id="npc_123")

# Store a memory
memory_db.add_memory(
    "Found food at the market stall",
    metadata={"location": "market", "time": "morning"}
)

# Query relevant memories
relevant_memories = memory_db.query_memories("Where can I find food?")
```

### Good Setup Description
```
Required Dependencies:
- ChromaDB for vector storage
- OpenAI/OpenRouter for LLM integration
- MCP for network communication
- Pydantic for data validation

Environment Variables:
- OPENAI_API_KEY: LLM provider key
- OPENROUTER_API_KEY: Alternative LLM provider
- NPC_DATA_DIR: Data storage location
```

### Good Integration Description
```
Data Flow:
1. Simulator sends observation
2. Agent processes with memory context
3. LLM generates decision
4. Action returned to simulator
```

## Additional Guidelines

1. Visual Hierarchy
   Use consistent heading levels and formatting to organize content logically. Keep sections focused and atomic, ordering from most to least important.

2. Technical Accuracy
   Verify all documented behaviors and configurations. **Always consult the relevant source code files when documenting technical details (e.g., class members, function signatures, configuration options) to ensure the documentation accurately reflects the implementation.** Test setup instructions to ensure they work as described. Keep documentation synchronized with code changes.

3. Cross-References
   Link related systems and note dependencies clearly. Show integration examples that demonstrate how components work together in practice.

4. Maintenance
   Regularly review and update documentation as the codebase evolves. Remove outdated information and clarify sections that users find confusing.
