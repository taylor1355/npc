"""MCP Prompt definitions for NPC decision making"""


def build_decision_prompt(
    agent_mental_state: dict,
    physical_state: dict
) -> str:
    """Build a unified decision prompt from mental and physical state.
    
    Args:
        agent_mental_state: Mental state from agent resource containing:
            - personality: traits
            - memory: working, long_term_recent
            - cognitive_state: last observations
        physical_state: Physical state from simulation containing:
            - observation: Current world state
            - available_actions: List of possible actions
            - position: Current location
            - state: Current activity state
    
    Returns:
        Formatted prompt string for LLM decision making
    """
    # Extract mental state components
    personality = agent_mental_state.get('personality', {})
    memory = agent_mental_state.get('memory', {})
    
    traits = personality.get('traits', [])
    working_memory = memory.get('working', '')
    recent_memories = memory.get('long_term_recent', [])
    
    # Extract physical state components
    observation = physical_state.get('observation', '')
    available_actions = physical_state.get('available_actions', [])
    
    # Format personality traits
    traits_text = ', '.join(traits) if traits else 'No defined traits'
    
    # Format recent memories
    if recent_memories:
        memory_text = '\n'.join(f"- {mem}" for mem in recent_memories[-5:])
    else:
        memory_text = "No recent memories."
    
    # Format available actions
    if isinstance(available_actions, dict):
        # Current format: dict with index -> description
        action_text = '\n'.join([
            f"- Action {idx}: {desc}"
            for idx, desc in available_actions.items()
        ])
    else:
        # Future format: list of action objects
        action_text = '\n'.join([
            f"- Action {i}: {action.get('name', '')} - {action.get('description', '')}"
            for i, action in enumerate(available_actions)
        ])
    
    # Build the unified prompt
    prompt = f"""You are an NPC with these characteristics:
Personality traits: {traits_text}

Your mental state:
Working memory: {working_memory}

Recent memories:
{memory_text}

Current situation:
{observation}

Available actions:
{action_text}

Based on your personality, memories, and the current situation, decide your next action.

Respond with JSON:
{{
  "reasoning": "Brief explanation of your choice",
  "action_index": <number>,
  "memory_note": "Important detail to remember (if any)"
}}"""
    
    return prompt


def build_conversation_prompt(
    agent_mental_state: dict,
    conversation_context: dict
) -> str:
    """Build a prompt for conversation responses.
    
    Args:
        agent_mental_state: Mental state from agent resource
        conversation_context: Current conversation state containing:
            - participants: List of participant IDs
            - history: Recent message history
            - current_speaker: Who just spoke
            - current_message: What they said
    
    Returns:
        Formatted prompt for conversation response
    """
    # Extract mental state
    personality = agent_mental_state.get('personality', {})
    memory = agent_mental_state.get('memory', {})
    
    traits = personality.get('traits', [])
    working_memory = memory.get('working', '')
    
    # Extract conversation context
    history = conversation_context.get('history', [])
    current_message = conversation_context.get('current_message', '')
    current_speaker = conversation_context.get('current_speaker', 'Someone')
    
    # Format conversation history
    history_text = '\n'.join([
        f"{msg.get('speaker', 'Unknown')}: {msg.get('text', '')}"
        for msg in history[-5:]  # Last 5 messages
    ])
    
    traits_text = ', '.join(traits) if traits else 'No defined traits'
    
    prompt = f"""You are an NPC in a conversation with these traits: {traits_text}

Your current thoughts: {working_memory}

Recent conversation:
{history_text}

{current_speaker} just said: "{current_message}"

How do you respond? Keep your response brief and in character.

Respond with JSON:
{{
  "message": "Your response",
  "emotion": "current emotional state",
  "wants_to_continue": true/false
}}"""
    
    return prompt


def build_planning_prompt(
    agent_mental_state: dict,
    goal: str,
    constraints: dict = None
) -> str:
    """Build a prompt for multi-step planning.
    
    Args:
        agent_mental_state: Mental state from agent resource
        goal: What the agent wants to achieve
        constraints: Optional constraints like time, resources, etc.
    
    Returns:
        Formatted prompt for planning
    """
    # This is a placeholder for future planning capabilities
    # when we want NPCs to think multiple steps ahead
    
    personality = agent_mental_state.get('personality', {})
    memory = agent_mental_state.get('memory', {})
    
    traits = personality.get('traits', [])
    working_memory = memory.get('working', '')
    
    constraints_text = ""
    if constraints:
        constraints_text = f"\nConstraints: {constraints}"
    
    traits_text = ', '.join(traits) if traits else 'No defined traits'
    
    prompt = f"""You are an NPC with traits: {traits_text}

Current understanding: {working_memory}

Goal: {goal}{constraints_text}

Create a simple plan to achieve this goal. List 3-5 concrete steps.

Respond with JSON:
{{
  "plan": ["step 1", "step 2", ...],
  "expected_outcome": "what you expect to achieve",
  "fallback": "what to do if the plan fails"
}}"""
    
    return prompt