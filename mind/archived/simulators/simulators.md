# Simulators

## Overview

Simulators provide environments where NPC agents can operate and make decisions. The system defines a base interface that all simulators must implement, ensuring consistent integration with the agent architecture. Currently includes a text adventure implementation as a standalone demo.

## Architecture

```
Simulators
├── Base Interface (base_interface.py)
│   └── BaseSimulator - Abstract protocol
├── Text Adventure (text_adventure.py)
│   ├── Story Generation - LLM-powered narratives
│   ├── Action System - Player choices
│   └── Outcome Engine - Probabilistic results
└── Text Adventure Components
    ├── StoryEngine - Core narrative logic
    ├── AgentGameplay - NPC integration
    └── TerminalUI - Interactive interface
```

## Base Interface

### BaseSimulator (`interfaces/base_interface.py`)

All simulators must implement this abstract base class:

**Core Methods:**
```python
class BaseSimulator(ABC):
    @abstractmethod
    def get_observation(self) -> str:
        """Return current state as natural language"""
        
    @abstractmethod  
    def get_available_actions(self) -> Dict[int, str]:
        """Return indexed action descriptions"""
        
    @abstractmethod
    def execute_action(self, action_index: int) -> str:
        """Execute action and return result"""
        
    @abstractmethod
    def is_complete(self) -> bool:
        """Check if simulation has ended"""
```

**Integration Pattern:**
```python
simulator = ConcreteSimulator()
agent = Agent(...)

while not simulator.is_complete():
    observation = simulator.get_observation()
    actions = simulator.get_available_actions()
    
    choice = agent.process_observation(observation, actions)
    result = simulator.execute_action(choice)
```

## Text Adventure Simulator

### Overview

The text adventure simulator (`simulators/text_adventure.py`) provides a narrative-driven environment where agents navigate branching storylines. It demonstrates agent decision-making in a creative, non-deterministic setting.

### Core Components

**TextAdventureSimulator Class:**
- Manages story progression
- Tracks narrative state
- Interfaces with story engine

**Key Features:**
- LLM-generated story content
- Probabilistic action outcomes
- Branching narrative paths
- Success/failure tracking

### Story Engine (`text_adventure/story_engine.py`)

Handles narrative generation and progression:

**Story Structure:**
```python
@dataclass
class StoryNode:
    content: str          # Narrative text
    actions: List[str]    # Available choices
    outcomes: Dict[str, List[Outcome]]  # Possible results
```

**Outcome System:**
- Each action has multiple possible outcomes
- Outcomes have success probabilities
- Failed outcomes can cascade effects
- Story adapts to outcome history

### Agent Integration (`text_adventure/agent_gameplay.py`)

Connects NPC agents to the story environment:

**Gameplay Loop:**
1. Present story section to agent
2. Agent observes narrative and choices
3. Agent selects action based on personality
4. Engine determines outcome
5. Story progresses based on result

**Agent Considerations:**
- Personality traits influence choices
- Memory affects decision context
- Past outcomes inform strategy

### Terminal UI (`text_adventure/terminal_ui.py`)

Provides interactive interface for testing:

**Features:**
- Rich terminal formatting
- Story section display
- Action selection menus
- Outcome visualization
- Agent decision explanations

## Creating New Simulators

### Implementation Steps

1. **Extend BaseSimulator:**
```python
class MySimulator(BaseSimulator):
    def __init__(self):
        self.state = initial_state
        
    def get_observation(self) -> str:
        return format_state_as_text(self.state)
        
    def get_available_actions(self) -> Dict[int, str]:
        return enumerate_valid_actions(self.state)
```

2. **Define State Management:**
- Track simulation state
- Validate action preconditions
- Update state on actions
- Determine completion conditions

3. **Format Observations:**
- Convert state to natural language
- Include relevant context
- Describe available interactions
- Maintain consistency

4. **Handle Actions:**
- Validate action indices
- Apply state changes
- Generate result descriptions
- Check for side effects

### Integration Considerations

**For Agent Compatibility:**
- Observations should be descriptive text
- Actions need clear descriptions
- Include personality-relevant details
- Provide meaningful choices

**For MCP Server:**
- Simulators run independently
- Agents access via standard interface
- State persists between decisions
- Support concurrent simulations

## Usage Examples

### Standalone Testing

```python
# Run text adventure with agent
from npc.simulators.text_adventure import TextAdventureSimulator
from npc.agent.agent import Agent

simulator = TextAdventureSimulator(story_concept="Space exploration")
agent = Agent(personality_traits=["brave", "curious"])

# Run simulation
while not simulator.is_complete():
    obs = simulator.get_observation()
    actions = simulator.get_available_actions()
    choice = agent.process_observation(obs, actions)
    simulator.execute_action(choice)
```

### Interactive Mode

```bash
# Play text adventure interactively
poetry run python -m npc.simulators.text_adventure

# With specific story concept
poetry run python -m npc.simulators.text_adventure --concept "Medieval quest"
```

## Future Directions

Potential simulator types:
- **Social Simulation**: Multi-agent conversations
- **Task Environment**: Goal-oriented scenarios  
- **Game Integration**: Direct Godot connection
- **Training Grounds**: Agent skill development