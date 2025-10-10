import textwrap
from mind.agent.agent import Agent, AgentLLMConfig
from mind.interfaces.text_adventure_interface import TextAdventureInterface, TextAdventureResponse, TextAdventureRequest
from .story_engine import TextAdventureSimulator

def run_agent_playthrough(
    simulator: TextAdventureSimulator, 
    llm_config: AgentLLMConfig, 
    max_steps: int = 3,
    personality_traits: list[str] = [],
    initial_working_memory: str = ""
):
    """Run an automated story playthrough with an AI agent"""
    simulator_interface = TextAdventureInterface(simulator)
    agent = Agent(
        llm_config=llm_config,
        personality_traits=personality_traits,
        initial_working_memory=initial_working_memory
    )

    simulator_response = TextAdventureResponse(
        success=True,
        message="Initial state",
        observation=simulator.state.observation,
        available_actions=simulator.state.available_actions,
    )

    for _ in range(max_steps):
        print(200 * "=")
        print("\n".join([
            "Observation:",
            textwrap.fill(simulator_response.observation, width=120),
            "",
        ]))
        
        # Process observation and choose action using updated agent interface
        action_index = agent.process_observation(
            observation=simulator_response.observation,
            available_actions=simulator_response.available_actions
        )
        
        # Create request with chosen action
        simulator_request = TextAdventureRequest(action_index=action_index)
        
        # Execute action in simulator
        simulator_response = simulator_interface.execute(simulator_request)
        
        if not simulator_response.success:
            print(f"Error: {simulator_response.message}")
            break
        
        if simulator.is_story_ended():
            print("Story has reached its conclusion.")
            break

    print("Playthrough completed!")
