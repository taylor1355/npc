import textwrap
from npc.agent.agent import Agent, LLMConfig
from npc.interfaces.text_adventure_interface import TextAdventureInterface, TextAdventureResponse
from .story_engine import TextAdventureSimulator

def run_agent_playthrough(
    simulator: TextAdventureSimulator, 
    config: LLMConfig, 
    max_steps: int = 3
):
    """Run an automated story playthrough with an AI agent"""
    simulator_interface = TextAdventureInterface(simulator)
    agent = Agent(simulator_interface=simulator_interface, llm_config=config)

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
        agent.update_state(simulator_response)
        simulator_request = agent.choose_action()
        simulator_response = simulator_interface.execute(simulator_request)
        
        if not simulator_response.success:
            print(f"Error: {simulator_response.message}")
            break
        
        if simulator.is_story_ended():
            print("Story has reached its conclusion.")
            break

    print("Playthrough completed!")
