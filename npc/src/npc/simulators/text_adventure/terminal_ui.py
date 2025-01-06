from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import IntPrompt

from .story_engine import TextAdventureSimulator

console = Console()

def print_wrapped_text(text: str):
    """Print text with Markdown formatting"""
    console.print(Markdown(text))

def run_interactive_story(simulator: TextAdventureSimulator):
    """Run an interactive story session with human input"""
    console.print(Panel(Markdown(simulator.story_guide), title="Story Guide"))

    while not simulator.is_story_ended():
        state = simulator.state

        # Display the current game state
        console.rule("Game State", style="cyan")
        print_wrapped_text(state.observation)

        # Display available actions
        action_text = "\n\n".join([f"**Option {i}**  \n{action}" for i, action in state.available_actions.items()])
        actions_panel = Panel(Markdown(action_text), title="[bold cyan]Available Actions[/bold cyan]", border_style="bright_blue")
        console.print(actions_panel)

        # Take an action
        action_index = IntPrompt.ask("Choose an action", choices=[str(i) for i in state.available_actions.keys()])
        console.print(f"\nYou chose: [bold green]Option {action_index}[/bold green]\n")
        simulator.take_action(action_index)

def print_story_nodes(simulator: TextAdventureSimulator):
    """Print the complete story path from root to current node"""
    nodes = simulator.current_node.path_from_root()
    for i, node in enumerate(nodes):
        heading = f"Node {i + 1}"
        llm_output = node.llm_response
        print("\n".join([
            f"{heading}",
            "=" * len(heading),
            llm_output["response"],
            "",
        ]))
