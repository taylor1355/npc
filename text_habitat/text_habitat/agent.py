from .openai_api import get_llm_completion
from .state import State

class Agent:
    ID_KEY = "name"
    REQUIRED_KEYS = [ID_KEY, "description", "status", "location", "room"]
    IMMUTABLE_KEYS = [ID_KEY]

    def __init__(self, state_dict):
        self.name = state_dict[Agent.ID_KEY]
        self.state = State(state_dict, self.REQUIRED_KEYS, self.IMMUTABLE_KEYS)

    @staticmethod
    def load_agents(agent_dicts):
        return {agent_dict[Agent.ID_KEY]: Agent(agent_dict) for agent_dict in agent_dicts}

    def room_id(self):
        return self.state["room"]

    def decide_action(self, room_description):
        action_str = "Walk into the room." # TODO: add action history to be able to fill this out and give a summary of past actions
        prompt = "\n".join([
            "You find yourself in a room in your house inside <room></room> tags. Your state is described below in the <you></you> tags.",
            "",
            "Considering your surroundings and how you feel right now, what do you decide to do next? You can do anything within reason (based on your capabilities, the state of the room, and the laws of physics). However, there are a few guidelines you must follow:",
            "- The action you take shouldn't be too small or too large in scope. For example, 'open the refrigerator' is too small, and 'cook dinner' is too large.",
            "- A good rule of thumb for action scope is that it should take at least 1 minute to complete. Longer actions are fine as long as they are simple or repetitive, like 'go to sleep' or 'play a game'.",
            "- Another good rule of thumb is that the action should be plausible as a single action in the Sims (with all imaginable expansion packs and additional content enabled).",
            "- Be brief and to the point when describing your action.",
            "",
            "The room's state is described below in the <room></room> tags. Note that the ONLY objects in the room are those described below. You can't assume that there are any other objects in the room.",
            "<room>",
           f"{room_description}",
            "</room>",
            "",
            "Your state is described below in the <you></you> tags:",
            "<you>",
           f"{self.state}",
            "</you>",
            "",
            "The last thing you did, as documented by an external observer is described below in <action></action> tags:",
            "<action>",
           f"{action_str}",
            "</action>",
        ])
        system_prompt = f"You are '{self.name}', an intelligent being in this virtual world."
        return get_llm_completion(prompt, system_prompt)

    def update(self, new_state_dict):
        self.state.update(new_state_dict)