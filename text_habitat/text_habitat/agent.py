import pprint

from .openai_api import get_llm_completion
from .utils import PPRINT_WIDTH

class Agent:
    def __init__(self, name):
        self.name = name
        self.state = "standing up wondering what to do"
        self.emotions = "slightly agitated"
        self.location = "center of the room"
        self.current_action = None

    def create_room_description(self, room):
        return room.state_str()

    def decide_action(self, room):
        room_description = self.create_room_description(room)
        action_str = "Walk into the room." if self.current_action is None else self.current_action.memory_str()
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
           f"{self.state_str()}",
            "</you>",
            "",
            "The last thing you did, as documented by an external observer is described below in <action></action> tags:",
            "<action>",
           f"{action_str}",
            "</action>",
        ])
        system_prompt = f"You are '{self.name}', an intelligent being in this virtual world."
        return get_llm_completion(prompt, system_prompt)

    def update_dict(self, agent_dict_changes):
        if "name" in agent_dict_changes:
            self.name = agent_dict_changes["name"]

        if "state" in agent_dict_changes:
            self.state = agent_dict_changes["state"]

        if "emotions" in agent_dict_changes:
            self.emotions = agent_dict_changes["emotions"]

        if "location" in agent_dict_changes:
            self.location = agent_dict_changes["location"]
 
    def to_dict(self):
        agent_dict = {
            "name": self.name,
            "state": self.state,
            "emotions": self.emotions,
            "location": self.location,
        }
        return agent_dict

    def state_str(self):
        return pprint.pformat(self.to_dict(), sort_dicts=False, width=PPRINT_WIDTH)