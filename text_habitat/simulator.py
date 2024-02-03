import config

from text_habitat.agent import Agent
from text_habitat.gamemaster import Gamemaster
from text_habitat.openai_api import authenticate_openai
from text_habitat.room import Room

class Simulator:
    def __init__(self, agent_dict, room_dicts):
        self.agent = Agent(agent_dict["name"], agent_dict)
        self.rooms = {room_dict["id"]: Room.from_dict(room_dict) for room_dict in room_dicts}
        self.gamemaster = Gamemaster()

        self.timestep = 0

    def execute_action(self, action):
        if action.end_time > self.timestep:
            action.acting_agent.current_action = action

        updated_room_state, updated_agent_state = action.execute(self)
        room = self.get_action_room(action)
        room.update_dict(updated_room_state)
        action.acting_agent.update(updated_agent_state)

        print(str(action))
        print()
        print(room.state_str())
        print(action.acting_agent.state_str())
        print()

    # TODO: come up with a better way of passing rooms around
    def get_agent_room(self, agent):
        return self.rooms[agent.state_dict["room"]]
    def get_action_room(self, action):
        return self.rooms[action.room]

    def run(self, num_timesteps):
        user_actions = []
        system_actions = []
        for _ in range(num_timesteps):
            if self.agent.current_action is None or self.agent.current_action.is_done(self.timestep):
                print(100 * "=")
                print(100 * "=")

                action_intent = self.agent.decide_action(self.get_agent_room(self.agent).state_str())
                user_actions.append(self.gamemaster.generate_user_action(action_intent, self))
                self.execute_action(user_actions[-1])

                # TODO: only do after every k user actions
                print(100 * "-")
                system_actions.append(self.gamemaster.generate_state_correction_action(self))
                self.execute_action(system_actions[-1])

            self.timestep += 1
    

if __name__ == "__main__":
    api_keys_path = "../api_keys.cfg"
    api_keys = config.Config(api_keys_path)
    authenticate_openai(api_keys['OPENAI_API_KEY'])

    agent_dict = {
        "name": "Alice",
        "description": "a curious and energetic person",
        "status": "awake",
        "emotional_status": "neutral",
        "location": "in the center of the living room",
        "room": "living_room"
    }

    living_room_dict = {
        "id": "living_room",
        "description": "a cozy living room",
        "entities": {
            "sofa": {
                "name": "sofa",
                "description": "a 3-person red sofa",
                "status": "unoccupied",
                "location": "against the east wall of the room"
            },
            "armchair": {
                "name": "armchair",
                "description": "a blue armchair",
                "status": "unoccupied",
                "location": "against the west wall of the room"
            },
            "laptop": {
                "description": "a laptop",
                "status": "on, with a new tab open in the browser",
                "location": "on the sofa"
            },
            "alexa": {
                "name": "alexa",
                "description": "a voice-activated smart speaker",
                "status": "ready for voice commands",
                "location": "mounted on the wall above the red sofa"
            },
            "kitchen_door": {
                "name": "kitchen_door",
                "description": "entry to the kitchen",
                "status": "open",
                "location": "on the north wall of the room",
                "leads_to": "kitchen"
            },
        }
    }

    kitchen_room_dict = {
        "id": "kitchen",
        "description": "a spacious kitchen",
        "entities": {
            "fridge": {
                "name": "fridge",
                "description": "a refrigerator",
                "status": "closed, has any possible food item you can imagine inside",
                "location": "against the north wall of the room"
            },
            "stove": {
                "name": "stove",
                "description": "a stove",
                "status": "off, with a pot on the left burner",
                "location": "to the left of the fridge"
            },
            "island": {
                "name": "island",
                "description": "a granite island countertop",
                "status": "has a cutting board with a knife on it",
                "location": "against the east wall of the room"
            },
            "living_room_door": {
                "name": "living_room_door",
                "description": "entry to the living room",
                "status": "open",
                "location": "on the south wall of the room",
                "leads_to": "living_room"
            },
        }
    }

    simulator = Simulator(agent_dict, [living_room_dict, kitchen_room_dict])

    for room in simulator.rooms.values():
        print(room.state_str())
    print(simulator.agent.state_str())
    print()

    simulator.run(20)