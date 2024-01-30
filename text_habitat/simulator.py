import config

from text_habitat.agent import Agent
from text_habitat.gamemaster import Gamemaster
from text_habitat.openai_api import authenticate_openai
from text_habitat.room import Room

class Simulator:
    def __init__(self, room_dict):
        self.room = Room.from_dict(room_dict)
        self.agent = Agent("Alice")
        self.gamemaster = Gamemaster()

        self.timestep = 0

    def execute_action(self, action):
        action.acting_agent.current_action = action

        updated_room_state, updated_agent_state = action.execute(self)
        self.room.update_dict(updated_room_state)
        self.agent.update_dict(updated_agent_state)

        print(str(action))
        print()
        print(self.room.state_str())
        print(self.agent.state_str())
        print()

    def run(self, num_timesteps):
        # TODO: have a priority queue of events to avoid needlessly simulating every timestep
        user_actions = []
        system_actions = []
        for _ in range(num_timesteps):
            if self.agent.current_action is None or self.agent.current_action.is_done(self.timestep):
                print(100 * "=")
                print(100 * "=")

                action_intent = self.agent.decide_action(self.room)
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

    room_dict = {
        "description": "a cozy living room",
        "entities": {
            "0": {
                "description": "a red sofa",
                "state": "empty",
                "location": "against the east wall of the room"
            },
            "1": {
                "description": "a blue armchair",
                "state": "empty",
                "location": "against the west wall of the room"
            },
            "2": {
                "description": "a laptop",
                "state": "on, with a new tab open in the browser",
                "location": "on the left cushion of the red sofa"
            },
            "3": {
                "description": "a voice-activated smart speaker",
                "state": "off, but ready for voice commands",
                "location": "mounted on the wall above the red sofa"
            },
            "4": {
                "description": "a digital picture frame that can display various artworks or personal photos",
                "state": "Displaying a slideshow of famous paintings. The current painting is 'The Starry Night' by Vincent van Gogh.",
                "location": "mounted on the wall above the blue armchair"
            }
        }
    }

    simulator = Simulator(room_dict)

    print(simulator.room.state_str())
    print(simulator.agent.state_str())
    print()

    simulator.run(30)