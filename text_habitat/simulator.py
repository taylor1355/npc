import config

from text_habitat.action import Action
from text_habitat.agent import Agent
from text_habitat.event import EventQueue, ChooseActionEvent, ConstructActionEvent, AffectRoomEvent, CheckRoomStateEvent
from text_habitat.gamemaster import Gamemaster
from text_habitat.openai_api import authenticate_openai
from text_habitat.room import Room
from text_habitat.utils import execute_state_updating_code, state_dict_str

# TODO: create state_dict class. This will collect utils.state_dict_str and the state dict logic in room, agent, entity into one place. It should handle required/immutable keys
class Simulator:
    def __init__(self, agent_dict, room_dicts):
        # TODO: allow multiple agents
        self.agent = Agent(agent_dict["name"], agent_dict)
        self.rooms = {room_dict["id"]: Room.from_dict(room_dict) for room_dict in room_dicts}
        self.gamemaster = Gamemaster()

        self.timestep = 0
        self.events = EventQueue([
            ChooseActionEvent(self.timestep, self.agent.name)
        ])

    # TODO: come up with a better way of passing rooms around
    def get_agent_room(self, agent):
        return self.rooms[agent.state_dict["room"]]
    def get_action_room(self, action):
        return self.rooms[action.room]

    # TODO: store event history
    def run(self, num_timesteps):
        end_timestep = self.timestep + num_timesteps
        while self.timestep < end_timestep and not self.events.is_empty():
            event = self.events.pop()
            self.timestep = event.timestep

            new_events = []
            if isinstance(event, ChooseActionEvent):
                room_description = state_dict_str(self.get_agent_room(self.agent).to_dict())
                action_intent = self.agent.decide_action(room_description)

                new_events.append(
                    ConstructActionEvent(self.timestep, self.agent.name, action_intent)
                )
            elif isinstance(event, ConstructActionEvent):
                room_state_dict = self.get_agent_room(self.agent).to_dict()
                agent_state_dict = self.agent.state_dict
                state_updating_code, time_taken, metadata = self.gamemaster.generate_user_action_code(event.action_intent, room_state_dict, agent_state_dict)
                action = Action(
                    start_time=self.timestep,
                    end_time=self.timestep + time_taken,
                    state_updating_code=state_updating_code,
                    acting_agent=self.agent,
                    metadata=metadata,
                )

                print(100 * "=")
                print(100 * "=")
                print(str(action))
                print()

                new_events.extend([
                    AffectRoomEvent(self.timestep, self.get_action_room(action).id, action.state_updating_code, check_state=True),
                    ChooseActionEvent(self.timestep + time_taken, self.agent.name)
                ])
            elif isinstance(event, AffectRoomEvent):
                room = self.rooms[event.room_id]
                agent = self.agent
                updated_room_state, updated_agent_state = execute_state_updating_code(event.state_modifying_code, room, agent)
                room.update_dict(updated_room_state)
                agent.update(updated_agent_state)

                print(state_dict_str(room.to_dict()))
                print(state_dict_str(action.acting_agent.state_dict))
                print()

                if event.check_state:
                    new_events.append(
                        CheckRoomStateEvent(self.timestep, room.id)
                    )
            elif isinstance(event, CheckRoomStateEvent):
                room_state_dict = self.rooms[event.room_id].to_dict()
                agent_state_dict = self.agent.state_dict
                state_correction_code, metadata = self.gamemaster.generate_state_correction_code(room_state_dict, agent_state_dict)
            
                print(100 * "-")

                new_events.append(
                    AffectRoomEvent(self.timestep, event.room_id, state_correction_code, check_state=False, metadata=metadata)
                )

            if new_events:
                self.events.push_batch(new_events)
        
        if self.events.is_empty():
            print("Warning: Simulation ended because there are no more events.")
            self.timestep = end_timestep
        print(f"Simulation ended at timestep {self.timestep}.")

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
        print(state_dict_str(room.to_dict()))
    print(simulator.agent.state_dict)
    print()

    simulator.run(20)