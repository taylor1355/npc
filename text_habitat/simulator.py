import config
import itertools
import pprint

from text_habitat.action import Action
from text_habitat.agent import Agent
from text_habitat.event import EventQueue, ChooseActionEvent, ConstructActionEvent, AffectRoomEvent, CheckRoomStateEvent
from text_habitat.gamemaster import Gamemaster
from text_habitat.openai_api import authenticate_openai
from text_habitat.room import Room
from text_habitat.utils import execute_state_updating_code

# TODO: make Agent, Room, and Entity classes subclasses of StatefulObject. StatefulObject should have:
# - ID_KEY, REQUIRED_KEYS, IMMUTABLE_KEYS class attributes
# - __init__ method that initializes a State object
# - update method that updates the State object
# - load_objects method that loads an {id: object} dict from a list of state dictionaries
# - id method that returns the object's ID (this should replace having an id/name attribute). In code, even the agent's name should be referred to as id for consistency.
# - __str__ method that returns a string representation of the object
class Simulator:
    def __init__(self, room_dicts):
        self.rooms = Room.load_rooms(room_dicts)
        self.gamemaster = Gamemaster()
        self.timestep = 0

        agents = itertools.chain.from_iterable(room.state["agents"].values() for room in self.rooms.values())
        self.events = EventQueue([
            ChooseActionEvent(self.timestep, agent.id)
            for agent in agents
        ])

    def locate_agent(self, agent_id):
        for room in self.rooms.values():
            room_agents = room.state["agents"]
            if agent_id in room_agents:
                return room, room_agents[agent_id]
        raise ValueError(f"Agent with id {agent_id} not found in any room.")

    # TODO: store event history
    def run(self, num_timesteps):
        end_timestep = self.timestep + num_timesteps
        while self.timestep < end_timestep and not self.events.is_empty():
            event = self.events.pop()
            self.timestep = event.timestep

            # TODO: move event handlers to the event classes
            new_events = []
            if isinstance(event, ChooseActionEvent):
                room, agent = self.locate_agent(event.agent_id)

                room_description = str(room.state)
                action_intent = agent.decide_action(room_description)

                new_events.append(
                    ConstructActionEvent(self.timestep, agent.id, action_intent)
                )
            elif isinstance(event, ConstructActionEvent):
                room, agent = self.locate_agent(event.agent_id)

                state_updating_code, time_taken, metadata = self.gamemaster.generate_user_action_code(agent.id, event.action_intent, room.state)
                # TODO: instead of having an Action object, move the relevant metadata to the AffectRoomEvent
                action = Action(
                    start_time=self.timestep,
                    end_time=self.timestep + time_taken,
                    state_updating_code=state_updating_code,
                    acting_agent=agent,
                    metadata=metadata,
                )

                print(100 * "=")
                print(100 * "=")
                print(str(action))
                print()

                new_events.extend([
                    AffectRoomEvent(self.timestep, room.id, state_updating_code, check_state=True),
                    ChooseActionEvent(self.timestep + time_taken, agent.id)
                ])
            elif isinstance(event, AffectRoomEvent):
                room = self.rooms[event.room_id]

                updated_room_state, agent_moves = execute_state_updating_code(event.state_modifying_code, room.state)
                room.update(updated_room_state)
                for agent_id, new_room_id in agent_moves.items():
                    if new_room_id not in self.rooms:
                        print(f"Warning: room with id {new_room_id} not found.")
                        continue
                    elif agent_id not in room.state.state_dict["agents"]:
                        print(f"Warning: agent with id {agent_id} not found in room {room.id}.")
                        continue

                    new_room = self.rooms[new_room_id]
                    room.move_agent(agent_id, new_room)

                print(room.state)
                print()

                if event.check_state:
                    new_events.append(
                        CheckRoomStateEvent(self.timestep, room.id)
                    )
            elif isinstance(event, CheckRoomStateEvent):
                room = self.rooms[event.room_id]

                state_correction_code, metadata = self.gamemaster.generate_state_correction_code(room.state)
            
                print(100 * "-")
                print(state_correction_code)
                pprint.pprint(metadata)

                new_events.append(
                    AffectRoomEvent(self.timestep, room.id, state_correction_code, check_state=False, metadata=metadata)
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

    living_room_dict = {
        "id": "living_room",
        "description": "a cozy living room",
        "agents": {
            "Alice": {
                "name": "Alice",
                "description": "a curious and energetic chef who really wants to cook something delicious",
                "physical_status": "hungry",
                "emotional_status": "excited to cook",
                "location": "standing in the center of the room"
            }
        },
        "entities": {
            "sofa": {
                "id": "sofa",
                "description": "a 3-person red sofa",
                "status": "unoccupied",
                "location": "against the east wall of the room"
            },
            "armchair": {
                "id": "armchair",
                "description": "a blue armchair",
                "status": "unoccupied",
                "location": "against the west wall of the room"
            },
            "laptop": {
                "id": "laptop",
                "description": "a laptop",
                "status": "on, with a new tab open in the browser",
                "location": "on the sofa"
            },
            "alexa": {
                "id": "alexa",
                "description": "a voice-activated smart speaker",
                "status": "ready for voice commands",
                "location": "mounted on the wall above the red sofa"
            },
            "kitchen_door": {
                "id": "kitchen_door",
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
        "agents": {
        },
        "entities": {
            "fridge": {
                "id": "fridge",
                "description": "a refrigerator",
                "status": "closed, has any possible food item you can imagine inside",
                "location": "against the north wall of the room"
            },
            "stove": {
                "id": "stove",
                "description": "a stove",
                "status": "off, with a pot on the left burner",
                "location": "to the left of the fridge"
            },
            "island": {
                "id": "island",
                "description": "a granite island countertop",
                "status": "has a cutting board with a knife on it",
                "location": "against the east wall of the room"
            },
            "living_room_door": {
                "id": "living_room_door",
                "description": "entry to the living room",
                "status": "open",
                "location": "on the south wall of the room",
                "leads_to": "living_room"
            },
        }
    }

    simulator = Simulator([living_room_dict, kitchen_room_dict])

    for room in simulator.rooms.values():
        print(room.state)
    print()

    simulator.run(20)