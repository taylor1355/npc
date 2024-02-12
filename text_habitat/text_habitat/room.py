import copy

from .agent import Agent
from .entity import Entity
from .state import State

class Room:
    ID_KEY = "id"
    REQUIRED_KEYS = [ID_KEY, "description", "agents", "entities"]
    IMMUTABLE_KEYS = [ID_KEY]

    def __init__(self, state_dict):
        self.id = state_dict[Room.ID_KEY]

        agents_constructor = lambda agents_dict: {id: Agent(agent_dict) for id, agent_dict in agents_dict.items()}
        entities_constructor = lambda entities_dict: {id: Entity(entity_dict) for id, entity_dict in entities_dict.items()}
        self.state = State(
            state_dict,
            self.REQUIRED_KEYS, self.IMMUTABLE_KEYS,
            key_serializers={"agents": agents_constructor, "entities": entities_constructor}
        )

    @staticmethod
    def load_rooms(room_dicts):
        return {room_dict[Room.ID_KEY]: Room(room_dict) for room_dict in room_dicts}

    def update(self, new_state_dict):
        # no new entities can be added
        for entity_id in new_state_dict["entities"]:
            if str(entity_id) not in self.state["entities"]:
                print(f"Warning: entity with id {entity_id} not found in room.")
                return

        # entities cannot be removed
        for entity_id in self.state["entities"]:
            if str(entity_id) not in new_state_dict["entities"]:
                print(f"Warning: entity with id {entity_id} not found in new state.")
                return

        self.state.update(new_state_dict)

    def move_agent(self, agent_id, new_room):
        new_room.state.state_dict["agents"][agent_id] = copy.deepcopy(self.state.state_dict["agents"][agent_id])
        del self.state.state_dict["agents"][agent_id]
        return new_room