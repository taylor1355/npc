from .state import State

class Entity:
    ID_KEY = "id"
    REQUIRED_KEYS = [ID_KEY, "description", "status", "location"]
    IMMUATABLE_KEYS = [ID_KEY]

    def __init__(self, state_dict):
        self.id = state_dict[Entity.ID_KEY]
        self.state = State(state_dict, self.REQUIRED_KEYS, self.IMMUATABLE_KEYS)

    def update(self, new_state_dict):
        self.state.update(new_state_dict)