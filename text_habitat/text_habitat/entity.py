class Entity:
    REQUIRED_KEYS = ["name", "description", "status", "location"]
    IMMUATABLE_KEYS = ["name"]

    def __init__(self, id, state_dict):
        self.id = id
        self.state_dict = state_dict

    def update(self, state_dict_changes):
        for key, value in state_dict_changes.items():
            if key in self.state_dict and key not in self.IMMUATABLE_KEYS:
                self.state_dict[key] = value