class Entity:
    def __init__(self, id, description, state, location):
        self.id = id
        self.description = description
        self.state = state
        self.location = location

    @staticmethod
    def from_dict(id, entity_dict):
        description = entity_dict["description"]
        state = entity_dict["state"]
        location = entity_dict["location"]
        return Entity(id, description, state, location)

    def update_dict(self, entity_dict_changes):
        if "description" in entity_dict_changes:
            self.description = entity_dict_changes["description"]

        if "state" in entity_dict_changes:
            self.state = entity_dict_changes["state"]

        if "location" in entity_dict_changes:
            self.location = entity_dict_changes["location"]

    def to_dict(self):
        state_dict = {
            "description": self.description,
            "state": self.state,
            "location": self.location
        }
        return state_dict