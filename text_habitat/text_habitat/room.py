import pprint

from .entity import Entity

# TODO: Room state should be represented as a knowledge graph. Each entity is a node, and each edge is a relationship between two entities.
# Entities can have attributes, and relationships can have attributes.
class Room:
    def __init__(self, id, description, entities):
        self.id = id
        self.description = description
        self.entities = entities
    
    @staticmethod
    def from_dict(room_dict):
        id = room_dict["id"]
        description = room_dict["description"]
        entities = {id: Entity(id, entity_dict) for id, entity_dict in room_dict["entities"].items()}
        return Room(id, description, entities)

    def update_dict(self, room_dict_changes):
        if "description" in room_dict_changes:
            self.description = room_dict_changes["description"]

        if "entities" in room_dict_changes:
            for entity_id, entity_dict_changes in room_dict_changes["entities"].items():
                key = str(entity_id)
                if key in self.entities:
                    self.entities[key].update(entity_dict_changes)
                else:
                    print(f"Warning: entity with id {key} not found in room.")

    def to_dict(self):
        room_dict = {
            "id": self.id,
            "description": self.description,
            "entities": {id: entity.state_dict for id, entity in self.entities.items()}
        }
        return room_dict