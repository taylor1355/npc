import pprint

from .entity import Entity
from .utils import PPRINT_WIDTH

# TODO: Room state should be represented as a knowledge graph. Each entity is a node, and each edge is a relationship between two entities.
# Entities can have attributes, and relationships can have attributes.
class Room:
    def __init__(self, description, entities):
        self.description = description
        self.entities = entities
    
    @staticmethod
    def from_dict(room_dict):
        description = room_dict["description"]
        entities = {id: Entity.from_dict(id, entity_dict) for id, entity_dict in room_dict["entities"].items()}
        return Room(description, entities)

    def update_dict(self, room_dict_changes):
        if "description" in room_dict_changes:
            self.description = room_dict_changes["description"]

        if "entities" in room_dict_changes:
            for entity_id, entity_dict_changes in room_dict_changes["entities"].items():
                key = str(entity_id)
                if key in self.entities:
                    self.entities[key].update_dict(entity_dict_changes)
                else:
                    print(f"Warning: entity with id {key} not found in room.")

    def to_dict(self):
        room_dict = {
            "description": self.description,
            "entities": {id: entity.to_dict() for id, entity in self.entities.items()}
        }
        return room_dict

    def state_str(self):
        return pprint.pformat(self.to_dict(), sort_dicts=False, width=PPRINT_WIDTH)