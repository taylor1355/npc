"""Action validation exceptions for the cognitive architecture"""


class ActionValidationError(ValueError):
    """Base exception for action validation failures that should trigger retry"""
    pass


class InvalidEntityError(ActionValidationError):
    """Entity referenced in action not found in visible entities"""

    def __init__(self, entity_id: str, available_entities: list[str]):
        self.entity_id = entity_id
        self.available_entities = available_entities
        entities_str = ", ".join(available_entities) if available_entities else "none visible"
        super().__init__(f"Entity '{entity_id}' not found. Available entities: {entities_str}")


class InvalidInteractionError(ActionValidationError):
    """Interaction not available on target entity"""

    def __init__(self, interaction: str, entity_id: str, available: list[str]):
        self.interaction = interaction
        self.entity_id = entity_id
        self.available_interactions = available
        interactions_str = ", ".join(available) if available else "none"
        super().__init__(
            f"Interaction '{interaction}' not available on entity '{entity_id}'. "
            f"Available: {interactions_str}"
        )


class MissingRequiredParameterError(ActionValidationError):
    """Required parameter missing from action"""

    def __init__(self, param_name: str, action_type: str):
        self.param_name = param_name
        self.action_type = action_type
        super().__init__(f"Required parameter '{param_name}' missing for action '{action_type}'")


class MovementLockedError(ActionValidationError):
    """Movement action attempted while movement is locked"""

    def __init__(self):
        super().__init__(
            "Movement actions not available - character is locked in current interaction. "
            "Use 'cancel_interaction' or 'continue' instead."
        )
