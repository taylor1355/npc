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


class NoAdvertisedParameterError(ActionValidationError):
    """An act carried none of the parameters its interaction advertises.

    Distinct from MissingRequiredParameterError, whose `param_name` is one
    parameter. Here the requirement is a disjunction over the advertised set, so
    the set is carried as a list rather than packed into a single-name field.
    """

    def __init__(self, param_names: list[str], action_type: str):
        self.param_names = param_names
        self.action_type = action_type
        super().__init__(
            f"Action '{action_type}' supplied none of the advertised parameters: "
            f"{', '.join(param_names)}"
        )


class MovementLockedError(ActionValidationError):
    """Movement action attempted while movement is locked"""

    def __init__(self):
        super().__init__(
            "Movement actions not available - character is locked in current interaction. "
            "Use 'cancel_interaction' or 'continue' instead."
        )


class ActivityLockedError(ActionValidationError):
    """A new activity was attempted while current state blocks transitions."""

    def __init__(self, action_type: str, reason: str):
        self.action_type = action_type
        self.reason = reason
        super().__init__(f"Action '{action_type}' cannot start because {reason}.")


class InvalidSelectedOptionError(ActionValidationError):
    """The selected option handle or its echoed first action is invalid."""

    def __init__(self, option_id: str, detail: str):
        self.option_id = option_id
        self.detail = detail
        super().__init__(f"Invalid selected_option_id '{option_id}': {detail}")
