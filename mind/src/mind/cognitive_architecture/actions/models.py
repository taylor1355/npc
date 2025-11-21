"""Action models for the cognitive architecture"""

from enum import Enum

from pydantic import BaseModel, Field, ValidationInfo, model_validator

from mind.cognitive_architecture.actions.exceptions import (
    InvalidEntityError,
    InvalidInteractionError,
    MissingRequiredParameterError,
    MovementLockedError,
)


class ActionType(str, Enum):
    """Available action types"""

    MOVE_TO = "move_to"
    MOVE_DIRECTION = "move_direction"
    INTERACT_WITH = "interact_with"
    WANDER = "wander"
    WAIT = "wait"
    CONTINUE = "continue"
    CANCEL_INTERACTION = "cancel_interaction"
    ACT_IN_INTERACTION = "act_in_interaction"
    RESPOND_TO_INTERACTION_BID = "respond_to_interaction_bid"
    BATCH_REJECT_INTERACTION_BIDS = "batch_reject_interaction_bids"


class Action(BaseModel):
    """Action to be executed"""

    model_config = {"use_enum_values": True}

    action: ActionType
    parameters: dict = Field(
        default_factory=dict,
        description="Action parameters as key-value pairs. Use exact parameter names from the action description."
    )

    def __str__(self) -> str:
        """Format action for LLM consumption"""
        params_str = (
            ", ".join([f"{k}={v}" for k, v in self.parameters.items()])
            if self.parameters
            else "no parameters"
        )
        return f"{self.action}({params_str})"

    @model_validator(mode='after')
    def validate_executable(self, info: ValidationInfo):
        """Validate action can be executed given pipeline state.

        Requires 'state' in validation context.
        Raises ValidationError (wrapping ActionValidationError) if invalid.
        """
        if not info.context:
            raise ValueError("Action validation requires context with 'state'")

        state = info.context.get('state')
        if not state:
            raise ValueError("Action validation requires 'state' in context")

        observation = state.observation

        # Run validation checks
        self._validate_movement_lock(observation)

        if self.action == ActionType.INTERACT_WITH:
            self._validate_interact_with(observation)
        elif self.action == ActionType.MOVE_TO:
            self._validate_move_to()
        elif self.action == ActionType.RESPOND_TO_INTERACTION_BID:
            self._validate_respond_to_bid(state)
        elif self.action == ActionType.BATCH_REJECT_INTERACTION_BIDS:
            self._validate_batch_reject_bids(state)
        elif self.action == ActionType.ACT_IN_INTERACTION:
            self._validate_act_in_interaction(observation)

        return self

    def _validate_movement_lock(self, observation):
        """Check if movement-based actions are blocked"""
        if observation.status and observation.status.movement_locked:
            if self.action in (ActionType.MOVE_TO, ActionType.MOVE_DIRECTION, ActionType.WANDER):
                raise MovementLockedError()

    def _validate_interact_with(self, observation):
        """Validate INTERACT_WITH action against observation"""
        entity_id = self.parameters.get("entity_id")
        interaction_name = self.parameters.get("interaction_name")

        if not entity_id:
            raise MissingRequiredParameterError("entity_id", self.action)
        if not interaction_name:
            raise MissingRequiredParameterError("interaction_name", self.action)

        # Check entity visibility
        if observation.vision:
            visible_entities = observation.vision.visible_entities
            visible_ids = [e.entity_id for e in visible_entities]

            if entity_id not in visible_ids:
                raise InvalidEntityError(entity_id, visible_ids)

            # Check interaction availability
            entity = next(e for e in visible_entities if e.entity_id == entity_id)
            if interaction_name not in entity.interactions:
                raise InvalidInteractionError(
                    interaction_name, entity_id, list(entity.interactions.keys())
                )

    def _validate_move_to(self):
        """Validate MOVE_TO action parameters"""
        if "destination" not in self.parameters:
            raise MissingRequiredParameterError("destination", self.action)

    def _validate_respond_to_bid(self, state):
        """Validate RESPOND_TO_INTERACTION_BID action against pending bids"""
        bid_id = self.parameters.get("bid_id")
        accept = self.parameters.get("accept")

        if not bid_id:
            raise MissingRequiredParameterError("bid_id", self.action)
        if accept is None:
            raise MissingRequiredParameterError("accept", self.action)

        # Check bid exists in pending bids
        if bid_id not in state.pending_incoming_bids:
            raise ValueError(
                f"Invalid bid_id '{bid_id}'. Available bids: {list(state.pending_incoming_bids.keys())}"
            )

        # If rejecting, reason is required
        if not accept and not self.parameters.get("reason"):
            raise MissingRequiredParameterError("reason", self.action)

    def _validate_batch_reject_bids(self, state):
        """Validate BATCH_REJECT_INTERACTION_BIDS action against pending bids"""
        ids = self.parameters.get("ids")
        reason = self.parameters.get("reason")

        if not ids:
            raise MissingRequiredParameterError("ids", self.action)
        if not reason:
            raise MissingRequiredParameterError("reason", self.action)

        # If not wildcard, validate the specified IDs exist
        if ids != "*":
            if not isinstance(ids, list):
                raise ValueError(f"Parameter 'ids' must be '*' or a list, got: {type(ids)}")

            # Check if these are bid IDs or entity IDs
            pending_bid_ids = set(state.pending_incoming_bids.keys())
            pending_entity_ids = {
                event.payload.get("bidder_id")
                for event in state.pending_incoming_bids.values()
            }

            # Validate each item is either a valid bid_id or entity_id
            for item in ids:
                if item not in pending_bid_ids and item not in pending_entity_ids:
                    raise ValueError(
                        f"'{item}' is not a valid bid_id or entity_id. "
                        f"Available bid_ids: {list(pending_bid_ids)}, "
                        f"Available entity_ids: {list(pending_entity_ids)}"
                    )

    def _validate_act_in_interaction(self, observation):
        """Validate ACT_IN_INTERACTION action requires appropriate parameters"""
        if not observation.status or not observation.status.current_interaction:
            raise ValueError("ACT_IN_INTERACTION requires an active interaction")

        interaction_name = observation.status.current_interaction.get('interaction_name', '')

        # For conversations, message parameter is required
        if interaction_name == 'conversation':
            message = self.parameters.get("message")
            if not message:
                raise MissingRequiredParameterError("message", self.action)


class AvailableAction(BaseModel):
    """An action that can be taken"""

    name: str = Field(description="Action identifier like 'move_to'")
    description: str = Field(description="Human-readable description of what this action does")
    parameters: dict[str, str] = Field(
        default_factory=dict, description="Parameter names mapped to their descriptions"
    )

    def __str__(self) -> str:
        """Format available action for LLM consumption"""
        if self.parameters:
            params_str = ", ".join([f"{param}: {desc}" for param, desc in self.parameters.items()])
            return f"{self.name}: {self.description} (params: {params_str})"
        return f"{self.name}: {self.description}"
