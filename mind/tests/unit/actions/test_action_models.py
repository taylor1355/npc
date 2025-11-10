"""Unit tests for action models"""

import pytest
from pydantic import ValidationError
from unittest.mock import Mock

from mind.cognitive_architecture.actions import Action
from mind.cognitive_architecture.actions.exceptions import (
    InvalidEntityError,
    InvalidInteractionError,
    MissingRequiredParameterError,
    MovementLockedError,
)
from mind.cognitive_architecture.observations import (
    EntityData,
    Observation,
    StatusObservation,
    VisionObservation,
)


class TestActionModel:
    """Test Action model creation (without validation)"""

    def test_create_basic_action(self):
        """Should create action with type"""
        action = Action.model_construct(action="wait")

        assert action.action == "wait"
        assert action.parameters == {}

    def test_create_action_with_parameters(self):
        """Should create action with parameters"""
        action = Action.model_construct(action="move_to", parameters={"destination": [10, 20]})

        assert action.action == "move_to"
        assert action.parameters["destination"] == [10, 20]

    def test_create_interaction_action(self):
        """Should create interact_with action"""
        action = Action.model_construct(
            action="interact_with",
            parameters={"entity_id": "food_1", "interaction_name": "eat"},
        )

        assert action.action == "interact_with"
        assert action.parameters["entity_id"] == "food_1"
        assert action.parameters["interaction_name"] == "eat"


class TestActionValidation:
    """Test Action validation with pipeline state context"""

    def _create_mock_state(self, observation: Observation):
        """Helper to create mock pipeline state"""
        state = Mock()
        state.observation = observation
        return state

    def _create_basic_observation(self):
        """Helper to create observation with visible entity"""
        return Observation(
            entity_id="npc_001",
            current_simulation_time=100,
            vision=VisionObservation(
                visible_entities=[
                    EntityData(
                        entity_id="chair_uuid_123",
                        display_name="Wooden Chair",
                        position=(5, 5),
                        interactions={
                            "sit": {
                                "description": "Sit on the chair",
                                "needs_filled": ["rest"],
                            }
                        },
                    )
                ]
            ),
        )

    def test_validation_requires_context(self):
        """Should fail if validation context is missing"""
        with pytest.raises(ValueError, match="Action validation requires context"):
            Action(action="wait", parameters={})

    def test_validation_requires_state_in_context(self):
        """Should fail if 'state' not in validation context"""
        with pytest.raises(ValidationError) as exc_info:
            Action.model_validate({"action": "wait", "parameters": {}}, context={})

        assert "requires context with 'state'" in str(exc_info.value)

    def test_valid_wait_action(self):
        """Should validate simple wait action"""
        observation = Observation(entity_id="npc_001", current_simulation_time=100)
        state = self._create_mock_state(observation)

        action = Action.model_validate(
            {"action": "wait", "parameters": {}}, context={"state": state}
        )

        assert action.action == "wait"

    def test_valid_move_to_action(self):
        """Should validate move_to with destination"""
        observation = Observation(entity_id="npc_001", current_simulation_time=100)
        state = self._create_mock_state(observation)

        action = Action.model_validate(
            {"action": "move_to", "parameters": {"destination": [10, 20]}},
            context={"state": state},
        )

        assert action.action == "move_to"
        assert action.parameters["destination"] == [10, 20]

    def test_move_to_missing_destination(self):
        """Should fail if move_to lacks destination parameter"""
        observation = Observation(entity_id="npc_001", current_simulation_time=100)
        state = self._create_mock_state(observation)

        with pytest.raises(ValidationError) as exc_info:
            Action.model_validate(
                {"action": "move_to", "parameters": {}}, context={"state": state}
            )

        # Check that it's wrapped MissingRequiredParameterError
        assert "destination" in str(exc_info.value)

    def test_movement_locked_blocks_move_to(self):
        """Should fail if trying to move while movement is locked"""
        observation = Observation(
            entity_id="npc_001",
            current_simulation_time=100,
            status=StatusObservation(position=(0, 0), movement_locked=True),
        )
        state = self._create_mock_state(observation)

        with pytest.raises(ValidationError) as exc_info:
            Action.model_validate(
                {"action": "move_to", "parameters": {"destination": [10, 20]}},
                context={"state": state},
            )

        assert "Movement actions not available" in str(exc_info.value)

    def test_movement_locked_blocks_wander(self):
        """Should fail if trying to wander while movement is locked"""
        observation = Observation(
            entity_id="npc_001",
            current_simulation_time=100,
            status=StatusObservation(position=(0, 0), movement_locked=True),
        )
        state = self._create_mock_state(observation)

        with pytest.raises(ValidationError) as exc_info:
            Action.model_validate(
                {"action": "wander", "parameters": {}}, context={"state": state}
            )

        assert "Movement actions not available" in str(exc_info.value)

    def test_movement_locked_allows_wait(self):
        """Should allow wait action even when movement is locked"""
        observation = Observation(
            entity_id="npc_001",
            current_simulation_time=100,
            status=StatusObservation(position=(0, 0), movement_locked=True),
        )
        state = self._create_mock_state(observation)

        action = Action.model_validate(
            {"action": "wait", "parameters": {}}, context={"state": state}
        )

        assert action.action == "wait"

    def test_valid_interact_with_action(self):
        """Should validate interact_with with valid entity and interaction"""
        observation = self._create_basic_observation()
        state = self._create_mock_state(observation)

        action = Action.model_validate(
            {
                "action": "interact_with",
                "parameters": {
                    "entity_id": "chair_uuid_123",
                    "interaction_name": "sit",
                },
            },
            context={"state": state},
        )

        assert action.action == "interact_with"
        assert action.parameters["entity_id"] == "chair_uuid_123"
        assert action.parameters["interaction_name"] == "sit"

    def test_interact_with_missing_entity_id(self):
        """Should fail if interact_with lacks entity_id"""
        observation = self._create_basic_observation()
        state = self._create_mock_state(observation)

        with pytest.raises(ValidationError) as exc_info:
            Action.model_validate(
                {
                    "action": "interact_with",
                    "parameters": {"interaction_name": "sit"},
                },
                context={"state": state},
            )

        assert "entity_id" in str(exc_info.value)

    def test_interact_with_missing_interaction_name(self):
        """Should fail if interact_with lacks interaction_name"""
        observation = self._create_basic_observation()
        state = self._create_mock_state(observation)

        with pytest.raises(ValidationError) as exc_info:
            Action.model_validate(
                {
                    "action": "interact_with",
                    "parameters": {"entity_id": "chair_uuid_123"},
                },
                context={"state": state},
            )

        assert "interaction_name" in str(exc_info.value)

    def test_interact_with_hallucinated_entity_id(self):
        """Should fail if entity_id not in visible entities (HALLUCINATION)"""
        observation = self._create_basic_observation()
        state = self._create_mock_state(observation)

        with pytest.raises(ValidationError) as exc_info:
            Action.model_validate(
                {
                    "action": "interact_with",
                    "parameters": {
                        "entity_id": "chair_001",  # Hallucinated simplified ID
                        "interaction_name": "sit",
                    },
                },
                context={"state": state},
            )

        error_str = str(exc_info.value)
        assert "chair_001" in error_str
        assert "not found" in error_str.lower()

    def test_interact_with_hallucinated_interaction_name(self):
        """Should fail if interaction_name not available on entity (HALLUCINATION)"""
        observation = self._create_basic_observation()
        state = self._create_mock_state(observation)

        with pytest.raises(ValidationError) as exc_info:
            Action.model_validate(
                {
                    "action": "interact_with",
                    "parameters": {
                        "entity_id": "chair_uuid_123",
                        "interaction_name": "rest",  # Hallucinated interaction
                    },
                },
                context={"state": state},
            )

        error_str = str(exc_info.value)
        assert "rest" in error_str
        assert "not available" in error_str.lower() or "invalid" in error_str.lower()

    def test_interact_with_no_vision_data(self):
        """Should handle case where vision is None"""
        observation = Observation(entity_id="npc_001", current_simulation_time=100)
        state = self._create_mock_state(observation)

        # Should not raise if no vision (can't validate entity visibility)
        action = Action.model_validate(
            {
                "action": "interact_with",
                "parameters": {
                    "entity_id": "some_entity",
                    "interaction_name": "some_action",
                },
            },
            context={"state": state},
        )

        assert action.action == "interact_with"
