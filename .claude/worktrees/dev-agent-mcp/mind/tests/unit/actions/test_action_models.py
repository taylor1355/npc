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


class TestBidResponseValidation:
    """Test validation of RESPOND_TO_INTERACTION_BID actions"""

    def _create_mock_state_with_bids(self, pending_bids: dict):
        """Helper to create mock pipeline state with pending bids"""
        from mind.cognitive_architecture.observations import Observation

        observation = Observation(entity_id="npc_001", current_simulation_time=100)
        state = Mock()
        state.observation = observation
        state.pending_incoming_bids = pending_bids
        return state

    def test_validate_accept_bid_with_valid_bid_id(self):
        """Should validate accept action when bid_id exists"""
        from mind.cognitive_architecture.observations import MindEvent, MindEventType

        bid_event = MindEvent(
            timestamp=100,
            event_type=MindEventType.INTERACTION_BID_RECEIVED,
            payload={
                "bid_id": "bid_789",
                "bidder_id": "charlie_001",
                "bidder_name": "Charlie",
                "interaction_name": "conversation",
            },
        )

        state = self._create_mock_state_with_bids({"bid_789": bid_event})

        action = Action.model_validate(
            {"action": "respond_to_interaction_bid", "parameters": {"bid_id": "bid_789", "accept": True}},
            context={"state": state},
        )

        assert action.action == "respond_to_interaction_bid"
        assert action.parameters["bid_id"] == "bid_789"
        assert action.parameters["accept"] is True

    def test_validate_reject_bid_with_reason(self):
        """Should validate reject action when reason is provided"""
        from mind.cognitive_architecture.observations import MindEvent, MindEventType

        bid_event = MindEvent(
            timestamp=100,
            event_type=MindEventType.INTERACTION_BID_RECEIVED,
            payload={
                "bid_id": "bid_999",
                "bidder_id": "dave_001",
                "bidder_name": "Dave",
                "interaction_name": "craft",
            },
        )

        state = self._create_mock_state_with_bids({"bid_999": bid_event})

        action = Action.model_validate(
            {
                "action": "respond_to_interaction_bid",
                "parameters": {"bid_id": "bid_999", "accept": False, "reason": "Currently busy"},
            },
            context={"state": state},
        )

        assert action.action == "respond_to_interaction_bid"
        assert action.parameters["accept"] is False
        assert action.parameters["reason"] == "Currently busy"

    def test_reject_bid_without_reason_fails(self):
        """Should fail validation when rejecting without reason"""
        from mind.cognitive_architecture.observations import MindEvent, MindEventType

        bid_event = MindEvent(
            timestamp=100,
            event_type=MindEventType.INTERACTION_BID_RECEIVED,
            payload={"bid_id": "bid_111", "bidder_id": "eve_001", "bidder_name": "Eve", "interaction_name": "trade"},
        )

        state = self._create_mock_state_with_bids({"bid_111": bid_event})

        with pytest.raises(ValidationError) as exc_info:
            Action.model_validate(
                {"action": "respond_to_interaction_bid", "parameters": {"bid_id": "bid_111", "accept": False}},
                context={"state": state},
            )

        assert "reason" in str(exc_info.value).lower()

    def test_invalid_bid_id_fails(self):
        """Should fail validation when bid_id doesn't exist"""
        state = self._create_mock_state_with_bids({})  # No pending bids

        with pytest.raises(ValidationError) as exc_info:
            Action.model_validate(
                {
                    "action": "respond_to_interaction_bid",
                    "parameters": {"bid_id": "nonexistent_bid", "accept": True},
                },
                context={"state": state},
            )

        assert "nonexistent_bid" in str(exc_info.value)

    def test_missing_bid_id_fails(self):
        """Should fail validation when bid_id parameter is missing"""
        state = self._create_mock_state_with_bids({})

        with pytest.raises(ValidationError) as exc_info:
            Action.model_validate(
                {"action": "respond_to_interaction_bid", "parameters": {"accept": True}},
                context={"state": state},
            )

        assert "bid_id" in str(exc_info.value).lower()

    def test_missing_accept_parameter_fails(self):
        """Should fail validation when accept parameter is missing"""
        from mind.cognitive_architecture.observations import MindEvent, MindEventType

        bid_event = MindEvent(
            timestamp=100,
            event_type=MindEventType.INTERACTION_BID_RECEIVED,
            payload={"bid_id": "bid_222", "bidder_id": "frank_001", "bidder_name": "Frank", "interaction_name": "sit"},
        )

        state = self._create_mock_state_with_bids({"bid_222": bid_event})

        with pytest.raises(ValidationError) as exc_info:
            Action.model_validate(
                {"action": "respond_to_interaction_bid", "parameters": {"bid_id": "bid_222"}},
                context={"state": state},
            )

        assert "accept" in str(exc_info.value).lower()
