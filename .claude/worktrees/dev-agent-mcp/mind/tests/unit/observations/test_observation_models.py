"""Unit tests for observation models"""

from mind.cognitive_architecture.observations import (
    ConversationMessage,
    ConversationObservation,
    EntityData,
    MindEvent,
    MindEventType,
    NeedsObservation,
    Observation,
    StatusObservation,
    VisionObservation,
)


class TestObservationModels:
    """Test observation model hierarchy"""

    def test_create_status_observation(self):
        """Should create status observation with position"""
        status = StatusObservation(position=(5, 10), movement_locked=False)

        assert status.position == (5, 10)
        assert status.movement_locked is False

    def test_create_needs_observation(self):
        """Should create needs observation with needs dict"""
        needs = NeedsObservation(
            needs={"hunger": 25.0, "energy": 70.0, "fun": 50.0}, max_value=100.0
        )

        assert needs.needs["hunger"] == 25.0
        assert needs.max_value == 100.0

    def test_create_vision_observation(self):
        """Should create vision observation with entities"""
        entity = EntityData(
            entity_id="tree_1",
            display_name="Oak Tree",
            position=(6, 10),
            interactions={
                "examine": {
                    "name": "examine",
                    "description": "Look closely",
                    "needs_filled": ["fun"],
                    "needs_drained": [],
                }
            },
        )

        vision = VisionObservation(visible_entities=[entity])

        assert len(vision.visible_entities) == 1
        assert vision.visible_entities[0].entity_id == "tree_1"

    def test_create_conversation_observation(self):
        """Should create conversation with message history"""
        msg1 = ConversationMessage(
            speaker_id="npc_1",
            speaker_name="Guard",
            message="Hello traveler",
            timestamp=100,
        )
        msg2 = ConversationMessage(
            speaker_id="npc_2", speaker_name="Merchant", message="Greetings", timestamp=105
        )

        conv = ConversationObservation(
            interaction_id="conv_1",
            interaction_name="conversation",
            participants=["npc_1", "npc_2"],
            conversation_history=[msg1, msg2],
        )

        assert conv.interaction_id == "conv_1"
        assert len(conv.conversation_history) == 2
        assert conv.participants == ["npc_1", "npc_2"]

    def test_create_composite_observation(self):
        """Should create full composite observation"""
        obs = Observation(
            entity_id="blacksmith_1",
            current_simulation_time=1000,
            status=StatusObservation(position=(10, 10), movement_locked=False),
            needs=NeedsObservation(needs={"hunger": 50.0, "energy": 60.0}, max_value=100.0),
            vision=VisionObservation(visible_entities=[]),
        )

        assert obs.entity_id == "blacksmith_1"
        assert obs.current_simulation_time == 1000
        assert obs.status.position == (10, 10)
        assert obs.needs.needs["hunger"] == 50.0

    def test_observation_with_minimal_fields(self):
        """Should allow creating observation with only required fields"""
        obs = Observation(
            entity_id="test_npc",
            current_simulation_time=0,
            status=StatusObservation(position=(0, 0), movement_locked=False),
        )

        assert obs.entity_id == "test_npc"
        assert obs.needs is None
        assert obs.vision is None
        assert obs.conversations == []


class TestMindEvent:
    """Test MindEvent model and formatting"""

    def test_create_mind_event(self):
        """Should create MindEvent with event_type field"""
        event = MindEvent(
            timestamp=100,
            event_type=MindEventType.INTERACTION_BID_REJECTED,
            payload={"interaction_name": "sit", "reason": "Too far away"},
        )

        assert event.timestamp == 100
        assert event.event_type == MindEventType.INTERACTION_BID_REJECTED
        assert event.payload["interaction_name"] == "sit"
        assert event.payload["reason"] == "Too far away"

    def test_format_rejection_event_with_reason(self):
        """Should format rejection event with reason in natural language"""
        event = MindEvent(
            timestamp=100,
            event_type=MindEventType.INTERACTION_BID_REJECTED,
            payload={"interaction_name": "sit", "reason": "Too far away"},
        )

        formatted = str(event)
        assert "Interaction bid rejected: sit" in formatted
        assert "Too far away" in formatted

    def test_format_rejection_event_without_reason(self):
        """Should format rejection event without reason"""
        event = MindEvent(
            timestamp=100,
            event_type=MindEventType.INTERACTION_BID_REJECTED,
            payload={"interaction_name": "sit"},
        )

        formatted = str(event)
        assert formatted == "Interaction bid rejected: sit"

    def test_format_interaction_started_event(self):
        """Should format interaction started event"""
        event = MindEvent(
            timestamp=100,
            event_type=MindEventType.INTERACTION_STARTED,
            payload={"interaction_name": "conversation"},
        )

        formatted = str(event)
        assert formatted == "Interaction started: conversation"

    def test_format_interaction_finished_event(self):
        """Should format interaction finished event"""
        event = MindEvent(
            timestamp=100,
            event_type=MindEventType.INTERACTION_FINISHED,
            payload={"interaction_name": "conversation"},
        )

        formatted = str(event)
        assert formatted == "Interaction finished: conversation"

    def test_format_error_event(self):
        """Should format error event"""
        event = MindEvent(
            timestamp=100,
            event_type=MindEventType.ERROR,
            payload={"message": "Something went wrong"},
        )

        formatted = str(event)
        assert formatted == "Error: Something went wrong"

    def test_format_movement_completed_arrived(self):
        """Should format movement completion event when arrived"""
        event = MindEvent(
            timestamp=100,
            event_type=MindEventType.MOVEMENT_COMPLETED,
            payload={
                "status": "ARRIVED",
                "intended_destination": [10, 20],
                "actual_destination": [10, 20],
            },
        )

        formatted = str(event)
        assert formatted == "Arrived at (10, 20)"

    def test_format_movement_completed_stopped_short(self):
        """Should format movement completion event when stopped short"""
        event = MindEvent(
            timestamp=100,
            event_type=MindEventType.MOVEMENT_COMPLETED,
            payload={
                "status": "STOPPED_SHORT",
                "intended_destination": [10, 20],
                "actual_destination": [9, 20],
            },
        )

        formatted = str(event)
        assert formatted == "Moved to (9, 20), intended destination (10, 20) was blocked"

    def test_format_movement_completed_blocked(self):
        """Should format movement completion event when completely blocked"""
        event = MindEvent(
            timestamp=100,
            event_type=MindEventType.MOVEMENT_COMPLETED,
            payload={
                "status": "BLOCKED",
                "intended_destination": [10, 20],
                "actual_destination": [5, 5],
            },
        )

        formatted = str(event)
        assert formatted == "Could not move to (10, 20), no valid path"


class TestBidActionGeneration:
    """Test generation of bid response actions"""

    def test_generate_bid_response_actions(self):
        """Should generate a unified respond action for each pending bid"""
        from mind.cognitive_architecture.actions import ActionType

        obs = Observation(
            entity_id="test_npc",
            current_simulation_time=100,
            status=StatusObservation(position=(0, 0), movement_locked=False),
        )

        # Create pending bid
        bid_event = MindEvent(
            timestamp=100,
            event_type=MindEventType.INTERACTION_BID_RECEIVED,
            payload={
                "bid_id": "bid_456",
                "bidder_id": "bob_001",
                "bidder_name": "Bob",
                "interaction_name": "trade",
            },
        )
        pending_bids = {"bid_456": bid_event}

        # Get available actions
        actions = obs.get_available_actions(pending_incoming_bids=pending_bids)

        # Find bid response actions
        bid_actions = [a for a in actions if a.name == ActionType.RESPOND_TO_INTERACTION_BID]

        # Should have 1 unified action per bid (with accept boolean parameter)
        assert len(bid_actions) == 1

        # Check the unified respond action
        respond_action = bid_actions[0]
        assert "Bob" in respond_action.description
        assert "trade" in respond_action.description
        assert "bid_456" in respond_action.description
        assert "bid_id" in respond_action.parameters
        assert "accept" in respond_action.parameters
        assert "reason" in respond_action.parameters

    def test_no_bid_actions_when_no_pending_bids(self):
        """Should not generate bid response actions when no bids pending"""
        from mind.cognitive_architecture.actions import ActionType

        obs = Observation(
            entity_id="test_npc",
            current_simulation_time=100,
            status=StatusObservation(position=(0, 0), movement_locked=False),
        )

        # Get available actions without pending bids
        actions = obs.get_available_actions()

        # Should have no bid response actions
        bid_actions = [a for a in actions if a.name == ActionType.RESPOND_TO_INTERACTION_BID]
        assert len(bid_actions) == 0
