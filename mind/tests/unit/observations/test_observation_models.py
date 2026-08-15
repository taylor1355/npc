"""Unit tests for observation models"""

import pytest
from pydantic import ValidationError

from mind.cognitive_architecture.observations import (
    ArousalBand,
    ConversationMessage,
    ConversationObservation,
    EntityData,
    GoalDetail,
    GoalObservation,
    MindEvent,
    MindEventType,
    MoodObservation,
    NeedsObservation,
    Observation,
    RelationshipState,
    StatusObservation,
    ValenceBand,
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


class TestSubstrateGoal:
    """The `goal` key the simulation has always sent, and used to lose.

    Before this field was declared, ``Observation`` had six fields and pydantic's
    default ``extra="ignore"`` dropped ``goal`` on every cycle, so the
    substrate's active goal never reached a prompt. These pin that it survives
    validation and that its absence stays harmless.
    """

    def test_goal_key_survives_validation(self):
        """Should keep the goal payload the simulation sends, not discard it"""
        obs = Observation.model_validate(
            {
                "entity_id": "test_npc",
                "current_simulation_time": 100,
                "status": {"position": (0, 0), "movement_locked": False},
                "goal": {
                    "active_goal": {
                        "label": "Find something to eat",
                        "urgency": 1.21,
                        "drive_source": "hunger",
                        "template_id": "satisfy_hunger",
                    },
                    "candidate_count": 5,
                },
            }
        )

        assert obs.goal is not None
        assert obs.goal.active_goal is not None
        assert obs.goal.active_goal.label == "Find something to eat"
        assert obs.goal.active_goal.drive_source == "hunger"
        assert obs.goal.active_goal.template_id == "satisfy_hunger"
        assert obs.goal.candidate_count == 5

    def test_goal_absent_is_valid(self):
        """Should treat a goal-less observation as valid with goal None"""
        obs = Observation(
            entity_id="test_npc",
            current_simulation_time=100,
            status=StatusObservation(position=(0, 0), movement_locked=False),
        )

        assert obs.goal is None
        assert str(obs)  # rendering must not raise

    def test_goal_without_active_goal_is_valid(self):
        """Should accept the shape the sim sends when no goal is active"""
        obs = Observation.model_validate(
            {
                "entity_id": "test_npc",
                "current_simulation_time": 100,
                "goal": {"candidate_count": 0},
            }
        )

        assert obs.goal is not None
        assert obs.goal.active_goal is None

    def test_urgency_is_not_clamped_to_one(self):
        """Should carry the simulation's wider-than-unit urgency domain intact

        Effective urgency is the template curve scaled by preference alignment,
        so a well-aligned homeostatic goal legitimately exceeds 1.0. Bounding it
        here would reject real payloads.
        """
        detail = GoalDetail(label="Eat", urgency=1.29)

        assert detail.urgency == 1.29

    def test_goal_requires_a_label(self):
        """Should fail loudly on a malformed goal rather than substituting one"""
        with pytest.raises(ValidationError):
            GoalObservation.model_validate({"active_goal": {"urgency": 0.5}})


class TestMoodObservation:
    """Mood crosses as band tokens plus a display word.

    Structure fails loud (bands are a StrEnum), copy fails soft (the label is a
    free str). These two behaviours together are the whole contract.
    """

    def test_mood_survives_validation(self):
        obs = Observation.model_validate(
            {
                "entity_id": "test_npc",
                "current_simulation_time": 100,
                "mood": {
                    "valence": -0.42,
                    "arousal": 0.81,
                    "valence_band": "neg",
                    "arousal_band": "high",
                    "label": "stressed",
                    "valence_baseline": -0.05,
                    "arousal_baseline": 0.5,
                },
            }
        )

        assert obs.mood is not None
        assert obs.mood.valence_band is ValenceBand.NEG
        assert obs.mood.arousal_band is ArousalBand.HIGH
        assert obs.mood.label == "stressed"
        assert obs.mood.valence_baseline == -0.05

    def test_unknown_band_is_rejected(self):
        """Structure fails loud: a fourth band is a breaking change, not a nuance"""
        with pytest.raises(ValidationError):
            MoodObservation(
                valence=0.0,
                arousal=0.5,
                valence_band="lukewarm",
                arousal_band="mid",
                label="calm",
            )

    def test_unseen_label_is_accepted(self):
        """Copy fails soft: relabelling a mood word must not break decisions"""
        mood = MoodObservation(
            valence=0.0,
            arousal=0.5,
            valence_band=ValenceBand.MID,
            arousal_band=ArousalBand.MID,
            label="equanimous",
        )

        assert mood.label == "equanimous"

    def test_out_of_domain_mood_values_are_accepted(self):
        """A numeric overshoot must not collapse the cycle into the WAIT fallback

        The simulation's baseline-drift path integrates rate * elapsed without a
        clamp, so a long gap between decision cycles can push valence past its
        nominal domain. Rejecting that would silently stop the NPC acting; the
        bands stay correct regardless.
        """
        mood = MoodObservation(
            valence=-1.4,
            arousal=1.2,
            valence_band=ValenceBand.NEG,
            arousal_band=ArousalBand.HIGH,
            label="stressed",
        )

        assert mood.valence == -1.4

    def test_mood_absent_is_valid(self):
        obs = Observation(entity_id="test_npc", current_simulation_time=100)

        assert obs.mood is None
        assert str(obs)


class TestRelationshipEnrichment:
    """Relationships ride the visible entity they describe, never a second list"""

    def test_relationship_survives_validation(self):
        obs = Observation.model_validate(
            {
                "entity_id": "test_npc",
                "current_simulation_time": 100,
                "vision": {
                    "visible_entities": [
                        {
                            "entity_id": "alice_npc",
                            "display_name": "Alice",
                            "position": (1, 2),
                            "interactions": {},
                            "relationship": {
                                "familiarity": 0.62,
                                "sentiment": 0.31,
                                "interaction_count": 14,
                            },
                        }
                    ]
                },
            }
        )

        entity = obs.vision.visible_entities[0]
        assert entity.relationship is not None
        assert entity.relationship.familiarity == 0.62
        assert entity.relationship.sentiment == 0.31
        assert entity.relationship.interaction_count == 14

    def test_entity_without_relationship_is_a_stranger(self):
        entity = EntityData(entity_id="x", display_name="X", position=(0, 0))

        assert entity.relationship is None

    def test_out_of_domain_relationship_values_are_rejected(self):
        """Unlike mood, every registry write clamps these — so a violation is a bug"""
        with pytest.raises(ValidationError):
            RelationshipState(familiarity=1.4, sentiment=0.0)

        with pytest.raises(ValidationError):
            RelationshipState(familiarity=0.5, sentiment=-2.0)
