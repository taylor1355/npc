"""Unit tests for observation models"""

import logging

import pytest
from pydantic import ValidationError

from mind.cognitive_architecture.observations import (
    ArousalBand,
    ConversationMessage,
    ConversationObservation,
    EntityData,
    GoalDetail,
    GoalObservation,
    InventoryObservation,
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

    # --- NPC-1335: __str__ is now the prompt-path renderer for the event
    # buffer, so every arm must be both informative and total. The tests below
    # encode the per-event-type fidelity audit that switch demanded.

    @pytest.mark.parametrize("event_type", list(MindEventType))
    def test_every_event_type_has_an_informative_arm(self, event_type):
        """Every MindEventType renders something the LLM can act on.

        The buffer is rendered straight into the reflection prompt, so an
        unhandled member would put a content-free line in front of the model.
        This passes today only because all members are covered; its job is to
        go red the day a twelfth one is added without an arm.
        """
        event = MindEvent(timestamp=100, event_type=event_type, payload={"interaction_name": "sit"})

        formatted = str(event)
        assert formatted.strip()
        # NOT a startswith check against the retired "Unknown event type"
        # literal: that string no longer appears anywhere in __str__, so the
        # assertion could not fail and guarded nothing. What must hold is that
        # the arm SAYS something specific -- the event type named in prose or
        # at minimum echoed with its payload, never a bare class repr.
        assert "MindEvent(" not in formatted
        assert formatted.strip() != ""
        assert formatted.strip() != str(event_type)

    def test_bid_received_renders_the_counter_offer(self):
        """A counter-offer must survive into the prompt.

        The counter fields reach the mind on the wire and are rendered by the
        simulation's own ``InteractionBidObservation.format_for_npc``, but no
        other path carries them to the LLM — dropping them here loses them
        outright.
        """
        event = MindEvent(
            timestamp=103,
            event_type=MindEventType.INTERACTION_BID_RECEIVED,
            payload={
                "interaction_name": "conversation",
                "bid_type": 0,
                "bid_id": "bid_e5f6a7b8",
                "bidder_id": "npc_carol",
                "provider_id": "npc_alice",
                "countered_bid_id": "bid_00112233",
                "target_interaction_id": "interaction_9",
                "existing_participants": ["npc_alice", "npc_bob"],
                "counter_reason": "already talking to someone",
                "timestamp": 103.0,
                "force": False,
            },
        )

        formatted = str(event)
        assert "Interaction bid received: conversation" in formatted
        assert "counter-offer" in formatted
        assert "npc_alice" in formatted
        assert "npc_bob" in formatted
        assert "already talking to someone" in formatted

    def test_bid_received_names_the_bidder_and_bid_id(self):
        """Who bid, and which bid — a response action needs the id to target."""
        event = MindEvent(
            timestamp=103,
            event_type=MindEventType.INTERACTION_BID_RECEIVED,
            payload={
                "interaction_name": "conversation",
                "bid_id": "bid_e5f6a7b8",
                "bidder_id": "npc_carol",
            },
        )

        formatted = str(event)
        assert "npc_carol" in formatted
        assert "bid_e5f6a7b8" in formatted

    def test_bid_pending_and_canceled_name_their_bid(self):
        """With two bids in flight, the interaction name cannot disambiguate."""
        for event_type, verb in (
            (MindEventType.INTERACTION_BID_PENDING, "pending"),
            (MindEventType.INTERACTION_BID_CANCELED, "canceled"),
        ):
            event = MindEvent(
                timestamp=103,
                event_type=event_type,
                payload={
                    "interaction_name": "conversation",
                    "bid_id": "bid_e5f6a7b8",
                    "bidder_id": "npc_carol",
                },
            )

            formatted = str(event)
            assert f"Interaction bid {verb}: conversation" in formatted
            assert "bid_e5f6a7b8" in formatted

    def test_bid_rejected_names_the_rejecting_entity(self):
        """``target_id`` is who refused — needed to avoid re-bidding at them."""
        event = MindEvent(
            timestamp=100,
            event_type=MindEventType.INTERACTION_BID_REJECTED,
            payload={
                "interaction_name": "sit",
                "reason": "Too far away",
                "target_id": "chair_7",
            },
        )

        formatted = str(event)
        assert "Interaction bid rejected: sit" in formatted
        assert "chair_7" in formatted
        assert "Too far away" in formatted

    @pytest.mark.parametrize(
        "payload",
        [
            {"status": "ARRIVED"},
            {"status": "ARRIVED", "actual_destination": None},
            {"status": "ARRIVED", "actual_destination": [4]},
            {"status": "STOPPED_SHORT", "actual_destination": [4, 5]},
            {"status": "BLOCKED"},
            {},
        ],
    )
    def test_movement_completed_with_missing_destination_does_not_raise(self, payload):
        """A malformed movement payload degrades; it must never raise.

        ``payload`` is an unvalidated dict, and this renders inside the
        argument expression that builds the reflection prompt — upstream of
        ``call_llm`` and therefore of its salvage fallback. An exception here
        costs the whole decision cycle and its telemetry (the NPC-1195 class).
        """
        event = MindEvent(
            timestamp=100, event_type=MindEventType.MOVEMENT_COMPLETED, payload=payload
        )

        formatted = str(event)
        assert formatted.strip()

    def test_unknown_event_type_still_carries_its_payload(self):
        """A future enum member degrades to a repr rather than to nothing.

        ``model_construct`` bypasses enum validation to stand in for a member
        the simulation ships before this package learns about it.
        """
        event = MindEvent.model_construct(
            timestamp=100,
            event_type="INTERACTION_ESCALATED",
            payload={"interaction_name": "duel"},
        )

        formatted = str(event)
        assert "INTERACTION_ESCALATED" in formatted
        assert "duel" in formatted

    def test_interaction_observation_still_carries_conversation_content(self):
        """Pin, not a preference: this arm is the sole speech channel.

        ``Observation.conversations`` is never populated in production and
        ``PipelineState.conversation_histories`` is rendered by no node, so an
        INTERACTION_OBSERVATION event's raw payload is the only way anything
        anyone said reaches the LLM. Compacting this arm would silently blind
        every NPC to speech. NPC-1298 owns closing that gap; until it does,
        this test must go red for anyone who tidies the arm up.
        """
        event = MindEvent(
            timestamp=100,
            event_type=MindEventType.INTERACTION_OBSERVATION,
            payload={
                "interaction_name": "conversation",
                "participants": ["npc_alice", "npc_bob"],
                "conversation_history": [
                    {"speaker_id": "npc_bob", "message": "Have you seen the smith today?"}
                ],
                "total_message_count": 1,
            },
        )

        formatted = str(event)
        assert "Have you seen the smith today?" in formatted
        assert "npc_bob" in formatted


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


# The exact sample payload from the goal-block wire contract (sim repo,
# docs/reference/minds/observations.md "Goal block wire contract"; NPC-1321).
# Parsing THIS payload verbatim is the contract test — do not "tidy" its values.
GOAL_BLOCK_CONTRACT_SAMPLE = {
    "contract_version": 1,
    "urgency_max": 1.3,
    "active_goal": {
        "template_id": "satisfy_hunger",
        "label": "Find food",
        "urgency": 0.8734,
        "drive_source": "hunger",
        "preference_alignment": 0.1204,
        "age_minutes": 14,
        "interruption_threshold": 1.1354,
    },
    "goals": [
        {
            "template_id": "satisfy_hunger",
            "label": "Find food",
            "urgency": 0.8734,
            "drive_source": "hunger",
            "preference_alignment": 0.1204,
            "is_active": True,
        },
        {
            "template_id": "seek_social_interaction",
            "label": "Find company",
            "urgency": 0.3410,
            "drive_source": "social",
            "preference_alignment": -0.0312,
            "is_active": False,
        },
        {
            "template_id": "explore_area",
            "label": "Explore the area",
            "urgency": 0.0500,
            "drive_source": "stimulation",
            "preference_alignment": 0.0,
            "is_active": False,
        },
    ],
    "options": [
        {
            "option_id": "satisfy_hunger:0",
            "description": "Apple (consume, 0 away)",
            "score": 0.6756,
            "segments": [
                {
                    "goal_template_id": "satisfy_hunger",
                    "goal_label": "Find food",
                    "steps": [
                        {
                            "action": {
                                "name": "INTERACT_WITH",
                                "parameters": {
                                    "entity_id": "apple_01",
                                    "interaction_name": "consume",
                                },
                            },
                            "target": {
                                "interaction_name": "consume",
                                "entity_id": "apple_01",
                            },
                            "factors": {
                                "urgency": 0.8734,
                                "utility": 0.9100,
                                "responsiveness": 0.8500,
                                "policy_modifier": 1.0,
                            },
                            "step_score": 0.6756,
                        }
                    ],
                }
            ],
        },
        {
            "option_id": "explore_area:1",
            "description": "Wander and explore",
            "score": 0.0050,
            "segments": [
                {
                    "goal_template_id": "explore_area",
                    "goal_label": "Explore the area",
                    "steps": [
                        {
                            "action": {"name": "WANDER", "parameters": {}},
                            "target": None,
                            "factors": {
                                "urgency": 0.0500,
                                "utility": 0.1000,
                                "responsiveness": 1.0,
                                "policy_modifier": 1.0,
                            },
                            "step_score": 0.0050,
                        }
                    ],
                }
            ],
        },
    ],
    "option_total": 14,
}


class TestSubstrateGoal:
    """The plan-shaped goal block: contract v1 parse posture.

    ``extra="forbid"`` on the goal models is the block's whole point — a key
    the sim emits that these models do not declare is contract drift that must
    fail loud, not be silently dropped (the pre-declaration era lost the entire
    block that way).
    """

    def test_full_contract_sample_payload_parses(self):
        """The wire contract's own sample payload must validate verbatim"""
        obs = Observation.model_validate(
            {
                "entity_id": "test_npc",
                "current_simulation_time": 100,
                "status": {"position": (0, 0), "movement_locked": False},
                "goal": GOAL_BLOCK_CONTRACT_SAMPLE,
            }
        )

        goal = obs.goal
        assert goal is not None
        assert goal.contract_version == 1
        assert goal.urgency_max == 1.3
        assert goal.option_total == 14

        assert goal.active_goal is not None
        assert goal.active_goal.label == "Find food"
        assert goal.active_goal.template_id == "satisfy_hunger"
        assert goal.active_goal.preference_alignment == 0.1204
        assert goal.active_goal.age_minutes == 14
        assert goal.active_goal.interruption_threshold == 1.1354

        assert [g.template_id for g in goal.goals] == [
            "satisfy_hunger",
            "seek_social_interaction",
            "explore_area",
        ]
        assert [g.is_active for g in goal.goals] == [True, False, False]

        assert len(goal.options) == 2
        first = goal.options[0]
        assert first.option_id == "satisfy_hunger:0"
        assert first.score == 0.6756
        assert first.confidence is None  # reserved key, absent at tier 0
        step = first.segments[0].steps[0]
        assert step.action.name == "INTERACT_WITH"
        assert step.action.parameters["entity_id"] == "apple_01"
        assert step.target is not None
        assert step.target.entity_id == "apple_01"
        assert step.factors.responsiveness == 0.85
        assert step.step_score == 0.6756

        # The wander escape hatch: null target, identity responsiveness
        wander_step = goal.options[1].segments[0].steps[0]
        assert wander_step.target is None
        assert wander_step.factors.responsiveness == 1.0

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
                "goal": {
                    "contract_version": 1,
                    "urgency_max": 1.3,
                    "goals": [],
                    "options": [],
                    "option_total": 0,
                },
            }
        )

        assert obs.goal is not None
        assert obs.goal.active_goal is None
        assert obs.goal.options == []

    def test_undeclared_root_key_is_rejected(self):
        """extra="forbid" working: the retired wire key is the probe.

        ``candidate_count`` stopped being emitted when the plan-shaped block
        shipped; a payload still carrying it is a version-skewed sim, and that
        must fail loud rather than silently drop the key. (Red-first verified:
        this test fails under pydantic's default ``extra="ignore"``.)
        """
        with pytest.raises(ValidationError):
            GoalObservation.model_validate({"candidate_count": 5})

    def test_undeclared_nested_key_is_rejected(self):
        """forbid reaches every level of the block, not just the root"""
        option = {
            "option_id": "satisfy_hunger:0",
            "description": "Apple",
            "score": 0.5,
            "segments": [],
            "not_in_the_contract": True,
        }
        with pytest.raises(ValidationError):
            GoalObservation.model_validate({"options": [option]})

    def test_unknown_contract_version_warns_and_degrades(self, caplog):
        """A future version must degrade (warn + best-effort), never raise.

        The probe payload carries both an unknown version AND a novel root key,
        because that is what a real future version looks like — the shed-and-
        parse path has to survive the key, and the warning must name both the
        received version and the known set.
        """
        payload = dict(GOAL_BLOCK_CONTRACT_SAMPLE)
        payload["contract_version"] = 2
        payload["novel_v2_field"] = {"anything": True}

        with caplog.at_level(logging.WARNING):
            goal = GoalObservation.model_validate(payload)

        assert goal.contract_version == 2
        assert goal.active_goal is not None  # declared fields still parsed
        warning = next(r for r in caplog.records if "contract_version" in r.getMessage())
        assert "2" in warning.getMessage()
        assert "1" in warning.getMessage()

    def test_known_version_does_not_warn(self, caplog):
        with caplog.at_level(logging.WARNING):
            GoalObservation.model_validate(GOAL_BLOCK_CONTRACT_SAMPLE)

        assert not [r for r in caplog.records if "contract_version" in r.getMessage()]

    def test_general_plan_shape_parses(self):
        """Tier 0 sends 1 segment x 1 step, but the contract's forward promise
        is that planner tiers grow both under the same version — so the parser
        must already accept a multi-segment, multi-step option."""
        step = {
            "action": {"name": "MOVE_TO", "parameters": {"destination": [3, 4]}},
            "target": None,
            "factors": {
                "urgency": 0.5,
                "utility": 0.6,
                "responsiveness": 1.0,
                "policy_modifier": 1.0,
            },
            "step_score": 0.3,
        }
        option = {
            "option_id": "planner:0",
            "description": "eat apple (hunger) then chat (social)",
            "score": 0.9,
            "segments": [
                {
                    "goal_template_id": "satisfy_hunger",
                    "goal_label": "Find food",
                    "steps": [step, dict(step)],
                },
                {
                    "goal_template_id": "seek_social_interaction",
                    "goal_label": "Find company",
                    "steps": [dict(step)],
                },
            ],
        }

        goal = GoalObservation.model_validate({"options": [option], "option_total": 1})

        parsed = goal.options[0]
        assert len(parsed.segments) == 2
        assert [len(seg.steps) for seg in parsed.segments] == [2, 1]

    def test_reserved_keys_parse_when_present(self):
        """``confidence`` and ``segments[].rationale`` are reserved by the
        contract: never sent at tier 0, but their additive arrival (a planner
        emitting them) must parse under the same version with no model change."""
        option = {
            "option_id": "planner:0",
            "description": "Planned chain",
            "score": 0.8,
            "confidence": 0.72,
            "segments": [
                {
                    "goal_template_id": "satisfy_hunger",
                    "goal_label": "Find food",
                    "steps": [],
                    "rationale": "Food first; company keeps.",
                }
            ],
        }

        goal = GoalObservation.model_validate({"options": [option], "option_total": 1})

        assert goal.options[0].confidence == 0.72
        assert goal.options[0].segments[0].rationale == "Food first; company keeps."

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


class TestInventoryObservation:
    """The ``inventory`` root key survives the boundary (NPC-1116).

    Before this model existed the whole block was discarded by pydantic's
    default ``extra="ignore"`` -- silently, on every decision cycle, for every
    NPC with an InventoryComponent.
    """

    def test_inventory_survives_validation(self):
        """Should parse the wire inventory block into a typed model"""
        from tests.fixtures.observations import wire_full_root_payload

        obs = Observation.model_validate(wire_full_root_payload())

        assert obs.inventory is not None
        assert obs.inventory.owner_id == "carrier_npc"
        assert obs.inventory.capacity == 4
        assert obs.inventory.used_slots == 2
        assert [item.entity_id for item in obs.inventory.items] == ["apple_001", "pebble_001"]
        assert obs.inventory.items[0].display_name == "Ripe Apple"
        assert "consume" in obs.inventory.items[0].interactions

    def test_used_slots_is_carried_not_derived(self):
        """Should report the simulation's own count, not ``len(items)``"""
        from tests.fixtures.observations import wire_inventory_block, wire_inventory_item

        block = wire_inventory_block(items=[wire_inventory_item("apple_001", "Ripe Apple")])
        block["used_slots"] = 3  # a paged/partial items list, as a future wire could send

        inv = InventoryObservation.model_validate(block)

        assert inv.used_slots == 3
        assert len(inv.items) == 1

    def test_inventory_block_forbids_extras(self):
        """Should reject an undeclared key inside the inventory block"""
        from tests.fixtures.observations import wire_inventory_block

        block = wire_inventory_block()
        block["weight_kg"] = 3

        with pytest.raises(ValidationError):
            InventoryObservation.model_validate(block)

    def test_idle_inventory_item_has_no_interaction(self):
        """Should map the simulation's ``{}`` idle sentinel onto None"""
        from tests.fixtures.observations import wire_inventory_block, wire_inventory_item

        inv = InventoryObservation.model_validate(
            wire_inventory_block(items=[wire_inventory_item("apple_001", "Ripe Apple")])
        )

        assert inv.items[0].current_interaction is None

    def test_empty_inventory_is_valid(self):
        """Should accept a carrier holding nothing"""
        from tests.fixtures.observations import wire_inventory_block

        inv = InventoryObservation.model_validate(wire_inventory_block(items=[]))

        assert inv.items == []
        assert inv.used_slots == 0


class TestNeedsCeilingWireKey:
    """``max_need_value`` is the wire spelling of ``max_value`` (NPC-1116).

    The mismatch was correct only by coincidence -- both spellings resolve to
    100.0 today -- so nothing failed while the real value was being dropped.
    """

    def test_max_need_value_wire_key_lands(self):
        """Should take the ceiling from the simulation's spelling"""
        needs = NeedsObservation.model_validate({"needs": {"hunger": 1.0}, "max_need_value": 250.0})

        assert needs.max_value == 250.0

    def test_max_value_field_name_still_accepted(self):
        """Should still accept the field name so ``model_dump()`` round-trips"""
        original = NeedsObservation(needs={"hunger": 1.0}, max_value=42.0)

        assert NeedsObservation.model_validate(original.model_dump()).max_value == 42.0


class TestObservationRootForbid:
    """An undeclared root key raises instead of vanishing (NPC-1116).

    The wire root key set is mechanically ``{entity_id,
    current_simulation_time}`` plus one key per ``Observation.get_type()``, so
    a key this model does not declare is contract drift rather than noise. The
    trade is deliberate and one-directional: the failure is loud (an error
    response, logged at ERROR simulation-side, naming the key) but total for
    MCP NPCs, which is why every emitted root key must be declared here BEFORE
    the simulation-side emission merges.
    """

    def test_undeclared_root_key_is_rejected_by_observation(self):
        """Should raise on a root key no field declares"""
        payload = {
            "entity_id": "x",
            "current_simulation_time": 1,
            # W4's key. It exists nowhere in the simulation tree today, so this
            # test doubles as the proof that the cross-repo ordering gate is
            # still free to establish: the mind must declare it first.
            "shared_place": {"place_id": "tavern"},
        }

        with pytest.raises(ValidationError):
            Observation.model_validate(payload)

    def test_root_forbid_is_falsifiable(self):
        """Should keep forbid set, so a future merge cannot silently revert it"""
        assert Observation.model_config["extra"] == "forbid"

    def test_full_wire_payload_parses_under_forbid(self):
        """Should accept every root key the simulation actually emits.

        The single test that would have caught a missed key before it reached a
        live NPC. It is only as good as ``wire_full_root_payload``'s
        transcription -- re-derive that from the simulation's ``get_data()``
        rather than trusting it if the producers change.
        """
        from tests.fixtures.observations import wire_full_root_payload

        obs = Observation.model_validate(wire_full_root_payload())

        assert obs.entity_id == "carrier_npc"
        assert obs.status is not None
        assert obs.needs is not None
        assert obs.vision is not None
        assert obs.goal is not None
        assert obs.mood is not None
        assert obs.inventory is not None

    def test_model_dump_round_trips_under_forbid(self):
        """Should re-validate its own dump -- integration tests rely on this"""
        from tests.fixtures.observations import create_carrying_observation

        original = create_carrying_observation()

        assert Observation.model_validate(original.model_dump()).inventory is not None
