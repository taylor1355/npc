"""Realistic observation fixtures for integration testing

These fixtures mirror real Godot simulation observations that would be
sent to the Mind server via MCP.
"""

from mind.cognitive_architecture.observations import (
    ArousalBand,
    ConversationMessage,
    ConversationObservation,
    EntityData,
    GoalDetail,
    GoalObservation,
    GoalOption,
    GoalOptionSegment,
    GoalOptionStep,
    GoalStepAction,
    GoalStepFactors,
    GoalStepTarget,
    GoalSummary,
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
from mind.cognitive_architecture.working_memory import WorkingMemory
from mind.constants import DEFAULT_EMBEDDING_MODEL, DEFAULT_SMALL_MODEL
from mind.interfaces.mcp.models import MindConfig


def wire_property_spec(type_name: str, default, description: str) -> dict:
    """One entry of ``act_in_interaction_parameters``, verbatim from the wire.

    This IS the Godot ``PropertySpec.to_dict()`` shape — ``{"type", "default",
    "description"}`` and nothing else. Tests build parameter payloads through
    this helper so a simulation-side shape change breaks in one place rather
    than in every fixture that happened to guess right.
    """
    return {"type": type_name, "default": default, "description": description}


def wire_current_interaction(
    name: str,
    description: str = "",
    act_parameters: dict | None = None,
    needs_filled: list[str] | None = None,
    needs_drained: list[str] | None = None,
) -> dict:
    """A ``StatusObservation.current_interaction`` payload, verbatim from the wire.

    This IS the Godot ``Interaction.to_dict()`` contract: the key set below is
    exactly what crosses the boundary, and there is deliberately no
    ``interaction_name`` key — asking for one is what made the mind blind to
    every interaction's identity and parameters (NPC-1278). Fixtures that
    invented their own shape were why the bug survived a full test suite, so
    every test that needs a current interaction builds it here.
    """
    return {
        "name": name,
        "description": description or f"The {name} interaction",
        "needs_filled": needs_filled or [],
        "needs_drained": needs_drained or [],
        "need_rates": {},
        "act_in_interaction_parameters": act_parameters or {},
    }


def wire_conversation_interaction() -> dict:
    """The conversation interaction as the simulation actually advertises it.

    Mirrors ``conversation_interaction.gd``'s two declared act parameters.
    """
    return wire_current_interaction(
        name="conversation",
        description="Multi-party conversation",
        act_parameters={
            "message": wire_property_spec("string", "", "Message text to send in conversation"),
            "is_farewell": wire_property_spec(
                "bool",
                False,
                "Whether the speaker intends this message to end the conversation",
            ),
        },
        needs_filled=["social"],
    )


def wire_entity_data(*args, **kwargs) -> dict:
    """The EntityData wire shape, under the name a non-inventory caller wants.

    Identical to :func:`wire_inventory_item` because the simulation builds BOTH
    inventory items and visible entities with the same ``EntityData.to_dict()``.
    The alias exists so a vision fixture does not have to call something named
    "inventory_item" and make a reader stop to work out why.
    """
    return wire_inventory_item(*args, **kwargs)


def wire_inventory_item(
    entity_id: str,
    display_name: str,
    position: tuple[int, int] = (0, 0),
    interactions: dict | None = None,
) -> dict:
    """One entry of ``InventoryObservation.items``, verbatim from the wire.

    This IS the Godot ``EntityData.to_dict()`` contract
    (``src/minds/observations/entity_data.gd``): ``entity_id``,
    ``display_name``, ``position`` as a two-element LIST (not a tuple -- the
    simulation converts ``Vector2i`` for Python), ``interactions``, and
    ``current_interaction`` emitted as ``{}`` rather than omitted when idle.

    Three keys are deliberately absent because ``to_dict()`` does not emit them:
    ``relationship`` (``InventoryComponent.get_items_as_entity_data`` calls
    ``EntityData.from_entity_ids`` with ``include_relationships`` defaulting
    false, so a carried item never carries one), ``entity_type``, and
    ``distance_to_observer``.
    """
    return {
        "entity_id": entity_id,
        "display_name": display_name,
        "position": [position[0], position[1]],
        "interactions": interactions or {},
        "current_interaction": {},
    }


def wire_inventory_block(
    owner_id: str = "carrier_npc",
    capacity: int = 4,
    items: list[dict] | None = None,
) -> dict:
    """An ``Observation``'s ``inventory`` value, verbatim from the wire.

    This IS the Godot ``InventoryObservation.get_data()`` contract
    (``src/minds/observations/inventory_observation.gd``): exactly ``owner_id``,
    ``capacity``, ``used_slots``, ``items`` and nothing else. ``used_slots`` is
    the simulation's own ``items.size()``, carried rather than re-derived.

    Transcribed from simulation ``origin/main`` @ ``a2ac2f5a``. It is only as
    good as that transcription -- if ``get_data()`` changes, re-derive this
    from the source rather than trusting it.
    """
    return {
        "owner_id": owner_id,
        "capacity": capacity,
        "used_slots": len(items or []),
        "items": items or [],
    }


def wire_full_root_payload(simulation_time: int = 100) -> dict:
    """Every root key the simulation can put on the ``decide_action`` wire.

    ``CompositeObservation.get_data()`` builds ``{entity_id,
    current_simulation_time}`` plus one key per ``Observation.get_type()`` of
    every observation added in ``entity_controller.gd`` /
    ``npc_controller.gd::get_current_state_observation``. That resolves to
    exactly the eight keys below (verified against simulation ``origin/main``
    @ ``a2ac2f5a``).

    ``conversations`` is deliberately NOT here: it is a mind-side field lifted
    out of ``INTERACTION_OBSERVATION`` events by
    ``server.py::_extract_conversation_observations``, never a wire root key.

    Each nested block is transcribed from its own ``get_data()`` -- note
    ``needs.max_need_value`` and ``status.position`` as a list.
    """
    return {
        "entity_id": "carrier_npc",
        "current_simulation_time": simulation_time,
        "status": {
            "position": [12, 8],
            "movement_locked": False,
            "current_interaction": {},
            "activity_state": {"state_name": "idle"},
        },
        "needs": {
            "needs": {"hunger": 22.0, "energy": 41.0, "stimulation": 60.0, "social": 35.0},
            "max_need_value": 100.0,
        },
        "vision": {
            "visible_entities": [
                wire_entity_data("alice_npc", "Alice", (13, 8)),
            ]
        },
        "goal": {
            "contract_version": 1,
            "urgency_max": 1.3,
            "active_goal": {
                "template_id": "satisfy_hunger",
                "label": "Find something to eat",
                "urgency": 1.18,
                "drive_source": "hunger",
                "preference_alignment": 0.12,
                "age_minutes": 14,
                "interruption_threshold": 1.25,
            },
            "goals": [
                {
                    "template_id": "satisfy_hunger",
                    "label": "Find something to eat",
                    "urgency": 1.18,
                    "drive_source": "hunger",
                    "preference_alignment": 0.12,
                    "is_active": True,
                }
            ],
            "options": [],
            "option_total": 0,
        },
        "mood": {
            "valence": -0.42,
            "arousal": 0.81,
            "valence_band": "neg",
            "arousal_band": "high",
            "label": "stressed",
            "valence_baseline": -0.05,
            "arousal_baseline": 0.5,
        },
        "inventory": wire_inventory_block(
            owner_id="carrier_npc",
            capacity=4,
            items=[
                wire_inventory_item(
                    "apple_001",
                    "Ripe Apple",
                    (12, 8),
                    interactions={
                        "consume": {
                            "name": "consume",
                            "description": "Eat the apple",
                            "needs_filled": ["hunger"],
                            "needs_drained": [],
                        }
                    },
                ),
                wire_inventory_item("pebble_001", "Smooth Pebble", (12, 8)),
            ],
        ),
    }


def create_carrying_observation(simulation_time: int = 100) -> Observation:
    """An NPC carrying two items, parsed from the wire payload rather than built.

    Built through ``model_validate`` on purpose: constructing the model with
    Python keywords would bypass exactly the boundary NPC-1116 is about.
    """
    return Observation.model_validate(wire_full_root_payload(simulation_time))


def create_blacksmith_observation(simulation_time: int = 100) -> Observation:
    """Blacksmith NPC at forge with low energy, seeing tools and customers"""
    return Observation(
        entity_id="blacksmith_npc",
        current_simulation_time=simulation_time,
        status=StatusObservation(
            position=(15, 20), movement_locked=False, current_interaction={}, activity_state={}
        ),
        needs=NeedsObservation(
            needs={"hunger": 65.0, "energy": 30.0, "fun": 40.0, "hygiene": 70.0, "social": 55.0},
            max_value=100.0,
        ),
        vision=VisionObservation(
            visible_entities=[
                EntityData(
                    entity_id="forge_001",
                    display_name="Blacksmith Forge",
                    position=(15, 21),
                    interactions={
                        "work_at_forge": {
                            "name": "work_at_forge",
                            "description": "Work at the forge to create items",
                            "needs_filled": ["fun"],
                            "needs_drained": ["energy"],
                        }
                    },
                ),
                EntityData(
                    entity_id="anvil_001",
                    display_name="Iron Anvil",
                    position=(16, 20),
                    interactions={
                        "examine": {
                            "name": "examine",
                            "description": "Examine the anvil",
                            "needs_filled": [],
                            "needs_drained": [],
                        }
                    },
                ),
                EntityData(
                    entity_id="customer_npc_01",
                    display_name="Traveling Merchant",
                    position=(14, 19),
                    interactions={
                        "chat": {
                            "name": "chat",
                            "description": "Talk with the merchant",
                            "needs_filled": ["social", "fun"],
                            "needs_drained": [],
                        }
                    },
                ),
                EntityData(
                    entity_id="bed_001",
                    display_name="Simple Bed",
                    position=(10, 20),
                    interactions={
                        "sleep": {
                            "name": "sleep",
                            "description": "Rest and recover energy",
                            "needs_filled": ["energy"],
                            "needs_drained": [],
                        }
                    },
                ),
            ]
        ),
        conversations=[],
    )


def create_explorer_observation(simulation_time: int = 100) -> Observation:
    """Explorer NPC in wilderness, hungry, seeing food and shelter"""
    return Observation(
        entity_id="explorer_npc",
        current_simulation_time=simulation_time,
        status=StatusObservation(position=(5, 10), movement_locked=False),
        needs=NeedsObservation(
            needs={"hunger": 20.0, "energy": 60.0, "fun": 75.0, "hygiene": 45.0, "social": 30.0},
            max_value=100.0,
        ),
        vision=VisionObservation(
            visible_entities=[
                EntityData(
                    entity_id="berry_bush_01",
                    display_name="Berry Bush",
                    position=(6, 10),
                    interactions={
                        "gather_berries": {
                            "name": "gather_berries",
                            "description": "Gather berries for food",
                            "needs_filled": ["hunger"],
                            "needs_drained": [],
                        }
                    },
                ),
                EntityData(
                    entity_id="cave_entrance_01",
                    display_name="Cave Entrance",
                    position=(7, 12),
                    interactions={
                        "enter_cave": {
                            "name": "enter_cave",
                            "description": "Enter the cave for shelter",
                            "needs_filled": [],
                            "needs_drained": [],
                        },
                        "examine": {
                            "name": "examine",
                            "description": "Look at the cave entrance",
                            "needs_filled": ["fun"],
                            "needs_drained": [],
                        },
                    },
                ),
                EntityData(
                    entity_id="stream_01",
                    display_name="Clear Stream",
                    position=(5, 12),
                    interactions={
                        "drink_water": {
                            "name": "drink_water",
                            "description": "Drink fresh water",
                            "needs_filled": ["hygiene"],
                            "needs_drained": [],
                        }
                    },
                ),
            ]
        ),
        conversations=[],
    )


def create_conversation_observation(simulation_time: int = 100) -> Observation:
    """NPC engaged in active conversation with another character"""
    return Observation(
        entity_id="social_npc",
        current_simulation_time=simulation_time,
        status=StatusObservation(
            position=(20, 15),
            movement_locked=True,  # Locked during conversation
            current_interaction=wire_conversation_interaction(),
            activity_state={"state_name": "interacting"},
        ),
        needs=NeedsObservation(
            needs={"hunger": 70.0, "energy": 80.0, "fun": 85.0, "hygiene": 90.0, "social": 95.0}
        ),
        vision=VisionObservation(
            visible_entities=[
                EntityData(
                    entity_id="alice_npc",
                    display_name="Alice",
                    position=(20, 14),
                    interactions={
                        "continue_chat": {
                            "name": "continue_chat",
                            "description": "Continue the conversation",
                            "needs_filled": ["social", "fun"],
                            "needs_drained": [],
                        }
                    },
                )
            ]
        ),
        conversations=[
            ConversationObservation(
                interaction_id="chat_with_alice",
                interaction_name="casual_chat",
                participants=["social_npc", "alice_npc"],
                conversation_history=[
                    ConversationMessage(
                        speaker_id="alice_npc",
                        speaker_name="Alice",
                        message="Hello! How are you doing today?",
                        timestamp=simulation_time - 5,
                    ),
                    ConversationMessage(
                        speaker_id="social_npc",
                        speaker_name="Bob",
                        message="I'm doing well, thanks! Just finished some work.",
                        timestamp=simulation_time - 3,
                    ),
                    ConversationMessage(
                        speaker_id="alice_npc",
                        speaker_name="Alice",
                        message="That's great! What have you been working on?",
                        timestamp=simulation_time - 1,
                    ),
                ],
            )
        ],
    )


def create_idle_observation(simulation_time: int = 100) -> Observation:
    """NPC with no pressing needs, in open area with various options"""
    return Observation(
        entity_id="idle_npc",
        current_simulation_time=simulation_time,
        status=StatusObservation(position=(10, 10), movement_locked=False),
        needs=NeedsObservation(
            needs={"hunger": 80.0, "energy": 85.0, "fun": 60.0, "hygiene": 90.0, "social": 70.0}
        ),
        vision=VisionObservation(
            visible_entities=[
                EntityData(
                    entity_id="tree_01",
                    display_name="Oak Tree",
                    position=(11, 10),
                    interactions={
                        "examine": {
                            "name": "examine",
                            "description": "Look at the tree",
                            "needs_filled": ["fun"],
                            "needs_drained": [],
                        }
                    },
                ),
                EntityData(
                    entity_id="bench_01",
                    display_name="Wooden Bench",
                    position=(10, 11),
                    interactions={
                        "sit": {
                            "name": "sit",
                            "description": "Sit and rest",
                            "needs_filled": ["energy"],
                            "needs_drained": [],
                        }
                    },
                ),
            ]
        ),
        conversations=[],
    )


def create_emergency_observation(simulation_time: int = 100) -> Observation:
    """NPC in urgent situation - multiple critical needs"""
    return Observation(
        entity_id="distressed_npc",
        current_simulation_time=simulation_time,
        status=StatusObservation(position=(8, 8), movement_locked=False),
        needs=NeedsObservation(
            needs={
                "hunger": 5.0,  # Critical
                "energy": 10.0,  # Critical
                "fun": 20.0,
                "hygiene": 15.0,
                "social": 30.0,
            }
        ),
        vision=VisionObservation(
            visible_entities=[
                EntityData(
                    entity_id="food_stall_01",
                    display_name="Food Stall",
                    position=(9, 8),
                    interactions={
                        "buy_food": {
                            "name": "buy_food",
                            "description": "Purchase food",
                            "needs_filled": ["hunger"],
                            "needs_drained": [],
                        }
                    },
                ),
                EntityData(
                    entity_id="inn_01",
                    display_name="Cozy Inn",
                    position=(8, 10),
                    interactions={
                        "rest_at_inn": {
                            "name": "rest_at_inn",
                            "description": "Rest at the inn",
                            "needs_filled": ["energy", "hygiene"],
                            "needs_drained": [],
                        }
                    },
                ),
            ]
        ),
        conversations=[],
    )


def create_blacksmith_config() -> MindConfig:
    """Configuration for a blacksmith NPC mind (entity_id is a create_mind arg, not config)"""
    return MindConfig(
        traits=["diligent", "perfectionist", "proud", "helpful"],
        llm_model=DEFAULT_SMALL_MODEL,
        embedding_model=DEFAULT_EMBEDDING_MODEL,
        memory_storage_path="./tmp/test_blacksmith_db",
        initial_working_memory=WorkingMemory(
            situation_assessment="I am a blacksmith running my own forge",
            active_goals=["Maintain the forge", "Serve customers", "Perfect my craft"],
            emotional_state="Proud and focused on my work",
        ),
        initial_long_term_memories=[
            "I learned blacksmithing from my father",
            "I specialize in crafting ceremonial blades",
            "Maintaining the forge fire is crucial for quality work",
            "Customer satisfaction is important for my reputation",
        ],
    )


def create_explorer_config() -> MindConfig:
    """Configuration for an explorer NPC mind (entity_id is a create_mind arg, not config)"""
    return MindConfig(
        traits=["curious", "brave", "resourceful", "independent"],
        llm_model=DEFAULT_SMALL_MODEL,
        embedding_model=DEFAULT_EMBEDDING_MODEL,
        memory_storage_path="./tmp/test_explorer_db",
        initial_working_memory=WorkingMemory(
            situation_assessment="I am exploring unknown wilderness",
            active_goals=["Find food and shelter", "Map the area", "Survive"],
            emotional_state="Alert and cautiously optimistic",
        ),
        initial_long_term_memories=[
            "I have basic survival skills",
            "Berry bushes often indicate fresh water nearby",
            "Caves can provide shelter but may be dangerous",
        ],
    )


def create_risk_scenario_observation(simulation_time: int = 100) -> Observation:
    """NPC with choice between safe and risky options

    Designed to test personality-driven decision making:
    - Safe option: Rest at camp (low reward, no risk)
    - Risky option: Explore mysterious cave (high reward, dangerous)
    """
    return Observation(
        entity_id="adventurer_npc",
        current_simulation_time=simulation_time,
        status=StatusObservation(position=(10, 10), movement_locked=False),
        needs=NeedsObservation(
            needs={"hunger": 70.0, "energy": 70.0, "fun": 50.0, "hygiene": 80.0, "social": 60.0},
            max_value=100.0,
        ),
        vision=VisionObservation(
            visible_entities=[
                EntityData(
                    entity_id="camp_001",
                    display_name="Safe Camp",
                    position=(9, 10),
                    interactions={
                        "rest_at_camp": {
                            "name": "rest_at_camp",
                            "description": "Rest safely at camp - boring but safe",
                            "needs_filled": ["energy"],
                            "needs_drained": [],
                        }
                    },
                ),
                EntityData(
                    entity_id="mysterious_cave_001",
                    display_name="Mysterious Dark Cave",
                    position=(11, 10),
                    interactions={
                        "explore_cave": {
                            "name": "explore_cave",
                            "description": "Explore the mysterious cave - exciting but dangerous, strange sounds echo from within",
                            "needs_filled": ["fun"],
                            "needs_drained": ["energy"],
                        }
                    },
                ),
            ]
        ),
        conversations=[],
    )


def create_enriched_observation(simulation_time: int = 100) -> Observation:
    """The enriched arm of the observation A/B.

    Every other fixture in this module is deliberately left unenriched so it
    keeps acting as a control-arm regression: the fields added by observation
    enrichment are all optional, and an observation that omits them must render
    exactly as it did before they existed.
    """
    return Observation(
        entity_id="enriched_npc",
        current_simulation_time=simulation_time,
        status=StatusObservation(position=(12, 8), movement_locked=False),
        needs=NeedsObservation(
            needs={"hunger": 22.0, "energy": 41.0, "stimulation": 60.0, "social": 35.0},
            max_value=100.0,
        ),
        goal=GoalObservation(
            urgency_max=1.3,
            active_goal=GoalDetail(
                label="Find something to eat",
                urgency=1.18,
                drive_source="hunger",
                template_id="satisfy_hunger",
                preference_alignment=0.12,
                age_minutes=14,
                interruption_threshold=1.25,
            ),
            goals=[
                GoalSummary(
                    template_id="satisfy_hunger",
                    label="Find something to eat",
                    urgency=1.18,
                    drive_source="hunger",
                    preference_alignment=0.12,
                    is_active=True,
                ),
                GoalSummary(
                    template_id="seek_social_interaction",
                    label="Find company",
                    urgency=0.34,
                    drive_source="social",
                    preference_alignment=-0.03,
                ),
            ],
            options=[
                GoalOption(
                    option_id="satisfy_hunger:0",
                    description="Talk with Alice about food",
                    score=0.68,
                    segments=[
                        GoalOptionSegment(
                            goal_template_id="satisfy_hunger",
                            goal_label="Find something to eat",
                            steps=[
                                GoalOptionStep(
                                    action=GoalStepAction(
                                        name="INTERACT_WITH",
                                        parameters={
                                            "entity_id": "alice_npc",
                                            "interaction_name": "conversation",
                                        },
                                    ),
                                    target=GoalStepTarget(
                                        interaction_name="conversation",
                                        entity_id="alice_npc",
                                    ),
                                    factors=GoalStepFactors(
                                        urgency=1.18,
                                        utility=0.68,
                                        responsiveness=0.85,
                                        policy_modifier=1.0,
                                    ),
                                    step_score=0.68,
                                )
                            ],
                        )
                    ],
                )
            ],
            option_total=5,
        ),
        mood=MoodObservation(
            valence=-0.42,
            arousal=0.81,
            valence_band=ValenceBand.NEG,
            arousal_band=ArousalBand.HIGH,
            label="stressed",
            valence_baseline=-0.05,
            arousal_baseline=0.5,
        ),
        vision=VisionObservation(
            visible_entities=[
                EntityData(
                    entity_id="alice_npc",
                    display_name="Alice",
                    position=(13, 8),
                    interactions={
                        "conversation": {
                            "name": "conversation",
                            "description": "Talk with Alice",
                            "needs_filled": ["social"],
                            "needs_drained": [],
                        }
                    },
                    relationship=RelationshipState(
                        familiarity=0.62, sentiment=0.31, interaction_count=14
                    ),
                ),
                EntityData(
                    entity_id="stranger_npc",
                    display_name="Unfamiliar Traveller",
                    position=(10, 9),
                    interactions={},
                ),
            ]
        ),
        conversations=[],
    )


# --- Event-buffer fixtures (NPC-1335) ---------------------------------------
#
# The observation fixtures above return an Observation and no MindEvents at
# all, which is why any harness built from them measures a decision cycle with
# an EMPTY recent-events buffer — and therefore measures nothing about how that
# buffer is rendered. These give the buffer a fixed, committed shape so a
# rendering cost is reproducible and comparable across commits.
#
# Payload keys are the Godot wire keys verbatim (the `get_data()` output of
# InteractionBidObservation, InteractionBidRejectedObservation,
# MovementObservation, ConversationObservation), shipped unchanged as
# MindEvent.payload by mcp_mind_client.gd.


def create_movement_events(start_time: int = 100) -> list[MindEvent]:
    """A short buffer with no conversation: the cheapest realistic shape."""
    return [
        MindEvent(
            timestamp=start_time,
            event_type=MindEventType.ACTION_CHOSEN,
            payload={"action": "move_to", "parameters": {"x": 12, "y": 8}},
        ),
        MindEvent(
            timestamp=start_time + 2,
            event_type=MindEventType.MOVEMENT_COMPLETED,
            payload={
                "intended_destination": [12, 8],
                "actual_destination": [12, 8],
                "status": "ARRIVED",
            },
        ),
        MindEvent(
            timestamp=start_time + 3,
            event_type=MindEventType.INTERACTION_BID_PENDING,
            payload={
                "interaction_name": "eat",
                "bid_type": 0,
                "bid_id": "bid_a1b2c3d4",
                "bidder_id": "npc_alice",
                "provider_id": "item_apple_3",
                "timestamp": float(start_time + 3),
                "force": False,
            },
        ),
    ]


def create_social_events(start_time: int = 100) -> list[MindEvent]:
    """A fuller buffer covering the bid lifecycle, still with no conversation."""
    events = create_movement_events(start_time)
    events.extend(
        [
            MindEvent(
                timestamp=start_time + 4,
                event_type=MindEventType.INTERACTION_BID_REJECTED,
                payload={
                    "interaction_name": "eat",
                    "bid_type": 0,
                    "reason": "Already being used by someone else",
                    "target_id": "item_apple_3",
                },
            ),
            MindEvent(
                timestamp=start_time + 5,
                event_type=MindEventType.INTERACTION_BID_RECEIVED,
                payload={
                    "interaction_name": "conversation",
                    "bid_type": 0,
                    "bid_id": "bid_e5f6a7b8",
                    "bidder_id": "npc_carol",
                    "provider_id": "npc_alice",
                    "countered_bid_id": "bid_11223344",
                    "target_interaction_id": "interaction_77",
                    "existing_participants": ["npc_bob", "npc_dave"],
                    "counter_reason": "Already talking with others",
                    "timestamp": float(start_time + 5),
                    "force": False,
                },
            ),
            MindEvent(
                timestamp=start_time + 6,
                event_type=MindEventType.INTERACTION_STARTED,
                payload={"interaction_name": "conversation", "update_type": "started"},
            ),
            MindEvent(
                timestamp=start_time + 9,
                event_type=MindEventType.INTERACTION_FINISHED,
                payload={"interaction_name": "conversation", "update_type": "finished"},
            ),
        ]
    )
    return events


def create_conversation_events(start_time: int = 100) -> list[MindEvent]:
    """The expensive shape: a buffer carrying an INTERACTION_OBSERVATION.

    That arm renders its raw payload deliberately — it is the only channel
    carrying conversation content to the LLM (NPC-1298) — so it is the one
    event type prose rendering barely shrinks. A buffer's conversation-event
    count therefore dominates any aggregate saving measured over it.
    """
    events = create_social_events(start_time)
    # Inserted before the INTERACTION_FINISHED entry: the buffer arrives in the
    # simulation's arrival order, and a conversation update necessarily precedes
    # the end of the conversation it describes.
    events.insert(
        -1,
        MindEvent(
            timestamp=start_time + 8,
            event_type=MindEventType.INTERACTION_OBSERVATION,
            payload={
                "interaction_name": "conversation",
                "interaction_id": "interaction_77",
                "initiator_id": "npc_bob",
                "participants": ["npc_alice", "npc_bob", "npc_dave"],
                "conversation_history": [
                    {
                        "speaker_id": "npc_bob",
                        "message": "Have you seen the smith today? I need a blade repaired.",
                        "timestamp": float(start_time + 7),
                    },
                    {
                        "speaker_id": "npc_alice",
                        "message": "He was at the forge this morning, hammering away.",
                        "timestamp": float(start_time + 7),
                    },
                    {
                        "speaker_id": "npc_dave",
                        "message": "I would not bother him before noon, he gets short with people.",
                        "timestamp": float(start_time + 8),
                    },
                ],
                "total_message_count": 3,
            },
        ),
    )
    return events
