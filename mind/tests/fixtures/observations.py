"""Realistic observation fixtures for integration testing

These fixtures mirror real Godot simulation observations that would be
sent to the Mind server via MCP.
"""

from mind.cognitive_architecture.models import (
    ConversationMessage,
    ConversationObservation,
    EntityData,
    NeedsObservation,
    Observation,
    StatusObservation,
    VisionObservation,
)
from mind.cognitive_architecture.nodes.cognitive_update.models import WorkingMemory
from mind.constants import DEFAULT_EMBEDDING_MODEL, DEFAULT_SMALL_MODEL
from mind.interfaces.mcp.models import MindConfig


def create_blacksmith_observation(simulation_time: int = 100) -> Observation:
    """Blacksmith NPC at forge with low energy, seeing tools and customers"""
    return Observation(
        entity_id="blacksmith_npc",
        current_simulation_time=simulation_time,
        status=StatusObservation(
            position=(15, 20), movement_locked=False, current_interaction={}, controller_state={}
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
            current_interaction={"interaction_id": "chat_with_alice", "type": "conversation"},
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
    """Configuration for a blacksmith NPC mind"""
    return MindConfig(
        entity_id="blacksmith_npc",
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
    """Configuration for an explorer NPC mind"""
    return MindConfig(
        entity_id="explorer_npc",
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
