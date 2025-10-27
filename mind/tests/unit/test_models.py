"""Unit tests for cognitive architecture models"""

import pytest
from pydantic import ValidationError

from mind.cognitive_architecture.models import (
    Action,
    ConversationMessage,
    ConversationObservation,
    EntityData,
    Memory,
    NeedsObservation,
    Observation,
    StatusObservation,
    VisionObservation,
)
from mind.cognitive_architecture.nodes.cognitive_update.models import (
    NewMemory,
    WorkingMemory,
)


class TestMemoryModel:
    """Test Memory model validation and behavior"""

    def test_create_basic_memory(self):
        """Should create memory with minimal required fields"""
        memory = Memory(id="mem_1", content="Test memory", importance=5.0)

        assert memory.id == "mem_1"
        assert memory.content == "Test memory"
        assert memory.importance == 5.0
        assert memory.timestamp is None
        assert memory.location is None

    def test_create_memory_with_metadata(self):
        """Should create memory with full metadata"""
        memory = Memory(
            id="mem_1",
            content="Test memory",
            importance=7.0,
            timestamp=100,
            location=(10, 20),
        )

        assert memory.timestamp == 100
        assert memory.location == (10, 20)

    def test_importance_range_validation(self):
        """Should validate importance is between 0 and 10"""
        # Valid importance
        Memory(id="mem_1", content="Test", importance=0.0)
        Memory(id="mem_2", content="Test", importance=10.0)
        Memory(id="mem_3", content="Test", importance=5.5)

        # Invalid importance should fail
        with pytest.raises(ValidationError):
            Memory(id="mem_4", content="Test", importance=-1.0)

        with pytest.raises(ValidationError):
            Memory(id="mem_5", content="Test", importance=11.0)

    def test_memory_string_representation(self):
        """Should format memory with metadata for LLM"""
        memory = Memory(
            id="mem_123",
            content="Worked at forge",
            importance=7.0,
            timestamp=500,
            location=(10, 15),
        )

        mem_str = str(memory)
        assert "mem_123" in mem_str
        assert "500" in mem_str
        assert "(10, 15)" in mem_str or "10" in mem_str  # Format may vary
        assert "Worked at forge" in mem_str


class TestWorkingMemoryModel:
    """Test WorkingMemory model"""

    def test_create_empty_working_memory(self):
        """Should create with default empty values"""
        wm = WorkingMemory()

        assert wm.situation_assessment == ""
        assert wm.active_goals == []
        assert wm.recent_events == []
        assert wm.current_plan == []
        assert wm.emotional_state == ""

    def test_create_populated_working_memory(self):
        """Should create with all fields populated"""
        wm = WorkingMemory(
            situation_assessment="At the forge, hammer in hand",
            active_goals=["Complete sword", "Earn gold"],
            recent_events=["Customer arrived", "Started work"],
            current_plan=["Heat metal", "Shape blade", "Quench"],
            emotional_state="Focused and satisfied",
        )

        assert wm.situation_assessment == "At the forge, hammer in hand"
        assert len(wm.active_goals) == 2
        assert len(wm.recent_events) == 2
        assert len(wm.current_plan) == 3
        assert wm.emotional_state == "Focused and satisfied"

    def test_working_memory_allows_extra_fields(self):
        """Should allow extra fields due to Config.extra = 'allow'"""
        wm = WorkingMemory(custom_field="custom value")

        # Extra fields should be accessible
        assert wm.custom_field == "custom value"  # type: ignore


class TestNewMemoryModel:
    """Test NewMemory model for memory formation"""

    def test_create_new_memory(self):
        """Should create new memory with content and importance"""
        new_mem = NewMemory(content="Finished crafting sword", importance=8.0)

        assert new_mem.content == "Finished crafting sword"
        assert new_mem.importance == 8.0

    def test_importance_validation(self):
        """Should validate importance range"""
        NewMemory(content="Test", importance=1.0)
        NewMemory(content="Test", importance=10.0)

        with pytest.raises(ValidationError):
            NewMemory(content="Test", importance=0.5)  # Below minimum

        with pytest.raises(ValidationError):
            NewMemory(content="Test", importance=11.0)  # Above maximum


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


class TestActionModel:
    """Test Action model"""

    def test_create_basic_action(self):
        """Should create action with type"""
        action = Action(action="wait")

        assert action.action == "wait"
        assert action.parameters == {}

    def test_create_action_with_parameters(self):
        """Should create action with parameters"""
        action = Action(action="move_to", parameters={"x": 10, "y": 20})

        assert action.action == "move_to"
        assert action.parameters["x"] == 10
        assert action.parameters["y"] == 20

    def test_create_interaction_action(self):
        """Should create interact_with action"""
        action = Action(
            action="interact_with",
            parameters={"entity_id": "food_1", "interaction_name": "eat"},
        )

        assert action.action == "interact_with"
        assert action.parameters["entity_id"] == "food_1"
        assert action.parameters["interaction_name"] == "eat"
