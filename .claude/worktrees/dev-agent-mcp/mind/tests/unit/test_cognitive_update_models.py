"""Unit tests for cognitive update models (WorkingMemory, NewMemory)"""

import pytest
from pydantic import ValidationError

from mind.cognitive_architecture.nodes.cognitive_update.models import (
    NewMemory,
    WorkingMemory,
)


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
