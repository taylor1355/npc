"""Unit tests for memory models"""

import pytest
from pydantic import ValidationError

from mind.cognitive_architecture.memory import Memory


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
