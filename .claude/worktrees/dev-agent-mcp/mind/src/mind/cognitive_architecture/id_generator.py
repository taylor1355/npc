"""ID generation utilities for cognitive architecture components"""

import uuid


class IdGenerator:
    """Static ID generation methods matching Godot IdGenerator pattern"""

    @staticmethod
    def generate_uuid() -> str:
        """Generate a UUID v4 string"""
        return str(uuid.uuid4())

    @staticmethod
    def generate_memory_id() -> str:
        """Generate a unique memory ID"""
        return f"memory_{IdGenerator.generate_uuid()}"
