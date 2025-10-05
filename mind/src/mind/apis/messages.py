"""Simple, serializable message types for LLM communication"""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel


class Role(str, Enum):
    """Message roles"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    """Simple message type that's fully serializable"""
    role: Role
    content: str

    def to_dict(self) -> dict:
        """Convert to OpenAI-compatible format"""
        return {"role": self.role.value, "content": self.content}


class ChatThread(BaseModel):
    """A thread of messages"""
    messages: List[Message]

    def to_openai_format(self) -> List[dict]:
        """Convert to OpenAI API format"""
        return [msg.to_dict() for msg in self.messages]

    def add_system(self, content: str) -> "ChatThread":
        """Add a system message"""
        self.messages.append(Message(role=Role.SYSTEM, content=content))
        return self

    def add_user(self, content: str) -> "ChatThread":
        """Add a user message"""
        self.messages.append(Message(role=Role.USER, content=content))
        return self

    def add_assistant(self, content: str) -> "ChatThread":
        """Add an assistant message"""
        self.messages.append(Message(role=Role.ASSISTANT, content=content))
        return self