from abc import ABC, abstractmethod
from typing import Any

class Tool(ABC):
    def __init__(self):
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Return a description of what this tool does"""
        pass