from abc import ABC, abstractmethod
from typing import Any

class Tool(ABC):
    def __init__(self, llm):
        self.llm = llm
        self._initialize_generators()
    
    @abstractmethod
    def _initialize_generators(self) -> None:
        """Initialize any LLM clients needed by this tool"""
        pass
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """Execute the tool's primary function"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Return a description of what this tool does"""
        pass
    
    @property
    @abstractmethod
    def required_inputs(self) -> dict[str, str]:
        """Return a dict of required input names and their descriptions"""
        pass