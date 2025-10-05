"""Protocols defining interfaces for cognitive architecture components"""

from typing import Protocol
from .state import PipelineState


class CognitiveNode(Protocol):
    """Protocol for nodes in the cognitive pipeline"""

    async def process(self, state: PipelineState) -> PipelineState:
        """Process the state and return updated state"""
        ...