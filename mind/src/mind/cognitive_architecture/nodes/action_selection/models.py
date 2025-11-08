"""Models for action selection node"""

from pydantic import BaseModel, Field

from ...actions import Action, AvailableAction


class ActionSelectionInput(BaseModel):
    """Input for action selection"""

    working_memory: str = Field(description="Current working memory state")
    cognitive_context: dict = Field(
        description="Current cognitive context including goals and emotions"
    )
    available_actions: list[AvailableAction] = Field(
        description="Actions available from the simulation"
    )
    personality_traits: list[str] = Field(description="Character personality traits")


class ActionSelectionOutput(BaseModel):
    """Output from action selection"""

    chosen_action: Action = Field(description="Selected action to execute")
