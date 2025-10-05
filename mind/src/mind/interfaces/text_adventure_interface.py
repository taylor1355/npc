from mind.interfaces import SimulatorInterface, SimulatorRequest, SimulatorResponse
from mind.simulators.text_adventure import TextAdventureSimulator
from pydantic import Field
from typing import Any, Type


class TextAdventureRequest(SimulatorRequest):
    action_index: int | None = Field(default=None, description="Chosen action index from available options")
        

class TextAdventureResponse(SimulatorResponse):
    observation: str = Field(required=True, description="New story segment after taking the action")
    available_actions: dict[int, str] = Field(required=True, description="Available actions in the new story segment")


class TextAdventureInterface(SimulatorInterface[TextAdventureRequest, TextAdventureResponse, TextAdventureSimulator]):
    def __init__(self, simulator: TextAdventureSimulator):
        super().__init__(simulator)

    @property
    def request_class(self) -> Type[TextAdventureRequest]:
        return TextAdventureRequest

    @property
    def response_class(self) -> Type[TextAdventureResponse]:
        return TextAdventureResponse

    def execute(self, request: TextAdventureRequest) -> TextAdventureResponse:
        # Empty action, used to get the initial story segment
        if not request.action_index:
            state = self.simulator.state
            return TextAdventureResponse(
                success=True,
                message="Initial story segment",
                observation=state.observation,
                available_actions=state.available_actions,
            )

        # Execute the chosen action
        try:
            new_state = self.simulator.take_action(request.action_index)
            return TextAdventureResponse(
                success=True,
                message="Action executed successfully",
                observation=new_state.observation,
                available_actions=new_state.available_actions,
            )
        except Exception as e:
            return TextAdventureResponse(
                success=False,
                message=f"Error executing action: {str(e)}",
                observation="",
                available_actions={},
            )