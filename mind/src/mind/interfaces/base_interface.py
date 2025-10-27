from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Union, get_args, get_origin

from json_repair import repair_json
from pydantic import BaseModel


class SimulatorRequest(BaseModel):
    """Base class for simulator request parameters.

    This class serves as a blueprint for all simulator requests. Subclasses should
    define specific request parameters as class attributes.
    """

    @classmethod
    def documentation_llm_str(cls) -> str:
        """Generate documentation of the request parameters in an LLM-friendly format.

        Returns:
            str: A formatted string listing each parameter, its type, and description.
        """

        def get_type_name(annotation) -> str:
            """Recursively get a readable type name from an annotation."""
            origin = get_origin(annotation)
            if origin is Union:
                args = get_args(annotation)
                return " | ".join(get_type_name(arg) for arg in args)
            elif hasattr(annotation, "__name__"):
                return annotation.__name__
            else:
                return str(annotation)

        return "\n".join(
            [
                f"- **{name}** ({get_type_name(field.annotation)}): {field.description}"
                for name, field in cls.model_fields.items()
            ]
        )

    @classmethod
    def parse_json(cls, json_str: str):
        """Parse a JSON string into an instance of the request model.

        Args:
            json_str (str): The JSON string to parse.

        Returns:
            SimulatorRequest: An instance of the request model populated with the parsed data.
        """
        if "{" not in json_str:
            json_str = "{" + json_str
        if "}" not in json_str:
            json_str = json_str + "}"
        json_str = repair_json(json_str)

        return cls.model_validate_json(json_str)


class SimulatorResponse(BaseModel):
    """Base class for simulator response parameters.

    Represents the standard response structure returned by the simulator after executing a request.
    Subclasses can add additional fields if necessary.
    """

    success: bool
    message: str
    observation: str
    available_actions: dict[int, str]

    def observation_llm_str(self) -> str:
        """Generate the observation in an LLM-friendly format.

        Returns:
            str: A formatted string containing the observation.
        """
        return f"# Observation:\n{self.observation}"

    def available_actions_llm_str(self) -> str:
        """Generate the available actions in an LLM-friendly format.

        Returns:
            str: A formatted string listing the available actions with their indices.
        """
        return "\n".join(
            [
                "# Available Actions",
                *[
                    f"- Action Index {index}: {action}"
                    for index, action in self.available_actions.items()
                ],
            ]
        )


SimulatorRequestType = TypeVar("SimulatorRequestType", bound=SimulatorRequest)
SimulatorResponseType = TypeVar("SimulatorResponseType", bound=SimulatorResponse)
SimulatorType = TypeVar("SimulatorType")


class SimulatorInterface(ABC, Generic[SimulatorRequestType, SimulatorResponseType, SimulatorType]):
    """
    Abstract base class for simulator interfaces.

    Defines the required methods for interacting with a simulator. This interface should be the only
    part of the NPC module that directly interacts with, or has knowledge of, the simulator.
    """

    def __init__(self, simulator: SimulatorType):
        """Initialize the simulator interface with a simulator instance.

        Args:
            simulator (SimulatorType): An instance of the simulator to interact with.
        """
        self.simulator = simulator

    @property
    @abstractmethod
    def request_class(self) -> type[SimulatorRequestType]:
        """
        Get the Pydantic model class for the simulator request.

        Returns:
            type[SimulatorRequestType]: The request model class.
        """
        pass

    @property
    @abstractmethod
    def response_class(self) -> type[SimulatorResponseType]:
        """
        Get the Pydantic model class for the simulator response.

        Returns:
            type[SimulatorResponseType]: The response model class.
        """
        pass

    @abstractmethod
    def execute(self, simulator_request: SimulatorRequestType) -> SimulatorResponseType:
        """
        Execute a request on the simulator and return the response.

        Args:
            simulator_request (SimulatorRequestType): The request parameters as defined by the request model.

        Returns:
            SimulatorResponseType: The simulator's response, as defined by the response model.
        """
        pass
