"""Test fixtures for integration testing"""

from .observations import (
    create_blacksmith_config,
    create_blacksmith_observation,
    create_conversation_observation,
    create_emergency_observation,
    create_explorer_config,
    create_explorer_observation,
    create_idle_observation,
    create_risk_scenario_observation,
)

__all__ = [
    "create_blacksmith_observation",
    "create_explorer_observation",
    "create_conversation_observation",
    "create_idle_observation",
    "create_emergency_observation",
    "create_risk_scenario_observation",
    "create_blacksmith_config",
    "create_explorer_config",
]
