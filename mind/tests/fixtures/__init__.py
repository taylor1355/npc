"""Test fixtures for integration testing"""

from .observations import (
    create_blacksmith_config,
    create_blacksmith_observation,
    create_conversation_events,
    create_conversation_observation,
    create_emergency_observation,
    create_enriched_observation,
    create_explorer_config,
    create_explorer_observation,
    create_idle_observation,
    create_movement_events,
    create_risk_scenario_observation,
    create_saturated_events,
    create_social_events,
)

__all__ = [
    "create_blacksmith_observation",
    "create_explorer_observation",
    "create_conversation_observation",
    "create_idle_observation",
    "create_emergency_observation",
    "create_enriched_observation",
    "create_risk_scenario_observation",
    "create_blacksmith_config",
    "create_explorer_config",
    # Event-buffer constructors. Exported so a caller can build a decision cycle
    # with a POPULATED recent-events buffer through the package rather than
    # reaching past it into tests.fixtures.observations — an empty buffer is
    # what made the 2026-08-20 decision-cycle measurement unusable (NPC-1318),
    # and it looks exactly like a normal one from the outside.
    "create_movement_events",
    "create_social_events",
    "create_conversation_events",
    "create_saturated_events",
]
