"""The decision-cycle measurement matrix (NPC-1318).

Everything the decision-cycle harness knows about *what* to measure lives here
rather than in ``tools/measure_decision_cycle.py``. Two reasons, and only the
second is about tidiness:

- ``tools/`` is outside the ruff and pytest scopes both CI and pre-commit
  declare (``src tests``), so logic placed there is ungated. Here it is linted,
  formatted, and pinned by ``tests/unit/test_measurement_fixtures.py``.
- The predecessor harness was never committed at all, and the fixtures it ran
  on carried no ``MindEvent``s — so it measured every cycle with an EMPTY
  recent-events buffer and nothing said so. Putting the matrix under test is
  what makes that failure loud the next time it is approached.

The scenarios pair a committed observation with a committed event buffer, and
span the two axes that actually move a cycle's token count: how many events the
buffer holds, and how many actions the observation makes available.
"""

from __future__ import annotations

from dataclasses import dataclass

from mind.cognitive_architecture.observations import MindEvent, Observation
from mind.interfaces.mcp.mind import Mind
from mind.interfaces.mcp.models import MindConfig
from mind.interfaces.mcp.server import _extract_conversation_observations

from .observations import (
    create_blacksmith_config,
    create_blacksmith_observation,
    create_conversation_events,
    create_emergency_observation,
    create_enriched_observation,
    create_explorer_config,
    create_explorer_observation,
    create_movement_events,
    create_saturated_events,
    create_social_events,
)


@dataclass(frozen=True)
class MeasurementScenario:
    """One decision cycle's inputs, as they would arrive over the wire.

    ``config`` is a distinct instance per scenario, so a caller may rewrite its
    ``memory_storage_path`` for a run without leaking that into any other
    scenario.
    """

    id: str
    why: str
    config: MindConfig
    observation: Observation
    events: list[MindEvent]

    @property
    def entity_id(self) -> str:
        """The driven entity, taken from the observation.

        ``decide_action`` rejects a cycle whose observation entity_id disagrees
        with the mind's, so a harness that invented its own id would be running
        a state production would have refused.
        """
        return self.observation.entity_id


def build_measurement_scenarios() -> list[MeasurementScenario]:
    """The matrix. Pure data — no LLM, no network, no filesystem."""
    return [
        MeasurementScenario(
            id="blacksmith_movement",
            why="cheap buffer, several visible entities — the floor",
            config=create_blacksmith_config(),
            observation=create_blacksmith_observation(),
            events=create_movement_events(),
        ),
        MeasurementScenario(
            id="blacksmith_conversation",
            why="the expensive event type: an INTERACTION_OBSERVATION in the buffer",
            config=create_blacksmith_config(),
            observation=create_blacksmith_observation(),
            events=create_conversation_events(),
        ),
        MeasurementScenario(
            id="explorer_social",
            why="full bid lifecycle, no conversation content",
            config=create_explorer_config(),
            observation=create_explorer_observation(),
            events=create_social_events(),
        ),
        MeasurementScenario(
            id="enriched_saturated",
            why="goal options plus a buffer at the retention ceiling — the upper bound",
            config=create_blacksmith_config(),
            observation=create_enriched_observation(),
            events=create_saturated_events(),
        ),
        MeasurementScenario(
            id="emergency_movement",
            why="high-urgency control against blacksmith_movement",
            config=create_explorer_config(),
            observation=create_emergency_observation(),
            events=create_movement_events(),
        ),
    ]


def scenario_config(scenario: MeasurementScenario, memory_storage_path: str) -> MindConfig:
    """The scenario's config, re-homed onto a caller-owned memory store.

    The fixture configs point at ``./tmp/test_*_db``, which persists between
    runs. A measurement run must not inherit a store an earlier run warmed, so
    every repetition gets a fresh directory and re-seeds
    ``initial_long_term_memories`` from scratch.
    """
    return scenario.config.model_copy(update={"memory_storage_path": memory_storage_path})


def apply_cycle_inputs(mind: Mind, scenario: MeasurementScenario) -> None:
    """Advance ``mind`` by this cycle's inputs, exactly as ``decide_action`` does.

    These are the three mutations ``server.py::decide_action`` performs before
    it builds the pipeline state — and ``update_events`` is the ONLY thing that
    ever fills ``mind.event_buffer``, which is why skipping it is what produces
    an empty-buffer measurement.

    Deliberately does NOT build the state: that is
    ``Mind.build_pipeline_state``, which production and the harness share.
    """
    conversation_obs = _extract_conversation_observations(scenario.events, mind.entity_id)
    mind.update_conversations(conversation_obs)
    mind.update_events(scenario.events, scenario.observation.current_simulation_time)
    mind.last_simulation_time = scenario.observation.current_simulation_time
