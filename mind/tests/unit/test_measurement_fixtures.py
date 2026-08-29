"""Pins the decision-cycle measurement matrix (NPC-1318).

These assertions exist because of a specific, silent failure: the 2026-08-20
per-cycle measurement was taken against observation fixtures carrying no
``MindEvent``s at all, so every cycle it measured ran with an EMPTY
recent-events buffer. Nothing errored, nothing looked wrong, and the number was
published — it was only unusable in retrospect.

Nothing here asserts a token count. Counts move with every prompt edit, and
pinning them would make an intended prompt change look like a regression. What
is pinned is the set of structural properties whose absence makes a measurement
mean something other than it claims.

No LLM, no network, no filesystem: the Mind is constructed inert (no pipeline,
no memory store) because every property under test is reached before either is
touched.
"""

from mind.cognitive_architecture.state import PipelineState
from mind.cognitive_architecture.working_memory import WorkingMemory
from mind.interfaces.mcp.mind import (
    EVENT_BUFFER_MAX_SIZE,
    EVENT_RETENTION_TIME_MINUTES,
    Mind,
)
from tests.fixtures import create_saturated_events
from tests.fixtures.measurement import (
    MeasurementScenario,
    apply_cycle_inputs,
    build_measurement_scenarios,
    scenario_config,
)

# PipelineState fields that build_pipeline_state deliberately does NOT set:
# outputs the pipeline writes, and working state the nodes fill in. Listed
# explicitly so a NEW field added to PipelineState fails this module until
# somebody decides which side of the line it falls on — that is the drift
# channel that let the predecessor harness's hand-copied construction go stale.
NON_INPUT_FIELDS = {
    "memory_queries",
    "retrieved_memories",
    "daily_memories",
    "chosen_action",
    "tokens_used",
    "time_ms",
}


def _inert_mind(scenario: MeasurementScenario) -> Mind:
    """A Mind with the scenario's identity and personality and nothing else.

    ``Mind.from_config`` would build a real LLM client and a real ChromaDB
    collection; neither is reachable from anything asserted below, and both
    would put a model download on the unit suite's critical path.
    """
    return Mind(
        mind_id=f"measure_{scenario.id}",
        entity_id=scenario.entity_id,
        traits=scenario.config.traits,
        pipeline=None,
        memory_store=None,
        working_memory=scenario.config.initial_working_memory or WorkingMemory(),
        llm_model=scenario.config.llm_model,
        personality_dimensions=scenario.config.personality_dimensions,
    )


def _drive_one_cycle(scenario: MeasurementScenario) -> PipelineState:
    mind = _inert_mind(scenario)
    apply_cycle_inputs(mind, scenario)
    return mind.build_pipeline_state(scenario.observation)


def test_matrix_has_at_least_five_scenarios_with_unique_ids():
    scenarios = build_measurement_scenarios()
    ids = [s.id for s in scenarios]

    assert len(scenarios) >= 5
    assert len(set(ids)) == len(ids), f"duplicate scenario ids: {ids}"


def test_every_scenario_yields_a_populated_event_buffer():
    """THE load-bearing assertion — the exact property whose absence in 2026-08
    made a published per-cycle figure measure a cycle nobody runs."""
    for scenario in build_measurement_scenarios():
        state = _drive_one_cycle(scenario)
        assert len(state.recent_events) > 0, (
            f"scenario {scenario.id} produced an EMPTY recent-events buffer; "
            f"any cycle measured from it understates the reflection prompt"
        )


def test_every_scenario_offers_at_least_one_action():
    """A cycle with no available actions is not a decision. The integration
    suite's ``available_actions=[]`` shortcut must not creep into the matrix."""
    for scenario in build_measurement_scenarios():
        state = _drive_one_cycle(scenario)
        assert len(state.available_actions) > 0, f"scenario {scenario.id} offered no actions"


def test_saturated_buffer_is_sized_to_the_retention_ceiling():
    events = create_saturated_events()

    assert len(events) == EVENT_BUFFER_MAX_SIZE

    # And every timestamp is inside the retention window, so update_events
    # keeps all of them rather than aging some out and silently measuring a
    # smaller buffer than the fixture's length advertises.
    start = min(event.timestamp for event in events)
    span = max(event.timestamp for event in events) - start
    assert span < EVENT_RETENTION_TIME_MINUTES


def test_saturated_scenario_actually_saturates_after_retention():
    """The fixture's length is not the measurement — what survives
    ``update_events`` is."""
    scenario = next(s for s in build_measurement_scenarios() if s.id == "enriched_saturated")
    state = _drive_one_cycle(scenario)

    assert len(state.recent_events) == EVENT_BUFFER_MAX_SIZE


def test_build_pipeline_state_populates_every_input_field():
    """Closes the drift channel a hand-copied state construction leaves open.

    ``model_fields_set`` is what the constructor was actually given, so adding
    an input to PipelineState without threading it through
    ``Mind.build_pipeline_state`` fails here instead of silently measuring (and
    running) a cycle with that input at its default.
    """
    scenario = build_measurement_scenarios()[0]
    state = _drive_one_cycle(scenario)

    expected_inputs = set(PipelineState.model_fields) - NON_INPUT_FIELDS
    assert state.model_fields_set == expected_inputs


def test_scenario_config_rehomes_the_memory_store_without_mutating_the_scenario():
    """Reps must be independent. A shared store would let rep 2 retrieve what
    rep 1 wrote, and the token count would climb for a reason nobody recorded."""
    scenario = build_measurement_scenarios()[0]
    original = scenario.config.memory_storage_path

    rehomed = scenario_config(scenario, "/tmp/measure-decision-cycle-probe")

    assert rehomed.memory_storage_path == "/tmp/measure-decision-cycle-probe"
    assert scenario.config.memory_storage_path == original
    assert rehomed.initial_long_term_memories == scenario.config.initial_long_term_memories


def test_harness_cli_imports():
    """``tools/`` is outside CI's ruff and pytest scopes, so this import is the
    only thing standing between the CLI and rotting into a syntax error nobody
    notices until the next measurement is due."""
    import tools.measure_decision_cycle as harness

    assert callable(harness.main)
