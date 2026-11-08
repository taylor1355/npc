"""Goal-step JSON exactly as the simulation's ``goal_observation.gd`` emits it.

The literals below mirror ``_serialize_steps`` / ``step_factors`` on the
simulation side (time_discount always present; sunk_progress only on a resumed
step; steps at chain index 1 and up are ``projected`` with no ``factors`` and a
target that may carry ``provider_type``). A refusal here would refuse the whole
goal observation, so each shape must validate.
"""

import pytest
from pydantic import ValidationError

from mind.cognitive_architecture.nodes.formatting import format_goal_options
from mind.cognitive_architecture.observations import GoalObservation, GoalOptionStep

PRICED_STEP = {
    "action": {
        "name": "INTERACT_WITH",
        "parameters": {"entity_id": "stove_1", "interaction_name": "cook"},
    },
    "target": {"interaction_name": "cook", "entity_id": "stove_1"},
    "step_score": 0.3,
    "factors": {
        "hope": 1.0,
        "policy_modifier": 1.0,
        "time_discount": 0.75,
        "urgency": 1.0,
        "utility": 0.6,
        "responsiveness": 0.8,
        "sunk_progress": 1.25,
    },
}

PROJECTED_STEP = {
    "action": {
        "name": "INTERACT_WITH",
        "parameters": {"entity_id": "table_2", "interaction_name": "eat"},
    },
    "target": {"interaction_name": "eat", "entity_id": "table_2", "provider_type": "table"},
    "step_score": 0.3,
    "projected": True,
}


def _goal(steps: list[dict]) -> GoalObservation:
    return GoalObservation.model_validate(
        {
            "contract_version": 1,
            "option_total": 1,
            "options": [
                {
                    "option_id": "satisfy_hunger:0",
                    "description": "Cook, then eat",
                    "score": 0.3,
                    "segments": [
                        {
                            "goal_template_id": "satisfy_hunger",
                            "goal_label": "Find food",
                            "steps": steps,
                        }
                    ],
                }
            ],
        }
    )


def test_single_priced_step_with_time_discount_and_sunk_progress_validates():
    step = _goal([PRICED_STEP]).options[0].segments[0].steps[0]
    assert step.factors.time_discount == 0.75
    assert step.factors.sunk_progress == 1.25
    assert step.projected is False


def test_priced_step_without_sunk_progress_defaults_to_identity():
    factors = {k: v for k, v in PRICED_STEP["factors"].items() if k != "sunk_progress"}
    step = _goal([{**PRICED_STEP, "factors": factors}]).options[0].segments[0].steps[0]
    assert step.factors.sunk_progress == 1.0


def test_two_step_chain_with_projected_step_validates():
    steps = _goal([PRICED_STEP, PROJECTED_STEP]).options[0].segments[0].steps
    assert steps[1].projected is True
    assert steps[1].factors is None
    assert steps[1].target.provider_type == "table"


def test_projected_step_target_may_omit_provider_type():
    target = {"interaction_name": "eat", "entity_id": "table_2"}
    step = (
        _goal([PRICED_STEP, {**PROJECTED_STEP, "target": target}]).options[0].segments[0].steps[1]
    )
    assert step.target.provider_type is None


def test_non_projected_step_without_factors_is_refused():
    bare = {k: v for k, v in PRICED_STEP.items() if k != "factors"}
    with pytest.raises(ValidationError, match="must carry factors"):
        GoalOptionStep.model_validate(bare)


@pytest.mark.parametrize(
    "where",
    ["step", "factors", "target"],
)
def test_unknown_key_is_still_refused(where):
    step = {
        **PRICED_STEP,
        "factors": dict(PRICED_STEP["factors"]),
        "target": dict(PRICED_STEP["target"]),
    }
    if where == "step":
        step["mystery"] = 1
    elif where == "factors":
        step["factors"]["mystery"] = 1.0
    else:
        step["target"]["mystery"] = "x"
    with pytest.raises(ValidationError):
        _goal([step])


def test_prompt_renders_discounts_and_projected_step():
    text = format_goal_options(_goal([PRICED_STEP, PROJECTED_STEP]))
    assert "time discount 0.75" in text
    assert "sunk progress 1.25" in text
    assert "then (projected): INTERACT_WITH(entity_id=table_2, interaction_name=eat)" in text
