"""``hope`` on goal-step factors (NPC-1896): additive, defaulted, range-checked."""

import pytest
from pydantic import ValidationError

from mind.cognitive_architecture.nodes.formatting import format_goal_options
from mind.cognitive_architecture.observations import (
    GoalObservation,
    GoalStepFactors,
)

BASE = {"urgency": 1.2, "utility": 0.5, "responsiveness": 0.8, "policy_modifier": 1.0}


def _goal(factors: dict, step_score: float) -> GoalObservation:
    return GoalObservation.model_validate(
        {
            "contract_version": 1,
            "option_total": 1,
            "options": [
                {
                    "option_id": "eat:0",
                    "description": "Eat the apple",
                    "score": step_score,
                    "segments": [
                        {
                            "goal_template_id": "satisfy_hunger",
                            "goal_label": "Find food",
                            "steps": [
                                {
                                    "action": {"name": "WAIT", "parameters": {}},
                                    "factors": factors,
                                    "step_score": step_score,
                                }
                            ],
                        }
                    ],
                }
            ],
        }
    )


def test_payload_without_hope_validates_as_full_hope():
    assert GoalStepFactors.model_validate(BASE).hope == 1.0


def test_payload_with_hope_validates():
    assert GoalStepFactors.model_validate({**BASE, "hope": 0.25}).hope == 0.25
    assert GoalStepFactors.model_validate({**BASE, "hope": 1.0}).hope == 1.0


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.01, float("nan")])
def test_out_of_range_hope_rejected(bad):
    with pytest.raises(ValidationError):
        GoalStepFactors.model_validate({**BASE, "hope": bad})


def test_other_unknown_factor_keys_still_rejected():
    with pytest.raises(ValidationError):
        GoalStepFactors.model_validate({**BASE, "mystery": 1.0})


def test_prompt_shows_hope_only_when_diminished():
    low = format_goal_options(_goal({**BASE, "hope": 0.4}, 0.19))
    assert "hope 0.40" in low
    full = format_goal_options(_goal(BASE, 0.48))
    assert "hope" not in full
    explicit_full = format_goal_options(_goal({**BASE, "hope": 1.0}, 0.48))
    assert "hope" not in explicit_full
