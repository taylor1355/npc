"""Shared prompt-input formatting helpers for cognitive pipeline nodes.

Lives outside base.py so the LLMNode base stays free of domain-specific
rendering logic. Imported by any node that surfaces this state to the LLM
(reflection today) so the representation stays identical if more ever do.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mind.cognitive_architecture.observations import GoalObservation, Observation


def format_personality(
    traits: list[str],
    dimensions: dict[str, float],
) -> tuple[str, str]:
    """Format personality traits and dimensions for prompt rendering.

    Returns sentinel strings when personality is absent because LangChain
    PromptTemplate requires every declared variable to be present.

    Returns:
        (traits_text, dimensions_text). Dimensions are sorted by name for
        deterministic prompts.
    """
    traits_text = ", ".join(traits) if traits else "No specific traits"

    if dimensions:
        dims_text = "\n".join(f"{name}: {value:.2f}" for name, value in sorted(dimensions.items()))
    else:
        dims_text = "No personality dimensions provided"

    return traits_text, dims_text


def format_interaction_status(observation: Observation | None) -> str:
    """Render the observation's authoritative interaction status for prompts.

    Grounds the LLM's "am I interacting?" belief in the current observation
    (current_interaction + activity_state), so a stale working-memory belief
    can be corrected each cycle rather than driving the NPC-688 desync loop.
    A single-source-of-truth rendering: every prompt surface that states the
    interaction status goes through here.

    Defaults to "NOT currently in any interaction" when status is absent or
    partial — a missing field never reads as "interacting".

    The name comes from ``StatusObservation.interaction_display_name()``, the
    one canonical reader of that wire key. Reading the dict directly here is
    what made this line render the literal word "interaction" for every
    interaction the NPC ever had (NPC-1278).
    """
    if observation is not None and observation.is_interacting():
        interaction_name = observation.status.interaction_display_name()
        return (
            f"You ARE currently in an interaction ({interaction_name}). "
            "Interaction-participation actions are valid."
        )
    return (
        "You are NOT currently in any interaction. Do not attempt to act in or "
        "continue an interaction; any belief that you are mid-interaction is stale."
    )


def format_substrate_goal(goal: GoalObservation | None) -> str:
    """Render the substrate's active goal as an advisory pull for prompts.

    The reflection prompt gives the pull its own dedicated section (with a
    standing advisory on how to weigh it) rather than leaving it buried inside
    ``observation_text``; the observation's own "Subconscious pull" line is the
    same datum echoed, and the prompt says so.

    Returns a sentinel string when no goal is present. This is mandatory, not
    defensive: LangChain PromptTemplate raises at format time on any declared
    variable that is missing, and in the reflection node that exception would
    burn every retry before the salvage fallback caught it. A None-returning
    helper would turn "this NPC has no active goal" into "this NPC never acts".

    Urgency is rendered as the raw simulation value rather than a percentage —
    the percent-of-maximum conversion belongs to the simulation tier, and
    re-deriving it here would fork the display scale.
    """
    if goal is None or goal.active_goal is None:
        return "Nothing in particular; your drives have not settled on a pull right now."

    active = goal.active_goal
    return f"Toward: {active.label} {active.urgency_clause()}."


def format_goal_options(goal: GoalObservation | None) -> str:
    """Render the substrate's evaluated option menu for prompts.

    The Subconscious Pull section is the substrate's settled direction; this is
    the menu it weighed to get there — the same pool, scores included, that the
    reflex tier's softmax samples from. Options are rendered in the wire's
    score-descending order, one headline line per option plus one line per step
    (steps carry the goal each serves, so a multi-goal plan reads as a chain).

    Written against the contract's GENERAL plan shape: multiple segments and
    steps per option render as successive step lines, even though the tier-0
    sim sends exactly one of each.

    Returns a sentinel string when there is no menu, for the same
    PromptTemplate-raises-on-missing-variable reason as
    ``format_substrate_goal``. Scores stay raw simulation values throughout —
    the percent conversion is owned by the simulation tier.
    """
    if goal is None or not goal.options:
        return "No evaluated options this cycle."

    shown = len(goal.options)
    header = f"{shown} of {goal.option_total} evaluated options, best-scored first"
    if goal.urgency_max is not None:
        header += f" (urgency scale runs 0 to {goal.urgency_max:g})"
    lines = [header + ":"]

    for option in goal.options:
        lines.append(f"Option {option.option_id}: {option.description} (score: {option.score:.2f})")
        for segment in option.segments:
            for step in segment.steps:
                params = ", ".join(f"{k}={v}" for k, v in step.action.parameters.items())
                action_text = f"{step.action.name}({params})" if params else step.action.name
                lines.append(
                    f"  - serves '{segment.goal_label}': {action_text} "
                    f"[utility {step.factors.utility:.2f}, "
                    f"habituation {step.factors.responsiveness:.2f}, "
                    f"step score {step.step_score:.2f}]"
                )

    if goal.option_total > shown:
        lines.append(
            f"({goal.option_total - shown} lower-scoring options exist but are not shown.)"
        )
    return "\n".join(lines)
