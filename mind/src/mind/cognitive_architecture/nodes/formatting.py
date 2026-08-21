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
