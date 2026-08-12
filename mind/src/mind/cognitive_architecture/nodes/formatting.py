"""Shared prompt-input formatting helpers for cognitive pipeline nodes.

Lives outside base.py so the LLMNode base stays free of domain-specific
rendering logic. Imported by any node that surfaces personality to the LLM
(cognitive_update, action_selection) so the representation stays identical.
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

    Shared by nodes that surface personality to the LLM (cognitive_update,
    action_selection) so the rendered representation stays identical across the
    pipeline. Returns sentinel strings when personality is absent because
    LangChain PromptTemplate requires every declared variable to be present.

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
    Shared by cognitive_update and action_selection so both nodes see an
    identical, single-source-of-truth rendering.

    Defaults to "NOT currently in any interaction" when status is absent or
    partial — a missing field never reads as "interacting".
    """
    if observation is not None and observation.is_interacting():
        interaction_name = observation.status.current_interaction.get(
            "interaction_name", "interaction"
        )
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

    action_selection never receives ``observation_text``, so without this the
    substrate's own answer to "why act?" reaches the node that actually picks
    actions only laundered through working memory, or not at all.

    Returns a sentinel string when no goal is present. This is mandatory, not
    defensive: LangChain PromptTemplate raises at format time on any declared
    variable that is missing, and in this node that exception is caught and
    collapses the cycle into the WAIT fallback. A None-returning helper would
    turn "this NPC has no active goal" into "this NPC never acts".

    Urgency is rendered as the raw simulation value rather than a percentage —
    the percent-of-maximum conversion belongs to the simulation tier, and
    re-deriving it here would fork the display scale.
    """
    if goal is None or goal.active_goal is None:
        return "Nothing in particular; your drives have not settled on a pull right now."

    active = goal.active_goal
    drive_clause = f", arising from your {active.drive_source} drive" if active.drive_source else ""
    return f"Toward: {active.label} (urgency {active.urgency:.2f}{drive_clause})."
