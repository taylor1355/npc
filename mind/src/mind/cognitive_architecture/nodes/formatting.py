"""Shared prompt-input formatting helpers for cognitive pipeline nodes.

Lives outside base.py so the LLMNode base stays free of domain-specific
rendering logic. Imported by any node that surfaces personality to the LLM
(cognitive_update, action_selection) so the representation stays identical.
"""


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
        dims_text = "\n".join(
            f"{name}: {value:.2f}" for name, value in sorted(dimensions.items())
        )
    else:
        dims_text = "No personality dimensions provided"

    return traits_text, dims_text


def format_interaction_status(observation) -> str:
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
        interaction_name = observation.status.current_interaction.get("interaction_name", "interaction")
        return (
            f"You ARE currently in an interaction ({interaction_name}). "
            "Interaction-participation actions are valid."
        )
    return (
        "You are NOT currently in any interaction. Do not attempt to act in or "
        "continue an interaction; any belief that you are mid-interaction is stale."
    )
