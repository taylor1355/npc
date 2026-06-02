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
