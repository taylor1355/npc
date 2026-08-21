"""Prompt-variable contract for the reflection prompt.

The merged prompt is split at a literal cache-breakpoint marker into a static
prefix (formatted once, byte-identical across calls) and a dynamic suffix
(formatted per call). This contract pins three things: the marker exists
exactly once, the static half declares ONLY construction-time variables, and
the dynamic half declares exactly the expected per-call set — a silently
dropped variable (e.g. substrate_goal) would strand behavior with no error.
"""

from pathlib import Path
from unittest.mock import AsyncMock

from langchain_core.prompts import PromptTemplate

from mind.cognitive_architecture.nodes.reflection.node import (
    CACHE_BREAKPOINT_MARKER,
    ReflectionNode,
)

PROMPT_PATH = (
    Path(__file__).parents[2] / "src/mind/cognitive_architecture/nodes/reflection/prompt.md"
)

STATIC_VARIABLES = {"world_knowledge", "format_instructions"}

DYNAMIC_VARIABLES = {
    "working_memory",
    "personality_traits",
    "personality_dimensions",
    "interaction_status",
    "substrate_goal",
    "retrieved_memories",
    "recent_events",
    "observation_text",
    "available_actions",
}


class TestReflectionPromptContract:
    def test_prompt_contains_exactly_one_cache_breakpoint_marker(self):
        assert PROMPT_PATH.read_text().count(CACHE_BREAKPOINT_MARKER) == 1

    def test_static_half_declares_only_construction_time_variables(self):
        """Anything else above the breakpoint would vary per call and silently
        fragment the provider cache (the node's __init__ would refuse it)."""
        static_text, _ = PROMPT_PATH.read_text().split(CACHE_BREAKPOINT_MARKER)

        variables = set(PromptTemplate.from_template(static_text).input_variables)

        assert variables == STATIC_VARIABLES

    def test_dynamic_half_declares_exactly_the_per_call_set(self):
        _, dynamic_text = PROMPT_PATH.read_text().split(CACHE_BREAKPOINT_MARKER)

        variables = set(PromptTemplate.from_template(dynamic_text).input_variables)

        assert variables == DYNAMIC_VARIABLES

    def test_node_prompt_matches_the_dynamic_contract(self):
        """The union of static + dynamic variables is the full input contract;
        the node's live template must carry exactly the dynamic half (the
        static half was consumed at construction)."""
        node = ReflectionNode(AsyncMock())

        assert set(node.prompt.input_variables) == DYNAMIC_VARIABLES
        assert node.static_prefix is not None
        # The marker is the split point, never part of the sent bytes
        assert CACHE_BREAKPOINT_MARKER not in node.static_prefix
        # The formatted prefix contains no unconsumed static placeholders
        assert "{world_knowledge}" not in node.static_prefix
        assert "{format_instructions}" not in node.static_prefix
