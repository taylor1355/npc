"""Offline guards on the configured LLM model identifiers (NPC-1012).

Preview and version-pinned model slugs are retired upstream on a schedule, and
OpenRouter rejects a retired slug with a 404 at CALL time, not at startup - so the
pipeline stops producing decisions without anything failing loudly.

These are network-free so they run in CI and offline, which bounds what they prove:
they catch the predictable subclass (a slug whose name marks it as scheduled for
retirement). They cannot detect an *undated* slug being retired - only a live probe
against OpenRouter can.
"""

import re

from mind import constants
from mind.apis.langchain_llm import LangChainModel

# Slug suffixes that mark a model as scheduled for retirement:
#   -preview, -exp                        undated preview / experimental
#   -preview-09-2025, -preview-2025-09    dated preview, either field order
#   -001                                  pinned version snapshot
#
# Calibrated against OpenRouter's live catalog (338 models, 2026-08-03): no served
# slug ends in -NNN, and the 13 served slugs ending in -preview/-exp are exactly the
# family this rule tells you not to configure, so flagging them is intended.
RETIRING_SLUG = re.compile(r"-(preview|exp)(-[\d-]+)?$|-\d{3}$")


def _model_constants(namespace) -> dict[str, str]:
    """Every uppercase str attribute on a namespace that looks like a provider slug."""
    return {
        name: value
        for name, value in vars(namespace).items()
        if name.isupper() and isinstance(value, str) and "/" in value
    }


def test_no_configured_model_uses_a_retiring_slug():
    """Preview and pinned slugs retire on a schedule and 404 at call time.

    Guards both namespaces at once; a new model constant is covered automatically.
    """
    offenders = {
        f"{namespace.__name__}.{name}": value
        for namespace in (constants, LangChainModel)
        for name, value in _model_constants(namespace).items()
        if RETIRING_SLUG.search(value)
    }

    assert not offenders, (
        f"These model slugs retire upstream on a schedule and 404 at call time: {offenders}. "
        "Use the undated, unpinned slug for the same family."
    )


def test_the_guard_can_actually_fail():
    """The pattern must match every retiring shape, and no maintained slug.

    Without this, a typo'd pattern would match nothing and pass forever - a detector
    that fails open. The first three literals are real slugs that were live in this
    repo and now 404 (verified against OpenRouter); the rest are the shapes the
    pattern was widened to cover.
    """
    retiring = [
        # Verified 404 "No endpoints found" against OpenRouter.
        "google/gemini-2.5-flash-lite-preview-09-2025",
        "google/gemini-2.5-flash-preview-09-2025",
        "google/gemini-2.0-flash-lite-001",
        # Shapes in the same retires-on-a-schedule family.
        "google/gemini-2.5-flash-preview",
        "google/gemini-2.5-flash-preview-2025-09",
        "google/gemini-2.0-flash-exp",
        "google/gemini-exp-1206",
    ]
    for slug in retiring:
        assert RETIRING_SLUG.search(slug), f"guard fails open on {slug}"

    maintained = [
        "google/gemini-2.5-flash-lite",
        "google/gemini-2.5-flash",
        "anthropic/claude-sonnet-4",
    ]
    for slug in maintained:
        assert not RETIRING_SLUG.search(slug), f"guard false-positives on {slug}"


def test_langchain_model_aliases_the_constants_table():
    """LangChainModel must stay a view over `constants`, not a second copy.

    The duplication that made NPC-1012 a two-file fix is gone by construction; this
    asserts the aliasing survives, so re-introducing a literal here fails loudly.
    """
    assert constants.SONNET is LangChainModel.CLAUDE_SONNET
    assert constants.GEMINI_FLASH is LangChainModel.GEMINI_FLASH
    assert constants.GEMINI_FLASH_LITE is LangChainModel.GEMINI_FLASH_LITE


def test_pipeline_defaults_resolve_to_declared_model_families():
    """DEFAULT_SMALL_MODEL is what the whole pipeline uses when a client sets no
    llm_model, so its value must be one of the declared families rather than a stray
    literal that no one thinks to update when the family moves.

    The family set is written out rather than derived from the module, because
    deriving it would re-include the DEFAULT_* values and make this pass by identity.
    """
    families = {constants.SONNET, constants.GEMINI_FLASH, constants.GEMINI_FLASH_LITE}
    assert constants.DEFAULT_SMALL_MODEL in families
    assert constants.DEFAULT_LARGE_MODEL in families
