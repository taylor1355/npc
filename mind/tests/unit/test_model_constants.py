"""Offline guards on the configured LLM model identifiers (NPC-1012).

The outage these cover: every dated Gemini preview slug the project defaulted to was
retired upstream, and OpenRouter rejects a retired slug with a 404 at CALL time, not
at startup. Nothing failed loudly - the pipeline just stopped producing decisions.

These are deliberately network-free, so they run in CI and offline. That bounds what
they can prove: they catch the *predictable* subclass (a dated preview slug, which
retires on a schedule) and a partial edit that updates one of the two duplicated
tables. They cannot detect an undated slug being retired - only a live probe against
OpenRouter can, which is why the model list was checked live when these were written.
"""

import re

from mind import constants
from mind.apis.langchain_llm import LangChainModel

# "google/gemini-2.5-flash-lite-preview-09-2025" - a month-dated preview release.
DATED_PREVIEW = re.compile(r"-preview-\d{2}-\d{4}$")


def _model_constants(namespace) -> dict[str, str]:
    """Every uppercase str attribute on a namespace that looks like a provider slug."""
    return {
        name: value
        for name, value in vars(namespace).items()
        if name.isupper() and isinstance(value, str) and "/" in value
    }


def test_no_configured_model_uses_a_dated_preview_slug():
    """Dated preview slugs retire on a schedule and 404 at call time.

    Guards both tables at once; a new model constant is covered automatically.
    """
    offenders = {
        f"{namespace.__name__}.{name}": value
        for namespace in (constants, LangChainModel)
        for name, value in _model_constants(namespace).items()
        if DATED_PREVIEW.search(value)
    }

    assert not offenders, (
        f"Dated preview model slugs are retired upstream and 404 at call time: {offenders}. "
        "Use the undated slug for the same family."
    )


def test_the_guard_can_actually_fail():
    """The regex above must match a real retired slug.

    Without this, a typo'd pattern would make the guard match nothing and pass
    forever - a detector that fails open. The literal is the exact slug from the
    NPC-1012 outage.
    """
    assert DATED_PREVIEW.search("google/gemini-2.5-flash-lite-preview-09-2025")
    assert not DATED_PREVIEW.search("google/gemini-2.5-flash-lite")


def test_constants_and_langchain_model_declare_the_same_slugs():
    """The two model tables are duplicated by hand and must not drift.

    NPC-1012 had to be fixed in both files. A future edit that updates only one
    leaves half the project on a dead model, which presents identically to the
    original outage.
    """
    assert constants.SONNET == LangChainModel.CLAUDE_SONNET
    assert constants.GEMINI_FLASH == LangChainModel.GEMINI_FLASH
    assert constants.GEMINI_FLASH_LITE == LangChainModel.GEMINI_FLASH_LITE


def test_pipeline_defaults_resolve_to_declared_model_families():
    """DEFAULT_SMALL_MODEL is what the whole pipeline uses when a client sets no
    llm_model, so it must alias a maintained family constant rather than a stray
    literal that no one thinks to update.

    The family set is written out rather than derived from the module, because
    deriving it would re-include the DEFAULT_* values and make this pass by identity.
    """
    families = {constants.SONNET, constants.GEMINI_FLASH, constants.GEMINI_FLASH_LITE}
    assert constants.DEFAULT_SMALL_MODEL in families
    assert constants.DEFAULT_LARGE_MODEL in families
