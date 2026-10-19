"""LangChain LLM wrapper configured for OpenRouter"""

from langchain_openai import ChatOpenAI

from mind import constants


class LangChainModel:
    """Model identifiers for OpenRouter-compatible models.

    A named view over `mind.constants`, not a second table. NPC-1012 had to be fixed
    in two hand-duplicated copies of these slugs; aliasing means a future slug change
    lands in one place and cannot leave half the project on a retired model.
    """

    CLAUDE_SONNET = constants.SONNET
    GEMINI_FLASH = constants.GEMINI_FLASH
    GEMINI_FLASH_LITE = constants.GEMINI_FLASH_LITE


# Per-call HTTP bound. Without it there is none: langchain-openai passes
# request_timeout=None through, which httpx treats as unbounded (the SDK's
# 600s default never applies). Kept below the game's 60s decide_action deadline
# (npc-simulation DEADLINE_SECONDS_DECIDE_ACTION) so this side fails first.
# tests/unit/test_server_bounds.py pins both properties.
LLM_REQUEST_TIMEOUT_SECONDS = 45.0

# Node-level retry (LLMNode max_retries) already exists; client retry would multiply it.
LLM_CLIENT_MAX_RETRIES = 0


def supports_cache_control(model: str) -> bool:
    """Whether this model slug may receive explicit cache_control breakpoints.

    Allowlist-based and deliberately conservative: an unknown slug (or a test
    double whose model name is not a real slug) gets no breakpoint, because a
    provider that rejects an unrecognised content-block key would otherwise
    take the whole decision down.
    """
    return model in constants.CACHE_CONTROL_MODELS


def get_llm(model: str, temperature: float = 0) -> ChatOpenAI:
    """Get a configured LangChain LLM via OpenRouter

    Args:
        model: Model identifier (use LangChainModel constants)
        temperature: Sampling temperature (0.0 to 1.0), default 0 for deterministic output

    Returns:
        Configured ChatOpenAI instance pointing to OpenRouter, bounded by
        LLM_REQUEST_TIMEOUT_SECONDS with client-level retry disabled — see the
        constants above for why both are explicit rather than left to defaults.

    Example:
        >>> from mind.apis.langchain_llm import get_llm, LangChainModel
        >>> llm = get_llm(LangChainModel.CLAUDE_SONNET)
        >>> response = llm.invoke("Hello!")
    """
    # Pure cache-policy consumers must not load credentials; real construction still does.
    from mind.project_config import OPENROUTER_API_KEY

    return ChatOpenAI(
        model=model,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=temperature,
        timeout=LLM_REQUEST_TIMEOUT_SECONDS,
        max_retries=LLM_CLIENT_MAX_RETRIES,
    )


# A local Ollama server's OpenAI-compatible endpoint. Ollama's own default port.
OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434/v1"

# A local model's first call loads its weights into VRAM, which alone can take
# longer than the OpenRouter bound above; this backend is for the on-demand
# experiment tier, not for the game's 60s decide_action deadline.
LOCAL_LLM_REQUEST_TIMEOUT_SECONDS = 300.0


def get_local_llm(
    model: str,
    *,
    base_url: str = OLLAMA_DEFAULT_BASE_URL,
    temperature: float = 0,
    reasoning_effort: str | None = None,
    timeout: float = LOCAL_LLM_REQUEST_TIMEOUT_SECONDS,
) -> ChatOpenAI:
    """A LangChain LLM for a local OpenAI-compatible server (Ollama by default).

    Same client class as ``get_llm``, so ``LLMNode`` -- retries, usage
    accounting, structured output -- needs no change: only the base URL and the
    credential differ. No credential is loaded (Ollama ignores the key, but the
    client requires a non-empty one). ``supports_cache_control`` is False for
    every local tag, so no cache breakpoint is ever sent.

    ``reasoning_effort="none"`` turns off a hybrid model's thinking through
    Ollama's OpenAI-compatible endpoint (verified on ``qwen3:8b``, Ollama
    0.40.5). Leave it None for models without a thinking switch: a
    thinking-only model answers it by moving its reasoning into ``content``,
    which breaks JSON parsing.
    """
    kwargs = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}
    return ChatOpenAI(
        model=model,
        openai_api_key="ollama",
        openai_api_base=base_url,
        temperature=temperature,
        timeout=timeout,
        max_retries=LLM_CLIENT_MAX_RETRIES,
        **kwargs,
    )
