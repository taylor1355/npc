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


# Per-call HTTP bound, in seconds, applied to every provider request.
#
# WHY THIS EXISTS AT ALL: without it there is NO timeout anywhere in the chain.
# `openai._constants.DEFAULT_TIMEOUT` is 600s and is widely assumed to apply
# here — it does not. langchain-openai passes its own `request_timeout` field
# through to the OpenAI client unconditionally, and that field defaults to None,
# which is a real value rather than the SDK's `not_given` sentinel. The SDK
# substitutes DEFAULT_TIMEOUT only for `not_given`, so an explicit None reaches
# httpx as `Timeout(timeout=None)` — unbounded connect, read, write and pool.
# Verified against the versions pinned in uv.lock (openai 1.109.1,
# langchain-openai 1.1.9, httpx 0.28.1):
#
#     ChatOpenAI(...).root_client._client.timeout  ->  Timeout(timeout=None)
#
# So a provider that accepts the connection and then goes quiet wedges the
# decision forever.
#
# WHY 45s. Today this is the ONLY bound in the chain: the game side calls
# `decide_action` with no operation timeout at all (`McpSdkClient.cs::
# DecideAction`; `McpServiceProxy.CallToolAsync` applies its timeout only to
# `EnsureConnectedClientAsync`, i.e. connection setup). Verified directly on the
# simulation repo's main, not taken on trust.
#
# [NPC-1682] adds the missing outer half — a per-attempt wall-clock deadline of
# 60s for `decide_action` plus a 30-game-minute decision watchdog. It is in
# flight, not merged. 45s is chosen to be correct in BOTH states:
#
#   * Before it lands, 45s is what stops a hung call wedging the NPC forever.
#   * After it lands, 45s sits INSIDE the client's 60s, so the inner bound fires
#     first and the server answers with a real error the client can attribute,
#     instead of the client cancelling a call the server still believes is live
#     and still paying a provider for. Matching 60s exactly would make which one
#     fires a race, which is why this is not simply the same number.
#
# Worst-case wall clock for one decide_action, with the settings below:
#
#     2 LLM nodes             MemoryQueryNode and ReflectionNode. The third
#                             pipeline node (MemoryRetrievalNode) makes no
#                             provider call, and MemoryConsolidationNode is a
#                             plain Node outside the graph.
#   x 3 attempts each         both pass max_retries=2 to LLMNode.
#   x 1 HTTP request each     LLM_CLIENT_MAX_RETRIES = 0, below.
#   = 6 requests x 45s        = 270s absolute ceiling.
#
# Reaching that ceiling needs BOTH nodes to burn every validation retry, because
# LLMNode's retry loop catches only (json.JSONDecodeError, ValidationError) — a
# timeout is not retried at the node level and escapes on its first occurrence.
# A single hung call therefore costs 45s, not 270s, and the single-hang case is
# the one this exists for.
#
# KNOWN RESIDUE: 270s still exceeds NPC-1682's 60s client deadline, so a
# decision that burns its validation retries is abandoned by the client while
# this server keeps working on it. Closing that needs a whole-PIPELINE deadline
# here, not a per-call one — a design question, deliberately not decided in a
# firefighting change.
LLM_REQUEST_TIMEOUT_SECONDS = 45.0

# Client-level retry is DISABLED because node-level retry already exists and the
# two MULTIPLY rather than compose. The OpenAI SDK defaults to 2 (verified:
# `openai._constants.DEFAULT_MAX_RETRIES == 2`), which would turn each of the 6
# attempts above into 3 HTTP requests — 18 requests per decision, every one of
# them unbounded today. The retry that belongs to this layer (connection errors,
# 429, 5xx) is left to the caller so there is exactly one retry policy in play.
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
