"""LangChain LLM wrapper configured for OpenRouter"""

from langchain_openai import ChatOpenAI

from mind import constants
from mind.project_config import OPENROUTER_API_KEY


class LangChainModel:
    """Model identifiers for OpenRouter-compatible models.

    A named view over `mind.constants`, not a second table. NPC-1012 had to be fixed
    in two hand-duplicated copies of these slugs; aliasing means a future slug change
    lands in one place and cannot leave half the project on a retired model.
    """

    CLAUDE_SONNET = constants.SONNET
    GEMINI_FLASH = constants.GEMINI_FLASH
    GEMINI_FLASH_LITE = constants.GEMINI_FLASH_LITE


def get_llm(model: str, temperature: float = 0) -> ChatOpenAI:
    """Get a configured LangChain LLM via OpenRouter

    Args:
        model: Model identifier (use LangChainModel constants)
        temperature: Sampling temperature (0.0 to 1.0), default 0 for deterministic output

    Returns:
        Configured ChatOpenAI instance pointing to OpenRouter

    Example:
        >>> from mind.apis.langchain_llm import get_llm, LangChainModel
        >>> llm = get_llm(LangChainModel.CLAUDE_SONNET)
        >>> response = llm.invoke("Hello!")
    """
    return ChatOpenAI(
        model=model,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=temperature,
    )
