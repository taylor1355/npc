"""LangChain LLM wrapper configured for OpenRouter"""

from langchain_openai import ChatOpenAI

from mind.project_config import OPENROUTER_API_KEY


class LangChainModel:
    """Model identifiers for OpenRouter-compatible models"""
    CLAUDE_SONNET = "anthropic/claude-sonnet-4"
    GEMINI_FLASH = "google/gemini-2.5-flash-preview-09-2025"
    GEMINI_FLASH_LITE = "google/gemini-2.5-flash-lite-preview-09-2025"


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
        temperature=temperature
    )
