"""Constants for Mind configuration including LLM models"""

# LLM Model Families - OpenRouter compatible models
#
# Prefer UNDATED slugs. Dated preview slugs ("...-preview-09-2025") are retired
# upstream on a schedule, and OpenRouter then rejects them with a 404 "No endpoints
# found for <slug>" at CALL time rather than at startup - so the whole pipeline
# (MCP mind decisions, memory consolidation, the LLM-judge harness) goes silently
# dead and presents as "the NPC stopped deciding". Both Gemini constants here were
# dated previews and both had been retired (NPC-1012).
SONNET = "anthropic/claude-sonnet-4"
GEMINI_FLASH = "google/gemini-2.5-flash"
GEMINI_FLASH_LITE = "google/gemini-2.5-flash-lite"

# Default Models
DEFAULT_SMALL_MODEL = GEMINI_FLASH_LITE  # Cheapest, fastest for testing
DEFAULT_LARGE_MODEL = SONNET  # High quality for complex reasoning

# Embedding Models
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Memory Configuration
DEFAULT_MEMORY_STORAGE_PATH = "./chroma_db"
DEFAULT_MEMORIES_PER_QUERY = 2
DEFAULT_MAX_RETRIEVED_MEMORIES = 5
