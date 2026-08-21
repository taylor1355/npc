"""Constants for Mind configuration including LLM models"""

# LLM Model Families - OpenRouter compatible models
#
# Prefer undated, unpinned slugs. Preview and version-pinned slugs
# ("...-preview-09-2025", "...-lite-001") are retired upstream on a schedule, and
# OpenRouter then rejects them with a 404 at CALL time rather than at startup - so
# the pipeline goes silently dead rather than failing loudly (NPC-1012).
#
# This is the single source for model slugs; LangChainModel aliases these values.
SONNET = "anthropic/claude-sonnet-4"
GEMINI_FLASH = "google/gemini-2.5-flash"
GEMINI_FLASH_LITE = "google/gemini-2.5-flash-lite"

# Default Models
DEFAULT_SMALL_MODEL = GEMINI_FLASH_LITE  # Cheapest, fastest for testing
DEFAULT_LARGE_MODEL = SONNET  # High quality for complex reasoning

# Prompt Caching
#
# Models allowed to receive an explicit cache_control breakpoint on prompt
# content blocks (OpenRouter passes it through to Anthropic/Gemini; OpenAI
# caches automatically and ignores it). Unknown slugs default OFF: a provider
# that 400s on an unrecognised key must not take the pipeline down.
CACHE_CONTROL_MODELS = frozenset({SONNET, GEMINI_FLASH, GEMINI_FLASH_LITE})

# Prefix length (characters) below which requesting a breakpoint is pointless:
# every relevant provider minimum is >= 1,024 tokens, roughly 4 KB of text.
MIN_CACHEABLE_PREFIX_CHARS = 4 * 1024

# Embedding Models
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Memory Configuration
DEFAULT_MEMORY_STORAGE_PATH = "./chroma_db"
DEFAULT_MEMORIES_PER_QUERY = 2
DEFAULT_MAX_RETRIEVED_MEMORIES = 5
