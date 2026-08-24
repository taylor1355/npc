"""Constants for Mind configuration including LLM models"""

# LLM Model Families - OpenRouter compatible models
SONNET = "anthropic/claude-sonnet-4"
GEMINI_FLASH = "google/gemini-2.5-flash-preview-09-2025"
GEMINI_FLASH_LITE = "google/gemini-2.5-flash-lite-preview-09-2025"

# Default Models
DEFAULT_SMALL_MODEL = GEMINI_FLASH_LITE  # Cheapest, fastest for testing
DEFAULT_LARGE_MODEL = SONNET  # High quality for complex reasoning

# Embedding Models
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Memory Configuration
DEFAULT_MEMORY_STORAGE_PATH = "./chroma_db"
DEFAULT_MEMORIES_PER_QUERY = 2
DEFAULT_MAX_RETRIEVED_MEMORIES = 5
