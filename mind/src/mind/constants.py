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

# Memory Retrieval Scoring
#
# Park et al. 2023, "Generative Agents: Interactive Simulacra of Human Behavior"
# (arXiv:2304.03442) section 4, retrieval: the score is a weighted sum of recency,
# importance and relevance, and "in our implementation, all alphas are set to 1".
# We ship the paper's numbers rather than a tuned set, so the defaults are citable
# rather than incidental. Tuning is deliberately out of scope [NPC-400].
DEFAULT_RETRIEVAL_WEIGHT_RELEVANCE = 1.0
DEFAULT_RETRIEVAL_WEIGHT_IMPORTANCE = 1.0
DEFAULT_RETRIEVAL_WEIGHT_RECENCY = 1.0

# Park's exponential forgetting curve, per GAME hour. Half-life is
# log(0.5)/log(0.995) ~= 138 game hours ~= 5.8 game days.
DEFAULT_RECENCY_DECAY_PER_GAME_HOUR = 0.995

# How strongly one retrieval pulls a memory's effective age toward "now":
#
#     effective_time <- alpha * now + (1 - alpha) * effective_time
#
# The constant is the knob, and the two named behaviours are its ENDPOINTS
# rather than forks of it:
#   alpha = 1.0  -- exactly Park's decay-from-last-retrieval
#   alpha = 0.0  -- exactly decay-from-creation
#
# 0.3 retains 70% of a memory's age per retrieval, so log(0.5)/log(0.7) ~= 1.94:
# **two retrievals halve a memory's effective age.** For a memory 100 game hours
# old at the moment it is recalled, effective age and recency score go:
#     never recalled  100.0 gh -> 0.606
#     recalled once    70.0 gh -> 0.704
#     recalled 5x      16.8 gh -> 0.919
# A single recall is meaningful but far from erasing, while sustained recall
# carries a memory back toward fresh. That asymmetry is the point: this models
# how persistently a memory has MATTERED, not merely when it was last touched.
#
# Not a tuned number - a defensible default. Tuning is out of scope [NPC-400].
DEFAULT_RECENCY_REINFORCEMENT_ALPHA = 0.3

# The simulation clock is elapsed game MINUTES
# (SimulationTime.get_elapsed_game_minutes), while the decay base above is
# per-hour. This is the conversion between them; it is not a tuning knob.
GAME_MINUTES_PER_HOUR = 60.0

# Importance is authored on Park's 1-10 poignancy scale (see the rubric in
# nodes/reflection/prompt.md), so this is the divisor that maps it onto [0, 1].
IMPORTANCE_SCALE_MAX = 10.0

# Candidate pool for retrieval. Park ranks over the whole memory stream; asking
# the vector index for exactly top_k would make the weighted score a reranker
# over the cosine top-k, unable to surface a high-importance memory that cosine
# did not already rank first. We over-fetch by cosine, then score the pool.
# An HNSW query returning tens of rows instead of two costs no LLM call and no
# extra embedding. A true whole-stream scan is deliberately not attempted here.
CANDIDATE_POOL_MULTIPLIER = 10
MIN_CANDIDATE_POOL = 30
