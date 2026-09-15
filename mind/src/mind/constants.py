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

# The spatial term is ours, not Park's, so it gets no citable number and must
# argue for its own [NPC-1476]. It is 0.5 rather than 1.0 because the module's
# own thesis about min-max normalization decides it: min-max and alpha=1 are a
# package precisely BECAUSE the three Park terms have unequal realized spreads,
# and normalizing puts them on equal footing so equal coefficients mean equal
# influence. A BINARY term breaks that symmetry from the other side - whenever it
# is live at all it realizes a spread of exactly 1.0, the widest possible - so
# giving it Park's coefficient would hand it MORE than Park's influence.
#
# **That premise weakened when the term stopped being binary [NPC-1476].** Graded
# by distance it no longer realizes 1.0: on the measurement corpus it lands at
# ~0.984 against recency's ~0.985, tied for widest rather than uniquely widest.
# The halving is therefore argued from something no longer quite true, and 0.5 is
# now a starting value with a weaker case rather than a reasoned one - if
# anything it under-weights a graded term, which spreads candidates across the
# band instead of polarizing them to the ends.
#
# Deliberately NOT retuned here. The right number comes from a measurement over
# real memory distributions, not from adjusting one argument to rescue another,
# and this corpus is a fixture rather than a world.
# test_term_spread_measurement.py::test_the_graded_spatial_term_lands_in_the_widest_band
# carries the reading.
DEFAULT_RETRIEVAL_WEIGHT_SPATIAL = 0.5

# Characteristic falloff length for SpatialTerm, in grid cells. Relevance decays
# as 0.5 ** (chebyshev_distance / this), so a memory formed this many cells away
# scores half what one formed underfoot does, and the curve never reaches zero.
#
# 8 is the simulation's sight radius, and that is the argument rather than a
# coincidence: "near" for an embodied agent is most naturally "about as far as I
# can see", and it is the one length in this world that already has a meaning
# both sides agree on. Chebyshev, not Euclidean, for the same reason - the
# simulation's own vision disc is Chebyshev over an 8-connected grid, so a
# Euclidean metric here would disagree with the geometry that produced the
# coordinates.
#
# A reasoned starting scale, not a measured one. What would falsify it: retrieval
# that surfaces same-room memories and across-the-map memories at
# indistinguishable rank (too large), or that cannot see anything outside the
# current cell (too small). test_term_spread_measurement.py measures the realized
# spread this produces; prefer that number to this argument.
DEFAULT_SPATIAL_DECAY_CELLS = 8.0

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
