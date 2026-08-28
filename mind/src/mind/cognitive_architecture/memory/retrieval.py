"""Generative-Agents retrieval scoring: relevance + importance + recency.

Implements the retrieval function of Park et al. 2023, "Generative Agents:
Interactive Simulacra of Human Behavior" (arXiv:2304.03442) section 4:

    score = a_relevance * relevance + a_importance * importance + a_recency * recency

with all alphas defaulting to 1.0, the paper's published setting.

**This module imports no storage backend and performs no I/O.** That is a hard
constraint, not an accident of the current implementation: it is what lets the
formula be tested as a pure function with hand-computed expected values. A
monotonicity test run through a live embedding model measures cosine noise; run
through this module it constrains the arithmetic. `test_retrieval_scoring.py`
pins the constraint with an import probe.

Two deliberate departures from the paper, both recorded rather than silently
taken:

1. **Fixed-scale normalization, not min-max.** Park min-max scales each of the
   three scores across the candidate set. That makes a memory's score depend on
   which other memories happened to be in the pool, so the same memory scores
   differently on two queries in the same cycle, and it is degenerate when the
   pool holds one candidate or all candidates tie (max == min). It is also
   incompatible with abstention (below), which needs the surviving terms to stay
   on a fixed, comparable scale. Each term here maps onto [0, 1] absolutely,
   with both endpoints given a stated meaning. Consequence worth naming: because
   importance rarely spans its full range in practice, equal weights here
   down-weight importance relative to Park's min-max stretch.
2. **Recency decays from a reinforced anchor, not from a single fixed event.**
   Park decays from the moment a memory was last retrieved, which is lossy: one
   recall erases a memory's entire age, so a decade-old memory recalled once
   becomes indistinguishable from one formed a minute ago. Here the anchor is an
   exponential moving average seeded at creation and pulled toward the present on
   each retrieval, which measures how persistently a memory has *mattered*. This
   generalizes the paper rather than contradicting it: Park's behaviour is the
   `alpha = 1.0` endpoint and pure creation-time decay is `alpha = 0.0`. See
   `reinforced_time`.

The shape is a term registry rather than one widened arithmetic expression,
because most of what is queued behind this (relationship boosting [NPC-401],
per-target stance [NPC-411], mind-written trigger boosting [NPC-406]) is an
additional term or a per-memory modifier. A new term is a new ScoringTerm plus a
weight field; the combiner never changes.
"""

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mind.constants import (
    CANDIDATE_POOL_MULTIPLIER,
    DEFAULT_RECENCY_DECAY_PER_GAME_HOUR,
    DEFAULT_RETRIEVAL_WEIGHT_IMPORTANCE,
    DEFAULT_RETRIEVAL_WEIGHT_RECENCY,
    DEFAULT_RETRIEVAL_WEIGHT_RELEVANCE,
    GAME_MINUTES_PER_HOUR,
    IMPORTANCE_SCALE_MAX,
    MIN_CANDIDATE_POOL,
)
from mind.logging_config import get_logger

from .models import VectorDBMetadata

logger = get_logger()


class ScoredCandidate(BaseModel):
    """One memory as it enters scoring: identity, content, metadata, distance."""

    memory_id: str
    content: str
    metadata: VectorDBMetadata

    # Cosine distance from the vector index, or None when the backend reported
    # none. Chroma's cosine distance is (1 - cosine_similarity) over a similarity
    # in [-1, 1], so the domain is [0, 2].
    distance: float | None = None


class RetrievalContext(BaseModel):
    """Everything a term may condition on besides the memory itself.

    Deliberately near-empty in this iteration. It exists so that the consumers
    queued behind this formula can add what they condition on without touching
    any signature: NPC-401's relationship map, NPC-411's target id, NPC-406's
    trigger context. Slots get added as typed fields per consumer issue - this is
    an extension point, not a dict-of-anything.
    """

    query: str
    current_simulation_time: int | None = None


class ScoreBreakdown(BaseModel):
    """What one candidate scored, and what could not be scored at all.

    `abstained` is the observable half. A store in which every memory abstains on
    recency is a broken write path, and without this it looks identical to a
    store ranking correctly on relevance alone.
    """

    # None means every term abstained - nothing at all was known about this
    # candidate. Callers drop it, loudly. Distinct from a total of 0.0, which is
    # a real measurement of a worthless memory.
    total: float | None
    contributions: dict[str, float] = Field(default_factory=dict)
    abstained: list[str] = Field(default_factory=list)

    # Sum of the weights of the terms that actually voted.
    live_weight: float = 0.0


@runtime_checkable
class ScoringTerm(Protocol):
    """One dimension of the retrieval score.

    `score` returns a value in [0, 1], or **None meaning "no data - abstain"**.
    That return type is the load-bearing design choice in this module. An
    untimestamped memory has no recency *fact*: scoring it 0 ranks it last on a
    property nobody measured, and scoring it 1.0 ranks it first on the same
    non-measurement. Both are a default impersonating data. Abstention scores the
    candidate on what is actually known and renormalizes, so it stays comparable
    with fully-populated candidates.
    """

    name: str

    def score(self, candidate: ScoredCandidate, context: RetrievalContext) -> float | None: ...


class RelevanceTerm:
    """Semantic similarity to the query. 0 = orthogonal or opposed, 1 = identical.

    The clamp folds negative cosine similarity (distance > 1) onto 0: "points the
    other way" and "unrelated" are both simply irrelevant, and there is no
    meaningful ordering between them for retrieval.
    """

    name = "relevance"

    def score(self, candidate: ScoredCandidate, context: RetrievalContext) -> float | None:
        if candidate.distance is None:
            # A backend that reported no distance told us nothing about
            # relevance. Ranking such a candidate as perfectly relevant - the
            # previous behaviour - promotes it above every real match.
            return None
        return max(0.0, min(1.0, 1.0 - candidate.distance))


class ImportanceTerm:
    """LLM-assigned poignancy. 0 = trivial, 1 = maximally significant.

    The rating itself costs nothing extra: reflection already emits an importance
    per new memory on Park's 1-10 rubric, inside the merged reflection call, so
    the paper's "an LLM call per memory at write time" is a call already made.
    """

    name = "importance"

    def score(self, candidate: ScoredCandidate, context: RetrievalContext) -> float | None:
        importance = candidate.metadata.importance
        if importance is None:
            # Never rated (a config seed, say). Abstain rather than vote a
            # midpoint - a fabricated 5.0 outranks every genuinely mundane
            # memory and is indistinguishable from a real rating downstream.
            return None
        return max(0.0, min(1.0, importance / IMPORTANCE_SCALE_MAX))


class RecencyTerm:
    """Park's exponential forgetting curve. ->0 = ancient, 1 = just now.

    `decay_base ** hours_elapsed`, where hours come from the game clock. Not the
    hyperbolic 1/(1 + delta/1000) this replaced: that had a fat tail (a 30-day-old
    memory still scored 0.023) and its 1000 was an unnamed scale constant that,
    read against a clock of game minutes, worked out to a ~16.7-game-hour
    half-scale nobody chose.
    """

    name = "recency"

    def __init__(self, default_decay_base: float = DEFAULT_RECENCY_DECAY_PER_GAME_HOUR):
        self.default_decay_base = default_decay_base

    def score(self, candidate: ScoredCandidate, context: RetrievalContext) -> float | None:
        now = context.current_simulation_time
        # The reinforced anchor, not the raw creation stamp - see
        # VectorDBMetadata.recency_anchor and reinforced_time below.
        anchor = candidate.metadata.recency_anchor
        if now is None or anchor is None:
            return None

        elapsed_minutes = now - anchor
        if elapsed_minutes < 0:
            # A monotone clock cannot produce this. A scenario restart under a
            # retained collection can: elapsed game minutes reset to ~0 while the
            # collection keeps its old stamps. Clamp so the score stays inside
            # [0, 1], but say so - the arithmetic is contained, the data is not.
            # Ownership of the underlying problem is NPC-400's open question D-5.
            logger.warning(
                f"Memory {candidate.memory_id} is stamped in the future "
                f"(now={now}, anchor={anchor}, elapsed={elapsed_minutes} game minutes) - "
                "clamping recency to 'just now'. A retained collection across a "
                "simulation restart is the known cause."
            )
            elapsed_minutes = 0

        # `is None`, not `or`: 0.0 is falsy, and letting it fall through to the
        # default would silently ignore a deliberately-written base.
        decay_base = candidate.metadata.decay_base
        if decay_base is None:
            decay_base = self.default_decay_base

        return decay_base ** (elapsed_minutes / GAME_MINUTES_PER_HOUR)


def reinforced_time(previous_anchor: float, now: float, alpha: float) -> float:
    """One retrieval's exponential-moving-average update of a memory's anchor.

        effective_time <- alpha * now + (1 - alpha) * effective_time

    seeded at the creation timestamp. **The two named behaviours are endpoints of
    `alpha`, not alternatives to this:** `alpha = 1.0` is exactly Park's
    decay-from-last-retrieval, and `alpha = 0.0` is exactly decay-from-creation.
    The constant is the knob, so neither is a fork.

    Why an EMA rather than Park's last-access directly: last access is lossy.
    One retrieval erases a memory's entire age, so a decade-old memory recalled
    once becomes indistinguishable from one formed a minute ago. The EMA instead
    models **how persistently a memory has mattered** - repeated recall drifts it
    toward "recent", a single touch leaves most of its age intact. The
    distinguishing property, and the one the tests pin, is that a memory
    retrieved once is *strictly less recent* than one created at that same
    instant; under `alpha = 1.0` those two are equal.

    Pure and total: the caller decides whether reinforcing is appropriate at all
    (see `should_reinforce`).
    """
    return alpha * now + (1.0 - alpha) * previous_anchor


def should_reinforce(previous_anchor: float | None, now: float | None) -> bool:
    """Whether a retrieval may write an EMA update back.

    False when either reading is missing, and false when `now` precedes the
    anchor. That second case is the scenario-restart hazard (open question D-5)
    and it is **worse under an EMA than it was under a read-time clamp**: the
    clamp contained a bad reading transiently, whereas an EMA fed a reset clock
    would persist a pulled-back anchor, and successive retrievals would drag it
    down until it fell below `now` - at which point elapsed goes positive again,
    the warning stops firing, and the corruption becomes invisible exactly when
    it has finished happening.

    Refusing to write on a known-inconsistent clock keeps the damage read-time
    only, which is where it was before. It does not FIX D-5 - the anchors are
    still wrong relative to the reset clock - it just stops this change from
    making them permanently wrong.
    """
    if previous_anchor is None or now is None:
        return False
    return now >= previous_anchor


class RetrievalWeights(BaseModel):
    """Relative weights of the scoring terms.

    Weights need not sum to 1. `combined_score` divides by the weight that
    actually voted, so these are ratios - which removes a whole class of
    "must sum to 1" errors and is what makes abstention renormalize cleanly.

    `ge=0.0` is structural, not decorative. The previous model expressed
    relevance as an implied `1 - importance_weight - recency_weight`, so
    `importance_weight=0.6, recency_weight=0.6` constructed cleanly and produced
    a relevance coefficient of **-0.2** - a formula that actively preferred
    irrelevant memories, silently, for the whole session. No configuration
    expressible here can produce a negative coefficient, and a degenerate
    all-zero set is rejected at construction rather than at the first query.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    relevance: float = Field(default=DEFAULT_RETRIEVAL_WEIGHT_RELEVANCE, ge=0.0)
    importance: float = Field(default=DEFAULT_RETRIEVAL_WEIGHT_IMPORTANCE, ge=0.0)
    recency: float = Field(default=DEFAULT_RETRIEVAL_WEIGHT_RECENCY, ge=0.0)

    @model_validator(mode="after")
    def _reject_degenerate(self) -> "RetrievalWeights":
        if self.relevance + self.importance + self.recency <= 0.0:
            raise ValueError(
                "at least one retrieval weight must be greater than zero; "
                "an all-zero set scores every memory identically"
            )
        return self

    def weight_for(self, term_name: str) -> float:
        """Weight of a named term.

        Raises rather than defaulting: an unweighted term is a wiring bug, and
        silently giving it 0.0 would delete a whole scoring dimension while every
        query kept returning plausible results.
        """
        try:
            return getattr(self, term_name)
        except AttributeError:
            raise KeyError(
                f"no retrieval weight declared for term {term_name!r}; "
                f"add a field to RetrievalWeights when adding a term"
            ) from None


def default_terms() -> list[ScoringTerm]:
    """The three Park terms, in a stable order."""
    return [RelevanceTerm(), ImportanceTerm(), RecencyTerm()]


def combined_score(
    candidate: ScoredCandidate,
    context: RetrievalContext,
    weights: RetrievalWeights,
    terms: list[ScoringTerm],
) -> ScoreBreakdown:
    """Weighted mean over the terms that had something to say.

    Abstaining terms are excluded from both numerator and denominator, so a
    candidate is never *penalized* for missing plumbing - the absence of a
    timestamp is a fact about our writers, not about the memory. The writers get
    fixed at the write site; the scorer must not paper over them.
    """
    contributions: dict[str, float] = {}
    abstained: list[str] = []
    live_weight = 0.0

    for term in terms:
        value = term.score(candidate, context)
        if value is None:
            abstained.append(term.name)
            continue
        contributions[term.name] = value
        live_weight += weights.weight_for(term.name)

    if live_weight <= 0.0:
        return ScoreBreakdown(
            total=None, contributions=contributions, abstained=abstained, live_weight=0.0
        )

    total = sum(weights.weight_for(name) * value for name, value in contributions.items())
    return ScoreBreakdown(
        total=total / live_weight,
        contributions=contributions,
        abstained=abstained,
        live_weight=live_weight,
    )


def rank(
    candidates: list[ScoredCandidate],
    context: RetrievalContext,
    weights: RetrievalWeights,
    terms: list[ScoringTerm],
    top_k: int,
) -> list[tuple[ScoreBreakdown, ScoredCandidate]]:
    """Score every candidate, order by score descending, keep the best `top_k`."""
    scored: list[tuple[ScoreBreakdown, ScoredCandidate]] = []

    for candidate in candidates:
        breakdown = combined_score(candidate, context, weights, terms)
        if breakdown.total is None:
            logger.warning(
                f"Every retrieval term abstained for memory {candidate.memory_id} "
                f"(abstained: {', '.join(breakdown.abstained) or 'none'}) - dropping it. "
                "Nothing at all is known about this candidate."
            )
            continue
        scored.append((breakdown, candidate))

    scored.sort(key=lambda pair: pair[0].total, reverse=True)
    return scored[:top_k]


def candidate_pool_size(top_k: int, collection_count: int) -> int:
    """How many rows to fetch by cosine before scoring.

    Asking the index for exactly `top_k` makes the weighted score a reranker over
    the cosine top-k rather than a retrieval formula: at the production top_k of
    2, it sorted two items, and a high-importance memory that cosine had not
    already surfaced could not be retrieved at any weight. Over-fetching restores
    the score's ability to promote.

    Still not Park's whole-stream scan - that is O(N) per query per cycle across
    3-5 queries, and is deliberately deferred (open question D-4). Clamped to the
    collection size because the index cannot return rows that do not exist.
    """
    if collection_count <= 0:
        return 0
    desired = max(top_k * CANDIDATE_POOL_MULTIPLIER, MIN_CANDIDATE_POOL)
    return min(desired, collection_count)
