"""Memory retrieval: Park relevance/importance/recency plus spatial context.

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

**Normalization is min-max over the candidate pool, following Park - and the
reason matters more than the precedent.** The paper states the choice flat, with
no rationale, no comparison, and an ablation that varied observation, reflection
and planning but never the normalization or the weights; the secondary literature
only restates it. So the argument below is ours, not theirs:

The three terms have very different **realized** ranges despite all being
nominally [0, 1]. Cosine similarity between an embedded query and related text
occupies a narrow band and rarely approaches either endpoint; LLM importance
ratings cluster toward the middle, because models seldom emit 1 or 10;
exponential recency decay spans nearly the whole interval. Under fixed absolute
scales, the term with the widest realized variance silently dominates whatever
the weights say. **So min-max and `alpha = 1` are a package**: without the
normalization, "all alphas are 1" means equal *coefficients* on unequal ranges,
not equal *influence*, and shipping the paper's weights without the paper's
normalization while calling them the paper's settings would be incoherent.

`test_term_spread_measurement.py` measures those realized spreads on our own
embeddings and our own importance distribution rather than assuming the paper's,
because that is the evidence Park did not publish.

Three consequences, none of them defects:

- **A candidate's score depends on which others were in the pool.** That is what
  a ranking function over a candidate set *is*. It is sound because nothing
  compares these scores across queries: `search()` discards the breakdowns and
  returns only the ordered memories, and no consumer of `ScoreBreakdown` exists
  outside this module and its tests.
- **A term whose values all tie carries no information for that pool**, and every
  candidate gets the same normalized value. See `TermStats.normalize` for why
  that value is 0.5 rather than 1.0 or an abstention.
- **Abstention composes with it** by normalizing each term over the candidates
  that did not abstain on it. See `term_statistics`.

Recency reinforcement is a deliberate departure from the paper:

1. **Recency decays from a reinforced anchor, not from a single fixed event.**
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

`SpatialTerm` [NPC-1476] is the first term added under that shape, and it
exercised it exactly as designed: a class, a `RetrievalWeights` field, a
`RetrievalContext` slot, an entry in `default_terms()`. `combined_score`,
`term_statistics`, `collect_raw_scores`, `score_pool`, `rank` and `TermStats` are
untouched by it. It is also the first term that is not Park's, so unlike the
other three its weight is argued rather than cited - see
`DEFAULT_RETRIEVAL_WEIGHT_SPATIAL`, and its decay scale likewise, see
`DEFAULT_SPATIAL_DECAY_CELLS`. It is the only CONTINUOUS term of the four whose
input is not already a similarity: relevance arrives as a vector distance,
importance and recency as rates, while this one converts a grid separation into
one.

**Deliberate divergence from simulation place selection.** Proposed ADR-40
("belief as a posterior", npc-simulation/docs/adrs/2026-09-belief-as-a-posterior.md)
chooses a Gamma-Poisson posterior and absolute travel utility for place decisions.
This module retains pool-normalized weighted retrieval: semantic relevance,
importance and recency have incommensurable realized ranges. Retrieving a memory
and deciding whether a place is worth visiting answer different questions; the
simulation's posterior policy does not replace this term registry. This records
the chosen design boundary, not a claim that either change is deployed.
"""

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mind.constants import (
    CANDIDATE_POOL_MULTIPLIER,
    DEFAULT_RECENCY_DECAY_PER_GAME_HOUR,
    DEFAULT_RETRIEVAL_WEIGHT_IMPORTANCE,
    DEFAULT_RETRIEVAL_WEIGHT_RECENCY,
    DEFAULT_RETRIEVAL_WEIGHT_RELEVANCE,
    DEFAULT_RETRIEVAL_WEIGHT_SPATIAL,
    DEFAULT_SPATIAL_DECAY_CELLS,
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

    # The place the NPC is standing in right now, as a simulation zone id, from
    # StatusObservation.current_zone_id. None means the NPC knows no place here -
    # which SpatialTerm treats as an abstention, not as a place called "nowhere".
    current_zone_id: str | None = None

    # Where the NPC is standing, in grid cells, from StatusObservation.position.
    # This is what makes SpatialTerm GRADED rather than binary: with it, a memory
    # formed one room away outranks one formed across the map, which a
    # same-place-or-not test cannot express. Absent for a status-less query, and
    # SpatialTerm then falls back to the zone comparison.
    current_position: tuple[int, int] | None = None


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

    # Per-term values AFTER min-max normalization over the pool - what actually
    # entered the weighted sum.
    contributions: dict[str, float] = Field(default_factory=dict)

    # Per-term values BEFORE normalization, on each term's own absolute scale.
    # Kept because a normalized value alone cannot distinguish "genuinely the
    # most relevant memory" from "the least bad in a pool of irrelevant ones" -
    # min-max maps the pool's best onto 1.0 either way. This is also what the
    # realized-spread measurement reads.
    raw_contributions: dict[str, float] = Field(default_factory=dict)

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


class SpatialTerm:
    """Memories formed near where the NPC is standing retrieve hotter.

    **Graded by distance, with same-place as its ceiling.** A memory formed one
    room away is more relevant than one formed across the map, and a binary
    same-place test cannot say so - it collapses every elsewhere into one bucket.
    So the score is ``0.5 ** (chebyshev_distance / DEFAULT_SPATIAL_DECAY_CELLS)``
    over the two formation cells: 1.0 underfoot, 0.5 a sight-radius away, never
    reaching zero. Chebyshev because the simulation's own vision disc is
    Chebyshev over an 8-connected grid; a Euclidean metric here would disagree
    with the geometry that produced the coordinates.

    **Same place short-circuits to 1.0, and that is not the same as distance 0.**
    A place has extent. Standing at one end of a clearing, a memory formed at its
    far end is thirty cells away by the metric and *here* by every other reading
    an NPC has - it happened in the place the NPC is in. Zone identity is the
    better answer when it is available, so it wins; distance grades everything
    the zone comparison would have flattened into a single 0.0.

    **Abstains rather than scoring 0.0 whenever neither reading is available**:
    the NPC knows no place here and carries no position, or the memory carries
    neither. A 0.0 there would rank a memory last on a property nobody measured -
    the same default-impersonating-data failure the abstention doctrine in this
    module exists to prevent, and the reason ``ImportanceTerm`` does not vote a
    midpoint for an unrated memory.

    Note what is NOT an abstention: a real distance is a measurement, however
    large, and "known to be different places" with no positions on either side is
    also a measurement. Those return numbers.
    """

    name = "spatial"

    def score(self, candidate: ScoredCandidate, context: RetrievalContext) -> float | None:
        here_zone = context.current_zone_id
        formed_zone = candidate.metadata.zone_id

        # Same place wins outright, before distance is consulted - see the
        # far-end-of-the-clearing case in the class docstring.
        if here_zone and formed_zone and here_zone == formed_zone:
            return 1.0

        here_position = context.current_position
        formed_position = candidate.metadata.get_location()
        if here_position is not None and formed_position is not None:
            separation = max(
                abs(here_position[0] - formed_position[0]),
                abs(here_position[1] - formed_position[1]),
            )
            return 0.5 ** (separation / DEFAULT_SPATIAL_DECAY_CELLS)

        # No usable coordinates. Two known and different places is still a
        # measurement, so it scores; anything less is not, so it abstains.
        if here_zone and formed_zone:
            return 0.0
        return None


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
    spatial: float = Field(default=DEFAULT_RETRIEVAL_WEIGHT_SPATIAL, ge=0.0)

    @model_validator(mode="after")
    def _reject_degenerate(self) -> "RetrievalWeights":
        # Every weight, not just Park's three: a place-only configuration
        # (relevance=importance=recency=0, spatial=1) is legitimate, and omitting
        # spatial from this sum would reject it as degenerate [NPC-1476]. Any
        # term added later must be added here too, or it becomes un-configurable
        # on its own.
        if self.relevance + self.importance + self.recency + self.spatial <= 0.0:
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
    """The three Park terms plus the spatial term, in a stable order.

    SpatialTerm goes last so the Park three keep the order they have always had.
    Order is not semantically load-bearing - `combined_score` sums - but it is
    the order raw/normalized contribution dicts iterate in, which several tests
    and the spread measurement read.
    """
    return [RelevanceTerm(), ImportanceTerm(), RecencyTerm(), SpatialTerm()]


class TermStats(BaseModel):
    """The realized range of one term across one candidate pool.

    Computed over the candidates that did NOT abstain on that term - see
    `term_statistics`.
    """

    minimum: float
    maximum: float

    # How many candidates contributed. A spread computed from one candidate is
    # not evidence about the term's range; the measurement reports this so a
    # narrow spread cannot be mistaken for a narrow distribution.
    count: int

    @property
    def spread(self) -> float:
        return self.maximum - self.minimum

    def normalize(self, value: float) -> float:
        """Map a raw score onto [0, 1] against this pool's observed range.

        When the range has collapsed to a point, the term carries **no
        information** for this pool - every candidate scored identically on it,
        so it cannot discriminate between them. Every candidate therefore gets
        the same value, and that value is the midpoint 0.5: the least arbitrary
        image of a degenerate interval, and neutral in magnitude relative to a
        term that did spread.

        Not 1.0, and not abstention. 1.0 would let a term that measured nothing
        maximally inflate every total, which matters because candidates with
        different abstention patterns divide by different live weights - so a
        constant is not a uniform shift and does move the ranking. Abstention
        would be cleaner still, except that a pool of ONE candidate is degenerate
        on every term at once, so abstaining would drop the only candidate and
        return nothing for a perfectly valid `top_k=1` query.
        """
        if self.spread <= 0.0:
            return 0.5
        return (value - self.minimum) / self.spread


def collect_raw_scores(
    candidates: list[ScoredCandidate],
    context: RetrievalContext,
    terms: list[ScoringTerm],
) -> list[dict[str, float]]:
    """Each candidate's raw per-term scores, on the terms' own absolute scales.

    A term that abstains is simply absent from that candidate's dict.
    """
    return [
        {
            term.name: value
            for term in terms
            if (value := term.score(candidate, context)) is not None
        }
        for candidate in candidates
    ]


def term_statistics(
    raw_scores: list[dict[str, float]], terms: list[ScoringTerm]
) -> dict[str, TermStats]:
    """Realized min/max per term, over the candidates that scored on it.

    **Abstaining candidates are excluded from a term's min/max**, which is the
    consistent extension of abstention rather than a special case: abstention
    already removes a candidate from that term's weight, so letting it influence
    that term's range would be reintroducing, through the normalizer, exactly the
    vote it was excluded from.

    A term nothing scored on gets no entry at all.
    """
    stats: dict[str, TermStats] = {}
    for term in terms:
        values = [raw[term.name] for raw in raw_scores if term.name in raw]
        if not values:
            continue
        stats[term.name] = TermStats(minimum=min(values), maximum=max(values), count=len(values))
    return stats


def combined_score(
    raw: dict[str, float],
    stats: dict[str, TermStats],
    weights: RetrievalWeights,
    terms: list[ScoringTerm],
) -> ScoreBreakdown:
    """Weighted mean of the min-max-normalized terms that had something to say.

    Two independent mechanisms meet here and the interaction is the subtle part:

    - **Min-max normalization** puts every term on the same realized scale, which
      is what makes equal weights mean equal *influence*. Without it, the term
      with the widest realized variance quietly dominates however the weights are
      set (see the module docstring).
    - **Abstention** excludes a term from both numerator and denominator for the
      candidates that had no data for it, so a candidate is never penalized for
      missing plumbing.

    They compose because normalization happens per term, per pool, over the
    non-abstaining subset only, while the weight renormalization happens per
    candidate. A candidate scoring 0.8 normalized on everything it knows totals
    0.8 whether it abstained on nothing or on two terms.
    """
    contributions: dict[str, float] = {}
    abstained: list[str] = []
    live_weight = 0.0

    for term in terms:
        if term.name not in raw:
            abstained.append(term.name)
            continue
        term_stats = stats.get(term.name)
        contributions[term.name] = (
            term_stats.normalize(raw[term.name]) if term_stats is not None else 0.5
        )
        live_weight += weights.weight_for(term.name)

    if live_weight <= 0.0:
        return ScoreBreakdown(
            total=None,
            contributions=contributions,
            raw_contributions=dict(raw),
            abstained=abstained,
            live_weight=0.0,
        )

    total = sum(weights.weight_for(name) * value for name, value in contributions.items())
    return ScoreBreakdown(
        total=total / live_weight,
        contributions=contributions,
        raw_contributions=dict(raw),
        abstained=abstained,
        live_weight=live_weight,
    )


def score_pool(
    candidates: list[ScoredCandidate],
    context: RetrievalContext,
    weights: RetrievalWeights,
    terms: list[ScoringTerm],
) -> list[ScoreBreakdown]:
    """Score a whole candidate pool together, in input order.

    Pool-wide by necessity, not by preference: min-max normalization is defined
    against the pool, so no candidate's score exists independently of the others.
    That is a property of a ranking function over a candidate set, and it is
    sound here precisely because nothing compares these scores across queries -
    `search()` discards the breakdowns and returns only the ordered memories.
    """
    raw_scores = collect_raw_scores(candidates, context, terms)
    stats = term_statistics(raw_scores, terms)
    return [combined_score(raw, stats, weights, terms) for raw in raw_scores]


def rank(
    candidates: list[ScoredCandidate],
    context: RetrievalContext,
    weights: RetrievalWeights,
    terms: list[ScoringTerm],
    top_k: int,
) -> list[tuple[ScoreBreakdown, ScoredCandidate]]:
    """Score every candidate, order by score descending, keep the best `top_k`."""
    scored: list[tuple[ScoreBreakdown, ScoredCandidate]] = []

    for breakdown, candidate in zip(score_pool(candidates, context, weights, terms), candidates):
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
