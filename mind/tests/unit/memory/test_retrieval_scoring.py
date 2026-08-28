"""Unit tests for the Generative-Agents retrieval scorer.

These drive `memory/retrieval.py` as a **pure function**: no ChromaDB, no
embedding model, no I/O. That is the whole point of the module boundary. A
monotonicity test run through a live encoder measures cosine noise and passes
whatever the arithmetic does; run against hand-built candidates with exact
distances it constrains the formula.

Expected scores below are computed offline from Park's published constants
(decay base 0.995 per game hour, importance on a 1-10 scale), not by re-running
the implementation's own expression - an assertion that recomputes the formula
it is checking constrains nothing.
"""

import os
import subprocess
import sys

import pytest
from pydantic import ValidationError

from mind.cognitive_architecture.memory.models import VectorDBMetadata
from mind.cognitive_architecture.memory.retrieval import (
    ImportanceTerm,
    RecencyTerm,
    RelevanceTerm,
    RetrievalContext,
    RetrievalWeights,
    ScoredCandidate,
    candidate_pool_size,
    collect_raw_scores,
    default_terms,
    rank,
    reinforced_time,
    score_pool,
    should_reinforce,
    term_statistics,
)

# 0.995 ** 10 and 0.995 ** 1000, to 8dp. Literals rather than expressions so that
# changing the decay base, or dropping the game-minutes-to-hours conversion,
# breaks these tests instead of silently travelling into them.
DECAY_10_HOURS = 0.95111014
DECAY_1000_HOURS = 0.00665397

NOW = 100_000  # elapsed game minutes


def make_candidate(
    memory_id: str,
    *,
    distance: float | None = 0.5,
    importance: float | None = 5.0,
    timestamp: int | None = NOW,
    decay_base: float | None = None,
    content: str = "a memory",
) -> ScoredCandidate:
    return ScoredCandidate(
        memory_id=memory_id,
        content=content,
        metadata=VectorDBMetadata(
            importance=importance, timestamp=timestamp, decay_base=decay_base
        ),
        distance=distance,
    )


def context(now: int | None = NOW) -> RetrievalContext:
    return RetrievalContext(query="anything", current_simulation_time=now)


def breakdowns_for(
    candidates: list[ScoredCandidate],
    weights: RetrievalWeights | None = None,
    now: int | None = NOW,
) -> list:
    """Score a pool, returned in input order.

    There is deliberately no single-candidate `score_of` helper any more.
    Normalization is min-max **over the pool**, so a lone candidate is degenerate
    on every term at once and always totals 0.5. A helper that hid that would
    invite tests which look like they constrain the arithmetic but constrain
    nothing.
    """
    return score_pool(candidates, context(now), weights or RetrievalWeights(), default_terms())


def totals_for(
    candidates: list[ScoredCandidate],
    weights: RetrievalWeights | None = None,
    now: int | None = NOW,
) -> list[float | None]:
    return [b.total for b in breakdowns_for(candidates, weights, now)]


def order_of(
    candidates: list[ScoredCandidate], weights: RetrievalWeights | None = None
) -> list[str]:
    ranked = rank(
        candidates, context(), weights or RetrievalWeights(), default_terms(), len(candidates)
    )
    return [candidate.memory_id for _, candidate in ranked]


class TestModuleBoundary:
    """The scorer must stay importable without a storage backend."""

    @staticmethod
    def _chromadb_loaded_after_importing(module_path: str) -> bool:
        """True iff importing `module_path` in a fresh interpreter pulls chromadb."""
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                f"import sys; import {module_path}; "
                "print('LOADED' if 'chromadb' in sys.modules else 'ABSENT')",
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode == 0, f"probe failed to import {module_path}: {result.stderr}"
        return "LOADED" in result.stdout

    def test_probe_detects_chromadb_when_it_is_present(self):
        """Known-positive control.

        Without this, the assertion below is satisfied just as well by a probe
        that can never say LOADED - a clean zero that never had the chance to be
        anything else. The storage module genuinely imports chromadb, so it must
        come back LOADED or the probe is not measuring what it claims.
        """
        assert self._chromadb_loaded_after_importing(
            "mind.cognitive_architecture.memory.vector_db_memory"
        )

    def test_retrieval_module_does_not_import_chromadb(self):
        """Importing the scorer must not drag in the vector store.

        Breaks the moment someone types `from .vector_db_memory import ...` into
        retrieval.py, which is exactly the change that would make these tests
        depend on a live index.
        """
        assert not self._chromadb_loaded_after_importing(
            "mind.cognitive_architecture.memory.retrieval"
        )


class TestKnownInputRanking:
    """Exact scores for exact inputs, with equal weights."""

    def test_ranks_three_candidates_by_hand_computed_score(self):
        best = make_candidate("best", distance=0.2, importance=9.0, timestamp=NOW)
        middle = make_candidate("middle", distance=0.5, importance=5.0, timestamp=NOW - 600)
        worst = make_candidate("worst", distance=0.9, importance=2.0, timestamp=NOW - 60_000)

        ranked = rank(
            [worst, best, middle], context(), RetrievalWeights(), default_terms(), top_k=3
        )

        assert [c.memory_id for _, c in ranked] == ["best", "middle", "worst"]

        # Min-max over the pool. `best` is the pool maximum on all three terms
        # and `worst` the pool minimum on all three, so they land exactly on the
        # interval endpoints - a property fixed-scale normalization does not have
        # and a sharp check that the normalizer ran at all.
        assert ranked[0][0].total == pytest.approx(1.0, abs=1e-9)
        assert ranked[2][0].total == pytest.approx(0.0, abs=1e-9)

        # `middle`, hand-computed against the pool's observed ranges:
        #   relevance  (0.5 - 0.1) / (0.8 - 0.1)                    = 0.5714286
        #   importance (0.5 - 0.2) / (0.9 - 0.2)                    = 0.4285714
        #   recency    (0.95111014 - 0.00665397) / (1 - 0.00665397) = 0.9507826
        assert ranked[1][0].total == pytest.approx(
            (0.5714286 + 0.4285714 + 0.9507826) / 3, abs=1e-5
        )

        # The raw, pre-normalization values stay on the terms' own absolute
        # scales, so the decay base and the /60 conversion remain pinned.
        assert ranked[1][0].raw_contributions["recency"] == pytest.approx(DECAY_10_HOURS, abs=1e-6)
        assert ranked[2][0].raw_contributions["recency"] == pytest.approx(
            DECAY_1000_HOURS, abs=1e-6
        )
        assert ranked[0][0].raw_contributions["importance"] == pytest.approx(0.9, abs=1e-9)

    def test_top_k_truncates_after_scoring_not_before(self):
        """The pool is scored whole; top_k selects from the ranking, not the input order."""
        candidates = [
            make_candidate("low", distance=0.9, importance=1.0),
            make_candidate("high", distance=0.1, importance=10.0),
            make_candidate("mid", distance=0.5, importance=5.0),
        ]

        ranked = rank(candidates, context(), RetrievalWeights(), default_terms(), top_k=1)

        assert [c.memory_id for _, c in ranked] == ["high"]

    def test_weights_are_ratios_not_required_to_sum_to_one(self):
        """Doubling every weight cannot change any score - they are relative."""
        pool = [
            make_candidate("a", distance=0.3, importance=7.0, timestamp=NOW - 600),
            make_candidate("b", distance=0.7, importance=3.0, timestamp=NOW - 6000),
        ]

        unit = totals_for(pool, RetrievalWeights(relevance=1.0, importance=1.0, recency=1.0))
        doubled = totals_for(pool, RetrievalWeights(relevance=2.0, importance=2.0, recency=2.0))

        assert unit == pytest.approx(doubled, abs=1e-12)

    def test_a_zero_weighted_term_cannot_influence_the_ranking(self):
        """Weighting relevance to zero must make an irrelevant memory competitive."""
        relevant_trivia = make_candidate("relevant", distance=0.0, importance=1.0)
        irrelevant_gravity = make_candidate("irrelevant", distance=1.0, importance=10.0)
        weights = RetrievalWeights(relevance=0.0, importance=1.0, recency=1.0)

        ranked = rank(
            [relevant_trivia, irrelevant_gravity], context(), weights, default_terms(), top_k=2
        )

        assert [c.memory_id for _, c in ranked] == ["irrelevant", "relevant"]


class TestMonotonicity:
    """Sweep one term across a pool with the other two held fixed.

    **This property changed shape under min-max normalization**, and the change
    is real rather than cosmetic. Previously each candidate had an absolute score
    and monotonicity was a statement about one memory in isolation. A score now
    exists only relative to its pool, so "the total rises with the term" is only
    meaningful *within one pool of co-scored candidates* - which is what these
    build. Holding the other two terms fixed also makes them degenerate, so they
    contribute an identical constant to every candidate and the swept term alone
    decides the order: the sharpest possible form of the property.
    """

    def test_ranking_follows_relevance_when_the_other_terms_are_fixed(self):
        pool = [
            make_candidate(f"d{i}", distance=d)
            for i, d in enumerate([1.0, 0.8, 0.6, 0.4, 0.2, 0.0])
        ]

        assert order_of(pool) == ["d5", "d4", "d3", "d2", "d1", "d0"], (
            "score must rise as cosine distance falls; using raw distance instead "
            "of (1 - distance) inverts this"
        )

    def test_ranking_follows_importance_when_the_other_terms_are_fixed(self):
        pool = [make_candidate(f"i{v}", importance=float(v)) for v in range(11)]

        assert order_of(pool) == [f"i{v}" for v in range(10, -1, -1)]

    def test_ranking_follows_age_when_the_other_terms_are_fixed(self):
        # 0, 1, 6, 24 and 240 game hours old.
        ages = [0, 60, 360, 1440, 14400]
        pool = [make_candidate(f"t{i}", timestamp=NOW - m) for i, m in enumerate(ages)]

        assert order_of(pool) == ["t0", "t1", "t2", "t3", "t4"], (
            "recency must decay with age; a decay base above 1.0 inverts this"
        )

    def test_totals_are_strictly_ordered_not_merely_sorted(self):
        """Distinct inputs must produce distinct scores.

        `order_of` above would be satisfied by an implementation that collapsed
        every candidate onto one value and returned them in input order, which is
        exactly what a broken normalizer does.
        """
        pool = [
            make_candidate(f"d{i}", distance=d)
            for i, d in enumerate([1.0, 0.8, 0.6, 0.4, 0.2, 0.0])
        ]

        totals = totals_for(pool)

        assert len(set(totals)) == len(totals)
        assert totals == sorted(totals)


class TestNormalizationBounds:
    """Every term stays inside [0, 1] across its full legal input domain."""

    @pytest.mark.parametrize("distance", [0.0, 0.5, 1.0, 1.5, 2.0])
    def test_relevance_stays_in_unit_range_across_the_full_cosine_domain(self, distance):
        """Chroma cosine distance spans [0, 2]; above 1.0 similarity is negative.

        Dropping the clamp makes distance=2.0 score -1.0, which would rank an
        opposed memory below a term that cannot otherwise go negative and quietly
        distort every total that included it.
        """
        value = RelevanceTerm().score(make_candidate("m", distance=distance), context())
        assert 0.0 <= value <= 1.0

    @pytest.mark.parametrize("importance", [0.0, 1.0, 5.0, 10.0])
    def test_importance_stays_in_unit_range(self, importance):
        value = ImportanceTerm().score(make_candidate("m", importance=importance), context())
        assert 0.0 <= value <= 1.0

    def test_importance_endpoints_map_to_the_unit_endpoints(self):
        """Breaks if the 1-10 scale is divided by anything but 10."""
        assert ImportanceTerm().score(make_candidate("m", importance=10.0), context()) == 1.0
        assert ImportanceTerm().score(make_candidate("m", importance=0.0), context()) == 0.0

    @pytest.mark.parametrize("age_minutes", [0, 1, 60, 100_000, 10**6])
    def test_recency_stays_in_unit_range_across_extreme_ages(self, age_minutes):
        value = RecencyTerm().score(
            make_candidate("m", timestamp=NOW - age_minutes), context(now=NOW)
        )
        assert 0.0 <= value <= 1.0

    def test_a_brand_new_memory_scores_exactly_one_on_recency(self):
        assert RecencyTerm().score(make_candidate("m", timestamp=NOW), context()) == 1.0


class TestMinMaxNormalization:
    """Park's normalization, and the cases where it has no range to work with."""

    def test_the_pool_extremes_land_on_the_interval_endpoints(self):
        """The defining property. Under fixed scales none of these reach 0 or 1."""
        pool = [
            make_candidate("mid", distance=0.5, importance=5.0),
            make_candidate("best", distance=0.4, importance=6.0),
            make_candidate("worst", distance=0.6, importance=4.0),
        ]

        by_id = dict(zip([c.memory_id for c in pool], breakdowns_for(pool)))

        assert by_id["best"].contributions["relevance"] == pytest.approx(1.0)
        assert by_id["worst"].contributions["relevance"] == pytest.approx(0.0)
        assert by_id["mid"].contributions["relevance"] == pytest.approx(0.5)

    def test_a_narrow_raw_band_is_stretched_to_the_full_interval(self):
        """Why min-max and equal weights are a package.

        These three differ by 0.02 of raw cosine similarity - the narrow band
        real embeddings occupy. Under fixed scales that band contributes almost
        nothing against a recency term spanning the whole interval, whatever the
        weights say. Min-max restores its influence.
        """
        pool = [make_candidate(f"c{i}", distance=d) for i, d in enumerate([0.50, 0.49, 0.48])]

        relevance = [b.contributions["relevance"] for b in breakdowns_for(pool)]

        assert relevance == pytest.approx([0.0, 0.5, 1.0], abs=1e-9)
        # The raw values really were nearly identical.
        raw = [b.raw_contributions["relevance"] for b in breakdowns_for(pool)]
        assert max(raw) - min(raw) == pytest.approx(0.02, abs=1e-9)

    def test_a_term_that_cannot_discriminate_contributes_the_midpoint(self):
        """min == max means the term carries no information for this pool.

        0.5 rather than 1.0: candidates with different abstention patterns divide
        by different live weights, so a constant is not a uniform shift and does
        move the ranking. The neutral midpoint is the least distorting choice.
        """
        pool = [
            make_candidate("a", distance=0.5, importance=5.0, timestamp=NOW),
            make_candidate("b", distance=0.5, importance=5.0, timestamp=NOW),
        ]

        for breakdown in breakdowns_for(pool):
            assert breakdown.contributions["relevance"] == 0.5
            assert breakdown.contributions["importance"] == 0.5
            assert breakdown.contributions["recency"] == 0.5
            assert breakdown.total == pytest.approx(0.5, abs=1e-9)

    def test_a_single_candidate_is_scored_rather_than_dropped(self):
        """A pool of one is degenerate on every term at once.

        Abstaining on a degenerate term would be cleaner in principle, but it
        would drop the only candidate of a perfectly valid top_k=1 query and
        return nothing. It must score.
        """
        ranked = rank(
            [make_candidate("only")], context(), RetrievalWeights(), default_terms(), top_k=1
        )

        assert [c.memory_id for _, c in ranked] == ["only"]
        assert ranked[0][0].total == pytest.approx(0.5, abs=1e-9)

    def test_a_term_only_one_candidate_scored_cannot_discriminate(self):
        """A documented consequence, not a defect - and a real behaviour change.

        When exactly one candidate has a timestamp, the recency range collapses
        to a point, so recency contributes the neutral midpoint and cannot
        separate that candidate from an abstaining one. Recency is inherently
        comparative under min-max: with a single observation there is nothing to
        be recent *relative to*.

        This is why the store-level backstory regression test weights relevance
        to zero and supplies two timestamped memories - stated here so the
        behaviour is visible rather than smoothed away.
        """
        pool = [
            make_candidate("dated", timestamp=NOW),
            make_candidate("undated", timestamp=None),
        ]

        dated, undated = breakdowns_for(pool)

        assert dated.contributions["recency"] == 0.5
        assert undated.abstained == ["recency"]
        assert dated.total == pytest.approx(undated.total, abs=1e-9)

    def test_raw_contributions_survive_normalization(self):
        """The absolute values stay available for diagnosis and measurement."""
        pool = [
            make_candidate("a", distance=0.2, importance=8.0, timestamp=NOW),
            make_candidate("b", distance=0.6, importance=3.0, timestamp=NOW - 600),
        ]

        a, b = breakdowns_for(pool)

        assert a.raw_contributions["relevance"] == pytest.approx(0.8, abs=1e-9)
        assert b.raw_contributions["importance"] == pytest.approx(0.3, abs=1e-9)
        assert b.raw_contributions["recency"] == pytest.approx(DECAY_10_HOURS, abs=1e-6)


class TestAbstention:
    """Missing data must not be voted on in either direction."""

    def test_missing_timestamp_abstains_on_recency(self):
        breakdown = breakdowns_for([make_candidate("m", timestamp=None)])[0]

        assert breakdown.abstained == ["recency"]
        assert "recency" not in breakdown.contributions

    def test_missing_current_time_abstains_on_recency(self):
        breakdown = breakdowns_for([make_candidate("m")], now=None)[0]

        assert breakdown.abstained == ["recency"]

    def test_missing_distance_abstains_rather_than_scoring_perfect_relevance(self):
        """A backend that reported no distance told us nothing about relevance.

        The prior behaviour substituted 1.0, which does not merely lose
        information - it ranks the unmeasured candidate above every genuine match.
        """
        breakdown = breakdowns_for([make_candidate("m", distance=None)])[0]

        assert breakdown.abstained == ["relevance"]
        assert breakdown.contributions.get("relevance") is None

    def test_never_rated_importance_abstains_rather_than_voting_a_midpoint(self):
        breakdown = breakdowns_for([make_candidate("m", importance=None)])[0]

        assert breakdown.abstained == ["importance"]

    def test_an_abstaining_candidate_is_excluded_from_that_terms_min_and_max(self):
        """The consistent extension of abstention into the normalizer.

        Abstention already removes a candidate from a term's *weight*; letting it
        influence that term's *range* would reintroduce, through the normalizer,
        exactly the vote it was excluded from. Here the untimestamped candidate
        must not affect the recency range, which the two timestamped ones define
        between them.
        """
        pool = [
            make_candidate("fresh", timestamp=NOW),
            make_candidate("stale", timestamp=NOW - 60_000),
            make_candidate("undated", timestamp=None),
        ]

        raw = collect_raw_scores(pool, context(), default_terms())
        stats = term_statistics(raw, default_terms())

        assert stats["recency"].count == 2, "the abstainer must not be counted"
        assert stats["recency"].maximum == pytest.approx(1.0, abs=1e-9)
        assert stats["recency"].minimum == pytest.approx(DECAY_1000_HOURS, abs=1e-6)

    def test_an_untimestamped_memory_ranks_neither_first_nor_last(self):
        """The paired assertion: both sentinel failures must be excluded.

        Scoring an unknown recency as 0 would sink this memory below a decade-old
        one; scoring it 1.0 - the behaviour this replaces - floats it above a
        memory formed seconds ago. Because config-seeded backstory is exactly the
        untimestamped case and lived memories are exactly the timestamped one,
        the 1.0 sentinel meant hardcoded backstory permanently outranked
        experience, by a margin that widened with playtime.

        Relevance and importance are matched across all three at 0.5, so the
        untimestamped memory's renormalized total is exactly 0.5 and must land
        between a fresh memory (recency 1.0) and a 1000-game-hour-old one.
        """
        fresh = make_candidate("fresh", distance=0.5, importance=5.0, timestamp=NOW)
        seeded = make_candidate("seeded", distance=0.5, importance=5.0, timestamp=None)
        ancient = make_candidate("ancient", distance=0.5, importance=5.0, timestamp=NOW - 60_000)

        ranked = rank(
            [seeded, ancient, fresh], context(), RetrievalWeights(), default_terms(), top_k=3
        )
        order = [c.memory_id for _, c in ranked]

        assert order[0] != "seeded", "an unscored memory must not rank first (the 1.0 sentinel)"
        assert order[-1] != "seeded", "an unscored memory must not rank last (the 0 sentinel)"
        assert order == ["fresh", "seeded", "ancient"]

        seeded_breakdown = next(b for b, c in ranked if c.memory_id == "seeded")
        assert seeded_breakdown.abstained == ["recency"]
        assert seeded_breakdown.total == pytest.approx(0.5, abs=1e-9)

    def test_abstention_renormalizes_onto_the_same_scale(self):
        """An abstaining candidate stays comparable, not merely un-penalized.

        `complete` and `partial` are identical on relevance and importance and
        differ only in that `partial` has no timestamp. Both are the pool maximum
        on every term they scored, so both normalize to 1.0 on those terms - and
        because the weight renormalization divides by only the live weight, they
        must tie at exactly 1.0 despite dividing by 3 and 2 respectively. That
        tie is the composition of min-max with abstention, and it is the
        interaction most likely to be subtly wrong.
        """
        pool = [
            make_candidate("complete", distance=0.2, importance=8.0, timestamp=NOW),
            make_candidate("partial", distance=0.2, importance=8.0, timestamp=None),
            make_candidate("worse", distance=0.9, importance=2.0, timestamp=NOW - 60_000),
        ]

        complete, partial, worse = totals_for(pool)

        assert complete == pytest.approx(1.0, abs=1e-9)
        assert partial == pytest.approx(1.0, abs=1e-9)
        assert worse == pytest.approx(0.0, abs=1e-9)

        breakdowns = breakdowns_for(pool)
        assert breakdowns[0].live_weight == 3.0
        assert breakdowns[1].live_weight == 2.0

    def test_a_candidate_with_nothing_known_is_dropped(self):
        blank = ScoredCandidate(
            memory_id="blank", content="?", metadata=VectorDBMetadata(), distance=None
        )
        real = make_candidate("real")

        ranked = rank([blank, real], context(now=None), RetrievalWeights(), default_terms(), 5)

        assert [c.memory_id for _, c in ranked] == ["real"]
        assert breakdowns_for([blank, real], now=None)[0].total is None

    def test_all_weight_on_an_abstaining_term_yields_no_score(self):
        """Not a total of 0.0 - nothing that was weighted could be measured."""
        weights = RetrievalWeights(relevance=0.0, importance=0.0, recency=1.0)
        breakdown = breakdowns_for([make_candidate("m", timestamp=None)], weights)[0]

        assert breakdown.total is None


class TestWeightValidation:
    """No configuration may produce a negative coefficient."""

    @pytest.mark.parametrize("field", ["relevance", "importance", "recency"])
    def test_negative_weights_are_rejected_at_construction(self, field):
        """The documented prior failure this closes:

            VectorDBQuery(importance_weight=0.6, recency_weight=0.6)

        constructed cleanly and expressed relevance as the leftover
        `1 - 0.6 - 0.6 = -0.2`, a formula that actively preferred *irrelevant*
        memories for the whole session with nothing logged. Relevance is now an
        explicit non-negative weight, so the state is unreachable.
        """
        with pytest.raises(ValidationError):
            RetrievalWeights(**{field: -0.1})

    def test_all_zero_weights_are_rejected(self):
        with pytest.raises(ValidationError):
            RetrievalWeights(relevance=0.0, importance=0.0, recency=0.0)

    def test_unknown_weight_field_is_rejected(self):
        with pytest.raises(ValidationError):
            RetrievalWeights(relevence=1.0)

    def test_weights_are_frozen(self):
        weights = RetrievalWeights()
        with pytest.raises(ValidationError):
            weights.relevance = 5.0

    def test_defaults_are_parks_equal_weights(self):
        """Park: "in our implementation, all alphas are set to 1"."""
        weights = RetrievalWeights()
        assert (weights.relevance, weights.importance, weights.recency) == (1.0, 1.0, 1.0)

    def test_weight_for_an_undeclared_term_raises(self):
        """A term with no weight is a wiring bug; defaulting it to 0.0 would
        delete a scoring dimension while every query kept returning results."""
        with pytest.raises(KeyError):
            RetrievalWeights().weight_for("relationship")


class TestPerMemoryDecayOverride:
    """The one forward hook built now: NPC-406's per-memory decay rate."""

    def test_a_memory_may_carry_its_own_decay_base(self):
        """A term reading its base only from a module constant could not accept a
        per-memory rate without being rewritten, so the read exists from day one.
        Nothing writes the field yet.

        0.5 per game hour over 2 hours is 0.25 - far below the 0.995 default's
        0.99 - so this cannot pass if the override is ignored.
        """
        fast_fading = make_candidate("fast", timestamp=NOW - 120, decay_base=0.5)

        assert RecencyTerm().score(fast_fading, context()) == pytest.approx(0.25, abs=1e-9)

    def test_absent_decay_base_uses_the_default(self):
        default_decay = make_candidate("default", timestamp=NOW - 120)

        assert RecencyTerm().score(default_decay, context()) == pytest.approx(0.995**2, abs=1e-12)

    def test_an_override_of_zero_is_read_as_a_value_not_as_unset(self):
        """`decay_base or default` would silently swap a falsy 0.0 for the default.

        The field is bounded `gt=0.0` at its writer, so a stored 0.0 is
        unreachable in practice - this drives the read path directly, via
        model_construct, to pin that the term branches on `is None` rather than
        on truthiness. Without it, the `or` spelling passes every other test here
        and the distinction survives only as a comment.
        """
        candidate = ScoredCandidate(
            memory_id="zero",
            content="forgotten instantly",
            metadata=VectorDBMetadata.model_construct(timestamp=NOW - 120, decay_base=0.0),
        )

        assert RecencyTerm().score(candidate, context()) == 0.0

    def test_a_zero_decay_base_is_rejected_where_it_is_written(self):
        with pytest.raises(ValidationError):
            VectorDBMetadata(decay_base=0.0)

    @pytest.mark.parametrize("bad_base", [1.5, 2.0, -0.5])
    def test_an_out_of_range_decay_base_is_rejected_where_it_is_written(self, bad_base):
        """A base above 1.0 makes recency GROW with age and escape [0, 1].

        Rejected at construction rather than clamped at use, so the bad value
        fails at its writer instead of being silently corrected at every read.
        """
        with pytest.raises(ValidationError):
            VectorDBMetadata(decay_base=bad_base)


class TestClockBoundary:
    """A restart under a retained collection can produce a future timestamp."""

    def test_a_future_timestamp_is_clamped_and_warned_about(self, caplog):
        import logging

        future = make_candidate("future", timestamp=NOW + 5000)

        with caplog.at_level(logging.WARNING, logger="mind"):
            value = RecencyTerm().score(future, context())

        assert value == 1.0, "clamped to 'just now' rather than scoring above 1.0"
        assert any("future" in record.getMessage() for record in caplog.records), (
            "clamping must not be silent - the arithmetic is contained but the data is wrong"
        )


class TestRecencyReinforcement:
    """The EMA that carries a repeatedly-recalled memory back toward 'recent'.

    Every test here is chosen to FAIL under at least one of the two endpoint
    behaviours. A test that passes at alpha=0.0, alpha=1.0 and alpha=0.3 alike
    would be describing something all three share and would constrain nothing
    about the EMA.
    """

    ALPHA = 0.3

    def test_alpha_of_one_reproduces_park_last_access_exactly(self):
        """Endpoint, not an approximation: the anchor becomes 'now'."""
        assert reinforced_time(previous_anchor=0.0, now=6000.0, alpha=1.0) == 6000.0

    def test_alpha_of_zero_reproduces_creation_time_decay_exactly(self):
        """Endpoint: retrieval moves nothing, which is W1's shipped behaviour."""
        assert reinforced_time(previous_anchor=0.0, now=6000.0, alpha=0.0) == 0.0

    def test_one_retrieval_is_less_recent_than_being_created_now(self):
        """**The assertion the whole EMA exists for.**

        Under pure last-access (alpha=1.0) a memory recalled once and a memory
        formed at that instant are indistinguishable - one retrieval erases the
        memory's entire age. Under the EMA the recalled memory keeps (1 - alpha)
        of its age, so it must remain strictly the older of the two.

        This test is red at alpha=1.0 by construction, which is what makes it
        evidence rather than decoration.
        """
        now = 6000.0  # 100 game hours after creation
        recalled_once = reinforced_time(previous_anchor=0.0, now=now, alpha=self.ALPHA)

        assert recalled_once < now, (
            "a single recall must not erase a memory's whole age (that is alpha=1.0)"
        )
        assert recalled_once > 0.0, "a recall must move the anchor at all (that is alpha=0.0)"
        # 70% of a 100-game-hour age survives one recall -> anchor at 30% of now.
        assert recalled_once == pytest.approx(1800.0, abs=1e-9)

    def test_repeated_retrieval_is_strictly_more_recent_than_a_single_one(self):
        now = 6000.0
        once = reinforced_time(0.0, now, self.ALPHA)

        five_times = 0.0
        for _ in range(5):
            five_times = reinforced_time(five_times, now, self.ALPHA)

        assert five_times > once
        # 0.7 ** 5 of the age survives -> anchor at (1 - 0.16807) * now.
        assert five_times == pytest.approx(now * (1 - 0.7**5), abs=1e-6)

    def test_anchor_is_monotonic_in_retrieval_count(self):
        now = 6000.0
        anchors = []
        anchor = 0.0
        for _ in range(6):
            anchor = reinforced_time(anchor, now, self.ALPHA)
            anchors.append(anchor)

        assert anchors == sorted(anchors)
        assert len(set(anchors)) == len(anchors), "each recall must move the anchor"
        assert all(a < now for a in anchors), "the anchor must never reach or pass the present"

    def test_two_retrievals_halve_the_effective_age(self):
        """Pins the documented meaning of the default alpha.

        The constant is justified in constants.py as "two retrievals halve a
        memory's effective age"; if the default moves, that sentence must move
        with it.
        """
        from mind.constants import DEFAULT_RECENCY_REINFORCEMENT_ALPHA

        now = 6000.0
        anchor = 0.0
        for _ in range(2):
            anchor = reinforced_time(anchor, now, DEFAULT_RECENCY_REINFORCEMENT_ALPHA)

        remaining_age = now - anchor
        assert remaining_age == pytest.approx(now * 0.5, rel=0.05)

    def test_recency_term_decays_from_the_reinforced_anchor(self):
        """A reinforced memory scores as younger than its creation stamp implies."""
        stale = ScoredCandidate(
            memory_id="stale",
            content="never recalled",
            metadata=VectorDBMetadata(timestamp=NOW - 6000, effective_time=NOW - 6000),
        )
        reinforced = ScoredCandidate(
            memory_id="reinforced",
            content="recalled often",
            metadata=VectorDBMetadata(timestamp=NOW - 6000, effective_time=NOW - 600),
        )

        assert RecencyTerm().score(reinforced, context()) > RecencyTerm().score(stale, context())

    def test_recency_falls_back_to_creation_for_rows_predating_the_field(self):
        """Data compatibility for an already-persisted collection.

        A memory written before effective_time existed has only a timestamp, and
        must still score rather than abstain.
        """
        legacy = ScoredCandidate(
            memory_id="legacy",
            content="written by an older schema",
            metadata=VectorDBMetadata(timestamp=NOW - 600, effective_time=None),
        )

        assert RecencyTerm().score(legacy, context()) == pytest.approx(DECAY_10_HOURS, abs=1e-6)


class TestReinforcementRefusal:
    """When a retrieval must NOT write an anchor back."""

    def test_missing_readings_do_not_reinforce(self):
        assert should_reinforce(None, 100.0) is False
        assert should_reinforce(100.0, None) is False

    def test_a_clock_behind_the_anchor_does_not_reinforce(self):
        """The scenario-restart guard, and the reason the EMA does not worsen D-5.

        A restart resets elapsed game minutes while a retained collection keeps
        its old anchors. Feeding that reset clock into the EMA would *persist* a
        pulled-back anchor, and successive retrievals would drag it down until it
        fell below `now` - at which point elapsed goes positive, the clamp's
        warning stops firing, and the corruption becomes invisible exactly when
        it has finished happening. Refusing to write keeps the damage read-time
        only, where it already was.
        """
        assert should_reinforce(100_000.0, 5.0) is False

    def test_a_normal_forward_clock_does_reinforce(self):
        """Control: without this, a guard that refused everything would pass above."""
        assert should_reinforce(100.0, 6000.0) is True
        assert should_reinforce(100.0, 100.0) is True


class TestCandidatePoolSize:
    """The fix for scoring a pool the vector index already narrowed to top_k."""

    def test_pool_is_much_wider_than_top_k(self):
        """At the production top_k of 2, a pool of 2 means the weighted score
        sorts two items and can never surface anything cosine missed."""
        assert candidate_pool_size(top_k=2, collection_count=1000) > 2

    def test_pool_respects_the_floor_for_small_top_k(self):
        assert candidate_pool_size(top_k=1, collection_count=1000) == 30

    def test_pool_scales_with_large_top_k(self):
        assert candidate_pool_size(top_k=50, collection_count=1000) == 500

    def test_pool_is_clamped_to_the_collection_size(self):
        """The index cannot return rows that do not exist."""
        assert candidate_pool_size(top_k=2, collection_count=3) == 3

    def test_empty_collection_yields_an_empty_pool(self):
        assert candidate_pool_size(top_k=5, collection_count=0) == 0
