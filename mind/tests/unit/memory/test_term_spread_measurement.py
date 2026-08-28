"""Measures the REALIZED range of each retrieval term on our own data.

The argument for min-max normalization is that the three terms occupy very
different realized ranges despite all being nominally [0, 1], so equal weights
over fixed absolute scales would mean equal *coefficients* on unequal ranges
rather than equal *influence*. Park et al. state their use of min-max scaling
flat, with no rationale, no comparison, and an ablation that never varied the
normalization or the weights - so that argument cannot be sourced from the paper
and has to be checked here.

This is offline and deterministic: real sentence-transformer embeddings, a fixed
corpus, a fixed importance distribution and fixed timestamps. **No LLM call.**

It mirrors production rather than approximating it: the pool it measures is the
cosine top-`candidate_pool_size(...)` of a larger store, which is exactly what
`VectorDBMemory.search` scores. Measuring over the whole store instead would
overstate the relevance spread, because narrowing to the cosine top-N is itself
what compresses that band.

Run it with output shown:

    PYTHONPATH=$PWD/src:$PWD python -m pytest \\
        tests/unit/memory/test_term_spread_measurement.py -s -q
"""

import uuid

import pytest

from mind.cognitive_architecture.memory.models import VectorDBMetadata
from mind.cognitive_architecture.memory.retrieval import (
    ScoredCandidate,
    candidate_pool_size,
    collect_raw_scores,
    default_terms,
    term_statistics,
)
from mind.cognitive_architecture.memory.vector_db_memory import VectorDBMemory

NOW = 100_000  # elapsed game minutes, ~69 game days
QUERY = "what happened recently at the forge and with my work"

# A blacksmith NPC's memory stream. Deliberately mixed: some squarely on-topic
# for the query above, some adjacent, some unrelated - the shape a real stream
# has after a few in-game weeks.
CORPUS = [
    "I hammered out a set of horseshoes for the miller's cart horse",
    "The forge bellows tore along the seam and I patched it with hide",
    "A traveller commissioned a hunting knife with an antler handle",
    "I quenched a blade too fast and it cracked down the middle",
    "Ore delivery arrived short by two sacks and I argued with the carter",
    "I taught the apprentice to read the colour of hot steel",
    "The anvil came loose on its stump and rang wrong all morning",
    "I finished the ceremonial sword for the captain of the guard",
    "Charcoal ran low so I banked the fire early",
    "A farmer brought a broken plough blade to be rewelded",
    "I burned my forearm reaching across the hearth",
    "The guild inspector approved my maker's mark",
    "I sharpened every blade in the barracks in one long day",
    "Rain came through the roof and hissed on the coals",
    "I traded a set of nails for a sack of barley",
    "The apprentice quit without warning and left his apron folded",
    "I stayed late refitting a hinge for the chapel door",
    "My tongs slipped and a billet rolled into the ash",
    "A merchant offered to buy my whole stock of knives",
    "I reforged my father's old hammer head onto a new haft",
    "The market square was crowded for the harvest festival",
    "I drank too much cider at the tavern and slept badly",
    "My sister sent word that she is coming in the spring",
    "The village dogs got into the butcher's yard again",
    "I walked to the river and watched the herons at dusk",
    "A funeral procession passed while I was shuttering the shop",
    "The baker's oven collapsed and half the street smelled of smoke",
    "I lost a wager on the wrestling at the fair",
    "Snow closed the mountain road for six days",
    "A minstrel sang a song I had not heard since childhood",
    "The well water tasted of iron for a week",
    "I mended my boots rather than buy new ones",
    "A cat had kittens under the woodpile",
    "The tax collector came and I paid in coin",
    "I dreamed of my mother's kitchen and woke unsettled",
    "The miller's daughter smiled at me across the square",
    "A stray dog followed me home and I fed it scraps",
    "I planted onions in the strip behind the workshop",
    "Thunder split a tree on the ridge above the village",
    "I sat with the old smith while he told war stories",
]

# Importance as a reflection node plausibly emits it: clustered toward the
# middle, endpoints rare. Models seldom answer 1 or 10 on a 1-10 rubric.
IMPORTANCES = [
    4.0,
    3.0,
    5.0,
    6.0,
    4.0,
    7.0,
    3.0,
    8.0,
    2.0,
    4.0,
    5.0,
    6.0,
    4.0,
    3.0,
    3.0,
    7.0,
    4.0,
    3.0,
    6.0,
    8.0,
    4.0,
    3.0,
    6.0,
    2.0,
    4.0,
    5.0,
    4.0,
    3.0,
    4.0,
    5.0,
    2.0,
    3.0,
    4.0,
    3.0,
    6.0,
    6.0,
    4.0,
    3.0,
    4.0,
    7.0,
]

# Ages spread across roughly the last 40 game days, densest recently - the shape
# a decaying memory stream has.
AGES_IN_MINUTES = [
    0,
    120,
    300,
    600,
    900,
    1440,
    2000,
    2880,
    3600,
    4320,
    5000,
    6000,
    7200,
    8640,
    10000,
    11520,
    13000,
    14400,
    16000,
    17280,
    19000,
    20160,
    22000,
    23040,
    25000,
    27000,
    28800,
    31000,
    33000,
    34560,
    37000,
    40000,
    43200,
    46000,
    48000,
    51840,
    54000,
    57600,
    60000,
    63000,
]


@pytest.fixture(scope="module")
def measured_stats():
    """Embed the corpus, take the cosine pool, measure each term's raw range."""
    from chromadb.api.client import SharedSystemClient

    SharedSystemClient.clear_system_cache()
    store = VectorDBMemory(collection_name=f"test_spread_{uuid.uuid4().hex[:12]}")

    for content, importance, age in zip(CORPUS, IMPORTANCES, AGES_IN_MINUTES):
        store.add_memory(content=content, importance=importance, timestamp=NOW - age)

    # Exactly what search() does: over-fetch by cosine, then score that pool.
    pool_size = candidate_pool_size(top_k=2, collection_count=store.collection.count())
    raw_results = store.collection.query(
        query_embeddings=[store.encoder.encode(QUERY, show_progress_bar=False).tolist()],
        n_results=pool_size,
        include=["documents", "metadatas", "distances"],
    )

    candidates = [
        ScoredCandidate(
            memory_id=memory_id,
            content=content,
            metadata=VectorDBMetadata.model_validate(metadata or {}),
            distance=distance,
        )
        for memory_id, content, metadata, distance in zip(
            raw_results["ids"][0],
            raw_results["documents"][0],
            raw_results["metadatas"][0],
            raw_results["distances"][0],
        )
    ]

    from mind.cognitive_architecture.memory.retrieval import RetrievalContext

    context = RetrievalContext(query=QUERY, current_simulation_time=NOW)
    terms = default_terms()
    stats = term_statistics(collect_raw_scores(candidates, context, terms), terms)

    SharedSystemClient.clear_system_cache()
    return {"stats": stats, "pool_size": len(candidates), "corpus_size": len(CORPUS)}


def test_report_realized_term_spreads(measured_stats):
    """Print the measurement. Always passes; the assertions are below."""
    stats = measured_stats["stats"]

    print(
        f"\n\nRealized raw term ranges over the cosine top-{measured_stats['pool_size']} "
        f"of {measured_stats['corpus_size']} memories"
    )
    print(f"query: {QUERY!r}\n")
    print(f"{'term':<12}{'min':>10}{'max':>10}{'spread':>10}{'n':>5}")
    print("-" * 47)
    for name in ("relevance", "importance", "recency"):
        s = stats[name]
        print(f"{name:<12}{s.minimum:>10.4f}{s.maximum:>10.4f}{s.spread:>10.4f}{s.count:>5}")
    print()


def test_every_term_scored_the_whole_pool(measured_stats):
    """Control. Spreads are only comparable if each term measured the same set."""
    stats = measured_stats["stats"]
    pool_size = measured_stats["pool_size"]

    assert pool_size > 1, "a one-candidate pool has no spread to measure"
    for name in ("relevance", "importance", "recency"):
        assert stats[name].count == pool_size


def test_relevance_occupies_a_far_narrower_band_than_recency(measured_stats):
    """**The load-bearing empirical claim behind min-max normalization.**

    If this fails, the case for min-max rests on Park's precedent alone rather
    than on our data - which is materially weaker and worth knowing. Written from
    the measured output rather than assumed: see the numbers this file prints.
    """
    stats = measured_stats["stats"]

    assert stats["relevance"].spread < stats["recency"].spread, (
        "relevance spread "
        f"{stats['relevance'].spread:.4f} is not narrower than recency spread "
        f"{stats['recency'].spread:.4f} - the empirical argument for min-max does "
        "not hold on this data and the decision rests on precedent alone"
    )


def test_no_term_uses_more_than_its_nominal_range(measured_stats):
    """Every raw score stays inside [0, 1], so the spreads are comparable."""
    stats = measured_stats["stats"]
    for name in ("relevance", "importance", "recency"):
        assert 0.0 <= stats[name].minimum <= 1.0
        assert 0.0 <= stats[name].maximum <= 1.0
