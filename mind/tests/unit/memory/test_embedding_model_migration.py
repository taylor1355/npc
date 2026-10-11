"""A memory collection must only ever be queried by the model that wrote its vectors.

Switching the default embedding model (all-MiniLM-L6-v2, 384-d -> EmbeddingGemma 2, 768-d)
strands every persisted collection. ChromaDB answers the width change with
InvalidDimension on the first query, which at least announces itself; a same-width change
of model is accepted and returns meaningless neighbours with nothing reporting it. These
tests pin the behaviour that closes both: each collection is stamped with the model that
wrote it, and a non-empty collection stamped otherwise (or not at all) is re-embedded from
its stored documents when opened.

The encoders here are deterministic fakes. What is under test is the migration, not the
quality of any embedding, and a fake whose width and identity the test controls is the only
way to produce "another model" at "the same width".
"""

import hashlib
import os
import uuid

import chromadb
import numpy as np
import pytest
from chromadb.api.client import SharedSystemClient

from mind.cognitive_architecture.memory.vector_db_memory import (
    EMBEDDING_MODEL_KEY,
    REEMBED_SUFFIX,
    VectorDBMemory,
    VectorDBQuery,
)
from mind.constants import DEFAULT_EMBEDDING_MODEL, EMBEDDING_MODEL_PROMPTS

LEGACY_MODEL = "legacy-model-4d"
NEW_MODEL = "new-model-6d"
SAME_WIDTH_OTHER_MODEL = "other-model-4d"
GEMMA_PROMPT = "task: sentence similarity | query: "


class FakeEncoder:
    """Deterministic embedder: a text maps to a unit vector of `width` by its hash and tag."""

    def __init__(self, width: int, tag: str):
        self.width = width
        self.tag = tag
        self.calls: list[tuple[object, str | None]] = []

    def encode(self, texts, prompt=None, show_progress_bar=False, **_kwargs):
        self.calls.append((texts, prompt))
        single = isinstance(texts, str)
        batch = [texts] if single else list(texts)
        rows = []
        for text in batch:
            digest = hashlib.sha256(f"{self.tag}|{prompt or ''}|{text}".encode()).digest()
            vec = np.array([digest[i] + 1 for i in range(self.width)], dtype=np.float32)
            rows.append(vec / np.linalg.norm(vec))
        out = np.stack(rows)
        return out[0] if single else out


@pytest.fixture(autouse=True)
def _isolated_chroma():
    SharedSystemClient.clear_system_cache()
    yield
    SharedSystemClient.clear_system_cache()


@pytest.fixture
def storage(tmp_path):
    return str(tmp_path / "chroma")


def _open(monkeypatch, storage, model, encoder, name="mind_a"):
    monkeypatch.setattr(
        "mind.cognitive_architecture.memory.vector_db_memory.SentenceTransformer",
        lambda requested: encoder,
    )
    return VectorDBMemory(collection_name=name, embedding_model=model, storage_path=storage)


def _legacy_collection(storage, name, rows, width, stamp=None):
    """A collection as an older release wrote it: no model stamp, vectors of `width`."""
    client = chromadb.PersistentClient(
        path=storage, settings=chromadb.Settings(anonymized_telemetry=False, allow_reset=True)
    )
    metadata = {"hnsw:space": "cosine"}
    if stamp:
        metadata[EMBEDDING_MODEL_KEY] = stamp
    collection = client.create_collection(name=name, metadata=metadata)
    legacy = FakeEncoder(width, "legacy")
    for row_id, document, meta in rows:
        kwargs = {"metadatas": [meta]} if meta else {}
        collection.add(
            ids=[row_id],
            embeddings=[legacy.encode(document).tolist()],
            documents=[document],
            **kwargs,
        )
    return client, collection


ROWS = [
    ("m1", "the orchard has apples", {"importance": 7.0, "timestamp": 10, "tags": ["food"]}),
    ("m2", "cass told me about the well", {"importance": 3.0, "timestamp": 20}),
    ("m3", "it rained all morning", {"timestamp": 30, "effective_time": 55, "last_accessed": 55}),
]


def _stored(collection):
    got = collection.get(include=["documents", "metadatas", "embeddings"])
    return {
        i: (d, m, np.asarray(e))
        for i, d, m, e in zip(
            got["ids"], got["documents"], got["metadatas"], got["embeddings"], strict=True
        )
    }


class TestFreshCollectionIsStamped:
    def test_a_new_collection_records_the_model_that_will_write_it(self, monkeypatch, storage):
        store = _open(monkeypatch, storage, NEW_MODEL, FakeEncoder(6, "new"))

        assert store.collection.metadata[EMBEDDING_MODEL_KEY] == NEW_MODEL
        assert store.collection.metadata["hnsw:space"] == "cosine"

    def test_clear_keeps_the_stamp(self, monkeypatch, storage):
        store = _open(monkeypatch, storage, NEW_MODEL, FakeEncoder(6, "new"))
        store.add_memory("something")

        store.clear()

        assert store.collection.metadata[EMBEDDING_MODEL_KEY] == NEW_MODEL
        assert store.collection.count() == 0


class TestReembedOnModelChange:
    def test_an_unstamped_narrow_collection_is_rewritten_at_the_new_width(
        self, monkeypatch, storage
    ):
        """The shipped situation: a 384-d MiniLM collection opened by a 768-d model."""
        _legacy_collection(storage, "mind_a", ROWS, width=4)
        new = FakeEncoder(6, "new")

        store = _open(monkeypatch, storage, NEW_MODEL, new)

        stored = _stored(store.collection)
        assert set(stored) == {"m1", "m2", "m3"}
        for row_id, document, _meta in ROWS:
            assert stored[row_id][0] == document
            np.testing.assert_allclose(stored[row_id][2], new.encode(document), rtol=1e-5)
            assert stored[row_id][2].shape == (6,)
        assert store.collection.metadata[EMBEDDING_MODEL_KEY] == NEW_MODEL

    def test_a_query_after_the_migration_no_longer_raises_invalid_dimension(
        self, monkeypatch, storage
    ):
        """The first-query failure ChromaDB gives on a width mismatch is gone."""
        _legacy_collection(storage, "mind_a", ROWS, width=4)
        store = _open(monkeypatch, storage, NEW_MODEL, FakeEncoder(6, "new"))

        import asyncio

        found = asyncio.run(store.search(VectorDBQuery(query="the orchard has apples", top_k=3)))

        assert {m.id for m in found} == {"m1", "m2", "m3"}
        assert found[0].content == "the orchard has apples"

    def test_documents_ids_and_metadata_survive_verbatim(self, monkeypatch, storage):
        _client, legacy = _legacy_collection(storage, "mind_a", ROWS, width=4)
        before = _stored(legacy)

        store = _open(monkeypatch, storage, NEW_MODEL, FakeEncoder(6, "new"))

        after = _stored(store.collection)
        assert set(after) == set(before)
        for row_id in before:
            assert after[row_id][0] == before[row_id][0]
            assert after[row_id][1] == before[row_id][1]

    def test_the_same_width_from_another_model_is_still_rewritten(self, monkeypatch, storage):
        """Identity, not width: equal widths are accepted by ChromaDB and silently wrong."""
        _legacy_collection(storage, "mind_a", ROWS, width=4, stamp=LEGACY_MODEL)
        other = FakeEncoder(4, "other")

        store = _open(monkeypatch, storage, SAME_WIDTH_OTHER_MODEL, other)

        stored = _stored(store.collection)
        for row_id, document, _meta in ROWS:
            np.testing.assert_allclose(stored[row_id][2], other.encode(document), rtol=1e-5)
        assert store.collection.metadata[EMBEDDING_MODEL_KEY] == SAME_WIDTH_OTHER_MODEL

    def test_a_collection_already_written_by_this_model_is_not_touched(self, monkeypatch, storage):
        _legacy_collection(storage, "mind_a", ROWS, width=6, stamp=NEW_MODEL)
        new = FakeEncoder(6, "new")

        store = _open(monkeypatch, storage, NEW_MODEL, new)

        assert new.calls == []
        assert store.collection.count() == len(ROWS)

    def test_a_row_stored_without_metadata_stays_without_it(self, monkeypatch, storage):
        rows = [("bare", "a memory nobody rated", None), *ROWS]
        _legacy_collection(storage, "mind_a", rows, width=4)

        store = _open(monkeypatch, storage, NEW_MODEL, FakeEncoder(6, "new"))

        stored = _stored(store.collection)
        assert stored["bare"][1] is None
        assert stored["m1"][1] == dict(ROWS[0][2])

    def test_more_rows_than_one_batch(self, monkeypatch, storage):
        rows = [(f"r{i}", f"memory number {i}", {"timestamp": i}) for i in range(150)]
        _legacy_collection(storage, "mind_a", rows, width=4)
        new = FakeEncoder(6, "new")

        store = _open(monkeypatch, storage, NEW_MODEL, new)

        assert store.collection.count() == 150
        assert len(new.calls) == 3  # 64 + 64 + 22
        assert set(_stored(store.collection)) == {r[0] for r in rows}

    def test_an_empty_unstamped_collection_is_recreated_with_the_stamp(self, monkeypatch, storage):
        _legacy_collection(storage, "mind_a", [], width=4)

        store = _open(monkeypatch, storage, NEW_MODEL, FakeEncoder(6, "new"))

        assert store.collection.metadata[EMBEDDING_MODEL_KEY] == NEW_MODEL
        assert store.collection.count() == 0

    def test_no_staging_collection_is_left_behind(self, monkeypatch, storage):
        client, _ = _legacy_collection(storage, "mind_a", ROWS, width=4)

        _open(monkeypatch, storage, NEW_MODEL, FakeEncoder(6, "new"))

        assert [c.name for c in client.list_collections()] == ["mind_a"]


class TestMigrationFailsSafe:
    def test_a_failure_mid_rebuild_leaves_the_original_collection_intact(
        self, monkeypatch, storage
    ):
        rows = [(f"r{i}", f"memory number {i}", {"timestamp": i}) for i in range(100)]
        client, legacy = _legacy_collection(storage, "mind_a", rows, width=4)

        class Dies(FakeEncoder):
            def encode(self, texts, **kwargs):
                if len(self.calls) >= 1:
                    raise RuntimeError("encoder died on the second batch")
                return super().encode(texts, **kwargs)

        with pytest.raises(RuntimeError, match="second batch"):
            _open(monkeypatch, storage, NEW_MODEL, Dies(6, "new"))

        assert [c.name for c in client.list_collections()] == ["mind_a"]
        survivor = client.get_collection("mind_a")
        assert survivor.count() == 100
        assert EMBEDDING_MODEL_KEY not in (survivor.metadata or {})
        assert _stored(survivor)["r5"][2].shape == (4,)

    def test_a_document_that_cannot_be_embedded_aborts_instead_of_dropping_the_row(
        self, monkeypatch, storage
    ):
        client, legacy = _legacy_collection(storage, "mind_a", ROWS, width=4)
        legacy.add(ids=["hollow"], embeddings=[[0.1, 0.2, 0.3, 0.4]], metadatas=[{"timestamp": 1}])

        with pytest.raises(RuntimeError, match="hollow"):
            _open(monkeypatch, storage, NEW_MODEL, FakeEncoder(6, "new"))

        assert client.get_collection("mind_a").count() == len(ROWS) + 1
        assert [c.name for c in client.list_collections()] == ["mind_a"]

    def test_a_staging_collection_beside_the_original_is_a_partial_rebuild_and_is_discarded(
        self, monkeypatch, storage
    ):
        client, _ = _legacy_collection(storage, "mind_a", ROWS, width=4)
        stale = client.create_collection(
            name=f"mind_a{REEMBED_SUFFIX}", metadata={"hnsw:space": "cosine"}
        )
        stale.add(ids=["half"], embeddings=[[1.0, 0.0]], documents=["half-written"])

        store = _open(monkeypatch, storage, NEW_MODEL, FakeEncoder(6, "new"))

        assert set(_stored(store.collection)) == {"m1", "m2", "m3"}
        assert [c.name for c in client.list_collections()] == ["mind_a"]

    def test_an_interrupted_swap_is_finished_not_restarted(self, monkeypatch, storage):
        """Original dropped, staging complete, rename never ran: promote it as it is."""
        client = chromadb.PersistentClient(
            path=storage, settings=chromadb.Settings(anonymized_telemetry=False, allow_reset=True)
        )
        new = FakeEncoder(6, "new")
        staging = client.create_collection(
            name=f"mind_a{REEMBED_SUFFIX}",
            metadata={"hnsw:space": "cosine", EMBEDDING_MODEL_KEY: NEW_MODEL},
        )
        for row_id, document, meta in ROWS:
            staging.add(
                ids=[row_id],
                embeddings=[new.encode(document).tolist()],
                documents=[document],
                metadatas=[meta],
            )
        assert "mind_a" not in {c.name for c in client.list_collections()}
        new.calls.clear()  # the fixture's own embedding of the staging rows

        store = _open(monkeypatch, storage, NEW_MODEL, new)

        assert store.collection.name == "mind_a"
        assert store.collection.count() == len(ROWS)
        assert new.calls == [], "a complete staging collection must be promoted, not rebuilt"
        assert [c.name for c in client.list_collections()] == ["mind_a"]


class TestCollectionLifecycleSeesStaging:
    def test_a_promotable_staging_collection_counts_as_an_existing_memory(
        self, monkeypatch, storage
    ):
        client = chromadb.PersistentClient(
            path=storage, settings=chromadb.Settings(anonymized_telemetry=False, allow_reset=True)
        )
        client.create_collection(name=f"mind_a{REEMBED_SUFFIX}", metadata={"hnsw:space": "cosine"})

        assert VectorDBMemory.collection_exists(storage, "mind_a") is True

    def test_forgetting_a_mind_removes_its_staging_collection_too(self, monkeypatch, storage):
        client, _ = _legacy_collection(storage, "mind_a", ROWS, width=4)
        client.create_collection(name=f"mind_a{REEMBED_SUFFIX}", metadata={"hnsw:space": "cosine"})

        VectorDBMemory.delete_collection(storage, "mind_a")

        assert client.list_collections() == []


class TestPrompt:
    def test_the_default_model_embeds_with_the_similarity_prompt(self, monkeypatch, storage):
        enc = FakeEncoder(6, "gemma")
        store = _open(monkeypatch, storage, DEFAULT_EMBEDDING_MODEL, enc)

        store.add_memory("the orchard has apples")

        assert EMBEDDING_MODEL_PROMPTS[DEFAULT_EMBEDDING_MODEL] == GEMMA_PROMPT
        assert enc.calls[-1] == ("the orchard has apples", GEMMA_PROMPT)

    def test_a_query_is_embedded_with_the_same_prompt_as_a_stored_memory(
        self, monkeypatch, storage
    ):
        import asyncio

        enc = FakeEncoder(6, "gemma")
        store = _open(monkeypatch, storage, DEFAULT_EMBEDDING_MODEL, enc)
        store.add_memory("the orchard has apples")

        found = asyncio.run(store.search(VectorDBQuery(query="the orchard has apples", top_k=1)))

        # Same text, same prompt, same vector: the query lands exactly on the memory.
        assert found and found[0].content == "the orchard has apples"
        assert [p for _t, p in enc.calls] == [GEMMA_PROMPT, GEMMA_PROMPT]

    def test_the_rebuild_applies_the_prompt_to_every_stored_document(self, monkeypatch, storage):
        _legacy_collection(storage, "mind_a", ROWS, width=4)
        enc = FakeEncoder(6, "gemma")

        _open(monkeypatch, storage, DEFAULT_EMBEDDING_MODEL, enc)

        assert [p for _t, p in enc.calls] == [GEMMA_PROMPT]
        assert list(enc.calls[0][0]) == [r[1] for r in ROWS]

    def test_a_model_without_a_prompt_gets_none(self, monkeypatch, storage):
        enc = FakeEncoder(6, "plain")
        store = _open(monkeypatch, storage, NEW_MODEL, enc)

        store.add_memory("x")

        assert enc.calls[-1] == ("x", None)


@pytest.mark.skipif(
    os.environ.get("MIND_TEST_REAL_EMBEDDING_MODEL") != "1",
    reason="downloads and loads the real default model (~1.5 GB); "
    "set MIND_TEST_REAL_EMBEDDING_MODEL=1",
)
class TestRealDefaultModel:
    """The one place the actual default model is loaded.

    Everything above uses fakes and the rest of the suite substitutes a small encoder
    (tests/conftest.py), so without this nothing proves the pinned sentence-transformers
    and transformers can load google/embeddinggemma-2 at all.
    """

    def test_the_default_model_loads_at_768_dimensions_and_ranks_a_paraphrase_first(self, tmp_path):
        import asyncio

        store = VectorDBMemory(
            collection_name=f"real_{uuid.uuid4().hex[:8]}", storage_path=str(tmp_path / "c")
        )
        store.add_memory("There is food at the orchard if you are hungry")
        store.add_memory("The river floods every spring")

        found = asyncio.run(
            store.search(VectorDBQuery(query="Where can I find something to eat?", top_k=2))
        )

        assert store.embedding_model == DEFAULT_EMBEDDING_MODEL
        assert store.encoder.get_embedding_dimension() == 768
        assert found[0].content.startswith("There is food")
        assert store.collection.metadata[EMBEDDING_MODEL_KEY] == DEFAULT_EMBEDDING_MODEL
