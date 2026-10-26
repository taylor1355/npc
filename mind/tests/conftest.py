"""Suite-wide fixtures.

The default embedding model is EmbeddingGemma 2: a ~1.5 GB download whose weights carry
vision and audio towers the memory store never uses. Roughly sixty tests construct a
VectorDBMemory without naming a model, so loading the real default for each of them
would make the suite's cost a function of a model nothing in those tests is about.
They are about retrieval scoring, id stability, recency and plumbing, none of which
depends on WHICH sentence-embedding model produced the vectors - only that a real one
did, because several assert semantic neighbourhoods.

So the default model's NAME is honoured (the store still stamps collections with it and
applies its prompt) while a small real encoder stands in for the weights. Tests that
need the real model opt in with MIND_TEST_REAL_EMBEDDING_MODEL=1 (see
tests/unit/memory/test_embedding_model_migration.py::TestRealDefaultModel).

A test that patches SentenceTransformer itself is unaffected: its patch wraps this one.
"""

import atexit
import os
import shutil
import tempfile
from pathlib import Path

import pytest
from sentence_transformers import SentenceTransformer as RealSentenceTransformer

# project_config reads credentials/api_keys.cfg AT IMPORT TIME and the import chain
# reaches it from unit tests. Point it at a throwaway MIND_HOME holding dummy keys so
# the suite never needs, reads or writes the developer's real credentials and cache.
# Set before any mind import below: this line is load-bearing.
_TEST_MIND_HOME = Path(tempfile.mkdtemp(prefix="mind-test-home-"))
(_TEST_MIND_HOME / "credentials").mkdir()
(_TEST_MIND_HOME / "credentials" / "api_keys.cfg").write_text(
    "OPENROUTER_API_KEY: 'test-dummy'\nPERPLEXITY_API_KEY: 'test-dummy'\n"
)
os.environ["MIND_HOME"] = str(_TEST_MIND_HOME)
atexit.register(shutil.rmtree, _TEST_MIND_HOME, ignore_errors=True)

import mind.cognitive_architecture.memory.vector_db_memory as vector_db_memory  # noqa: E402
from mind.constants import DEFAULT_EMBEDDING_MODEL  # noqa: E402

STAND_IN_MODEL = "all-MiniLM-L6-v2"

_loaded: dict[str, RealSentenceTransformer] = {}


def _build_encoder(name, *args, **kwargs):
    if name == DEFAULT_EMBEDDING_MODEL:
        name = STAND_IN_MODEL
    # One instance per model per session: the weights are read-only, and reloading them
    # for every test is the cost this fixture exists to avoid.
    if name not in _loaded:
        _loaded[name] = RealSentenceTransformer(name, *args, **kwargs)
    return _loaded[name]


@pytest.fixture(scope="session", autouse=True)
def _light_default_embedding_model():
    """Session-scoped, because module-scoped fixtures build their stores before any
    function-scoped autouse fixture runs and would otherwise load the real weights."""
    if os.environ.get("MIND_TEST_REAL_EMBEDDING_MODEL") == "1":
        yield
        return
    patcher = pytest.MonkeyPatch()
    patcher.setattr(vector_db_memory, "SentenceTransformer", _build_encoder)
    yield
    patcher.undo()
