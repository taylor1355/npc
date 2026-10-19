"""No remark key, declaration kind or declaring parameter is spelled in ``src/``.

The ``ConversationMessage.declarations`` rule, extended to remarks (NPC-1916):
the simulation owns the grammar, so a remark it adds must reach the prompt and
the validator through the catalogue with no Python change here. A literal in
``src/`` is how that silently stops being true.

The vocabulary is read from the committed catalogue fixture, so this scan
widens by itself when the fixture is re-derived from a newer grammar. String
constants are scanned through the AST with docstrings excluded: prose explaining
why a remark must not be named is not a hardcoded remark.
"""

import ast
import re
from pathlib import Path

import pytest

from tests.fixtures.remark_catalogue import wire_catalogue

SRC = Path(__file__).parents[2] / "src" / "mind"


def _vocabulary() -> set[str]:
    words: set[str] = set()
    for row in wire_catalogue()["remarks"]:
        words.add(row["key"])
        if row["declares"]:
            words.add(row["declares"])
        words.update(row["declare_with"])
    return words


def _docstring_nodes(tree: ast.AST) -> set[int]:
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                ids.add(id(body[0].value))
    return ids


def _offenders(source: str, vocabulary: set[str]) -> list[tuple[int, str, str]]:
    tree = ast.parse(source)
    docstrings = _docstring_nodes(tree)
    patterns = {word: re.compile(rf"(?<![\w.]){re.escape(word)}(?![\w.])") for word in vocabulary}
    hits = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
            continue
        if id(node) in docstrings:
            continue
        for word, pattern in patterns.items():
            if _names(node.value, word, pattern):
                hits.append((node.lineno, node.value, word))
    return hits


def _names(literal: str, word: str, pattern: re.Pattern) -> bool:
    """Whether a string literal spells ``word`` as vocabulary.

    A compound word (``answer_who_told.told``, ``is_farewell``) is vocabulary
    wherever it appears as a whole token. A plain English word that is also a key
    (``response``, ``greeting``) counts only when it IS the literal, so a log line
    saying "after response" is prose rather than a hardcoded remark.
    """
    if literal.strip() == word:
        return True
    compound = "_" in word or "." in word
    return compound and bool(pattern.search(literal))


def test_the_vocabulary_is_not_empty():
    """An empty vocabulary would make the scan below pass over nothing."""
    assert len(_vocabulary()) >= 20


def test_src_names_no_remark_key_kind_or_declaring_parameter():
    vocabulary = _vocabulary()
    offenders = {
        str(path.relative_to(SRC)): hits
        for path in sorted(SRC.rglob("*.py"))
        if (hits := _offenders(path.read_text(encoding="utf-8"), vocabulary))
    }

    assert offenders == {}, (
        "src/ hardcodes simulation grammar vocabulary; read it from the remark "
        f"catalogue instead: {offenders}"
    )


@pytest.mark.parametrize(
    "source",
    [
        'def check(params):\n    return params.get("utterance_element") == "greeting"\n',
        'FLAG = "is_farewell"\n',
        'KEY = " response "\n',
        'KIND = "share_belief"\n',
        'x = f"say {1} answer_who_told.told"\n',
    ],
)
def test_the_scan_catches_a_hardcoded_remark(source):
    """Known positives: a clean report above is only evidence if these go red."""
    assert _offenders(source, _vocabulary())


def test_the_scan_ignores_docstrings_and_longer_identifiers():
    source = (
        'def f():\n    """A greeting is a remark; so is farewell."""\n'
        '    return "greeting_responses_total", "response_metadata"\n'
    )

    assert _offenders(source, _vocabulary()) == []
