"""Every module under src/ must import.

The gap this closes: nothing else in the project ever imports an unreferenced
module. ruff parses files without executing their imports, pytest only reaches
what a test pulls in, and the build step compiles nothing. So a module whose
import target was deleted, renamed, or never packaged sits in the tree looking
alive and fails only when someone finally imports it.

That is not hypothetical - it is why this test exists, and the first run found
two instances rather than the one it was written for ([NPC-1061]):

  * `interfaces/text_adventure_interface.py` imported `mind.simulators.
    text_adventure` against a `mind/simulators/` package that exists nowhere in
    the repo.
  * `prompts/` imported `llama_index`, which is not a dependency in
    pyproject.toml and is not in uv.lock.

Both had survived a ruff adoption that reordered their imports without ever
noticing they could not resolve. Both now live under `mind/archived/`, beside
the only code that ever referenced them.

Importing IS the assertion: `import_module` raises on an unresolvable import, so
a module that comes back at all has had its import graph executed.

The walk is deliberately filesystem-driven rather than `pkgutil.walk_packages`.
walk_packages recurses by importing each package and reading its `__path__`, and
it swallows the ImportError when that fails - so a broken or namespace package
silently takes its whole subtree out of the sweep, leaving a green result over
code nobody checked. `mind`, `mind/apis` and `mind/cognitive_architecture` are
all namespace packages (no `__init__.py`), and an earlier pkgutil-based draft of
this file skipped `mind.apis` entirely for that reason.
"""

import importlib
from pathlib import Path

import pytest

SOURCE_ROOT = Path(__file__).resolve().parents[2] / "src"


def _module_names() -> list[str]:
    """Dotted name of every .py file under src/, derived from its path alone."""
    names = []
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        parts = path.relative_to(SOURCE_ROOT).with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        names.append(".".join(parts))
    assert names, f"discovered no modules under {SOURCE_ROOT} - the walk is not looking"
    return names


@pytest.mark.parametrize("module_name", _module_names())
def test_module_imports(module_name: str) -> None:
    """Fails naming the module and the import that could not resolve."""
    importlib.import_module(module_name)


def test_the_walk_reaches_every_subtree() -> None:
    """Guards the guard.

    A walk that quietly stopped short would leave the unreached modules
    unchecked while this file still reported green - which is the exact failure
    the pkgutil draft had. The names below are deep and sit in four different
    subtrees, two of them under a namespace package with no `__init__.py`, so no
    single missed recursion can satisfy all of them.
    """
    discovered = set(_module_names())
    expected = {
        "mind.apis.langchain_llm",
        "mind.cognitive_architecture.nodes.reflection.node",
        "mind.interfaces.mcp.main",
        "mind.knowledge",
    }
    assert expected <= discovered, f"never discovered by the walk: {sorted(expected - discovered)}"
