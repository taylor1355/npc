"""Offline A/B of the recent-events rendering cost (NPC-1335).

Measures ONE thing, deterministically and without an LLM: how many prompt
tokens the recent-events buffer costs rendered as ``pformat`` (what the
reflection node used to send) versus ``format_recent_events`` (what it sends
now), over identical committed buffers.

Why this rather than a live per-cycle figure. A live decision-cycle total moves
with model, memory-store contents, and the LLM's own output; it cannot isolate
a rendering change. It is also not reproducible from this repo: the NPC-1318
harness that produced the 5,367-token baseline was never committed, and the
observation fixtures it was built from contain no MindEvents at all — so a
cycle measured from them has an EMPTY buffer and says nothing about this cost
either way.

Every token counted here is an uncached one. ``{recent_events}`` sits below the
reflection prompt's cache breakpoint, in the per-call dynamic suffix, so the
buffer is billed at the full input rate on every call to every NPC.

Usage (from the ``mind`` project root):

    PYTHONPATH=$PWD/src:$PWD python tools/measure_recent_events_rendering.py

Counts use ``cl100k_base``, which is an approximation for non-OpenAI models —
the ratio is the durable result, not the absolute counts.
"""

from __future__ import annotations

import subprocess
from pprint import pformat

import tiktoken

from mind.cognitive_architecture.nodes.formatting import format_recent_events
from mind.cognitive_architecture.observations import MindEventType
from tests.fixtures.observations import (
    create_conversation_events,
    create_movement_events,
    create_social_events,
)

ENCODING_NAME = "cl100k_base"


def _resolved_commit() -> str:
    """The commit these numbers describe, so the figure carries its own scope."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main() -> None:
    encoding = tiktoken.get_encoding(ENCODING_NAME)

    def count(text: str) -> int:
        return len(encoding.encode(text))

    buffers = {
        "movement (3 events, no conversation)": create_movement_events(),
        "social (7 events, no conversation)": create_social_events(),
        "conversation (8 events, 1 conversation)": create_conversation_events(),
    }

    print(f"commit={_resolved_commit()}  encoding={ENCODING_NAME}  tiktoken={tiktoken.__version__}")
    print(f"{'buffer':<44}{'pformat':>10}{'__str__':>10}{'saved':>10}")

    for label, events in buffers.items():
        before = count(pformat(events))
        after = count(format_recent_events(events))
        saved = (before - after) / before if before else 0.0
        print(f"{label:<44}{before:>10}{after:>10}{saved:>9.0%}")

    # Isolated, because this one event type dominates any buffer it appears in
    # and is deliberately left uncompacted (NPC-1298).
    solo = [
        event
        for event in create_conversation_events()
        if event.event_type == MindEventType.INTERACTION_OBSERVATION
    ]
    before = count(pformat(solo))
    after = count(format_recent_events(solo))
    print(
        f"{'  of which: one INTERACTION_OBSERVATION alone':<44}"
        f"{before:>10}{after:>10}{(before - after) / before:>9.0%}"
    )


if __name__ == "__main__":
    main()
