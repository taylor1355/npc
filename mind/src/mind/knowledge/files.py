"""Knowledge file enum with runtime validation."""

from enum import StrEnum
from pathlib import Path


class KnowledgeFile(StrEnum):
    NEEDS = "needs"
    MOVEMENT = "movement"
    INTERACTIONS = "interactions"
    ACTIVITY = "activity"


def _validate_knowledge_files():
    """Raise RuntimeError at import if enum and files disagree."""
    knowledge_dir = Path(__file__).parent
    missing = []
    for kf in KnowledgeFile:
        if not (knowledge_dir / f"{kf.value}.md").exists():
            missing.append(f"{kf.value}.md")
    if missing:
        raise RuntimeError(f"Missing knowledge files: {missing}")


_validate_knowledge_files()
