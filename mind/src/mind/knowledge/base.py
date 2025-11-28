"""KnowledgeBase for loading simulation knowledge."""

from functools import lru_cache
from pathlib import Path

from .files import KnowledgeFile

KNOWLEDGE_DIR = Path(__file__).parent


class KnowledgeBase:
    @classmethod
    def get(cls, files: KnowledgeFile | list[KnowledgeFile]) -> str:
        """Load knowledge file content(s).

        Args:
            files: Single KnowledgeFile or list of KnowledgeFiles

        Returns:
            Content of file(s), joined with double newlines if multiple
        """
        if isinstance(files, list):
            return "\n\n".join(cls._get_single(f) for f in files)
        return cls._get_single(files)

    @classmethod
    @lru_cache(maxsize=20)
    def _get_single(cls, file: KnowledgeFile) -> str:
        """Load a single knowledge file content."""
        path = KNOWLEDGE_DIR / f"{file.value}.md"
        return path.read_text().strip()
