"""Persistence interfaces for backend-neutral searchable objects."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

from .schema import SearchableObject


class SearchableObjectRepository(Protocol):
    """Storage boundary implemented by JSONL, Elasticsearch, or MongoDB."""

    def replace(self, objects: Iterable[SearchableObject]) -> None:
        """Replace all searchable objects in this repository.

        Args:
            objects: Complete replacement set of searchable objects.
        """

    def load(self) -> list[SearchableObject]:
        """Load all searchable objects from this repository.

        Returns:
            Persisted searchable objects.
        """


class JsonlSearchableObjectRepository:
    """Persist searchable objects as newline-delimited JSON.

    Args:
        path: Destination JSONL file path.
    """

    def __init__(self, path: Path | str) -> None:
        """Initialize a JSONL searchable-object repository.

        Args:
            path: Destination JSONL file path.
        """
        self.path = Path(path)

    def replace(self, objects: Iterable[SearchableObject]) -> None:
        """Atomically replace the JSONL file contents.

        Args:
            objects: Complete replacement set of searchable objects.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        with temporary_path.open("w", encoding="utf-8") as output:
            for obj in objects:
                output.write(json.dumps(obj.to_dict(), ensure_ascii=False) + "\n")
        temporary_path.replace(self.path)

    def load(self) -> list[SearchableObject]:
        """Load searchable objects, returning an empty list if absent.

        Returns:
            Persisted searchable objects, or an empty list when the file is absent.

        Raises:
            ValueError: If a JSONL row is invalid or fails schema validation.
        """
        if not self.path.exists():
            return []
        objects: list[SearchableObject] = []
        with self.path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    objects.append(SearchableObject.from_dict(json.loads(line)))
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid searchable object at {self.path}:{line_number}"
                    ) from exc
        return objects
