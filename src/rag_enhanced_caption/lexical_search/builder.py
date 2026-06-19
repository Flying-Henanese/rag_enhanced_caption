"""Build stable searchable objects from enriched chunks."""

from __future__ import annotations

import hashlib
from typing import Any

from .extractors import DeterministicSearchFieldExtractor, SearchFieldExtractor
from .schema import SearchableObject


def build_searchable_objects(
    chunk: dict[str, Any], extractor: SearchFieldExtractor | None = None
) -> list[SearchableObject]:
    """Build deduplicated searchable objects for one chunk.

    Args:
        chunk: Enriched chunk containing an ``id`` and source content.
        extractor: Optional field extractor. The deterministic extractor is used
            by default.

    Returns:
        Stable searchable objects linked to the chunk ID.

    Raises:
        ValueError: If the chunk has no ID.
    """
    owner_node_id = str(chunk.get("id") or "").strip()
    if not owner_node_id:
        raise ValueError("A searchable chunk must have an id")

    active_extractor = extractor or DeterministicSearchFieldExtractor()
    seen: set[tuple[str, str]] = set()
    objects: list[SearchableObject] = []
    for field in active_extractor.extract(chunk):
        text = field.text.strip()
        key = (field.field_type, text)
        if not text or key in seen:
            continue
        seen.add(key)
        digest = hashlib.sha256(
            f"{owner_node_id}\0{field.field_type}\0{text}".encode("utf-8")
        ).hexdigest()[:16]
        objects.append(
            SearchableObject(
                id=f"{owner_node_id}:{field.field_type}:{digest}",
                owner_node_id=owner_node_id,
                searchable_text=text,
                field_type=field.field_type,
                metadata=field.metadata,
            )
        )
    return objects
