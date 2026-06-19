"""Data models shared by lexical search storage and query backends."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class SearchableObject:
    """A backend-neutral unit of text addressable by lexical search.

    Args:
        id: Stable identifier for this searchable field.
        owner_node_id: Existing chunk/node ID returned after a lexical match.
        searchable_text: Canonical, untokenized text stored across backends.
        field_type: Kind of extracted field, such as ``table_row``.
        metadata: Optional source information useful to external backends.
        schema_version: Persistence schema version.
    """

    id: str
    owner_node_id: str
    searchable_text: str
    field_type: str
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Serialize this object into a JSON-compatible dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SearchableObject:
        """Validate and deserialize a searchable object dictionary."""
        required = ("id", "owner_node_id", "searchable_text", "field_type")
        missing = [key for key in required if not str(data.get(key, "")).strip()]
        if missing:
            raise ValueError(f"Missing searchable object fields: {', '.join(missing)}")
        return cls(
            id=str(data["id"]),
            owner_node_id=str(data["owner_node_id"]),
            searchable_text=str(data["searchable_text"]),
            field_type=str(data["field_type"]),
            metadata=dict(data.get("metadata") or {}),
            schema_version=int(data.get("schema_version", 1)),
        )
