"""Helpers for resolving chunk identifiers across persisted artifacts."""

from __future__ import annotations


def resolve_parent_chunk_id(parent_id: str | None, child_chunk_id: str) -> str | None:
    """Resolve a short or complete parent ID to a complete chunk ID.

    Args:
        parent_id: Parent reference such as ``"5"`` or
            ``"document_chunk_5"``.
        child_chunk_id: Complete ID of the child chunk, used to recover the
            document-specific prefix for short parent references.

    Returns:
        The complete parent chunk ID, or ``None`` when no parent is present.
        If the child ID does not follow the ``<prefix>_chunk_<index>`` format,
        the original parent reference is returned unchanged.
    """
    if parent_id is None:
        return None

    normalized_parent_id = str(parent_id).strip()
    if not normalized_parent_id:
        return None
    if "_chunk_" in normalized_parent_id:
        return normalized_parent_id

    child_prefix, separator, _ = child_chunk_id.rpartition("_chunk_")
    if not separator:
        return normalized_parent_id
    return f"{child_prefix}_chunk_{normalized_parent_id}"
