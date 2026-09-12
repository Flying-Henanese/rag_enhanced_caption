"""
定义了核心的数据结构。
"""

from dataclasses import dataclass, field
from typing import Any


TEXTUAL_ELEMENT_TYPES: frozenset[str | None] = frozenset(
    {"text", "", None, "bullet_list_open", "ordered_list_open"}
)


def is_textual_element_type(element_type: str | None) -> bool:
    """Return whether an element should participate in ordinary text retrieval.

    Args:
        element_type: Parser element type stored on a semantic chunk.

    Returns:
        ``True`` for prose and structural list chunks.
    """
    return element_type in TEXTUAL_ELEMENT_TYPES


@dataclass
class SemanticChunk:
    """Represent one structure-aware Markdown chunk.

    Args:
        content: Markdown content of the chunk.
        header_path: Heading hierarchy containing the chunk.
        element_type: Semantic element type, such as ``text`` or ``Image``.
        text_for_embedding: Clean retrieval-facing text or summary.
        full_content: Complete content retained for retrieval-time expansion.
        parent_id: Optional parent chunk reference.
        metadata: Additional parser and source metadata.
    """

    content: str
    header_path: list[str] = field(default_factory=list)
    element_type: str = "text"
    text_for_embedding: str = ""
    full_content: str = ""
    parent_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
