"""
定义了核心的数据结构。
"""

from dataclasses import dataclass, field
from typing import Any


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
