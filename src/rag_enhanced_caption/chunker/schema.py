"""
定义了核心的数据结构。
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


@dataclass
class SemanticChunk:
    content: str
    header_path: List[str] = field(default_factory=list)
    element_type: str = "text"
    text_for_embedding: str = ""
    full_content: str = ""
    parent_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
