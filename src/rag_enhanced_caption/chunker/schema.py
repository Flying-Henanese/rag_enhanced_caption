"""
定义了核心的数据结构。
"""
from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class SemanticChunk:
    content: str
    header: str = ""
    element_type: str = "text"
    metadata: Dict[str, Any] = field(default_factory=dict)
