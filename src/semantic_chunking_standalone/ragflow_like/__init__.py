"""
RAGFlow-like 语义分块模块入口。

本模块提供类似 RAGFlow 的核心 Markdown 语义分块能力，主要包含：
1. semantic_chunk_raw: 仅返回切分后的纯文本字符串列表。
2. semantic_chunk_with_metadata: 返回带有元数据（如 file_id, chunk_index 等）的字典列表，方便直接存入向量数据库。
"""
from .parsers.semantic import chunk_markdown as semantic_chunk_raw
from .dispatcher import chunk_markdown as semantic_chunk_with_metadata

__all__ = ["semantic_chunk_raw", "semantic_chunk_with_metadata"]
