"""
语义分块调度器模块。

负责对外提供高层接口，它对底层的 `semantic.chunk_markdown` 核心函数进行包装，
在其返回结构化对象的基础上，追加文件来源、索引等全局元数据（metadata），
使其输出格式满足向量数据库存入的标准格式。
"""

from __future__ import annotations

from typing import Any
import copy

from .parsers import semantic
from .schema import SemanticChunk


def _build_chunk_records(
    semantic_chunks: list[SemanticChunk], file_id: str, filename: str
) -> list[dict[str, Any]]:
    """
    将语义块转换为带全局元数据的结构化记录字典。

    Args:
        semantic_chunks: 结构化语义块列表。
        file_id: 文档在系统中的唯一标识。
        filename: 文档的原始文件名。

    Returns:
        包含 id, content, file_id, filename, chunk_index, source, chunk_id 以及 header, element_type 的字典列表。
    """
    records: list[dict[str, Any]] = []

    for idx, chunk in enumerate(semantic_chunks):
        text = (chunk.content or "").strip()
        if not text:
            continue

        # 将 SemanticChunk 特有的 metadata 与全局 metadata 合并
        chunk_metadata = copy.deepcopy(chunk.metadata)
        chunk_metadata.update(
            {
                "header_path": chunk.header_path,
                "element_type": chunk.element_type,
                "file_id": file_id,
                "filename": filename,
                "chunk_index": idx,
                "source": filename,
                "parent_id": chunk.parent_id,
            }
        )

        records.append(
            {
                "id": f"{file_id}_chunk_{idx}",
                "content": text,
                "text_for_embedding": chunk.text_for_embedding,
                "full_content": chunk.full_content,
                "parent_id": chunk.parent_id,
                "file_id": file_id,
                "filename": filename,
                "chunk_index": idx,
                "source": filename,
                "chunk_id": f"{file_id}_chunk_{idx}",
                "metadata": chunk_metadata,
            }
        )

    return records


def chunk_markdown(
    markdown_content: str,
    file_id: str = "default",
    filename: str = "document.md",
    parser_config: dict[str, Any] | None = None,
    embed_fn: Any | None = None,
) -> list[dict[str, Any]]:
    """
    语义化切分 Markdown 并返回带元数据的记录。

    该函数是外部系统调用语义分块的首选入口点。

    Args:
        markdown_content: 原始 Markdown 字符串。
        file_id: 业务系统指定的文档ID，用于追踪溯源。
        filename: 原始文档名称。
        parser_config: 配置参数字典，如 {"chunk_token_num": 512} 用于控制最大分块大小。
        embed_fn: 传入的向量化函数（用于后续的文本向量聚类）。不传会自动加载默认客户端。

    Returns:
        一系列经过清洗、封装、带有完整上下文路径和元素类型的结构化记录字典。
    """
    semantic_chunks = semantic.chunk_markdown(markdown_content, parser_config, embed_fn)
    return _build_chunk_records(semantic_chunks, file_id, filename)
