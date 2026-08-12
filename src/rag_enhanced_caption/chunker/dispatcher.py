"""
语义分块调度器模块。

负责对外提供高层接口，它对底层的 `semantic.chunk_markdown` 核心函数进行包装，
在其返回结构化对象的基础上，追加文件来源、索引等全局元数据（metadata），
使其输出格式满足向量数据库存入的标准格式。
"""

from __future__ import annotations

from typing import Any, Literal
import copy

from .parsers import semantic
from .schema import SemanticChunk


def _format_chunk_as_markdown(chunk: SemanticChunk) -> str:
    """Format a semantic chunk into markdown with level-aware heading prefix."""
    content = (chunk.content or "").strip()
    if not content:
        return ""

    header_path = [h.strip() for h in (chunk.header_path or []) if h and h.strip()]
    if not header_path:
        return content

    metadata_level = 0
    if isinstance(chunk.metadata, dict):
        try:
            metadata_level = int(chunk.metadata.get("heading_level", 0) or 0)
        except (TypeError, ValueError):
            metadata_level = 0

    heading_level = min(max(metadata_level or len(header_path), 1), 6)
    heading_prefix = "#" * heading_level
    heading = " > ".join(header_path)
    return f"{heading_prefix} {heading}\n\n{content}"


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
    output_format: Literal["records", "markdown"] = "records",
) -> list[dict[str, Any]] | list[str]:
    """
    语义化切分 Markdown 并返回带元数据的记录。

    该函数是外部系统调用语义分块的首选入口点。

    Args:
        markdown_content: 原始 Markdown 字符串。
        file_id: 业务系统指定的文档ID，用于追踪溯源。
        filename: 原始文档名称。
        parser_config: 配置参数字典，如 {"chunk_token_num": 512} 用于控制最大分块大小。
        embed_fn: 传入的向量化函数（用于后续的文本向量聚类）。不传会自动加载默认客户端。
        output_format: 输出结构；``records`` 返回记录字典，``markdown`` 返回文本块。

    Returns:
        ``records`` 模式返回带完整上下文路径和元素类型的记录字典；
        ``markdown`` 模式返回格式化后的 Markdown 文本块。
    """
    semantic_chunks = semantic.chunk_markdown(markdown_content, parser_config, embed_fn)
    if output_format == "markdown":
        return [
            formatted
            for formatted in (
                _format_chunk_as_markdown(chunk) for chunk in semantic_chunks
            )
            if formatted
        ]
    return _build_chunk_records(semantic_chunks, file_id, filename)
