"""
语义分块调度器模块。

负责对外提供高层接口，它对底层的 `semantic.chunk_markdown` 核心函数进行包装，
在其返回纯文本切片的基础上，追加文件来源、索引等元数据（metadata），
使其输出格式满足向量数据库存入的标准格式。
"""
from __future__ import annotations

from typing import Any

from .parsers import semantic


def _build_chunk_records(text_chunks: list[str], file_id: str, filename: str) -> list[dict[str, Any]]:
    """
    将纯文本切片转换为带元数据的结构化记录字典。
    
    Args:
        text_chunks: 切分后的纯文本字符串列表。
        file_id: 文档在系统中的唯一标识。
        filename: 文档的原始文件名。
        
    Returns:
        包含 id, content, file_id, filename, chunk_index, source, chunk_id 等信息的字典列表。
    """
    records: list[dict[str, Any]] = []

    for idx, chunk_content in enumerate(text_chunks):
        text = (chunk_content or "").strip()
        # 跳过空切片
        if not text:
            continue

        records.append(
            {
                "id": f"{file_id}_chunk_{idx}",      # 全局唯一块ID
                "content": text,                     # 核心文本内容
                "file_id": file_id,                  # 关联文件ID
                "filename": filename,                # 来源文件名
                "chunk_index": idx,                  # 块的顺序索引
                "source": filename,                  # 数据源标识
                "chunk_id": f"{file_id}_chunk_{idx}", # 备用块ID
            }
        )

    return records


def chunk_markdown(
    markdown_content: str, 
    file_id: str = "default", 
    filename: str = "document.md", 
    parser_config: dict[str, Any] | None = None,
    embed_fn: Any | None = None
) -> list[dict[str, Any]]:
    """
    语义化切分 Markdown 并返回带元数据的记录。
    
    该函数是外部系统（如 FastAPI 接口、Celery Worker）调用语义分块的首选入口点。
    
    Args:
        markdown_content: 原始 Markdown 字符串。
        file_id: 业务系统指定的文档ID，用于追踪溯源。
        filename: 原始文档名称。
        parser_config: 配置参数字典，如 {"chunk_token_num": 512} 用于控制最大分块大小。
        embed_fn: 传入的向量化函数（用于后续的文本向量聚类）。不传会自动加载默认客户端。
        
    Returns:
        一系列经过清洗、封装、带有完整上下文路径的结构化文本记录。
    """
    # 核心调用：使用语义解析器对 Markdown 执行分块
    text_chunks = semantic.chunk_markdown(markdown_content, parser_config, embed_fn)
    # 结果封装：将纯文本包装成标准字典格式
    return _build_chunk_records(text_chunks, file_id, filename)
