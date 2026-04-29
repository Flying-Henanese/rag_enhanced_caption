from __future__ import annotations

from typing import Any

from .parsers import semantic


def _build_chunk_records(text_chunks: list[str], file_id: str, filename: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for idx, chunk_content in enumerate(text_chunks):
        text = (chunk_content or "").strip()
        if not text:
            continue

        records.append(
            {
                "id": f"{file_id}_chunk_{idx}",
                "content": text,
                "file_id": file_id,
                "filename": filename,
                "chunk_index": idx,
                "source": filename,
                "chunk_id": f"{file_id}_chunk_{idx}",
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
    """
    text_chunks = semantic.chunk_markdown(markdown_content, parser_config, embed_fn)
    return _build_chunk_records(text_chunks, file_id, filename)
