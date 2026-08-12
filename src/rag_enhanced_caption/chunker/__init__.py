from typing import Any

from .parsers.semantic import chunk_markdown as semantic_chunk_raw
from .dispatcher import chunk_markdown as semantic_chunk_with_metadata


def semantic_chunk_markdown_strings(*args: Any, **kwargs: Any) -> list[str]:
    """切分 Markdown 并以格式化后的 Markdown 字符串列表返回。

    Args:
        *args: 转发给 ``semantic_chunk_with_metadata`` 的位置参数。
        **kwargs: 转发给 ``semantic_chunk_with_metadata`` 的关键字参数。

    Returns:
        格式化后的非空 Markdown 分块。
    """
    kwargs["output_format"] = "markdown"
    return semantic_chunk_with_metadata(*args, **kwargs)


__all__ = [
    "semantic_chunk_raw",
    "semantic_chunk_with_metadata",
    "semantic_chunk_markdown_strings",
]
