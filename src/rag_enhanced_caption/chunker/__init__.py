from .parsers.semantic import chunk_markdown as semantic_chunk_raw
from .dispatcher import chunk_markdown as semantic_chunk_with_metadata


def semantic_chunk_markdown_strings(*args, **kwargs):
    kwargs["output_format"] = "markdown"
    return semantic_chunk_with_metadata(*args, **kwargs)


__all__ = [
    "semantic_chunk_raw",
    "semantic_chunk_with_metadata",
    "semantic_chunk_markdown_strings",
]
