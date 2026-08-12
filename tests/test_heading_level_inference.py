from rag_enhanced_caption.chunker import semantic_chunk_markdown_strings
from rag_enhanced_caption.chunker.parsers import semantic


def test_numbered_heading_recovers_hierarchy_in_header_path() -> None:
    md = """# 5.5.5 Third Level

Body under mis-leveled heading.
"""
    chunks = semantic.chunk_markdown(
        md,
        parser_config={"chunk_token_num": 512},
        embed_fn=None,
    )
    text_chunks = [c for c in chunks if c.element_type == "text" and c.content.strip()]

    assert text_chunks, "Expected at least one text chunk"
    assert text_chunks[0].header_path == ["5.5.5 Third Level"]
    assert text_chunks[0].metadata.get("heading_level") == 3


def test_numbered_heading_recovers_markdown_output_level() -> None:
    md = """# 5.5.5 Third Level

Body under mis-leveled heading.
"""
    formatted_chunks = semantic_chunk_markdown_strings(
        md,
        file_id="test",
        filename="test.md",
        parser_config={"chunk_token_num": 512},
        embed_fn=lambda texts: [[0.0] * 4 for _ in texts],
    )

    assert formatted_chunks, "Expected markdown output chunks"
    assert formatted_chunks[0].startswith("### 5.5.5 Third Level"), formatted_chunks[0]
