"""Regression tests for the parent-child linkage fixes.

Covers two bugs that silently broke the multimodal parent/child pipeline:

1. ``cli._build_record`` persisted the chunker's raw ``parent_id`` (a semantic-block
   index like ``"0"``) verbatim, while record ids look like ``p_{file_id}_chunk_{n}``.
   The two namespaces never matched, so a child could never resolve its parent.

2. ``MarkdownMultimodalProcessor.enrich_chunks`` keyed ``chunk_map`` by ``chunk["id"]``
   while ``parent_id`` is a ``chunk_index``. The lookup ``parent_id in chunk_map`` was
   therefore always False, so captions were generated without parent context.
"""

import asyncio
from pathlib import Path

from rag_enhanced_caption.cli import _build_record
from rag_enhanced_caption.enhancer.processor import MarkdownMultimodalProcessor


def test_build_record_child_parent_id_resolves_to_real_parent_id():
    parent = {
        "id": "doc_chunk_0",
        "content": "parent text body",
        "file_id": "doc",
        "filename": "doc.md",
        "text_for_embedding": "parent text body",
        "metadata": {"element_type": "text", "parent_id": None, "chunk_index": 0},
    }
    child = {
        "id": "doc_chunk_1",
        "content": "![](img.png)",
        "file_id": "doc",
        "filename": "doc.md",
        "text_for_embedding": "a caption",
        "metadata": {"element_type": "Image", "parent_id": "0", "chunk_index": 1},
    }

    parent_idx, parent_doc = _build_record(parent, Path("doc.md"))
    child_idx, child_doc = _build_record(child, Path("doc.md"))

    # parent record gets the p_ prefix; child must point at that exact id.
    assert parent_doc["id"] == "p_doc_chunk_0"
    assert child_doc["parent_id"] == parent_doc["id"]
    # index_record must also carry the resolved parent_id for downstream consumers.
    assert child_idx["parent_id"] == parent_doc["id"]
    assert parent_idx["parent_id"] is None
    # chunk_index is preserved as a bridging fallback.
    assert child_doc["metadata"]["chunk_index"] == 1


def test_build_record_text_chunk_has_no_parent():
    text_chunk = {
        "id": "doc_chunk_0",
        "content": "body",
        "file_id": "doc",
        "filename": "doc.md",
        "text_for_embedding": "body",
        "metadata": {"element_type": "text", "parent_id": None, "chunk_index": 0},
    }
    idx, doc = _build_record(text_chunk, Path("doc.md"))
    assert doc["type"] == "parent"
    assert doc["parent_id"] is None
    assert idx["parent_id"] is None


def test_enrich_chunks_finds_parent_context_via_chunk_index_key():
    captured_prompts = []

    async def fake_vlm(user_prompt, system_prompt, image_base64=None, image_bytes=None):
        captured_prompts.append(user_prompt)
        return '{"summary": "S", "entities": []}'

    chunks = [
        {
            "content": "PARENT_TEXT_MARKER body of the section",
            "metadata": {"element_type": "text", "chunk_index": 0, "parent_id": None},
        },
        {
            "content": "<table><tr><td>a</td><td>b</td></tr></table>",
            "metadata": {"element_type": "Table", "chunk_index": 1, "parent_id": "0"},
        },
    ]

    proc = MarkdownMultimodalProcessor(vlm_func=fake_vlm, max_concurrency=1)
    asyncio.run(proc.enrich_chunks(chunks, base_dir="."))

    # Parent context was found and injected into the VLM prompt for the table chunk.
    assert any("PARENT_TEXT_MARKER" in p for p in captured_prompts)
    # Caption landed on the table chunk's embedding field.
    assert chunks[1]["text_for_embedding"] == "S"
    # Text chunk's embedding is its own content.
    assert chunks[0]["text_for_embedding"] == "PARENT_TEXT_MARKER body of the section"
