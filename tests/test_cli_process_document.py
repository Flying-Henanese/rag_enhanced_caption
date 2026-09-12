import asyncio
from collections.abc import Awaitable, Callable
import json
from pathlib import Path
from typing import Any

import pytest

from rag_enhanced_caption import cli
from rag_enhanced_caption.enhancer.vlm_client import VlmCall


def test_process_document_uses_enrich_chunks_and_completes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_md = tmp_path / "sample.md"
    output_dir = tmp_path / "out"
    input_md.write_text("# Title\n\nSome text.\n", encoding="utf-8")

    def fake_chunk_markdown(**kwargs: Any) -> list[dict[str, Any]]:
        return [
            {
                "id": "chunk_1",
                "content": "# Title\n\nSome text.\n",
                "file_id": "sample",
                "filename": "sample.md",
                "metadata": {"element_type": "text"},
            },
            {
                "id": "chunk_2",
                "content": "1. first\n2. second",
                "file_id": "sample",
                "filename": "sample.md",
                "metadata": {"element_type": "ordered_list_open"},
            },
        ]

    async def fake_vlm(*args: object, **kwargs: object) -> str:
        return '{"summary":"ok","entities":[]}'

    class FakeProcessor:
        def __init__(self, vlm_func: VlmCall, max_concurrency: int = 5) -> None:
            self.vlm_func = vlm_func
            self.max_concurrency = max_concurrency

        async def enrich_chunks(
            self,
            chunks: list[dict[str, Any]],
            image_resolver: Callable[[str], Awaitable[bytes | None]] | None = None,
            base_dir: str | Path = ".",
        ) -> list[dict[str, Any]]:
            for chunk in chunks:
                chunk["text_for_embedding"] = chunk["content"]
            return chunks

    monkeypatch.setattr(cli, "semantic_chunk_with_metadata", fake_chunk_markdown)
    monkeypatch.setattr(cli, "create_default_vlm_client", lambda: fake_vlm)
    monkeypatch.setattr(cli, "MarkdownMultimodalProcessor", FakeProcessor)

    asyncio.run(cli.process_document(input_md, output_dir))

    index_path = output_dir / "sample_index.jsonl"
    docstore_path = output_dir / "sample_docstore.jsonl"
    enhanced_md_path = output_dir / "sample_enhanced.md"

    assert index_path.exists()
    assert docstore_path.exists()
    assert enhanced_md_path.exists()

    index_lines = index_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(index_lines) == 2
    index_records = [json.loads(line) for line in index_lines]
    first = index_records[0]
    assert first["id"].startswith("p_")
    assert first["metadata"]["type"] == "parent"
    assert first["text_for_embedding"] == "# Title\n\nSome text.\n"
    assert all(record["text_for_embedding"] for record in index_records)
    assert all(record["metadata"]["type"] == "parent" for record in index_records)

    docstore_lines = docstore_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(docstore_lines) == len(index_lines)
    docstore_records = [json.loads(line) for line in docstore_lines]
    first_docstore = docstore_records[0]
    assert first_docstore["id"] == first["id"]
    assert {record["id"] for record in index_records} == {
        record["id"] for record in docstore_records
    }
    assert {
        "id",
        "type",
        "parent_id",
        "text_for_embedding",
        "full_content",
        "header_path",
        "element_type",
        "entities",
        "metadata",
    } <= first_docstore.keys()

    enhanced_markdown = enhanced_md_path.read_text(encoding="utf-8")
    assert enhanced_markdown == (
        "# Title\n\nSome text.\n\n\n---\n\n1. first\n2. second\n\n---\n\n"
    )
