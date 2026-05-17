import asyncio
import json

from rag_enhanced_caption import cli


def test_process_document_uses_enrich_chunks_and_completes(tmp_path, monkeypatch):
    input_md = tmp_path / "sample.md"
    output_dir = tmp_path / "out"
    input_md.write_text("# Title\n\nSome text.\n", encoding="utf-8")

    def fake_chunk_markdown(**kwargs):
        return [
            {
                "id": "chunk_1",
                "content": "# Title\n\nSome text.\n",
                "file_id": "sample",
                "filename": "sample.md",
                "metadata": {"element_type": "text"},
            }
        ]

    async def fake_vlm(*args, **kwargs):
        return '{"summary":"ok","entities":[]}'

    class FakeProcessor:
        def __init__(self, vlm_func, max_concurrency=5):
            self.vlm_func = vlm_func
            self.max_concurrency = max_concurrency

        async def enrich_chunks(self, chunks, image_resolver=None, base_dir="."):
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
    assert len(index_lines) >= 1
    first = json.loads(index_lines[0])
    assert first["id"].startswith("p_")
    assert first["metadata"]["type"] == "parent"
