import asyncio
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from rag_enhanced_caption.integrations.llama_index.hybrid_retriever import (
    HybridRetriever,
)
from rag_enhanced_caption.lexical_search.builder import build_searchable_objects
from rag_enhanced_caption.lexical_search.repository import (
    JsonlSearchableObjectRepository,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_example(name: str, filename: str) -> ModuleType:
    path = ROOT / "examples" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class EmptyVectorRetriever(BaseRetriever):
    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        return []


def test_ingestion_example_writes_sparse_jsonl(tmp_path: Path) -> None:
    ingestion = _load_example(
        "data_ingestion_pipeline_test", "data_ingestion_pipeline.py"
    )
    sparse_path = tmp_path / "sample_sparse.jsonl"
    chunks = [
        {
            "id": "sample_chunk_1",
            "content": "| 名称 | 日期 |\n| --- | --- |\n| MiniRAG | 2026-06-19 |",
            "metadata": {"element_type": "Table"},
        }
    ]

    ingestion.write_searchable_objects(chunks, sparse_path)

    objects = JsonlSearchableObjectRepository(sparse_path).load()
    assert objects
    assert {obj.owner_node_id for obj in objects} == {"sample_chunk_1"}


def test_ingestion_example_writes_non_empty_index_and_complete_docstore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ingestion = _load_example(
        "data_ingestion_pipeline_contract_test", "data_ingestion_pipeline.py"
    )
    input_path = tmp_path / "sample.md"
    input_path.write_text("- first\n- second\n", encoding="utf-8")
    chunks = [
        {
            "id": "sample_chunk_0",
            "content": "- first\n- second",
            "full_content": "- first\n- second",
            "text_for_embedding": "",
            "metadata": {
                "element_type": "bullet_list_open",
                "parent_id": None,
                "header_path": ["Sample"],
            },
        }
    ]

    class FakeProcessor:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        async def enrich_chunks(
            self, chunks: list[dict[str, Any]], base_dir: str
        ) -> list[dict[str, Any]]:
            return chunks

    monkeypatch.setattr(ingestion, "root_dir", tmp_path)
    monkeypatch.setattr(ingestion, "chunk_markdown", lambda **kwargs: chunks)
    monkeypatch.setattr(ingestion, "compress_and_relink_chunks", lambda value: value)
    monkeypatch.setattr(ingestion, "create_default_vlm_client", lambda: object())
    monkeypatch.setattr(ingestion, "MarkdownMultimodalProcessor", FakeProcessor)

    asyncio.run(ingestion.process_document(input_path))

    index_path = tmp_path / "output" / "sample_index_new.jsonl"
    docstore_path = tmp_path / "output" / "sample_docstore_new.jsonl"
    index_record = json.loads(index_path.read_text(encoding="utf-8"))
    docstore_record = json.loads(docstore_path.read_text(encoding="utf-8"))
    assert index_record["text_for_embedding"] == "- first\n- second"
    assert index_record["id"] == docstore_record["id"]
    assert {
        "id",
        "full_content",
        "parent_id",
        "header_path",
        "element_type",
        "entities",
    } == docstore_record.keys()


def test_advanced_example_builds_hybrid_candidate_retriever(tmp_path: Path) -> None:
    advanced = _load_example("advanced_rag_hybrid_test", "llama_index_advanced_rag.py")
    sparse_path = tmp_path / "sample_sparse.jsonl"
    JsonlSearchableObjectRepository(sparse_path).replace(
        build_searchable_objects(
            {
                "id": "sample_chunk_1",
                "content": "owner@example.com",
                "metadata": {"element_type": "text"},
            }
        )
    )
    leaf = TextNode(id_="sample_chunk_1", text="semantic summary")

    retriever = advanced.build_candidate_retriever(
        EmptyVectorRetriever(), [leaf], sparse_path
    )

    assert isinstance(retriever, HybridRetriever)


def test_advanced_example_falls_back_when_sparse_file_is_missing(
    tmp_path: Path,
) -> None:
    advanced = _load_example(
        "advanced_rag_fallback_test", "llama_index_advanced_rag.py"
    )
    vector_retriever = EmptyVectorRetriever()

    retriever = advanced.build_candidate_retriever(
        vector_retriever, [], tmp_path / "missing_sparse.jsonl"
    )

    assert retriever is vector_retriever
