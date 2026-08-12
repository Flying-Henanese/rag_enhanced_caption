import importlib.util
from pathlib import Path
from types import ModuleType

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
