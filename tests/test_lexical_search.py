from pathlib import Path

from rag_enhanced_caption.lexical_search.bm25 import InMemoryBM25Backend
from rag_enhanced_caption.lexical_search.fusion import reciprocal_rank_fusion
from rag_enhanced_caption.lexical_search.repository import (
    JsonlSearchableObjectRepository,
)
from rag_enhanced_caption.lexical_search.schema import SearchableObject


def _object(object_id: str, owner_id: str, text: str) -> SearchableObject:
    return SearchableObject(
        id=object_id,
        owner_node_id=owner_id,
        searchable_text=text,
        field_type="table_row",
    )


def test_jsonl_repository_round_trips_searchable_objects(tmp_path: Path) -> None:
    path = tmp_path / "document_sparse.jsonl"
    expected = [
        _object("row-1", "chunk-1", "模型：MiniRAG"),
        _object("row-2", "chunk-2", "模型：LightRAG"),
    ]
    repository = JsonlSearchableObjectRepository(path)

    repository.replace(expected)

    assert repository.load() == expected


def test_bm25_ranks_exact_table_value_first() -> None:
    backend = InMemoryBM25Backend(
        [
            _object("row-1", "chunk-1", "模型 MiniRAG 极简 RAG 系统"),
            _object("row-2", "chunk-2", "模型 LightRAG 简单快速 RAG 系统"),
            _object("row-3", "chunk-3", "通用检索增强生成系统"),
        ]
    )

    results = backend.search("MiniRAG", top_k=2)

    assert results[0].owner_node_id == "chunk-1"
    assert results[0].score > 0


def test_bm25_deduplicates_results_by_owner_node() -> None:
    backend = InMemoryBM25Backend(
        [
            _object("row-1", "chunk-1", "发布时间 2026-06-19"),
            _object("row-2", "chunk-1", "发布日期 2026-06-19"),
            _object("row-3", "chunk-2", "发布日期 2025-01-01"),
        ]
    )

    results = backend.search("2026-06-19", top_k=5)

    assert [result.owner_node_id for result in results] == ["chunk-1"]


def test_reciprocal_rank_fusion_combines_independent_rankings() -> None:
    fused = reciprocal_rank_fusion(
        [["vector-only", "shared"], ["lexical-only", "shared"]], constant=60
    )

    assert list(fused) == ["shared", "vector-only", "lexical-only"]
