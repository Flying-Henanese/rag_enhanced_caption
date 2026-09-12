import asyncio
import importlib.util
import json
import os
from pathlib import Path

import dotenv
import pytest
from llama_index.core import StorageContext
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import (
    BaseNode,
    IndexNode,
    NodeWithScore,
    QueryBundle,
    TextNode,
)
from llama_index.core.storage.docstore import SimpleDocumentStore

from rag_enhanced_caption.integrations.llama_index import (
    ShortContextExpandingRetriever,
)


dotenv.load_dotenv()

MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "examples" / "llama_index_advanced_rag.py"
)
spec = importlib.util.spec_from_file_location("advanced_rag_example", MODULE_PATH)
advanced_rag = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(advanced_rag)


def test_advanced_example_targets_rag_anything_artifacts() -> None:
    assert advanced_rag.INDEX_PATH.name == "rag-anything_index_new.jsonl"
    assert advanced_rag.DOCSTORE_PATH.name == "rag-anything_docstore_new.jsonl"


def test_link_content_nodes_in_order_stays_within_section() -> None:
    paragraph = TextNode(
        id_="paragraph",
        text="正文",
        metadata={"type": "chunk", "section_path": ["章节 A"]},
    )
    table = TextNode(
        id_="table",
        text="表格",
        metadata={"type": "element", "section_path": ["章节 A"]},
    )
    next_section = TextNode(
        id_="next-section",
        text="下一章节",
        metadata={"type": "chunk", "section_path": ["章节 B"]},
    )

    advanced_rag._link_content_nodes_in_order([paragraph, table, next_section])

    assert paragraph.next_node is not None
    assert paragraph.next_node.node_id == table.node_id
    assert table.prev_node is not None
    assert table.prev_node.node_id == paragraph.node_id
    assert table.next_node is None
    assert next_section.prev_node is None


def test_short_parent_id_maps_to_complete_content_node_id() -> None:
    node_id_map = {
        "rag-anything_chunk_5": "rag-anything_chunk_5",
        "rag-anything_chunk_6": "rag-anything_chunk_6_full",
    }

    resolved = advanced_rag._resolve_parent_node_id(
        "5", "rag-anything_chunk_6", node_id_map
    )

    assert resolved == "rag-anything_chunk_5"


def test_context_expansion_pipeline_wraps_reranked_retriever() -> None:
    docstore = SimpleDocumentStore()
    storage_context = StorageContext.from_defaults(docstore=docstore)
    reranked = advanced_rag.RerankedRetriever(
        base_retriever=object(),
        reranker=None,
    )

    auto_merging = advanced_rag._build_auto_merging_retriever(
        reranked,
        storage_context,
    )

    expanding = auto_merging._vector_retriever
    assert isinstance(expanding, ShortContextExpandingRetriever)
    assert expanding.base_retriever is reranked


def test_advanced_components_map_list_records_to_text_nodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index_path = tmp_path / "sample_index_new.jsonl"
    docstore_path = tmp_path / "sample_docstore_new.jsonl"
    sparse_path = tmp_path / "sample_sparse.jsonl"
    index_path.write_text(
        json.dumps(
            {
                "id": "sample_chunk_0",
                "text_for_embedding": "",
                "metadata": {"element_type": "ordered_list_open"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    docstore_path.write_text(
        json.dumps(
            {
                "id": "sample_chunk_0",
                "full_content": "1. first\n2. second",
                "parent_id": None,
                "header_path": [],
                "element_type": "ordered_list_open",
                "entities": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    captured_nodes: list[BaseNode] = []

    class EmptyRetriever(BaseRetriever):
        def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
            return []

    class FakeVectorStoreIndex:
        def __init__(
            self, nodes: list[BaseNode], storage_context: StorageContext
        ) -> None:
            captured_nodes.extend(nodes)

        def as_retriever(self, similarity_top_k: int) -> BaseRetriever:
            return EmptyRetriever()

    monkeypatch.setattr(advanced_rag, "INDEX_PATH", index_path)
    monkeypatch.setattr(advanced_rag, "DOCSTORE_PATH", docstore_path)
    monkeypatch.setattr(advanced_rag, "SPARSE_PATH", sparse_path)
    monkeypatch.setattr(advanced_rag, "VectorStoreIndex", FakeVectorStoreIndex)
    monkeypatch.delenv("RERANK_API_KEY", raising=False)

    advanced_rag.get_advanced_components()

    assert len(captured_nodes) == 1
    assert isinstance(captured_nodes[0], TextNode)
    assert captured_nodes[0].text == "1. first\n2. second"


def test_async_recursive_retriever_resolves_index_node() -> None:
    full_node = TextNode(id_="image_full", text="full image description")
    index_node = IndexNode(
        id_="image", text="image summary", index_id=full_node.node_id
    )

    class StaticRetriever(BaseRetriever):
        def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
            return [NodeWithScore(node=index_node, score=0.8)]

    retriever = advanced_rag.AsyncRecursiveRetriever(
        "candidate",
        retriever_dict={"candidate": StaticRetriever()},
        node_dict={full_node.node_id: full_node},
    )

    results = asyncio.run(retriever.aretrieve("describe the image"))

    assert len(results) == 1
    assert results[0].node.node_id == full_node.node_id
    assert results[0].score == 0.8


def test_advanced_retrieval_pipeline_real_components() -> None:
    if not os.getenv("EMBEDDING_API_KEY"):
        pytest.skip("EMBEDDING_API_KEY is not configured.")

    if not advanced_rag.DOCSTORE_PATH.exists() or not advanced_rag.INDEX_PATH.exists():
        pytest.skip("Required JSONL resources for retrieval are missing.")

    _, auto_merging_retriever, _ = advanced_rag.get_advanced_components()
    query = "RAG-Anything 的架构图展示了哪些主要处理阶段？"

    final_results = asyncio.run(auto_merging_retriever.aretrieve(query))

    assert len(final_results) > 0
    assert final_results[0].node is not None
    assert final_results[0].node.text
