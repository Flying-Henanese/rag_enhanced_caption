import importlib.util
import os
from pathlib import Path

import dotenv
import pytest
from llama_index.core import StorageContext
from llama_index.core.schema import TextNode
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


def test_advanced_retrieval_pipeline_real_components():
    if not os.getenv("EMBEDDING_API_KEY"):
        pytest.skip("EMBEDDING_API_KEY is not configured.")

    if not advanced_rag.DOCSTORE_PATH.exists() or not advanced_rag.INDEX_PATH.exists():
        pytest.skip("Required JSONL resources for retrieval are missing.")

    _, auto_merging_retriever, _ = advanced_rag.get_advanced_components()
    query = "RAG-Anything 的架构图展示了哪些主要处理阶段？"

    final_results = auto_merging_retriever.retrieve(query)

    assert len(final_results) > 0
    assert final_results[0].node is not None
    assert final_results[0].node.text
