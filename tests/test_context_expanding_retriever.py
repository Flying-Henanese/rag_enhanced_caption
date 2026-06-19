"""Tests for post-rerank expansion of short LlamaIndex content nodes."""

from __future__ import annotations

from llama_index.core import StorageContext
from llama_index.core.retrievers import AutoMergingRetriever, BaseRetriever
from llama_index.core.schema import (
    NodeRelationship,
    NodeWithScore,
    QueryBundle,
    RelatedNodeInfo,
    TextNode,
)
from llama_index.core.storage.docstore import SimpleDocumentStore

from rag_enhanced_caption.integrations.llama_index import (
    ShortContextExpandingRetriever,
)


class _StaticRetriever(BaseRetriever):
    """Return a fixed set of nodes without external services."""

    def __init__(self, nodes: list[NodeWithScore]) -> None:
        self.nodes = nodes
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        return list(self.nodes)


def _link(left: TextNode, right: TextNode) -> None:
    left.relationships[NodeRelationship.NEXT] = RelatedNodeInfo(node_id=right.node_id)
    right.relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(
        node_id=left.node_id
    )


def _content_node(node_id: str, text: str, section: str = "A") -> TextNode:
    return TextNode(
        id_=node_id,
        text=text,
        metadata={"type": "chunk", "section_path": [section]},
    )


def _build_linear_docstore() -> tuple[SimpleDocumentStore, list[TextNode]]:
    nodes = [
        _content_node("previous", "前文段落"),
        _content_node("anchor", "短锚点"),
        _content_node("next", "后文段落"),
    ]
    _link(nodes[0], nodes[1])
    _link(nodes[1], nodes[2])
    docstore = SimpleDocumentStore()
    docstore.add_documents(nodes)
    return docstore, nodes


def test_short_anchor_adds_previous_and_next_with_decayed_scores() -> None:
    docstore, nodes = _build_linear_docstore()
    retriever = ShortContextExpandingRetriever(
        base_retriever=_StaticRetriever([NodeWithScore(node=nodes[1], score=0.8)]),
        docstore=docstore,
        token_count_fn=len,
        short_node_token_threshold=10,
        score_decay=0.5,
    )

    results = retriever.retrieve("测试问题")

    by_id = {result.node.node_id: result for result in results}
    assert set(by_id) == {"previous", "anchor", "next"}
    assert by_id["anchor"].score == 0.8
    assert by_id["previous"].score == 0.4
    assert by_id["next"].score == 0.4


def test_long_anchor_is_not_expanded() -> None:
    docstore, nodes = _build_linear_docstore()
    nodes[1].text = "长度达到阈值"
    retriever = ShortContextExpandingRetriever(
        base_retriever=_StaticRetriever([NodeWithScore(node=nodes[1], score=0.8)]),
        docstore=docstore,
        token_count_fn=len,
        short_node_token_threshold=len(nodes[1].text),
    )

    results = retriever.retrieve("测试问题")

    assert [result.node.node_id for result in results] == ["anchor"]


def test_expansion_stops_at_section_boundary() -> None:
    _, nodes = _build_linear_docstore()
    nodes[2].metadata["section_path"] = ["B"]
    docstore = SimpleDocumentStore()
    docstore.add_documents(nodes)
    retriever = ShortContextExpandingRetriever(
        base_retriever=_StaticRetriever([NodeWithScore(node=nodes[1], score=1.0)]),
        docstore=docstore,
        token_count_fn=len,
        short_node_token_threshold=10,
    )

    results = retriever.retrieve("测试问题")

    assert {result.node.node_id for result in results} == {"previous", "anchor"}


def test_overlapping_expansions_deduplicate_nodes() -> None:
    docstore, nodes = _build_linear_docstore()
    retriever = ShortContextExpandingRetriever(
        base_retriever=_StaticRetriever(
            [
                NodeWithScore(node=nodes[0], score=0.9),
                NodeWithScore(node=nodes[1], score=0.8),
            ]
        ),
        docstore=docstore,
        token_count_fn=len,
        short_node_token_threshold=10,
    )

    results = retriever.retrieve("测试问题")

    ids = [result.node.node_id for result in results]
    assert len(ids) == len(set(ids))
    assert set(ids) == {"previous", "anchor", "next"}


def test_missing_related_node_is_ignored() -> None:
    anchor = _content_node("anchor", "短锚点")
    anchor.relationships[NodeRelationship.NEXT] = RelatedNodeInfo(node_id="missing")
    docstore = SimpleDocumentStore()
    docstore.add_documents([anchor])
    retriever = ShortContextExpandingRetriever(
        base_retriever=_StaticRetriever([NodeWithScore(node=anchor, score=1.0)]),
        docstore=docstore,
        token_count_fn=len,
        short_node_token_threshold=10,
    )

    results = retriever.retrieve("测试问题")

    assert [result.node.node_id for result in results] == ["anchor"]


def test_relationship_cycle_terminates_during_multi_hop_expansion() -> None:
    anchor = _content_node("anchor", "短锚点")
    neighbor = _content_node("neighbor", "邻居")
    _link(anchor, neighbor)
    neighbor.relationships[NodeRelationship.NEXT] = RelatedNodeInfo(
        node_id=anchor.node_id
    )
    docstore = SimpleDocumentStore()
    docstore.add_documents([anchor, neighbor])
    retriever = ShortContextExpandingRetriever(
        base_retriever=_StaticRetriever([NodeWithScore(node=anchor, score=1.0)]),
        docstore=docstore,
        token_count_fn=len,
        short_node_token_threshold=10,
        previous_nodes=0,
        next_nodes=5,
        max_added_nodes=5,
    )

    results = retriever.retrieve("测试问题")

    assert {result.node.node_id for result in results} == {"anchor", "neighbor"}


def test_added_node_and_token_budgets_cap_expansion() -> None:
    docstore, nodes = _build_linear_docstore()
    retriever = ShortContextExpandingRetriever(
        base_retriever=_StaticRetriever([NodeWithScore(node=nodes[1], score=1.0)]),
        docstore=docstore,
        token_count_fn=len,
        short_node_token_threshold=10,
        max_added_nodes=1,
        max_expansion_tokens=len(nodes[0].text),
    )

    results = retriever.retrieve("测试问题")

    assert [result.node.node_id for result in results] == ["anchor", "previous"]


def test_expanded_sibling_can_trigger_parent_automerge() -> None:
    parent = TextNode(
        id_="parent",
        text="完整父级上下文",
        relationships={
            NodeRelationship.CHILD: [
                RelatedNodeInfo(node_id="child-1"),
                RelatedNodeInfo(node_id="child-2"),
            ]
        },
    )
    child_1 = _content_node("child-1", "短锚点")
    child_2 = _content_node("child-2", "相邻子节点")
    child_1.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(
        node_id=parent.node_id
    )
    child_2.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(
        node_id=parent.node_id
    )
    _link(child_1, child_2)
    docstore = SimpleDocumentStore()
    docstore.add_documents([parent, child_1, child_2])
    expanding_retriever = ShortContextExpandingRetriever(
        base_retriever=_StaticRetriever([NodeWithScore(node=child_1, score=0.9)]),
        docstore=docstore,
        token_count_fn=len,
        short_node_token_threshold=10,
        previous_nodes=0,
        next_nodes=1,
    )
    storage_context = StorageContext.from_defaults(docstore=docstore)
    retriever = AutoMergingRetriever(
        expanding_retriever,
        storage_context,
        simple_ratio_thresh=0.5,
    )

    results = retriever.retrieve("测试问题")

    assert [result.node.node_id for result in results] == ["parent"]
