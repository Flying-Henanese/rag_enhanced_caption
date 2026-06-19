from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from rag_enhanced_caption.integrations.llama_index.hybrid_retriever import (
    HybridRetriever,
)
from rag_enhanced_caption.lexical_search.bm25 import SparseSearchResult


class StaticVectorRetriever(BaseRetriever):
    def __init__(self, nodes: list[NodeWithScore]) -> None:
        self.nodes = nodes
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        return self.nodes


class StaticLexicalBackend:
    def search(self, query: str, top_k: int = 10) -> list[SparseSearchResult]:
        return [
            SparseSearchResult(
                owner_node_id="lexical-only", object_id="obj-1", score=2.0
            ),
            SparseSearchResult(owner_node_id="shared", object_id="obj-2", score=1.0),
        ][:top_k]


def test_hybrid_retriever_fuses_vector_and_lexical_nodes() -> None:
    nodes = {
        node_id: TextNode(id_=node_id, text=node_id)
        for node_id in ("vector-only", "shared", "lexical-only")
    }
    vector_retriever = StaticVectorRetriever(
        [
            NodeWithScore(node=nodes["vector-only"], score=0.9),
            NodeWithScore(node=nodes["shared"], score=0.8),
        ]
    )
    retriever = HybridRetriever(
        vector_retriever=vector_retriever,
        lexical_backend=StaticLexicalBackend(),
        node_by_id=nodes,
        lexical_top_k=2,
    )

    results = retriever.retrieve("MiniRAG")

    assert [result.node.node_id for result in results] == [
        "shared",
        "vector-only",
        "lexical-only",
    ]
    assert all(result.score is not None for result in results)
