"""LlamaIndex adapter for vector and lexical rank fusion."""

from __future__ import annotations

from collections.abc import Mapping

from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import BaseNode, NodeWithScore, QueryBundle

from rag_enhanced_caption.lexical_search.bm25 import SparseSearchBackend
from rag_enhanced_caption.lexical_search.fusion import reciprocal_rank_fusion


class HybridRetriever(BaseRetriever):
    """Fuse vector candidates with lexical owner-node matches using RRF.

    Args:
        vector_retriever: Retriever that supplies vector-search candidates.
        lexical_backend: Sparse backend that supplies lexical owner-node matches.
        node_by_id: Mapping used to resolve lexical owner IDs to nodes.
        lexical_top_k: Maximum lexical matches included in rank fusion.
        fusion_constant: Reciprocal-rank-fusion smoothing constant.
    """

    def __init__(
        self,
        vector_retriever: BaseRetriever,
        lexical_backend: SparseSearchBackend,
        node_by_id: Mapping[str, BaseNode],
        *,
        lexical_top_k: int = 15,
        fusion_constant: int = 60,
    ) -> None:
        """Initialize the hybrid retriever.

        Args:
            vector_retriever: Retriever that supplies vector-search candidates.
            lexical_backend: Sparse backend that supplies lexical owner-node matches.
            node_by_id: Mapping used to resolve lexical owner IDs to nodes.
            lexical_top_k: Maximum lexical matches included in rank fusion.
            fusion_constant: Reciprocal-rank-fusion smoothing constant.
        """
        self._vector_retriever = vector_retriever
        self._lexical_backend = lexical_backend
        self._node_by_id = node_by_id
        self._lexical_top_k = lexical_top_k
        self._fusion_constant = fusion_constant
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        lexical_results = self._lexical_backend.search(
            query_bundle.query_str, top_k=self._lexical_top_k
        )
        vector_ids = [node.node.node_id for node in vector_nodes]
        lexical_ids = [result.owner_node_id for result in lexical_results]
        fused = reciprocal_rank_fusion(
            [vector_ids, lexical_ids], constant=self._fusion_constant
        )

        candidates = {node.node.node_id: node.node for node in vector_nodes}
        candidates.update(
            {
                node_id: self._node_by_id[node_id]
                for node_id in lexical_ids
                if node_id in self._node_by_id
            }
        )
        return [
            NodeWithScore(node=candidates[node_id], score=score)
            for node_id, score in fused.items()
            if node_id in candidates
        ]
