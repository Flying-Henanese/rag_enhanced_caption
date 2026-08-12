"""Custom retrievers for composing contextual expansion with LlamaIndex."""

from __future__ import annotations

from collections.abc import Callable

from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import (
    BaseNode,
    MetadataMode,
    NodeRelationship,
    NodeWithScore,
    QueryBundle,
)
from llama_index.core.storage.docstore.types import BaseDocumentStore
from llama_index.core.utils import get_tokenizer


class ShortContextExpandingRetriever(BaseRetriever):
    """Expand short reranked paragraph hits with document-order neighbors.

    Args:
        base_retriever: Retriever that returns reranked anchor nodes.
        docstore: Store containing nodes referenced by document-order links.
        token_count_fn: Function returning the token count for node text.
        short_node_token_threshold: Expand anchors shorter than this count.
        previous_nodes: Maximum previous-node hops per anchor.
        next_nodes: Maximum next-node hops per anchor.
        max_added_nodes: Maximum neighbors added across one retrieval.
        max_expansion_tokens: Maximum total tokens in added neighbors.
        score_decay: Multiplicative score decay applied per relationship hop.
        same_section: Whether expansion must remain in the anchor section.
        section_metadata_key: Metadata key containing section identity.
        eligible_node_types: Metadata ``type`` values eligible as anchors.
    """

    def __init__(
        self,
        base_retriever: BaseRetriever,
        docstore: BaseDocumentStore,
        token_count_fn: Callable[[str], int] | None = None,
        short_node_token_threshold: int = 100,
        previous_nodes: int = 1,
        next_nodes: int = 1,
        max_added_nodes: int = 2,
        max_expansion_tokens: int = 512,
        score_decay: float = 0.85,
        same_section: bool = True,
        section_metadata_key: str = "section_path",
        eligible_node_types: set[str] | None = None,
    ) -> None:
        """Initialize the context-expanding retriever.

        Args:
            base_retriever: Retriever that returns reranked anchor nodes.
            docstore: Store containing nodes referenced by document-order links.
            token_count_fn: Function returning the token count for node text.
            short_node_token_threshold: Expand anchors shorter than this count.
            previous_nodes: Maximum previous-node hops per anchor.
            next_nodes: Maximum next-node hops per anchor.
            max_added_nodes: Maximum neighbors added across one retrieval.
            max_expansion_tokens: Maximum total tokens in added neighbors.
            score_decay: Multiplicative score decay applied per relationship hop.
            same_section: Whether expansion must remain in the anchor section.
            section_metadata_key: Metadata key containing section identity.
            eligible_node_types: Metadata ``type`` values eligible as anchors.
        """
        self.base_retriever = base_retriever
        self.docstore = docstore
        if token_count_fn is None:
            tokenizer = get_tokenizer()

            def token_count_fn(text: str) -> int:
                return len(tokenizer(text))

        self.token_count_fn = token_count_fn
        self.short_node_token_threshold = short_node_token_threshold
        self.previous_nodes = previous_nodes
        self.next_nodes = next_nodes
        self.max_added_nodes = max_added_nodes
        self.max_expansion_tokens = max_expansion_tokens
        self.score_decay = score_decay
        self.same_section = same_section
        self.section_metadata_key = section_metadata_key
        self.eligible_node_types = eligible_node_types or {"chunk"}
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        anchors = self.base_retriever.retrieve(query_bundle)
        results = list(anchors)
        known_node_ids = {result.node.node_id for result in results}
        added_nodes = 0
        expansion_tokens = 0

        for anchor in anchors:
            if not self._is_short_eligible_anchor(anchor.node):
                continue
            anchor_section = anchor.node.metadata.get(self.section_metadata_key)
            directions = (
                (NodeRelationship.PREVIOUS, self.previous_nodes),
                (NodeRelationship.NEXT, self.next_nodes),
            )
            for relationship, max_hops in directions:
                current_node = anchor.node
                visited_node_ids = {anchor.node.node_id}
                for hop in range(1, max_hops + 1):
                    related = current_node.relationships.get(relationship)
                    if related is None or related.node_id in visited_node_ids:
                        break
                    visited_node_ids.add(related.node_id)

                    neighbor = self.docstore.get_document(
                        related.node_id, raise_error=False
                    )
                    if neighbor is None:
                        break
                    if (
                        self.same_section
                        and neighbor.metadata.get(self.section_metadata_key)
                        != anchor_section
                    ):
                        break

                    current_node = neighbor
                    if neighbor.node_id in known_node_ids:
                        continue
                    if added_nodes >= self.max_added_nodes:
                        break

                    neighbor_text = neighbor.get_content(
                        metadata_mode=MetadataMode.NONE
                    )
                    neighbor_tokens = self.token_count_fn(neighbor_text)
                    if expansion_tokens + neighbor_tokens > self.max_expansion_tokens:
                        break

                    results.append(
                        NodeWithScore(
                            node=neighbor,
                            score=(anchor.score or 0.0) * self.score_decay**hop,
                        )
                    )
                    known_node_ids.add(neighbor.node_id)
                    added_nodes += 1
                    expansion_tokens += neighbor_tokens

        return results

    def _is_short_eligible_anchor(self, node: BaseNode) -> bool:
        node_type = node.metadata.get("type")
        if node_type not in self.eligible_node_types:
            return False
        text = node.get_content(metadata_mode=MetadataMode.NONE)
        return self.token_count_fn(text) < self.short_node_token_threshold
