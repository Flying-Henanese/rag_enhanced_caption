import json
import os
import sys
import requests
from pydantic import Field
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.schema import (
    BaseNode,
    TextNode,
    IndexNode,
    NodeWithScore,
    QueryBundle,
    NodeRelationship,
    RelatedNodeInfo,
)
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core import StorageContext
from llama_index.core.retrievers import AutoMergingRetriever, BaseRetriever
from rag_enhanced_caption.integrations.llama_index.hybrid_retriever import (
    HybridRetriever,
)
from rag_enhanced_caption.lexical_search.bm25 import InMemoryBM25Backend
from rag_enhanced_caption.lexical_search.repository import (
    JsonlSearchableObjectRepository,
)

from rag_enhanced_caption.integrations.llama_index import (
    ShortContextExpandingRetriever,
)
from rag_enhanced_caption.chunker.utils.id_utils import resolve_parent_chunk_id

load_dotenv()
logger.remove()
logger.add(sys.stderr, level="WARNING")


class SiliconFlowRerank(BaseNodePostprocessor):
    """Rerank retrieved nodes through the SiliconFlow rerank API.

    Args:
        api_key: SiliconFlow API key. Empty keys disable remote reranking.
        endpoint: Rerank API endpoint.
        model: Rerank model identifier.
        top_n: Maximum number of reranked nodes to return.
        timeout: Request timeout in seconds.
    """

    api_key: str = Field(default="")
    endpoint: str = Field(default="https://api.siliconflow.cn/v1/rerank")
    model: str = Field(default="Pro/BAAI/bge-reranker-v2-m3")
    top_n: int = Field(default=2)
    timeout: float = Field(default=30.0)

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> list[NodeWithScore]:
        if query_bundle is None or not nodes or not self.api_key:
            return nodes

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        texts = [node.node.get_content() for node in nodes]
        payload = {
            "model": self.model,
            "query": query_bundle.query_str,
            "documents": texts,
            "return_documents": False,
        }

        try:
            response = requests.post(
                self.endpoint,
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            results = data.get("results", [])
            new_nodes = []
            for result in results:
                idx = result["index"]
                score = result["relevance_score"]
                nodes[idx].score = score
                new_nodes.append(nodes[idx])
            new_nodes.sort(key=lambda x: x.score or 0.0, reverse=True)
            return new_nodes[: self.top_n]
        except requests.Timeout:
            logger.warning(f"Rerank request timed out after {self.timeout} seconds")
            return nodes[: self.top_n]
        except Exception as e:
            logger.error(f"Rerank failed: {e}")
            return nodes[: self.top_n]


embed_model_name = os.getenv("EMBEDDING_MODEL_NAME")
if embed_model_name:
    try:
        from llama_index.embeddings.openai import OpenAIEmbedding

        api_key = os.getenv("EMBEDDING_API_KEY", "")
        endpoint = os.getenv(
            "EMBEDDING_ENDPOINT", "https://api.siliconflow.cn/v1/embeddings"
        )
        if endpoint.endswith("/embeddings"):
            endpoint = endpoint[:-11]

        Settings.embed_model = OpenAIEmbedding(
            model_name=embed_model_name,
            api_key=api_key,
            api_base=endpoint,
            embed_batch_size=64,
        )
    except ImportError:
        pass

RESOURCE_DIR = Path("test_resource")
OUTPUT_DIR = Path("output")
DOCSTORE_PATH = OUTPUT_DIR / "rag-anything_docstore_new.jsonl"
INDEX_PATH = OUTPUT_DIR / "rag-anything_index_new.jsonl"
SPARSE_PATH = OUTPUT_DIR / "rag-anything_sparse.jsonl"


def build_candidate_retriever(
    vector_retriever: BaseRetriever,
    leaf_nodes: list[BaseNode],
    sparse_path: Path,
) -> BaseRetriever:
    """Build vector-only or hybrid candidates depending on sparse data.

    Args:
        vector_retriever: Existing embedding-based leaf retriever.
        leaf_nodes: Nodes addressable by IDs stored in searchable objects.
        sparse_path: Persisted searchable object JSONL path.

    Returns:
        A hybrid retriever when searchable objects exist, otherwise the original
        vector retriever.
    """
    searchable_objects = JsonlSearchableObjectRepository(sparse_path).load()
    if not searchable_objects:
        return vector_retriever
    lexical_backend = InMemoryBM25Backend(searchable_objects)
    return HybridRetriever(
        vector_retriever=vector_retriever,
        lexical_backend=lexical_backend,
        node_by_id={node.node_id: node for node in leaf_nodes},
        lexical_top_k=15,
    )


def _link_content_nodes_in_order(nodes: list[TextNode]) -> None:
    """Link consecutive content nodes when they share a section path."""
    for previous, current in zip(nodes, nodes[1:]):
        if previous.metadata.get("section_path") != current.metadata.get(
            "section_path"
        ):
            continue
        previous.relationships[NodeRelationship.NEXT] = RelatedNodeInfo(
            node_id=current.node_id
        )
        current.relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(
            node_id=previous.node_id
        )


def _resolve_parent_node_id(
    parent_id: str | None,
    child_chunk_id: str,
    node_id_map: dict[str, str],
) -> str | None:
    """Map a persisted parent reference to its LlamaIndex content node ID.

    Args:
        parent_id: Short or complete parent chunk ID from the docstore record.
        child_chunk_id: Complete ID of the child record.
        node_id_map: Mapping from chunk IDs to their content-node IDs.

    Returns:
        The mapped LlamaIndex node ID, or ``None`` when the parent is absent or
        has not been constructed.
    """
    resolved_parent_id = resolve_parent_chunk_id(parent_id, child_chunk_id)
    if resolved_parent_id is None:
        return None
    return node_id_map.get(resolved_parent_id)


class RerankedRetriever(BaseRetriever):
    """在上下文合并前执行可选精排。

    避免超大章节被精排模型截断。

    Args:
        base_retriever: 生成候选节点的基础检索器。
        reranker: 可选的节点精排器。
    """

    def __init__(
        self,
        base_retriever: BaseRetriever,
        reranker: BaseNodePostprocessor | None,
    ) -> None:
        """初始化精排检索器。

        Args:
            base_retriever: 生成候选节点的基础检索器。
            reranker: 可选的节点精排器。
        """
        self.base_retriever = base_retriever
        self.reranker = reranker
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        nodes = self.base_retriever.retrieve(query_bundle)
        if self.reranker and nodes:
            nodes = self.reranker.postprocess_nodes(nodes, query_bundle)
        return nodes


def _build_auto_merging_retriever(
    reranked_retriever: BaseRetriever,
    storage_context: StorageContext,
) -> AutoMergingRetriever:
    """Compose post-rerank short-context expansion with AutoMerge."""
    context_expanding_retriever = ShortContextExpandingRetriever(
        base_retriever=reranked_retriever,
        docstore=storage_context.docstore,
        short_node_token_threshold=100,
        previous_nodes=1,
        next_nodes=1,
        max_added_nodes=2,
        max_expansion_tokens=512,
        score_decay=0.85,
        same_section=True,
    )
    return AutoMergingRetriever(
        context_expanding_retriever,
        storage_context,
        verbose=True,
        simple_ratio_thresh=0.5,
    )


def get_advanced_components() -> tuple[
    BaseRetriever,
    AutoMergingRetriever,
    BaseNodePostprocessor | None,
]:
    """基于生成好的 JSONL 构建 LlamaIndex 检索树。

    Returns:
        向量检索器、上下文自动合并检索器和可选的精排器。
    """
    docstore_nodes = {}
    path_nodes = {}
    leaf_nodes = []
    content_nodes_in_order = []

    # 1. 加载数据
    doc_records = {}
    with open(DOCSTORE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            doc_records[data["id"]] = data

    index_records = {}
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            index_records[data["id"]] = data

    node_id_map = {}

    # 2. 构建树状结构
    for chunk_id, doc_data in doc_records.items():
        header_path = doc_data.get("header_path", [])
        element_type = doc_data.get("element_type", "text")
        parent_id = doc_data.get("parent_id")

        # a. 构建章节节点 (Path Nodes)
        current_parent_node_id = None
        current_path_str = ""
        for level in header_path:
            current_path_str = (
                f"{current_path_str}|{level}" if current_path_str else level
            )
            if current_path_str not in path_nodes:
                safe_path = current_path_str.replace("|", "_").replace(" ", "")
                sect_id = f"path_{safe_path}"
                sect_node = TextNode(id_=sect_id, text=f"【章节聚合：{level}】")
                docstore_nodes[sect_id] = sect_node
                path_nodes[current_path_str] = sect_id

                if current_parent_node_id:
                    sect_node.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(
                        node_id=current_parent_node_id
                    )
                    docstore_nodes[current_parent_node_id].relationships.setdefault(
                        NodeRelationship.CHILD, []
                    ).append(RelatedNodeInfo(node_id=sect_id))

            current_parent_node_id = path_nodes[current_path_str]

        # b. 构建主节点
        if element_type == "text":
            text_for_embed = index_records.get(chunk_id, {}).get(
                "text_for_embedding", ""
            )
            node = TextNode(
                id_=chunk_id,
                text=text_for_embed,
                metadata={
                    "type": "chunk",
                    "full_content": doc_data["full_content"],
                    "section_path": header_path,
                },
            )
            if current_parent_node_id:
                node.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(
                    node_id=current_parent_node_id
                )
                docstore_nodes[current_parent_node_id].relationships.setdefault(
                    NodeRelationship.CHILD, []
                ).append(RelatedNodeInfo(node_id=chunk_id))

            docstore_nodes[chunk_id] = node
            leaf_nodes.append(node)
            content_nodes_in_order.append(node)
            node_id_map[chunk_id] = chunk_id

        else:
            # 独立多模态元素
            element_node = TextNode(
                id_=f"{chunk_id}_full",
                text=doc_data["full_content"],
                metadata={"type": "element", "section_path": header_path},
            )
            docstore_nodes[element_node.id_] = element_node

            actual_parent_id = _resolve_parent_node_id(parent_id, chunk_id, node_id_map)
            if actual_parent_id:
                element_node.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(
                    node_id=actual_parent_id
                )
                docstore_nodes[actual_parent_id].relationships.setdefault(
                    NodeRelationship.CHILD, []
                ).append(RelatedNodeInfo(node_id=element_node.id_))
            elif current_parent_node_id:
                element_node.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(
                    node_id=current_parent_node_id
                )
                docstore_nodes[current_parent_node_id].relationships.setdefault(
                    NodeRelationship.CHILD, []
                ).append(RelatedNodeInfo(node_id=element_node.id_))

            # 指针节点
            text_for_embed = index_records.get(chunk_id, {}).get(
                "text_for_embedding", ""
            )
            index_node = IndexNode(
                id_=chunk_id, text=text_for_embed, index_id=element_node.id_
            )
            docstore_nodes[index_node.id_] = index_node
            leaf_nodes.append(index_node)
            content_nodes_in_order.append(element_node)
            node_id_map[chunk_id] = element_node.id_

    _link_content_nodes_in_order(content_nodes_in_order)

    # c. 聚合章节文本
    sorted_paths = sorted(
        list(path_nodes.keys()), key=lambda x: len(x.split("|")), reverse=True
    )
    for path in sorted_paths:
        node_id = path_nodes[path]
        node = docstore_nodes[node_id]
        child_texts = []
        for child_info in node.relationships.get(NodeRelationship.CHILD, []):
            child_node = docstore_nodes[child_info.node_id]
            child_texts.append(child_node.metadata.get("full_content", child_node.text))
        node.text = f"【章节聚合：{path.split('|')[-1]}】\n\n" + "\n\n".join(
            child_texts
        )
        node.metadata["full_content"] = node.text

    docstore = SimpleDocumentStore()
    docstore.add_documents(list(docstore_nodes.values()))
    storage_context = StorageContext.from_defaults(docstore=docstore)

    vector_index = VectorStoreIndex(leaf_nodes, storage_context=storage_context)
    vector_retriever = vector_index.as_retriever(similarity_top_k=15)
    candidate_retriever = build_candidate_retriever(
        vector_retriever, leaf_nodes, SPARSE_PATH
    )

    # 递归回表检索器
    recursive_retriever = RecursiveRetriever(
        "candidate",
        retriever_dict={"candidate": candidate_retriever},
        node_dict=storage_context.docstore.docs,
    )

    reranker = None
    rerank_api_key = os.getenv("RERANK_API_KEY")
    if rerank_api_key:
        reranker = SiliconFlowRerank(
            api_key=rerank_api_key,
            endpoint=os.getenv(
                "RERANK_ENDPOINT", "https://api.siliconflow.cn/v1/rerank"
            ),
            model=os.getenv("RERANK_MODEL_NAME", "Qwen/Qwen3-Reranker-0.6B"),
            top_n=3,
        )

    reranked_retriever = RerankedRetriever(
        base_retriever=recursive_retriever, reranker=reranker
    )

    # 上下文合并检索器
    auto_merging_retriever = _build_auto_merging_retriever(
        reranked_retriever,
        storage_context,
    )

    return vector_retriever, auto_merging_retriever, reranker


def print_box(title: str, content: object) -> None:
    """在终端中打印带标题的文本框。

    Args:
        title: 文本框标题。
        content: 要显示的内容。
    """
    print(f"\n┌── {title} {'─' * (60 - len(title))}")
    for line in str(content).split("\n"):
        print(f"│ {line[:90] + '...' if len(line) > 90 else line}")
    print(f"└{'─' * 64}")


def run_narrative_evaluation(query: str) -> None:
    """运行一个查询并打印高级检索结果。

    Args:
        query: 待评测的自然语言查询。
    """
    print(
        "\n========================================================================================="
    )
    print(f" 🎯 评测问题: {query}")
    print(
        "========================================================================================="
    )

    # ---------------------------------------------------------
    # 方案 B: 我们的方案 (AST 全局多叉树 + 先精排 + 后自动合并)
    # ---------------------------------------------------------
    print("\n\n" + "=" * 80)
    print(
        " ✅ 方案 B: 我们的方案 (AST 全局多叉树 + 先精排过滤 + AutoMerging 上下文合并)"
    )
    print("=" * 80)

    adv_vector_retriever, adv_auto_merging_retriever, reranker = (
        get_advanced_components()
    )

    print("\n【精排过滤后触发上下文合并 (Rerank -> Auto-Merging)】")
    final_results = adv_auto_merging_retriever.retrieve(query)

    for i, final_result in enumerate(final_results):
        score_info = (
            f", Rerank Score: {final_result.score:.4f}"
            if reranker and final_result.score is not None
            else ""
        )
        print_box(
            f"最终召回豪华上下文 (Merged Node {i + 1}{score_info})",
            final_result.node.text[:500] + "\n...",
        )


if __name__ == "__main__":
    queries = [
        "RAG-Anything 的架构图（Framework）展示了哪些主要的处理阶段？",
        "针对数学表达式（Mathematical Expression），系统是如何提供原生支持的？",
        "LiteWrite 是什么？它和项目有什么关系？",
    ]
    for q in queries:
        run_narrative_evaluation(q)
