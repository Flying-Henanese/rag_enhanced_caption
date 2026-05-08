import os
import requests
from typing import List, Dict, Any, Tuple, Optional
from pydantic import Field, PrivateAttr
from loguru import logger

from llama_index.core import StorageContext, VectorStoreIndex, Settings, QueryBundle
from llama_index.core.schema import (
    TextNode,
    IndexNode,
    NodeRelationship,
    RelatedNodeInfo,
    NodeWithScore,
)
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.retrievers import (
    AutoMergingRetriever,
    RecursiveRetriever,
    BaseRetriever,
)
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.embeddings import BaseEmbedding

from rag_enhanced_caption.chunker.dispatcher import chunk_markdown
from rag_enhanced_caption.enhancer.processor import MarkdownMultimodalProcessor
from rag_enhanced_caption.enhancer.vlm_client import create_default_vlm_client
from rag_enhanced_caption.chunker.embed_client import get_default_embedding_client

# --- 1. LlamaIndex 扩展组件配置 ---


class ProjectEmbedding(BaseEmbedding):
    """包装自定义向量模型供 LlamaIndex 使用"""

    _client: Any = PrivateAttr()

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._client = get_default_embedding_client()
        if not self._client:
            logger.error(
                "ProjectEmbedding: 无法初始化 Embedding 客户端，请检查环境变量 EMBEDDING_API_KEY。"
            )

    def _get_query_embedding(self, query: str) -> List[float]:
        if not self._client:
            raise ValueError(
                "Embedding 客户端未配置。请在 .env 中设置 EMBEDDING_API_KEY。"
            )
        embeddings = self._client([query])
        if not embeddings:
            raise ValueError("Embedding API 返回空结果。请检查网络或 API 额度。")
        return embeddings[0]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        if not self._client:
            raise ValueError(
                "Embedding 客户端未配置。请在 .env 中设置 EMBEDDING_API_KEY。"
            )
        embeddings = self._client([text])
        if not embeddings:
            raise ValueError("Embedding API 返回空结果。请检查网络或 API 额度。")
        return embeddings[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not self._client:
            raise ValueError(
                "Embedding 客户端未配置。请在 .env 中设置 EMBEDDING_API_KEY。"
            )
        embeddings = self._client(texts)
        if not embeddings:
            raise ValueError("Embedding API 返回空结果。请检查网络或 API 额度。")
        return embeddings


class SiliconFlowRerank(BaseNodePostprocessor):
    """自定义 SiliconFlow 重排组件"""

    api_key: str = Field(default="")
    endpoint: str = Field(default="https://api.siliconflow.cn/v1/rerank")
    model: str = Field(default="Pro/BAAI/bge-reranker-v2-m3")
    top_n: int = Field(default=3)

    def _postprocess_nodes(
        self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle] = None
    ) -> List[NodeWithScore]:
        if not query_bundle or not nodes or not self.api_key:
            return nodes[: self.top_n]

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
            resp = requests.post(self.endpoint, json=payload, headers=headers)
            resp.raise_for_status()
            results = resp.json().get("results", [])

            new_nodes = []
            for res in results:
                idx = res["index"]
                nodes[idx].score = float(res["relevance_score"])
                new_nodes.append(nodes[idx])

            new_nodes.sort(key=lambda x: x.score or 0.0, reverse=True)
            return new_nodes[: self.top_n]
        except Exception as e:
            logger.error(f"Rerank API failed: {e}")
            return nodes[: self.top_n]


# 应用全局向量模型配置
Settings.embed_model = ProjectEmbedding()

# --- 2. 核心构建与提取逻辑 ---


async def build_advanced_rag_index(
    md_content: str, session_id: str, md_filename: str, base_dir: str
) -> Tuple[VectorStoreIndex, StorageContext, List[Dict[str, Any]]]:
    """
    处理全流程：分块 -> VLM 增强 -> 构建「层级 AST 树」 -> 建立向量索引
    """
    # 1. 语义分块 (返回包含 metadata 的纯净结构字典)
    chunks = chunk_markdown(
        markdown_content=md_content,
        file_id=session_id,
        filename=md_filename,
        parser_config={"chunk_token_num": 512},
    )

    # 2. VLM 多模态增强 (仅处理分离出去的 Table/Image/Math，写入 text_for_embedding)
    vlm_client = create_default_vlm_client()
    processor = MarkdownMultimodalProcessor(vlm_func=vlm_client, max_concurrency=2)
    chunks = await processor.enrich_chunks(chunks, base_dir=base_dir)

    docstore_nodes = {}
    leaf_nodes = []
    path_nodes = {}
    ui_results = []
    node_id_map = {}

    for i, chunk in enumerate(chunks):
        chunk_id = chunk["id"]
        header_path = chunk["metadata"].get("header_path", [])
        element_type = chunk["metadata"].get("element_type", "text")
        parent_id = chunk["metadata"].get("parent_id")

        # ----- a. 解析 Heading 建立 Section 树状结构 -----
        current_parent_node_id = None
        current_path_str = ""
        for level in header_path:
            level = level.strip()
            current_path_str = (
                f"{current_path_str}|{level}" if current_path_str else level
            )
            if current_path_str not in path_nodes:
                safe_path = current_path_str.replace("|", "_").replace(" ", "")
                sect_id = f"path_{safe_path}_{session_id}"
                # 初始化为临时占位符，稍后聚合子节点内容
                sect_node = TextNode(id_=sect_id, text=f"【章节聚合：{level}】")
                docstore_nodes[sect_id] = sect_node
                path_nodes[current_path_str] = sect_id

                # 连接层级：子 Section -> 父 Section
                if current_parent_node_id:
                    sect_node.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(
                        node_id=current_parent_node_id
                    )
                    docstore_nodes[current_parent_node_id].relationships.setdefault(
                        NodeRelationship.CHILD, []
                    ).append(RelatedNodeInfo(node_id=sect_id))

            current_parent_node_id = path_nodes[current_path_str]

        # ----- b. 建立主干 Node / 叶子 Node -----
        if element_type == "text":
            # 文本段落：既是向量检索的目标，也是子元素（多模态）的 Parent
            node = TextNode(
                id_=chunk_id,
                text=chunk["text_for_embedding"],
                metadata={
                    "type": "chunk",
                    "element_type": element_type,
                    "full_content": chunk["full_content"],
                    "ui_id": str(i),
                    "path": current_path_str,
                },
            )
            # 挂载到对应的章节下
            if current_parent_node_id:
                node.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(
                    node_id=current_parent_node_id
                )
                docstore_nodes[current_parent_node_id].relationships.setdefault(
                    NodeRelationship.CHILD, []
                ).append(RelatedNodeInfo(node_id=chunk_id))

            docstore_nodes[chunk_id] = node
            leaf_nodes.append(node)
            node_id_map[chunk_id] = chunk_id

        else:
            # 独立元素（Table/Image）：它是某段文字的孩子
            # 我们用 IndexNode 作为指针，它的 text 是高浓度摘要（用来被搜），它的 index_id 指向原段落（或它自己，如果想独立返回）
            # 这里我们让它指向自己，这样召回时就能拿到原汁原味的 Markdown Table
            element_node = TextNode(
                id_=f"{chunk_id}_full",
                text=chunk["full_content"],  # 给 LLM 看的原文
                metadata={
                    "type": "element",
                    "element_type": element_type,
                    "entities": chunk["metadata"].get("entities", []),
                    "ui_id": str(i),
                    "path": current_path_str,
                },
            )
            docstore_nodes[element_node.id_] = element_node

            # 如果它有父段落（在 semantic.py 中识别出来的上文）
            if parent_id and parent_id in node_id_map:
                actual_parent_id = node_id_map[parent_id]
                element_node.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(
                    node_id=actual_parent_id
                )
                docstore_nodes[actual_parent_id].relationships.setdefault(
                    NodeRelationship.CHILD, []
                ).append(RelatedNodeInfo(node_id=element_node.id_))
            elif current_parent_node_id:
                # 降级：如果没有指定的父段落，就挂在章节下
                element_node.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(
                    node_id=current_parent_node_id
                )
                docstore_nodes[current_parent_node_id].relationships.setdefault(
                    NodeRelationship.CHILD, []
                ).append(RelatedNodeInfo(node_id=element_node.id_))

            # 建立指针用于向量检索
            index_node = IndexNode(
                id_=chunk_id,
                text=chunk["text_for_embedding"],  # 纯净的 VLM 摘要，用来喂给 Embedding
                index_id=element_node.id_,  # 指向刚刚创建的，装有完整 Markdown 的节点
            )
            docstore_nodes[index_node.id_] = index_node
            leaf_nodes.append(index_node)
            node_id_map[chunk_id] = element_node.id_

        ui_results.append(
            {
                "id": str(i),
                "original_content": chunk["content"],
                "enriched_content": chunk["full_content"]
                + (
                    "\n\n> AI Summary: " + chunk["text_for_embedding"]
                    if element_type != "text"
                    else ""
                ),
                "images": [],  # In decoupled mode, we rely on the markdown renderer
                "metadata": chunk.get("metadata", {}),
            }
        )

    # ----- c. 后处理：聚合 Section 节点的文本，满足宏观检索需求 -----
    sorted_paths = sorted(
        list(path_nodes.keys()), key=lambda x: len(x.split("|")), reverse=True
    )
    for path in sorted_paths:
        sect_id = path_nodes[path]
        sect_node = docstore_nodes[sect_id]

        child_texts = []
        for child_rel in sect_node.relationships.get(NodeRelationship.CHILD, []):
            child_node = docstore_nodes[child_rel.node_id]
            # 合并文本时，使用带有完整上下文的 full_content
            child_texts.append(child_node.metadata.get("full_content", child_node.text))

        sect_node.text = f"【章节聚合：{path.split('|')[-1]}】\n\n" + "\n\n".join(
            child_texts
        )
        # 章节节点自己也保留完整内容，供上级合并使用
        sect_node.metadata["full_content"] = sect_node.text

    # 3. 注入数据到存储和向量库
    docstore = SimpleDocumentStore()
    docstore.add_documents(list(docstore_nodes.values()))
    storage_context = StorageContext.from_defaults(docstore=docstore)

    # 仅向量化 Leaf Nodes（底层的文本段落和摘要指针）
    vector_index = VectorStoreIndex(leaf_nodes, storage_context=storage_context)

    return vector_index, storage_context, ui_results


# --- 3. 高级检索接口 ---


class RerankedRetriever(BaseRetriever):
    """
    一个简单的包装检索器，用于在合并前执行精排。
    避免超大章节被精排模型在 512 tokens 处无情截断。
    """

    def __init__(
        self, base_retriever: BaseRetriever, reranker: Optional[BaseNodePostprocessor]
    ):
        self.base_retriever = base_retriever
        self.reranker = reranker
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        nodes = self.base_retriever.retrieve(query_bundle)
        if self.reranker and nodes:
            nodes = self.reranker.postprocess_nodes(nodes, query_bundle)
        return nodes


def retrieve_advanced(
    query: str,
    index: VectorStoreIndex,
    storage_context: StorageContext,
    top_k: int = 15,
    rerank_top_n: int = 5,
) -> List[Dict[str, Any]]:
    """
    使用四级火箭：Vector Index -> Recursive (回表) -> Rerank (短文本高精度精排) -> AutoMerging (合并章节)
    """
    vector_retriever = index.as_retriever(similarity_top_k=top_k)

    # 1. 递归回表：当命中 IndexNode(摘要) 时，自动回表拉取对应的完整表格节点
    recursive_retriever = RecursiveRetriever(
        "vector",
        retriever_dict={"vector": vector_retriever},
        node_dict=storage_context.docstore.docs,
    )

    # 2. 构造精排器
    reranker = None
    reranker_key = os.getenv("RERANK_API_KEY")
    if reranker_key:
        reranker = SiliconFlowRerank(
            api_key=reranker_key,
            endpoint=os.getenv(
                "RERANK_ENDPOINT", "https://api.siliconflow.cn/v1/rerank"
            ),
            model=os.getenv("RERANK_MODEL_NAME", "Qwen/Qwen3-Reranker-0.6B"),
            top_n=rerank_top_n,
        )

    # 3. 将 Rerank 包裹在 RecursiveRetriever 之后，合并之前
    # 此时进入 Rerank 的都是短小的摘要或原始段落，绝对不会触发 512 token 截断
    reranked_retriever = RerankedRetriever(
        base_retriever=recursive_retriever, reranker=reranker
    )

    # 4. 树状爬升合并：接收被 Rerank 筛选过的 Top 5 精华切片。
    # 如果同一个章节下被命中的精华切片超过阈值，就把整个大章节丢出去
    auto_merging_retriever = AutoMergingRetriever(
        reranked_retriever,  # 关键：入参变成了被重排过的检索器
        storage_context,
        verbose=False,
        simple_ratio_thresh=0.5,
    )

    # 执行全链路检索
    nodes = auto_merging_retriever.retrieve(query)

    # 5. 封装返回给 API
    results = []
    for node in nodes:
        node_text = node.node.get_content()
        metadata = node.node.metadata
        ui_id = metadata.get("ui_id")

        is_merged_section = "【章节聚合：" in node_text

        results.append(
            {
                "chunk_id": node.node.id_,
                "content": node_text,
                "ui_id": ui_id,
                "score": float(node.score or 0.0),
                "is_merged_section": is_merged_section,
            }
        )

    return results
