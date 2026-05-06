import asyncio
import re
import os
import requests
from typing import List, Dict, Any, Tuple, Optional
from pydantic import Field
from loguru import logger

from llama_index.core import StorageContext, VectorStoreIndex, Settings, QueryBundle
from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo, NodeWithScore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.embeddings import BaseEmbedding

from rag_enhanced_caption.chunker.dispatcher import chunk_markdown
from rag_enhanced_caption.enhancer.processor import MarkdownMultimodalProcessor
from rag_enhanced_caption.enhancer.vlm_client import create_default_vlm_client
from rag_enhanced_caption.chunker.embed_client import get_default_embedding_client

# --- 1. LlamaIndex 扩展组件配置 ---

class ProjectEmbedding(BaseEmbedding):
    """包装自定义向量模型供 LlamaIndex 使用"""
    def _get_query_embedding(self, query: str) -> List[float]:
        return get_default_embedding_client()([query])[0]
    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)
    def _get_text_embedding(self, text: str) -> List[float]:
        return get_default_embedding_client()([text])[0]
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return get_default_embedding_client()(texts)

class SiliconFlowRerank(BaseNodePostprocessor):
    """自定义 SiliconFlow 重排组件"""
    api_key: str = Field(default="")
    endpoint: str = Field(default="https://api.siliconflow.cn/v1/rerank")
    model: str = Field(default="Pro/BAAI/bge-reranker-v2-m3")
    top_n: int = Field(default=3)

    def _postprocess_nodes(self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle] = None) -> List[NodeWithScore]:
        if not query_bundle or not nodes or not self.api_key:
            return nodes[:self.top_n]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        texts = [node.node.get_content() for node in nodes]
        payload = {
            "model": self.model,
            "query": query_bundle.query_str,
            "documents": texts,
            "return_documents": False
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
            return new_nodes[:self.top_n]
        except Exception as e:
            logger.error(f"Rerank API failed: {e}")
            return nodes[:self.top_n]

# 应用全局向量模型配置
Settings.embed_model = ProjectEmbedding()

# --- 2. 核心构建与提取逻辑 ---

_IMAGE_ANALYSIS_RE = re.compile(r"<image_analysis>(.*?)</image_analysis>", re.DOTALL)
_ENTITY_RE = re.compile(r"-\s+\*\*关键实体\*\*:\s*(.*)")
_SUMMARY_RE = re.compile(r"-\s+\*\*简短摘要\*\*:\s*(.*)")

def _extract_metadata(analysis_body: str) -> Dict[str, Any]:
    """提取 AI 注释中的元数据供检索参考"""
    ent_match = _ENTITY_RE.search(analysis_body)
    sum_match = _SUMMARY_RE.search(analysis_body)
    return {
        "entities": [e.strip() for e in ent_match.group(1).split(",") if e.strip()] if ent_match else [],
        "summary": sum_match.group(1).strip() if sum_match else analysis_body[:100]
    }

async def build_advanced_rag_index(
    md_content: str, 
    session_id: str, 
    md_filename: str, 
    base_dir: str
) -> Tuple[VectorStoreIndex, StorageContext, List[Dict[str, Any]]]:
    """
    处理全流程：分块 -> VLM 增强 -> 构建「层级 AST 树（Section -> Chunk -> Leaf）」 -> 建立向量索引
    返回供检索的 VectorStoreIndex, StorageContext 和供 UI 渲染的 results
    """
    # 1. 语义分块 (返回包含 metadata 的纯净结构字典)
    chunks = chunk_markdown(
        markdown_content=md_content,
        file_id=session_id,
        filename=md_filename,
        parser_config={"chunk_token_num": 512}
    )

    # 2. VLM 多模态增强
    vlm_client = create_default_vlm_client()
    processor = MarkdownMultimodalProcessor(vlm_func=vlm_client, max_concurrency=2)
    
    docstore_nodes = {}
    leaf_nodes = []
    path_nodes = {}
    ui_results = []

    for i, chunk in enumerate(chunks):
        # 执行增强
        enriched_text = await processor.enrich_markdown(chunk["content"], base_dir=base_dir)
        
        # 处理净化文本
        clean_content = _IMAGE_ANALYSIS_RE.sub("", enriched_text)
        clean_content = re.sub(r"<details>.*?</details>", "", clean_content, flags=re.DOTALL).strip()
        
        chunk_id = f"chunk_{i}_{session_id}"
        
        # ----- a. 解析 Heading 建立 Section 树状结构 (从 metadata 获取) -----
        header_str = chunk["metadata"].get("header", "")
        element_type = chunk["metadata"].get("element_type", "text")
        
        match = re.match(r'^#+\s+(.*)', header_str)
        levels = match.group(1).split('|') if match else ["默认文档"]
        
        current_parent_id = None
        current_path = ""
        for level in levels:
            level = level.strip()
            current_path = f"{current_path}|{level}" if current_path else level
            if current_path not in path_nodes:
                safe_path = current_path.replace("|", "_").replace(" ", "")
                sect_id = f"path_{safe_path}_{session_id}"
                # 初始化为临时占位符，稍后聚合子节点内容
                sect_node = TextNode(id_=sect_id, text=f"【章节聚合：{level}】") 
                docstore_nodes[sect_id] = sect_node
                path_nodes[current_path] = sect_id
                
                # 连接层级：子 Section -> 父 Section
                if current_parent_id:
                    sect_node.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(node_id=current_parent_id)
                    docstore_nodes[current_parent_id].relationships.setdefault(NodeRelationship.CHILD, []).append(RelatedNodeInfo(node_id=sect_id))
            
            current_parent_id = path_nodes[current_path]
            
        # 拼接 header 和 content 作为最终呈现给 Node 的完整内容
        full_node_text = f"{header_str}\n\n{enriched_text}" if header_str else enriched_text
        full_clean_text = f"{header_str}\n\n{clean_content}" if header_str else clean_content

        # ----- b. 建立段落主干 Node -----
        chunk_node = TextNode(
            id_=chunk_id,
            text=full_node_text,
            metadata={
                "type": "chunk", 
                "element_type": element_type,
                "full_content": full_node_text, 
                "ui_id": str(i), 
                "path": current_path
            }
        )
        # 连接段落 -> Section
        chunk_node.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(node_id=current_parent_id)
        docstore_nodes[current_parent_id].relationships.setdefault(NodeRelationship.CHILD, []).append(RelatedNodeInfo(node_id=chunk_id))
        docstore_nodes[chunk_id] = chunk_node
        
        # ----- c. 建立搜索钩子 Leaves -----
        
        # C-1. 纯文本钩子 (搜索干净内容)
        search_node = TextNode(id_=f"search_txt_{chunk_id}", text=full_clean_text)
        search_node.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(node_id=chunk_id)
        chunk_node.relationships.setdefault(NodeRelationship.CHILD, []).append(RelatedNodeInfo(node_id=search_node.id_))
        docstore_nodes[search_node.id_] = search_node
        leaf_nodes.append(search_node)
        
        # C-2. 多模态/图片 Caption 钩子
        images = []
        analysis_blocks = _IMAGE_ANALYSIS_RE.findall(enriched_text)
        img_matches = re.findall(r"!\[(.*?)\]\((.*?)\)", chunk["content"])
        
        for j, analysis_body in enumerate(analysis_blocks):
            caption_id = f"search_img_{j}_{chunk_id}"
            meta = _extract_metadata(analysis_body)
            url = img_matches[j][1] if j < len(img_matches) else "unknown"
            images.append({"src": url, "alt": img_matches[j][0] if j < len(img_matches) else ""})
            
            # 使用 AI 简短摘要作为搜素入口
            caption_node = TextNode(id_=caption_id, text=meta["summary"])
            caption_node.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(node_id=chunk_id)
            chunk_node.relationships.setdefault(NodeRelationship.CHILD, []).append(RelatedNodeInfo(node_id=caption_node.id_))
            docstore_nodes[caption_node.id_] = caption_node
            leaf_nodes.append(caption_node)
            
        ui_results.append({
            "id": str(i),
            "original_content": chunk["content"],
            "enriched_content": full_node_text,
            "images": images,
            "metadata": chunk.get("metadata", {})
        })

    # ----- d. 后处理：聚合 Section 节点的文本，满足宏观检索需求 -----
    sorted_paths = sorted(list(path_nodes.keys()), key=lambda x: len(x.split('|')), reverse=True)
    for path in sorted_paths:
        sect_id = path_nodes[path]
        sect_node = docstore_nodes[sect_id]
        
        child_texts = []
        for child_rel in sect_node.relationships.get(NodeRelationship.CHILD, []):
            child_node = docstore_nodes[child_rel.node_id]
            child_texts.append(child_node.text)
            
        sect_node.text = f"【章节聚合：{path.split('|')[-1]}】\n\n" + "\n\n".join(child_texts)

    # 3. 注入数据到存储和向量库
    docstore = SimpleDocumentStore()
    docstore.add_documents(list(docstore_nodes.values()))
    storage_context = StorageContext.from_defaults(docstore=docstore)
    
    # 仅向量化 Leaf Nodes（底层段落与图片描述），上层靠合并召回
    vector_index = VectorStoreIndex(leaf_nodes, storage_context=storage_context)
    
    return vector_index, storage_context, ui_results


# --- 3. 高级检索接口 ---

def retrieve_advanced(
    query: str, 
    index: VectorStoreIndex, 
    storage_context: StorageContext, 
    top_k: int = 12, 
    rerank_top_n: int = 3
) -> List[Dict[str, Any]]:
    """
    使用 AutoMergingRetriever 及可选的 Rerank 处理召回。
    """
    vector_retriever = index.as_retriever(similarity_top_k=top_k)
    
    # 开启自动向上合并能力 (将阈值提高到 0.5，要求至少一半子节点命中才合并)
    auto_merging_retriever = AutoMergingRetriever(
        vector_retriever,
        storage_context,
        verbose=False,
        simple_ratio_thresh=0.5
    )
    
    # 1. 初筛 + 树状合并
    nodes = auto_merging_retriever.retrieve(query)
    
    # 2. 精排
    reranker_key = os.getenv("RERANK_API_KEY")
    if reranker_key and nodes:
        reranker = SiliconFlowRerank(
            api_key=reranker_key,
            endpoint=os.getenv("RERANK_ENDPOINT", "https://api.siliconflow.cn/v1/rerank"),
            model=os.getenv("RERANK_MODEL_NAME", "Qwen/Qwen3-Reranker-0.6B"),
            top_n=rerank_top_n
        )
        nodes = reranker.postprocess_nodes(nodes, query_bundle=QueryBundle(query_str=query))
    else:
        nodes = nodes[:rerank_top_n]
        
    # 3. 封装返回给 API
    results = []
    for node in nodes:
        node_text = node.node.get_content()
        metadata = node.node.metadata
        ui_id = metadata.get("ui_id")
        
        is_merged_section = "【章节聚合：" in node_text
        
        results.append({
            "chunk_id": node.node.id_,
            "content": node_text, 
            "ui_id": ui_id,
            "score": float(node.score or 0.0),
            "is_merged_section": is_merged_section
        })
        
    return results
