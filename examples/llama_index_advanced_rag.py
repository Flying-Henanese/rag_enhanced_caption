import json
import os
import requests
from typing import List, Optional
from pydantic import Field
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core.schema import TextNode, IndexNode, NodeWithScore, QueryBundle
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.node_parser import SentenceSplitter

load_dotenv()
logger.remove()
import sys
logger.add(sys.stderr, level="WARNING")

class SiliconFlowRerank(BaseNodePostprocessor):
    api_key: str = Field(default="")
    endpoint: str = Field(default="https://api.siliconflow.cn/v1/rerank")
    model: str = Field(default="Pro/BAAI/bge-reranker-v2-m3")
    top_n: int = Field(default=2)

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        if query_bundle is None or not nodes:
            return nodes

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
            response = requests.post(self.endpoint, json=payload, headers=headers)
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
            return new_nodes[:self.top_n]
        except Exception as e:
            logger.error(f"Rerank failed: {e}")
            return nodes[:self.top_n]

embed_model_name = os.getenv("EMBEDDING_MODEL_NAME")
if embed_model_name:
    try:
        from llama_index.embeddings.openai import OpenAIEmbedding
        api_key = os.getenv("EMBEDDING_API_KEY", "")
        endpoint = os.getenv("EMBEDDING_ENDPOINT", "https://api.siliconflow.cn/v1/embeddings")
        if endpoint.endswith("/embeddings"):
            endpoint = endpoint[:-11]
            
        Settings.embed_model = OpenAIEmbedding(
            model_name=embed_model_name,
            api_key=api_key,
            api_base=endpoint,
            embed_batch_size=64
        )
    except ImportError:
        pass

RESOURCE_DIR = Path("test_resource")
OUTPUT_DIR = Path("output")
DOCSTORE_PATH = OUTPUT_DIR / "rag-anything_docstore.jsonl"
ORIGINAL_MD_PATH = RESOURCE_DIR / "rag-anything.md"

def get_baseline_components():
    with open(ORIGINAL_MD_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    doc = Document(text=text, id_="doc_001")
    parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
    nodes = parser.get_nodes_from_documents([doc])
    index = VectorStoreIndex(nodes)
    retriever = index.as_retriever(similarity_top_k=2)
    return nodes, retriever

def get_advanced_components():
    from llama_index.core.schema import NodeRelationship, RelatedNodeInfo
    from llama_index.core.storage.docstore import SimpleDocumentStore
    from llama_index.core import StorageContext
    from llama_index.core.retrievers import AutoMergingRetriever
    import re
    
    docstore_nodes = {}
    path_nodes = {}
    leaf_nodes = []
    
    sample_parent = None
    sample_child = None
    
    with open(DOCSTORE_PATH, "r", encoding="utf-8") as f:
        docstore_records = [json.loads(line) for line in f]
        
    for data in docstore_records:
        if data.get("type") == "parent":
            node_id = data.get("id")
            full_text = data.get("full_content", "")
            
            first_line = full_text.split('\n')[0].strip()
            match = re.match(r'^#+\s+(.*)', first_line)
            levels = match.group(1).split('|') if match else ["默认文档"]
            
            current_parent_id = None
            current_path = ""
            for level in levels:
                current_path = f"{current_path}|{level}" if current_path else level
                if current_path not in path_nodes:
                    sect_id = f"path_{current_path}"
                    sect_node = TextNode(id_=sect_id, text=f"【章节：{level}】")
                    docstore_nodes[sect_id] = sect_node
                    path_nodes[current_path] = sect_id
                    
                    if current_parent_id:
                        sect_node.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(node_id=current_parent_id)
                        docstore_nodes[current_parent_id].relationships.setdefault(NodeRelationship.CHILD, []).append(RelatedNodeInfo(node_id=sect_id))
                
                current_parent_id = path_nodes[current_path]
            
            chunk_node = TextNode(
                id_=node_id,
                text=full_text,
                metadata=data.get("metadata", {})
            )
            chunk_node.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(node_id=current_parent_id)
            docstore_nodes[current_parent_id].relationships.setdefault(NodeRelationship.CHILD, []).append(RelatedNodeInfo(node_id=node_id))
            docstore_nodes[node_id] = chunk_node
            
            if sample_parent is None and "RAG-Anything" in full_text:
                sample_parent = data
                
            search_node = TextNode(
                id_=f"search_{node_id}",
                text=data.get("text_for_embedding", "")
            )
            search_node.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(node_id=node_id)
            chunk_node.relationships.setdefault(NodeRelationship.CHILD, []).append(RelatedNodeInfo(node_id=search_node.id_))
            
            docstore_nodes[search_node.id_] = search_node
            leaf_nodes.append(search_node)
            
        elif data.get("type") == "child":
            parent_id = data.get("parent_id")
            node_id = data.get("id")
            
            child_node = TextNode(
                id_=node_id,
                text=data.get("text_for_embedding", ""),
                metadata=data.get("metadata", {})
            )
            child_node.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(node_id=parent_id)
            
            if parent_id in docstore_nodes:
                docstore_nodes[parent_id].relationships.setdefault(NodeRelationship.CHILD, []).append(RelatedNodeInfo(node_id=node_id))
            
            docstore_nodes[node_id] = child_node
            leaf_nodes.append(child_node)
            
            if sample_child is None and "Framework" in child_node.text:
                sample_child = data

    if not sample_parent: sample_parent = [d for d in docstore_records if d["type"] == "parent"][0]
    if not sample_child: sample_child = [d for d in docstore_records if d["type"] == "child"][0]

    sorted_paths = sorted(list(path_nodes.keys()), key=lambda x: len(x.split('|')), reverse=True)
    for path in sorted_paths:
        node_id = path_nodes[path]
        node = docstore_nodes[node_id]
        children_info = node.relationships.get(NodeRelationship.CHILD, [])
        child_texts = []
        for child_info in children_info:
            child_node = docstore_nodes[child_info.node_id]
            child_texts.append(child_node.text)
        node.text = f"【章节聚合：{path.split('|')[-1]}】\n" + "\n\n".join(child_texts)

    docstore = SimpleDocumentStore()
    docstore.add_documents(list(docstore_nodes.values()))
    storage_context = StorageContext.from_defaults(docstore=docstore)

    vector_index = VectorStoreIndex(leaf_nodes, storage_context=storage_context)
    # Increased top_k for vector retriever to give reranker more options
    vector_retriever = vector_index.as_retriever(similarity_top_k=20)

    auto_merging_retriever = AutoMergingRetriever(
        vector_retriever,
        storage_context,
        verbose=True,
        simple_ratio_thresh=0.3
    )
    
    reranker = None
    rerank_api_key = os.getenv("RERANK_API_KEY")
    if rerank_api_key:
        reranker = SiliconFlowRerank(
            api_key=rerank_api_key,
            endpoint=os.getenv("RERANK_ENDPOINT", "https://api.siliconflow.cn/v1/rerank"),
            model=os.getenv("RERANK_MODEL_NAME", "Pro/BAAI/bge-reranker-v2-m3"),
            top_n=3
        )
    
    return sample_parent, sample_child, vector_retriever, auto_merging_retriever, reranker

def print_box(title, content):
    print(f"\n┌── {title} {'─'*(60-len(title))}")
    for line in str(content).split('\n'):
        print(f"│ {line[:90] + '...' if len(line) > 90 else line}")
    print(f"└{'─'*64}")

def run_narrative_evaluation(query: str):
    print("\n=========================================================================================")
    print(f" 🎯 评测问题: {query}")
    print("=========================================================================================")
    
    # ---------------------------------------------------------
    # 方案 A: 传统基线 (Baseline Naive Chunking)
    # ---------------------------------------------------------
    print("\n\n" + "="*80)
    print(" ❌ 方案 A: 传统基线 (Naive Chunking)")
    print("="*80)
    
    base_nodes, base_retriever = get_baseline_components()
    
    print("\n【检索 (Retrieval)】")
    base_results = base_retriever.retrieve(query)
    for i, base_result in enumerate(base_results):
        score_val = base_result.score if base_result.score is not None else 0.0
        print_box(f"命中节点 {i+1} (Score: {score_val:.4f})", base_result.node.text[:200] + "\n...")
    
    # ---------------------------------------------------------
    # 方案 B: 我们的方案 (Global Tree + AutoMerging)
    # ---------------------------------------------------------
    print("\n\n" + "="*80)
    print(" ✅ 方案 B: 我们的方案 (AST 全局多叉树 + AutoMerging 自动合并 + 可选 Rerank)")
    print("="*80)
    
    sample_parent, sample_child, adv_vector_retriever, adv_auto_merging_retriever, reranker = get_advanced_components()
    
    print("\n【自动向上合并召回与精排 (Auto-Merging & Reranking)】")
    final_results = adv_auto_merging_retriever.retrieve(query)
    
    if reranker:
        print(f"使用 {reranker.model} 模型进行精排...")
        final_results = reranker.postprocess_nodes(final_results, query_bundle=QueryBundle(query_str=query))
    
    for i, final_result in enumerate(final_results):
        score_info = f", Rerank Score: {final_result.score:.4f}" if reranker and final_result.score is not None else ""
        print_box(f"最终召回豪华上下文 (Merged Node {i+1}{score_info})", final_result.node.text[:500] + "\n...")

if __name__ == "__main__":
    queries = [
        "RAG-Anything 的架构图（Framework）展示了哪些主要的处理阶段？",
        "针对数学表达式（Mathematical Expression），系统是如何提供原生支持的？",
        "LiteWrite 是什么？它和项目有什么关系？"
    ]
    for q in queries:
        run_narrative_evaluation(q)