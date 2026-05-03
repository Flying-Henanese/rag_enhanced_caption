import json
import os
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core.schema import TextNode, IndexNode
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.node_parser import SentenceSplitter

load_dotenv()
logger.remove()
import sys
logger.add(sys.stderr, level="WARNING")

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
            api_base=endpoint
        )
    except ImportError:
        pass

RESOURCE_DIR = Path("test_resource")
DOCSTORE_PATH = RESOURCE_DIR / "高性能文档解析方案 2e2848cda67f8020abf0d58252a28708_docstore.jsonl"
ORIGINAL_MD_PATH = RESOURCE_DIR / "高性能文档解析方案 2e2848cda67f8020abf0d58252a28708.md"

def get_baseline_components():
    with open(ORIGINAL_MD_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    doc = Document(text=text, id_="doc_001")
    parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
    nodes = parser.get_nodes_from_documents([doc])
    index = VectorStoreIndex(nodes)
    retriever = index.as_retriever(similarity_top_k=1)
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
            
            if sample_parent is None and "mineru-api" in full_text:
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
            
            if sample_child is None and "系统架构" in child_node.text:
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
    vector_retriever = vector_index.as_retriever(similarity_top_k=2)

    auto_merging_retriever = AutoMergingRetriever(
        vector_retriever,
        storage_context,
        verbose=True,
        simple_ratio_thresh=0.3
    )
    
    return sample_parent, sample_child, vector_retriever, auto_merging_retriever

def print_box(title, content):
    print(f"\n┌── {title} {'─'*(60-len(title))}")
    for line in str(content).split('\n'):
        print(f"│ {line[:90] + '...' if len(line) > 90 else line}")
    print(f"└{'─'*64}")

def run_narrative_evaluation():
    query = "该文档中包含了一个系统架构图，请问这个系统架构图的核心组件包含哪些？它的作用是什么？"
    
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
    sample_base_node = [n for n in base_nodes if "系统架构图" in n.text or "mineru-api" in n.text]
    if not sample_base_node: sample_base_node = base_nodes
    
    print("\n【Step 1. 切分 (Chunking)】")
    print_box("原始切片 (TextNode)", sample_base_node[0].text[:300] + "\n...")
    
    print("\n【Step 2. 入库 (Indexing)】")
    
    print("\n【Step 3. 检索 (Retrieval)】")
    base_result = base_retriever.retrieve(query)[0]
    score_val = base_result.score if base_result.score is not None else 0.0
    print_box(f"命中节点 (Score: {score_val:.4f})", base_result.node.text[:200] + "\n...")
    
    print("\n【Step 4. 召回 (Recall & Generation)】")
    
    # ---------------------------------------------------------
    # 方案 B: 我们的方案 (Global Tree + AutoMerging)
    # ---------------------------------------------------------
    print("\n\n" + "="*80)
    print(" ✅ 方案 B: 我们的方案 (AST 全局多叉树 + AutoMerging 自动合并)")
    print("="*80)
    
    sample_parent, sample_child, adv_vector_retriever, adv_auto_merging_retriever = get_advanced_components()
    
    print("\n【Step 1. 结构化多叉树切分 (Global Tree Hierarchy Chunking)】")
    print("抛弃扁平切分，基于 Markdown 标题构建了一棵真正的树：Root -> 章节 (H1/H2) -> 物理段落 -> AI 摘要/纯文本锚点。")
    print_box("树的最底层：AI 意图叶子节点", sample_child["text_for_embedding"])
    print_box("树的倒数第二层：完整段落节点", sample_parent["full_content"][:300] + "\n...")
    
    print("\n【Step 2. 叶子节点入库 (Leaf Indexing)】")
    print("只将树的最底层（精炼纯粹的短文本）送入向量数据库，作为无噪音的搜索锚点。")
    
    print("\n【Step 3. 检索 (Vector Retrieval)】")
    child_result = adv_vector_retriever.retrieve(query)[0]
    score_val = child_result.score if child_result.score is not None else 0.0
    print_box(f"向量库精准命中底层锚点 (Score: {score_val:.4f})", child_result.node.text)
    
    print("\n【Step 4. 自动向上合并召回 (Auto-Merging & Recall)】")
    print("AutoMergingRetriever 感知到叶子节点被命中，开始沿着我们建立的树干向上攀爬！")
    print("如果多个段落命中，它甚至会合并出整个 H1/H2 章节！")
    
    # We set top_k=2 on vector_retriever. If both leaf hits belong to the same parent, they merge.
    final_results = adv_auto_merging_retriever.retrieve(query)
    for i, final_result in enumerate(final_results):
        print_box(f"最终喂给大模型的豪华上下文 (Merged Node {i+1})", final_result.node.text[:500] + "\n...")

if __name__ == "__main__":
    run_narrative_evaluation()
